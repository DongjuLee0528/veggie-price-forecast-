"""Publish additive radish direction records; never rewrite earlier forecast metadata."""
import hashlib
import json
import os
import subprocess

import pandas as pd

from src.db.sync_mysql import export_version
from src.pipeline.radish_logistic import MODEL, probabilities


def sync(command, outputs, *, install=False):
    def query(sql):
        result = subprocess.run(command, input="SET SESSION time_zone='+00:00';\n" + sql,
                                text=True, capture_output=True, timeout=120,
                                env={**os.environ, "MYSQL_PWD": os.environ["DB_PASSWORD"]})
        if result.returncode:
            raise RuntimeError("Radish direction synchronization failed: " + result.stderr[:300])
        return result.stdout.splitlines()

    if install:
        query((outputs.parent / "db/radish_direction.sql").read_text())
    missing = query("""
        SELECT r.id, r.base_date, r.source, r.model_version
        FROM forecast_run r JOIN forecast_short s ON s.run_id=r.id
        JOIN item i ON i.id=s.item_id
        LEFT JOIN radish_direction_logistic l ON l.run_id=r.id
        WHERE i.name='무' AND s.horizon=1 AND l.run_id IS NULL ORDER BY r.id;
    """)
    if not missing:
        return
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    issued = {}
    for p in outputs.glob("selected_forecasts_*/short/forecast.json"):
        short = p.read_bytes()
        report = json.loads(short)
        if report.get("radish_logistic_hash") == digest:
            mid = p.parent.parent / "mid/forecast.json"
            issued[export_version([short, mid.read_bytes()])] = float(report["radish_probability_up"])
    histories = sorted((outputs / "wholesale_verify").glob("forecast_inputs_????????/prices.parquet"))
    past = probabilities(pd.read_parquet(histories[-1])) if histories else pd.Series(dtype=float)
    values = []
    for line in missing:
        run_id, date, source, version = line.split("\t")
        snapshot = source == "LIVE" and version in issued
        prob = issued[version] if snapshot else past.get(pd.Timestamp(date))
        if pd.isna(prob):
            continue
        if not 0 <= prob <= 1:
            raise ValueError("Invalid radish probability")
        # Only first publication on the origin day can count towards operational scores.
        retro = f"DATE(UTC_TIMESTAMP() + INTERVAL 9 HOUR) <> '{pd.Timestamp(date).date()}'" if snapshot and not install else "TRUE"
        values.append(f"({int(run_id)},{prob:.17g},({retro}),'{digest}')")
    if values:
        query("INSERT INTO radish_direction_logistic (run_id, probability_up, reconstructed, model_hash) VALUES "
              + ",".join(values) + " ON DUPLICATE KEY UPDATE run_id=radish_direction_logistic.run_id;")
    print(f"Radish logistic direction records processed: {len(values)}")
