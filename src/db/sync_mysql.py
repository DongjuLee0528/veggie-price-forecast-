import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from src.evaluate.prospective_direction import forecast_signals
from src.evaluate.prospective_forecasts import EXPECTED, ROOT
from src.features.targets import dekad_bounds, dekad_index
from src.models.compare_all_items import ITEMS


def literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, (float, int)):
        if not math.isfinite(value):
            raise ValueError("Expected a finite number")
        return str(value)
    return "CONVERT(X'" + value.encode("utf-8").hex() + "' USING utf8mb4)"


def date_value(value):
    return pd.Timestamp(value).date().isoformat()


def price_rows(directory, today):
    for path in sorted(directory.glob("daily_????????.csv")):
        day = pd.to_datetime(path.stem[-8:], format="%Y%m%d")
        if day > today:
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        if len(frame) != len(ITEMS) or set(frame.item) != ITEMS or not frame.date.eq(day).all():
            raise ValueError(f"Invalid price report: {path.name}")
        usable = frame[frame.observed_markets.gt(0) & frame.unconfirmed_markets.eq(0)
                       & frame.review_markets.eq(0) & frame.provisional_price_per_kg.notna()]
        for row in usable.itertuples():
            if min(row.provisional_price_per_kg, row.volume_kg) <= 0:
                raise ValueError(f"Invalid price or volume: {path.name}")
            yield row.item, date_value(day), float(row.provisional_price_per_kg), float(row.volume_kg)


def forecast_rows(directory, inputs_root, today):
    paths = [directory / term / "forecast.json" for term in ("short", "mid")]
    payloads = [path.read_bytes() for path in paths]
    short, mid = map(json.loads, payloads)
    origin = pd.Timestamp(short["base_date"])
    if origin > today or any(report["base_date"] != short["base_date"] or
                             report["source"] != "mafra_api_provisional" for report in (short, mid)):
        raise ValueError("Incompatible short and mid forecast exports")
    records = [(term, row) for term, report in (("short", short), ("mid", mid))
               for row in report["forecasts"]]
    if len(records) != len(EXPECTED) or {(term, row["item"], row["horizon"]) for term, row in records} != EXPECTED:
        raise ValueError("Expected all 25 selected forecast routes")
    inputs = inputs_root / f"forecast_inputs_{origin:%Y%m%d}"
    features = pd.read_parquet(inputs / "price_features.parquet")
    base = features.loc[features.date.eq(origin)].set_index("item").base_price
    if not base.index.is_unique or set(base.index) != ITEMS or not base.gt(0).all():
        raise ValueError("Missing forecast reference prices")
    short_rows, mid_rows = [], []
    for term, row in records:
        prediction = row["prediction_krw_per_kg"]
        if row["base_date"] != short["base_date"] or not math.isfinite(prediction) or prediction <= 0:
            raise ValueError("Invalid forecast origin or price")
        item, horizon = row["item"], row["horizon"]
        if term == "short":
            target = row.get("target_date")
            if target is not None and pd.Timestamp(target) <= origin:
                raise ValueError("Invalid short forecast target")
            short_rows.append((item, horizon, date_value(target) if target else None,
                               prediction, float(base.loc[item]), row["model"]))
        else:
            start, end = dekad_bounds(dekad_index(origin) + horizon)
            if (row["target_start"], row["target_end"]) != (date_value(start), date_value(end)):
                raise ValueError("Invalid mid forecast target")
            start, end = date_value(start), date_value(end)
            mid_rows.append((item, horizon, f"{start} ~ {end}", start, end, prediction))
    version = "api-provisional-" + hashlib.sha256(b"\x00".join(payloads)).hexdigest()[:16]
    return date_value(origin), version, short_rows, mid_rows, forecast_signals(
        short["forecasts"], inputs / "prices.parquet", origin)


def item_id(name):
    return f"(SELECT id FROM item WHERE name={literal(name)})"


def insert(table, columns, values, condition):
    return f"INSERT INTO {table} ({columns}) SELECT {', '.join(values)} WHERE {condition};"


def sync_sql(prices, forecast):
    yield "SET SESSION time_zone = '+00:00';"
    yield "CREATE TEMPORARY TABLE forecast_sync_guard (held INT NOT NULL CHECK (held = 1));"
    yield "INSERT INTO forecast_sync_guard VALUES (GET_LOCK(SHA2(CONCAT(DATABASE(), ':forecast-sync'), 256), 10));"
    yield "START TRANSACTION;"
    yield "SET @price_count=0, @forecast_count=0;"
    for item, date, price, volume in prices:
        key, day = item_id(item), literal(date)
        yield insert("price_daily", "item_id,trade_date,price_per_kg,volume_kg",
                     [key, day, literal(price), literal(volume)],
                     f"NOT EXISTS (SELECT 1 FROM price_daily WHERE item_id={key} AND trade_date={day})")
        yield "SET @price_count=@price_count+ROW_COUNT();"
    if forecast is not None:
        date, version, short, mid, signals = forecast
        day, tag = literal(date), literal(version)
        yield f"SET @run = (SELECT id FROM forecast_run WHERE base_date={day} AND source='LIVE' AND model_version={tag});"
        yield "SET @new_run = (SELECT COALESCE(MAX(id),0)+1 FROM forecast_run);"
        yield insert("forecast_run", "id,base_date,source,model_version,created_at",
                     ["@new_run", day, "'LIVE'", tag, "UTC_TIMESTAMP()"], "@run IS NULL")
        yield "SET @forecast_count=ROW_COUNT(), @run=COALESCE(@run,@new_run);"
        for table, rows, columns, horizon_col in (
            ("forecast_short", short, "horizon,target_date,p50,base_price,model", "horizon"),
            ("forecast_mid", mid, "k,period_label,period_start,period_end,p50", "k"),
        ):
            for item, horizon, *rest in rows:
                key = item_id(item)
                yield insert(table, f"run_id,item_id,{columns}",
                             ["@run", key, *map(literal, (horizon, *rest))],
                             f"NOT EXISTS (SELECT 1 FROM {table} WHERE run_id=@run AND item_id={key} AND {horizon_col}={horizon})")
        for item, signal in signals.items():
            key, value = item_id(item), literal(signal)
            stamp = "JSON_ARRAY(CONCAT('direction_recorded_on:', DATE(UTC_TIMESTAMP() + INTERVAL 9 HOUR)))"
            yield insert("forecast_meta", "run_id,item_id,`signal`,reasons", ["@run", key, value, stamp],
                         f"NOT EXISTS (SELECT 1 FROM forecast_meta WHERE run_id=@run AND item_id={key})")
    yield "COMMIT;"
    yield "SELECT @price_count, @forecast_count;"


def main():
    parser = argparse.ArgumentParser(description="Publish saved settlement prices and selected forecasts to MySQL.")
    parser.add_argument("--forecast-dir", type=Path, help="Publish a specific complete selected forecast export.")
    parser.add_argument("--prices-only", action="store_true", help="Publish prices without processing any forecast.")
    args = parser.parse_args()
    if args.prices_only and args.forecast_dir is not None:
        parser.error("--forecast-dir and --prices-only cannot be combined")
    missing = [key for key in ("DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD") if not os.getenv(key)]
    if missing:
        parser.error("Missing environment variables: " + ", ".join(missing))
    client = shutil.which("mysql")
    if client is None:
        parser.error("The mysql command-line client is required")
    outputs = ROOT / "outputs"
    inputs_root = outputs / "wholesale_verify"
    today = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    selected = args.forecast_dir
    if selected is None and not args.prices_only:
        selected = max((path.parent.parent for path in outputs.glob("selected_forecasts_????????/short/forecast.json")
                        if (path.parent.parent / "mid/forecast.json").is_file()), default=None)
    forecast = forecast_rows(selected, inputs_root, today) if selected else None
    prices = list(price_rows(inputs_root / "mafra_settlement", today))
    command = [client, "--no-defaults", "--protocol=TCP", "--connect-timeout=5",
               "--default-character-set=utf8mb4", "--host", os.environ["DB_HOST"],
               "--port", os.getenv("DB_PORT") or "3306", "--user", os.environ["DB_USER"],
               "--database", os.environ["DB_NAME"], "--batch", "--skip-column-names"]
    try:
        result = subprocess.run(command, input="\n".join(sync_sql(prices, forecast)), text=True,
                                capture_output=True, timeout=60,
                                env={**os.environ, "MYSQL_PWD": os.environ["DB_PASSWORD"]})
    except subprocess.TimeoutExpired:
        parser.exit(1, "MySQL synchronization timed out; retrying is safe.\n")
    if result.returncode:
        error = re.search(r"ERROR (\d+)", result.stderr)
        parser.exit(1, f"MySQL synchronization failed (error {error[1] if error else 'unknown'}).\n")
    inserted_prices, inserted_runs = result.stdout.strip().split("\t")
    print(f"New price rows: {inserted_prices}; new forecast runs: {inserted_runs}")


if __name__ == "__main__":
    main()
