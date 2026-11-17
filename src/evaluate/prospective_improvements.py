import argparse
import hashlib
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.prospective_forecasts import ROOT, official_prices, score
from src.features.build import price_features
from src.models.train_lgbm import model_inputs


def onion_forecast(origin):
    settlement = ROOT / "outputs/wholesale_verify/mafra_settlement"
    first_api_date = pd.Timestamp("2026-10-01")
    frames = []
    for day in pd.date_range(first_api_date, origin):
        path = settlement / f"daily_{day:%Y%m%d}.csv"
        if not path.is_file():
            raise ValueError(f"Missing settlement report for {day.date()}")
        row = pd.read_csv(path, parse_dates=["date"]).query("item == '양파'")
        if len(row) != 1 or not row.date.eq(day).all():
            raise ValueError(f"Invalid onion settlement report for {day.date()}")
        observed = row.observed_markets.gt(0)
        if day == origin and (not observed.all() or row.review_markets.ne(0).any()
                              or row.unconfirmed_markets.ne(0).any()):
            raise ValueError(f"No usable onion settlement for {day.date()}")
        frames.append(row[observed | row.review_markets.gt(0) | row.unconfirmed_markets.gt(0)])
    api = pd.concat(frames, ignore_index=True).rename(columns={"provisional_price_per_kg": "price_per_kg"})
    columns = ["date", "item", "price_per_kg"]
    official = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet", columns=columns)
    history = pd.concat([official[official.date.lt(first_api_date) & official.item.eq("양파")],
                         api[columns]], ignore_index=True).sort_values("date")
    if history.date.duplicated().any():
        raise ValueError("Duplicate onion price dates")
    features = price_features(history).tail(1).assign(horizon=5).set_index("item")
    directory = ROOT / "results/lgbm_production"
    manifest = json.loads((directory / "run.json").read_text())
    route, = [row for row in manifest["selected"] if (row["item"], row["horizon"]) == ("양파", 5)]
    if manifest["status"] != "complete" or pd.Timestamp(manifest["cutoff"]) >= origin or route["model"] != "price_pooled":
        raise ValueError("Expected the fixed onion production model")
    model_bytes = (directory / route["file"]).read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    artifact, = [entry for entry in manifest["models"] if entry["file"] == route["file"]]
    if digest != artifact["sha256"]:
        raise ValueError("Onion production model checksum mismatch")
    inputs = model_inputs(features, route["inputs"])
    if not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise ValueError("Missing or invalid onion price features")
    booster = lgb.Booster(model_str=model_bytes.decode("utf-8"))
    change, = booster.predict(inputs, num_iteration=route["best_iteration"], num_threads=2)
    base = float(features.base_price.iloc[0])
    row = dict(term="short", item="양파", horizon=5, base_date=str(origin.date()), target_date=None,
               model=route["model"], source="mafra_api_provisional", base_price=base,
               last_value_krw_per_kg=float(history.price_per_kg.iloc[-1]), prediction_krw_per_kg=base * (1 + float(change)))
    version = hashlib.sha256(json.dumps([route, digest], sort_keys=True).encode()).hexdigest()
    return dict(base_date=str(origin.date()), model_version=version, forecasts=[row])


def record(source, ledger, now):
    now = now.tz_convert("Asia/Seoul")
    origin = pd.Timestamp(now.date())
    if source is None:
        run = onion_forecast(origin)
        run["recorded_at"] = now.isoformat()
        payload = json.dumps(run, sort_keys=True).encode()
    else:
        payload = source.read_bytes()
        run = json.loads(payload)
        recorded = pd.Timestamp(run["recorded_at"]).tz_convert("Asia/Seoul")
        if pd.Timestamp(run["base_date"]) != origin or recorded.date() != now.date() or recorded > now:
            raise ValueError("Expected a model forecast recorded earlier today")
    matches = [row for row in run["forecasts"]
               if (row["term"], row["item"], row["horizon"]) == ("short", "양파", 5)]
    if len(matches) != 1:
        raise ValueError("Expected one onion forecast at five trading days")
    row, = matches
    if pd.Timestamp(row["base_date"]) != origin or (
        row.get("target_date") and pd.Timestamp(row["target_date"]) <= origin
    ):
        raise ValueError("Expected today's forecast with a future target")
    current = row["prediction_krw_per_kg"]
    last = row["last_value_krw_per_kg"]
    base = row["base_price"]
    if not all(math.isfinite(price) and price > 0 for price in (current, last, base)):
        raise ValueError("Invalid onion candidate inputs")
    disagree = np.sign(current - base) != np.sign(last - base)
    prices = {"current": current, "last_value": last,
              "conditional_half_blend": current + (last - current) / 2 if disagree else current}
    variants = [dict(row, variant=name, prediction_krw_per_kg=price)
                for name, price in prices.items()]
    run.update(recorded_at=now.isoformat(), forecasts=variants,
               source_forecast_sha256=hashlib.sha256(payload).hexdigest(),
               candidate_rule_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    data = json.dumps(run, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    ledger.mkdir(parents=True, exist_ok=True)
    path = ledger / f"forecast_{origin:%Y%m%d}.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(data)
    return path


def main():
    parser = argparse.ArgumentParser(description="Record and score five-trading-day onion forecast candidates.")
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("record")
    save.add_argument("--source", type=Path, help="Use an existing fixed-model ledger JSON for today.")
    save.add_argument("--ledger", type=Path, required=True, help="Separate candidate ledger directory.")
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--ledger", type=Path, required=True)
    evaluate.add_argument("--official-csv", type=Path, action="append", required=True)
    evaluate.add_argument("--official-through", type=pd.Timestamp, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if args.command == "record":
        try:
            print(record(args.source, args.ledger, now))
        except (ValueError, FileExistsError) as exc:
            parser.exit(1, f"{exc}\n")
        return
    if args.official_through >= pd.Timestamp(now.date()):
        parser.error("Official coverage must end before today")
    runs = [json.loads(path.read_text()) for path in args.ledger.glob("forecast_????????.json")]
    for field in ("model_version", "candidate_rule_sha256"):
        if len({run.get(field) for run in runs}) != 1 or not runs[0].get(field):
            parser.error("Expected one fixed model and candidate rule in the candidate ledger")
    prices = official_prices(args.official_csv, args.official_through)
    rows = score(args.ledger, prices, args.official_through)
    scored = rows.loc[rows.status.eq("scored")]
    summary = scored.groupby(["item", "horizon", "variant"]).agg(
        n=("actual_krw_per_kg", "size"),
        direction_accuracy_pct=("direction_hit", lambda hits: 100 * hits.mean()),
        price_error_pct=("absolute_error_pct", "mean"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)
    print(summary.to_string() if not summary.empty else "No official targets available yet.")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
