import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate.prospective_forecasts import official_prices, score
from src.features.targets import dekad_bounds, dekad_index


FIRST_ORIGIN = pd.Timestamp("2027-01-01")


def cabbage_baselines(row, origin, history_path):
    daily = pd.read_parquet(history_path, columns=["date", "item", "price_per_kg"])
    daily = daily.loc[daily.item.eq("배추") & daily.date.le(origin)]
    if daily.date.duplicated().any() or not np.isfinite(daily.price_per_kg).all() or not daily.price_per_kg.gt(0).all():
        raise ValueError("Invalid cabbage daily history")
    means = daily.groupby(daily.date.map(dekad_index)).price_per_kg.mean()
    origin_period = dekad_index(origin)
    completed = origin_period - int(origin < dekad_bounds(origin_period)[1])
    target = dekad_index(row["target_start"])
    if target != origin_period + 2 or pd.Timestamp(row["target_end"]) != dekad_bounds(target)[1]:
        raise ValueError("Unexpected cabbage target period")
    last, previous_last, previous_target = means.reindex([completed, completed - 36, target - 36])
    if not np.isfinite([last, previous_last, previous_target]).all():
        raise ValueError("Missing completed or prior-year cabbage dekad prices")
    return last, previous_target * last / previous_last


def record(source, history, ledger, now):
    origin = pd.Timestamp(now.date())
    if origin < FIRST_ORIGIN:
        raise ValueError("Candidate recording begins on 2027-01-01")
    payload = source.read_bytes()
    run = json.loads(payload)
    recorded = pd.Timestamp(run["recorded_at"]).tz_convert("Asia/Seoul")
    if pd.Timestamp(run["base_date"]) != origin or recorded.date() != now.date() or recorded > now:
        raise ValueError("Expected a model forecast recorded earlier today")
    variants = []
    for term, item, horizon in (("short", "양파", 5), ("mid", "배추", 2)):
        matches = [row for row in run["forecasts"]
                   if (row["term"], row["item"], row["horizon"]) == (term, item, horizon)]
        if len(matches) != 1:
            raise ValueError(f"Expected one forecast for {item} {horizon}")
        row, = matches
        current = row["prediction_krw_per_kg"]
        base = row["base_price"]
        if term == "short":
            last = row["last_value_krw_per_kg"]
            disagree = np.sign(current - base) != np.sign(last - base)
            prices = {"current": current, "last_value": last,
                      "conditional_half_blend": (current + last) / 2 if disagree else current}
        else:
            last, seasonal = cabbage_baselines(row, origin, history)
            prices = {"current": current, "last_completed": last, "adjusted_seasonal": seasonal,
                      "seasonal_half_blend": (current + seasonal) / 2}
        if not all(math.isfinite(price) and price > 0 for price in [base, *prices.values()]):
            raise ValueError(f"Invalid candidate inputs for {item} {horizon}")
        variants.extend(dict(row, variant=name, prediction_krw_per_kg=float(price))
                        for name, price in prices.items())
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
    parser = argparse.ArgumentParser(description="Record and score post-Q4 onion and cabbage forecast candidates.")
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("record")
    save.add_argument("--source", type=Path, required=True, help="Today's fixed-model ledger JSON.")
    save.add_argument("--history", type=Path, required=True, help="Daily prices available at forecast time.")
    save.add_argument("--ledger", type=Path, required=True, help="Separate candidate ledger directory.")
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--ledger", type=Path, required=True)
    evaluate.add_argument("--official-csv", type=Path, action="append", required=True)
    evaluate.add_argument("--official-through", type=pd.Timestamp, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if args.command == "record":
        print(record(args.source, args.history, args.ledger, now))
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
