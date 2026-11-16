import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate.prospective_forecasts import official_prices, score


def record(source, ledger, now):
    now = now.tz_convert("Asia/Seoul")
    origin = pd.Timestamp(now.date())
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
    save.add_argument("--source", type=Path, required=True, help="Today's fixed-model ledger JSON.")
    save.add_argument("--ledger", type=Path, required=True, help="Separate candidate ledger directory.")
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--ledger", type=Path, required=True)
    evaluate.add_argument("--official-csv", type=Path, action="append", required=True)
    evaluate.add_argument("--official-through", type=pd.Timestamp, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if args.command == "record":
        print(record(args.source, args.ledger, now))
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
