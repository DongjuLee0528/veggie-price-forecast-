import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT
from src.models.compare_all_items import ITEMS

EXPECTED = {(term, item, horizon) for item in ITEMS
            for term, horizons in (("short", (1, 5, 10)), ("mid", (2, 3))) for horizon in horizons}


def model_manifest(paths):
    return [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(set(paths))]


def record(forecast_dir, features_path, actuals_path, artifacts, ledger, now, *, allow_partial=False):
    forecasts = []
    origin = pd.Timestamp(now.date())
    for term in ("short", "mid"):
        report = json.loads((forecast_dir / term / "forecast.json").read_text())
        if pd.Timestamp(report["base_date"]) != origin:
            raise ValueError("Only forecasts with today's origin can enter the prospective ledger")
        for row in report["forecasts"]:
            if pd.Timestamp(row["base_date"]) != origin:
                raise ValueError("Forecast rows must match today's origin")
            if term == "mid":
                start, end = (pd.Timestamp(row[field]) for field in ("target_start", "target_end"))
                if not origin < start <= end:
                    raise ValueError("Mid-term target must be entirely in the future")
            elif row.get("target_date") and pd.Timestamp(row["target_date"]) <= origin:
                raise ValueError("Short-term target must be in the future")
            if not math.isfinite(row["prediction_krw_per_kg"]) or row["prediction_krw_per_kg"] <= 0:
                raise ValueError("Expected positive forecast prices")
            forecasts.append(dict(row, term=term))
    selected = {row["item"] for row in forecasts} if allow_partial else ITEMS
    expected = {route for route in EXPECTED if route[1] in selected}
    if (not expected or len(forecasts) != len(expected)
            or {(row["term"], row["item"], row["horizon"]) for row in forecasts} != expected):
        raise ValueError("Expected 25 short- and mid-term forecast routes")
    features = pd.read_parquet(features_path)
    prices = features.loc[features.date.eq(origin)].set_index("item").base_price
    prices = prices.reindex(sorted(selected))
    if not prices.index.is_unique or not prices.gt(0).all():
        raise ValueError("Missing origin reference prices")
    actuals = pd.read_parquet(actuals_path, columns=["date", "item", "price_per_kg"])
    last = actuals.loc[actuals.date.eq(origin) & actuals.item.isin(ITEMS)].set_index("item").price_per_kg
    last = last.reindex(sorted(selected))
    if not last.index.is_unique or not last.gt(0).all() or not np.isfinite(last).all():
        raise ValueError("Expected all five observed origin prices for the last-value benchmark")
    manifest = model_manifest(artifacts)
    if not manifest or len({row["name"] for row in manifest}) != len(manifest):
        raise ValueError("Provide all model artifacts with distinct names")
    version = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    for row in forecasts:
        row["base_price"] = float(prices.loc[row["item"]])
        row["last_value_krw_per_kg"] = float(last.loc[row["item"]])
    run = dict(recorded_at=now.isoformat(), base_date=str(origin.date()), model_version=version,
               artifacts=manifest, forecasts=forecasts)
    ledger.mkdir(parents=True, exist_ok=True)
    path = ledger / f"forecast_{origin:%Y%m%d}.json"
    payload = json.dumps(run, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return path


def official_prices(paths, through):
    prices = pd.concat([pd.read_csv(path, parse_dates=["date"],
                                   usecols=["date", "item", "price_per_kg"])
                        for path in paths], ignore_index=True)
    prices = prices.loc[prices.item.isin(ITEMS)]
    if (set(prices.item) != ITEMS or prices.date.isna().any()
            or prices.duplicated(["date", "item"]).any()
            or not np.isfinite(prices.price_per_kg).all() or not prices.price_per_kg.gt(0).all()
            or prices.date.max() > through):
        raise ValueError("Invalid official price series or coverage date")
    return prices.sort_values("date")


def score(ledger, prices, through):
    paths = sorted(ledger.glob("forecast_????????.json"))
    if not paths:
        raise ValueError("No prospectively recorded forecasts")
    rows = []
    for path in paths:
        run = json.loads(path.read_text())
        origin = pd.Timestamp(run["base_date"])
        for row in run["forecasts"]:
            daily = prices.loc[prices.item.eq(row["item"]) & prices.date.gt(origin)]
            if row["term"] == "short":
                sample = daily.iloc[row["horizon"] - 1:row["horizon"]]
                if not sample.empty and row.get("target_date") and (
                    sample.date.iloc[0] != pd.Timestamp(row["target_date"])
                ):
                    raise ValueError("Saved short target differs from official trading calendar")
            else:
                start, end = (pd.Timestamp(row[field]) for field in ("target_start", "target_end"))
                sample = daily[daily.date.between(start, end)] if through >= end else daily.iloc[:0]
            actual = float(sample.price_per_kg.mean()) if not sample.empty else None
            forecast = row["prediction_krw_per_kg"]
            base = row["base_price"]
            last = row.get("last_value_krw_per_kg")
            rows.append(dict(
                term=row["term"], item=row["item"], horizon=row["horizon"], base_date=run["base_date"],
                **({"variant": row["variant"]} if "variant" in row else {}),
                model_version=run["model_version"], recorded_at=run["recorded_at"],
                status="scored" if actual is not None else "official_actual_pending",
                actual_end=None if sample.empty else str(sample.date.iloc[-1].date()),
                target_end=None if sample.empty else str((end if row["term"] == "mid" else sample.date.iloc[-1]).date()),
                last_value_krw_per_kg=last,
                last_value_absolute_error_pct=None if actual is None or last is None else abs(last / actual - 1) * 100,
                last_value_direction_hit=None if actual is None or last is None else np.sign(last - base) == np.sign(actual - base),
                forecast_krw_per_kg=forecast, actual_krw_per_kg=actual,
                absolute_error_pct=None if actual is None else abs(forecast / actual - 1) * 100,
                direction_hit=None if actual is None else np.sign(forecast - base) == np.sign(actual - base),
            ))
    return pd.DataFrame(rows)


def evaluate_q4(ledger, official_paths, through, output):
    start, end = pd.Timestamp("2026-10-01"), pd.Timestamp("2026-12-31")
    if output.exists():
        raise FileExistsError("The one-time Q4 holdout report already exists")
    if through < end:
        raise ValueError("Wait for official settlement coverage through 2026-12-31 before opening the holdout")
    prices = official_prices(official_paths, through)
    rows = score(ledger, prices, through)
    recorded = pd.to_datetime(rows.recorded_at, utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    base = pd.to_datetime(rows.base_date)
    target_end = pd.to_datetime(rows.target_end)
    eligible = base.between(start, end) & target_end.between(start, end) & rows.status.eq("scored")
    if (recorded[eligible].dt.normalize().ne(base[eligible]).any()
            or recorded[eligible].dt.normalize().ge(target_end[eligible]).any()):
        raise ValueError("A holdout forecast was not recorded prospectively")
    scored = rows.loc[eligible].copy()
    if scored.empty or scored.last_value_krw_per_kg.isna().any():
        raise ValueError("No eligible prospectively recorded model and baseline forecast pairs")
    if scored.model_version.nunique() != 1:
        raise ValueError("The one-time holdout requires one fixed model version")
    summary = scored.groupby(["term", "horizon"], as_index=False).agg(
        n=("item", "size"), model_error_pct=("absolute_error_pct", "mean"),
        last_value_error_pct=("last_value_absolute_error_pct", "mean"),
        model_direction_rate=("direction_hit", "mean"),
        last_value_direction_rate=("last_value_direction_hit", "mean"),
    )
    metadata = dict(period_start=str(start.date()), period_end=str(end.date()),
                    official_through=str(through.date()), model_version=scored.model_version.iloc[0],
                    official_files=model_manifest(official_paths),
                    sample_count=len(scored))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as work:
        stage = Path(work) / "report"
        stage.mkdir()
        scored.to_csv(stage / "scored_forecasts.csv", index=False)
        summary.to_csv(stage / "metrics.csv", index=False)
        (stage / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        if output.exists():
            raise FileExistsError("The one-time Q4 holdout report already exists")
        stage.rename(output)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Record forward forecasts and score against official settlement CSV exports.")
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("record")
    save.add_argument("--forecast-dir", type=Path, required=True)
    save.add_argument("--features", type=Path, required=True)
    save.add_argument("--origin-prices", type=Path, required=True,
                      help="Daily price parquet containing the last observed price per item at the origin.")
    save.add_argument("--artifact", type=Path, action="append", required=True)
    save.add_argument("--ledger", type=Path, required=True)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--ledger", type=Path, required=True)
    evaluate.add_argument("--official-csv", type=Path, action="append", required=True)
    evaluate.add_argument("--official-through", type=pd.Timestamp, required=True,
                          help="Last date fully covered by the official files across all five items.")
    evaluate.add_argument("--output", type=Path, required=True)
    holdout = commands.add_parser("holdout-q4")
    holdout.add_argument("--ledger", type=Path, required=True)
    holdout.add_argument("--official-csv", type=Path, action="append", required=True)
    holdout.add_argument("--official-through", type=pd.Timestamp, required=True)
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if args.command == "record":
        print(record(args.forecast_dir, args.features, args.origin_prices, args.artifact, args.ledger, now))
    else:
        if args.official_through >= pd.Timestamp(now.date()):
            parser.error("Official coverage must end before today")
        if args.command == "holdout-q4":
            print(evaluate_q4(args.ledger, args.official_csv, args.official_through,
                              ROOT / "results/holdout_2026q4").to_string(index=False))
            return
        prices = official_prices(args.official_csv, args.official_through)
        result = score(args.ledger, prices, args.official_through)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(result.groupby(["term", "status"]).size().to_string())
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
