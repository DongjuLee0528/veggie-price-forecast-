import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT
from src.models.compare_all_items import ITEMS


def read_actuals(directory, origin, as_of):
    dates = pd.date_range(origin + pd.Timedelta(days=1), as_of)
    frames = []
    for date in dates:
        path = directory / f"daily_{date:%Y%m%d}.csv"
        if not path.is_file():
            continue
        daily = pd.read_csv(path, parse_dates=["date"])
        if (len(daily) != len(ITEMS) or set(daily.item) != ITEMS
                or not daily.date.eq(date).all()):
            raise ValueError(f"Invalid daily settlement file: {path.name}")
        frames.append(daily)
    if not frames:
        return {}
    actuals = pd.concat(frames, ignore_index=True)
    return {(row.item, row.date): row for row in actuals.itertuples()}


def settle(record, actuals, origin, as_of):
    mid = record.term == "mid"
    if mid and pd.Timestamp(record.target_end) > as_of:
        return "target_period_open", None, None
    start = pd.Timestamp(record.target_start) if mid else origin + pd.Timedelta(days=1)
    end = pd.Timestamp(record.target_end) if mid else as_of
    observed = []
    for day in pd.date_range(start, end):
        row = actuals.get((record.item, day))
        if row is None:
            return "settlement_pending", None, None
        if row.unconfirmed_markets or row.review_markets:
            return "settlement_review_required", None, None
        price = row.provisional_price_per_kg
        if np.isfinite(price) and price > 0:
            observed.append((day, float(price)))
            if not mid and len(observed) == record.horizon:
                return "scored_provisional", day, float(price)
        elif row.observed_markets:
            return "settlement_review_required", None, None
    if mid and observed:
        return "scored_provisional", end, float(np.mean([value for _, value in observed]))
    return "trading_horizon_pending", None, None


def main():
    parser = argparse.ArgumentParser(description="Score all saved forecasts when provisional settlements are available.")
    parser.add_argument("--origin", type=pd.Timestamp, default=pd.Timestamp("2026-11-13"))
    today = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    parser.add_argument("--as-of", type=pd.Timestamp, default=today)
    parser.add_argument("--refresh", action="store_true", help="Collect settlements before scoring.")
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    origin, as_of = args.origin.normalize(), args.as_of.normalize()
    if as_of < origin:
        parser.error("The scoring date must be on or after the forecast origin")
    out = ROOT / "outputs"
    stamp = f"{origin:%Y%m%d}"
    directory = out / "wholesale_verify"
    short = pd.read_csv(out / f"short_forecast_{stamp}/predictions.csv").assign(term="short")
    mid = pd.read_csv(out / f"mid_forecast_{stamp}/predictions.csv").assign(term="mid")
    candidate_dir = args.candidate_dir or out / f"candidate_forecasts_{stamp}"
    candidates = pd.read_csv(candidate_dir / "comparison.csv").rename(
        columns={"candidate_price": "prediction_krw_per_kg", "candidate_model": "model",
                 "status": "candidate_input_status"}
    )
    base = pd.concat([short, mid], ignore_index=True).assign(variant="existing")
    forecasts = pd.concat([base, candidates.assign(variant="candidate")], ignore_index=True)
    keys = ["term", "item", "horizon"]
    expected = {(term, item, horizon) for item in ITEMS
                for term, horizons in (("short", (1, 5, 10)), ("mid", (2, 3))) for horizon in horizons}
    if (len(base) != len(expected) or set(map(tuple, base[keys].to_numpy())) != expected
            or len(candidates) != 5
            or forecasts.duplicated(keys + ["variant"]).any()
            or not pd.to_datetime(forecasts.base_date).eq(origin).all()
            or not np.isfinite(forecasts.prediction_krw_per_kg).all()
            or not forecasts.prediction_krw_per_kg.gt(0).all()):
        raise ValueError("Unexpected forecast routes, origins, or prices")
    reference = candidates.merge(base, on=keys, suffixes=("", "_base"), validate="one_to_one")
    if len(reference) != len(candidates):
        raise ValueError("Candidate has no matching existing forecast")
    np.testing.assert_allclose(reference.existing_price, reference.prediction_krw_per_kg_base)
    features = pd.read_parquet(directory / f"forecast_inputs_{stamp}" / "price_features.parquet")
    base_price = features.loc[features.date.eq(origin)].set_index("item").base_price
    if set(base_price.index) != ITEMS or not base_price.index.is_unique or not base_price.gt(0).all():
        raise ValueError("Invalid origin reference prices")
    if args.refresh:
        start = origin + pd.Timedelta(days=1)
        end = min(as_of, today - pd.Timedelta(days=1))
        if end >= start and os.getenv("MAFRA_WHOLESALE_API_KEY", "").strip():
            command = [sys.executable, "-m", "scripts.wholesale_verify.compare_national",
                       "--collect", f"{start:%Y-%m-%d}", f"{end:%Y-%m-%d}", "--refresh-empty"]
            if subprocess.run(command, cwd=ROOT, check=False).returncode:
                print("정산 조회가 중단되어 저장된 자료로 채점합니다.", flush=True)
        elif end >= start:
            print("MAFRA_WHOLESALE_API_KEY가 전달되지 않아 저장된 자료로 채점합니다.", flush=True)

    actuals = read_actuals(directory / "mafra_settlement", origin, as_of)
    results = []
    for row in forecasts.itertuples():
        status, actual_date, actual = settle(row, actuals, origin, as_of)
        forecast = row.prediction_krw_per_kg
        baseline = base_price.loc[row.item]
        results.append(dict(
            term=row.term, item=row.item, horizon=row.horizon, variant=row.variant, model=row.model,
            candidate_input_status=row.candidate_input_status, base_date=origin.date(),
            generated_at=getattr(row, "generated_at", None),
            target_start=row.target_start, target_end=row.target_end,
            actual_date=actual_date, status=status, forecast_krw_per_kg=forecast,
            actual_krw_per_kg=actual, absolute_error_pct=None if actual is None else abs(forecast / actual - 1) * 100,
            direction_hit=None if actual is None else np.sign(forecast - baseline) == np.sign(actual - baseline),
        ))
    report = pd.DataFrame(results)
    target = args.output or out / f"forecast_scores_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"scores_as_of_{as_of:%Y%m%d}.csv"
    temporary = path.with_suffix(".tmp")
    report.to_csv(temporary, index=False)
    temporary.replace(path)
    print(report.groupby(["variant", "status"]).size().to_string())
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
