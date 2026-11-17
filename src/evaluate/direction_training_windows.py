import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT
from src.evaluate.freeze_direction_candidates import FEATURES, HORIZONS, PARAMS, block_interval, price_inputs
from src.models.compare_all_items import ITEMS


WINDOWS = {"expanding": None, "recent_2y": 2, "recent_3y": 3}


def predict_windows(data, test, year):
    start = pd.Timestamp(year, 1, 1)
    embargo = data.loc[data.date.lt(start), "date"].iloc[-10]
    available = data[data.target_date.lt(embargo)]
    if available.empty or not available.target_date.max() < embargo < test.date.min():
        raise ValueError("Invalid time separation")
    predictions = {}
    for name, years in WINDOWS.items():
        train = available if years is None else available[available.date.ge(start - pd.DateOffset(years=years))]
        if train.actual_direction.nunique() < 2:
            raise ValueError(f"Insufficient training labels: {year}, {name}")
        model = lgb.LGBMClassifier(**PARAMS).fit(train[FEATURES], train.actual_direction)
        predictions[name] = model.predict(test[FEATURES])
    return predictions


def evaluate(data, archive, held):
    validation = data[data.date.ge("2024-01-01") & data.target_date.lt("2025-01-01")]
    calibration = predict_windows(data, validation, 2024)
    rates = {name: float(np.mean(pred == validation.actual_direction)) for name, pred in calibration.items()}
    winner = max(WINDOWS, key=rates.get)
    evaluation = archive.merge(data[["date", "target_date", "price_per_kg", "actual_direction", *FEATURES]],
                               left_on=["base_date", "target_date"], right_on=["date", "target_date"],
                               validate="one_to_one")
    if len(evaluation) != len(archive) or set(evaluation.date.dt.year) != {2025, 2026}:
        raise ValueError("Incomplete paired evaluation")
    np.testing.assert_allclose(evaluation.price_per_kg, evaluation.last_value)
    np.testing.assert_array_equal(np.sign(evaluation.actual - evaluation.last_value), evaluation.actual_direction)

    frames = []
    for year, frame in evaluation.groupby(evaluation.date.dt.year):
        frame = frame.assign(**predict_windows(data, frame, int(year)))
        frame["reference"] = np.sign(frame.avg7) if held else np.sign(frame.final - frame.last_value)
        frame["selected"] = frame[winner]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), winner, rates


def main():
    parser = argparse.ArgumentParser(description="Compare direction training windows across all fifteen short-term routes.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/direction_training_windows")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    reference = pd.read_parquet(ROOT / "results/accuracy_followup/presentation_predictions.parquet")
    years = reference.target_date.dt.year
    reference = reference[years.isin([2025, 2026]) & reference.target_date.dt.month.le(9)
                          & reference.base_date.dt.year.eq(years)]
    daily = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    potato = pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet")
    tables, selections, predictions = [], [], []
    methods = ["reference", *WINDOWS, "selected"]
    for item in sorted(ITEMS):
        for horizon in HORIZONS:
            history = price_inputs(item, potato if item == "감자 수미" else daily, horizon)
            subset = reference[reference.item.eq(item) & reference.horizon.eq(horizon)]
            held = horizon == 1 and item in {"무", "양파"}
            paired, winner, rates = evaluate(history, subset, held)
            benchmark = "mean7" if held else "current_price_forecast"
            selections.append(dict(item=item, horizon=horizon, window=winner, benchmark=benchmark, **rates))
            for period, sample in [*paired.groupby(paired.base_date.dt.year), ("pooled", paired)]:
                hit = sample[methods].eq(sample.actual_direction, axis=0).mean() * 100
                low, high = block_interval(sample.assign(candidate_hit=sample.selected.eq(sample.actual_direction),
                                                         benchmark_hit=sample.reference.eq(sample.actual_direction)),
                                           "benchmark_hit")
                tables.append(dict(item=item, horizon=horizon, year=period, n=len(sample),
                                   selected_window=winner, benchmark=benchmark, **hit.to_dict(),
                                   delta_pp=hit.selected-hit.reference, ci_low_pp=low, ci_high_pp=high))
            predictions.append(paired[["item", "horizon", "base_date", "target_date", "actual_direction", *methods]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp) / "report"
        stage.mkdir()
        pd.DataFrame(tables).to_csv(stage / "metrics.csv", index=False)
        pd.DataFrame(selections).to_csv(stage / "selection_2024.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_parquet(stage / "paired_predictions.parquet", index=False)
        plan = dict(features=FEATURES, parameters=PARAMS, windows=WINDOWS, selection_year=2024,
                    evaluation_years=[2025, 2026], evaluation_months="January–September",
                    label="sign(target price - origin observed price)", embargo_trading_days=10)
        (stage / "settings.json").write_text(json.dumps(plan, indent=2) + "\n")
        stage.rename(args.output)
    print(f"Saved 15 route comparisons to {args.output}")


if __name__ == "__main__":
    main()
