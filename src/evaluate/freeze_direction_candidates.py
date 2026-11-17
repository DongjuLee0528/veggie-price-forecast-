import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT
from src.evaluate.short_batch import history_for_item
from src.models.compare_all_items import ITEMS


HORIZONS = (1, 5, 10)
FEATURES = [
    "ret1", "ret3", "ret7", "ret14", "ret28", "avg3", "avg7", "avg14", "avg28",
    "vol_28", "yoy", "volume_ratio", "weekday", "sin", "cos",
]
PARAMS = dict(n_estimators=100, num_leaves=7, min_child_samples=40, learning_rate=.03,
              n_jobs=2, verbosity=-1, random_state=42)
REFERENCE = pd.Timestamp("2024-01-01")


def direction_inputs(frame):
    frame = frame.sort_values("date").copy()
    price = frame.price_per_kg
    for lag in (1, 3, 7, 14, 28):
        frame[f"ret{lag}"] = price.pct_change(lag, fill_method=None)
    for window in (3, 7, 14, 28):
        frame[f"avg{window}"] = price.rolling(window).mean() / price - 1
    frame["volume_ratio"] = frame.volume_kg / frame.volume_kg.shift().rolling(14).median() - 1
    frame["weekday"] = frame.date.dt.day_of_week
    phase = 2 * np.pi * frame.date.dt.day_of_year / 365.25
    frame["sin"], frame["cos"] = np.sin(phase), np.cos(phase)
    if np.isinf(frame[FEATURES].to_numpy()).any():
        raise ValueError("Infinite direction input")
    return frame


def price_inputs(item, daily, horizon):
    frame = history_for_item(item, daily).loc[lambda rows: rows.horizon.eq(horizon)]
    frame = direction_inputs(frame)
    return frame.assign(actual_direction=np.sign(frame.actual - frame.price_per_kg).astype(int))


def block_interval(frame, baseline):
    blocks = (frame.assign(delta=(frame.candidate_hit.astype(int) - frame[baseline].astype(int)) * 100, n=1)
              .groupby(frame.base_date.dt.to_period("W"))[["delta", "n"]].sum().to_numpy())
    rng = np.random.default_rng(42)
    draws = blocks[rng.integers(len(blocks), size=(12000, len(blocks)))].sum(axis=1)
    return np.quantile(draws[:, 0] / draws[:, 1], [.025, .975])


def evaluate(item, horizon, daily, archive, output):
    data = price_inputs(item, daily, horizon)
    boundary = data.loc[data.date.lt(REFERENCE), "date"].iloc[-10]
    train = data.loc[data.target_date.lt(boundary)]
    if train.empty or train.actual_direction.nunique() < 2:
        raise ValueError(f"Insufficient training data: {item}, {horizon}")
    model = lgb.LGBMClassifier(**PARAMS).fit(train[FEATURES], train.actual_direction)
    valid = data.loc[data.date.ge(REFERENCE) & data.target_date.lt("2025-01-01")]
    frame = archive.merge(data[["date", "target_date", "price_per_kg", "actual_direction", *FEATURES]],
                          left_on=["base_date", "target_date"], right_on=["date", "target_date"],
                          validate="one_to_one")
    if len(frame) != len(archive) or set(frame.base_date.dt.year) != {2025, 2026}:
        raise ValueError(f"Incomplete evaluation: {item}, {horizon}")
    np.testing.assert_allclose(frame.price_per_kg, frame.last_value)
    np.testing.assert_array_equal(np.sign(frame.actual - frame.last_value), frame.actual_direction)
    held = horizon == 1 and item in {"무", "양파"}
    frame["current_direction"] = 0 if held else np.sign(frame.final - frame.last_value)
    frame["mean7_direction"] = np.sign(frame.avg7)
    frame["majority_direction"] = train.actual_direction.mode().iloc[0]
    frame["candidate_direction"] = model.predict(frame[FEATURES])
    for name in ("current", "mean7", "majority", "candidate"):
        frame[f"{name}_hit"] = frame[f"{name}_direction"].eq(frame.actual_direction)
    columns = [name for name in frame if name.endswith("_hit")]
    annual = frame.groupby(frame.base_date.dt.year.rename("year"))[columns].mean() * 100
    annual["n"] = frame.groupby(frame.base_date.dt.year).size()
    decision = "keep_direction_pending" if held else "keep_existing_direction"
    baselines = ["mean7_hit", "majority_hit"] if held else ["current_hit", "mean7_hit", "majority_hit"]
    gates = {}
    for name in baselines:
        low, high = block_interval(frame, name)
        gates[name] = dict(delta_pp=float((frame.candidate_hit.mean() - frame[name].mean()) * 100),
                           ci_low_pp=float(low), ci_high_pp=float(high),
                           wins_each_year=bool(annual.candidate_hit.gt(annual[name]).all()))
    if all(gate["wins_each_year"] and gate["ci_low_pp"] > 0 for gate in gates.values()):
        decision = "candidate_for_new_period"
    filename = f"{item}_{horizon}.txt"
    path = output / filename
    model.booster_.save_model(str(path))
    restored = lgb.Booster(model_file=str(path))
    probability = restored.predict(frame[FEATURES])
    labels = model.classes_.take(probability.argmax(axis=1) if probability.ndim == 2 else (probability > .5).astype(int))
    np.testing.assert_array_equal(labels, frame.candidate_direction)
    manifest = dict(item=item, horizon=int(horizon), file=filename, classes=model.classes_.tolist(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(), train_rows=len(train),
                    train_target_through=str(train.target_date.max().date()),
                    validation_accuracy=float(np.mean(model.predict(valid[FEATURES]) == valid.actual_direction)),
                    current_direction_pending=held, decision=decision, retrospective_comparisons=gates)
    return annual.reset_index().assign(item=item, horizon=horizon, decision=decision), manifest


def main():
    parser = argparse.ArgumentParser(description="Freeze and backtest direction candidates for all five items.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/direction_candidates_observed_price")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    reference = pd.read_parquet(ROOT / "results/accuracy_followup/presentation_predictions.parquet")
    year = reference.target_date.dt.year
    reference = reference[year.isin([2025, 2026]) & reference.target_date.dt.month.le(9)
                          & reference.base_date.dt.year.eq(year)]
    daily = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    potato = pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp) / "report"
        stage.mkdir()
        metrics, routes = [], []
        for item in sorted(ITEMS):
            for horizon in HORIZONS:
                subset = reference[reference.item.eq(item) & reference.horizon.eq(horizon)]
                table, route = evaluate(item, horizon, potato if item == "감자 수미" else daily, subset, stage)
                metrics.append(table)
                routes.append(route)
        pd.concat(metrics, ignore_index=True).to_csv(stage / "metrics.csv", index=False)
        manifest = dict(features=FEATURES, parameters=PARAMS, label="sign(target_price - origin_observed_price)",
                        evaluation="Retrospective, January through September 2025 and 2026",
                        prospective_status="pending", routes=routes)
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        stage.rename(args.output)
    print(f"Saved {len(routes)} candidate models and paired retrospective scores to {args.output}")


if __name__ == "__main__":
    main()
