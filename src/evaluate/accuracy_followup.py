import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT
from src.evaluate.short_batch import history_for_item, records
from src.models.compare_all_items import PARAMS
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs

KEYS = ["item", "horizon", "base_date", "target_date"]
ITEMS = ("배추", "양파", "무")
WINDOWS = ("expanding", "recent_2y", "recent_3y", "recent_1y_weight_2x")


def hits(frame, column):
    return np.sign(frame[column] - frame.base_price).eq(np.sign(frame.actual - frame.base_price))


def compare(frame, method, baseline, seed=42):
    hit = hits(frame, method)
    other = hits(frame, baseline)
    error = (frame[method] / frame.actual - 1).abs()
    alternate = (frame[baseline] / frame.actual - 1).abs()
    disagreement = np.sign(frame[method] - frame.base_price).ne(
        np.sign(frame.last_value - frame.base_price))
    baseline_disagreement = np.sign(frame[baseline] - frame.base_price).ne(
        np.sign(frame.last_value - frame.base_price))
    def interval(values):
        blocks = pd.DataFrame({"delta": values, "week": frame.base_date.dt.to_period("W-SUN")})
        b = blocks.groupby("week").delta.agg(["sum", "size"]).to_numpy()
        samples = b[np.random.default_rng(seed).integers(len(b), size=(5000, len(b)))].sum(axis=1)
        return tuple(np.quantile(samples[:, 0] / samples[:, 1], [.025, .975]) * 100)
    acc_lo, acc_hi = interval(hit.astype(float))
    err_lo, err_hi = interval(error)
    delta_lo, delta_hi = interval(hit.astype(int) - other.astype(int))
    return dict(model=method, baseline=baseline, n=len(frame), direction_accuracy_pct=100 * hit.mean(),
                accuracy_ci_low=acc_lo, accuracy_ci_high=acc_hi,
                price_error_pct=100 * error.mean(), error_ci_low=err_lo, error_ci_high=err_hi,
                direction_delta_pp=100 * (hit.mean() - other.mean()),
                direction_delta_ci_low=delta_lo, direction_delta_ci_high=delta_hi,
                price_error_delta_pp=100 * (error.mean() - alternate.mean()),
                disagreement_n=int(disagreement.sum()), disagreement_accuracy_pct=100 * hit[disagreement].mean(),
                last_value_disagreement_accuracy_pct=100 * hits(frame, "last_value")[disagreement].mean(),
                baseline_disagreement_n=int(baseline_disagreement.sum()),
                accuracy_on_baseline_disagreement_pct=100 * hit[baseline_disagreement].mean(),
                last_value_on_baseline_disagreement_pct=100 * hits(frame, "last_value")[baseline_disagreement].mean())


def presentation(targeted_path):
    original = pd.read_parquet(ROOT / "results/all_items_experiment/predictions.parquet")
    current = original.query("split == 'test' and term == 'short' and model == 'current'").copy()
    current["year"] = current.base_date.dt.year
    excluded = current.loc[current.target_date.dt.year.ne(current.year)].copy()
    comparable = current[current.target_date.dt.year.eq(current.year)].copy()
    last = original.query("split == 'test' and term == 'short' and model == 'last_value'")
    frame = comparable.merge(last[KEYS + ["prediction", "base_price", "actual"]], on=KEYS,
                             suffixes=("", "_last"), validate="one_to_one")
    if len(frame) != len(comparable):
        raise ValueError("Missing last-value reference")
    for field in ("actual", "base_price"):
        np.testing.assert_allclose(frame[field], frame[field + "_last"])
    frame = frame.rename(columns={"prediction": "final", "prediction_last": "last_value"})
    frame["moving_average_7"] = frame.base_price
    new = pd.read_parquet(ROOT / "results/final_candidates/new_predictions.parquet")
    batch = pd.read_parquet(ROOT / "results/short_batch/predictions.parquet")
    targeted = pd.read_parquet(targeted_path)
    for rows, item, horizon, model in [
        (new, "무", 10, "volume_only_lgbm"), (new, "양파", 10, "median_volume_lgbm"),
        (batch, "무", 5, "price_lgbm_blend"), (targeted, "배추", 10, "variety_mix"),
    ]:
        ref = rows[rows.item.eq(item) & rows.horizon.eq(horizon) & rows.model.eq(model)]
        route = frame.item.eq(item) & frame.horizon.eq(horizon)
        aligned = frame.loc[route, KEYS + ["actual", "base_price"]].merge(
            ref[KEYS + ["prediction", "actual", "base_price"]], on=KEYS,
            suffixes=("", "_candidate"), validate="one_to_one")
        if len(aligned) != route.sum():
            raise ValueError(f"Missing candidate prices: {item}, {horizon}")
        np.testing.assert_allclose(aligned.actual, aligned.actual_candidate)
        np.testing.assert_allclose(aligned.base_price, aligned.base_price_candidate)
        frame.loc[route, "final"] = aligned.prediction.to_numpy()
    summary = [
        dict(year=year, horizon=horizon, **compare(group, model, "last_value"))
        for year, annual in frame.groupby("year")
        for horizon, group in [(0, annual), *list(annual.groupby("horizon"))]
        for model in ("final", "last_value", "moving_average_7")
    ]
    return frame, pd.DataFrame(summary), excluded


def window_forecasts(history, year):
    boundary = pd.Timestamp(year, 1, 1)
    end = pd.Timestamp(year + 1, 1, 1) if year < 2026 else pd.Timestamp("2026-10-01")
    past = history[history.date.lt(boundary)].date.drop_duplicates().sort_values()
    train = history[history.target_date.lt(past.iloc[-10])]
    test = history[history.date.ge(boundary) & history.target_date.lt(end) & history.horizon.isin([5, 10])]
    columns = PRICE_FEATURES + ["horizon"]
    frame = records(test, "expanding", test.price_per_kg, year).rename(columns={"prediction": "last_value"})
    for name in WINDOWS:
        subset = train
        if name in ("recent_2y", "recent_3y"):
            subset = train[train.date.ge(boundary - pd.DateOffset(years=int(name[7])))]
        weight = np.where(subset.date.ge(boundary - pd.DateOffset(years=1)), 2., 1.) if name.endswith("2x") else None
        model = lgb.LGBMRegressor(**PARAMS, n_estimators=100)
        model.fit(model_inputs(subset, columns), subset.actual / subset.base_price - 1, sample_weight=weight)
        frame[name] = test.base_price.to_numpy() * (1 + model.predict(model_inputs(test, columns)))
    if not np.isfinite(frame[list(WINDOWS)]).all().all() or not frame[list(WINDOWS)].gt(0).all().all():
        raise ValueError("Invalid window forecast")
    return frame.drop(columns="model")


def gated(frame, threshold, weight):
    direction = np.sign(frame["expanding"] - frame.base_price)
    trigger = direction.ne(np.sign(frame.last_value - frame.base_price)) & (
        (frame["expanding"] / frame.base_price - 1).abs().le(threshold))
    return frame["expanding"].where(~trigger, (1 - weight) * frame["expanding"] + weight * frame.last_value)


def experiments():
    prices = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    frames = pd.concat([
        window_forecasts(history_for_item(item, prices), year)
        for item in ITEMS for year in (2024, 2025, 2026)
    ], ignore_index=True)
    calibration = frames[frames.split.eq("2024")]
    trial = frames[~frames.split.eq("2024")].copy()
    selected = max(WINDOWS, key=lambda name: (
        hits(calibration, name).mean(), -(calibration[name] / calibration.actual - 1).abs().mean()))
    best = (-1., -np.inf, 0., 0.)
    for threshold in (.0, .005, .01, .02, .03, .05, .10):
        for weight in (0., .25, .5, .75, 1.):
            valid = calibration.assign(gated=gated(calibration, threshold, weight))
            error = (valid.gated / valid.actual - 1).abs().mean()
            if error - (valid["expanding"] / valid.actual - 1).abs().mean() <= .001:
                best = max(best, (hits(valid, "gated").mean(), -error, threshold, weight))
    _, _, threshold, weight = best
    trial["conditional_blend"] = gated(trial, threshold, weight)
    records_out = []
    for year, annual in trial.groupby("split"):
        groups = [("전체", 0, annual), *[(i, h, g) for (i, h), g in annual.groupby(["item", "horizon"])]]
        for item, horizon, group in groups:
            for name in (*WINDOWS, "conditional_blend", "last_value"):
                records_out.append(dict(year=year, item=item, horizon=horizon,
                                        selected_in_2024=name == selected,
                                        **compare(group, name, "expanding")))
    metrics = pd.DataFrame(records_out)
    total = metrics[metrics.item.eq("전체") & metrics.model.isin([selected, "conditional_blend"])].copy()
    passed = total.direction_delta_pp.gt(0) & total.direction_delta_ci_low.gt(0) & total.price_error_delta_pp.le(.1)
    total["pass_both_years"] = passed.groupby(total.model).transform("all")
    return trial, metrics, total, dict(window=selected, threshold=threshold, weight=weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targeted-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/accuracy_followup")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    frame, table, excluded = presentation(args.targeted_predictions)
    trial, metrics, decisions, settings = experiments()
    args.output.mkdir(parents=True)
    for name, data in {
        "presentation": table, "excluded_cross_year": excluded, "window_metrics": metrics,
        "decision": decisions, "selection_2024": pd.DataFrame([settings]),
    }.items():
        data.to_csv(args.output / f"{name}.csv", index=False)
    frame.to_parquet(args.output / "presentation_predictions.parquet", index=False)
    trial.to_parquet(args.output / "window_predictions.parquet", index=False)
    print(decisions.round(3).to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
