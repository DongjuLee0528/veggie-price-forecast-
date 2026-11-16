import argparse
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.evaluate.short_batch import WEIGHTS, confidence_rows, history_for_item, records, segments
from src.models.compare_all_items import PARAMS
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs

KEYS = ["item", "horizon", "base_date", "target_date"]
SEASONS = {"spring": "봄", "highland": "고냉지|고랭지|여름", "autumn": "가을|김장",
           "winter": "월동|겨울"}
SHORT = {
    "건고추": ("low_volume_smoothing", [7, 28], [5, 10]),
    "배추": ("variety_mix", list(SEASONS), [5, 10]),
    "무": ("variety_mix", list(SEASONS), [5, 10]),
    "감자 수미": ("volume_only", [7, 28], [5]),
}
COMPONENTS = ["price_lgbm", "sarimax", "ets"]


def variety_shares():
    path = ROOT / "data/raw/price/settlement_rows.parquet"
    raw = pd.read_parquet(path, columns=["date", "item", "variety", "unit_qty", "kg_per_unit"],
                          filters=[("item", "in", ["배추", "무"])])
    kg = raw.unit_qty * raw.kg_per_unit
    for name, pattern in SEASONS.items():
        raw[name] = kg.where(raw.variety.str.contains(pattern, na=False), 0)
    totals = raw.assign(kg=kg).groupby(["date", "item"])[["kg", *SEASONS]].sum()
    return totals[list(SEASONS)].div(totals.kg, axis=0).reset_index()


def short_experiments(dry_run):
    history = pd.concat([
        pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet"),
        pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet"),
    ])
    shares = variety_shares()
    archive = pd.read_parquet(ROOT / "results/all_items_experiment/predictions.parquet")
    archive = archive.query("split == 'test' and term == 'short' and model == 'current'")
    outputs, choices, confidence = [], [], []
    for item, (name, settings, horizons) in SHORT.items():
        rows = history_for_item(item, history).sort_values("date", kind="stable")
        if name == "variety_mix":
            rows = rows.merge(shares, on=["date", "item"], how="left", validate="many_to_one")
            extra = settings
        else:
            day = rows.drop_duplicates("date").set_index("date")
            extra = []
            for window in settings:
                feature = f"{name}_{window}"
                volume = day.volume_kg
                if name == "low_volume_smoothing":
                    weight = volume.pow(.5)
                    series = (day.price_per_kg * weight).rolling(window).sum() / weight.rolling(window).sum()
                    series = series / day.base_price
                else:
                    series = volume / volume.shift().rolling(window).mean()
                rows[feature] = rows.date.map(series)
                extra.append(feature)
        for year in (2025, 2026):
            train, valid, test = [frame[frame.horizon.isin(horizons)] for frame in segments(rows, year)]
            if any(frame.empty or not np.isfinite(frame[extra].to_numpy(dtype=float)).any(axis=0).all()
                   for frame in (train, valid, test)):
                raise ValueError(f"Missing experiment inputs: {item}, {year}")
            reference = records(test, "archived_current", test.price_per_kg, year)
            aligned = reference.drop(columns="prediction").merge(
                archive[KEYS + ["actual", "base_price", "prediction"]], on=KEYS,
                suffixes=("", "_archive"), validate="one_to_one",
            )
            if len(aligned) != len(test):
                raise ValueError(f"Incomplete archived prices: {item}, {year}")
            np.testing.assert_allclose(aligned.actual, aligned.actual_archive)
            np.testing.assert_allclose(aligned.base_price, aligned.base_price_archive)
            print(f"{item} {year}: {len(test)} matched short-term samples", flush=True)
            if dry_run:
                continue
            outputs.extend([aligned[reference.columns],
                            records(test, "last_value", test.price_per_kg, year)])
            for model_name, features in (("price_lgbm", []), (name, extra)):
                columns = PRICE_FEATURES + ["horizon", *features]
                model = lgb.LGBMRegressor(**PARAMS, n_estimators=100)
                model.fit(model_inputs(train, columns), train.actual / train.base_price - 1)
                vp = valid.base_price * (1 + model.predict(model_inputs(valid, columns)))
                tp = test.base_price * (1 + model.predict(model_inputs(test, columns)))
                outputs.append(records(test, model_name, tp, year))
                for horizon in horizons:
                    v, t = valid.horizon.eq(horizon), test.horizon.eq(horizon)
                    valid_rows, test_rows = valid.loc[v], test.loc[t]
                    def mixed(frame, prediction, weight):
                        return (1 - weight) * prediction + weight * frame.price_per_kg
                    weight = min(WEIGHTS, key=lambda w: (
                        mixed(valid_rows, vp[v], w) / valid_rows.actual - 1
                    ).abs().mean())
                    blend = model_name + "_blend"
                    calibration = records(valid_rows, blend, mixed(valid_rows, vp[v], weight), year)
                    evaluation = records(test_rows, blend, mixed(test_rows, tp[t], weight), year)
                    outputs.append(evaluation)
                    confidence.extend(confidence_rows(calibration, evaluation, blend, year))
                    choices.append(dict(item=item, year=year, horizon=horizon, model=blend, last_value_weight=weight))
    return outputs, choices, confidence


def mid_experiments():
    results = ROOT / "results"
    original = pd.read_parquet(results / "lgbm_mid_comparison/predictions.parquet")
    ets = pd.read_parquet(results / "mid_tuning/predictions.parquet")
    potato = pd.read_parquet(results / "potato_sumi_tuning/predictions.parquet").query("term == 'mid'")
    old_potato = potato.query("model == 'holt_winters'").assign(model="archived_current")
    components = pd.concat([
        original[original.model.isin(COMPONENTS)],
        ets.query("model == 'ets'"),
        ets.query("model == 'blend_with_ets'").assign(model="archived_current"),
        potato.query("model in ['price_lgbm', 'sarimax', 'ets_no_trend']").replace(
            {"model": {"ets_no_trend": "ets"}}),
        old_potato,
    ])
    index = KEYS + ["split", "actual", "base_price"]
    wide = components.pivot(index=index, columns="model", values="prediction").reset_index()
    if wide[COMPONENTS + ["archived_current"]].isna().any().any():
        raise ValueError("Mid-term models must share dates, actuals, and base prices")
    weights = [tuple(value / 4 for value in steps)
               for steps in product(range(5), repeat=3) if sum(steps) == 4]
    outputs, choices = [], []
    for (item, horizon), group in wide.groupby(["item", "horizon"]):
        for year in (2025, 2026):
            valid = group[group.base_date.dt.year.eq(year - 1) & group.target_date.lt(f"{year}-01-01")]
            test = group[group.base_date.dt.year.eq(year) & group.target_date.lt(
                min(f"{year + 1}-01-01", "2026-10-01"))]
            if valid.empty or test.empty:
                raise ValueError(f"Missing mid-term rows: {item} {horizon} {year}")
            weight = min(weights, key=lambda w: np.mean(np.abs(valid[COMPONENTS].to_numpy() @ w / valid.actual - 1)))
            trial = test.assign(blend=test[COMPONENTS].to_numpy() @ weight, split=str(year))
            outputs.append(trial.melt(id_vars=index, value_vars=[*COMPONENTS, "archived_current", "blend"],
                                      var_name="model", value_name="prediction"))
            choices.append(dict(item=item, horizon=horizon, year=year, **dict(zip(COMPONENTS, weight))))
    return pd.concat(outputs), pd.DataFrame(choices)


def main():
    parser = argparse.ArgumentParser(description="Run targeted short- and mid-term price experiments.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/targeted_batch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error("Use a new output directory")
    mid, mid_weights = mid_experiments()
    short, short_weights, confidence = short_experiments(args.dry_run)
    monthly = pd.read_parquet(ROOT / "results/short_batch/predictions.parquet").query(
        "item == '양파' and horizon == 10 and model == 'combined_lgbm'").copy()
    baseline = pd.read_parquet(ROOT / "results/all_items_experiment/predictions.parquet").query(
        "split == 'test' and term == 'short' and item == '양파' and horizon == 10 and model == 'current'")
    reference = baseline.merge(monthly[KEYS + ["actual", "base_price"]], on=KEYS,
                               suffixes=("", "_candidate"), validate="one_to_one")
    if monthly.empty or len(reference) != len(monthly):
        raise ValueError("Incomplete onion comparison")
    np.testing.assert_allclose(reference.actual, reference.actual_candidate)
    np.testing.assert_allclose(reference.base_price, reference.base_price_candidate)
    monthly = pd.concat([monthly, reference.assign(model="archived_current")])
    monthly["split"] = monthly.base_date.dt.strftime("%Y-%m")
    if args.dry_run:
        print(f"Validated {len(mid)} mid-term forecasts and {len(monthly)} onion forecasts")
        return
    forecasts = pd.concat(short, ignore_index=True)
    for frame in (forecasts, mid, monthly):
        if (frame.duplicated(KEYS + ["model"]).any() or not np.isfinite(frame.prediction).all()
                or not frame.prediction.gt(0).all()):
            raise ValueError("Invalid experiment predictions")
    args.output.mkdir(parents=True)
    for name, frame in {
        "short_metrics": score_predictions(forecasts), "mid_metrics": score_predictions(mid),
        "onion_monthly": score_predictions(monthly), "short_weights": pd.DataFrame(short_weights),
        "mid_weights": mid_weights, "confidence": pd.DataFrame(confidence),
    }.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    forecasts.to_parquet(args.output / "short_predictions.parquet", index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
