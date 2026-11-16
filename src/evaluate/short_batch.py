import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.models.compare_all_items import ITEMS, PARAMS
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs

HORIZONS = (1, 5, 10)
VARIANTS = {
    "price_lgbm": [],
    "median_lgbm": ["median_to_mean_7"],
    "low_activity_lgbm": ["low_count", "low_volume"],
    "combined_lgbm": ["median_to_mean_7", "low_count", "low_volume"],
}
WEIGHTS = (0., .25, .5, .75, 1.)
COVERAGES = (1., .7, .5, .3)


def history_for_item(item, history):
    daily = history[history.item.eq(item)].sort_values("date").reset_index(drop=True)
    if daily.empty or daily.date.duplicated().any() or not daily.price_per_kg.gt(0).all():
        raise ValueError(f"Invalid historical prices: {item}")
    price = daily.price_per_kg
    daily["base_price"] = price.rolling(7).mean()
    for lag in (1, 5, 10, 20):
        daily[f"lag_{lag}"] = price.shift(lag) / daily.base_price
    daily["ma7_ma28_ratio"] = daily.base_price / price.rolling(28).mean()
    for lag in (7, 28):
        daily[f"ret_{lag}"] = price / price.shift(lag) - 1
    daily["vol_28"] = price.pct_change(fill_method=None).rolling(28).std()
    prior = (daily.date - pd.DateOffset(years=1)).map(daily.set_index("date").price_per_kg)
    daily["yoy"] = price / prior - 1
    daily["median_to_mean_7"] = price.rolling(7).median() / daily.base_price
    for column, name in (("n_rows", "low_count"), ("volume_kg", "low_volume")):
        past_median = daily[column].shift().rolling(28).median()
        daily[name] = daily[column].lt(past_median * .5).astype(float).where(past_median.gt(0))
    return pd.concat([
        daily.assign(horizon=h, target_date=daily.date.shift(-h), actual=price.shift(-h))
        for h in HORIZONS
    ], ignore_index=True).dropna(subset=["base_price", "target_date"])


def segments(rows, year):
    valid_start = pd.Timestamp(year - 1, 1, 1)
    test_start = pd.Timestamp(year, 1, 1)
    test_end = min(pd.Timestamp(year + 1, 1, 1), pd.Timestamp("2026-10-01"))
    earlier = rows.loc[rows.date.lt(valid_start), "date"].drop_duplicates().sort_values()
    train = rows[rows.target_date.lt(earlier.iloc[-10])]
    calibration = rows[rows.date.ge(valid_start) & rows.target_date.lt(test_start)]
    evaluation = rows[rows.date.ge(test_start) & rows.target_date.lt(test_end)]
    for frame in (train, calibration, evaluation):
        if set(frame.horizon) != set(HORIZONS):
            raise ValueError(f"Missing horizon coverage for {year}")
    return train, calibration, evaluation


def records(rows, name, prices, year):
    return rows[["item", "date", "target_date", "horizon", "actual", "base_price"]].rename(
        columns={"date": "base_date"}
    ).assign(split=str(year), model=name, prediction=np.asarray(prices))


def confidence_rows(calibration, evaluation, name, year):
    output = []
    for horizon, valid in calibration.groupby("horizon"):
        test = evaluation[evaluation.horizon.eq(horizon)]
        valid_score = (valid.prediction / valid.base_price - 1).abs()
        test_score = (test.prediction / test.base_price - 1).abs()
        valid_direction = np.sign(valid.actual - valid.base_price)
        direction = np.sign(test.actual - test.base_price)
        hit = np.sign(test.prediction - test.base_price).eq(direction)
        majority = valid_direction.mode().iloc[0]
        for coverage in COVERAGES:
            threshold = 0. if coverage == 1. else valid_score.quantile(1 - coverage)
            selected = test_score.ge(threshold)
            output.append(dict(year=year, item=test.item.iloc[0], horizon=horizon, model=name,
                               target_coverage=coverage, threshold=threshold,
                               total=len(test), selected=int(selected.sum()),
                               actual_coverage=selected.mean(), selected_accuracy=hit[selected].mean(),
                               withheld_accuracy=hit[~selected].mean(),
                               majority_accuracy=direction[selected].eq(majority).mean()))
    return output


def main():
    parser = argparse.ArgumentParser(description="Compare short-term price experiments on two yearly folds.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/short_batch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error("Use a new output directory")
    daily = pd.concat([
        pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet"),
        pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet"),
    ])
    predictions, selections, confidence = [], [], []
    for item in sorted(ITEMS):
        history = history_for_item(item, daily)
        for year in (2025, 2026):
            train, valid, test = segments(history, year)
            print(f"{item} {year}: train={len(train)}, calibration={len(valid)}, test={len(test)}", flush=True)
            if args.dry_run:
                continue
            predictions.append(records(test, "last_value", test.price_per_kg, year))
            for name, extra in VARIANTS.items():
                columns = PRICE_FEATURES + ["horizon", *extra]
                model = lgb.LGBMRegressor(**PARAMS, n_estimators=100)
                model.fit(model_inputs(train, columns), train.actual / train.base_price - 1)
                valid_price = valid.base_price * (1 + model.predict(model_inputs(valid, columns)))
                test_price = test.base_price * (1 + model.predict(model_inputs(test, columns)))
                predictions.append(records(test, name, test_price, year))
                for horizon in HORIZONS:
                    v = valid.horizon.eq(horizon)
                    t = test.horizon.eq(horizon)
                    def error(weight):
                        mixed = (1 - weight) * valid_price[v] + weight * valid.price_per_kg[v]
                        return ((mixed / valid.actual[v] - 1).abs()).mean()
                    weight = min(WEIGHTS, key=error)
                    selections.append(dict(year=year, item=item, horizon=horizon, model=name,
                                           last_value_weight=weight))
                    blended_name = name + "_blend"
                    calibrated = records(valid.loc[v], blended_name,
                                         (1 - weight) * valid_price[v] + weight * valid.price_per_kg[v], year)
                    forecast = records(test.loc[t], blended_name,
                                       (1 - weight) * test_price[t] + weight * test.price_per_kg[t], year)
                    predictions.append(forecast)
                    confidence.extend(confidence_rows(calibrated, forecast, blended_name, year))
                print(f"  {name}: finished", flush=True)
    if args.dry_run:
        return
    rows = pd.concat(predictions, ignore_index=True)
    keys = ["split", "item", "horizon", "base_date", "target_date"]
    if (rows.duplicated(keys + ["model"]).any()
            or not rows.groupby(keys).model.nunique().eq(1 + 2 * len(VARIANTS)).all()
            or not np.isfinite(rows.prediction).all() or not rows.prediction.gt(0).all()):
        raise ValueError("Every experiment must produce a valid prediction on the same samples")
    metrics = score_predictions(rows)
    archive = pd.read_parquet(ROOT / "results/all_items_experiment/predictions.parquet")
    archive = archive.query("split == 'test' and term == 'short' and model == 'current'").copy()
    archive["split"] = archive.base_date.dt.year.astype(str)
    reference = rows.loc[rows.model.eq("last_value"), keys + ["actual", "base_price"]]
    matched = archive.merge(reference, on=keys, suffixes=("", "_reference"), validate="one_to_one")
    np.testing.assert_allclose(matched.actual, matched.actual_reference)
    np.testing.assert_allclose(matched.base_price, matched.base_price_reference)
    if len(matched) != len(reference):
        raise ValueError("Archived comparison does not cover the experiment dates")
    archived_metrics = score_predictions(matched.assign(model="archived_current"))
    args.output.mkdir(parents=True)
    rows.to_parquet(args.output / "predictions.parquet", index=False)
    pd.concat([metrics, archived_metrics]).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(selections).to_csv(args.output / "blend_weights.csv", index=False)
    pd.DataFrame(confidence).to_csv(args.output / "confidence.csv", index=False)
    print(metrics.groupby(["split", "model"])[["nmae", "direction_accuracy"]].mean().to_string())
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
