import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.targets import dekad_bounds, dekad_index
from src.models.compare_all_items import ITEMS
from src.models.train_lgbm import model_inputs
from src.models.train_mid import series_predictions
from src.models.tune_mid import ets_predictions

ROOT = Path(__file__).resolve().parents[2]


def predict_mid(inputs, model_results, *, selected_pepper=False, items=None):
    prices = pd.read_parquet(inputs / "prices.parquet")
    features = pd.read_parquet(inputs / "price_features.parquet")
    if set(features.item) != ITEMS or features.item.duplicated().any():
        raise ValueError("Expected one feature row per item")
    origin, = features.date.unique()
    origin = pd.Timestamp(origin)
    period = dekad_index(origin)
    completed = period - int(dekad_bounds(period)[1] > origin)
    prices = prices.loc[prices.date.le(origin)].copy()
    if prices.duplicated(["date", "item"]).any():
        raise ValueError("Duplicate price observation")
    prices["dekad"] = prices.date.map(dekad_index)
    weights = json.loads((model_results / "mid_tuning/selection.json").read_text())
    ets_weight = weights["ets_weight"]
    sarimax_weight = weights["sarimax_weight_in_fixed_blend"]
    if not all(0 <= weight <= 1 for weight in (ets_weight, sarimax_weight)):
        raise ValueError("Invalid blend weights")
    potato = pd.read_csv(model_results / "potato_sumi_tuning/validation_selection.csv")
    if potato.query("term == 'mid'").set_index("horizon").model.to_dict() != {
        2: "ets_no_trend", 3: "ets_no_trend_with_sarimax"
    }:
        raise ValueError("Unexpected potato mid-term routes")
    selected = ITEMS if items is None else set(items)
    if not selected or not selected <= ITEMS:
        raise ValueError("Invalid selected items")
    records = []
    for item in sorted(selected):
        daily = prices.loc[prices.item.eq(item)]
        history = daily.loc[daily.dekad.le(completed)]
        means = history.groupby("dekad").price_per_kg.mean()
        if means.isna().any() or not means.gt(0).all():
            raise ValueError(f"Unavailable completed dekad: {item}")
        origin_rows = pd.concat([features.loc[features.item.eq(item)].assign(horizon=horizon)
                                 for horizon in (2, 3)], ignore_index=True)
        origin_rows = origin_rows.assign(base_date=origin, origin_dekad=period,
                                         target_dekad=lambda frame: period + frame.horizon)
        sarimax = series_predictions(origin_rows, daily, means).sarimax
        if item == "감자 수미":
            ets = ets_predictions(origin_rows, means, daily.date.min(), trend=None, damped=False)
            predictions = ets.where(origin_rows.horizon.eq(2), .75 * ets + .25 * sarimax)
            methods = ["ets_no_trend", "ets_no_trend_with_sarimax"]
        else:
            ets = ets_predictions(origin_rows, means, daily.date.min())
            model = lgb.Booster(model_file=str(model_results / "lgbm_mid_comparison" / f"{item}_price_lgbm.txt"))
            returns = model.predict(model_inputs(origin_rows, model.feature_name()), num_threads=2)
            fixed = (1 - sarimax_weight) * origin_rows.base_price * (1 + returns) + sarimax_weight * sarimax
            predictions = (1 - ets_weight) * fixed + ets_weight * ets
            methods = ["blend_with_ets"] * 2
            if item == "건고추" and selected_pepper:
                predictions = predictions.where(origin_rows.horizon.ne(2), sarimax)
                methods[0] = "sarimax"
        for row, prediction, method in zip(origin_rows.itertuples(), predictions, methods):
            start, end = dekad_bounds(row.target_dekad)
            if not np.isfinite(prediction) or prediction <= 0:
                raise ValueError(f"Invalid forecast: {item}, {row.horizon}")
            records.append(dict(item=item, horizon=row.horizon, horizon_unit="dekads",
                                base_date=str(origin.date()), target_start=str(start.date()),
                                target_end=str(end.date()), prediction_krw_per_kg=float(prediction),
                                model=method, source="mafra_api_provisional",
                                missing_prices_in_completed_dekads=int(history.price_per_kg.isna().sum())))
    return dict(base_date=str(origin.date()), completed_through=str(dekad_bounds(completed)[1].date()),
                source="mafra_api_provisional", unit="KRW/kg", forecasts=records)


def main():
    parser = argparse.ArgumentParser(description="Generate selected mid-term forecasts.")
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    report = predict_mid(args.inputs, ROOT / "results")
    args.output.mkdir(parents=True)
    rows = pd.DataFrame(report["forecasts"])
    rows.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "forecast.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(rows[["item", "horizon", "target_start", "target_end", "prediction_krw_per_kg"]].to_string(index=False))


if __name__ == "__main__":
    main()
