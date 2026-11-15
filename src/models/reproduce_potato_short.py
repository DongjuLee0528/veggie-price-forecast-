import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs, validate_split

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
root = Path.cwd()
work = Path(__file__).resolve().parents[2] / "outputs" / "wholesale_verify"
data = root / "data/processed/potato_sumi"
results = root / "results/potato_sumi_tuning"
inputs = work / "forecast_inputs_20261113"
output = inputs / "potato_tuned"
columns = PRICE_FEATURES + ["horizon"]
train, validation, test = [
    pd.read_parquet(data / f"{split}.parquet").query("term == 'short'")
    for split in ("train", "validation", "test")
]
for split, frame in zip(("train", "validation", "test"), (train, validation, test)):
    validate_split(frame, split)
selection = pd.read_csv(results / "validation_selection.csv").query("term == 'short'").set_index("horizon").model
archive = pd.read_parquet(results / "predictions.parquet").query("term == 'short' and split == 'test'")
observations = pd.read_parquet(inputs / "prices.parquet").query("item == '감자 수미'").set_index("date")
origin_row = pd.read_parquet(inputs / "price_features.parquet").query("item == '감자 수미'")
origin, = origin_row.date
last = observations.loc[origin, "price_per_kg"]
routes = [(1, 4, 25, .5, "lgbm_4y_7leaves_25child_last_0.5"),
          (10, 2, 50, 0., "lgbm_2y_7leaves_50child")]
models, forecasts = {}, []
for horizon, years, min_child, weight, name in routes:
    if selection.loc[horizon] != name:
        raise ValueError(f"Archived selection changed: {horizon}")
    subset = train[train.date.ge(f"{2024 - years}-01-01")]
    # The original tuning trained across horizons with horizon as a feature.
    holdout = test[test.horizon.eq(horizon)]
    saved = archive[archive.model.eq(name) & archive.horizon.eq(horizon)].set_index(["base_date", "target_date"])
    expected = saved.reindex(pd.MultiIndex.from_frame(holdout[["date", "target_end"]]))
    if len(expected) != len(saved) or expected.prediction.isna().any():
        raise ValueError(f"Incomplete archive: {name}")
    np.testing.assert_allclose(holdout.target_price, expected.actual)
    np.testing.assert_allclose(holdout.base_price, expected.base_price)
    origin_inputs = model_inputs(origin_row.assign(horizon=horizon), columns)
    if not np.isfinite(origin_inputs).all().all() or not np.isfinite(last) or last <= 0:
        raise ValueError("Invalid latest price features")
    print(f"{horizon} trading days: train={len(subset)}, validation={len(validation)}, replay={len(holdout)}", flush=True)
    if args.dry_run:
        continue
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=500, learning_rate=.03,
        num_leaves=7, min_child_samples=min_child, reg_lambda=1.,
        n_jobs=2, random_state=42, verbosity=-1,
    )
    model.fit(model_inputs(subset, columns), subset.target_return,
              eval_set=[(model_inputs(validation, columns), validation.target_return)],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(10)])
    replay = holdout.base_price.to_numpy() * (1 + model.predict(model_inputs(holdout, columns)))
    replay = (1 - weight) * replay + weight * holdout.date.map(observations.price_per_kg).to_numpy()
    np.testing.assert_allclose(replay, expected.prediction, rtol=1e-9, atol=1e-8)
    change, = model.predict(origin_inputs)
    forecast = (1 - weight) * origin_row.base_price.iloc[0] * (1 + change) + weight * last
    if not np.isfinite(forecast) or forecast <= 0:
        raise ValueError(f"Invalid price forecast: {name}")
    models[name] = model.booster_
    forecasts.append(dict(item="감자 수미", horizon=horizon, base_date=origin.date(),
                          prediction_krw_per_kg=forecast, model=name,
                          best_iteration=model.best_iteration_, source="mafra_api_provisional"))
    print(f"Replayed archived predictions: {name}; best round={model.best_iteration_}", flush=True)

if not args.dry_run:
    output.mkdir(exist_ok=False)
    for name, model in models.items():
        model.save_model(str(output / f"{name}.txt"))
    pd.DataFrame(forecasts).to_csv(output / "predictions.csv", index=False)
    print(pd.DataFrame(forecasts).to_string(index=False))
    print(f"Saved: {output}")
