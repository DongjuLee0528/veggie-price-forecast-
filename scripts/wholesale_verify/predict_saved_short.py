import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.train_lgbm import model_inputs

root = Path.cwd()
inputs = Path(__file__).resolve().parents[2] / "outputs" / "wholesale_verify" / "forecast_inputs_20261113"
model_dir = root / "results/lgbm_production"
manifest = json.loads((model_dir / "run.json").read_text())
features = pd.read_parquet(inputs / "price_features.parquet").set_index("item")
prices = pd.read_parquet(inputs / "prices.parquet")
origin, = features.date.unique()
if manifest["status"] != "complete" or pd.Timestamp(manifest["cutoff"]) > origin:
    raise ValueError("Invalid production model cutoff or status")
current = prices.loc[prices.date.eq(origin)].set_index("item").price_per_kg
hashes = {row["file"]: row["sha256"] for row in manifest["models"]}
routes = {(row["item"], row["horizon"]): row for row in manifest["selected"]}
models = {}
records = []
for item in features.index:
    for horizon in (1, 5, 10):
        route = routes.get((item, horizon))
        row = features.loc[[item]].assign(horizon=horizon)
        base = float(row.base_price.iloc[0])
        prediction = None
        if route is None:
            method = "potato_selected"
            reason = ("Selected activity blend requires original transaction counts; "
                      "API group counts have a different meaning." if horizon == 5 else
                      "Selected tuning model has archived predictions but no saved booster.")
        else:
            method = route["model"]
            reason = ""
            missing = sorted(set(route.get("inputs", [])) - set(row.columns))
            if missing:
                reason = "Missing feature inputs: " + ", ".join(missing)
            elif method == "last_value":
                prediction = float(current.loc[item])
            elif method == "moving_average_7":
                prediction = base
            elif method in {"price_direct", "price_pooled"}:
                filename = route["file"]
                if filename not in models:
                    path = model_dir / filename
                    if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[filename]:
                        raise ValueError(f"Model checksum mismatch: {filename}")
                    models[filename] = lgb.Booster(model_file=str(path))
                value, = models[filename].predict(model_inputs(row, route["inputs"]),
                                                   num_iteration=route["best_iteration"], num_threads=2)
                prediction = base * (1 + float(value))
            else:
                raise ValueError(f"Unexpected route: {method}")
        if prediction is not None and (not np.isfinite(prediction) or prediction <= 0):
            raise ValueError(f"Invalid prediction: {item}, {horizon}")
        records.append(dict(item=item, horizon=horizon, horizon_unit="observed_trading_days",
                            model=method, status="predicted" if prediction is not None else "pending",
                            prediction_krw_per_kg=prediction, target_date=None, reason=reason))
assert len(records) == 15
report = dict(base_date=str(pd.Timestamp(origin).date()), trained_through=manifest["cutoff"],
              source="mafra_api_provisional", model_set="saved_production_routes",
              forecasts=records)
(inputs / "short_forecast.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
pd.DataFrame(records).assign(base_date=report["base_date"], source=report["source"]).to_csv(
    inputs / "short_forecast.csv", index=False)
print(pd.DataFrame(records)[["item", "horizon", "status", "prediction_krw_per_kg"]].to_string(index=False))
