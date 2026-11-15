import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.config import load_items
from src.features.build import build_features
from src.models.train_lgbm import model_inputs
from src.preprocess.merge import merge_sources

root = Path.cwd()
inputs = Path(__file__).resolve().parents[2] / "outputs" / "wholesale_verify" / "forecast_inputs_20261113"
prices = pd.read_parquet(inputs / "prices.parquet")
price_features = pd.read_parquet(inputs / "price_features.parquet")
origin, = price_features.date.unique()
origin = pd.Timestamp(origin)
items = load_items()["items"]
config = yaml.safe_load((root / "configs/features.yaml").read_text())
interim = root / "data/interim"
weather = pd.read_parquet(interim / "weather/asos_daily.parquet")
fx = pd.read_parquet(interim / "fx/fx_daily.parquet")
if fx.date.max() < origin:
    fx = pd.concat([fx, pd.DataFrame({"date": [origin]})], ignore_index=True)
trade = pd.read_parquet(interim / "trade/imports_by_form.parquet")
calendar = pd.read_parquet(interim / "calendar/calendar_daily.parquet")
volume_files = sorted((root / "data/raw/volume").glob("garak_????????.parquet"))
volume = pd.concat([pd.read_parquet(path) for path in volume_files], ignore_index=True)
radish = prices.loc[prices.item.eq("무")].copy()
radish["n_rows"] = radish.n_rows.fillna(1).astype(int)
radish = radish.dropna(subset=["price_per_kg"])
# merge_sources validates historical settlement row counts. The synthetic counts
# are never used by build_features or persisted.
integrated = merge_sources(
    pd.concat([radish, pd.read_parquet(interim / "price/settlement_daily_clean.parquet")
               .loc[lambda frame: frame.item.ne("무")]], ignore_index=True),
    volume, weather, fx, trade, calendar, items, config["publish_lag"],
    pd.read_csv(root / "configs/trade_releases.csv",
                parse_dates=["month", "published_on", "available_on"]),
)
features = build_features(integrated[integrated.item.eq("무")], weather, fx, trade,
                          {"무": items["무"]}, config)
current = features.loc[features.date.eq(origin)]
price_columns = price_features.columns.drop(["date", "item"])
np.testing.assert_allclose(current[price_columns], price_features.loc[
    price_features.item.eq("무"), price_columns])
model_dir = root / "results/lgbm_production"
manifest = json.loads((model_dir / "run.json").read_text())
if manifest["status"] != "complete" or pd.Timestamp(manifest["cutoff"]) > origin:
    raise ValueError("Invalid saved model manifest")
routes = [r for r in manifest["selected"] if r["item"] == "무" and r["horizon"] in (5, 10)]
records = []
for route in routes:
    path = model_dir / route["file"]
    detail, = [r for r in manifest["models"] if r["file"] == path.name]
    if hashlib.sha256(path.read_bytes()).hexdigest() != detail["sha256"]:
        raise ValueError("Model checksum mismatch")
    model = lgb.Booster(model_file=str(path))
    row = model_inputs(current.assign(horizon=route["horizon"]), route["inputs"])
    missing = row.columns[row.isna().any()].tolist()
    value, = model.predict(row, num_iteration=route["best_iteration"], num_threads=2)
    records.append(dict(item="무", horizon=route["horizon"], status="preview_missing_features",
                        prediction_krw_per_kg=float(current.base_price.iloc[0] * (1 + value)),
                        model=path.name, missing_features=", ".join(missing)))

path = root / "results/potato_sumi/short_lgbm.txt"
model = lgb.Booster(model_file=str(path))
potato = price_features.loc[price_features.item.eq("감자 수미")]
for horizon in (1, 5, 10):
    row = model_inputs(potato.assign(horizon=horizon), model.feature_name())
    value, = model.predict(row, num_threads=2)
    records.append(dict(item="감자 수미", horizon=horizon, status="preview_saved_fallback",
                        prediction_krw_per_kg=float(potato.base_price.iloc[0] * (1 + value)),
                        model="potato_sumi/short_lgbm.txt", missing_features=""))

report = pd.DataFrame(records).assign(base_date=origin.date(), source="mafra_api_provisional")
assert len(report) == 5 and not report.duplicated(["item", "horizon"]).any()
assert np.isfinite(report.prediction_krw_per_kg).all() and report.prediction_krw_per_kg.gt(0).all()
report.to_csv(inputs / "short_forecast_pending_preview.csv", index=False)
print(report.to_string(index=False))
