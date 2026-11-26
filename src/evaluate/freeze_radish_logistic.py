"""Export standardization, imputation and logistic coefficients without a pickle dependency."""
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluate.freeze_next_day_direction import OUTPUT, ROOT, peer_inputs, price_inputs
from src.pipeline.radish_logistic import MODEL, probabilities


def main():
    daily = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    potato = pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet")
    prices = pd.concat([daily, potato], ignore_index=True)
    route = next(r for r in json.loads((OUTPUT / "manifest.json").read_text())["routes"] if r["item"] == "무")
    data = peer_inputs(price_inputs("무", daily, 1), prices)
    train = data[data.target_date.le(route["train_target_through"])]
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x = imputer.fit_transform(train[route["features"]])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=.01, max_iter=2000).fit(scaler.transform(x), train.actual_direction)
    assert model.classes_.tolist() == [-1, 1]
    spec = dict(cutoff=route["train_target_through"], features=route["features"], C=.01,
                median=imputer.statistics_.tolist(), indicators=imputer.indicator_.features_.tolist(),
                mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                weights=model.coef_[0].tolist(), bias=float(model.intercept_[0]))
    MODEL.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
    archive = pd.read_csv(ROOT / "results/radish_direction_tuning/probabilities.csv", parse_dates=["date"])
    np.testing.assert_allclose(probabilities(prices).reindex(archive.date), archive.logistic_tuned, atol=1e-10)
    print(f"Verified {len(archive)} archived probabilities")


if __name__ == "__main__":
    main()
