"""Tune on 2024 only; compare frozen choices against archived untuned forecasts."""
import json
from itertools import product

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate.freeze_next_day_direction import OUTPUT, ROOT, peer_inputs, price_inputs
from src.evaluate.radish_direction_models import tcn_probabilities


def main():
    daily = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    potato = pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet")
    frame = peer_inputs(price_inputs("무", daily, 1), pd.concat([daily, potato])).reset_index(drop=True)
    route = next(r for r in json.loads((OUTPUT / "manifest.json").read_text())["routes"] if r["item"] == "무")
    x = frame[route["features"]]
    y = frame.actual_direction.to_numpy()
    train = np.flatnonzero(frame.target_date.le(route["train_target_through"]))
    valid = np.flatnonzero(frame.date.dt.year.eq(2024) & frame.target_date.dt.year.eq(2024))
    test = np.flatnonzero(frame.date.dt.year.isin([2025, 2026]) &
                          frame.target_date.dt.year.eq(frame.date.dt.year) & frame.target_date.dt.month.le(9))
    assert len(train) == route["train_rows"] and len(test) == 466
    assert frame.target_date.iloc[train].max() < frame.date.iloc[valid].min()
    assert frame.target_date.iloc[valid].max() < frame.date.iloc[test].min()
    transform = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    values = transform.fit(x.iloc[train]).transform(x)

    def predict(name, params, indices):
        if name == "tcn":
            return tcn_probabilities(values, y > 0, train, indices, **params)
        if name == "logistic":
            model = LogisticRegression(max_iter=2000, **params).fit(values[train], y[train])
            return model.predict_proba(values[indices])[:, 1]
        if name == "lightgbm":
            model = lgb.LGBMClassifier(random_state=42, verbosity=-1, n_jobs=2, **params)
        else:
            model = CatBoostClassifier(random_seed=42, thread_count=2, verbose=False,
                                       allow_writing_files=False, **params)
        model.fit(x.iloc[train], y[train])
        return model.predict_proba(x.iloc[indices])[:, 1]

    grids = {
        "lightgbm": [dict(num_leaves=n, min_child_samples=c, n_estimators=e, learning_rate=.03)
                     for n, c, e in product([7, 3, 15], [40, 20, 80], [100, 200])],
        "catboost": [dict(depth=d, iterations=i, learning_rate=r)
                     for d, i, r in product([4, 2, 6], [150, 300], [.035, .07])],
        "logistic": [dict(C=c) for c in [1, .001, .01, .1, 10, 100]],
        "tcn": [dict(width=w, epochs=e, lr=r)
                for w, e, r in product([16, 8], [20, 10], [.001, .0003])],
    }
    output = ROOT / "results/radish_direction_tuning"
    output.mkdir(exist_ok=True)
    predictions = frame.iloc[test][["date", "target_date", "actual_direction"]].reset_index(drop=True)
    archive = pd.read_csv(ROOT / "results/radish_direction_models/probabilities.csv",
                          parse_dates=["date", "target_date"])
    predictions = predictions.merge(archive, on=list(predictions), validate="one_to_one")
    assert len(predictions) == len(test) and predictions.date.tolist() == frame.date.iloc[test].tolist()
    trials, metrics = [], []
    for name, grid in grids.items():
        hits = []
        for params in grid:
            prob = predict(name, params, valid)
            assert np.isfinite(prob).all()
            correct = int((np.where(prob > .5, 1, -1) == y[valid]).sum())
            hits.append(correct)
            trials.append(dict(model=name, params=json.dumps(params), issued=len(valid), correct=correct))
        best = grid[int(np.argmax(hits))]  # Fixed grid order breaks ties.
        prob = predict(name, best, test)
        assert np.isfinite(prob).all()
        predictions[name + "_tuned"] = prob
        for year, g in predictions.groupby(predictions.date.dt.year):
            correct = int((np.where(g[name + "_tuned"] > .5, 1, -1) == g.actual_direction).sum())
            original = int((np.where(g[name] > .5, 1, -1) == g.actual_direction).sum())
            metrics.append(dict(model=name, year=year, n=len(g), correct=correct, untuned=original,
                                val_correct=max(hits), val_n=len(valid), params=json.dumps(best)))
        print(name, metrics[-2:], flush=True)
    pd.DataFrame(trials).to_csv(output / "validation.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "probabilities.csv", index=False)


if __name__ == "__main__":
    main()
