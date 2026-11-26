"""Compare next-trade radish directions. Experiment dependencies: torch, catboost==1.2.8."""
import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.evaluate.freeze_next_day_direction import OUTPUT, ROOT, peer_inputs, price_inputs


def tcn_probabilities(values, labels, train, test, *, width=16, epochs=20, lr=.001):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    length = 15
    # Left padding keeps the first observations; every window ends on its forecast origin.
    padded = np.pad(values.astype("float32"), ((length - 1, 0), (0, 0)))
    windows = torch.from_numpy(np.lib.stride_tricks.sliding_window_view(padded, length, axis=0).copy())
    layers = []
    channels = values.shape[1]
    for dilation in (1, 2, 4):
        layers.extend([nn.ConstantPad1d((2 * dilation, 0), 0),
                       nn.Conv1d(channels, width, 3, dilation=dilation), nn.ReLU()])
        channels = width
    network = nn.Sequential(*layers, nn.Conv1d(width, 1, 1))
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr)
    targets = torch.tensor(labels, dtype=torch.float32)
    for _ in range(epochs):
        for indices in torch.tensor(train)[torch.randperm(len(train))].split(64):
            logits = network(windows[indices])[:, 0, -1]
            loss = nn.functional.binary_cross_entropy_with_logits(logits, targets[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    network.eval()
    with torch.no_grad():
        return network(windows[test])[:, 0, -1].sigmoid().numpy()


def main():
    daily = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    potato = pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet")
    frame = peer_inputs(price_inputs("무", daily, 1), pd.concat([daily, potato])).reset_index(drop=True)
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    route = next(row for row in manifest["routes"] if row["item"] == "무")
    payload = (OUTPUT / route["file"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == route["sha256"]
    inputs = frame[route["features"]]
    labels = frame.actual_direction
    train = np.flatnonzero(frame.target_date.le(route["train_target_through"]))
    assert len(train) == route["train_rows"] and set(labels.iloc[train]) == {-1, 1}
    test = np.flatnonzero((frame.date.dt.year.isin([2025, 2026])) &
                          (frame.target_date.dt.year == frame.date.dt.year) &
                          (frame.target_date.dt.month <= 9))
    result = frame.iloc[test][["date", "target_date", "actual_direction"]].copy()
    assert result.groupby(result.date.dt.year).size().tolist() == [233, 233]
    result["lightgbm"] = lgb.Booster(model_str=payload.decode()).predict(inputs.iloc[test], num_threads=2)
    transform = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
    transform.fit(inputs.iloc[train])
    values = transform.transform(inputs)
    logistic = LogisticRegression(max_iter=2000).fit(values[train], labels.iloc[train])
    result["logistic"] = logistic.predict_proba(values[test])[:, 1]
    cat = CatBoostClassifier(iterations=150, depth=4, learning_rate=.035, random_seed=42,
                             thread_count=2, verbose=False, allow_writing_files=False)
    cat.fit(inputs.iloc[train], labels.iloc[train])
    result["catboost"] = cat.predict_proba(inputs.iloc[test])[:, 1]
    result["tcn"] = tcn_probabilities(values, labels.gt(0).to_numpy(), train, test)
    summaries = []
    for year, group in result.groupby(result.date.dt.year):
        for model in ("lightgbm", "logistic", "catboost", "tcn"):
            probability = group[model]
            assert probability.between(0, 1).all()
            pred = np.where(probability > .5, 1, -1)
            hit = pred == group.actual_direction
            summaries.append(dict(year=year, model=model, issued=len(group), correct=int(hit.sum()),
                                  accuracy=hit.mean(), up=int(sum(pred == 1)), up_correct=int(sum(hit & (pred == 1))),
                                  down=int(sum(pred == -1)), down_correct=int(sum(hit & (pred == -1)))))
    summary = pd.DataFrame(summaries)
    assert summary.loc[summary.model.eq("lightgbm"), "correct"].tolist() == [134, 127]
    dest = ROOT / "results/radish_direction_models"
    dest.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest / "probabilities.csv", index=False)
    summary.to_csv(dest / "metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
