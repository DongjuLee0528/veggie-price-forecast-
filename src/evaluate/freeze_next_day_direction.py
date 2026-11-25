import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.freeze_direction_candidates import FEATURES, PARAMS, ROOT, price_inputs


OUTPUT = ROOT / "results/next_day_direction_all"


PEERS = {"배추": "cabbage", "양파": "onion", "건고추": "pepper", "감자 수미": "potato"}


def peer_inputs(frame, prices):
    for item, prefix in PEERS.items():
        peer = prices.loc[prices.item.eq(item)].sort_values("date").set_index("date")
        if peer.empty or not peer.index.is_unique:
            raise ValueError(f"Invalid peer price history: {item}")
        price = peer.price_per_kg
        signals = pd.DataFrame({
            "ret1": price.pct_change(fill_method=None),
            "ret7": price.pct_change(7, fill_method=None),
            "avg7": price.rolling(7).mean() / price - 1,
            "volume_ratio": peer.volume_kg / peer.volume_kg.shift().rolling(14).median() - 1,
        }).add_prefix(f"peer_{prefix}_")
        frame = frame.join(signals, on="date", validate="one_to_one")
    if np.isinf(frame.filter(regex="^peer_").to_numpy()).any():
        raise ValueError("Infinite peer inputs")
    return frame


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    archive = ROOT / "results/direction_candidates_observed_price"
    manifest = json.loads((archive / "manifest.json").read_text())
    onion, = [r for r in manifest["routes"] if r["item"] == "양파" and r["horizon"] == 1]
    payload = (archive / onion["file"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != onion["sha256"] or manifest["features"] != FEATURES:
        raise ValueError("Frozen onion candidate differs from the evaluated model")
    onion.update(min_probability=.65, benchmark="mean7")

    raw = pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet")
    history = price_inputs("감자 수미", raw, 1)
    start = pd.Timestamp("2026-01-01")
    embargo = history.loc[history.date.lt(start), "date"].iloc[-10]
    train = history[history.date.ge(start - pd.DateOffset(years=2)) & history.target_date.lt(embargo)]
    model = lgb.LGBMClassifier(**PARAMS).fit(train[FEATURES], train.actual_direction)
    tested = pd.read_parquet(ROOT / "results/direction_training_windows/paired_predictions.parquet")
    tested = tested[tested.item.eq("감자 수미") & tested.horizon.eq(1) & tested.base_date.dt.year.eq(2026)]
    paired = tested.merge(history, left_on=["base_date", "target_date"], right_on=["date", "target_date"],
                          validate="one_to_one")
    if paired.empty or len(paired) != len(tested):
        raise ValueError("Incomplete replay of the evaluated potato candidate")
    np.testing.assert_array_equal(model.predict(paired[FEATURES]), paired.recent_2y)
    daily = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    columns = ["date", "item", "price_per_kg", "volume_kg"]
    peer_prices = pd.concat([daily[columns], raw[columns]], ignore_index=True)
    radish_data = price_inputs("무", daily, 1)
    radish_data = peer_inputs(radish_data, peer_prices)
    radish_features = FEATURES + [column for column in radish_data if column.startswith("peer_")]
    boundary = radish_data.loc[radish_data.date.lt("2024-01-01"), "date"].iloc[-10]
    radish_train = radish_data.loc[radish_data.target_date.lt(boundary)]
    radish_model = lgb.LGBMClassifier(**PARAMS).fit(radish_train[radish_features], radish_train.actual_direction)
    threshold = .65
    reproduced = {}
    for year, expected in ((2025, (41, 28)), (2026, (42, 32))):
        test = radish_data[radish_data.date.ge(f"{year}-01-01") & radish_data.target_date.lt(f"{year}-10-01")]
        probabilities = radish_model.predict_proba(test[radish_features])
        shown = probabilities.max(axis=1) >= threshold
        directions = radish_model.classes_[probabilities.argmax(axis=1)]
        correct = int((shown & test.actual_direction.eq(directions)).sum())
        if (int(shown.sum()), correct) != expected:
            raise ValueError(f"Radish candidate did not reproduce the evaluated results: {year}")
        reproduced[year] = dict(issued=int(shown.sum()), correct=correct, total=len(test))
    with TemporaryDirectory(dir=OUTPUT.parent) as tmp:
        stage = Path(tmp) / "models"
        stage.mkdir()
        (stage / onion["file"]).write_bytes(payload)
        potato_file = stage / "감자 수미_1.txt"
        model.booster_.save_model(str(potato_file))
        potato = dict(item="감자 수미", horizon=1, file=potato_file.name,
                      sha256=hashlib.sha256(potato_file.read_bytes()).hexdigest(),
                      classes=model.classes_.tolist(), train_rows=len(train),
                      train_target_through=str(train.target_date.max().date()), window_years=2,
                      min_probability=None, benchmark="current", decision="candidate_for_new_period")
        radish_file = stage / "무_1.txt"
        radish_model.booster_.save_model(str(radish_file))
        radish = dict(item="무", horizon=1, file=radish_file.name, features=radish_features,
                      sha256=hashlib.sha256(radish_file.read_bytes()).hexdigest(),
                      classes=radish_model.classes_.tolist(), train_rows=len(radish_train),
                      train_target_through=str(radish_train.target_date.max().date()),
                      min_probability=threshold, benchmark="mean7", decision="candidate_for_new_period",
                      retrospective_replay=reproduced)
        (stage / "manifest.json").write_text(
            json.dumps(dict(features=FEATURES, parameters=PARAMS, routes=[onion, potato, radish]),
                       ensure_ascii=False, indent=2) + "\n")
        if OUTPUT.exists():
            raise FileExistsError(OUTPUT)
        stage.rename(OUTPUT)
    print(f"Frozen three candidate models; verified {len(tested)} potato forecasts and radish coverage/accuracy")


if __name__ == "__main__":
    main()
