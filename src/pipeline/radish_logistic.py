"""Frozen next-trade radish direction classifier; C=0.01, no abstention."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from src.evaluate.freeze_direction_candidates import direction_inputs
from src.evaluate.freeze_next_day_direction import peer_inputs
from src.features.build import price_features

MODEL = Path(__file__).with_name("radish_logistic.json")


def probabilities(prices):
    spec = json.loads(MODEL.read_text())
    daily = prices.loc[prices.item.eq("무")].sort_values("date").reset_index(drop=True)
    features = price_features(daily)
    daily = daily.assign(vol_28=features.vol_28, yoy=features.yoy)
    inputs = peer_inputs(direction_inputs(daily.loc[features.base_price.notna()]), prices)
    valid_dates = daily.loc[daily.price_per_kg.rolling(29).count().eq(29) & daily.volume_kg.gt(0), "date"]
    inputs = inputs.loc[inputs.date.isin(valid_dates) & inputs.date.gt(spec["cutoff"])]
    x = inputs[spec["features"]].to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError("Infinite radish inputs")
    missing = np.isnan(x)
    values = np.column_stack([np.where(missing, spec["median"], x), missing[:, spec["indicators"]]])
    z = (values - spec["mean"]) / spec["scale"]
    return pd.Series(expit(z @ spec["weights"] + spec["bias"]), index=inputs.date)
