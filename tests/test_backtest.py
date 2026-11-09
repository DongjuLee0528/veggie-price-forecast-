import numpy as np
import pandas as pd

from src.config import load_items
from src.evaluate.backtest import backtest, score_predictions


def test_metrics_missing_predictions_and_direction():
    records = pd.DataFrame({
        'split': ['validation'] * 3, 'item': ['배추'] * 3,
        'horizon': [1] * 3, 'model': ['example'] * 3,
        'prediction': [12., 8., np.nan], 'actual': [10., 10., 10.],
        'base_price': [9., 9., 9.],
    })
    scores = score_predictions(records).iloc[0]
    assert scores.n_total == 3 and scores.n_scored == 2
    assert scores.nmae == .2 and scores.rmse == 2
    assert scores.direction_accuracy == .5
    assert scores.coverage == 2 / 3


def test_trading_horizons_boundary_and_future_independence():
    dates = pd.bdate_range('2023-12-01', '2025-01-31')
    prices = pd.concat([pd.DataFrame({
        'date': dates, 'item': item, 'price_per_kg': 100.,
        'amount': 100., 'volume_kg': 1., 'n_rows': 1,
    }) for item in load_items()['items']], ignore_index=True)
    result = backtest(prices)
    assert result.horizon.between(1, 10).all()
    assert result.target_date.gt(result.base_date).all()
    assert result[result.split.eq('validation')].target_date.dt.year.eq(2024).all()
    assert result[result.model.eq('last_value')].prediction.eq(100).all()
    base = result.base_date.min()
    prices.loc[prices.date.gt(base), ['price_per_kg', 'amount']] = 200.
    changed = backtest(prices)
    pd.testing.assert_series_equal(
        result[result.base_date.eq(base)].prediction.reset_index(drop=True),
        changed[changed.base_date.eq(base)].prediction.reset_index(drop=True),
    )
