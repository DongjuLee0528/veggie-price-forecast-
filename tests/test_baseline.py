import numpy as np
import pandas as pd
import pytest

from src.config import load_items
from src.models.baseline import predict_baselines


def sample_prices():
    dates = pd.date_range('2023-01-01', '2024-01-10')
    return pd.concat([pd.DataFrame({
        'date': dates, 'item': item, 'price_per_kg': np.arange(len(dates)) + 1.,
        'amount': np.arange(len(dates)) + 1., 'volume_kg': 1., 'n_rows': 1,
    }) for item in load_items()['items']], ignore_index=True)


def test_formulas_and_future_independence():
    prices = sample_prices()
    result = predict_baselines(prices, '2024-01-07', ['2024-01-08'])
    values = result[result.item.eq('배추')].set_index('model').prediction
    assert values['last_value'] == 372
    assert values['moving_average_7'] == 369
    assert values['year_ago_adjusted'] == 8 * 369 / 4
    past = prices[prices.date.le('2024-01-07')]
    pd.testing.assert_frame_equal(result, predict_baselines(past, '2024-01-07', ['2024-01-08']))


def test_missing_year_ago_remains_nan():
    result = predict_baselines(sample_prices(), '2023-01-07', ['2023-01-08'])
    assert result.loc[result.model.eq('year_ago_adjusted'), 'prediction'].isna().all()


@pytest.mark.parametrize('targets', [[], ['2024-01-07'], ['2024-01-08', '2024-01-08']])
def test_invalid_targets(targets):
    with pytest.raises(ValueError):
        predict_baselines(sample_prices(), '2024-01-07', targets)


def test_insufficient_history():
    with pytest.raises(ValueError, match='seven'):
        predict_baselines(sample_prices(), '2023-01-03', ['2023-01-04'])
