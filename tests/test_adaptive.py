import numpy as np
import pandas as pd

from src.ensemble.adaptive import recent_weights


def test_rolling_weights_use_only_already_observed_targets():
    dates = pd.date_range('2024-01-01', periods=60)
    rows = pd.DataFrame({'target_date': dates + pd.Timedelta(days=5), 'actual': 105.,
                         'base_price': 100., 'validation_blend': 106.,
                         'validation_selected': 96., 'last_value': 102., 'moving_average_7': 100.})
    origin = dates[35]
    expected = recent_weights(rows, [origin], window=28, penalty=1, temperature=12)
    changed = rows.copy()
    changed.loc[changed.target_date.gt(origin), 'actual'] = 5.
    np.testing.assert_array_equal(recent_weights(changed, [origin], 28, 1, 12), expected)
    assert expected[0, 0] > expected[0, 1]
    np.testing.assert_allclose(expected.sum(axis=1), 1.)
