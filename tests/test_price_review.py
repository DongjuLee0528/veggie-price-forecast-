import pandas as pd
import pytest

from src.preprocess.price_review import review_prices


def test_trial_exclusion():
    rows = pd.DataFrame({
        'date': pd.to_datetime(['2020-03-17', '2020-03-17', '2020-03-19', '2024-04-11', '2024-04-11']),
        'market_code': ['380401', 'other', '380401', '210009', 'other'],
        'item': ['건고추'] * 3 + ['깐마늘'] * 2,
        'variety': ['건고추(일반)'] * 3 + ['깐마늘 남도'] * 2,
        'unit_qty': [120., 1., 60., 1., 10.],
        'price_per_unit': [125000., 10000., 85000., 1600., 5000.],
        'kg_per_unit': [0.6, 1., 0.6, 500., 1.],
    })
    original = rows.copy(deep=True)
    after, flagged, comparison = review_prices(rows)
    assert flagged.action.tolist() == ['exclude_trial', 'exclude_trial', 'exclude_trial']
    assert after.n_rows.sum() == 2
    assert comparison.excluded_rows.sum() == 3
    assert after.loc[after.date.eq('2020-03-17'), 'price_per_kg'].item() == 10000
    pd.testing.assert_frame_equal(rows, original)
    with pytest.raises(ValueError):
        review_prices(rows.drop(columns='kg_per_unit'))


def test_entire_day_excluded():
    rows = pd.DataFrame({
        'date': pd.to_datetime(['2024-04-11']), 'market_code': ['210009'],
        'item': ['깐마늘'], 'variety': ['깐마늘 남도'], 'unit_qty': [1.],
        'price_per_unit': [1600.], 'kg_per_unit': [500.],
    })
    after, _, comparison = review_prices(rows)
    assert after.empty
    assert comparison.excluded_rows.item() == 1
    assert pd.isna(comparison.price_per_kg_after.item())


def test_unreviewed_pattern_is_retained():
    rows = pd.DataFrame({
        'date': pd.to_datetime(['2020-03-19', '2021-03-19']),
        'market_code': ['380401', '380401'],
        'item': ['건고추', '건고추'], 'variety': ['건고추(일반)'] * 2,
        'unit_qty': [61., 60.], 'price_per_unit': [85000., 85000.],
        'kg_per_unit': [0.6, 0.6],
    })
    after, flagged, comparison = review_prices(rows)
    assert flagged.action.tolist() == ['review_only', 'review_only']
    assert after.n_rows.sum() == 2
    assert comparison.excluded_rows.sum() == 0
