import numpy as np
import pandas as pd
import pytest

from src.collect.asos import station_ids
from src.preprocess.weather import load_weather


def fixture_files(tmp_path):
    data = pd.DataFrame({'date': pd.Timestamp('2024-01-01'), 'station_id': station_ids(),
        'observed': True, 'temp_mean_c': 1., 'temp_min_c': -1., 'temp_max_c': 3.,
        'rain_mm': np.nan, 'sunshine_hours': 0.})
    data.to_parquet(tmp_path / 'asos_all.parquet', index=False)
    return data


def test_identical_duplicates_and_nan_preserved(tmp_path):
    data = fixture_files(tmp_path)
    data.iloc[:1].to_parquet(tmp_path / 'asos_overlap.parquet', index=False)
    result = load_weather(tmp_path, '2024-01-01', '2024-01-01')
    assert len(result) == len(station_ids())
    assert result.rain_mm.isna().all()
    assert result.sunshine_hours.eq(0).all()


def test_conflicting_duplicates(tmp_path):
    data = fixture_files(tmp_path).iloc[:1].copy()
    data['rain_mm'] = 1.
    data.to_parquet(tmp_path / 'asos_conflict.parquet', index=False)
    with pytest.raises(ValueError, match='Conflicting'):
        load_weather(tmp_path, '2024-01-01', '2024-01-01')


def test_missing_day_not_fabricated(tmp_path):
    fixture_files(tmp_path)
    with pytest.raises(ValueError, match='Missing weather'):
        load_weather(tmp_path, '2024-01-01', '2024-01-02')
