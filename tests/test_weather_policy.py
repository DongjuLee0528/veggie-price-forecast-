import pandas as pd
import yaml

from src.config import CONFIG_DIR
from src.collect.asos import parse_response


def test_rain_not_active_for_first_training():
    with (CONFIG_DIR / 'features.yaml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    active = config['features']['weather']
    policy = config['weather_missing_policy']
    assert policy['rain_blank'] == 'preserve_nan'
    assert policy['rain_features_enabled'] is False
    assert set(policy['deferred_rain_features']).isdisjoint(active)
    assert not any('rain' in name for name in active)
    assert 'temp_anom_14' in active and 'sun_sum_14' in active


def test_reported_zero_and_blank_rain_are_distinct():
    payload = {'response': {'header': {'resultCode': '00'}, 'body': {
        'totalCount': 2, 'items': {'item': [
            {'tm': '2024-01-01', 'stnId': '100', 'sumRn': '0.0'},
            {'tm': '2024-01-02', 'stnId': '100', 'sumRn': ''},
        ]}}}}
    result = parse_response(payload, 100, pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02'))
    assert result.rain_mm.iloc[0] == 0
    assert pd.isna(result.rain_mm.iloc[1])
    assert result.observed.all()
