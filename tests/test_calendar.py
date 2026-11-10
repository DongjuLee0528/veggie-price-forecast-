import pandas as pd
import pytest

from src.collect.calendar import build_calendar, festival_dates, parse_year


def holidays():
    return pd.DataFrame({'date': pd.to_datetime(['2024-02-09','2024-02-10','2024-02-11','2024-09-16','2024-09-17','2024-09-18']), 'name': ['설날'] * 3 + ['추석'] * 3, 'is_holiday': True})


def test_festival_center_and_countdown():
    data = build_calendar(holidays(), '2024-02-09', '2024-02-11')
    assert data.days_to_seollal.iloc[:2].tolist() == [1, 0]
    assert pd.isna(data.days_to_seollal.iloc[2])
    assert data.is_holiday.all()
    assert festival_dates(holidays(), '추석')[0] == pd.Timestamp('2024-09-17')


def test_kimjang_boundaries():
    data = build_calendar(holidays(), '2024-10-31', '2024-12-11').set_index('date')
    assert not data.loc['2024-10-31','is_kimjang']
    assert data.loc['2024-11-01','is_kimjang']
    assert data.loc['2024-12-10','is_kimjang']
    assert not data.loc['2024-12-11','is_kimjang']


def test_ambiguous_festival_rejected():
    with pytest.raises(ValueError):
        festival_dates(holidays().iloc[1:], '설날')


def test_incomplete_response_rejected():
    with pytest.raises(ValueError):
        parse_year({'response': {'header': {'resultCode': '00'}, 'body': {'totalCount': 10,'items': ''}}}, 2024)
