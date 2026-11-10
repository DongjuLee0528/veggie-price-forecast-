from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from src.collect.asos import parse_response, fetch_station, station_ids


START, END = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-02')


def payload(rows):
    return {'response': {'header': {'resultCode': '00'}, 'body': {'totalCount': len(rows), 'items': {'item': rows}}}}


def row():
    return {'stnId': '100', 'tm': '2024-01-01', 'avgTa': '-2', 'maxTa': '3', 'minTa': '-5', 'sumRn': '', 'sumSsHr': '0'}


def test_missing_not_zero_and_calendar():
    result = parse_response(payload([row()]), 100, START, END)
    assert len(result) == 2
    assert pd.isna(result.rain_mm.iloc[0])
    assert result.sunshine_hours.iloc[0] == 0
    assert result.observed.tolist() == [True, False]
    assert result.temp_mean_c.iloc[0] == -2


@pytest.mark.parametrize('field,value', [('stnId', '216'), ('tm', '2024-02-01'), ('sumRn', '-1')])
def test_bad_source_values(field, value):
    entry = row(); entry[field] = value
    with pytest.raises(ValueError):
        parse_response(payload([entry]), 100, START, END)


def test_duplicate_and_incomplete():
    with pytest.raises(ValueError):
        parse_response(payload([row(), row()]), 100, START, END)
    data = payload([row()]); data['response']['body']['totalCount'] = 20
    with pytest.raises(ValueError):
        parse_response(data, 100, START, END)


def test_no_data_and_error():
    result = parse_response({'response': {'header': {'resultCode': '03'}}}, 100, START, END)
    assert not result.observed.any()
    with pytest.raises(ValueError):
        parse_response({'response': {'header': {'resultCode': '30'}}}, 100, START, END)


def test_secret_safe_failure(monkeypatch):
    monkeypatch.setattr('src.collect.asos.time.sleep', lambda _: None)
    session = Mock()
    session.get.side_effect = requests.ConnectionError('secret-value')
    with pytest.raises(RuntimeError) as error:
        fetch_station(session, 'secret-value', 100, START, END)
    assert 'secret-value' not in str(error.value)
    assert session.get.call_count == 3


def test_station_union():
    assert {100, 216, 261, 184, 189, 165, 285, 288, 262, 185, 136, 271, 272, 278} == set(station_ids())


@pytest.mark.parametrize('key', ['example%2Bkey%3D', 'example+key='])
def test_key_decoded_once_before_requests_encoding(key):
    session = Mock()
    response = Mock(status_code=200)
    response.json.return_value = payload([row()])
    session.get.return_value = response
    fetch_station(session, key, 100, START, END)
    assert session.get.call_args.kwargs['params']['serviceKey'] == 'example+key='
