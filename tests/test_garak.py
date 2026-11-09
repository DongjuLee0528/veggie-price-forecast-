from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from src.collect.garak import fetch_day, parse_day


DAY = pd.Timestamp('2024-04-11')


def response(rows):
    r = Mock(status_code=200)
    r.json.return_value = {'resultData': rows}
    return r


def test_pagination():
    session = Mock()
    first = [{'x': n} for n in range(100)]
    second = [{'x': n} for n in range(100, 200)]
    session.get.side_effect = [response(first), response(second), response([])]
    assert fetch_day(DAY, session, 'dummy-id', 'dummy-password') == first + second
    assert session.get.call_count == 3


def test_secret_safe_error():
    session = Mock()
    session.get.side_effect = requests.ConnectionError('id=dummy-id&passwd=dummy-password')
    with pytest.raises(RuntimeError) as error:
        fetch_day(DAY, session, 'dummy-id', 'dummy-password')
    assert 'dummy-id' not in str(error.value)
    assert 'dummy-password' not in str(error.value)
    assert error.value.__suppress_context__


def test_repeated_page():
    session = Mock()
    session.get.return_value = response([{'x': n} for n in range(100)])
    with pytest.raises(ValueError, match='Repeated'):
        fetch_day(DAY, session, 'dummy-id', 'dummy-password')


def test_matching_and_missing():
    rows = [{'BURYU': '엽채류', '구분': '배추', 'TOT': '1,234.5'},
            {'BURYU': '엽채류', '구분': '얼갈이배추', 'TOT': '999'},
            {'BURYU': '조미채류', '구분': '마늘', 'TOT': '0'}]
    result = parse_day(DAY, rows).set_index('item')
    assert result.loc['배추', 'volume_ton'] == 1234.5
    assert result.loc['깐마늘', 'volume_ton'] == 0
    assert result.loc['깐마늘', 'source_item'] == '마늘'
    assert pd.isna(result.loc['무', 'volume_ton'])
    assert result.loc['무', 'status'] == 'item_missing'
    assert parse_day(DAY, []).status.eq('no_data').all()


@pytest.mark.parametrize('value', ['bad', '-1', 'NaN', 'inf'])
def test_invalid_volume(value):
    with pytest.raises(ValueError):
        parse_day(DAY, [{'BURYU': '근채류', '구분': '무', 'TOT': value}])


def test_duplicate_total():
    row = {'BURYU': '근채류', '구분': '무', 'TOT': '1'}
    with pytest.raises(ValueError, match='Duplicate'):
        parse_day(DAY, [row, row])


@pytest.mark.parametrize('status', [302, 403, 500])
def test_http_failure_reports_only_status(status):
    session = Mock()
    session.get.return_value = Mock(status_code=status)
    with pytest.raises(RuntimeError, match=f'HTTP {status}'):
        fetch_day(DAY, session, 'dummy-id', 'dummy-password')


def test_non_json_response():
    session = Mock()
    session.get.return_value = Mock(status_code=200)
    session.get.return_value.content = b'<html>secret response content</html>'
    session.get.return_value.json.side_effect = ValueError('secret response content')
    with pytest.raises(RuntimeError, match='non-JSON') as error:
        fetch_day(DAY, session, 'dummy-id', 'dummy-password')
    assert 'secret response content' not in str(error.value)


def test_short_page_stops_before_next_request():
    session = Mock()
    session.get.side_effect = [response([{'x': n} for n in range(100)]), response([{'x': 100}])]
    assert len(fetch_day(DAY, session, 'dummy-id', 'dummy-password')) == 101
    assert session.get.call_count == 2


def test_empty_non_json_is_not_assumed_success():
    session = Mock()
    result = Mock(status_code=200, content=b'')
    result.json.side_effect = ValueError('bad')
    session.get.return_value = result
    with pytest.raises(RuntimeError, match='kind=empty'):
        fetch_day(DAY, session, 'dummy-id', 'dummy-password')
