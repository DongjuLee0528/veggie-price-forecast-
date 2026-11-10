from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from src.collect.ecos import parse_rates, fetch_rates


START, END = pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-31')


def payload():
    return {'StatisticSearch': {'list_total_count': 1, 'row': [{'STAT_CODE': '731Y001', 'ITEM_CODE1': '0000001', 'TIME': '20240102', 'DATA_VALUE': '1,300.5', 'UNIT_NAME': '원', 'ITEM_NAME1': '원/미국달러(매매기준율)'}]}}


def test_rate_and_metadata():
    result = parse_rates(payload(), '731Y001', '0000001', 'usd', START, END)
    assert result.krw_per_unit.item() == 1300.5
    assert result.item_code.item() == '0000001'


@pytest.mark.parametrize('field,value', [('DATA_VALUE','0'),('DATA_VALUE','NaN'),('ITEM_CODE1','wrong'),('TIME','20240201')])
def test_bad_rate(field, value):
    data = payload(); data['StatisticSearch']['row'][0][field] = value
    with pytest.raises(ValueError):
        parse_rates(data, '731Y001', '0000001', 'usd', START, END)


def test_incomplete_or_error_response():
    data = payload(); data['StatisticSearch']['list_total_count'] = 10
    with pytest.raises(ValueError):
        parse_rates(data, '731Y001', '0000001', 'usd', START, END)
    with pytest.raises(ValueError):
        parse_rates({'RESULT': {'CODE':'ERROR'}}, '731Y001', '0000001', 'usd', START, END)


def test_safe_network_error(monkeypatch):
    monkeypatch.setattr('src.collect.ecos.time.sleep',lambda _:None)
    session = Mock(); session.get.side_effect = requests.ConnectionError('secret-key')
    with pytest.raises(RuntimeError) as error:
        fetch_rates(session,'secret-key','731Y001','D','0000001','usd',START,END)
    assert 'secret-key' not in str(error.value)
    assert session.get.call_count == 3
