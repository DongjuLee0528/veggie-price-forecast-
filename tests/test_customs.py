import pandas as pd
import pytest

from src.collect.customs import parse_trade, aggregate_trade


def test_source_and_missing():
    xml = '<response><header><resultCode>00</resultCode></header><body><items><item><year>2024.01</year><hsCd>0703101010</hsCd><statCd>CN</statCd><impWgt>1,000</impWgt><impDlr>50</impDlr></item><item><year>총계</year></item></items></body></response>'
    result = parse_trade(xml, '0703101010', 'CN', pd.Period('2024-01'), pd.Period('2024-02'))
    assert result.import_kg.iloc[0] == 1000
    assert pd.isna(result.import_kg.iloc[1])
    assert result.status.tolist() == ['observed', 'no_record']
    with pytest.raises(ValueError):
        parse_trade(xml, '0703101090', 'CN', pd.Period('2024-01'), pd.Period('2024-02'))


def test_incomplete_code_sum_remains_missing():
    rows = pd.DataFrame({'month': [pd.Timestamp('2024-01-01')] * 2, 'item': ['양파'] * 2, 'country': ['CN'] * 2, 'hsk': ['a','b'], 'import_kg': [1., float('nan')], 'import_usd': [2., float('nan')]})
    result = aggregate_trade(rows).iloc[0]
    assert pd.isna(result.import_kg)
    assert result.codes_observed == 1
