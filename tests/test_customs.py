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


def test_parent_reconciliation_and_selective_sum():
    from src.collect.customs import parse_parent, summarize_parent
    def entry(period, code, weight):
        return f'<item><year>{period}</year><hsCd>{code}</hsCd><statCd>CN</statCd><impWgt>{weight}</impWgt><impDlr>{weight}</impDlr></item>'
    xml = '<response><header><resultCode>00</resultCode></header><body><items>' + entry('총계','-',60) + entry('2022.01','0703101000',10) + entry('2022.01','0703101010',20) + entry('2022.01','0703103000',30) + '</items></body></response>'
    start, end = pd.Period('2022-01'), pd.Period('2022-02')
    details, status = parse_parent(xml, '070310', 'CN', start, end)
    assert status == 'reconciled'
    result = summarize_parent(details, {'0703101000','0703101010','0703101090'}, start, end)
    assert result.import_kg.iloc[0] == 30
    assert pd.isna(result.import_kg.iloc[1])
    with pytest.raises(ValueError, match='reconcile'):
        parse_parent(xml.replace('<impWgt>60</impWgt>', '<impWgt>160</impWgt>'), '070310', 'CN', start, end)
