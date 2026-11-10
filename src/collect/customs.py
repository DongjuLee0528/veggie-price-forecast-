import argparse
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import load_items, require_api_key


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = 'https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList'


def parse_trade(content, code, country, start, end):
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        raise ValueError('Customs response is not XML') from None
    if root.findtext('./header/resultCode') != '00':
        raise ValueError('Customs API rejected request')
    records = []
    for row in root.findall('./body/items/item'):
        period = (row.findtext('year') or '').strip()
        if period in ('총계', '합계'):
            continue
        if not re.fullmatch(r'\d{4}\.\d{2}', period):
            raise ValueError('Unexpected customs period')
        month = pd.Period(period.replace('.', '-'), freq='M')
        if not start <= month <= end or row.findtext('hsCd') != code or row.findtext('statCd') != country:
            raise ValueError('Customs response does not match request')
        values = []
        for field in ('impWgt', 'impDlr'):
            try:
                value = float((row.findtext(field) or '').replace(',', ''))
            except ValueError:
                raise ValueError('Invalid customs import value') from None
            if not np.isfinite(value) or value < 0:
                raise ValueError('Invalid customs import value')
            values.append(value)
        records.append({'month': month.to_timestamp(), 'import_kg': values[0], 'import_usd': values[1]})
    frame = pd.DataFrame(records, columns=['month', 'import_kg', 'import_usd'])
    if frame.month.duplicated().any():
        raise ValueError('Duplicate customs monthly rows')
    frame = frame.set_index('month').reindex(pd.period_range(start, end, freq='M').to_timestamp().rename('month'))
    frame['hsk'] = code
    frame['country'] = country
    frame['status'] = np.where(frame.import_kg.notna(), 'observed', 'no_record')
    return frame.reset_index()


def fetch_trade(session, key, code, country, start, end):
    params = {'serviceKey': unquote(key), 'strtYymm': start.strftime('%Y%m'), 'endYymm': end.strftime('%Y%m'), 'hsSgn': code, 'cntyCd': country}
    for attempt in range(3):
        try:
            response = session.get(ENDPOINT, params=params, timeout=30, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError(f'Customs HTTP {response.status_code}')
            return parse_trade(response.content, code, country, start, end)
        except (requests.RequestException, RuntimeError) as error:
            if attempt == 2:
                kind = type(error).__name__ if isinstance(error, requests.RequestException) else str(error)
                raise RuntimeError(f'{kind}; HSK={code}; serviceKey=***') from None
            time.sleep(2 ** attempt)


def aggregate_trade(rows):
    return rows.groupby(['month', 'item', 'country'], as_index=False).agg(
        import_kg=('import_kg', lambda values: values.sum(min_count=len(values))),
        import_usd=('import_usd', lambda values: values.sum(min_count=len(values))),
        codes_expected=('hsk', 'size'), codes_observed=('import_kg', 'count'),
    )


GROUPS = {
    '양파': [('070310', {'0703101000', '0703101010', '0703101090'}, 'onion')],
    '깐마늘': [('070320', {'0703201000', '0703209000'}, 'fresh_chilled'),
              ('0710802000', {'0710802000'}, 'frozen')],
    '건고추': [('0904210000', {'0904210000'}, 'dried'),
              ('0904220000', {'0904220000'}, 'powder')],
}


def parse_parent(content, prefix, country, start, end):
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        raise ValueError('Customs response is not XML') from None
    if root.findtext('./header/resultCode') != '00':
        raise ValueError('Customs API rejected request')
    records, totals = [], []
    for row in root.findall('./body/items/item'):
        period = (row.findtext('year') or '').strip()
        values = []
        for field in ('impWgt', 'impDlr'):
            try:
                number = float((row.findtext(field) or '').replace(',', ''))
            except ValueError:
                raise ValueError('Invalid customs total or detail value') from None
            if not np.isfinite(number) or number < 0:
                raise ValueError('Invalid customs total or detail value')
            values.append(number)
        if period in ('총계', '합계'):
            totals.append(values)
            continue
        if not re.fullmatch(r'\d{4}\.\d{2}', period):
            raise ValueError('Unexpected customs period')
        month = pd.Period(period.replace('.', '-'), freq='M')
        code = (row.findtext('hsCd') or '').strip()
        if not start <= month <= end or len(code) != 10 or not code.startswith(prefix) or row.findtext('statCd') != country:
            raise ValueError('Customs parent response does not match request')
        records.append({'month': month.to_timestamp(), 'hsk': code,
                        'country': country, 'import_kg': values[0], 'import_usd': values[1]})
    columns = ['month', 'hsk', 'country', 'import_kg', 'import_usd']
    frame = pd.DataFrame(records, columns=columns)
    if frame.duplicated(['month', 'hsk', 'country']).any():
        raise ValueError('Duplicate customs parent details')
    if frame.empty:
        if totals and any(value != 0 for total in totals for value in total):
            raise ValueError('Customs total has no corresponding details')
        return frame, 'no_records'
    if len(totals) != 1:
        raise ValueError('Expected one customs grand total')
    differences = {}
    for index, field in enumerate(('import_kg', 'import_usd')):
        difference = float(totals[0][index] - frame[field].sum())
        differences[field] = difference
        if abs(difference) > (len(frame) + 1) * 0.5:
            raise ValueError('Customs details do not reconcile with grand total')
    status = 'reconciled' if all(value == 0 for value in differences.values()) else f'source_total_difference:{differences}'
    return frame, status


def summarize_parent(frame, allowed, start, end):
    selected = frame[frame.hsk.isin(allowed)]
    monthly = selected.groupby('month')[['import_kg', 'import_usd']].sum()
    monthly = monthly.reindex(pd.period_range(start, end, freq='M').to_timestamp().rename('month'))
    monthly['status'] = np.where(monthly.import_kg.notna(), 'observed', 'no_record')
    return monthly.reset_index()


def collect(start, end):
    start, end = pd.Period(start, freq='M'), pd.Period(end, freq='M')
    if start > end or end >= pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).to_period('M'):
        raise ValueError('Use an ordered range of completed months')
    config = load_items()
    key = require_api_key('DATA_GO_KR_KEY')
    country = config['trade_country']
    output = ROOT / 'data/raw/trade/parent_v1'
    output.mkdir(parents=True, exist_ok=True)
    frames, audits = [], []
    with requests.Session() as session:
        for item, groups in GROUPS.items():
            for prefix, allowed, form in groups:
                if not (allowed - {'0703101000'}) <= set(config['items'][item]['hsk']):
                    raise ValueError('Customs policy conflicts with configured HSK codes')
                for year in range(start.year, end.year + 1):
                    first, last = max(start, pd.Period(f'{year}-01')), min(end, pd.Period(f'{year}-12'))
                    path = output / f'customs_{country}_{prefix}_{first:%Y%m}_{last:%Y%m}.xml'
                    if path.exists():
                        content = path.read_bytes()
                    else:
                        params = {'serviceKey': unquote(key), 'strtYymm': first.strftime('%Y%m'), 'endYymm': last.strftime('%Y%m'), 'hsSgn': prefix, 'cntyCd': country}
                        for attempt in range(3):
                            try:
                                response = session.get(ENDPOINT, params=params, timeout=30, allow_redirects=False)
                                if response.status_code != 200:
                                    raise RuntimeError(f'Customs HTTP {response.status_code}')
                                content = response.content
                                break
                            except (requests.RequestException, RuntimeError) as error:
                                if attempt == 2:
                                    kind = type(error).__name__ if isinstance(error, requests.RequestException) else str(error)
                                    raise RuntimeError(f'{kind}; prefix={prefix}; serviceKey=***') from None
                                time.sleep(2 ** attempt)
                    detail, audit = parse_parent(content, prefix, country, first, last)
                    if not path.exists():
                        if key.encode() in content or unquote(key).encode() in content:
                            raise ValueError('Refusing to store credential-bearing response')
                        temporary = path.with_suffix('.tmp')
                        temporary.write_bytes(content)
                        temporary.replace(path)
                    detail.to_parquet(path.with_suffix('.parquet'), index=False)
                    frame = summarize_parent(detail, allowed, first, last).assign(item=item, country=country, form=form)
                    frames.append(frame)
                    audits.append({'item': item, 'form': form, 'prefix': prefix, 'start': str(first), 'end': str(last), 'reconciliation': audit, 'detail_rows': len(detail), 'selected_rows': int(detail.hsk.isin(allowed).sum())})
                    print(f'{item} {form} {year}: {audit}, observed={int(frame.status.eq("observed").sum())}/{len(frame)}')
    forms = pd.concat(frames, ignore_index=True)
    monthly = forms.groupby(['month', 'item', 'country'], as_index=False).agg(
        import_kg=('import_kg', lambda values: values.sum(min_count=len(values))),
        import_usd=('import_usd', lambda values: values.sum(min_count=len(values))),
        forms_expected=('form', 'size'), forms_observed=('import_kg', 'count'))
    target = ROOT / 'data/interim/trade'
    target.mkdir(parents=True, exist_ok=True)
    forms.to_parquet(target / 'imports_by_form.parquet', index=False)
    monthly.to_parquet(target / 'imports_monthly.parquet', index=False)
    pd.DataFrame(audits).to_csv(target / 'parent_reconciliation.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2020-01')
    parser.add_argument('--end', required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / '.env', override=False)
    try:
        collect(args.start, args.end)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
