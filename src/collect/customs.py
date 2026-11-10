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


def collect(start, end):
    start, end = pd.Period(start, freq='M'), pd.Period(end, freq='M')
    if start > end or end >= pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).to_period('M'):
        raise ValueError('Use an ordered range of completed months')
    config = load_items()
    key = require_api_key('DATA_GO_KR_KEY')
    country = config['trade_country']
    output = ROOT / 'data/raw/trade'
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    with requests.Session() as session:
        for item, settings in config['items'].items():
            if settings['type'] != 'storage':
                continue
            for code in settings['hsk']:
                for year in range(start.year, end.year + 1):
                    first, last = max(start, pd.Period(f'{year}-01')), min(end, pd.Period(f'{year}-12'))
                    path = output / f'customs_{country}_{code}_{first:%Y%m}_{last:%Y%m}.parquet'
                    if path.exists():
                        frame = pd.read_parquet(path)
                        expected = pd.period_range(first, last, freq='M').to_timestamp()
                        if len(frame) != len(expected) or set(frame.month) != set(expected) or not frame.hsk.eq(code).all() or not frame.country.eq(country).all():
                            raise ValueError('Invalid cached customs data')
                    else:
                        frame = fetch_trade(session, key, code, country, first, last)
                        temp = path.with_suffix('.tmp.parquet')
                        frame.to_parquet(temp, index=False)
                        temp.replace(path)
                    frame = frame.assign(item=item)
                    frames.append(frame)
                    print(f'{item} {code} {year}: observed={int(frame.status.eq("observed").sum())}/{len(frame)}')
    rows = pd.concat(frames, ignore_index=True)
    target = ROOT / 'data/interim/trade'
    target.mkdir(parents=True, exist_ok=True)
    aggregate_trade(rows).to_parquet(target / 'imports_monthly.parquet', index=False)
    rows.to_csv(target / 'hsk_coverage.csv', index=False)


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
