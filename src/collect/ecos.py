import argparse
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import load_items, require_api_key


ROOT = Path(__file__).resolve().parents[2]


def parse_rates(payload, table, item_code, currency, start, end):
    if 'StatisticSearch' not in payload:
        raise ValueError('ECOS API rejected request or returned no data')
    body = payload['StatisticSearch']
    rows = body.get('row')
    if not isinstance(rows, list) or not rows or int(body['list_total_count']) != len(rows):
        raise ValueError('Incomplete ECOS response')
    records = []
    for row in rows:
        if row.get('STAT_CODE') != table or row.get('ITEM_CODE1') != item_code:
            raise ValueError('ECOS series does not match request')
        date = pd.to_datetime(row['TIME'], format='%Y%m%d')
        if not start <= date <= end:
            raise ValueError('ECOS date outside requested range')
        try:
            value = float(str(row['DATA_VALUE']).replace(',', ''))
        except ValueError:
            raise ValueError('Invalid ECOS rate') from None
        if not np.isfinite(value) or value <= 0:
            raise ValueError('ECOS rates must be finite and positive')
        records.append({'date': date, 'currency': currency, 'krw_per_unit': value,
                        'item_code': item_code, 'unit': row['UNIT_NAME'], 'series_name': row['ITEM_NAME1']})
    frame = pd.DataFrame(records)
    if frame.date.duplicated().any() or frame.unit.nunique() != 1:
        raise ValueError('Duplicate ECOS dates or inconsistent units')
    return frame.sort_values('date').reset_index(drop=True)


def fetch_rates(session, key, table, cycle, item_code, currency, start, end):
    url = f'https://ecos.bok.or.kr/api/StatisticSearch/{quote(key, safe="")}/json/kr/1/1000/{table}/{cycle}/{start:%Y%m%d}/{end:%Y%m%d}/{item_code}'
    for attempt in range(3):
        try:
            response = session.get(url, timeout=30, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError(f'ECOS HTTP {response.status_code}')
            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError('ECOS response is not JSON') from None
            return parse_rates(payload, table, item_code, currency, start, end)
        except (requests.RequestException, RuntimeError) as error:
            if attempt == 2:
                detail = type(error).__name__ if isinstance(error, requests.RequestException) else str(error)
                raise RuntimeError(f'{detail}; currency={currency}, year={start.year}; ECOS_KEY=***') from None
            time.sleep(2 ** attempt)


def collect(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start > end or start != start.normalize() or end != end.normalize():
        raise ValueError('Invalid FX date range')
    if end >= pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).normalize():
        raise ValueError('Use dates before today in Korea')
    settings = load_items()['ecos']
    if settings['cycle'] != 'D':
        raise ValueError('FX collector requires daily cycle')
    key = require_api_key('ECOS_KEY')
    output = ROOT / 'data/raw/fx'
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    with requests.Session() as session:
        for currency, item_code in settings['items'].items():
            for year in range(start.year, end.year + 1):
                first, last = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))
                path = output / f'ecos_{currency}_{first:%Y%m%d}_{last:%Y%m%d}.parquet'
                if path.exists():
                    frame = pd.read_parquet(path)
                    if frame.empty or frame.date.duplicated().any() or not frame.date.between(first, last).all() or not frame.item_code.eq(item_code).all() or not frame.currency.eq(currency).all():
                        raise ValueError('Invalid cached FX data')
                else:
                    frame = fetch_rates(session, key, settings['table'], settings['cycle'], item_code, currency, first, last)
                    temporary = path.with_suffix('.tmp.parquet')
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                frames.append(frame)
                print(f'{currency} {year}: observations={len(frame)}, unit={frame.unit.iloc[0]}')
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(['date', 'currency']).any():
        raise ValueError('Duplicate FX date/currency keys')
    daily = data.pivot(index='date', columns='currency', values='krw_per_unit').reindex(pd.date_range(start, end, name='date'))
    daily.columns = [f'krw_per_{currency}' for currency in daily.columns]
    target = ROOT / 'data/interim/fx'
    target.mkdir(parents=True, exist_ok=True)
    daily.reset_index().to_parquet(target / 'fx_daily.parquet', index=False)
    print('Calendar-day missing values (not filled):', daily.isna().sum().to_dict())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2020-01-01')
    parser.add_argument('--end', required=True)
    args = parser.parse_args()
    load_dotenv(ROOT / '.env', override=False)
    try:
        collect(args.start, args.end)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
