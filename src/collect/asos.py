import argparse
import time
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import load_items, require_api_key


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = 'https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
FIELDS = {'avgTa': 'temp_mean_c', 'maxTa': 'temp_max_c', 'minTa': 'temp_min_c', 'sumRn': 'rain_mm', 'sumSsHr': 'sunshine_hours'}


def station_ids():
    return sorted({station for item in load_items()['items'].values() for rule in item['stations'] for station in rule['ids']})


def parse_response(payload, station, start, end):
    try:
        response = payload['response']
        code = str(response['header']['resultCode'])
        if code == '03':
            rows = []
        elif code == '00':
            body = response['body']
            items = body.get('items')
            rows = items.get('item', []) if isinstance(items, dict) else []
            if isinstance(rows, dict):
                rows = [rows]
            if not isinstance(rows, list) or int(body['totalCount']) != len(rows):
                raise ValueError('Incomplete ASOS response')
        else:
            raise ValueError('ASOS API rejected request')
    except (KeyError, TypeError):
        raise ValueError('Invalid ASOS response schema') from None
    records = []
    for row in rows:
        if int(row['stnId']) != station:
            raise ValueError('Unexpected ASOS station')
        date = pd.Timestamp(row['tm'])
        if not start <= date <= end or date != date.normalize():
            raise ValueError('Unexpected ASOS date')
        record = {'date': date, 'station_id': station, 'observed': True}
        for source, target in FIELDS.items():
            value = row.get(source)
            number = np.nan if value is None or str(value).strip() == '' else float(value)
            if not pd.isna(number) and not np.isfinite(number):
                raise ValueError('Nonfinite ASOS value')
            if target in ('rain_mm', 'sunshine_hours') and number < 0:
                raise ValueError('Negative ASOS accumulation')
            record[target] = number
        records.append(record)
    frame = pd.DataFrame(records, columns=['date', 'station_id', 'observed', *FIELDS.values()])
    if frame.date.duplicated().any():
        raise ValueError('Duplicate ASOS dates')
    frame = frame.set_index('date').reindex(pd.date_range(start, end, name='date'))
    frame['station_id'] = station
    frame['observed'] = frame.observed.fillna(False).astype(bool)
    return frame.reset_index()


def fetch_station(session, key, station, start, end):
    params = {'serviceKey': unquote(key), 'pageNo': 1, 'numOfRows': 999, 'dataType': 'JSON', 'dataCd': 'ASOS', 'dateCd': 'DAY', 'startDt': start.strftime('%Y%m%d'), 'endDt': end.strftime('%Y%m%d'), 'stnIds': station}
    for attempt in range(3):
        try:
            response = session.get(ENDPOINT, params=params, timeout=30, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError(f'ASOS HTTP {response.status_code}')
            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError('ASOS returned non-JSON content') from None
            break
        except (requests.RequestException, RuntimeError) as error:
            if attempt == 2:
                kind = type(error).__name__ if isinstance(error, requests.RequestException) else str(error)
                raise RuntimeError(f'{kind}; station={station}, start={start.date()}; serviceKey=***') from None
            time.sleep(2 ** attempt)
    return parse_response(payload, station, start, end)


def collect(start, end, stations=None, output=ROOT / 'data/raw/weather'):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start > end or start != start.normalize() or end != end.normalize():
        raise ValueError('Invalid ASOS date range')
    if end >= pd.Timestamp.now(tz='Asia/Seoul').tz_localize(None).normalize():
        raise ValueError('ASOS end date must be before today in Korea')
    key = require_api_key('DATA_GO_KR_KEY')
    stations = station_ids() if stations is None else stations
    if not stations or not set(stations) <= set(station_ids()):
        raise ValueError('Use configured ASOS stations only')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for station in stations:
            for month in pd.period_range(start, end, freq='M'):
                first, last = max(start, month.start_time), min(end, month.end_time.normalize())
                path = output / f'asos_{station}_{first:%Y%m%d}_{last:%Y%m%d}.parquet'
                if path.exists():
                    frame = pd.read_parquet(path)
                    if not frame.date.equals(pd.Series(pd.date_range(first, last), name='date')) or not frame.station_id.eq(station).all():
                        raise ValueError(f'Invalid existing ASOS file: {path.name}')
                else:
                    frame = fetch_station(session, key, station, first, last)
                    temporary = path.with_suffix('.tmp.parquet')
                    frame.to_parquet(temporary, index=False)
                    temporary.replace(path)
                print(f'{station} {first:%Y-%m-%d}..{last:%Y-%m-%d}: observed={int(frame.observed.sum())}/{len(frame)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--station', type=int, action='append')
    args = parser.parse_args()
    load_dotenv(ROOT / '.env', override=False)
    try:
        collect(args.start, args.end, args.station)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
