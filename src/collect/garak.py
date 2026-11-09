import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import require_api_key


ENDPOINT = 'http://www.garak.co.kr/homepage/publicdata/dataJsonOpen.do'
MATCHES = {'배추': ('엽채류', '배추'), '무': ('근채류', '무'), '양파': ('조미채류', '양파'), '깐마늘': ('조미채류', '마늘'), '건고추': ('조미채류', '건고추')}
OUTPUT = Path(__file__).resolve().parents[2] / 'data/raw/volume'


def request_page(day, page, session, params, collected_rows):
    for attempt in range(3):
        try:
            response = session.get(ENDPOINT, params=params, timeout=30, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError(f'Garak HTTP {response.status_code}')
            try:
                return response.json()
            except ValueError:
                body = response.content
                text = body.decode('utf-8', errors='replace')
                if re.search(r'LIST_COUNT\s*:\s*0\s*[,}]', text) and re.search(r'ApiErrorMsg\s*:\s*데이터가 없습니다\.', text):
                    return {'resultData': []}
                kind = 'empty' if not body.strip() else ('html' if body.lstrip().lower().startswith((b'<!doctype html', b'<html')) else 'other')
                raise RuntimeError(f'Garak returned non-JSON content; kind={kind}, bytes={len(body)}') from None
        except (requests.RequestException, RuntimeError) as error:
            message = type(error).__name__ if isinstance(error, requests.RequestException) else str(error)
            if attempt == 2:
                raise RuntimeError(f'{message}; date={day.date()}, page={page}, collected_rows={collected_rows}, attempts=3; id=***&passwd=***') from None
            time.sleep(2 ** attempt)


def fetch_day(day, session, user_id, password):
    records, seen = [], set()
    for page in range(1, 101):
        params = {'id': user_id, 'passwd': password, 'dataid': 'data20', 'pagesize': 100, 'pageidx': page, 'portal.templet': 'false', 'date': day.strftime('%Y%m%d')}
        payload = request_page(day, page, session, params, len(records))
        if not isinstance(payload, dict) or not isinstance(payload.get('resultData'), list):
            raise ValueError(f'Invalid Garak response for {day.date()}, page {page}')
        rows = payload['resultData']
        if not rows:
            return records
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError('Invalid Garak result row')
        signature = json.dumps(rows, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            raise ValueError(f'Repeated Garak page for {day.date()}')
        seen.add(signature)
        records.extend(rows)
        if len(rows) < params['pagesize']:
            return records
    raise ValueError(f'Garak pagination limit reached for {day.date()}')


def parse_day(day, rows):
    matches = {item: [] for item in MATCHES}
    for row in rows:
        if not {'BURYU', '구분', 'TOT'} <= row.keys():
            raise ValueError('Missing Garak BURYU, 구분 or TOT field')
        category, name = str(row['BURYU']).strip(), str(row['구분']).strip()
        for item, pair in MATCHES.items():
            if (category, name) == pair:
                try:
                    value = float(str(row['TOT']).replace(',', '').strip())
                except ValueError:
                    raise ValueError(f'Invalid Garak volume for {item} on {day.date()}') from None
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f'Invalid Garak volume for {item} on {day.date()}')
                matches[item].append(value)
    records = []
    for item, values in matches.items():
        if len(values) > 1:
            raise ValueError(f'Duplicate Garak item total for {item} on {day.date()}')
        records.append({'date': day, 'item': item, 'source_item': MATCHES[item][1], 'volume_ton': values[0] if values else np.nan, 'status': 'observed' if values else ('no_data' if not rows else 'item_missing')})
    return pd.DataFrame(records)


def collect(start, end, output=OUTPUT):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start < pd.Timestamp('2021-01-01') or end < start or start != start.normalize() or end != end.normalize():
        raise ValueError('Expected date range starting on or after 2021-01-01')
    user_id, password = require_api_key('GARAK_ID'), require_api_key('GARAK_PASSWD')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    with requests.Session() as session:
        for day in pd.date_range(start, end):
            path = output / f'garak_{day:%Y%m%d}.parquet'
            if path.exists():
                frame = pd.read_parquet(path)
                if len(frame) != 5 or set(frame.item) != set(MATCHES) or not frame.date.eq(day).all():
                    raise ValueError(f'Invalid existing Garak file: {path.name}')
            else:
                frame = parse_day(day, fetch_day(day, session, user_id, password))
                temporary = path.with_suffix('.tmp.parquet')
                frame.to_parquet(temporary, index=False)
                temporary.replace(path)
            frames.append(frame)
            print(f'{day:%Y-%m-%d}: {frame.status.value_counts().to_dict()}')
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / '.env', override=False)
    try:
        collect(args.start, args.end)
    except (RuntimeError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
