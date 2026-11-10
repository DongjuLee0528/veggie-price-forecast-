import argparse
import time
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

from src.config import require_api_key


ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = 'https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo'


def parse_year(payload, year):
    try:
        response = payload['response']
        if str(response['header']['resultCode']) != '00':
            raise ValueError('Holiday API rejected request')
        body = response['body']
        items = body.get('items')
        rows = items.get('item', []) if isinstance(items, dict) else []
        if isinstance(rows, dict):
            rows = [rows]
        if int(body['totalCount']) != len(rows):
            raise ValueError('Incomplete holiday response')
    except (KeyError, TypeError):
        raise ValueError('Invalid holiday response schema') from None
    records = []
    for row in rows:
        date = pd.to_datetime(str(row['locdate']), format='%Y%m%d')
        if date.year != year or row['isHoliday'] not in ('Y', 'N'):
            raise ValueError('Invalid holiday record')
        records.append({'date': date, 'name': str(row['dateName']).strip(), 'is_holiday': row['isHoliday'] == 'Y'})
    result = pd.DataFrame(records, columns=['date', 'name', 'is_holiday'])
    if result.empty or result.duplicated(['date', 'name']).any():
        raise ValueError('Empty or duplicate holiday records')
    return result.sort_values(['date', 'name']).reset_index(drop=True)


def festival_dates(holidays, name):
    dates = []
    for year in sorted(holidays.date.dt.year.unique()):
        selected = holidays.loc[holidays.date.dt.year.eq(year) & holidays.name.eq(name), 'date'].sort_values().drop_duplicates()
        if len(selected) != 3 or not selected.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError(f'Expected three consecutive {name} dates in {year}')
        dates.append(selected.iloc[1])
    return pd.DatetimeIndex(dates)


def build_calendar(holidays, start, end, kimjang_start='11-01', kimjang_end='12-10'):
    dates = pd.date_range(start, end, name='date')
    result = pd.DataFrame({'date': dates})
    result['is_holiday'] = result.date.isin(holidays.loc[holidays.is_holiday, 'date'])
    result['month'] = result.date.dt.month
    result['dow'] = result.date.dt.dayofweek
    md = result.date.dt.strftime('%m-%d')
    result['is_kimjang'] = md.between(kimjang_start, kimjang_end)
    for name, column in [('설날', 'days_to_seollal'), ('추석', 'days_to_chuseok')]:
        festivals = festival_dates(holidays, name)
        next_dates = pd.Series(festivals, index=festivals).reindex(dates.union(festivals)).sort_index().bfill().reindex(dates)
        result[column] = (next_dates.to_numpy() - dates.to_numpy()) / pd.Timedelta(days=1)
    return result


def fetch_year(session, key, year):
    params = {'serviceKey': unquote(key), 'solYear': year, 'numOfRows': 100, 'pageNo': 1, '_type': 'json'}
    for attempt in range(3):
        try:
            response = session.get(ENDPOINT, params=params, timeout=30, allow_redirects=False)
            if response.status_code != 200:
                raise RuntimeError(f'Holiday HTTP {response.status_code}')
            try:
                payload = response.json()
            except ValueError:
                raise RuntimeError('Holiday response is not JSON') from None
            return parse_year(payload, year)
        except (requests.RequestException, RuntimeError) as error:
            if attempt == 2:
                detail = type(error).__name__ if isinstance(error, requests.RequestException) else str(error)
                raise RuntimeError(f'{detail}; year={year}; serviceKey=***') from None
            time.sleep(2 ** attempt)


def collect(start_year=2019, end_year=2027):
    if not 1900 <= start_year <= end_year <= 2100:
        raise ValueError('Invalid calendar year range')
    key = require_api_key('DATA_GO_KR_KEY')
    output = ROOT / 'data/raw/calendar'
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    with requests.Session() as session:
        for year in range(start_year, end_year + 1):
            path = output / f'holidays_{year}.parquet'
            if path.exists():
                frame = pd.read_parquet(path)
                if not frame.date.dt.year.eq(year).all():
                    raise ValueError('Cached holiday year mismatch')
            else:
                frame = fetch_year(session, key, year)
                festival_dates(frame, '설날')
                festival_dates(frame, '추석')
                temporary = path.with_suffix('.tmp.parquet')
                frame.to_parquet(temporary, index=False)
                temporary.replace(path)
            frames.append(frame)
            print(f'{year}: holiday records={len(frame)}')
    holidays = pd.concat(frames, ignore_index=True)
    with (ROOT / 'configs/features.yaml').open(encoding='utf-8') as stream:
        settings = yaml.safe_load(stream)['calendar_rules']
    daily = build_calendar(holidays, f'{start_year}-01-01', f'{end_year}-12-31', **settings)
    target = ROOT / 'data/interim/calendar'
    target.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(target / 'calendar_daily.parquet', index=False)
    print(f'Daily calendar: {len(daily)} dates; next-festival values beyond collected years remain NaN.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-year', type=int, default=2019)
    parser.add_argument('--end-year', type=int, default=2027)
    args = parser.parse_args()
    load_dotenv(ROOT / '.env', override=False)
    try:
        collect(args.start_year, args.end_year)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
