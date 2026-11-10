from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.asos import FIELDS, ROOT, station_ids


def load_weather(directory=ROOT / 'data/raw/weather', start='2020-01-01', end='2026-11-08'):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError('Invalid weather date range')
    files = sorted(Path(directory).glob('asos_*.parquet'))
    if not files:
        raise ValueError('No ASOS files found')
    columns = ['date', 'station_id', 'observed', *FIELDS.values()]
    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        if set(columns) - set(frame.columns):
            raise ValueError(f'Missing weather columns: {path.name}')
        frames.append(frame[columns])
    data = pd.concat(frames, ignore_index=True)
    if not pd.api.types.is_datetime64_any_dtype(data.date) or data.date.isna().any():
        raise ValueError('Invalid weather dates')
    if not data.date.eq(data.date.dt.normalize()).all():
        raise ValueError('Weather dates contain times')
    if not pd.api.types.is_bool_dtype(data.observed) or data.observed.isna().any():
        raise ValueError('Invalid weather observation flags')
    if data.station_id.isna().any() or not set(data.station_id) <= set(station_ids()):
        raise ValueError('Unknown weather station')
    for column in FIELDS.values():
        if not pd.api.types.is_numeric_dtype(data[column]) or not np.isfinite(data[column].dropna()).all():
            raise ValueError(f'Invalid weather numbers: {column}')
    if ((data.rain_mm < 0) | (data.sunshine_hours < 0)).any():
        raise ValueError('Negative weather accumulations')
    if ((data.temp_min_c > data.temp_mean_c) | (data.temp_mean_c > data.temp_max_c) | (data.temp_min_c > data.temp_max_c)).any():
        raise ValueError('Inconsistent weather temperatures')
    if data.loc[~data.observed, list(FIELDS.values())].notna().any().any():
        raise ValueError('Unobserved day contains weather values')
    data = data.drop_duplicates()
    if data.duplicated(['station_id', 'date']).any():
        raise ValueError('Conflicting weather records for station/date')
    data = data[data.date.between(start, end)]
    expected = pd.MultiIndex.from_product([station_ids(), pd.date_range(start, end)], names=['station_id', 'date'])
    actual = pd.MultiIndex.from_frame(data[['station_id', 'date']])
    if len(expected.difference(actual)):
        raise ValueError('Missing weather station/date rows')
    return data.sort_values(['station_id', 'date']).reset_index(drop=True)


def main():
    data = load_weather()
    output = ROOT / 'data/interim/weather'
    output.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output / 'asos_daily.parquet', index=False)
    summary = data.groupby('station_id')[list(FIELDS.values())].agg(lambda values: int(values.isna().sum()))
    summary.columns = [f'{column}_missing' for column in summary.columns]
    summary['source_missing_days'] = data.groupby('station_id')['observed'].apply(lambda values: int((~values).sum()))
    summary.to_csv(output / 'missing_summary.csv')
    print(f'Validated {len(data)} station-days; {int((~data.observed).sum())} source-missing days. Rain blanks retained as NaN.')


if __name__ == '__main__':
    main()
