import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.config import load_items
from src.evaluate.backtest import score_predictions


def load_forecast(directory, items):
    source = (directory / 'forecast.json').read_bytes()
    report = json.loads(source)
    frame = pd.DataFrame(report['forecasts'])
    csv = pd.read_csv(directory / 'predictions.csv')
    frame['target_date'] = frame.target_date.fillna('')
    csv['target_date'] = csv.target_date.astype('string').fillna('')
    pd.testing.assert_frame_equal(
        frame[sorted(frame.columns)].sort_values(['item', 'horizon']).reset_index(drop=True),
        csv[sorted(csv.columns)].sort_values(['item', 'horizon']).reset_index(drop=True),
        check_dtype=False, rtol=1e-12, atol=1e-12,
    )
    expected = {(item, horizon) for item in items for horizon in [1, 5, 10]}
    date = pd.Timestamp(report['base_date'])
    if (report['status'] != 'complete' or report['unit'] != 'KRW/kg'
            or len(frame) != len(expected) or set(zip(frame.item, frame.horizon)) != expected
            or not frame.base_date.eq(report['base_date']).all()
            or pd.isna(date) or date != date.normalize()
            or pd.Timestamp(report['trained_through']) > date):
        raise ValueError('Expected complete point forecasts for one valid origin')
    numeric = frame[['prediction_krw_per_kg', 'base_price_7day_krw_per_kg']]
    if not np.isfinite(numeric.to_numpy()).all() or not numeric.gt(0).all().all():
        raise ValueError('Forecast and base prices must be finite and positive')
    if not pd.to_datetime(frame.loc[frame.target_date.ne(''), 'target_date']).gt(date).all():
        raise ValueError('Target dates must be after origin')
    return date, frame, hashlib.sha256(source).hexdigest()[:16]


def seed_tables(directory, backtest_path, backtest_model):
    items = load_items()['items']
    item_ids = {item: number for number, item in enumerate(items, 1)}
    date, forecast, version = load_forecast(directory, items)
    prices = pd.read_parquet(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    prices = prices[prices.date.le(date)].copy()
    prices['item_id'] = prices.item.map(item_ids)
    if prices.item_id.isna().any() or prices.duplicated(['item_id', 'date']).any():
        raise ValueError('Unexpected or duplicate price series')
    if set(prices.loc[prices.date.eq(date), 'item']) != set(items):
        raise ValueError('Price series is missing forecast-origin observations')
    table_items = pd.DataFrame([{'id': item_ids[item], 'name': item,
                                 'type': config['type'].upper(), 'unit': 'KRW/kg'}
                                for item, config in items.items()])
    table_prices = prices.rename(columns={'date': 'trade_date'})[
        ['item_id', 'trade_date', 'price_per_kg', 'volume_kg']]
    created = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    runs = [{'id': 1, 'base_date': date, 'source': 'LIVE',
             'model_version': version, 'created_at': created}]
    short = forecast.rename(columns={'prediction_krw_per_kg': 'p50', 'base_price_7day_krw_per_kg': 'base_price'})
    short = short.assign(run_id=1, item_id=short.item.map(item_ids), p10=None, p90=None)
    metric_columns = ['item_id', 'horizon', 'period_start', 'period_end', 'nmae', 'direction_acc', 'coverage']
    metrics = pd.DataFrame(columns=metric_columns)
    if backtest_path is not None:
        backtest = pd.read_parquet(backtest_path)
        backtest = backtest[backtest.model.eq(backtest_model)].copy()
        if (backtest.empty or not backtest.split.eq('test').all()
                or not backtest.target_date.gt(backtest.base_date).all()
                or not backtest.target_date.le(date).all()
                or backtest.duplicated(['item', 'base_date', 'horizon']).any()):
            raise ValueError('Invalid retrospective predictions')
        check = backtest.merge(prices[['item', 'date', 'price_per_kg']], left_on=['item', 'target_date'],
                               right_on=['item', 'date'], how='left', validate='many_to_one')
        if not np.allclose(check.actual, check.price_per_kg, rtol=1e-12, atol=1e-12):
            raise ValueError('Backtest actual prices do not match the imported history')
        measured = score_predictions(backtest)
        dates = backtest.groupby(['item', 'horizon']).target_date.agg(period_start='min', period_end='max').reset_index()
        metrics = measured.merge(dates, on=['item', 'horizon'], validate='one_to_one')
        metrics = metrics.rename(columns={'direction_accuracy': 'direction_acc'}).assign(
            item_id=metrics.item.map(item_ids), coverage=None)[metric_columns]
        for run_id, (origin, _) in enumerate(backtest.groupby('base_date'), 2):
            runs.append({'id': run_id, 'base_date': origin, 'source': 'BACKTEST',
                         'model_version': backtest_model, 'created_at': created})
        ids = {run['base_date']: run['id'] for run in runs if run['source'] == 'BACKTEST'}
        history = backtest.rename(columns={'prediction': 'p50'}).assign(
            item_id=backtest.item.map(item_ids), run_id=backtest.base_date.map(ids), p10=None, p90=None)
        short = pd.concat([short, history], ignore_index=True)
    short['target_date'] = pd.to_datetime(short.target_date).dt.strftime('%Y-%m-%d')
    return {'item': table_items, 'price_daily': table_prices, 'forecast_run': pd.DataFrame(runs),
            'forecast_short': short[['run_id', 'item_id', 'horizon', 'target_date', 'p10', 'p50', 'p90', 'base_price', 'model']],
            'forecast_metric': metrics}


def main():
    parser = argparse.ArgumentParser(description='Export forecast CSV/JSON and price history as H2 seed tables.')
    parser.add_argument('--forecast-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'db/seed/current')
    parser.add_argument('--backtest', type=Path)
    parser.add_argument('--backtest-model', default='adaptive')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new seed directory')
    tables = seed_tables(args.forecast_dir, args.backtest, args.backtest_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as directory:
        stage = Path(directory) / 'seed'
        stage.mkdir()
        for name, frame in tables.items():
            frame.to_csv(stage / f'{name}.csv', index=False)
            print(f'{name}: {len(frame)} rows')
        stage.rename(args.output)
    print(f'Seed tables saved: {args.output}')


if __name__ == '__main__':
    main()
