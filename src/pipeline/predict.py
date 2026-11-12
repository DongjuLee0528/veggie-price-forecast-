import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.config import load_items
from src.models.train_lgbm import model_inputs


def forecast(model_dir, features, prices, as_of):
    manifest = json.loads((model_dir / 'run.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or pd.Timestamp(manifest['cutoff']) > as_of:
        raise ValueError('Use completed models trained through or before the forecast origin')
    expected = {(item, horizon) for item in load_items()['items'] for horizon in [1, 5, 10]}
    routes = manifest['selected']
    if len(routes) != len(expected) or {(r['item'], r['horizon']) for r in routes} != expected:
        raise ValueError('Expected exactly fifteen unique item/horizon routes')
    current = features[features.date.eq(as_of)].copy()
    if current.item.duplicated().any() or set(current.item) != {item for item, _ in expected}:
        raise ValueError('Forecast origin must have one feature row for each item')
    history = prices[prices.date.le(as_of)]
    if history.duplicated(['item', 'date']).any():
        raise ValueError('Duplicate price history')
    models = {}
    records = []
    for route in routes:
        row = current[current.item.eq(route['item'])].copy()
        base = float(row.base_price.iloc[0])
        row['horizon'] = route['horizon']
        name = route['model']
        if name == 'last_value':
            observed = history[history.item.eq(route['item']) & history.date.eq(as_of)]
            if len(observed) != 1:
                raise ValueError('Missing origin price for last-value forecast')
            prediction = float(observed.price_per_kg.iloc[0])
        elif name == 'moving_average_7':
            prediction = base
        elif name in {'price_pooled', 'price_direct', 'full_pooled'}:
            filename = route['file']
            if filename not in models:
                detail, = [m for m in manifest['models'] if m['file'] == filename]
                path = model_dir / filename
                if hashlib.sha256(path.read_bytes()).hexdigest() != detail['sha256']:
                    raise ValueError(f'Model checksum mismatch: {filename}')
                models[filename] = lgb.Booster(model_file=str(path))
            returns = models[filename].predict(
                model_inputs(row, route['inputs']),
                num_iteration=route['best_iteration'], num_threads=2)
            prediction = base * (1 + float(returns[0]))
        else:
            raise ValueError(f'Unsupported model: {name}')
        if not np.isfinite([base, prediction]).all() or min(base, prediction) <= 0:
            raise ValueError('Forecast and base price must be finite and positive')
        records.append({'item': route['item'], 'base_date': str(as_of.date()),
                        'horizon': route['horizon'], 'horizon_unit': 'observed_trading_days',
                        'target_date': None, 'prediction_krw_per_kg': prediction,
                        'base_price_7day_krw_per_kg': base,
                        'change_from_base_pct': (prediction / base - 1) * 100,
                        'model': name})
    return pd.DataFrame(records).sort_values(['item', 'horizon']).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description='Generate short-term point forecasts from saved production models.')
    parser.add_argument('--model-dir', type=Path, default=ROOT / 'results/lgbm_production')
    parser.add_argument('--date', type=pd.Timestamp)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.model_dir / 'run.json').read_text(encoding='utf-8'))
    as_of = args.date if args.date is not None else pd.Timestamp(manifest['cutoff'])
    output = args.output if args.output is not None else ROOT / 'outputs' / f'forecast_{as_of.date()}'
    if output.exists():
        parser.error('Output exists; use a new output directory')
    features = pd.read_parquet(ROOT / 'data/processed/features.parquet')
    prices = pd.read_parquet(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    predictions = forecast(args.model_dir, features, prices, as_of)
    report = {'base_date': str(as_of.date()), 'trained_through': manifest['cutoff'],
              'status': 'complete', 'unit': 'KRW/kg',
              'limitations': ['Point forecasts only for horizons 1, 5, 10.',
                              'Future item-specific trading dates are unavailable; target dates are null.',
                              'Forecast origin uses available historical data; deployment refit accuracy is unmeasured.'],
              'forecasts': predictions.to_dict(orient='records')}
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_csv(output / 'predictions.csv', index=False)
    temporary = output / 'forecast.tmp.json'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output / 'forecast.json')
    print(predictions[['item', 'horizon', 'prediction_krw_per_kg', 'model']].to_string(index=False))
    print(f'Forecasts saved: {output}')


if __name__ == '__main__':
    main()
