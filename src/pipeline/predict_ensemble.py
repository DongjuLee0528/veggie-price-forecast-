import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.ensemble.adaptive import METHODS
from src.pipeline.predict import forecast


def apply_blends(predictions, choices, prices):
    expected = set(zip(predictions.item, predictions.horizon))
    if len(choices) != len(expected) or {(c['item'], c['horizon']) for c in choices} != expected:
        raise ValueError('Blend routes differ from forecast routes')
    result = predictions.copy()
    for choice in choices:
        mask = result.item.eq(choice['item']) & result.horizon.eq(choice['horizon'])
        row = result.loc[mask].iloc[0]
        if row.model != choice['model'] or not 0 <= choice['model_weight'] <= 1:
            raise ValueError('Blend must reference the frozen production method')
        if choice['model_weight'] == 1:
            continue
        if choice['baseline'] == 'last_value':
            history = prices[prices.item.eq(row['item']) & prices.date.eq(row.base_date)]
            if len(history) != 1:
                raise ValueError('Missing origin price')
            baseline = history.price_per_kg.iloc[0]
        elif choice['baseline'] == 'moving_average_7':
            baseline = row.base_price_7day_krw_per_kg
        else:
            raise ValueError('Unknown blend baseline')
        value = choice['model_weight'] * row.prediction_krw_per_kg + (1 - choice['model_weight']) * baseline
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid blended price')
        result.loc[mask, 'prediction_krw_per_kg'] = value
        result.loc[mask, 'change_from_base_pct'] = (value / row.base_price_7day_krw_per_kg - 1) * 100
        result.loc[mask, 'model'] = f'{row.model}:{choice["model_weight"]}+{choice["baseline"]}'
    return result


def apply_adaptive(blended, original, prices, snapshot, as_of):
    if snapshot['base_date'] != str(as_of.date()) or snapshot['selection_source'] != '2024 validation only':
        raise ValueError('Use adaptive weights for this forecast origin')
    selected = snapshot['selected']
    routes = {(row['item'], row['horizon']): row['weights'] for row in selected}
    if len(selected) != len(routes) or set(routes) != set(zip(blended.item, blended.horizon)):
        raise ValueError('Adaptive routes differ from forecast routes')
    result = blended.copy()
    current = prices[prices.date.eq(as_of)].set_index('item').price_per_kg
    original_prices = original.set_index(['item', 'horizon']).prediction_krw_per_kg
    for row in blended.itertuples():
        weights = routes[row.item, row.horizon]
        if set(weights) != set(METHODS) or min(weights.values()) < 0 or not np.isclose(sum(weights.values()), 1):
            raise ValueError('Invalid adaptive weight snapshot')
        candidates = [row.prediction_krw_per_kg, original_prices.loc[row.item, row.horizon],
                      current.loc[row.item], row.base_price_7day_krw_per_kg]
        price = float(np.dot([weights[method] for method in METHODS], candidates))
        if not np.isfinite(price) or price <= 0:
            raise ValueError('Invalid adaptive price')
        result.loc[row.Index, 'prediction_krw_per_kg'] = price
        result.loc[row.Index, 'change_from_base_pct'] = (price / row.base_price_7day_krw_per_kg - 1) * 100
        result.loc[row.Index, 'model'] = 'adaptive'
    return result


def main():
    parser = argparse.ArgumentParser(description='Apply validation-only blend weights to saved production forecasts.')
    parser.add_argument('--weights', type=Path, default=ROOT / 'results/lgbm_blend/blends.json')
    parser.add_argument('--adaptive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    manifest = json.loads((ROOT / 'results/lgbm_production/run.json').read_text())
    weights = json.loads(args.weights.read_text())
    if weights['selection_source'] != '2024 validation only':
        raise ValueError('Unexpected blend selection source')
    as_of = pd.Timestamp(manifest['cutoff'])
    prices = pd.read_parquet(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    original = forecast(ROOT / 'results/lgbm_production', pd.read_parquet(ROOT / 'data/processed/features.parquet'), prices, as_of)
    predicted = apply_blends(original, weights['selected'], prices)
    if args.adaptive is not None:
        snapshot = json.loads(args.adaptive.read_text())
        predicted = apply_adaptive(predicted, original, prices, snapshot, as_of)
    report = {'base_date': str(as_of.date()), 'trained_through': manifest['cutoff'], 'status': 'complete',
              'unit': 'KRW/kg', 'limitations': ['Historical forecast origin; future trading dates and intervals unavailable.',
              'Ensemble settings selected on 2024 validation; refit accuracy has no new holdout evaluation.'],
              'forecasts': predicted.to_dict(orient='records')}
    args.output.mkdir(parents=True, exist_ok=False)
    predicted.to_csv(args.output / 'predictions.csv', index=False)
    (args.output / 'forecast.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(predicted[['item', 'horizon', 'prediction_krw_per_kg']].to_string(index=False))
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
