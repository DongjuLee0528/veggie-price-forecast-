import argparse
import json
from pathlib import Path

import pandas as pd

from src.collect.asos import ROOT
from src.ensemble.adaptive import METHODS, apply_recent_weights, recent_weights, select_recent_config
from src.evaluate.backtest import score_predictions
from src.models.train_production import freeze_selection


KEYS = ['item', 'horizon', 'base_date', 'target_date']


def align_predictions(frame):
    prices = frame.drop_duplicates(KEYS).set_index(KEYS)[['actual', 'base_price']]
    pivot = frame.pivot(index=KEYS, columns='model', values='prediction')
    return prices.join(pivot, validate='one_to_one').sort_index().reset_index()


def main():
    parser = argparse.ArgumentParser(description='Select rolling weights on 2024, replay known outcomes, and snapshot weights for the latest origin.')
    parser.add_argument('--blend-dir', type=Path, default=ROOT / 'results/lgbm_blend')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/lgbm_adaptive')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    run_dir = ROOT / 'results/lgbm_experiment_v2'
    _, selected = freeze_selection(run_dir)
    blend = json.loads((args.blend_dir / 'blends.json').read_text())
    choices = {(entry['item'], entry['horizon']): entry for entry in blend['selected']}
    validation = pd.read_parquet(run_dir / 'validation_predictions.parquet')
    if (not validation.split.eq('validation').all() or not validation.base_date.dt.year.eq(2024).all()
            or not validation.target_date.lt('2025-01-01').all()):
        raise ValueError('Adaptive selection requires 2024 validation only')
    before = {}
    configs = {}
    for route in selected:
        item, horizon = key = route['item'], route['horizon']
        rows = align_predictions(validation[validation.item.eq(item) & validation.horizon.eq(horizon)])
        choice = choices[key]
        weight = choice['model_weight']
        rows['validation_selected'] = rows[route['model']]
        rows['validation_blend'] = weight * rows.validation_selected + (1 - weight) * rows[choice['baseline']]
        before[key] = rows
        configs[key] = select_recent_config(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = [{'item': item, 'horizon': horizon, 'config': config} for (item, horizon), config in configs.items()]
    (args.output / 'selection.json').write_text(json.dumps({'source': '2024 validation only', 'selected': frozen},
                                                           ensure_ascii=False, indent=2)+'\n')
    test = pd.read_parquet(args.blend_dir / 'test_predictions.parquet')
    if not test.split.eq('test').all() or not test.base_date.ge('2025-01-01').all():
        raise ValueError('Expected retrospective samples from 2025 onward')
    cutoff = test.target_date.max()
    predictions = []
    live = []
    for (item, horizon), config in configs.items():
        after = align_predictions(test[test.item.eq(item) & test.horizon.eq(horizon)])
        rows = pd.concat([before[item, horizon], after], ignore_index=True)
        after['prediction'] = apply_recent_weights(rows, config)[-len(after):]
        after['model'] = 'adaptive'
        after['split'] = 'test'
        predictions.append(after[[*KEYS, 'actual', 'base_price', 'prediction', 'model', 'split']])
        weights = recent_weights(rows, [cutoff], **config)[0].tolist() if config else [1., 0., 0., 0.]
        live.append({'item': item, 'horizon': horizon, 'weights': dict(zip(METHODS, weights))})
    adaptive = pd.concat(predictions, ignore_index=True)
    combined = pd.concat([test, adaptive], ignore_index=True)
    if (combined.duplicated(KEYS + ['model']).any() or not combined.groupby(KEYS).model.nunique().eq(6).all()
            or not combined.prediction.gt(0).all()):
        raise ValueError('Adaptive and fixed methods must have identical complete samples')
    metrics = score_predictions(combined)
    combined.to_parquet(args.output / 'test_predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    (args.output / 'live_weights.json').write_text(json.dumps({
        'base_date': str(cutoff.date()), 'selection_source': '2024 validation only', 'selected': live,
        'note': 'Weight scores use targets already observed by the forecast origin. Retrospective period has been inspected.'},
        ensure_ascii=False, indent=2)+'\n')
    print(metrics.groupby('model')[['nmae', 'direction_accuracy']].mean().to_string())


if __name__ == '__main__':
    main()
