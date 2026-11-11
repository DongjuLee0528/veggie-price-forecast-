import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.features.targets import build_targets
from src.models.train_lgbm import model_inputs, progress_callback


BASELINES = {'last_value', 'moving_average_7'}


def freeze_selection(run_dir):
    run = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    metrics = pd.read_csv(run_dir / 'metrics_by_item_horizon.csv')
    candidates = BASELINES | {'full_pooled', 'price_pooled', 'price_direct'}
    metrics = metrics[metrics.model.isin(candidates)].copy()
    keys = ['item', 'horizon']
    if (metrics.empty or not metrics.split.eq('validation').all()
            or not metrics.coverage.eq(1).all()
            or not np.isfinite(metrics.nmae).all()
            or metrics.duplicated(keys + ['model']).any()):
        raise ValueError('Expected complete, unique validation metrics')
    groups = metrics.groupby(keys)
    if (len(groups) != 15 or set(metrics.horizon) != {1, 5, 10}
            or not groups.model.agg(set).map(lambda names: names == candidates).all()):
        raise ValueError('Expected five items, three horizons and five candidates')
    winners = metrics.sort_values(['nmae', 'model']).drop_duplicates(keys).sort_values(keys)
    selected = []
    for row in winners.itertuples():
        entry = {'item': row.item, 'horizon': row.horizon, 'model': row.model,
                 'validation_nmae': row.nmae}
        if row.model not in BASELINES:
            horizon = row.horizon if row.model == 'price_direct' else None
            detail, = [m for m in run['models'] if m['item'] == row.item
                       and m['experiment'] == row.model and m['horizon'] == horizon]
            if not isinstance(detail['best_iteration'], int) or detail['best_iteration'] < 1:
                raise ValueError('Invalid fixed iteration count')
            entry.update(detail)
            entry['horizon'] = row.horizon
        selected.append(entry)
    return run, selected


def prepare_data(cutoff):
    prices = pd.read_parquet(ROOT / 'data/interim/integrated/daily.parquet')
    prices = prices[prices.date.le(cutoff)].copy()
    trial = pd.read_parquet(ROOT / 'data/processed/experiment_v2/price_history.parquet')
    historical = prices.date.lt('2025-01-01')
    replacement = prices.loc[historical, ['date', 'item']].merge(
        trial, on=['date', 'item'], validate='one_to_one', how='left')
    if replacement.price_per_kg.isna().any():
        raise ValueError('Incomplete experimental price history')
    prices.loc[historical, 'price_per_kg'] = replacement.price_per_kg.to_numpy()
    old = pd.read_parquet(ROOT / 'data/processed/experiment_v2/features.parquet')
    recent = pd.read_parquet(ROOT / 'data/processed/features.parquet')
    features = pd.concat([old, recent[recent.date.ge('2025-01-01')]], ignore_index=True)
    features = features[features.date.le(cutoff)]
    if features.duplicated(['date', 'item']).any():
        raise ValueError('Duplicate feature rows')
    config = {'horizons_short': list(range(1, 11)), 'horizons_mid': []}
    targets = build_targets(prices, features, config)
    data = targets.merge(features, on=['date', 'item'], validate='many_to_one')
    data = data[data.base_price.notna() & data.target_return.notna()].copy()
    if (data.empty or not data.target_end.le(cutoff).all()
            or not data.target_start.gt(data.date).all()
            or not data.base_price.gt(0).all() or not data.target_price.gt(0).all()
            or not np.isfinite(data.target_return).all()
            or data.duplicated(['date', 'item', 'horizon']).any()):
        raise ValueError('Invalid production training data')
    return data


def main():
    parser = argparse.ArgumentParser(description='Refit validation-selected short-term methods using fixed iterations.')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'results/lgbm_experiment_v2')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/lgbm_production')
    parser.add_argument('--cutoff', type=pd.Timestamp, default=pd.Timestamp('2026-09-30'))
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error('Checkpoint interval must be positive')
    if args.output.exists() and not args.dry_run:
        parser.error('Output exists; use a new output directory')
    run, selected = freeze_selection(args.run_dir)
    data = prepare_data(args.cutoff)
    tasks = {}
    for entry in selected:
        group = data[data.item.eq(entry['item'])]
        if group.empty or group.target_end.max() != args.cutoff:
            raise ValueError('Cutoff must have observed targets for every selected item')
        if entry['model'] in BASELINES:
            continue
        if entry['model'] == 'price_direct':
            group = group[group.horizon.eq(entry['horizon'])]
        inputs = model_inputs(group, entry['inputs'])
        tasks[entry['file']] = (entry, group, inputs)
    print(f'Prepared {len(data)} rows, {len(tasks)} models and {len(selected)} routes; target cutoff={args.cutoff.date()}', flush=True)
    for name, (entry, group, _) in tasks.items():
        print(f'{name}: rows={len(group)}, rounds={entry["best_iteration"]}', flush=True)
    if args.dry_run:
        print('Dry run complete. No models fitted or outputs written.')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'training', 'selection_source': '2024 validation only',
                'cutoff': str(args.cutoff.date()), 'selected': selected, 'models': [],
                'note': 'Deployment refit uses previously evaluated data. No new accuracy estimate. Only horizons 1, 5, 10 are routed. Two experimental 2020 exclusions retained.'}
    (args.output / 'frozen_selection.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for number, (name, (entry, group, inputs)) in enumerate(tasks.items(), 1):
        rounds = entry['best_iteration']
        params = {**run['params'], 'n_estimators': rounds}
        model = lgb.LGBMRegressor(**params)
        print(f'[{number}/{len(tasks)}] {entry["item"]} {entry["model"]}', flush=True)
        model.fit(inputs, group.target_return, callbacks=[progress_callback(
            args.output, Path(name).stem, number, len(tasks), rounds, args.checkpoint_every)])
        destination = args.output / name
        temporary = destination.with_suffix('.tmp')
        model.booster_.save_model(str(temporary))
        temporary.replace(destination)
        manifest['models'].append({'file': name, 'rows': len(group), 'iterations': rounds,
                                   'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
        state = {'status': 'model_completed', 'completed_models': number, 'model_count': len(tasks)}
        (args.output / 'progress.json').write_text(json.dumps(state, indent=2) + '\n')
    manifest['status'] = 'complete'
    (args.output / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    state['status'] = 'complete'
    (args.output / 'progress.json').write_text(json.dumps(state, indent=2) + '\n')
    print(f'Production refit complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
