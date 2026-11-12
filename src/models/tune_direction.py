import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.evaluate.backtest import score_predictions
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs, prediction_records
from src.models.train_production import prepare_data


def scores(group, returns):
    actual = group.target_price.to_numpy()
    prediction = group.base_price.to_numpy() * (1 + returns)
    return {'nmae': float(np.mean(np.abs(prediction - actual) / actual)),
            'direction_accuracy': float(np.mean(np.sign(returns) == np.sign(group.target_return)))}


def fit_candidate(train, valid, columns, config, early_stopping=True):
    params = {'n_estimators': config.get('rounds', 500), 'learning_rate': .03,
              'num_leaves': config['leaves'], 'min_child_samples': config['min_samples'],
              'reg_lambda': 5., 'n_jobs': 2, 'random_state': 42, 'verbosity': -1}
    x = model_inputs(train, columns)
    vx = model_inputs(valid, columns)
    for column in ['item', 'item_type']:
        if column in x:
            vx[column] = vx[column].cat.set_categories(x[column].cat.categories)
    callbacks = [lgb.early_stopping(40, verbose=False)] if early_stopping else []
    target = train.target_return
    vtarget = valid.target_return
    if config['kind'] == 'direction':
        estimator = lgb.LGBMClassifier(objective='binary', **params)
        target = target.gt(0).astype(int)
        vtarget = vtarget.gt(0).astype(int)
        fit_args = {'eval_set': [(vx, vtarget)], 'eval_metric': 'binary_logloss'} if early_stopping else {}
    else:
        estimator = lgb.LGBMRegressor(objective=config['kind'], **params)
        fit_args = {'eval_set': [(vx, vtarget)], 'eval_metric': 'l1'} if early_stopping else {}
    estimator.fit(x, target, callbacks=callbacks, **fit_args)
    if config['kind'] == 'direction':
        probability = estimator.predict_proba(vx)[:, 1]
        magnitude = train.target_return.abs().median()
        prediction = np.where(probability > config.get('threshold', .5), magnitude, -magnitude)
    else:
        probability = None
        magnitude = None
        prediction = estimator.predict(vx)
    rounds = estimator.best_iteration_ if early_stopping else params['n_estimators']
    return estimator, np.asarray(prediction), probability, magnitude, rounds


def main():
    parser = argparse.ArgumentParser(description='Tune on 2024 only, evaluate yearly refits retrospectively.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/direction_tuning')
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    data = pd.read_parquet(args.data) if args.data else prepare_data(pd.Timestamp('2026-09-30'))
    data = data[data.horizon.isin([1, 5, 10])].copy()
    prices = pd.read_parquet(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    origin = prices.set_index(['item', 'date']).price_per_kg
    data['origin_price'] = origin.reindex(pd.MultiIndex.from_frame(data[['item', 'date']])).to_numpy()
    if data.origin_price.isna().any():
        raise ValueError('Missing last-value baseline prices')
    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    candidates = []
    selected = []
    groups = list(data.groupby(['item', 'horizon']))
    for number, ((item, horizon), group) in enumerate(groups, 1):
        valid = group[group.date.dt.year.eq(2024) & group.target_end.lt('2025-01-01')]
        if valid.empty:
            raise ValueError('Missing validation data')
        best = None
        for baseline in ['last_value', 'moving_average_7']:
            prediction = valid.origin_price / valid.base_price - 1 if baseline == 'last_value' else np.zeros(len(valid))
            candidate = {'item': item, 'horizon': int(horizon), 'kind': baseline, **scores(valid, prediction)}
            candidates.append(candidate)
            if best is None or (-candidate['direction_accuracy'], candidate['nmae']) < (-best['direction_accuracy'], best['nmae']):
                best = candidate
        for years in [2, 4]:
            train = group[group.date.ge(f'{2024-years}-01-01') & group.target_end.lt('2024-01-01')]
            for feature_set, columns in [('price', PRICE_FEATURES), ('full', [c for c in manifest['input_columns'] if c != 'horizon'])]:
                for kind in ['regression_l1', 'regression', 'direction']:
                    for leaves, min_samples in [(7, 50), (15, 30)]:
                        config = {'kind': kind, 'leaves': leaves, 'min_samples': min_samples}
                        _, prediction, probability, magnitude, rounds = fit_candidate(train, valid, columns, config)
                        thresholds = [.4, .5, .6] if probability is not None else [None]
                        for threshold in thresholds:
                            if probability is not None:
                                prediction = np.where(probability > threshold, magnitude, -magnitude)
                            candidate = {'item': item, 'horizon': int(horizon), **config, 'years': years,
                                         'feature_set': feature_set, 'inputs': columns, 'rounds': int(rounds),
                                         'threshold': threshold, **scores(valid, prediction)}
                            candidates.append(candidate)
                            if (-candidate['direction_accuracy'], candidate['nmae']) < (-best['direction_accuracy'], best['nmae']):
                                best = candidate
        selected.append(best)
        pd.DataFrame([{k: v for k, v in r.items() if k != 'inputs'} for r in candidates]).to_csv(args.output / 'validation_candidates.csv', index=False)
        (args.output / 'selection.json').write_text(json.dumps({'selection_source': '2024 validation only',
            'objective': 'maximum direction accuracy; ties use minimum percentage error', 'selected': selected}, ensure_ascii=False, indent=2)+'\n')
        print(f'[{number}/{len(groups)}] {item} h={horizon}: {best["kind"]}, validation direction={best["direction_accuracy"]:.4f}, error={best["nmae"]:.4f}', flush=True)
    records = []
    for choice in selected:
        group = data[data.item.eq(choice['item']) & data.horizon.eq(choice['horizon'])]
        for year in [2025, 2026]:
            valid = group[group.date.dt.year.eq(year) & group.target_end.lt(f'{year+1}-01-01')]
            if choice['kind'] in {'last_value', 'moving_average_7'}:
                prediction = valid.origin_price.to_numpy() if choice['kind'] == 'last_value' else valid.base_price.to_numpy()
            else:
                train = group[group.date.ge(f'{year-choice["years"]}-01-01') & group.target_end.lt(f'{year}-01-01')]
                _, returns, _, _, _ = fit_candidate(train, valid, choice['inputs'], choice, early_stopping=False)
                prediction = valid.base_price.to_numpy() * (1 + returns)
            records.append(prediction_records(valid, prediction, 'direction_tuned'))
    records = pd.concat(records, ignore_index=True).assign(split='test')
    records.to_parquet(args.output / 'test_predictions.parquet', index=False)
    metrics = score_predictions(records)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    print('RETROSPECTIVE RESULTS', flush=True)
    print(metrics.groupby('item')[['nmae', 'direction_accuracy']].mean().to_string(), flush=True)
    print(metrics[['nmae', 'direction_accuracy']].mean().to_string(), flush=True)
    (args.output / 'evaluation.json').write_text(json.dumps({'status': 'complete', 'note':
        'Tuning uses 2024 only. Yearly refits include only labels observed before each evaluation year. 2025-2026 was previously inspected; retrospective evaluation only. Direction is relative to the origin seven-trading-day average. Price error must be reported alongside direction accuracy.'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
