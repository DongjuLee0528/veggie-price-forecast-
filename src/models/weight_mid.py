import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs
from src.models.train_mid import ITEMS, load_data
from src.models.tune_mid import KEYS, archived_comparison


def training_weights(train, scheme):
    if scheme == 'relative-error':
        return train.base_price / train.target_price
    counts = train.groupby(['item', 'horizon', 'target_end']).target_price.transform('size')
    weights = 1 / counts
    return weights / weights.mean()


def main():
    parser = argparse.ArgumentParser(description='Compare training weights for mid-term LightGBM.')
    parser.add_argument('--scheme', choices=['relative-error', 'equal-dekad'], default='relative-error')
    parser.add_argument('--dry-run', action='store_true', help='Verify the original forecasts and weights without training.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    default_output = 'mid_equal_dekad' if args.scheme == 'equal-dekad' else 'mid_relative_error'
    output = args.output or ROOT / 'results' / default_output
    if output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    _, _, frames = load_data()
    reference = ROOT / 'results/lgbm_mid_comparison'
    evaluation = pd.concat([frames['validation'], frames['test']], ignore_index=True)
    rows = archived_comparison(reference, evaluation)
    archive = pd.read_parquet(ROOT / 'results/mid_tuning/predictions.parquet')
    ets = archive[archive.model.eq('ets')][KEYS + ['prediction']].rename(columns={'prediction': 'ets'})
    rows = rows.merge(ets, on=KEYS, how='left', validate='one_to_one')
    if not np.isfinite(rows.ets).all() or not rows.ets.gt(0).all():
        raise ValueError('Holt–Winters forecasts must match the comparison dates')
    rows['current_blend'] = .5625 * rows.price_lgbm + .1875 * rows.sarimax + .25 * rows.ets
    params = dict(objective='regression_l1', learning_rate=.03, num_leaves=15,
                  min_child_samples=50, reg_lambda=1., n_jobs=2, random_state=42, verbosity=-1)
    columns = PRICE_FEATURES + ['horizon']
    models = {}
    for item in sorted(ITEMS):
        train = frames['train'][frames['train'].item.eq(item)]
        sample = rows[rows.item.eq(item)]
        original = lgb.Booster(model_file=str(reference / f'{item}_price_lgbm.txt'))
        inputs = model_inputs(sample, columns)
        np.testing.assert_allclose(sample.base_price * (1 + original.predict(inputs, num_threads=2)),
                                   sample.price_lgbm, rtol=1e-9, atol=1e-8)
        rounds = original.current_iteration()
        weights = training_weights(train, args.scheme)
        if not np.isfinite(weights).all() or not weights.gt(0).all():
            raise ValueError('Training weights must be positive and finite')
        print(f'{item}: {rounds} fixed rounds, {len(train)} training rows, '
              f'weight range {weights.min():.3f}–{weights.max():.3f}', flush=True)
        if args.dry_run:
            continue
        model = lgb.LGBMRegressor(n_estimators=rounds, **params)
        model.fit(model_inputs(train, columns), train.target_return, sample_weight=weights)
        rows.loc[sample.index, 'weighted_lgbm'] = sample.base_price * (1 + model.predict(inputs))
        models[item] = model.booster_
    if args.dry_run:
        return
    rows['weighted_blend'] = .5625 * rows.weighted_lgbm + .1875 * rows.sarimax + .25 * rows.ets
    methods = ['current_blend', 'weighted_blend', 'price_lgbm', 'weighted_lgbm']
    if not np.isfinite(rows[methods]).all().all() or not rows[methods].gt(0).all().all():
        raise ValueError('Expected complete positive predictions')
    predictions = rows.melt(id_vars=KEYS + ['actual', 'base_price'], value_vars=methods,
                            var_name='model', value_name='prediction')
    metrics = score_predictions(predictions)
    test = predictions[predictions.split.eq('test')]
    by_year = score_predictions(test.assign(split=test.base_date.dt.year.astype(str)))
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / 'predictions.parquet', index=False)
    metrics.to_csv(output / 'metrics.csv', index=False)
    by_year.to_csv(output / 'metrics_by_year.csv', index=False)
    for item, model in models.items():
        model.save_model(str(output / f'{item}_weighted_lgbm.txt'))
    print(metrics.groupby(['split', 'model'])[['nmae', 'direction_accuracy']].mean().to_string())
    print(by_year.groupby(['split', 'model'])[['nmae', 'direction_accuracy']].mean().to_string())
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
