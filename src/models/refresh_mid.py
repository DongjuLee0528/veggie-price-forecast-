import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.features.targets import dekad_index
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs
from src.models.train_mid import ITEMS, load_data
from src.models.tune_mid import KEYS, archived_comparison


WEIGHT_LGBM = .5625


def yearly_windows(history, year):
    cutoff = pd.Timestamp(year, 1, 1)
    prior = pd.Timestamp(year - 1, 1, 1)
    available = history[history.target_end.lt(cutoff)]
    training = available[available.target_end.lt(prior)]
    calibration = available[available.date.ge(prior)].copy()
    calibration['dekad'] = calibration.date.map(dekad_index)
    calibration = calibration.sort_values('date').drop_duplicates(['dekad', 'horizon'], keep='last')
    if training.empty or calibration.empty:
        raise ValueError(f'Incomplete training windows for {year}')
    return training, calibration, available


def main():
    parser = argparse.ArgumentParser(description='Compare annual LightGBM refits in the existing mid-term ensemble.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mid_annual_refit')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    _, _, frames = load_data()
    data = pd.concat([frames['validation'], frames['test']], ignore_index=True)
    rows = archived_comparison(ROOT / 'results/lgbm_mid_comparison', data)
    rows = rows[rows.split.eq('test')].copy()
    saved = pd.read_parquet(ROOT / 'results/mid_tuning/predictions.parquet')
    fixed = saved[saved.split.eq('test') & saved.model.eq('blend_with_ets')]
    rows = rows.merge(fixed[KEYS + ['actual', 'base_price', 'prediction']],
                      on=KEYS, suffixes=('', '_saved'), validate='one_to_one')
    if (len(rows) != len(frames['test']) or len(rows) != len(fixed)
            or not np.allclose(rows.actual, rows.actual_saved)
            or not np.allclose(rows.base_price, rows.base_price_saved)):
        raise ValueError('The saved ensemble does not match these test samples')
    history = pd.read_parquet(ROOT / 'data/processed/dataset.parquet')
    history = history[history.term.eq('mid') & history.item.isin(ITEMS)]
    params = dict(objective='regression_l1', learning_rate=.03, num_leaves=15,
                  min_child_samples=50, reg_lambda=1., n_jobs=2, random_state=42, verbosity=-1)
    columns = PRICE_FEATURES + ['horizon']
    details = []
    for (year, item), sample in rows.groupby([rows.base_date.dt.year, 'item']):
        train, valid, available = yearly_windows(history[history.item.eq(item)], year)
        print(f'{year} {item}: train={len(train)}, calibration={len(valid)}, '
              f'refit={len(available)}, test={len(sample)}', flush=True)
        if args.dry_run:
            continue
        search = lgb.LGBMRegressor(n_estimators=500, **params)
        search.fit(model_inputs(train, columns), train.target_return,
                   eval_set=[(model_inputs(valid, columns), valid.target_return)],
                   callbacks=[lgb.early_stopping(40, verbose=False)])
        rounds = search.best_iteration_
        model = lgb.LGBMRegressor(n_estimators=rounds, **params)
        model.fit(model_inputs(available, columns), available.target_return)
        price = sample.base_price * (1 + model.predict(model_inputs(sample, columns)))
        rows.loc[sample.index, 'refreshed'] = sample.prediction + WEIGHT_LGBM * (price - sample.price_lgbm)
        details.append({'year': year, 'item': item, 'rounds': rounds,
                        'last_training_target': str(available.target_end.max().date())})
    if args.dry_run:
        return
    if not np.isfinite(rows[['prediction', 'refreshed']]).all().all() or not rows[['prediction', 'refreshed']].gt(0).all().all():
        raise ValueError('Expected complete positive forecasts')
    comparison = pd.concat([
        rows.assign(model='current_blend'),
        rows.assign(model='annual_refit_blend', prediction=rows.refreshed),
    ], ignore_index=True)[KEYS + ['actual', 'base_price', 'model', 'prediction']]
    metrics = score_predictions(comparison)
    by_year = score_predictions(comparison.assign(split=comparison.base_date.dt.year.astype(str)))
    args.output.mkdir(parents=True, exist_ok=False)
    comparison.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    by_year.to_csv(args.output / 'metrics_by_year.csv', index=False)
    pd.DataFrame(details).to_csv(args.output / 'refits.csv', index=False)
    print(metrics.groupby('model')[['nmae', 'direction_accuracy']].mean().to_string())
    print(by_year.groupby(['split', 'model'])[['nmae', 'direction_accuracy']].mean().to_string())
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
