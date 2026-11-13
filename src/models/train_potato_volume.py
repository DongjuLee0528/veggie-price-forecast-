import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.models.train_lgbm import model_inputs, prediction_records, validate_split


def main():
    parser = argparse.ArgumentParser(description='Compare potato five-day forecasts with trading activity inputs.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/potato_sumi_volume')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')

    data = ROOT / 'data/processed/potato_sumi'
    columns = json.loads((data / 'manifest.json').read_text())['input_columns']
    daily = pd.read_parquet(data / 'prices.parquet').sort_values('date').set_index('date')
    activity = daily[['volume_kg', 'n_rows']]
    if not daily.index.is_unique or not np.isfinite(activity).all().all() or not activity.gt(0).all().all():
        raise ValueError('Expected unique trading days and positive finite activity')
    ratios = (activity / activity.shift().rolling(7).mean()).add_suffix('_ratio_prev7')

    frames = {}
    for split in ['train', 'validation', 'test']:
        frame = pd.read_parquet(data / f'{split}.parquet').query("term == 'short'").copy()
        frame = frame.join(ratios, on='date', validate='many_to_one')
        validate_split(frame, split)
        frames[split] = frame
    train = frames['train'][frames['train'].date.ge('2022-01-01')]
    valid = frames['validation']
    rows = pd.concat([valid, frames['test']], ignore_index=True)
    five = rows[rows.horizon.eq(5)].copy()
    if not np.isfinite(rows[ratios.columns]).all().all() or not np.isfinite(train[ratios.columns]).all().all():
        raise ValueError('Activity features are missing from the comparison data')

    archive = pd.read_parquet(ROOT / 'results/potato_sumi_tuning/predictions.parquet')
    saved = archive.query("term == 'short' and horizon == 5 and model == 'lgbm_2y_7leaves_25child_last_0.25'")
    keys = ['split', 'item', 'horizon', 'base_date', 'target_date']
    expected = prediction_records(five, five.base_price, 'alignment').assign(split=five.split.to_numpy())
    saved = saved.set_index(keys)
    aligned = saved.reindex(pd.MultiIndex.from_frame(expected[keys]))
    if len(saved) != len(five) or aligned.prediction.isna().any():
        raise ValueError('Saved five-day forecasts do not match comparison dates')
    np.testing.assert_allclose(aligned.actual, expected.actual)
    np.testing.assert_allclose(aligned.base_price, expected.base_price)
    print(f'Training: {len(train)} rows; five-day validation: {five.split.eq("validation").sum()}, '
          f'evaluation: {five.split.eq("test").sum()}', flush=True)
    if args.dry_run:
        return

    last = five.date.map(daily.price_per_kg).to_numpy()
    params = dict(objective='regression_l1', learning_rate=.03, num_leaves=7, min_child_samples=25,
                  reg_lambda=1., n_jobs=2, random_state=42, verbosity=-1)
    original = lgb.LGBMRegressor(n_estimators=500, **params)
    original.fit(model_inputs(train, columns), train.target_return,
                 eval_set=[(model_inputs(valid, columns), valid.target_return)],
                 callbacks=[lgb.early_stopping(40, verbose=False)])
    original_price = five.base_price.to_numpy() * (1 + original.predict(model_inputs(five, columns)))
    np.testing.assert_allclose(.75 * original_price + .25 * last, aligned.prediction,
                               rtol=1e-9, atol=1e-8)
    rounds = original.best_iteration_
    activity_columns = columns + list(ratios.columns)
    model = lgb.LGBMRegressor(n_estimators=rounds, **params)
    model.fit(model_inputs(train, activity_columns), train.target_return)
    forecast = five.base_price.to_numpy() * (1 + model.predict(model_inputs(five, activity_columns)))
    methods = {'current_blend': aligned.prediction.to_numpy(),
               'activity_blend': .75 * forecast + .25 * last, 'last_value': last}
    predictions = pd.concat([
        prediction_records(five, values, name).assign(split=five.split.to_numpy())
        for name, values in methods.items()
    ], ignore_index=True)
    if not predictions.prediction.gt(0).all():
        raise ValueError('Expected positive price predictions')
    metrics = score_predictions(predictions)
    test = predictions[predictions.split.eq('test')]
    yearly = score_predictions(test.assign(split=test.base_date.dt.year.astype(str)))
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    yearly.to_csv(args.output / 'metrics_by_year.csv', index=False)
    model.booster_.save_model(str(args.output / 'activity_lgbm.txt'))
    print(f'Original model reproduced; both models use {rounds} rounds')
    print(metrics[['split', 'model', 'n_scored', 'nmae', 'direction_accuracy']].to_string(index=False))
    print(yearly[['split', 'model', 'n_scored', 'nmae', 'direction_accuracy']].to_string(index=False))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
