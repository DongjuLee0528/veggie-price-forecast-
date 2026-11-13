import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs
from src.models.train_mid import load_data
from src.models.tune_mid import KEYS, archived_comparison


def main():
    parser = argparse.ArgumentParser(description='Compare separate 2- and 3-dekad LightGBM models in the mid-term ensemble.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mid_direct_horizons')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    _, _, frames = load_data()
    evaluation = pd.concat([frames['validation'], frames['test']], ignore_index=True)
    rows = archived_comparison(ROOT / 'results/lgbm_mid_comparison', evaluation)
    archive = pd.read_parquet(ROOT / 'results/mid_tuning/predictions.parquet')
    ets = archive[archive.model.eq('ets')][KEYS + ['prediction']].rename(columns={'prediction': 'ets'})
    rows = rows.merge(ets, on=KEYS, how='left', validate='one_to_one')
    if not np.isfinite(rows.ets).all() or not rows.ets.gt(0).all():
        raise ValueError('Holt–Winters forecasts must match the comparison dates')
    rows['current_blend'] = .5625 * rows.price_lgbm + .1875 * rows.sarimax + .25 * rows.ets
    params = dict(objective='regression_l1', n_estimators=500, learning_rate=.03, num_leaves=15,
                  min_child_samples=50, reg_lambda=1., n_jobs=2, random_state=42, verbosity=-1)
    models = {}
    for (item, horizon), sample in rows.groupby(['item', 'horizon']):
        train = frames['train'].query('item == @item and horizon == @horizon')
        valid = sample[sample.split.eq('validation')]
        if train.empty or valid.empty or not sample.split.eq('test').any():
            raise ValueError(f'Incomplete splits for {item}, horizon {horizon}')
        print(f'{item} horizon={horizon}: train={len(train)}, validation={len(valid)}, '
              f'test={sample.split.eq("test").sum()}', flush=True)
        if args.dry_run:
            continue
        model = lgb.LGBMRegressor(**params)
        model.fit(model_inputs(train, PRICE_FEATURES), train.target_return,
                  eval_set=[(model_inputs(valid, PRICE_FEATURES), valid.target_return)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        print(f'  best round={model.best_iteration_}', flush=True)
        rows.loc[sample.index, 'direct_lgbm'] = sample.base_price * (
            1 + model.predict(model_inputs(sample, PRICE_FEATURES)))
        models[item, horizon] = model.booster_
    if args.dry_run:
        return
    rows['direct_blend'] = .5625 * rows.direct_lgbm + .1875 * rows.sarimax + .25 * rows.ets
    methods = ['current_blend', 'direct_blend', 'price_lgbm', 'direct_lgbm']
    if not np.isfinite(rows[methods]).all().all() or not rows[methods].gt(0).all().all():
        raise ValueError('Expected complete positive forecasts')
    predictions = rows.melt(id_vars=KEYS + ['actual', 'base_price'], value_vars=methods,
                            var_name='model', value_name='prediction')
    metrics = score_predictions(predictions)
    test = predictions[predictions.split.eq('test')]
    by_year = score_predictions(test.assign(split=test.base_date.dt.year.astype(str)))
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    by_year.to_csv(args.output / 'metrics_by_year.csv', index=False)
    for (item, horizon), booster in models.items():
        booster.save_model(str(args.output / f'{item}_h{horizon}_lgbm.txt'))
    print(metrics.groupby(['split', 'model'])[['nmae', 'direction_accuracy']].mean().to_string())
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
