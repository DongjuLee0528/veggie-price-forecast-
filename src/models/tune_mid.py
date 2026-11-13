import argparse
from itertools import product
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.features.targets import dekad_bounds
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs
from src.models.train_mid import ITEMS, load_data


KEYS = ['split', 'item', 'horizon', 'base_date', 'target_date']


def archived_comparison(directory, evaluation):
    archive = pd.read_parquet(directory / 'predictions.parquet')
    columns = KEYS + ['actual', 'base_price']
    archive = archive[archive.model.isin(['price_lgbm', 'sarimax'])].pivot(
        index=columns, columns='model', values='prediction').reset_index()
    rows = evaluation.rename(columns={'date': 'base_date', 'target_end': 'target_date'})
    rows = rows.merge(archive, on=KEYS, suffixes=('', '_archive'), validate='one_to_one')
    if (len(rows) != len(evaluation) or len(rows) != len(archive)
            or not np.allclose(rows.actual, rows.target_price)
            or not np.allclose(rows.base_price, rows.base_price_archive)
            or not np.isfinite(rows[['price_lgbm', 'sarimax']]).all().all()):
        raise ValueError('Archived forecasts do not match the current data')
    rows['fixed_blend'] = .75 * rows.price_lgbm + .25 * rows.sarimax
    return rows


def ets_predictions(rows, means, first_observation, *, trend="add", damped=True, lookback=None):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    first = int(means.index.min())
    if first_observation > dekad_bounds(first)[0]:
        first += 1
    means = means.loc[first:]
    if not means.index.equals(pd.Index(range(first, int(means.index.max()) + 1), name=means.index.name)):
        raise ValueError('Holt–Winters requires consecutive dekad observations')
    completed = rows.origin_dekad - [
        int(dekad_bounds(period)[1] > date) for period, date in zip(rows.origin_dekad, rows.base_date)
    ]
    result = pd.Series(index=rows.index, dtype=float)
    for period, indices in completed.groupby(completed).groups.items():
        history = means.loc[:period]
        if lookback is not None:
            history = history.tail(lookback)
        if len(history) < 72 or not history.gt(0).all():
            raise ValueError('Holt–Winters requires two years of positive prices')
        model = ExponentialSmoothing(np.log(history.to_numpy()), trend=trend, damped_trend=damped,
                                     seasonal='add', seasonal_periods=36, initialization_method='heuristic')
        fitted = model.fit()
        if not fitted.mle_retvals.success:
            fitted = model.fit(method='Powell')
            if not fitted.mle_retvals.success:
                raise ValueError(f'Holt–Winters did not converge for dekad {period}')
        steps = rows.loc[indices, 'target_dekad'].to_numpy() - period
        forecast = np.exp(fitted.forecast(int(steps.max())))
        result.loc[indices] = forecast[steps - 1]
    return result


def scores(rows, prediction, name):
    return score_predictions(rows.assign(prediction=prediction, model=name))


def tune_lgbm(train, rows):
    columns = PRICE_FEATURES + ['horizon']
    params = dict(objective='regression_l1', n_estimators=500, learning_rate=.03,
                  min_child_samples=50, reg_lambda=1., n_jobs=2, random_state=42, verbosity=-1)
    validation = rows[rows.split.eq('validation')]
    trials = []
    models = {}
    for window, leaves in product(['all', '2y'], [7, 15]):
        fitted = {}
        forecasts = pd.Series(index=validation.index, dtype=float)
        for item in sorted(ITEMS):
            subset = train[train.item.eq(item)]
            if window == '2y':
                subset = subset[subset.date.ge('2022-01-01')]
            valid = validation[validation.item.eq(item)]
            x_valid = model_inputs(valid, columns)
            model = lgb.LGBMRegressor(**params, num_leaves=leaves)
            model.fit(model_inputs(subset, columns), subset.target_return,
                      eval_set=[(x_valid, valid.target_return)],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
            forecasts.loc[valid.index] = valid.base_price * (1 + model.predict(x_valid))
            fitted[item] = model
        blend = .75 * forecasts + .25 * validation.sarimax
        metric = scores(validation, blend, 'tuned_blend')[['nmae', 'direction_accuracy']].mean()
        trials.append({'window': window, 'leaves': leaves, **metric.to_dict()})
        models[window, leaves] = fitted
        print(f'LightGBM {window}, leaves={leaves}: validation blend NMAE={metric.nmae:.4f}', flush=True)
    trials = pd.DataFrame(trials)
    chosen = trials.sort_values(['nmae', 'direction_accuracy', 'leaves'],
                                ascending=[True, False, True]).iloc[0]
    chosen_models = models[chosen.window, chosen.leaves]
    forecasts = pd.Series(index=rows.index, dtype=float)
    for item, model in chosen_models.items():
        sample = rows[rows.item.eq(item)]
        forecasts.loc[sample.index] = sample.base_price * (1 + model.predict(model_inputs(sample, columns)))
    return forecasts, trials, chosen, chosen_models


def main():
    parser = argparse.ArgumentParser(description='Tune mid-term LightGBM and compare Holt–Winters on identical dates.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reference', type=Path, default=ROOT / 'results/lgbm_mid_comparison')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/mid_tuning')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    prices, means, frames = load_data()
    evaluation = pd.concat([frames['validation'], frames['test']], ignore_index=True)
    rows = archived_comparison(args.reference, evaluation)
    if args.dry_run:
        print(f'Archived comparison matched: {len(rows)} rows')
        return

    tuned, trials, chosen, models = tune_lgbm(frames['train'], rows)
    rows['tuned_blend'] = .75 * tuned + .25 * rows.sarimax
    rows['ets'] = np.nan
    for item in sorted(ITEMS):
        sample = rows[rows.item.eq(item)]
        rows.loc[sample.index, 'ets'] = ets_predictions(
            sample, means.loc[item], prices.loc[prices.item.eq(item), 'date'].min())
        print(f'{item}: Holt–Winters complete', flush=True)
    valid = rows[rows.split.eq('validation')]
    candidates = []
    for weight in [0., .25, .5]:
        prediction = (1 - weight) * valid.fixed_blend + weight * valid.ets
        metric = scores(valid, prediction, 'blend_with_ets')[['nmae', 'direction_accuracy']].mean()
        candidates.append({'ets_weight': weight, **metric.to_dict()})
    ets_trials = pd.DataFrame(candidates)
    weight = ets_trials.sort_values(['nmae', 'direction_accuracy', 'ets_weight'],
                                    ascending=[True, False, True]).iloc[0].ets_weight
    rows['blend_with_ets'] = (1 - weight) * rows.fixed_blend + weight * rows.ets
    methods = ['fixed_blend', 'tuned_blend', 'ets', 'blend_with_ets']
    if not np.isfinite(rows[methods]).all().all() or not rows[methods].gt(0).all().all():
        raise ValueError('Expected complete positive prices for every method')
    predictions = rows.melt(id_vars=KEYS + ['actual', 'base_price'], value_vars=methods,
                            var_name='model', value_name='prediction')
    metrics = score_predictions(predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    trials.to_csv(args.output / 'lgbm_calibration.csv', index=False)
    ets_trials.to_csv(args.output / 'ets_calibration.csv', index=False)
    predictions.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    for item, model in models.items():
        model.booster_.save_model(str(args.output / f'{item}_tuned_lgbm.txt'))
    (args.output / 'selection.json').write_text(json.dumps({
        'window': chosen.window, 'leaves': int(chosen.leaves), 'ets_weight': float(weight),
        'sarimax_weight_in_fixed_blend': .25, 'selected_on': '2024 validation',
        'note': '2025–2026 test period previously inspected. ETS refits use completed dekads only.',
    }, indent=2) + '\n')
    print(metrics.groupby(['split', 'model'])[['nmae', 'direction_accuracy']].mean().to_string())
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
