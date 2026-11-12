import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.collect.price import load_prices
from src.evaluate.backtest import ROOT, score_predictions
from src.features.targets import dekad_bounds, dekad_index
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs, prediction_records, validate_split


ITEMS = {'배추', '무', '양파', '건고추'}
HORIZONS = {2, 3}


def load_data():
    prices = load_prices(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    prices = prices[prices.item.isin(ITEMS)].copy()
    prices['dekad'] = prices.date.map(dekad_index)
    means = prices.groupby(['item', 'dekad']).price_per_kg.mean()
    frames = {}
    for split in ['train', 'validation', 'test']:
        data = pd.read_parquet(ROOT / f'data/processed/{split}.parquet')
        data = data[data.term.eq('mid') & data.item.isin(ITEMS)].copy()
        validate_split(data, split)
        if set(data.item) != ITEMS or set(data.horizon) != HORIZONS:
            raise ValueError(f'Unexpected item or horizon coverage: {split}')
        data['origin_dekad'] = data.date.map(dekad_index)
        data['target_dekad'] = data.origin_dekad + data.horizon
        bounds = data.target_dekad.map(dekad_bounds)
        if not (bounds.map(lambda pair: pair[0]).eq(data.target_start).all()
                and bounds.map(lambda pair: pair[1]).eq(data.target_end).all()):
            raise ValueError(f'Dekad boundaries differ: {split}')
        actual = means.reindex(pd.MultiIndex.from_arrays([data.item, data.target_dekad]))
        if not np.allclose(actual, data.target_price):
            raise ValueError(f'Dekad target prices differ from daily history: {split}')
        if split != 'train':
            data = data.sort_values('date').drop_duplicates(
                ['item', 'origin_dekad', 'horizon'], keep='last')
        frames[split] = data
        print(f'{split}: {len(data)} rows, {data.date.min().date()} to {data.date.max().date()}',
              flush=True)
    return prices, means, frames


def seasonal_terms(periods):
    angle = 2 * np.pi * np.asarray(periods) / 36
    return np.column_stack([np.sin(angle), np.cos(angle)])


def series_predictions(rows, daily, means):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    first = int(means.index.min())
    if daily.date.min() > dekad_bounds(first)[0]:
        first += 1
    periods = np.arange(first, int(rows.target_dekad.max()) + 1)
    series = means.reindex(periods)
    log_price = np.log(series.to_numpy())
    last_train = dekad_index('2023-12-31')
    size = last_train - first + 1
    model = SARIMAX(log_price[:size], exog=seasonal_terms(periods[:size]),
                    order=(1, 0, 0), trend='c')
    fitted = model.fit(disp=False, maxiter=150)
    if not fitted.mle_retvals['converged']:
        fitted = model.fit(method='powell', disp=False, maxiter=200)
        if not fitted.mle_retvals['converged']:
            raise ValueError('Mid-term SARIMAX failed to converge')
    last = last_train
    records = []
    for row in rows.sort_values('date').itertuples():
        completed = row.origin_dekad
        if dekad_bounds(completed)[1] > row.date:
            completed -= 1
        if completed > last:
            available = np.arange(last + 1, completed + 1)
            fitted = fitted.extend(log_price[available - first], exog=seasonal_terms(available))
            last = completed
        future = np.arange(completed + 1, row.target_dekad + 1)
        log_prediction = fitted.get_forecast(steps=len(future), exog=seasonal_terms(future)).predicted_mean
        records.append((row.Index, np.exp(log_prediction[-1]),
                        series.loc[:completed].dropna().iloc[-1], series.get(row.target_dekad - 36)))
    return pd.DataFrame(records, columns=['row', 'sarimax', 'last_completed', 'year_ago']).set_index('row')


def main():
    parser = argparse.ArgumentParser(description='Compare five methods for 2- and 3-dekad forecasts.')
    parser.add_argument('--dry-run', action='store_true', help='Check target prices and splits without fitting.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/lgbm_mid_comparison')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    prices, means, frames = load_data()
    if args.dry_run:
        return

    manifest = json.loads((ROOT / 'data/processed/manifest.json').read_text())
    predictions = []
    boosters = {}
    evaluation = pd.concat([frames['validation'], frames['test']], ignore_index=True)
    params = dict(objective='regression_l1', n_estimators=500, learning_rate=.03,
                  num_leaves=15, min_child_samples=50, reg_lambda=1., n_jobs=2,
                  random_state=42, verbosity=-1)
    for item in sorted(ITEMS):
        train = frames['train'][frames['train'].item.eq(item)]
        rows = evaluation[evaluation.item.eq(item)]
        valid = rows[rows.split.eq('validation')]
        series = series_predictions(rows, prices[prices.item.eq(item)], means.loc[item])
        methods = series.reindex(rows.index).to_dict('series')
        for name, columns in [('price_lgbm', PRICE_FEATURES + ['horizon']),
                              ('full_lgbm', manifest['input_columns'])]:
            model = lgb.LGBMRegressor(**params)
            model.fit(model_inputs(train, columns), train.target_return,
                      eval_set=[(model_inputs(valid, columns), valid.target_return)],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
            values = rows.base_price.to_numpy() * (1 + model.predict(model_inputs(rows, columns)))
            methods[name] = values
            boosters[f'{item}_{name}.txt'] = model.booster_
            print(f'{item} {name}: best round {model.best_iteration_}', flush=True)
        for name, values in methods.items():
            records = prediction_records(rows, values, name)
            records['split'] = rows.split.to_numpy()
            predictions.append(records)
    predictions = pd.concat(predictions, ignore_index=True)
    if not predictions.prediction.gt(0).all():
        raise ValueError('Expected positive forecasts for all methods')
    keys = ['split', 'item', 'horizon', 'base_date', 'target_date']
    if (predictions.duplicated(keys + ['model']).any()
            or not predictions.groupby(keys).model.nunique().eq(5).all()):
        raise ValueError('Forecast methods must be compared on identical dates')
    metrics = score_predictions(predictions)
    winners = metrics[metrics.split.eq('validation')].sort_values(
        ['nmae', 'direction_accuracy', 'model'], ascending=[True, False, True]
    ).drop_duplicates(['item', 'horizon'])[['item', 'horizon', 'model']]
    selected = predictions[predictions.split.eq('test')].merge(
        winners, on=['item', 'horizon', 'model'], validate='many_to_one')
    selected['model'] = 'validation_selected'
    metrics = pd.concat([metrics, score_predictions(selected)], ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=False)
    for file, booster in boosters.items():
        booster.save_model(str(args.output / file))
    predictions.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    winners.to_csv(args.output / 'validation_selection.csv', index=False)
    print(metrics.groupby(['split', 'model'])[['nmae', 'direction_accuracy']].mean().to_string())
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
