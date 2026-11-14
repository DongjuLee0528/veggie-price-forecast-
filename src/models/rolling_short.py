import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate.backtest import ROOT, score_predictions
from src.models.compare_all_items import ITEMS, KEYS, PARAMS
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import prediction_records


HORIZONS = [5, 10]
METHODS = ['last_value', 'monthly_lgbm', 'monthly_mlp', 'monthly_blend']


def prepare(item, reference):
    processed = ROOT / 'data/processed'
    if item == '감자 수미':
        directory = processed / 'potato_sumi'
        prices = pd.read_parquet(directory / 'prices.parquet')
        history = pd.concat([pd.read_parquet(directory / f'{split}.parquet')
                             for split in ['train', 'validation', 'test']], ignore_index=True)
    else:
        prices = pd.read_parquet(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
        history = pd.read_parquet(processed / 'dataset.parquet')
    prices = prices[prices.item.eq(item)].set_index('date').sort_index()
    history = history[history.item.eq(item) & history.term.eq('short')
                      & history.horizon.isin(HORIZONS)].copy()
    activity = prices[['volume_kg', 'n_rows']]
    ratios = (activity / activity.shift().rolling(7).mean()).add_suffix('_ratio_prev7')
    history = history.join(ratios, on='date', validate='many_to_one')
    history['last_value'] = history.date.map(prices.price_per_kg)
    if (history.duplicated(['date', 'horizon']).any()
            or not history.target_end.gt(history.date).all()
            or not np.isfinite(history[['last_value', 'target_return', 'base_price']]).all().all()):
        raise ValueError(f'Invalid training history: {item}')
    validation = history[history.split.eq('validation')].copy()
    if validation.empty or not validation.date.dt.year.eq(2024).all() or not validation.target_end.lt('2025-01-01').all():
        raise ValueError(f'Invalid 2024 validation dates: {item}')
    test = history[history.split.eq('test')].rename(columns={'date': 'base_date', 'target_end': 'target_date'})
    matched = test.merge(reference[KEYS + ['actual', 'base_price']], on=KEYS,
                         suffixes=('', '_reference'), validate='one_to_one')
    if (len(matched) != len(reference) or not matched.base_date.ge('2025-01-01').all()
            or not np.allclose(matched.actual, matched.target_price)
            or not np.allclose(matched.base_price, matched.base_price_reference)):
        raise ValueError(f'Archived forecasts do not match training data: {item}')
    test = matched.rename(columns={'base_date': 'date', 'target_date': 'target_end'})
    return history, validation, test, PRICE_FEATURES + ['horizon', *ratios.columns]


def run_months(history, origins, columns, dry_run):
    predictions = []
    for month, rows in origins.groupby(origins.date.dt.to_period('M')):
        cutoff = month.start_time
        start = cutoff - pd.DateOffset(years=3)
        train = history[history.date.ge(start) & history.target_end.lt(cutoff)]
        if len(train) < 300:
            raise ValueError(f'Insufficient completed training targets for {month}')
        if dry_run:
            continue
        x = train[columns].replace([np.inf, -np.inf], np.nan)
        future = rows[columns].replace([np.inf, -np.inf], np.nan)
        models = {
            'monthly_lgbm': lgb.LGBMRegressor(**PARAMS, n_estimators=80),
            'monthly_mlp': make_pipeline(
                SimpleImputer(strategy='median'), StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(32, 16), alpha=.01, max_iter=100,
                             random_state=42)),
        }
        prices = {'last_value': rows.last_value.to_numpy()}
        for name, model in models.items():
            model.fit(x, train.target_return)
            prices[name] = rows.base_price.to_numpy() * (1 + model.predict(future))
        prices['monthly_blend'] = .5 * prices['monthly_lgbm'] + .25 * prices['monthly_mlp'] + .25 * prices['last_value']
        for name, values in prices.items():
            predictions.append(prediction_records(rows, values, name).assign(split=rows.split.to_numpy()))
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Compare monthly short-term refits and a small MLP.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/short_monthly_models')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    reference = pd.read_parquet(ROOT / 'results/all_items_experiment/predictions.parquet')
    reference = reference.query("term == 'short' and split == 'test' and model == 'current' and horizon in @HORIZONS")
    results, selections = [], []
    for item in sorted(ITEMS):
        current = reference[reference.item.eq(item)]
        if current.empty:
            raise ValueError(f'Missing current forecasts: {item}')
        history, validation, test, columns = prepare(item, current)
        print(f'{item}: validation={len(validation)}, test={len(test)}', flush=True)
        candidates = run_months(history, validation, columns, args.dry_run)
        if args.dry_run:
            run_months(history, test, columns, True)
            continue
        candidates = pd.concat(candidates, ignore_index=True)
        metrics = score_predictions(candidates)
        winners = metrics.sort_values(['nmae', 'direction_accuracy', 'model'],
                                      ascending=[True, False, True]).drop_duplicates('horizon')
        winners = winners[['item', 'horizon', 'model']]
        selections.append(winners)
        scored = pd.concat(run_months(history, test, columns, False), ignore_index=True)
        if (not scored.prediction.gt(0).all()
                or not scored.groupby(KEYS).model.nunique().eq(len(METHODS)).all()):
            raise ValueError(f'Invalid or incomplete forecasts: {item}')
        selected = scored.merge(winners, on=['item', 'horizon', 'model'], validate='many_to_one')
        if len(selected) != len(current):
            raise ValueError(f'Incomplete selected forecasts: {item}')
        results.extend([candidates, scored, current, selected.assign(model='validation_selected')])
    if args.dry_run:
        return
    predictions = pd.concat(results, ignore_index=True)
    metrics = score_predictions(predictions)
    test = predictions[predictions.split.eq('test')]
    yearly = score_predictions(test.assign(split=test.base_date.dt.year.astype(str)))
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    yearly.to_csv(args.output / 'metrics_by_year.csv', index=False)
    pd.concat(selections, ignore_index=True).to_csv(args.output / 'validation_selection.csv', index=False)
    print(metrics[metrics.split.eq('test')].groupby('model')[
        ['nmae', 'direction_accuracy']].mean().to_string())


if __name__ == '__main__':
    main()
