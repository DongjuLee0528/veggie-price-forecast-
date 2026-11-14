import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.features.targets import dekad_index
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs, prediction_records, validate_split


ITEMS = {'감자 수미', '배추', '무', '양파', '건고추'}
KEYS = ['split', 'item', 'horizon', 'base_date', 'target_date']
PARAMS = dict(objective='regression_l1', learning_rate=.03, num_leaves=15,
              min_child_samples=50, reg_lambda=1., n_jobs=2, random_state=42, verbosity=-1)


def last_origins(rows):
    return (rows.assign(dekad=rows.date.map(dekad_index))
            .sort_values('date').drop_duplicates(['split', 'item', 'dekad', 'horizon'], keep='last')
            .drop(columns='dekad'))


def archived(rows, directory, method):
    filename = 'test_predictions.parquet' if directory == 'lgbm_adaptive' else 'predictions.parquet'
    frame = pd.read_parquet(ROOT / 'results' / directory / filename)
    frame = frame[frame.model.eq(method) & frame.item.eq(rows.item.iloc[0])]
    expected = prediction_records(rows, rows.base_price, 'reference').assign(split=rows.split.to_numpy())
    aligned = frame.set_index(KEYS).reindex(pd.MultiIndex.from_frame(expected[KEYS]))
    if not np.isfinite(aligned.prediction).all():
        raise ValueError(f'Missing archived forecasts: {directory}, {method}')
    np.testing.assert_allclose(aligned.actual, expected.actual)
    np.testing.assert_allclose(aligned.base_price, expected.base_price)
    return aligned.prediction.to_numpy(copy=True)


def evaluate(data_dir, prices, item, term, dry_run):
    horizons = [1, 5, 10] if term == 'short' else [2, 3]
    potato = item == '감자 수미'
    frames = {}
    for split in ['train', 'validation', 'test']:
        source = data_dir
        if term == 'short' and not potato and split != 'test':
            source = data_dir / 'experiment_v2'
        frame = pd.read_parquet(source / f'{split}.parquet')
        frame = frame[frame.item.eq(item) & frame.term.eq(term)].copy()
        validate_split(frame, split)
        if split != 'train':
            frame = frame[frame.horizon.isin(horizons)]
            if term == 'mid':
                frame = last_origins(frame)
        if not set(horizons).issubset(frame.horizon):
            raise ValueError(f'Missing horizons: {item}, {term}, {split}')
        frames[split] = frame
    train = frames['train']
    if term == 'short' and potato:
        train = train[train.date.ge('2022-01-01')]
    rows = pd.concat([frames['validation'], frames['test']], ignore_index=True)
    valid = rows.split.eq('validation')
    test = rows[~valid]
    if not test.date.ge('2025-01-01').all():
        raise ValueError('Evaluation origin predates test period')

    reference = {}
    if term == 'short':
        activity = prices.set_index('date')[['volume_kg', 'n_rows']].sort_index()
        ratios = (activity / activity.shift().rolling(7).mean()).add_suffix('_ratio_prev7')
        train = train.join(ratios, on='date', validate='many_to_one')
        rows = rows.join(ratios, on='date', validate='many_to_one')
        for frame in (train, rows):
            if np.isinf(frame[ratios.columns]).any().any() or frame[ratios.columns].isna().all().any():
                raise ValueError(f'Invalid activity features: {item}')
        if potato:
            current = archived(test, 'potato_sumi_tuning', 'validation_selected')
            current[test.horizon.eq(5).to_numpy()] = archived(
                test[test.horizon.eq(5)], 'potato_sumi_volume', 'activity_blend')
        else:
            current = archived(test, 'lgbm_adaptive', 'adaptive')
        variants = [('price_lgbm', train, []), ('activity_lgbm', train, list(ratios.columns))]
        baseline = test.date.map(prices.set_index('date').price_per_kg).to_numpy()
        reference = {'current': current, 'last_value': baseline}
    else:
        variants = [('price_lgbm', train, []), ('last_origin_lgbm', last_origins(train), [])]
        if potato:
            reference['current'] = archived(test, 'potato_sumi', 'holt_winters')
        else:
            reference['current'] = archived(test, 'mid_tuning', 'blend_with_ets')
            original = archived(rows, 'lgbm_mid_comparison', 'price_lgbm')
            fixed = archived(rows, 'mid_tuning', 'blend_with_ets')
    print(f'{item} {term}: train={len(train)}, validation={valid.sum()}, test={len(test)}'
          + (f', last-origin train={len(variants[1][1])}' if term == 'mid' else ''), flush=True)
    if dry_run:
        return []

    results = []
    columns = PRICE_FEATURES + ['horizon']
    rounds = 500
    for name, sample, added in variants:
        inputs = columns + added
        model = lgb.LGBMRegressor(**PARAMS, n_estimators=rounds if name == 'activity_lgbm' else 500)
        options = {}
        if name != 'activity_lgbm':
            options = dict(eval_set=[(model_inputs(rows.loc[valid], inputs), rows.loc[valid, 'target_return'])],
                           callbacks=[lgb.early_stopping(40, verbose=False)])
        model.fit(model_inputs(sample, inputs), sample.target_return, **options)
        if name == 'price_lgbm':
            rounds = model.best_iteration_
        prediction = rows.base_price.to_numpy() * (1 + model.predict(model_inputs(rows, inputs)))
        results.append(prediction_records(rows, prediction, name).assign(split=rows.split.to_numpy()))
        if term == 'mid' and not potato:
            blended = fixed + .5625 * (prediction - original)
            results.append(prediction_records(rows, blended, f'{name}_blend').assign(
                split=rows.split.to_numpy()))
    for name, prediction in reference.items():
        results.append(prediction_records(test, prediction, name).assign(split='test'))
    return [frame.assign(term=term) for frame in results]


def main():
    parser = argparse.ArgumentParser(description='Compare short-term activity and mid-term origin sampling across five items.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/all_items_experiment')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    processed = ROOT / 'data/processed'
    potato_dir = processed / 'potato_sumi'
    daily = pd.read_parquet(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    potato = pd.read_parquet(potato_dir / 'prices.parquet')
    results = []
    for item in sorted(ITEMS):
        data_dir, prices = (potato_dir, potato) if item == '감자 수미' else (processed, daily)
        prices = prices[prices.item.eq(item)]
        for term in ['short', 'mid']:
            results.extend(evaluate(data_dir, prices, item, term, args.dry_run))
    if args.dry_run:
        return
    predictions = pd.concat(results, ignore_index=True)
    if not predictions.prediction.gt(0).all() or predictions.duplicated(KEYS + ['term', 'model']).any():
        raise ValueError('Invalid comparison forecasts')
    metrics = pd.concat([score_predictions(frame).assign(term=term)
                         for term, frame in predictions.groupby('term')], ignore_index=True)
    test = predictions[predictions.split.eq('test')]
    yearly = pd.concat([score_predictions(frame.assign(split=frame.base_date.dt.year.astype(str))).assign(term=term)
                        for term, frame in test.groupby('term')], ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    yearly.to_csv(args.output / 'metrics_by_year.csv', index=False)
    print(metrics[metrics.split.eq('test')][
        ['term', 'item', 'horizon', 'model', 'n_scored', 'nmae', 'direction_accuracy']
    ].to_string(index=False))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
