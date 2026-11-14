import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ensemble.adaptive import apply_recent_weights
from src.evaluate.adaptive import align_predictions
from src.models.compare_all_items import archived
from src.models.rolling_short import HORIZONS, ITEMS, KEYS, ROOT, prepare
from src.models.train_production import freeze_selection


def classify_models(tune):
    trees = [(7, 50), (7, 25), (15, 50), (15, 25)] if tune else [(7, 50)]
    strengths = [1., .1, 10.] if tune else [1.]
    models = {}
    for leaves, child in trees:
        name = 'direction_lgbm' if (leaves, child) == (7, 50) else f'direction_lgbm_{leaves}_{child}'
        models[name] = lgb.LGBMClassifier(
            n_estimators=80, learning_rate=.03, num_leaves=leaves, min_child_samples=child,
            reg_lambda=1., n_jobs=2, random_state=42, verbosity=-1)
    for strength in strengths:
        name = 'direction_logistic' if strength == 1 else f'direction_logistic_c{strength:g}'
        models[name] = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                                     LogisticRegression(C=strength, max_iter=500, random_state=42))
    return models


def validation_prices():
    results = ROOT / 'results'
    run_dir = results / 'lgbm_experiment_v2'
    _, routes = freeze_selection(run_dir)
    rows = pd.read_parquet(run_dir / 'validation_predictions.parquet')
    blends = json.loads((results / 'lgbm_blend/blends.json').read_text())['selected']
    configs = json.loads((results / 'lgbm_adaptive/selection.json').read_text())['selected']
    blend_by_key = {(row['item'], row['horizon']): row for row in blends}
    config_by_key = {(row['item'], row['horizon']): row['config'] for row in configs}
    predictions = []
    for route in routes:
        item, horizon = key = route['item'], route['horizon']
        if item not in ITEMS or horizon not in HORIZONS:
            continue
        group = align_predictions(rows[rows.item.eq(item) & rows.horizon.eq(horizon)])
        blend = blend_by_key[key]
        weight = blend['model_weight']
        group['validation_selected'] = group[route['model']]
        group['validation_blend'] = weight * group.validation_selected + (1 - weight) * group[blend['baseline']]
        group['prediction'] = apply_recent_weights(group, config_by_key[key])
        predictions.append(group.assign(split='validation')[KEYS + ['prediction', 'actual', 'base_price']])
    return pd.concat(predictions, ignore_index=True)


def current_validation(rows, prices):
    expected = rows.rename(columns={'date': 'base_date', 'target_end': 'target_date'})
    if rows.item.iloc[0] == '감자 수미':
        selection = pd.read_csv(ROOT / 'results/potato_sumi_tuning/validation_selection.csv')
        method, = selection.loc[selection.term.eq('short') & selection.horizon.eq(10), 'model']
        predictions = pd.Series(index=rows.index, dtype=float)
        for horizon, directory, name in [
            (5, 'potato_sumi_volume', 'activity_blend'),
            (10, 'potato_sumi_tuning', method),
        ]:
            matches = rows.horizon.eq(horizon)
            predictions.loc[matches] = archived(rows[matches], directory, name)
        expected = expected.assign(prediction=predictions, actual=rows.target_price)
    else:
        aligned = expected[KEYS].merge(prices, on=KEYS, how='left', validate='one_to_one')
        np.testing.assert_allclose(aligned.base_price, expected.base_price)
        np.testing.assert_allclose(aligned.actual, expected.target_price)
        expected = aligned
    if not np.isfinite(expected.prediction).all():
        raise ValueError(f'Incomplete current validation forecasts: {rows.item.iloc[0]}')
    return current_directions(expected)


def current_directions(rows):
    return rows[KEYS].assign(
        model='current',
        direction=np.sign(rows.prediction - rows.base_price).astype(int),
        actual_direction=np.sign(rows.actual - rows.base_price).astype(int))


def signals(history, origins, columns, dry_run, tune=False):
    records = []
    for (month, horizon), rows in origins.groupby([origins.date.dt.to_period('M'), 'horizon']):
        cutoff = month.start_time
        train = history[history.horizon.eq(horizon)
                        & history.date.ge(cutoff - pd.DateOffset(years=3))
                        & history.target_end.lt(cutoff)]
        labels = np.sign(train.target_price - train.base_price).astype(int)
        if len(train) < 150 or not {-1, 1}.issubset(labels.unique()):
            raise ValueError(f'Insufficient directional training data: {rows.item.iloc[0]} {month} h={horizon}')
        if dry_run:
            continue
        x = train[columns].replace([np.inf, -np.inf], np.nan)
        future = rows[columns].replace([np.inf, -np.inf], np.nan)
        models = classify_models(tune)
        common = rows[['split', 'item', 'horizon', 'date', 'target_end']].rename(
            columns={'date': 'base_date', 'target_end': 'target_date'})
        common['actual_direction'] = np.sign(rows.target_price - rows.base_price).astype(int)
        records.append(common.assign(model='last_value',
                                     direction=np.sign(rows.last_value - rows.base_price).astype(int)))
        for name, model in models.items():
            model.fit(x, labels)
            records.append(common.assign(model=name, direction=model.predict(future)))
    if dry_run:
        return None
    result = pd.concat(records, ignore_index=True)
    if (result.duplicated(KEYS + ['model']).any()
            or len(result) != (1 + len(models)) * len(origins)
            or not result.direction.isin([-1, 0, 1]).all()):
        raise ValueError('Incomplete or invalid direction forecasts')
    return result


def score(records):
    return (records.assign(hit=records.direction.eq(records.actual_direction))
            .groupby(['split', 'item', 'horizon', 'model'], as_index=False)
            .agg(n_scored=('hit', 'size'), correct=('hit', 'sum'), direction_accuracy=('hit', 'mean')))


def main():
    parser = argparse.ArgumentParser(description='Compare direct short-term direction classifiers.')
    parser.add_argument('--dry-run', action='store_true', help='Validate dates and labels without fitting.')
    parser.add_argument('--tune', action='store_true', help='Tune direction classifiers and include the current route in validation.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / 'results' / ('short_direction_tuning' if args.tune else 'short_direction_models')
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    reference = pd.read_parquet(ROOT / 'results/all_items_experiment/predictions.parquet')
    reference = reference.query("term == 'short' and split == 'test' and model == 'current' and horizon in @HORIZONS")
    prices = validation_prices() if args.tune else None
    forecasts, choices = [], []
    for item in sorted(ITEMS):
        current = reference[reference.item.eq(item)]
        if current.empty:
            raise ValueError(f'Missing current forecasts: {item}')
        history, validation, test, columns = prepare(item, current)
        columns = [column for column in columns if column != 'horizon']
        print(f'{item}: validation={len(validation)}, test={len(test)}', flush=True)
        baseline_before = current_validation(validation, prices) if args.tune else None
        before = signals(history, validation, columns, args.dry_run, args.tune)
        after = signals(history, test, columns, args.dry_run, args.tune)
        if args.dry_run:
            continue
        baseline = current_directions(current)
        if args.tune:
            before = pd.concat([before, baseline_before], ignore_index=True)
            after = pd.concat([after, baseline], ignore_index=True)
            validation_groups = before.groupby(KEYS)
            if (before.duplicated(KEYS + ['model']).any()
                    or not validation_groups.actual_direction.nunique().eq(1).all()
                    or not validation_groups.model.nunique().eq(len(classify_models(True)) + 2).all()):
                raise ValueError(f'Expected identical validation samples: {item}')
        winners = (score(before).sort_values(['direction_accuracy', 'model'], ascending=[False, True])
                   .drop_duplicates('horizon').drop(columns='split'))
        choices.append(winners)
        winner_keys = ['item', 'horizon', 'model']
        chosen = after.merge(winners[winner_keys], on=winner_keys, validate='many_to_one')
        if len(chosen) != len(current):
            raise ValueError(f'Incomplete selected forecasts: {item}')
        if not args.tune:
            choices[-1] = winners[winner_keys]
            forecasts.append(baseline)
        forecasts.extend([before, after, chosen.assign(model='validation_selected')])
    if args.dry_run:
        return
    rows = pd.concat(forecasts, ignore_index=True)
    test = rows[rows.split.eq('test')]
    coverage = test.groupby(KEYS).model.nunique()
    expected_methods = len(classify_models(args.tune)) + 3
    if (len(coverage) != len(reference) or not coverage.eq(expected_methods).all()
            or test.duplicated(KEYS + ['model']).any()):
        raise ValueError('Expected identical samples for all methods')
    metrics = score(rows)
    yearly = score(test.assign(split=test.base_date.dt.year.astype(str)))
    args.output.mkdir(parents=True, exist_ok=False)
    rows.to_parquet(args.output / 'directions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    yearly.to_csv(args.output / 'metrics_by_year.csv', index=False)
    pd.concat(choices, ignore_index=True).to_csv(args.output / 'validation_selection.csv', index=False)
    summary = metrics[metrics.split.eq('test')].groupby('model').agg(
        n_scored=('n_scored', 'sum'), correct=('correct', 'sum'), macro_accuracy=('direction_accuracy', 'mean'))
    if args.tune:
        summary['direction_accuracy'] = summary.correct / summary.n_scored
        yearly_summary = yearly.groupby(['split', 'model'])[['correct', 'n_scored']].sum()
        print((yearly_summary.correct / yearly_summary.n_scored).unstack().to_string())
    print(summary.to_string())
    if args.tune:
        print(f"Validation-selected direction accuracy: {summary.loc['validation_selected', 'direction_accuracy']:.2%} (goal: 70%)")
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
