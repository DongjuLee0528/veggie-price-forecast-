import argparse
from itertools import product
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.features.targets import dekad_index
from src.models.train_lgbm import model_inputs, prediction_records, validate_split
from src.models.tune_mid import ets_predictions


def main():
    parser = argparse.ArgumentParser(description='Tune potato forecasts using 2024 validation.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/potato_sumi_tuning')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory')
    data = ROOT / 'data/processed/potato_sumi'
    columns = json.loads((data / 'manifest.json').read_text())['input_columns']
    train = pd.read_parquet(data / 'train.parquet')
    short_train = train[train.term.eq('short')]
    validate_split(short_train, 'train')
    daily = pd.read_parquet(data / 'prices.parquet')
    means = daily.groupby(daily.date.map(dekad_index)).price_per_kg.mean()
    archive = pd.read_parquet(ROOT / 'results/potato_sumi/predictions.parquet')
    predictions, selections = [], []
    for term in ['short', 'mid']:
        rows = (pd.concat([pd.read_parquet(data / f'{split}.parquet')
                           for split in ['validation', 'test']]) if term == 'short'
                else pd.read_parquet(data / 'mid_evaluation.parquet'))
        rows = rows[rows.term.eq(term)].reset_index(drop=True)
        keys = ['split', 'item', 'horizon', 'base_date', 'target_date']
        expected = prediction_records(rows, rows.base_price, 'alignment')
        expected['split'] = rows.split
        source = archive[archive.term.eq(term) & archive.model.ne('validation_selected')]
        wide = source.pivot(index=keys, columns='model', values='prediction')
        aligned = wide.reindex(pd.MultiIndex.from_frame(expected[keys]))
        if len(aligned) != len(wide) or aligned.isna().any().any():
            raise ValueError('Saved forecast dates do not match the tuning data')
        candidates = {name: aligned[name].to_numpy() for name in aligned}
        valid = rows[rows.split.eq('validation')]
        test = rows[rows.split.eq('test')]
        validate_split(valid, 'validation')
        validate_split(test, 'test')
        print(f'{term}: validation={len(valid)}, evaluation={len(test)}', flush=True)
        if args.dry_run:
            continue
        if term == 'short':
            for years, leaves, child in product([2, 4], [7, 15], [25, 50]):
                subset = short_train[short_train.date.ge(f'{2024-years}-01-01')]
                model = lgb.LGBMRegressor(
                    objective='regression_l1', n_estimators=500, learning_rate=.03,
                    num_leaves=leaves, min_child_samples=child, reg_lambda=1.,
                    n_jobs=2, random_state=42, verbosity=-1)
                model.fit(model_inputs(subset, columns), subset.target_return,
                          eval_set=[(model_inputs(valid, columns), valid.target_return)],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
                name = f'lgbm_{years}y_{leaves}leaves_{child}child'
                candidates[name] = rows.base_price.to_numpy() * (1 + model.predict(model_inputs(rows, columns)))
                print(f'{name}: best round={model.best_iteration_}', flush=True)
            for name in list(candidates):
                if name.startswith('lgbm_') or name == 'price_lgbm':
                    for weight in [.25, .5]:
                        candidates[f'{name}_last_{weight:g}'] = (
                            (1 - weight) * candidates[name] + weight * candidates['last_value'])
        else:
            rows['target_dekad'] = rows.origin_dekad + rows.horizon
            origins = rows.rename(columns={'date': 'base_date'})
            for name, options in [
                ('ets_no_trend', {'trend': None, 'damped': False}),
                ('ets_undamped', {'damped': False}),
                ('ets_recent_3y', {'lookback': 108}),
            ]:
                candidates[name] = ets_predictions(origins, means, daily.date.min(), **options).to_numpy()
                print(f'{name}: complete', flush=True)
            for name in ['holt_winters', 'ets_no_trend', 'ets_undamped', 'ets_recent_3y']:
                for component in ['price_lgbm', 'sarimax', 'last_completed']:
                    candidates[f'{name}_with_{component}'] = .75 * candidates[name] + .25 * candidates[component]
        records = []
        for name, values in candidates.items():
            forecast = prediction_records(rows, values, name)
            if not forecast.prediction.gt(0).all():
                raise ValueError(f'Nonpositive forecast: {name}')
            records.append(forecast.assign(split=rows.split.to_numpy(), term=term))
        scored = pd.concat(records, ignore_index=True)
        metrics = score_predictions(scored)
        winners = metrics[metrics.split.eq('validation')].sort_values(
            ['nmae', 'direction_accuracy', 'model'], ascending=[True, False, True]
        ).drop_duplicates('horizon')[['horizon', 'model']]
        selections.append(winners.assign(term=term))
        selected = scored[scored.split.eq('test')].merge(winners, on=['horizon', 'model'], validate='many_to_one')
        predictions.append(pd.concat([scored, selected.assign(model='validation_selected')]))
    if args.dry_run:
        return
    combined = pd.concat(predictions, ignore_index=True)
    metrics = pd.concat([score_predictions(frame).assign(term=term)
                         for term, frame in combined.groupby('term')], ignore_index=True)
    test = combined[combined.split.eq('test')].copy()
    test['split'] = test.base_date.dt.year.astype(str)
    yearly = pd.concat([score_predictions(frame).assign(term=term)
                        for term, frame in test.groupby('term')], ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=False)
    combined.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    yearly.to_csv(args.output / 'metrics_by_year.csv', index=False)
    selected = pd.concat(selections, ignore_index=True)
    selected.to_csv(args.output / 'validation_selection.csv', index=False)
    print(selected.to_string(index=False))
    print(metrics[metrics.split.eq('test') & metrics.model.isin(
        ['price_lgbm', 'last_value', 'holt_winters', 'validation_selected']
    )][['term', 'horizon', 'model', 'nmae', 'direction_accuracy']].to_string(index=False))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
