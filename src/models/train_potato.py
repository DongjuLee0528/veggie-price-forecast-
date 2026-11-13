import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.features.targets import dekad_index
from src.models.train_lgbm import model_inputs, prediction_records, progress_callback, validate_split
from src.models.train_mid import series_predictions
from src.models.tune_mid import ets_predictions


def main():
    parser = argparse.ArgumentParser(description='Compare short- and mid-term potato forecasts.')
    parser.add_argument('--dry-run', action='store_true', help='Check data and sample alignment without fitting.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/potato_sumi')
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error('Choose a new output directory to preserve previous results')
    data_dir = ROOT / 'data/processed/potato_sumi'
    columns = json.loads((data_dir / 'manifest.json').read_text())['input_columns']
    daily = pd.read_parquet(data_dir / 'prices.parquet').sort_values('date')
    history = daily.set_index('date').price_per_kg
    means = history.groupby(history.index.map(dekad_index)).mean()
    frames = {split: pd.read_parquet(data_dir / f'{split}.parquet')
              for split in ['train', 'validation', 'test']}
    mid_eval = pd.read_parquet(data_dir / 'mid_evaluation.parquet')
    params = dict(objective='regression_l1', n_estimators=500, learning_rate=.03,
                  num_leaves=15, min_child_samples=50, reg_lambda=1., n_jobs=2,
                  random_state=42, verbosity=-1)
    results = []
    selections = []
    for number, (term, horizons) in enumerate([('short', {1, 5, 10}), ('mid', {2, 3})], 1):
        train = frames['train'][frames['train'].term.eq(term)]
        rows = (pd.concat([frames['validation'], frames['test']], ignore_index=True)
                if term == 'short' else mid_eval)
        rows = rows[rows.term.eq(term)].reset_index(drop=True)
        valid = rows[rows.split.eq('validation')]
        test = rows[rows.split.eq('test')]
        for split, frame in [('train', train), ('validation', valid), ('test', test)]:
            validate_split(frame, split)
            if set(frame.item) != {'감자 수미'} or set(frame.horizon) != horizons:
                raise ValueError(f'Unexpected item or horizon coverage: {term} {split}')
        if not test.date.ge('2025-01-01').all() or not rows.target_end.le(history.index.max()).all():
            raise ValueError('Evaluation dates are outside price history')
        if term == 'short':
            actual = rows.target_end.map(history)
        else:
            rows['target_dekad'] = rows.origin_dekad + rows.horizon
            actual = rows.target_dekad.map(means)
        np.testing.assert_allclose(actual, rows.target_price)
        np.testing.assert_allclose(rows.date.map(history.rolling(7).mean()), rows.base_price)
        x_train, x_valid, x_eval = [model_inputs(frame, columns) for frame in (train, valid, rows)]
        print(f'{term}: train={len(train)}, validation={len(valid)}, test={len(test)}', flush=True)
        if args.dry_run:
            continue
        if number == 1:
            args.output.mkdir(parents=True, exist_ok=False)
        model = lgb.LGBMRegressor(**params)
        model.fit(x_train, train.target_return, eval_set=[(x_valid, valid.target_return)],
                  callbacks=[lgb.early_stopping(40, verbose=False),
                             progress_callback(args.output, term, number, 2, 500, 10)])
        model.booster_.save_model(str(args.output / f'{term}_lgbm.txt'))
        price = rows.base_price * (1 + model.predict(x_eval))
        methods = {'price_lgbm': price}
        if term == 'short':
            methods.update(last_value=rows.date.map(history), moving_average_7=rows.base_price)
        else:
            methods.update(series_predictions(rows, daily, means).reindex(rows.index).to_dict('series'))
            methods['holt_winters'] = ets_predictions(
                rows.rename(columns={'date': 'base_date'}), means, daily.date.min())
            methods['mid_blend'] = .5625 * price + .1875 * methods['sarimax'] + .25 * methods['holt_winters']
        term_predictions = []
        for name, values in methods.items():
            records = prediction_records(rows, values, name)
            records['split'] = rows.split.to_numpy()
            term_predictions.append(records.assign(term=term))
        predictions = pd.concat(term_predictions, ignore_index=True)
        keys = ['split', 'item', 'horizon', 'base_date', 'target_date']
        if (not predictions.prediction.gt(0).all()
                or predictions.duplicated(keys + ['model']).any()
                or not predictions.groupby(keys).size().eq(len(methods)).all()):
            raise ValueError('Expected complete positive forecasts on matching dates')
        metrics = score_predictions(predictions)
        chosen = metrics[metrics.split.eq('validation')].sort_values(
            ['nmae', 'direction_accuracy', 'model'], ascending=[True, False, True]
        ).drop_duplicates('horizon')[['horizon', 'model']]
        selections.append(chosen.assign(term=term))
        selected = predictions[predictions.split.eq('test')].merge(
            chosen, on=['horizon', 'model'], validate='many_to_one')
        results.append(pd.concat([predictions, selected.assign(model='validation_selected')]))
        print(f'{term}: best round={model.best_iteration_}', flush=True)
    if args.dry_run:
        return
    predictions = pd.concat(results, ignore_index=True)
    metrics = pd.concat([score_predictions(frame).assign(term=term)
                         for term, frame in predictions.groupby('term')], ignore_index=True)
    predictions.to_parquet(args.output / 'predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    pd.concat(selections).to_csv(args.output / 'validation_selection.csv', index=False)
    print(metrics[metrics.split.eq('test')][
        ['term', 'horizon', 'model', 'n_scored', 'nmae', 'direction_accuracy']].to_string(index=False))
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
