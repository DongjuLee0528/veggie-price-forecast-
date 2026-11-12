import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.evaluate.backtest import score_predictions
from src.models.train_lgbm import model_inputs, prediction_records
from src.models.train_production import freeze_selection


KEYS = ['item', 'horizon', 'base_date', 'target_date']


def choose_blends(validation, selected):
    if not validation.split.eq('validation').all() or not validation.base_date.dt.year.eq(2024).all():
        raise ValueError('Blend selection requires 2024 validation only')
    if validation.duplicated(KEYS + ['model']).any():
        raise ValueError('Duplicate validation predictions')
    choices = []
    for route in selected:
        group = validation[validation.item.eq(route['item']) & validation.horizon.eq(route['horizon'])]
        pivot = group.pivot(index=KEYS, columns='model', values='prediction')
        actual = group.drop_duplicates(KEYS).set_index(KEYS).actual.reindex(pivot.index)
        original = pivot[route['model']]
        best = {'baseline': 'last_value', 'model_weight': 1.,
                'validation_nmae': float(((original - actual).abs() / actual).mean())}
        for baseline in ['last_value', 'moving_average_7']:
            for weight in [.75, .5, .25, 0.]:
                prediction = weight * original + (1 - weight) * pivot[baseline]
                if prediction.isna().any() or not actual.gt(0).all():
                    raise ValueError('Incomplete validation comparisons')
                error = float(((prediction - actual).abs() / actual).mean())
                if error < best['validation_nmae']:
                    best = {'baseline': baseline, 'model_weight': weight, 'validation_nmae': error}
        choices.append({'item': route['item'], 'horizon': route['horizon'],
                        'model': route['model'], **best})
    return choices


def main():
    parser = argparse.ArgumentParser(description='Select simple blends on 2024; retrospectively evaluate stored models without training.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/lgbm_blend')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    run_dir = ROOT / 'results/lgbm_experiment_v2'
    _, selected = freeze_selection(run_dir)
    choices = choose_blends(pd.read_parquet(run_dir / 'validation_predictions.parquet'), selected)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'blends.json').write_text(json.dumps({
        'selection_source': '2024 validation only', 'selected': choices,
        'note': 'Retrospective evaluation period previously inspected; no fresh holdout claim.'},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    test = pd.read_parquet(ROOT / 'data/processed/test.parquet')
    test = test[test.term.eq('short') & test.horizon.isin([1, 5, 10])]
    archived = pd.read_parquet(ROOT / 'results/lgbm_final_test/test_predictions.parquet')
    archived = archived[archived.horizon.isin([1, 5, 10])]
    records = []
    for route, choice in zip(selected, choices):
        group = test[test.item.eq(route['item']) & test.horizon.eq(route['horizon'])]
        common = archived[archived.item.eq(route['item']) & archived.horizon.eq(route['horizon'])]
        scaffold = prediction_records(group, group.base_price, 'validation_selected')
        baseline = common[common.model.eq(choice['baseline'])].set_index(KEYS)
        aligned = baseline.reindex(pd.MultiIndex.from_frame(scaffold[KEYS]))
        if aligned.prediction.isna().any() or not np.array_equal(aligned.actual, scaffold.actual):
            raise ValueError('Test baseline samples differ')
        if route['model'] in {'last_value', 'moving_average_7'}:
            source = common[common.model.eq(route['model'])].set_index(KEYS).prediction
            prediction = source.reindex(pd.MultiIndex.from_frame(scaffold[KEYS])).to_numpy()
        else:
            booster = lgb.Booster(model_file=str(run_dir / route['file']))
            prediction = group.base_price.to_numpy() * (1 + booster.predict(
                model_inputs(group, route['inputs']), num_iteration=route['best_iteration'], num_threads=2))
        scaffold['prediction'] = prediction
        records.append(scaffold)
        blended = scaffold.copy()
        blended['model'] = 'validation_blend'
        blended['prediction'] = choice['model_weight'] * prediction + (1 - choice['model_weight']) * aligned.prediction.to_numpy()
        records.append(blended)
    new = pd.concat(records, ignore_index=True).assign(split='test')
    combined = pd.concat([new, archived], ignore_index=True)
    if not np.isfinite(combined.prediction).all() or not combined.prediction.gt(0).all():
        raise ValueError('Invalid retrospective predictions')
    if (combined.duplicated(KEYS + ['model']).any()
            or not combined.groupby(KEYS).model.nunique().eq(5).all()):
        raise ValueError('Comparisons must use identical samples')
    metrics = score_predictions(combined)
    combined.to_parquet(args.output / 'test_predictions.parquet', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    summary = metrics.groupby(['item', 'model'], as_index=False)[['nmae', 'direction_accuracy']].mean()
    summary.to_csv(args.output / 'comparison.csv', index=False)
    garlic = new[new.item.eq('깐마늘') & new.model.eq('validation_selected')].copy()
    garlic['absolute_percentage_error'] = (garlic.prediction / garlic.actual - 1).abs()
    garlic.nlargest(20, 'absolute_percentage_error').to_csv(args.output / 'garlic_error_review.csv', index=False)
    print(summary.to_string(index=False))
    print(metrics.groupby('model')[['nmae', 'direction_accuracy']].mean().to_string())


if __name__ == '__main__':
    main()
