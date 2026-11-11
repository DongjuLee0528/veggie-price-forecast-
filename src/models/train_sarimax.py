import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.collect.price import load_prices
from src.evaluate.backtest import ROOT, score_predictions


ORDER = (1, 1, 0)
SEASONAL_ORDER = (1, 0, 0, 6)
HORIZON = 10
CALIBRATION_END = pd.Timestamp('2025-07-01')
WEIGHTS = [0., .25, .5, .75]


def prepare_samples(prices, reference):
    required = ['item', 'horizon', 'base_date', 'target_date', 'actual', 'base_price', 'prediction']
    reference = reference.loc[
        reference.model.eq('adaptive') & reference.horizon.eq(HORIZON), required
    ].rename(columns={'prediction': 'adaptive'}).copy()
    if reference.empty or reference.duplicated(['item', 'base_date']).any():
        raise ValueError('Expected one archived adaptive forecast per item and origin')
    series = {}
    for item, group in prices.groupby('item'):
        daily = group.sort_values('date').set_index('date').price_per_kg
        sample = reference.item.eq(item)
        origins = daily.index.get_indexer(reference.loc[sample, 'base_date'])
        targets = origins + HORIZON
        if (not sample.any() or (origins < 6).any() or (targets >= len(daily)).any()
                or not np.array_equal(daily.index[targets], reference.loc[sample, 'target_date'])
                or not np.allclose(daily.iloc[targets], reference.loc[sample, 'actual'])
                or not np.allclose(daily.rolling(7).mean().iloc[origins], reference.loc[sample, 'base_price'])):
            raise ValueError(f'Archived forecast dates or prices do not match price history: {item}')
        reference.loc[sample, 'position'] = origins
        series[item] = np.log(daily.to_numpy(dtype=float))
    reference['position'] = reference.position.astype(int)
    reference['quarter'] = reference.base_date.dt.to_period('Q')
    return reference.sort_values(['item', 'base_date']), series


def predict_quarter(log_prices, positions):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    first = int(positions[0])
    model = SARIMAX(log_prices[:first + 1], order=ORDER, seasonal_order=SEASONAL_ORDER)
    fitted = model.fit(disp=False, maxiter=100)
    if not fitted.mle_retvals['converged']:
        print('SARIMAX L-BFGS did not converge; retrying with Powell.', flush=True)
        fitted = model.fit(method='powell', disp=False, maxiter=200)
        if not fitted.mle_retvals['converged']:
            raise ValueError('SARIMAX fit did not converge with either L-BFGS or Powell')
    previous = first
    forecasts = []
    for position in positions:
        if position > previous:
            fitted = fitted.extend(log_prices[previous + 1:position + 1])
        forecast = fitted.get_forecast(steps=HORIZON).predicted_mean
        forecasts.append(np.exp(forecast[-1]))
        previous = position
    return forecasts


def select_weight(calibration):
    trials = []
    for weight in WEIGHTS:
        rows = calibration.assign(split='calibration', model='blend',
                                  prediction=(1 - weight) * calibration.adaptive + weight * calibration.sarimax)
        metric = score_predictions(rows)[['nmae', 'direction_accuracy']].mean()
        trials.append({'sarimax_weight': weight, **metric.to_dict()})
    baseline_error = trials[0]['nmae']
    chosen = max((trial for trial in trials if trial['nmae'] <= baseline_error),
                 key=lambda trial: (trial['direction_accuracy'], -trial['nmae'], -trial['sarimax_weight']))
    return chosen['sarimax_weight'], pd.DataFrame(trials)


def main():
    parser = argparse.ArgumentParser(description='Compare quarterly refit SARIMAX and the archived 10-trading-day ensemble.')
    parser.add_argument('--dry-run', action='store_true', help='Validate dates and samples without fitting models.')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/sarimax_experiment')
    args = parser.parse_args()
    if not args.dry_run and args.output.exists():
        parser.error('Use a new output directory')
    prices = load_prices(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    reference = pd.read_parquet(ROOT / 'results/lgbm_adaptive/test_predictions.parquet')
    samples, series = prepare_samples(prices, reference)
    calibration_mask = samples.target_date.lt(CALIBRATION_END)
    evaluation_mask = samples.base_date.ge(CALIBRATION_END)
    if not calibration_mask.any() or not evaluation_mask.any() or (calibration_mask & evaluation_mask).any():
        raise ValueError('Invalid calibration or evaluation dates')
    print(f'Calibration: {calibration_mask.sum()}, evaluation: {evaluation_mask.sum()}', flush=True)
    if args.dry_run:
        return

    samples = samples.loc[calibration_mask | evaluation_mask].copy()
    groups = samples.groupby(['item', 'quarter'])
    for number, ((item, quarter), rows) in enumerate(groups, 1):
        start = perf_counter()
        try:
            predictions = predict_quarter(series[item], rows.position.to_numpy())
        except ValueError as error:
            raise ValueError(f'{item} {quarter}: {error}') from error
        samples.loc[rows.index, 'sarimax'] = predictions
        print(f'[{number}/{len(groups)}] {item} {quarter}: {len(rows)} origins, {perf_counter()-start:.2f}s', flush=True)
    if not np.isfinite(samples[['adaptive', 'sarimax']]).all().all() or not samples[['adaptive', 'sarimax']].gt(0).all().all():
        raise ValueError('Expected complete positive price forecasts')
    calibration = samples[samples.target_date.lt(CALIBRATION_END)]
    weight, trials = select_weight(calibration)
    evaluation = samples[samples.base_date.ge(CALIBRATION_END)].copy()
    evaluation['split'] = evaluation.base_date.dt.year.astype(str) + '-H' + np.where(evaluation.base_date.dt.month.le(6), '1', '2')
    evaluation['blend'] = (1 - weight) * evaluation.adaptive + weight * evaluation.sarimax
    predictions = evaluation.melt(
        id_vars=['item', 'horizon', 'base_date', 'target_date', 'actual', 'base_price', 'split'],
        value_vars=['adaptive', 'sarimax', 'blend'], var_name='model', value_name='prediction')
    metrics = score_predictions(predictions)
    args.output.mkdir(parents=True, exist_ok=False)
    trials.to_csv(args.output / 'calibration.csv', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    predictions.to_parquet(args.output / 'predictions.parquet', index=False)
    (args.output / 'run.json').write_text(json.dumps({
        'order': ORDER, 'seasonal_order': SEASONAL_ORDER, 'sarimax_weight': weight,
        'calibration_targets_before': str(CALIBRATION_END.date()),
        'evaluation_origins_from': str(CALIBRATION_END.date()),
        'samples_per_method': len(evaluation),
        'note': 'Retrospective comparison on a previously inspected period. No exogenous inputs.',
    }, indent=2) + '\n')
    print(f'Calibrated SARIMAX weight: {weight:.0%}')
    print(metrics.groupby(['split', 'model'])[['nmae', 'direction_accuracy']].mean().to_string())
    print(metrics.groupby('model')[['nmae', 'direction_accuracy']].mean().to_string())
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
