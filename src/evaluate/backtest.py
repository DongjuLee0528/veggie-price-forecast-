from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.price import load_prices, validate_prices
from src.models.baseline import predict_baselines


ROOT = Path(__file__).resolve().parents[2]


def backtest(prices: pd.DataFrame) -> pd.DataFrame:
    validate_prices(prices)
    records = []
    for item, group in prices.groupby('item'):
        group = group.sort_values('date').reset_index(drop=True)
        for split, year_start, year_end in [('validation', 2024, 2024), ('test', 2025, 9999)]:
            positions = group.index[group.date.dt.year.between(year_start, year_end)]
            for position in positions[::10]:
                if position < 6:
                    continue
                targets = group.iloc[position + 1:position + 11]
                targets = targets[targets.date.dt.year.le(year_end)]
                if targets.empty:
                    continue
                base = group.iloc[position]
                predictions = predict_baselines(prices, base.date, targets.date)
                predictions = predictions[predictions.item.eq(item)]
                actual = targets[['date', 'price_per_kg']].rename(columns={'date': 'target_date', 'price_per_kg': 'actual'})
                actual['horizon'] = np.arange(1, len(actual) + 1)
                predictions = predictions.merge(actual, on='target_date', validate='many_to_one')
                predictions['split'] = split
                predictions['base_price'] = group.iloc[position - 6:position + 1].price_per_kg.mean()
                records.append(predictions)
    if not records:
        raise ValueError('No evaluable dates in 2024 or later')
    return pd.concat(records, ignore_index=True)


def score_predictions(records: pd.DataFrame) -> pd.DataFrame:
    data = records.copy()
    valid = data.prediction.notna()
    data['absolute_relative_error'] = ((data.prediction - data.actual).abs() / data.actual).where(valid)
    data['squared_error'] = ((data.prediction - data.actual) ** 2).where(valid)
    data['direction_correct'] = (np.sign(data.prediction - data.base_price) == np.sign(data.actual - data.base_price)).astype(float).where(valid)
    scores = data.groupby(['split', 'item', 'horizon', 'model'], as_index=False).agg(
        n_total=('actual', 'size'), n_scored=('prediction', 'count'),
        nmae=('absolute_relative_error', 'mean'), rmse=('squared_error', 'mean'),
        direction_accuracy=('direction_correct', 'mean'),
    )
    scores['rmse'] = np.sqrt(scores.rmse)
    scores['coverage'] = scores.n_scored / scores.n_total
    return scores


def main():
    prices = load_prices(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    records = backtest(prices)
    scores = score_predictions(records)
    output = ROOT / 'results/baseline'
    output.mkdir(parents=True, exist_ok=True)
    records.to_parquet(output / 'backtest.parquet', index=False)
    for split in ('validation', 'test'):
        scores[scores.split.eq(split)].to_csv(output / f'{split}_metrics.csv', index=False)
    keys = ['split', 'item', 'base_date', 'target_date', 'horizon']
    complete = records.groupby(keys).prediction.transform(lambda values: values.notna().all())
    common = score_predictions(records.loc[complete])
    common.to_csv(output / 'common_sample_metrics.csv', index=False)
    print(scores.groupby(['split', 'model'])[['nmae', 'coverage']].mean().to_string())


if __name__ == '__main__':
    main()
