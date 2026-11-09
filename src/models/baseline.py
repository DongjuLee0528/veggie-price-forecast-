import numpy as np
import pandas as pd

from src.collect.price import validate_prices


def predict_baselines(prices: pd.DataFrame, base_date, target_dates) -> pd.DataFrame:
    validate_prices(prices)
    base_date = pd.Timestamp(base_date)
    targets = pd.DatetimeIndex(target_dates)
    if pd.isna(base_date) or base_date != base_date.normalize():
        raise ValueError("base_date must be a valid calendar date")
    if targets.empty or targets.hasnans or targets.has_duplicates or (targets <= base_date).any() or (targets != targets.normalize()).any():
        raise ValueError("Target dates must be unique valid dates after base_date")
    history = prices.loc[prices.date.le(base_date)].sort_values('date')
    records = []
    for item in prices.item.unique():
        series = history.loc[history.item.eq(item)].set_index('date').price_per_kg
        if len(series) < 7:
            raise ValueError(f"At least seven historical trading days required: {item}")
        current_mean = series.iloc[-7:].mean()
        previous_date = base_date - pd.DateOffset(years=1)
        previous_window = series.loc[series.index <= previous_date].tail(7)
        previous_mean = previous_window.mean() if len(previous_window) == 7 else np.nan
        for target in targets:
            previous_target = target - pd.DateOffset(years=1)
            previous_price = series.get(previous_target, np.nan) if previous_target <= base_date else np.nan
            values = {
                'last_value': series.iloc[-1],
                'moving_average_7': current_mean,
                'year_ago_adjusted': previous_price * current_mean / previous_mean,
            }
            for model, prediction in values.items():
                records.append({'base_date': base_date, 'item': item, 'target_date': target, 'model': model, 'prediction': prediction})
    return pd.DataFrame(records)
