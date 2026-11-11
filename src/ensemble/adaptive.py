import numpy as np
import pandas as pd


METHODS = ['validation_blend', 'validation_selected', 'last_value', 'moving_average_7']


def recent_weights(rows, origins, window, penalty, temperature):
    actual = rows.actual.to_numpy()
    base = rows.base_price.to_numpy()
    predictions = rows[METHODS].to_numpy()
    targets = pd.DatetimeIndex(rows.target_date)
    if (not targets.is_monotonic_increasing or targets.hasnans
            or not np.isfinite(predictions).all() or not np.isfinite(actual).all()
            or not np.isfinite(base).all() or not (actual > 0).all() or not (base > 0).all()):
        raise ValueError('Adaptive scores require ordered, complete realized predictions')
    hits = np.sign(predictions - base[:, None]) == np.sign(actual - base)[:, None]
    error = np.abs(predictions - actual[:, None]) / actual[:, None]
    score = hits.astype(float) - penalty * error
    known = targets.searchsorted(pd.DatetimeIndex(origins), side='right')
    start = np.maximum(0, known - window)
    counts = (known - start)[:, None]
    cumulative = np.vstack([np.zeros(len(METHODS)), score.cumsum(axis=0)])
    logits = temperature * (cumulative[known] - cumulative[start]) / np.maximum(1, counts)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    weights[known < 20] = [1., 0., 0., 0.]
    return weights


def apply_recent_weights(rows, config):
    if config is None:
        return rows.validation_blend.to_numpy()
    weights = recent_weights(rows, rows.base_date, **config)
    return (weights * rows[METHODS].to_numpy()).sum(axis=1)


def select_recent_config(rows):
    def scores(predicted):
        error = np.mean(np.abs(predicted - rows.actual) / rows.actual)
        direction = np.mean(np.sign(predicted - rows.base_price) == np.sign(rows.actual - rows.base_price))
        return -direction, error

    best = scores(rows.validation_blend.to_numpy())
    error_limit = best[1] * 1.02
    selected = None
    for window in [28, 60, 120, 240]:
        for penalty in [.5, 1., 2.]:
            for temperature in [4., 12., 25.]:
                config = {'window': window, 'penalty': penalty, 'temperature': temperature}
                candidate = scores(apply_recent_weights(rows, config))
                if candidate[1] <= error_limit and candidate < best:
                    best, selected = candidate, config
    return selected
