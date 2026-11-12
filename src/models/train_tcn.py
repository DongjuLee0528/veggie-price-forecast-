import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.collect.price import load_prices
from src.evaluate.backtest import ROOT, score_predictions


WINDOW = 30
HORIZON = 10
BLEND_WEIGHT = 0.25
KEYS = ['item', 'base_date', 'target_date']


def price_windows(prices):
    records, windows = [], []
    for item, group in prices.groupby('item', sort=True):
        group = group.sort_values('date')
        values = group.price_per_kg.to_numpy()
        dates = group.date.to_numpy()
        history = np.lib.stride_tricks.sliding_window_view(values, WINDOW)[:-HORIZON]
        origin = np.arange(WINDOW - 1, len(values) - HORIZON)
        base = history[:, -7:].mean(axis=1)
        records.append(pd.DataFrame({'item': item, 'base_date': dates[origin],
            'target_date': dates[origin + HORIZON], 'actual': values[origin + HORIZON],
            'base_price': base}))
        windows.append(np.log(history / base[:, None]))
    return pd.concat(records, ignore_index=True), np.concatenate(windows).astype(np.float32)


def fold_masks(records, cutoff):
    calibration = cutoff - pd.DateOffset(months=6)
    return (
        records.target_date.lt(calibration),
        records.base_date.ge(calibration) & records.target_date.lt(cutoff),
        records.base_date.ge(cutoff) & records.base_date.lt(cutoff + pd.DateOffset(months=6)),
    )


def build_model(item_count, architecture):
    import torch
    from torch import nn

    if architecture == 'nbeats':
        class NBeats(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([
                    nn.Sequential(nn.Linear(WINDOW + item_count, 32), nn.ReLU(),
                                  nn.Linear(32, 32), nn.ReLU(),
                                  nn.Linear(32, WINDOW + 1 if block == 0 else 1))
                    for block in range(2)
                ])

            def forward(self, x, item):
                residual = x.squeeze(1)
                first = self.blocks[0](torch.cat([residual, item], dim=1))
                residual = residual - first[:, :WINDOW]
                second = self.blocks[1](torch.cat([residual, item], dim=1))
                return first[:, -1] + second[:, 0]

        return NBeats()

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            channels = 1
            for dilation in [1, 2, 4, 8]:
                layers.extend([nn.ConstantPad1d((2 * dilation, 0), 0),
                               nn.Conv1d(channels, 16, kernel_size=3, dilation=dilation), nn.ReLU()])
                channels = 16
            self.temporal = nn.Sequential(*layers)
            self.head = nn.Linear(channels + item_count, 1)

        def forward(self, x, item):
            return self.head(torch.cat([self.temporal(x)[:, :, -1], item], dim=1)).squeeze(1)

    return TCN()


def fit_fold(torch, features, items, target, train, valid, epochs, directory, architecture='tcn'):
    from torch import nn

    torch.manual_seed(42)
    model = build_model(items.shape[1], architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    loss_fn = nn.L1Loss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features[train], items[train], target[train]),
        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(42))

    directory.mkdir()
    best = float('inf')
    history = []
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        model.train()
        total = 0.
        for x, item, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x, item), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        model.eval()
        with torch.inference_mode():
            difference = model(features[valid], items[valid]) - target[valid]
            error = torch.expm1(difference).abs().mean().item()
        if not np.isfinite(error):
            raise ValueError('Nonfinite validation loss')
        if error < best:
            best, best_epoch = error, epoch
            best_state = deepcopy(model.state_dict())
            torch.save({'epoch': epoch, 'state_dict': best_state}, directory / 'best.pt')
        if epoch % 5 == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, directory / f'epoch_{epoch:03d}.pt')
        seconds = perf_counter() - start
        history.append({'epoch': epoch, 'train_log_mae': total / len(loader.dataset),
                        'validation_nmae': error, 'seconds': seconds})
        pd.DataFrame(history).to_csv(directory / 'learning_curve.csv', index=False)
        print(f'{directory.name} epoch {epoch}/{epochs}: validation={error:.4f}, '
              f'{seconds:.2f}s, best epoch={best_epoch}', flush=True)
        if epoch - best_epoch >= 5:
            break
    print(f'{directory.name}: median epoch time={np.median([row["seconds"] for row in history]):.2f}s', flush=True)
    model.load_state_dict(best_state)
    return model


def compare(records, reference, output, architecture='tcn'):
    reference = reference.rename(columns={'actual': 'reference_actual', 'base_price': 'reference_base',
                                          'prediction': 'reference_prediction'})
    joined = records.merge(reference[KEYS + ['reference_actual', 'reference_base', 'reference_prediction']],
                           on=KEYS, validate='one_to_one')
    if len(joined) != len(reference) or not np.allclose(joined.actual, joined.reference_actual) or not np.allclose(joined.base_price, joined.reference_base):
        raise ValueError('Neural and reference samples or prices differ')
    columns = KEYS + ['actual', 'base_price', 'split', 'horizon']
    scored = pd.concat([
        joined[columns].assign(model='adaptive', prediction=joined.reference_prediction),
        joined[columns].assign(model=architecture, prediction=joined.prediction),
        joined[columns].assign(model='blend', prediction=(1-BLEND_WEIGHT)*joined.reference_prediction + BLEND_WEIGHT*joined.prediction),
    ], ignore_index=True)
    if not np.isfinite(scored.prediction).all() or not scored.prediction.gt(0).all():
        raise ValueError('Invalid predicted prices')
    metrics = score_predictions(scored)
    scored.to_parquet(output / 'predictions.parquet', index=False)
    metrics.to_csv(output / 'metrics.csv', index=False)
    print(metrics.groupby(['split','model'])[['nmae','direction_accuracy']].mean().to_string())
    print(metrics.groupby('model')[['nmae','direction_accuracy']].mean().to_string())
    print(f'Common samples per method: {len(joined)}')
    if architecture == 'nbeats':
        compare_calibrated(joined, columns, output)


def compare_calibrated(joined, columns, output):
    boundary = pd.Timestamp('2025-07-01')
    calibration = joined[joined.target_date.lt(boundary)]
    evaluation = joined[joined.base_date.ge(boundary)]
    trials = []
    for weight in [0., .25, .5, .75]:
        prediction = (1 - weight) * calibration.reference_prediction + weight * calibration.prediction
        trial = score_predictions(calibration.assign(model='blend', prediction=prediction))
        trials.append({'nbeats_weight': weight, **trial[['nmae', 'direction_accuracy']].mean().to_dict()})
    chosen = max((trial for trial in trials if trial['nmae'] <= trials[0]['nmae']),
                 key=lambda trial: (trial['direction_accuracy'], -trial['nmae'], -trial['nbeats_weight']))
    weight = chosen['nbeats_weight']
    prediction = (1 - weight) * evaluation.reference_prediction + weight * evaluation.prediction
    scored = pd.concat([
        evaluation[columns].assign(model='adaptive', prediction=evaluation.reference_prediction),
        evaluation[columns].assign(model='nbeats', prediction=evaluation.prediction),
        evaluation[columns].assign(model='calibrated_blend', prediction=prediction),
    ], ignore_index=True)
    metrics = score_predictions(scored)
    pd.DataFrame(trials).to_csv(output / 'blend_calibration.csv', index=False)
    metrics.assign(nbeats_weight=weight).to_csv(output / 'calibrated_metrics.csv', index=False)
    print(f'Calibrated N-BEATS weight: {weight:.0%} ({len(calibration)} calibration, {len(evaluation)} evaluation samples)')
    print(metrics.groupby('model')[['nmae','direction_accuracy']].mean().to_string())


def main():
    parser = argparse.ArgumentParser(description='Train a pooled TCN or compact N-BEATS for retrospective 10-trading-day evaluation.')
    parser.add_argument('--architecture', choices=['tcn', 'nbeats'], default='tcn')
    parser.add_argument('--smoke', action='store_true', help='Run three epochs for the first fold and report timing.')
    parser.add_argument('--epochs', type=int, default=30, help='Maximum epochs per fold for full evaluation.')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name != 'torch':
            raise
        parser.exit(1, 'Install PyTorch first: .venv/bin/python -m pip install torch\n')
    torch.set_num_threads(2)
    suffix = 'smoke' if args.smoke else 'experiment'
    output = args.output or ROOT / 'results' / f'{args.architecture}_{suffix}'
    prices = load_prices(ROOT / 'data/interim/price/settlement_daily_clean.parquet')
    records, windows = price_windows(prices)
    reference = pd.read_parquet(ROOT / 'results/lgbm_adaptive/test_predictions.parquet')
    reference = reference[reference.horizon.eq(HORIZON) & reference.model.eq('adaptive')]
    if reference.empty or reference.duplicated(KEYS).any():
        raise ValueError('Expected unique archived adaptive predictions')
    matching = pd.MultiIndex.from_frame(records[KEYS]).isin(pd.MultiIndex.from_frame(reference[KEYS]))
    if matching.sum() != len(reference):
        raise ValueError('Price history does not cover all archived evaluation samples')
    features = torch.from_numpy(windows[:, None, :])
    indicators = pd.get_dummies(records.item, dtype=np.float32)
    items = torch.from_numpy(indicators.to_numpy(copy=True))
    target = torch.from_numpy(np.log(records.actual / records.base_price).to_numpy(dtype=np.float32, copy=True))
    cutoffs = pd.date_range('2025-01-01', '2026-07-01', freq='6MS')
    if args.smoke:
        cutoffs = cutoffs[:1]
    output.mkdir(parents=True, exist_ok=False)
    pd.Series(indicators.columns).to_csv(output / 'item_order.csv', index=False, header=['item'])
    predictions = []
    for cutoff in cutoffs:
        train, valid, test = [mask.to_numpy(copy=True) for mask in fold_masks(records, cutoff)]
        test = test & matching
        if not (train.any() and valid.any() and test.any()) or np.any(train & valid):
            raise ValueError(f'Invalid time split for {cutoff.date()}')
        split = cutoff.strftime('%Y-%m')
        print(f'{split}: train={train.sum()}, validation={valid.sum()}, evaluation={test.sum()}', flush=True)
        predict = fit_fold(torch, features, items, target, train, valid,
                           3 if args.smoke else args.epochs, output / split, args.architecture)
        if not args.smoke:
            with torch.inference_mode():
                ratio = predict(features[test], items[test]).exp().numpy()
            rows = records.loc[test].copy()
            rows['prediction'] = rows.base_price * ratio
            predictions.append(rows.assign(split=split, horizon=HORIZON))
    if predictions:
        compare(pd.concat(predictions, ignore_index=True), reference, output, args.architecture)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
