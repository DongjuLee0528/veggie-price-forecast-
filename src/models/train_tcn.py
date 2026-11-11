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


def fit_fold(torch, features, items, target, train, valid, epochs, directory):
    from torch import nn

    torch.manual_seed(42)
    layers = []
    channels = 1
    for dilation in [1, 2, 4, 8]:
        layers.extend([nn.ConstantPad1d((2 * dilation, 0), 0),
                       nn.Conv1d(channels, 16, kernel_size=3, dilation=dilation), nn.ReLU()])
        channels = 16
    temporal = nn.Sequential(*layers)
    head = nn.Linear(channels + items.shape[1], 1)
    model = nn.ModuleDict({'temporal': temporal, 'head': head})
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    loss_fn = nn.L1Loss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features[train], items[train], target[train]),
        batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(42))

    def forward(x, item):
        return head(torch.cat([temporal(x)[:, :, -1], item], dim=1)).squeeze(1)

    directory.mkdir()
    best = float('inf')
    history = []
    for epoch in range(1, epochs + 1):
        start = perf_counter()
        model.train()
        total = 0.
        for x, item, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(forward(x, item), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        model.eval()
        with torch.inference_mode():
            difference = forward(features[valid], items[valid]) - target[valid]
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
    return forward


def compare(records, reference, output):
    reference = reference.rename(columns={'actual': 'reference_actual', 'base_price': 'reference_base',
                                          'prediction': 'reference_prediction'})
    joined = records.merge(reference[KEYS + ['reference_actual', 'reference_base', 'reference_prediction']],
                           on=KEYS, validate='one_to_one')
    if len(joined) != len(reference) or not np.allclose(joined.actual, joined.reference_actual) or not np.allclose(joined.base_price, joined.reference_base):
        raise ValueError('TCN and reference samples or prices differ')
    columns = KEYS + ['actual', 'base_price', 'split', 'horizon']
    scored = pd.concat([
        joined[columns].assign(model='adaptive', prediction=joined.reference_prediction),
        joined[columns].assign(model='tcn', prediction=joined.prediction),
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


def main():
    parser = argparse.ArgumentParser(description='Train a small pooled TCN for retrospective 10-trading-day price evaluation.')
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
    output = args.output or ROOT / 'results' / ('tcn_smoke' if args.smoke else 'tcn_experiment')
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
    items = torch.from_numpy(indicators.to_numpy())
    target = torch.from_numpy(np.log(records.actual / records.base_price).to_numpy(dtype=np.float32))
    cutoffs = pd.date_range('2025-01-01', '2026-07-01', freq='6MS')
    if args.smoke:
        cutoffs = cutoffs[:1]
    output.mkdir(parents=True, exist_ok=False)
    pd.Series(indicators.columns).to_csv(output / 'item_order.csv', index=False, header=['item'])
    predictions = []
    for cutoff in cutoffs:
        train, valid, test = [mask.to_numpy() for mask in fold_masks(records, cutoff)]
        test &= matching
        if not (train.any() and valid.any() and test.any()) or np.any(train & valid):
            raise ValueError(f'Invalid time split for {cutoff.date()}')
        split = cutoff.strftime('%Y-%m')
        print(f'{split}: train={train.sum()}, validation={valid.sum()}, evaluation={test.sum()}', flush=True)
        predict = fit_fold(torch, features, items, target, train, valid,
                           3 if args.smoke else args.epochs, output / split)
        if not args.smoke:
            with torch.inference_mode():
                ratio = predict(features[test], items[test]).exp().numpy()
            rows = records.loc[test].copy()
            rows['prediction'] = rows.base_price * ratio
            predictions.append(rows.assign(split=split, horizon=HORIZON))
    if predictions:
        compare(pd.concat(predictions, ignore_index=True), reference, output)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
