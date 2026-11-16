import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate.accuracy_followup import hits
from src.evaluate.backtest import ROOT

KEYS = ["item", "horizon", "base_date", "target_date"]


def metrics(rows, horizon):
    a = (rows.final / rows.actual - 1).abs()
    b = (rows.last_value / rows.actual - 1).abs()
    hit_final = hits(rows, "final")
    hit_last = hits(rows, "last_value")
    blocks = pd.DataFrame({
        "week": rows.base_date.dt.to_period("W-SUN"),
        "error_delta": a - b,
        "direction_delta": hit_final.astype(int) - hit_last.astype(int),
    }).groupby("week").agg(error=("error_delta", "sum"), direction=("direction_delta", "sum"),
                          n=("direction_delta", "size")).to_numpy()
    draw = np.random.default_rng(42).integers(len(blocks), size=(5000, len(blocks)))
    totals = blocks[draw].sum(axis=1)
    ci = np.quantile(100 * totals[:, :2] / totals[:, 2:], [.025, .975], axis=0)
    return dict(horizon=horizon, n=len(rows), final_error_pct=100 * a.mean(),
                last_error_pct=100 * b.mean(),
                average_7_error_pct=100 * (rows.moving_average_7 / rows.actual - 1).abs().mean(),
                error_delta_pp=100 * (a - b).mean(), error_ci_low_pp=ci[0, 0], error_ci_high_pp=ci[1, 0],
                final_direction_pct=100 * hit_final.mean(), last_direction_pct=100 * hit_last.mean(),
                direction_delta_pp=100 * (hit_final.mean() - hit_last.mean()),
                direction_ci_low_pp=ci[0, 1], direction_ci_high_pp=ci[1, 1])


def main():
    parser = argparse.ArgumentParser(description="Export the fixed presentation evaluation.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/presentation_accuracy")
    args = parser.parse_args()
    root = ROOT / "results"
    rows = pd.read_parquet(root / "accuracy_followup/presentation_predictions.parquet")
    cabbage = pd.read_parquet(root / "targeted_batch/short_predictions.parquet").query(
        "item == '배추' and horizon == 10 and model == 'variety_mix'")
    restored = rows.query("item == '배추' and horizon == 10").merge(
        cabbage[KEYS + ["prediction"]], on=KEYS, validate="one_to_one")
    if len(restored) != 523 or rows.duplicated(KEYS).any():
        raise ValueError("Missing or duplicate historical evaluation rows")
    np.testing.assert_allclose(restored.final, restored.prediction)
    if (not rows.target_date.gt(rows.base_date).all() or not rows.target_date.dt.year.eq(rows.year).all()
            or rows.target_date.max() > pd.Timestamp("2026-09-30")):
        raise ValueError("Unexpected retrospective boundary")
    table = pd.DataFrame([
        dict(year=year, **metrics(group, horizon))
        for year, annual in rows.groupby("year")
        for horizon, group in [(0, annual), *list(annual.groupby("horizon"))]
    ])
    table = table.sort_values(["horizon", "year"])
    disagreement = rows[rows.horizon.isin([5, 10])].copy()
    disagreement["final_hit"] = hits(disagreement, "final")
    disagreement["last_hit"] = hits(disagreement, "last_value")
    disagreement = disagreement[
        np.sign(disagreement.final - disagreement.base_price).ne(
            np.sign(disagreement.last_value - disagreement.base_price))]
    mismatch = disagreement.groupby("year").agg(n=("final_hit", "size"),
                                                 final_accuracy=("final_hit", "mean"),
                                                 last_accuracy=("last_hit", "mean"))
    args.output.mkdir(parents=True, exist_ok=False)
    table.to_csv(args.output / "metrics.csv", index=False)
    mismatch.to_csv(args.output / "direction_disagreements.csv")
    print(table.round(3).to_string(index=False))
    print(mismatch.to_string())


if __name__ == "__main__":
    main()
