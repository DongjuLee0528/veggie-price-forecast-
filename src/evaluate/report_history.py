import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions

KEYS = ["item", "horizon", "base_date", "target_date"]


def direction_comparison(archive):
    rows = archive.query("split == 'test' and model in ['current', 'direction_logistic']").copy()
    rows["hit"] = rows.direction.eq(rows.actual_direction).astype(int)
    hits = rows.pivot(index=KEYS, columns="model", values="hit").reset_index()
    if hits[["current", "direction_logistic"]].isna().any().any():
        raise ValueError("Direction methods must share evaluation dates")
    rng = np.random.default_rng(42)
    summary = []
    for year, group in hits.groupby(hits.base_date.dt.year):
        blocks = group.groupby(group.base_date.dt.to_period("M")).agg(
            current=("current", "sum"), logistic=("direction_logistic", "sum"), count=("current", "size")
        )
        draws = rng.integers(len(blocks), size=(5000, len(blocks)))
        totals = blocks.to_numpy()[draws].sum(axis=1)
        bounds = np.quantile((totals[:, 1] - totals[:, 0]) / totals[:, 2], [.025, .975])
        summary.append(dict(year=year, n=len(group), months=len(blocks),
                            current_accuracy=group.current.mean(),
                            logistic_accuracy=group.direction_logistic.mean(),
                            delta=group.direction_logistic.mean() - group.current.mean(),
                            delta_ci_low=bounds[0], delta_ci_high=bounds[1]))
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description="Report archived wholesale price forecast performance by year.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = ROOT / "results"
    archive = pd.read_parquet(results / "all_items_experiment/predictions.parquet")
    rows = archive.query("split == 'test' and model in ['current', 'last_value']").copy()
    if rows.duplicated(["term", "model", *KEYS]).any():
        raise ValueError("Duplicate historical forecasts")
    if not rows.target_date.gt(rows.base_date).all():
        raise ValueError("Historical target must follow its forecast date")
    rows["split"] = rows.base_date.dt.year.astype(str)
    comparison = rows[rows.term.eq("short")].copy()
    common = comparison.groupby(KEYS).model.transform("nunique").eq(2)
    comparison = comparison.loc[common]
    prices = pd.concat([
        score_predictions(comparison).assign(term="short"),
        score_predictions(rows.query("term == 'mid' and model == 'current'")).assign(term="mid"),
    ]).rename(columns={"split": "origin_year"})
    directions = direction_comparison(pd.read_parquet(results / "short_direction_tuning/directions.parquet"))
    args.output.mkdir(parents=True, exist_ok=False)
    rows.loc[rows.model.eq("current")].rename(columns={"split": "origin_year"}).to_csv(
        args.output / "historical_forecasts.csv", index=False)
    prices.to_csv(args.output / "price_metrics.csv", index=False)
    directions.to_csv(args.output / "direction_comparison.csv", index=False)
    print(directions.to_string(index=False))


if __name__ == "__main__":
    main()
