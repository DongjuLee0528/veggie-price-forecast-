import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.evaluate.short_batch import history_for_item, records, segments
from src.models.compare_all_items import PARAMS
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs

KEYS = ["split", "item", "horizon", "base_date", "target_date"]
NEW = {
    "무": ("volume_only_lgbm", ["low_volume"], "low_activity_lgbm"),
    "양파": ("median_volume_lgbm", ["median_to_mean_7", "low_volume"], "combined_lgbm"),
}
SAVED = (
    ("무", 5, "short_batch/metrics.csv", "price_lgbm_blend"),
    ("배추", 10, "targeted_batch/short_metrics.csv", "variety_mix"),
    ("건고추", 2, "targeted_batch/mid_metrics.csv", "sarimax"),
)


def paired_metrics(metrics, item, horizon, model):
    pair = metrics[
        metrics.item.eq(item) & metrics.horizon.eq(horizon)
        & metrics.model.isin(["archived_current", model])
    ]
    if len(pair) != 4 or set(pair.split.astype(str)) != {"2025", "2026"}:
        raise ValueError(f"Incomplete baseline and candidate metrics: {item} {horizon}")
    old = pair[pair.model.eq("archived_current")]
    new = pair[pair.model.eq(model)]
    rows = old.merge(new, on=["split", "item", "horizon"], suffixes=("_existing", "_candidate"),
                     validate="one_to_one")
    if len(rows) != 2 or not rows.n_scored_existing.eq(rows.n_scored_candidate).all():
        raise ValueError(f"Comparison sample counts differ: {item} {horizon}")
    for metric in ("nmae", "direction_accuracy"):
        rows[f"{metric}_delta_pp"] = 100 * (rows[f"{metric}_candidate"] - rows[f"{metric}_existing"])
    return rows[["split", "item", "horizon", "model_candidate", "n_scored_candidate",
                 "nmae_existing", "nmae_candidate", "nmae_delta_pp",
                 "direction_accuracy_existing", "direction_accuracy_candidate", "direction_accuracy_delta_pp"]]


def main():
    parser = argparse.ArgumentParser(description="Compare candidates with inputs available at inference.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/final_candidates")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error("Use a new output directory")
    results = ROOT / "results"
    batch = pd.read_csv(results / "short_batch/metrics.csv", dtype={"split": str})
    previous = pd.read_parquet(results / "short_batch/predictions.parquet")
    prices = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    summary = [
        paired_metrics(pd.read_csv(results / path, dtype={"split": str}), item, horizon, name)
        for item, horizon, path, name in SAVED
    ]
    forecasts = []
    for item, (name, extra, prior_name) in NEW.items():
        history = history_for_item(item, prices)
        baseline = batch[batch.item.eq(item) & batch.horizon.eq(10) & batch.model.eq("archived_current")]
        reused = previous[previous.item.eq(item) & previous.horizon.eq(10) & previous.model.eq(prior_name)]
        columns = PRICE_FEATURES + ["horizon", *extra]
        trials = []
        for year in (2025, 2026):
            train, _, test = segments(history, year)
            test = test[test.horizon.eq(10)]
            expected = records(test, name, test.price_per_kg, year).merge(
                reused[KEYS + ["actual", "base_price"]], on=KEYS,
                suffixes=("", "_prior"), validate="one_to_one",
            )
            if len(expected) != len(test) or len(test) != len(reused[reused.split.eq(str(year))]):
                raise ValueError(f"Experiment dates differ from previous run: {item}, {year}")
            np.testing.assert_allclose(expected.actual, expected.actual_prior)
            np.testing.assert_allclose(expected.base_price, expected.base_price_prior)
            if not train[extra].notna().any().all() or not test[extra].notna().all().all():
                raise ValueError(f"Unavailable volume inputs: {item}, {year}")
            print(f"{item} {year}: {len(train)} training rows, {len(test)} evaluation rows", flush=True)
            if args.dry_run:
                continue
            model = lgb.LGBMRegressor(**PARAMS, n_estimators=100)
            model.fit(model_inputs(train, columns), train.actual / train.base_price - 1)
            prediction = test.base_price * (1 + model.predict(model_inputs(test, columns)))
            if not np.isfinite(prediction).all() or not prediction.gt(0).all():
                raise ValueError(f"Invalid forecasts: {item}, {year}")
            trials.append(records(test, name, prediction, year))
        if not args.dry_run:
            trial = pd.concat(trials, ignore_index=True)
            forecasts.append(trial)
            summary.append(paired_metrics(pd.concat([baseline, score_predictions(trial)]), item, 10, name))
    if args.dry_run:
        return
    table = pd.concat(summary, ignore_index=True)
    if len(table) != 10 or table.duplicated(["split", "item", "horizon"]).any():
        raise ValueError("Expected five candidates across two evaluation years")
    table["term"] = np.where(table.horizon.eq(2), "mid", "short")
    improved = table.nmae_delta_pp.lt(0) & table.direction_accuracy_delta_pp.gt(0)
    table["improved_both_years"] = improved.groupby([table.item, table.horizon]).transform("all")
    table = table.sort_values(["term", "item", "horizon", "split"])
    args.output.mkdir(parents=True)
    table.to_csv(args.output / "comparison.csv", index=False)
    pd.concat(forecasts, ignore_index=True).to_parquet(args.output / "new_predictions.parquet", index=False)
    print(table[["split", "item", "horizon", "nmae_delta_pp", "direction_accuracy_delta_pp",
                 "improved_both_years"]].round(3).to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
