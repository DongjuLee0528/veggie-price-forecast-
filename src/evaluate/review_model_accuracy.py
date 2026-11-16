import argparse
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT, score_predictions
from src.evaluate.short_batch import WEIGHTS, confidence_rows, history_for_item, records, segments
from src.models.classify_short import classify_models
from src.models.compare_all_items import ITEMS, PARAMS
from src.features.targets import dekad_bounds, dekad_index
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_mid import seasonal_terms
from src.models.tune_mid import ets_predictions
from src.models.train_lgbm import model_inputs

HORIZONS = (1, 5, 10)
KEYS = ["item", "horizon", "base_date", "target_date"]
EXTRA = ["market_index", "sunday", "low_count", "low_volume", "median_to_mean_7"]


def daily_sources():
    daily = pd.concat([
        pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet"),
        pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet"),
    ])
    daily = daily[daily.item.isin(ITEMS)]
    raw = pd.read_parquet(ROOT / "data/raw/price/settlement_rows.parquet")
    market = raw.assign(kg=raw.unit_qty * raw.kg_per_unit, won=raw.unit_qty * raw.price_per_unit)
    market = market.groupby(["item", "date", "market_code"])[["kg", "won"]].sum()
    external = pd.read_parquet(ROOT / "data/processed/features.parquet").drop(
        columns=["item_type", *PRICE_FEATURES])
    return daily, market, external


def market_feature(market, calibration_start):
    historical = market.loc[market.index < calibration_start]
    coverage = historical.notna().mean().sort_values(ascending=False)
    selected = coverage[coverage.ge(.6)].head(5).index
    if len(selected) < 2:
        return pd.Series(np.nan, index=market.index)
    selected_prices = market[selected]
    scales = historical[selected].median()
    index = selected_prices.div(scales).mean(axis=1, skipna=False)
    return index / index.shift().rolling(7, min_periods=3).median() - 1


def levels(daily, noise_anchor, weighted):
    observations = np.log(daily.price_per_kg.to_numpy())
    if weighted:
        relative = (np.sqrt(daily.volume_kg / noise_anchor[0] * daily.n_rows / noise_anchor[1])
                    .clip(.1, 10).to_numpy())
    else:
        relative = np.ones(len(daily))
    state, variance = observations[0], 1.
    output = []
    for value, activity in zip(observations, relative):
        variance += .05
        noise = 1 / activity
        gain = variance / (variance + noise)
        state += gain * (value - state)
        variance *= 1 - gain
        output.append(state)
    return pd.Series(np.exp(output), index=daily.date)


def fit_price(train, sample, columns, normalized=False, quantile=None):
    params = PARAMS if quantile is None else {**PARAMS, "objective": "quantile", "alpha": quantile}
    model = lgb.LGBMRegressor(**params, n_estimators=100)
    scale = train.vol_28.clip(lower=.01).fillna(1) if normalized else 1
    model.fit(model_inputs(train, columns), (train.actual / train.base_price - 1) / scale)
    scale = sample.vol_28.clip(lower=.01).fillna(1) if normalized else 1
    return sample.base_price * (1 + model.predict(model_inputs(sample, columns)) * scale)


def interval_rows(valid, test, method, year, term):
    output = []
    for (item, horizon), calibration in valid.groupby(["item", "horizon"]):
        evaluation = test[test.item.eq(item) & test.horizon.eq(horizon)]
        residual = np.sort(np.abs(np.log(calibration.actual / calibration.prediction)))
        rank = min(len(residual), int(np.ceil((len(residual) + 1) * .8)))
        radius = residual[rank - 1]
        bounds = evaluation.prediction * np.exp(radius) - evaluation.prediction * np.exp(-radius)
        hit = np.abs(np.log(evaluation.actual / evaluation.prediction)).le(radius)
        output.append(dict(term=term, year=year, item=item, horizon=horizon, model=method,
                           calibration_n=len(residual), test_n=len(evaluation),
                           coverage=hit.mean(), mean_width_krw_per_kg=bounds.mean()))
    return output


def short_review(daily, market, external):
    histories = {item: history_for_item(item, daily) for item in sorted(ITEMS)}
    price_columns = PRICE_FEATURES + ["horizon"]
    price_rows, direction_rows, confidence, intervals = [], [], [], []
    for year in (2023, 2024, 2025, 2026):
        fold = {}
        for item, history in histories.items():
            rows = history.copy()
            rows["sunday"] = rows.date.dt.day_of_week.eq(6).astype(int)
            if item in market.index.get_level_values("item"):
                by_market = market.loc[item]
                wide = (by_market.won / by_market.kg).unstack()
                rows["market_index"] = rows.date.map(market_feature(wide, pd.Timestamp(year - 1, 1, 1)))
            else:
                rows["market_index"] = np.nan
            rows = rows.merge(external, on=["item", "date"], how="left", validate="many_to_one")
            fold[item] = segments(rows, year)
        pooled_train = pd.concat([triplet[0] for triplet in fold.values()])
        pooled_sample = pd.concat([part for _, valid, test in fold.values() for part in (valid, test)])
        pooled_predictions = fit_price(pooled_train, pooled_sample, price_columns + ["item"], normalized=True)
        pooled_sample = pooled_sample.assign(pooled_prediction=pooled_predictions.to_numpy())
        for item, (train, valid, test) in fold.items():
            sample = pd.concat([valid, test], ignore_index=True)
            ends = len(valid)
            days = histories[item].sort_values("date").drop_duplicates("date")
            noise_anchor = train[["volume_kg", "n_rows"]].median().to_numpy()
            variants = {
                "price_lgbm": price_columns,
                "robust_inputs_lgbm": price_columns + EXTRA,
                "sunday_flag_lgbm": price_columns + ["sunday"],
            }
            if item != "감자 수미":
                variants["external_inputs_lgbm"] = price_columns + [
                    column for column in external.columns if column not in ("date", "item")
                ]
                variants["fixed_market_lgbm"] = price_columns + ["market_index"]
            forecasts = {"last_value": sample.price_per_kg.to_numpy(),
                         "moving_average_7": sample.base_price.to_numpy()}
            for name, columns in variants.items():
                forecasts[name] = fit_price(train, sample, columns).to_numpy()
            clean = train.loc[train.sunday.eq(0) & train.low_count.eq(0) & train.low_volume.eq(0)]
            forecasts["filtered_training_lgbm"] = fit_price(clean, sample, price_columns).to_numpy()
            quantiles = np.sort(np.stack([
                fit_price(train, sample, price_columns, quantile=q).to_numpy() for q in (.1, .9)
            ]), axis=0)
            for weighted in (False, True):
                level = levels(days, noise_anchor, weighted)
                forecasts["kalman_volume_count" if weighted else "kalman_equal"] = sample.date.map(level).to_numpy()
            pooled = pooled_sample.loc[pooled_sample.item.eq(item)].set_index(["date", "horizon"])
            forecasts["pooled_normalized_lgbm"] = pooled.pooled_prediction.reindex(
                pd.MultiIndex.from_frame(sample[["date", "horizon"]])).to_numpy()
            for horizon in HORIZONS:
                v, t = valid.horizon.eq(horizon), test.horizon.eq(horizon)
                vr, tr = valid.loc[v], test.loc[t]
                labels = np.sign(train.actual - train.base_price).astype(int)
                subset = train.horizon.eq(horizon)
                logistic = classify_models(False)["direction_logistic"]
                logistic.fit(model_inputs(train.loc[subset], PRICE_FEATURES), labels[subset])
                actual_direction = np.sign(tr.actual - tr.base_price)
                for name, direction in {
                    "logistic": logistic.predict(model_inputs(tr, PRICE_FEATURES)),
                    "majority": np.repeat(np.sign(vr.actual - vr.base_price).mode().iloc[0], len(tr)),
                }.items():
                    direction_rows.append(records(tr, name, tr.price_per_kg, year).drop(columns="prediction")
                                          .assign(hit=np.asarray(direction) == actual_direction.to_numpy()))
                options = {}
                for method, prediction in forecasts.items():
                    vp, tp = prediction[:ends][v.to_numpy()], prediction[ends:][t.to_numpy()]
                    price_rows.append(records(tr, method, tp, year))
                    options[method] = (vp, tp)
                vp, tp = options["robust_inputs_lgbm"]
                weight = min(WEIGHTS, key=lambda w: np.mean(np.abs(
                    ((1 - w) * vp + w * vr.price_per_kg) / vr.actual - 1)))
                blended_v = records(vr, "robust_blend", (1 - weight) * vp + weight * vr.price_per_kg, year)
                blended_t = records(tr, "robust_blend", (1 - weight) * tp + weight * tr.price_per_kg, year)
                price_rows.append(blended_t)
                confidence.extend(confidence_rows(blended_v, blended_t, "robust_blend", year))
                intervals.extend(interval_rows(blended_v, blended_t, "robust_blend", year, "short"))
                lower, upper = quantiles[:, :ends][:, v.to_numpy()]
                residual = np.maximum.reduce([np.zeros(len(vr)), lower - vr.actual, vr.actual - upper])
                rank = min(len(residual), int(np.ceil((len(residual) + 1) * .8)))
                padding = np.sort(residual)[rank - 1]
                lower, upper = quantiles[:, ends:][:, t.to_numpy()]
                lower = np.maximum(lower - padding, 0)
                upper = upper + padding
                intervals.append(dict(term="short", year=year, item=item, horizon=horizon,
                                      model="calibrated_quantile_lgbm", calibration_n=len(vr), test_n=len(tr),
                                      coverage=tr.actual.between(lower, upper).mean(),
                                      mean_width_krw_per_kg=np.mean(upper - lower)))
            print(f"{year} {item}: {len(test)} forecasts per method", flush=True)
    rows = pd.concat(price_rows, ignore_index=True)
    if not np.isfinite(rows.prediction).all() or not rows.prediction.gt(0).all():
        raise ValueError("Nonpositive or missing forecast")
    directions = pd.concat(direction_rows, ignore_index=True)
    return rows, directions, pd.DataFrame(confidence), pd.DataFrame(intervals)


def portfolio():
    archive = pd.read_parquet(ROOT / "results/all_items_experiment/predictions.parquet")
    archive = archive.query("split == 'test' and model == 'current'").copy()
    archive["split"] = archive.base_date.dt.year
    archive = archive[archive.target_date.lt("2026-10-01") & archive.target_date.dt.year.eq(archive.split)]
    selected = pd.read_csv(ROOT / "results/final_candidates/comparison.csv")
    old = score_predictions(archive).merge(
        archive[["split", "item", "horizon", "term"]].drop_duplicates(),
        on=["split", "item", "horizon"], validate="one_to_one")
    pair = old.merge(selected, on=["split", "term", "item", "horizon"], how="left", validate="one_to_one")
    matched = pair.n_scored_candidate.notna()
    np.testing.assert_array_equal(pair.n_scored[matched], pair.n_scored_candidate[matched])
    for metric in ("nmae", "direction_accuracy"):
        np.testing.assert_allclose(pair.loc[matched, metric], pair.loc[matched, f"{metric}_existing"])
        pair[f"{metric}_proposed"] = pair[f"{metric}_candidate"].fillna(pair[metric])
    summary = []
    for (year, term), group in pair.groupby(["split", "term"]):
        summary.append(dict(year=year, term=term, count=group.n_scored.sum(),
                            **{col: np.average(group[col], weights=group.n_scored) for col in
                               ("nmae", "nmae_proposed", "direction_accuracy", "direction_accuracy_proposed")}))
    return pd.DataFrame(summary)


def diagnostics(daily, market):
    oracle, signals = [], []
    for item, days in daily.sort_values("date").groupby("item"):
        series = days.price_per_kg.reset_index(drop=True)
        log_return = np.log(series).diff()
        d = days.reset_index(drop=True)
        median = series.rolling(15, min_periods=7).median()
        sunday = d.date.dt.day_of_week.eq(6)
        r = dict(item=item, return_std=log_return.std(), lag1=log_return.autocorr(),
                 median_trades=d.n_rows.median(), median_kg=d.volume_kg.median(),
                 sunday_days=int(sunday.sum()), sunday_median_relative_price=(series[sunday] / median[sunday] - 1).median(),
                 price_volume_correlation=(series / median - 1).corr(
                     d.volume_kg / d.volume_kg.rolling(15, min_periods=7).median() - 1))
        if item in market.index.get_level_values("item"):
            group = market.loc[item]
            wide = (group.won / group.kg).unstack()
            index = market_feature(wide, pd.Timestamp("2025-01-01"))
            r["market_feature_observed"] = index.notna().sum()
            fixed = np.log1p(index).diff()
            r["market_relative_change_lag1"] = fixed.autocorr()
        signals.append(r)
        base = series.rolling(7).mean()
        low = d.n_rows.lt(d.n_rows.shift().rolling(28).median() * .5)
        for year in (2025, 2026):
            for horizon in (5, 10):
                target = series.shift(-horizon)
                target_date = d.date.shift(-horizon)
                for window in (5, 9):
                    future_median = series.rolling(window, center=True).median().shift(-horizon)
                    ok = d.date.dt.year.eq(year) & target_date.dt.year.eq(year) & base.notna() & future_median.notna()
                    for name, actual, usable in [
                        ("raw_daily", target, ok),
                        ("without_sunday_or_low_count", target, ok & ~sunday.shift(-horizon, fill_value=True)
                         & ~low.shift(-horizon, fill_value=True)),
                        ("centered_three_day_target", series.rolling(3, center=True).mean().shift(-horizon), ok),
                    ]:
                        direction = np.sign(actual[usable] - base[usable])
                        hit = np.sign(future_median[usable] - base[usable]).eq(direction)
                        oracle.append(dict(item=item, year=year, horizon=horizon, window=window,
                                           target_definition=name, samples=len(hit), diagnostic_accuracy=hit.mean()))
    return pd.DataFrame(signals), pd.DataFrame(oracle)


def mid_intervals():
    results = ROOT / "results"
    nonpotato = pd.read_parquet(results / "mid_tuning/predictions.parquet").query("model == 'blend_with_ets'")
    potato = pd.read_parquet(results / "potato_sumi_tuning/predictions.parquet").query(
        "term == 'mid' and model == 'holt_winters'")
    frames = pd.concat([nonpotato, potato], ignore_index=True)
    output = []
    for year in (2025, 2026):
        valid = frames[frames.base_date.dt.year.eq(year - 1) & frames.target_date.lt(f"{year}-01-01")]
        test = frames[frames.base_date.dt.year.eq(year) & frames.target_date.lt(
            min(f"{year + 1}-01-01", "2026-10-01"))]
        output.extend(interval_rows(valid, test, "archived_mid", year, "mid"))
    pepper = frames.loc[frames.item.eq("건고추") & frames.split.eq("test")].copy()
    pepper = pepper.assign(split=np.where(pepper.target_date.dt.month.isin([8, 9, 10]), "harvest", "other"))
    ref = pd.read_parquet(results / "lgbm_mid_comparison/predictions.parquet").query(
        "item == '건고추' and split == 'test' and model in ['last_completed', 'sarimax']")
    ref = ref.merge(pepper[KEYS + ["split"]], on=KEYS, suffixes=("_archive", ""), validate="many_to_one")
    return pd.DataFrame(output), score_predictions(pd.concat([pepper, ref]))


def block_comparison(rows):
    subset = rows[rows.model.isin(["price_lgbm", "robust_blend"])].copy()
    subset["hit"] = np.sign(subset.prediction - subset.base_price).eq(
        np.sign(subset.actual - subset.base_price)).astype(int)
    subset["error"] = (subset.prediction / subset.actual - 1).abs()
    a = subset.pivot(index=["split", *KEYS], columns="model", values=["hit", "error"])
    if a.isna().any().any():
        raise ValueError("Incomplete paired comparisons")
    b = pd.DataFrame({metric: a[metric].robust_blend - a[metric].price_lgbm for metric in ("hit", "error")}).reset_index()
    rng = np.random.default_rng(42)
    output = []
    for year, group in b.groupby("split"):
        months = group.groupby(group.base_date.dt.to_period("M")).agg(
            hit=("hit", "sum"), error=("error", "sum"), count=("hit", "size"))
        totals = months.to_numpy()[rng.integers(len(months), size=(5000, len(months)))].sum(axis=1)
        limits = np.quantile(totals[:, :2] / totals[:, 2:], [.025, .975], axis=0)
        output.append(dict(year=year, count=len(group), months=len(months), direction_delta=group.hit.mean(),
                           error_delta=group.error.mean(), direction_low=limits[0, 0],
                           direction_high=limits[1, 0], error_low=limits[0, 1], error_high=limits[1, 1]))
    return pd.DataFrame(output)


def early_mid_review(daily):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    outputs = []
    weights = [np.array(w) / 4 for w in product(range(5), repeat=3) if sum(w) == 4]
    columns = PRICE_FEATURES + ["horizon"]
    for item in sorted(ITEMS):
        history = history_for_item(item, daily).query("horizon == 1").sort_values("date").copy()
        original = daily[daily.item.eq(item)].sort_values("date").copy()
        means = original.groupby(original.date.map(dekad_index)).price_per_kg.mean()
        history["origin_dekad"] = history.date.map(dekad_index)
        history = history.drop_duplicates("origin_dekad", keep="last")
        rows = pd.concat([history.assign(horizon=h, target_dekad=history.origin_dekad + h)
                          for h in (2, 3)], ignore_index=True)
        rows["target_date"] = rows.target_dekad.map(lambda value: dekad_bounds(value)[1])
        rows["actual"] = rows.target_dekad.map(means)
        rows = rows.dropna(subset=["actual"])
        first_period = means.index.min() + 1
        for year in (2023, 2024):
            validation_start = pd.Timestamp(year - 1, 1, 1)
            test_start = pd.Timestamp(year, 1, 1)
            test_end = pd.Timestamp(year + 1, 1, 1)
            embargo = dekad_bounds(dekad_index(validation_start) - 3)[0]
            train = rows[rows.target_date.lt(embargo)]
            sample = rows[rows.date.ge(validation_start) & rows.target_date.lt(test_end)].copy()
            sample = sample[sample.date.ge(test_start) | sample.target_date.lt(test_start)]
            sample["completed"] = sample.origin_dekad - [
                int(dekad_bounds(period)[1] > date) for period, date in zip(sample.origin_dekad, sample.date)
            ]
            sample = sample[sample.completed.ge(first_period + 71)].sort_values("date")
            sample["price_lgbm"] = fit_price(train, sample, columns)
            sample["ets"] = ets_predictions(sample.assign(base_date=sample.date), means, original.date.min())
            last = means.loc[:dekad_index(validation_start) - 1].index.max()
            observed = means.loc[first_period:last]
            model = SARIMAX(np.log(observed.to_numpy()), exog=seasonal_terms(observed.index),
                            order=(1, 0, 0), trend="c")
            state = model.fit(disp=False, maxiter=150)
            if not state.mle_retvals["converged"]:
                state = model.fit(method="powell", disp=False, maxiter=200)
            if not state.mle_retvals["converged"]:
                raise ValueError(f"Early mid-term SARIMAX failed: {item}, {year}")
            for row in sample.itertuples():
                if row.completed > last:
                    new = np.arange(last + 1, row.completed + 1)
                    state = state.extend(np.log(means.loc[new].to_numpy()), exog=seasonal_terms(new))
                    last = row.completed
                periods = np.arange(last + 1, row.target_dekad + 1)
                sample.loc[row.Index, "sarimax"] = np.exp(
                    state.get_forecast(steps=len(periods), exog=seasonal_terms(periods)).predicted_mean[-1])
            sample["last_completed"] = sample.completed.map(means)
            methods = ["price_lgbm", "sarimax", "ets"]
            for horizon, group in sample.groupby("horizon"):
                valid, test = group[group.date.lt(test_start)], group[group.date.ge(test_start)]
                weight = min(weights, key=lambda w: np.mean(np.abs(valid[methods].to_numpy() @ w / valid.actual - 1)))
                outputs.extend(records(test, name, test[name], year) for name in methods + ["last_completed"])
                outputs.append(records(test, "tuned_blend", test[methods].to_numpy() @ weight, year)
                               .assign(lgbm_weight=weight[0], sarimax_weight=weight[1], ets_weight=weight[2]))
            print(f"{year} {item}: early mid-term comparison complete", flush=True)
    rows = pd.concat(outputs, ignore_index=True)
    if not np.isfinite(rows.prediction).all() or not rows.prediction.gt(0).all():
        raise ValueError("Invalid early mid-term forecasts")
    return score_predictions(rows)


def archive_inventory():
    frames = []
    for path in sorted((ROOT / "results").glob("*/metrics*.csv")):
        if path.parent.name == "model_accuracy_review":
            continue
        if path.name not in {"metrics.csv", "metrics_by_year.csv", "metrics_by_item_horizon.csv"}:
            continue
        frame = pd.read_csv(path)
        if "model" in frame and {"nmae", "direction_accuracy"}.intersection(frame.columns):
            frames.append(frame.assign(experiment=path.parent.name, metric_file=path.name))
    return pd.concat(frames, ignore_index=True)


def monthly_drift(rows, daily):
    r = rows[rows.model.eq("robust_blend")].copy()
    r["month"] = r.base_date.dt.to_period("M")
    r["hit"] = np.sign(r.prediction - r.base_price).eq(np.sign(r.actual - r.base_price))
    r["abs_change"] = (r.actual / r.base_price - 1).abs()
    stats = r.groupby(["item", "month"]).agg(samples=("hit", "size"), direction_accuracy=("hit", "mean"),
                                                large_move_pct=("abs_change", lambda x: x.gt(.10).mean()))
    traded = daily.sort_values("date").copy()
    traded["month"] = traded.date.dt.to_period("M")
    traded["log_return"] = traded.groupby("item").price_per_kg.transform(np.log).groupby(traded.item).diff()
    volume = traded.groupby(["item", "month"]).agg(median_trades=("n_rows", "median"),
                                                   log_return_std=("log_return", "std"))
    return stats.join(volume).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Audit existing models and evaluate remaining accuracy experiments.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/model_accuracy_review")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    daily, market, external = daily_sources()
    print("Comparing selected routes with the archived portfolio", flush=True)
    combined = portfolio()
    print(combined.to_string(index=False), flush=True)
    rows, directions, confidence, intervals = short_review(daily, market, external)
    signals, oracle = diagnostics(daily, market)
    mid_bounds, harvest = mid_intervals()
    early_mid = early_mid_review(daily)
    args.output.mkdir(parents=True)
    tables = {
        "portfolio": combined,
        "early_mid_metrics": early_mid,
        "archived_model_metrics": archive_inventory(),
        "monthly_drift": monthly_drift(rows, daily),
        "short_metrics": score_predictions(rows),
        "direction_metrics": directions.groupby(["split", "item", "horizon", "model"]).hit.agg(
            n="size", direction_accuracy="mean").reset_index(),
        "confidence": confidence,
        "intervals": pd.concat([intervals, mid_bounds]),
        "data_diagnostics": signals,
        "oracle_diagnostics": oracle,
        "pepper_mid_season": harvest,
        "paired_monthly_intervals": block_comparison(rows),
    }
    for name, frame in tables.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    rows.to_parquet(args.output / "short_predictions.parquet", index=False)
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
