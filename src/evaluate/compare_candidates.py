import argparse
import hashlib
import json
import runpy
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.backtest import ROOT
from src.evaluate.short_batch import history_for_item
from src.evaluate.targeted_batch import SEASONS, variety_shares
from src.features.targets import dekad_index
from src.models.compare_all_items import PARAMS
from src.models.train_experiments import PRICE_FEATURES
from src.models.train_lgbm import model_inputs
from src.models.train_mid import series_predictions

ROUTES = [
    ("무", 5, "price_lgbm_blend", []),
    ("무", 10, "volume_only_lgbm", ["low_volume"]),
    ("양파", 10, "median_volume_lgbm", ["median_to_mean_7", "low_volume"]),
    ("배추", 10, "variety_mix", list(SEASONS)),
]
CUTOFF = pd.Timestamp("2026-09-30")


def origin_shares(cache, item):
    classify = runpy.run_path(str(ROOT / "scripts/filter_settlement.py"))["classify"]
    markets = json.loads(cache.read_text())
    wanted = [row for rows in markets.values() for row in rows if row["PDLT_NM"] == item]
    frame = pd.DataFrame(wanted)
    frame.loc[frame.api_variety_code.eq("100105") & frame.SPCS_NM.eq(""), "SPCS_NM"] = "고냉지배추"
    frame = frame[classify(frame).eq(item)]
    kg = pd.to_numeric(frame.api_kg.str.replace(",", ""), errors="raise")
    won = pd.to_numeric(frame.api_won.str.replace(",", ""), errors="raise")
    eligible = kg.gt(0) & won.gt(0)
    if not eligible.any() or not frame.unit[eligible].eq("12").all() or frame.SPCS_NM[eligible].eq("").any():
        raise ValueError("Cannot reproduce origin variety volume")
    return {name: kg[eligible & frame.SPCS_NM.str.contains(pattern, na=False)].sum() / kg[eligible].sum()
            for name, pattern in SEASONS.items()}


def main():
    parser = argparse.ArgumentParser(description="Fit and publish experimental candidates beside saved forecasts.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/final_candidate_forecasts_20261113")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.dry_run:
        parser.error("Use a new output directory")
    output = ROOT / "outputs"
    verification = output / "wholesale_verify"
    inputs = verification / "forecast_inputs_20261113"
    prices = pd.read_parquet(inputs / "prices.parquet")
    current = pd.read_parquet(inputs / "price_features.parquet")
    origin, = current.date.unique()
    origin = pd.Timestamp(origin)
    short = pd.read_csv(output / "short_forecast_20261113/predictions.csv")
    mid = pd.read_csv(output / "mid_forecast_20261113/predictions.csv")
    if not all(pd.to_datetime(frame.base_date).eq(origin).all() for frame in (short, mid)):
        raise ValueError("Forecast origins do not match")
    official = prices[prices.date.le(CUTOFF)]
    shares = variety_shares()
    weights = pd.read_csv(ROOT / "results/short_batch/blend_weights.csv")
    weight, = weights.query("year == 2026 and item == '무' and horizon == 5 and model == 'price_lgbm'").last_value_weight
    models, forecasts = {}, []
    for item, horizon, name, extra in ROUTES:
        train = history_for_item(item, official)
        row = current[current.item.eq(item)].assign(horizon=horizon)
        item_history = prices.loc[prices.item.eq(item) & prices.date.le(origin)].sort_values("date")
        latest = item_history.iloc[-1]
        if latest.date != origin:
            raise ValueError(f"Origin price missing: {item}")
        if name == "variety_mix":
            train = train.merge(shares, on=["date", "item"], validate="many_to_one")
            row = row.assign(**origin_shares(verification / "mafra_settlement" / f"markets_{origin:%Y%m%d}.json", item))
        if "median_to_mean_7" in extra:
            row["median_to_mean_7"] = item_history.price_per_kg.tail(7).median() / row.base_price
        if "low_volume" in extra:
            recent_volume = item_history.volume_kg.tail(29)
            if len(recent_volume) != 29 or not recent_volume.gt(0).all():
                raise ValueError(f"Incomplete recent trade volumes: {item}")
            row["low_volume"] = float(recent_volume.iloc[-1] < .5 * recent_volume.iloc[:-1].median())
        columns = PRICE_FEATURES + ["horizon", *extra]
        x = model_inputs(row, columns)
        unavailable = x.columns[x.isna().any()].tolist()
        base, = short.loc[short.item.eq(item) & short.horizon.eq(horizon)].to_dict("records")
        print(f"{item} {horizon}: {len(train)} training rows; unavailable={unavailable}", flush=True)
        if args.dry_run:
            continue
        fit = lgb.LGBMRegressor(**PARAMS, n_estimators=100)
        fit.fit(model_inputs(train, columns), train.actual / train.base_price - 1)
        change, = fit.predict(x)
        candidate = float(row.base_price.iloc[0] * (1 + change))
        if name == "price_lgbm_blend":
            candidate = (1 - weight) * candidate + weight * latest.price_per_kg
        models[name] = fit.booster_
        forecasts.append(dict(term="short", item=item, horizon=horizon,
                              target_start=None, target_end=None, candidate_model=name,
                              existing_price=base["prediction_krw_per_kg"], candidate_price=candidate,
                              status="preview_missing_inputs" if unavailable else "candidate",
                              missing_features=", ".join(unavailable)))
    if args.dry_run:
        return

    pepper = prices.loc[prices.item.eq("건고추") & prices.date.le(origin)].copy()
    means = pepper.groupby(pepper.date.map(dekad_index)).price_per_kg.mean()
    old, = mid.loc[mid.item.eq("건고추") & mid.horizon.eq(2)].to_dict("records")
    sarimax = series_predictions(pd.DataFrame([dict(date=origin, origin_dekad=dekad_index(origin),
                                                    target_dekad=dekad_index(origin) + 2)]), pepper, means)
    forecasts.append(dict(term="mid", item="건고추", horizon=2, target_start=old["target_start"],
                          target_end=old["target_end"], candidate_model="sarimax",
                          existing_price=old["prediction_krw_per_kg"], candidate_price=float(sarimax.sarimax.iloc[0]),
                          status="candidate", missing_features=""))
    present = [r["candidate_price"] for r in forecasts if r["candidate_price"] is not None]
    if not all(np.isfinite(p) and p > 0 for p in present):
        raise ValueError("Invalid candidate price")
    args.output.mkdir(parents=True)
    hashes = {}
    for name, model in models.items():
        path = args.output / f"{name}.txt"
        model.save_model(str(path))
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    route_settings = [dict(item=item, horizon=horizon, model=name, input_features=PRICE_FEATURES + ["horizon", *extra])
                      for item, horizon, name, extra in ROUTES]
    route_settings.append(dict(item="건고추", horizon=2, model="sarimax", input_features=["completed_dekad_prices"]))
    generated_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    report = dict(generated_at=generated_at, run_type="reconstructed_origin",
                  routes=route_settings, base_date=str(origin.date()), lgbm_training_through=str(CUTOFF.date()),
                  source="mafra_api_provisional", radish_h5_last_value_weight=float(weight),
                  model_sha256=hashes, forecasts=forecasts)
    table = pd.DataFrame(forecasts).assign(base_date=report["base_date"], generated_at=generated_at)
    table.to_csv(args.output / "comparison.csv", index=False)
    (args.output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
