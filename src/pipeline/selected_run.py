import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.wholesale_verify.prepare_inputs import WORK, prepare_inputs
from src.evaluate.prospective_direction import forecast_signals
from src.evaluate.compare_candidates import ROUTES, origin_shares
from src.evaluate.prospective_forecasts import EXPECTED, ROOT, record
from src.models.train_lgbm import model_inputs
from src.pipeline.radish_logistic import MODEL as RADISH_MODEL, probabilities
from src.pipeline.predict_mid import predict_mid


def operational_inputs(origin, inputs):
    settlement = WORK / "mafra_settlement"
    today = pd.read_csv(settlement / f"daily_{origin:%Y%m%d}.csv")
    expected_items = {item for _, item, _ in EXPECTED}
    if len(today) != len(expected_items) or set(today.item) != expected_items or not today.date.eq(str(origin.date())).all():
        raise ValueError("Invalid origin settlement report")
    valid = today.observed_markets.gt(0) & today.review_markets.eq(0) & today.unconfirmed_markets.eq(0)
    valid &= today.provisional_price_per_kg.gt(0) & today.volume_kg.gt(0)
    missing = expected_items - set(today.loc[valid, "item"])
    prices, features = prepare_inputs(origin, inputs)
    pending = {item: "Origin settlement is incomplete" for item in missing}
    for item in expected_items - missing:
        history = prices.loc[prices.item.eq(item), "price_per_kg"]
        required = features.loc[features.item.eq(item)].drop(columns=["date", "item", "yoy"], errors="ignore")
        if history.tail(29).isna().any() or not np.isfinite(required.to_numpy(dtype=float)).all():
            pending[item] = "Recent trading prices are incomplete"
    return prices, features, pending


def input_signature(prices, settlement, origin):
    digest = hashlib.sha256(pd.util.hash_pandas_object(prices, index=False).values.tobytes())
    market = settlement / f"markets_{origin:%Y%m%d}.json"
    digest.update(market.read_bytes())
    return digest.hexdigest()


def run(origin, inputs, output):
    if output.exists():
        raise FileExistsError(output)
    expected_items = {item for _, item, _ in EXPECTED}
    settlement = WORK / "mafra_settlement"
    prices, features, pending = operational_inputs(origin, inputs)
    available = expected_items - pending.keys()
    if not available:
        raise ValueError("No items have complete operational inputs")
    references = prices[prices.date.eq(origin)].set_index("item").price_per_kg
    features = features.set_index("item")
    models = ROOT / "results"
    snapshot = ROOT / "outputs"
    production = models / "lgbm_production"
    production_file = production / "run.json"
    manifest = json.loads(production_file.read_text())
    if manifest["status"] != "complete" or pd.Timestamp(manifest["cutoff"]) >= origin:
        raise ValueError("Invalid short model manifest or cutoff")
    routes = {(r["item"], r["horizon"]): r for r in manifest["selected"]}
    hashes = {r["file"]: r["sha256"] for r in manifest["models"]}
    candidates = snapshot / "final_candidate_forecasts_20261113"
    comparison_file = candidates / "comparison.json"
    selection = json.loads(comparison_file.read_text())
    if pd.Timestamp(selection["lgbm_training_through"]) >= origin:
        raise ValueError("Selected candidate training cutoff is too recent")
    replaced = {(item, horizon): (name, extras) for item, horizon, name, extras in ROUTES}
    potato = snapshot / "wholesale_verify/forecast_inputs_20261113/potato_tuned"
    potato_files = {
        1: potato / "lgbm_4y_7leaves_25child_last_0.5.txt",
        5: models / "potato_sumi/short_lgbm.txt",
        10: potato / "lgbm_2y_7leaves_50child.txt",
    }
    artifacts = [production_file, comparison_file, Path(__file__),
                 ROOT / "scripts/filter_settlement.py", ROOT / "scripts/wholesale_verify/prepare_inputs.py",
                 ROOT / "src/features/build.py", ROOT / "src/pipeline/predict_mid.py",
                 ROOT / "src/models/train_mid.py", ROOT / "src/models/tune_mid.py", ROOT / "src/models/train_lgbm.py",
                 ROOT / "src/evaluate/compare_candidates.py", ROOT / "src/evaluate/targeted_batch.py",
                 models / "mid_tuning/selection.json", models / "potato_sumi_tuning/validation_selection.csv",
                 *(models / "lgbm_mid_comparison" / f"{item}_price_lgbm.txt"
                   for item in sorted(expected_items - {"감자 수미"}))]
    shorts = []
    for _, item, horizon in sorted(route for route in EXPECTED if route[0] == "short"):
        if item not in available:
            continue
        row = features.loc[[item]].assign(horizon=horizon)
        base = float(row.base_price.iloc[0])
        last = float(references.loc[item])
        history = prices[prices.item.eq(item)]
        weight, rounds, sha = 0., None, None
        if (item, horizon) in replaced:
            name, extras = replaced[item, horizon]
            path = candidates / f"{name}.txt"
            sha = selection["model_sha256"][name]
            if "low_volume" in extras:
                volume = history.volume_kg.tail(29)
                if len(volume) != 29 or not volume.gt(0).all():
                    raise ValueError(f"Missing trading volume for {item}")
                row["low_volume"] = float(volume.iloc[-1] < .5 * volume.iloc[:-1].median())
            if "median_to_mean_7" in extras:
                row["median_to_mean_7"] = history.price_per_kg.tail(7).median() / base
            if name == "variety_mix":
                row = row.assign(**origin_shares(settlement / f"markets_{origin:%Y%m%d}.json", item))
            if name == "price_lgbm_blend":
                weight = selection["radish_h5_last_value_weight"]
        elif item == "감자 수미":
            path = potato_files[horizon]
            name = "potato_sumi/short_lgbm.txt" if horizon == 5 else path.stem
            weight = .5 if horizon == 1 else 0.
        else:
            route = routes[item, horizon]
            name = route["model"]
            path = production / route["file"] if name != "last_value" else None
            if path is not None:
                sha, rounds = hashes[path.name], route["best_iteration"]
        if path is None:
            prediction = last
        else:
            payload = path.read_bytes()
            if sha is not None and hashlib.sha256(payload).hexdigest() != sha:
                raise ValueError(f"Model checksum mismatch: {path.name}")
            booster = lgb.Booster(model_str=payload.decode())
            model_row = model_inputs(row, booster.feature_name())
            # Prior-year prices can be unavailable on non-trading dates, as in training.
            required = model_row.drop(columns=["yoy"], errors="ignore")
            missing = required.columns[required.isna().any()].tolist()
            if missing:
                raise ValueError(f"Missing operational features: {item} {horizon}: {missing}")
            change, = booster.predict(model_row, num_iteration=rounds, num_threads=2)
            prediction = (1 - weight) * base * (1 + float(change)) + weight * last
            artifacts.append(path)
        if not np.isfinite(prediction) or prediction <= 0:
            raise ValueError(f"Invalid forecast price: {item} {horizon}")
        shorts.append(dict(item=item, horizon=horizon, horizon_unit="observed_trading_days",
                           base_date=str(origin.date()), target_date=None, prediction_krw_per_kg=prediction,
                           model=name, source="mafra_api_provisional"))
    short = dict(base_date=str(origin.date()), unit="KRW/kg", source="mafra_api_provisional",
                 forecasts=shorts)
    short["origin_prices"] = {item: float(references[item]) for item in sorted(available)}
    short["signals"] = forecast_signals([r for r in shorts if r["item"] != "무"], inputs / "prices.parquet", origin)
    if "무" in available:
        prob = float(probabilities(prices).loc[origin])
        short["radish_probability_up"] = prob
        short["radish_logistic_hash"] = hashlib.sha256(RADISH_MODEL.read_bytes()).hexdigest()
        short["signals"]["무"] = "NEXT_DAY_UP" if prob > .5 else "NEXT_DAY_DOWN"
        artifacts.append(RADISH_MODEL)
    short["input_signature"] = input_signature(prices, settlement, origin)
    mid = predict_mid(inputs, models, selected_pepper=True, items=available)
    for report in (short, mid):
        report["pending_items"] = pending
        report["generated_at"] = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
        report["model_hash"] = hashlib.sha256("".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(artifacts))).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "selected"
        for term, report in (("short", short), ("mid", mid)):
            folder = staging / term
            folder.mkdir(parents=True)
            (folder / "forecast.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
            pd.DataFrame(report["forecasts"]).to_csv(folder / "predictions.csv", index=False)
        staging.rename(output)
    return artifacts


def main():
    parser = argparse.ArgumentParser(description="Run and record the fixed short- and mid-term models for today.")
    parser.add_argument("--date", type=pd.Timestamp)
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    today = pd.Timestamp(now.date())
    origin = args.date.normalize() if args.date is not None else today
    if origin > today:
        parser.error("Forecast origin cannot be in the future")
    inputs = WORK / f"forecast_inputs_{origin:%Y%m%d}"
    output = ROOT / "outputs" / f"selected_forecasts_{origin:%Y%m%d}"
    ledger = ROOT / "outputs/production_prospective"
    if output.exists():
        output = output.with_name(output.name + now.strftime("_%H%M%S%f"))
    try:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temp:
            staging = Path(temp) / "forecast"
            artifacts = run(origin, inputs, staging)
            if origin == today:
                report = json.loads((staging / "short/forecast.json").read_text())
                partial = bool(report.get("pending_items"))
                destination = ledger / output.name if partial or "_" in output.name.removeprefix("selected_forecasts_") else ledger
                path = record(staging, inputs / "price_features.parquet", inputs / "prices.parquet",
                              artifacts, destination, pd.Timestamp.now(tz="Asia/Seoul"), allow_partial=partial)
            else:
                for term in ("short", "mid"):
                    report_path = staging / term / "forecast.json"
                    report = json.loads(report_path.read_text())
                    report.update(generated_at=pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                                  run_type="delayed_settlement")
                    report_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False))
                path = output
            staging.rename(output)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Saved selected forecasts: {output}; record: {path}")


if __name__ == "__main__":
    main()
