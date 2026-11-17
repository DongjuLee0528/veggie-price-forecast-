import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluate.freeze_direction_candidates import FEATURES, block_interval, direction_inputs
from src.evaluate.freeze_next_day_direction import OUTPUT, peer_inputs
from src.evaluate.prospective_forecasts import ROOT, official_prices
from src.features.build import price_features


ITEM = "양파"


def record(source, history, models, ledger, now):
    now = now.tz_convert("Asia/Seoul")
    origin = pd.Timestamp(now.date())
    run = json.loads(source.read_text())
    recorded = pd.Timestamp(run["recorded_at"]).tz_convert("Asia/Seoul")
    if pd.Timestamp(run["base_date"]) != origin or recorded.date() != now.date() or recorded > now:
        raise ValueError("Expected today's prospectively recorded price forecast")
    manifest = json.loads((models / "manifest.json").read_text())
    if manifest["features"] != FEATURES:
        raise ValueError("Candidate feature definitions differ")
    candidates = {(r["item"], r["horizon"]): r for r in manifest["routes"]
                  if r["decision"] == "candidate_for_new_period"}
    rows = [r for r in run["forecasts"] if r["term"] == "short" and
            (r["item"], r["horizon"]) in candidates]
    if len(rows) != len(candidates) or {(r["item"], r["horizon"]) for r in rows} != candidates.keys():
        raise ValueError("Missing or duplicate candidate price forecast routes")

    prices = pd.read_parquet(history, columns=["date", "item", "price_per_kg", "volume_kg"])
    prices = prices.loc[prices.date.le(origin)].copy()
    latest = {}
    for item in {item for item, _ in candidates}:
        daily = prices[prices.item.eq(item)].sort_values("date").reset_index(drop=True)
        if daily.date.duplicated().any() or daily.date.max() != origin:
            raise ValueError(f"Expected unique prices through today's origin: {item}")
        features = price_features(daily)
        eligible = features.base_price.notna()
        frame = daily.loc[eligible].assign(vol_28=features.vol_28, yoy=features.yoy)
        frame = direction_inputs(frame)
        if any(route.get("features") for (name, _), route in candidates.items() if name == item):
            frame = peer_inputs(frame, prices)
        row = frame.tail(1)
        required = [name for name in FEATURES if name != "yoy"]
        if (len(row) != 1 or row.date.iloc[0] != origin or not row[required].notna().all().all()
                or not np.isfinite(row.price_per_kg.iloc[0]) or not row.price_per_kg.gt(0).all()
                or not row.volume_kg.gt(0).all()):
            raise ValueError(f"Incomplete direction inputs at the origin: {item}")
        latest[item] = row

    forecasts = []
    for row in rows:
        item, horizon = row["item"], row["horizon"]
        origin_row = latest[item]
        last = float(origin_row.price_per_kg.iloc[0])
        candidate = candidates[item, horizon]
        if pd.Timestamp(candidate["train_target_through"]) >= origin:
            raise ValueError(f"Invalid candidate cutoff: {item} {horizon}")
        if not np.isclose(row["last_value_krw_per_kg"], last, rtol=1e-12):
            raise ValueError("Candidate and current forecast reference prices differ")
        payload = (models / candidate["file"]).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != candidate["sha256"]:
            raise ValueError(f"Candidate checksum mismatch: {item} {horizon}")
        features = candidate.get("features", FEATURES)
        booster = lgb.Booster(model_str=payload.decode())
        if booster.feature_name() != features:
            raise ValueError(f"Candidate feature order differs: {item} {horizon}")
        probabilities = booster.predict(origin_row[features], num_threads=2)
        prob = probabilities[0] if probabilities.ndim == 2 else [1 - probabilities[0], probabilities[0]]
        index = int(np.argmax(prob))
        confidence = float(prob[index])
        threshold = candidate.get("min_probability")
        direction = int(candidate["classes"][index]) if threshold is None or confidence >= threshold else None
        forecasts.append(dict(item=item, horizon=horizon, origin_price=last, target_date=row.get("target_date"),
                              candidate=direction, confidence=confidence, candidate_sha256=digest,
                              threshold=threshold, benchmark=candidate.get("benchmark", "mean7" if horizon == 1 else "current"),
                              mean7=int(np.sign(origin_row.avg7.iloc[0])),
                              current=None if row.get("model") == "last_value" else
                              int(np.sign(row["prediction_krw_per_kg"] - last))))
    report = dict(base_date=str(origin.date()), recorded_at=now.isoformat(),
                  price_model_version=run["model_version"], forecasts=forecasts)
    ledger.mkdir(parents=True, exist_ok=True)
    path = ledger / f"direction_{origin:%Y%m%d}.json"
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    return path


def score(ledger, prices):
    paths = sorted(ledger.glob("direction_????????.json"))
    if not paths:
        raise ValueError("No prospectively recorded directions")
    rows = []
    for path in paths:
        run = json.loads(path.read_text())
        origin = pd.Timestamp(run["base_date"])
        if pd.Timestamp(run["recorded_at"]).tz_convert("Asia/Seoul").date() != origin.date():
            raise ValueError("Found a direction record created outside its origin date")
        for row in run["forecasts"]:
            item = row.get("item", ITEM)
            daily = prices.loc[prices.item.eq(item) & prices.date.gt(origin)].sort_values("date")
            target = daily.iloc[row["horizon"] - 1:row["horizon"]]
            date = target.date.iloc[0] if not target.empty else pd.NaT
            if row["target_date"] and not target.empty and date != pd.Timestamp(row["target_date"]):
                raise ValueError("Saved target differs from official trading calendar")
            actual = int(np.sign(target.price_per_kg.iloc[0] - row["origin_price"])) if not target.empty else None
            rows.append(dict(row, item=item, base_date=origin, target_date=date, actual_direction=actual,
                             price_model_version=run["price_model_version"], scored=actual is not None))
    frame = pd.DataFrame(rows)
    if frame.groupby(["item", "horizon"]).candidate_sha256.nunique().gt(1).any() or frame.price_model_version.nunique() != 1:
        raise ValueError("Expected fixed candidate and price model versions")
    return frame


def main():
    out = ROOT / "outputs"
    parser = argparse.ArgumentParser(description="Record and score frozen next-day direction candidates.")
    parser.add_argument("--ledger", type=Path, default=out / "three_item_direction_prospective")
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("record")
    save.add_argument("--models", type=Path, default=OUTPUT)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("--official-csv", type=Path, action="append", required=True)
    evaluate.add_argument("--official-through", type=pd.Timestamp, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    stamp = now.strftime("%Y%m%d")
    try:
        if args.command == "record":
            source = out / f"production_prospective/forecast_{stamp}.json"
            if not source.is_file():
                raise ValueError(f"No prospectively recorded price forecast for {now.date()}")
            history = out / f"wholesale_verify/forecast_inputs_{stamp}/prices.parquet"
            print(record(source, history, args.models, args.ledger, now))
            return
        if args.official_through >= pd.Timestamp(now.date()):
            raise ValueError("Official coverage must end before today")
        rows = score(args.ledger, official_prices(args.official_csv, args.official_through))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"{exc}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.output, index=False)
    for (item, horizon), group in rows[rows.scored].groupby(["item", "horizon"]):
        available = len(group)
        group = group[group.candidate.notna()]
        if group.empty:
            print(f"{item} {horizon}: 0/{available} directions issued")
            continue
        baseline = group.benchmark.iloc[0] if "benchmark" in group else "mean7" if horizon == 1 else "current"
        if group[baseline].isna().any():
            raise ValueError("Missing direction benchmark")
        group = group.assign(candidate_hit=group.candidate.eq(group.actual_direction),
                             baseline_hit=group[baseline].eq(group.actual_direction))
        low, high = block_interval(group, "baseline_hit")
        print(f"{item} {horizon}: issued={len(group)}/{available}, direction={100 * group.candidate_hit.mean():.1f}%, "
              f"{baseline}={100 * group.baseline_hit.mean():.1f}%, difference CI=[{low:+.1f}, {high:+.1f}] pp")
    print(f"Scored {int(rows.scored.sum())}/{len(rows)} forecasts. Saved: {args.output}")


if __name__ == "__main__":
    main()
