import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.collect.asos import ROOT
from src.evaluate.backtest import score_predictions
from src.models.compare_all_items import ITEMS

HORIZONS = (1, 5, 10)


def main():
    parser = argparse.ArgumentParser(description="Build local dashboard seed data.")
    parser.add_argument("--date", type=pd.Timestamp)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    outputs = ROOT / "outputs"
    short_dates = {path.parent.name.removeprefix("short_forecast_")
                   for path in outputs.glob("short_forecast_????????/predictions.csv")}
    mid_dates = {path.parent.name.removeprefix("mid_forecast_")
                 for path in outputs.glob("mid_forecast_????????/predictions.csv")}
    complete_dates = short_dates & mid_dates
    if args.date is None and not complete_dates:
        parser.error("No matching short- and mid-term forecast exports")
    date = args.date if args.date is not None else pd.to_datetime(max(complete_dates), format="%Y%m%d")
    stamp = date.strftime("%Y%m%d")
    source = outputs / "wholesale_verify"
    history = source / f"forecast_inputs_{stamp}" / "prices.parquet"
    short_path = outputs / f"short_forecast_{stamp}" / "predictions.csv"
    mid_path = outputs / f"mid_forecast_{stamp}" / "predictions.csv"
    old = ROOT / "db/seed/current"
    items = pd.read_csv(old / "item.csv").replace({"name": {"깐마늘": "감자 수미"}})
    ids = items.set_index("name").id
    if set(ids.index) != ITEMS:
        raise ValueError("Unexpected dashboard items")

    prices = pd.read_parquet(history).dropna(subset=["price_per_kg"])
    additional = []
    for path in sorted((source / "mafra_settlement").glob("daily_*.csv")):
        daily = pd.read_csv(path, parse_dates=["date"])
        additional.append(daily.loc[daily.date.gt(date)].rename(
            columns={"provisional_price_per_kg": "price_per_kg"}))
    if additional:
        latest = pd.concat(additional).dropna(subset=["price_per_kg"])
        prices = pd.concat([prices, latest[["date", "item", "price_per_kg", "volume_kg"]]],
                           ignore_index=True)
    if (prices.duplicated(["date", "item"]).any() or set(prices.item) != ITEMS
            or not np.isfinite(prices[["price_per_kg", "volume_kg"]]).all().all()
            or not prices[["price_per_kg", "volume_kg"]].gt(0).all().all()):
        raise ValueError("Invalid price history")
    features = pd.read_parquet(history.with_name("price_features.parquet"))
    base = features.loc[features.date.eq(date)].set_index("item").base_price

    short = pd.read_csv(short_path)
    mid = pd.read_csv(mid_path)
    for rows, horizons in ((short, HORIZONS), (mid, (2, 3))):
        expected = {(item, h) for item in ITEMS for h in horizons}
        if (len(rows) != len(expected) or set(zip(rows.item, rows.horizon)) != expected
                or not pd.to_datetime(rows.base_date).eq(date).all()
                or not np.isfinite(rows.prediction_krw_per_kg).all()
                or not rows.prediction_krw_per_kg.gt(0).all()):
            raise ValueError("Invalid current forecasts")
    reference = pd.read_csv(old / "forecast_short.csv")
    runs = pd.read_csv(old / "forecast_run.csv")
    runs = runs[runs.source.eq("BACKTEST")].reset_index(drop=True)
    potato_id = ids.loc["감자 수미"]
    reference = reference[reference.run_id.isin(runs.id) & reference.item_id.ne(potato_id)]
    metrics = pd.read_csv(old / "forecast_metric.csv").query("item_id != @potato_id")
    archived = pd.read_parquet(ROOT / "results/potato_sumi_tuning/predictions.parquet")
    archived = archived.query("split == 'test' and term == 'short' and model == 'validation_selected' and horizon in @HORIZONS")
    potato = archived.merge(prices[["item", "date", "price_per_kg"]],
                            left_on=["item", "target_date"], right_on=["item", "date"],
                            validate="many_to_one")
    if len(potato) != len(archived) or not np.allclose(potato.actual, potato.price_per_kg):
        raise ValueError("Potato backtest actuals do not match price history")
    run_ids = runs.set_index(pd.to_datetime(runs.base_date)).id
    missing = sorted(set(potato.base_date) - set(run_ids.index))
    if missing:
        new = pd.DataFrame({"id": range(int(runs.id.max()) + 1, int(runs.id.max()) + 1 + len(missing)),
                            "base_date": [day.date().isoformat() for day in missing], "source": "BACKTEST",
                            "model_version": "potato_validation_selected", "created_at": runs.created_at.iloc[0]})
        runs = pd.concat([runs, new], ignore_index=True)
        run_ids = runs.set_index(pd.to_datetime(runs.base_date)).id
    historical = potato.rename(columns={"prediction": "p50"}).assign(
        run_id=potato.base_date.map(run_ids), item_id=potato_id, p10=None, p90=None)
    measured = score_predictions(potato)
    bounds = potato.groupby("horizon").target_date.agg(period_start="min", period_end="max")
    measured = measured.join(bounds, on="horizon").rename(columns={"direction_accuracy": "direction_acc"})
    metrics = pd.concat([metrics, measured.assign(item_id=potato_id, coverage=None)[metrics.columns]])
    version = hashlib.sha256(short_path.read_bytes() + mid_path.read_bytes()).hexdigest()[:16]
    live_id = int(runs.id.max()) + 1
    runs.loc[len(runs)] = [live_id, date.date().isoformat(), "LIVE", f"api-provisional-{version}", pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")]
    short = short.rename(columns={"prediction_krw_per_kg": "p50"}).assign(
        run_id=live_id, item_id=short.item.map(ids), p10=None, p90=None,
        base_price=short.item.map(base))
    if not short.base_price.gt(0).all():
        raise ValueError("Missing base price")
    mid = mid.rename(columns={"horizon": "k", "target_start": "period_start", "target_end": "period_end",
                              "prediction_krw_per_kg": "p50"})
    mid = mid.assign(run_id=live_id, item_id=mid.item.map(ids), p10=None, p90=None,
                     period_label=mid.period_start + " ~ " + mid.period_end)
    tables = {
        "item": items, "forecast_run": runs, "forecast_metric": metrics,
        "price_daily": prices.assign(item_id=prices.item.map(ids), trade_date=prices.date.dt.normalize())[
            ["item_id", "trade_date", "price_per_kg", "volume_kg"]],
        "forecast_short": pd.concat([reference, historical[reference.columns], short[reference.columns]]),
        "forecast_mid": mid[["run_id", "item_id", "k", "period_label", "period_start", "period_end", "p10", "p50", "p90"]],
    }
    args.output.mkdir(parents=True)
    for name, frame in tables.items():
        for column in ("base_date", "trade_date", "target_date", "period_start", "period_end"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], format="mixed").dt.strftime("%Y-%m-%d")
        frame.to_csv(args.output / f"{name}.csv", index=False)
    print(f"Forecast origin {date.date()}, latest observed price {prices.date.max().date()}")


if __name__ == "__main__":
    main()
