from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import price_features

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parents[2] / "outputs" / "wholesale_verify"
ITEMS = {"감자 수미", "건고추", "무", "배추", "양파"}


def prepare_inputs(origin, output):
    history = pd.concat([
        pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet"),
        pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet"),
    ], ignore_index=True)
    history = history[history.item.isin(ITEMS) & history.date.lt("2026-10-01")].copy()
    history["source"] = "settlement_csv"

    dates = pd.date_range("2026-10-01", origin)
    api = pd.concat([
        pd.read_csv(WORK / "mafra_settlement" / f"daily_{day:%Y%m%d}.csv", parse_dates=["date"])
        for day in dates
    ], ignore_index=True)
    expected = {(day, item) for day in dates for item in ITEMS}
    if api.duplicated(["date", "item"]).any() or set(zip(api.date, api.item)) != expected:
        raise ValueError("Unexpected API date or item coverage")

    traded = api.observed_markets.gt(0) | api.review_markets.gt(0)
    api = api.loc[traded, ["date", "item", "amount", "volume_kg", "provisional_price_per_kg"]]
    api = api.rename(columns={"provisional_price_per_kg": "price_per_kg"})
    api["source"] = "mafra_api_provisional"
    api.loc[api.price_per_kg.isna(), ["amount", "volume_kg"]] = np.nan
    prices = pd.concat([history, api], ignore_index=True).sort_values(["item", "date"]).reset_index(drop=True)
    if prices.duplicated(["item", "date"]).any():
        raise ValueError("Duplicate price history")
    observed = prices.dropna(subset=["price_per_kg"])
    np.testing.assert_allclose(observed.amount / observed.volume_kg, observed.price_per_kg)

    features = price_features(prices)
    current = features[features.date.eq(origin)].copy()
    if set(current.item) != ITEMS:
        raise ValueError("Expected all five items at forecast origin")
    columns = current.columns.drop(["date", "item"])
    readiness = current[["date", "item"]].copy()
    readiness["missing_price_features"] = current[columns].apply(
        lambda row: ", ".join(row.index[row.isna()]), axis=1
    )
    readiness["observed_api_days"] = readiness.item.map(api.groupby("item").price_per_kg.count())
    readiness["pending_api_days"] = readiness.item.map(api.groupby("item").price_per_kg.apply(lambda s: s.isna().sum()))
    readiness["missing_last_29_prices"] = readiness.item.map(
        prices.groupby("item").price_per_kg.apply(lambda s: s.tail(29).isna().sum())
    )

    output.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output / "prices.parquet", index=False)
    current.to_parquet(output / "price_features.parquet", index=False)
    readiness.to_csv(output / "readiness.csv", index=False)
    print(readiness.to_string(index=False))
    return prices, current


if __name__ == "__main__":
    prepare_inputs(pd.Timestamp("2026-11-13"), WORK / "forecast_inputs_20261113")
