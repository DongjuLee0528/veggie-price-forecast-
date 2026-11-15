from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
WORK = Path(__file__).resolve().parents[2] / "outputs" / "wholesale_verify"
AS_OF = pd.Timestamp("2026-11-13")
ITEMS = {"감자 수미", "건고추", "무", "배추", "양파"}


def price_features(history):
    output = []
    for _, group in history.groupby("item"):
        frame = group[["date", "item"]].copy()
        price = group.price_per_kg
        frame["base_price"] = price.rolling(7).mean()
        for lag in (1, 5, 10, 20):
            frame[f"lag_{lag}"] = price.shift(lag) / frame.base_price
        frame["ma7_ma28_ratio"] = frame.base_price / price.rolling(28).mean()
        for lag in (7, 28):
            frame[f"ret_{lag}"] = price / price.shift(lag) - 1
        frame["vol_28"] = price.pct_change(fill_method=None).rolling(28).std()
        prior = (group.date - pd.DateOffset(years=1)).map(group.set_index("date").price_per_kg)
        frame["yoy"] = price / prior - 1
        output.append(frame)
    return pd.concat(output, ignore_index=True)


history = pd.concat([
    pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet"),
    pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet"),
], ignore_index=True)
history = history[history.item.isin(ITEMS) & history.date.lt("2026-10-01")].copy()
history["source"] = "settlement_csv"

dates = pd.date_range("2026-10-01", AS_OF)
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
current = features[features.date.eq(AS_OF)].copy()
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

output = WORK / f"forecast_inputs_{AS_OF:%Y%m%d}"
output.mkdir(exist_ok=True)
prices.to_parquet(output / "prices.parquet", index=False)
current.to_parquet(output / "price_features.parquet", index=False)
readiness.to_csv(output / "readiness.csv", index=False)
print(readiness.to_string(index=False))
