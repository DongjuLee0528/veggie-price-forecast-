from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REVIEWED_PEPPER_TRADES = (
    ("2020-01-30", 100, 110000),
    ("2020-01-31", 53, 106000),
    ("2020-02-27", 100, 150000),
    ("2020-02-29", 160, 125000),
    ("2020-03-17", 120, 125000),
    ("2020-03-19", 60, 85000),
    ("2020-03-20", 60, 85000),
    ("2020-03-21", 20, 85000),
    ("2020-04-04", 20, 125000),
    ("2020-04-18", 30, 100000),
    ("2020-04-18", 25, 100000),
    ("2020-04-22", 20, 100000),
)


def review_prices(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"date", "market_code", "item", "variety", "unit_qty", "price_per_unit", "kg_per_unit"}
    if required - set(rows.columns):
        raise ValueError("Missing required settlement row columns")
    if rows.empty or rows[list(required)].isna().any().any():
        raise ValueError("Settlement rows must be nonempty and complete")
    if not pd.api.types.is_datetime64_any_dtype(rows.date):
        raise ValueError("Settlement dates must use a datetime dtype")
    for column in ("unit_qty", "price_per_unit", "kg_per_unit"):
        if not pd.api.types.is_numeric_dtype(rows[column]) or not np.isfinite(rows[column]).all() or (rows[column] <= 0).any():
            raise ValueError(f"Invalid positive numeric column: {column}")
    data = rows.copy()
    data["source_row"] = np.arange(len(data))
    garlic = data.item.eq("깐마늘") & data.market_code.eq("210009") & data.variety.eq("깐마늘 남도") & data.kg_per_unit.eq(500)
    pepper = data.item.eq("건고추") & data.market_code.eq("380401") & data.variety.eq("건고추(일반)") & data.kg_per_unit.eq(0.6) & data.price_per_unit.ge(85000)
    reviewed_pepper = pd.Series(False, index=data.index)
    for date, quantity, price in REVIEWED_PEPPER_TRADES:
        reviewed_pepper |= data.date.eq(date) & data.unit_qty.eq(quantity) & data.price_per_unit.eq(price)
    excluded = (garlic & data.date.eq("2024-04-11")) | (pepper & reviewed_pepper)
    data["reason"] = ""
    data.loc[garlic, "reason"] = "garlic_500kg_unit_suspect"
    data.loc[pepper, "reason"] = "pepper_0.6kg_high_unit_price_suspect"
    data["amount"] = data.unit_qty * data.price_per_unit
    data["volume_kg"] = data.unit_qty * data.kg_per_unit
    flagged = data.loc[garlic | pepper].copy()
    flagged["action"] = np.where(excluded.loc[flagged.index], "exclude_trial", "review_only")

    def aggregate(frame):
        daily = frame.groupby(["date", "item"], as_index=False).agg(
            amount=("amount", "sum"), volume_kg=("volume_kg", "sum"), n_rows=("amount", "size")
        )
        daily["price_per_kg"] = daily.amount / daily.volume_kg
        return daily.sort_values(["item", "date"]).reset_index(drop=True)

    before = aggregate(data)
    after = aggregate(data.loc[~excluded])
    comparison = before.merge(after, on=["date", "item"], how="left", suffixes=("_before", "_after"), validate="one_to_one")
    comparison["excluded_rows"] = comparison.n_rows_before - comparison.n_rows_after.fillna(0)
    comparison["price_change_pct"] = (comparison.price_per_kg_after / comparison.price_per_kg_before - 1) * 100
    return after, flagged, comparison


def main():
    rows = pd.read_parquet(ROOT / "data/raw/price/settlement_rows.parquet")
    after, flagged, comparison = review_prices(rows)
    output = ROOT / "data/interim/price_review"
    output.mkdir(parents=True, exist_ok=True)
    after.to_parquet(output / "settlement_daily_trial.parquet", index=False)
    flagged.to_csv(output / "flagged_rows.csv", index=False)
    comparison.to_csv(output / "daily_comparison.csv", index=False)
    print(flagged.groupby(["reason", "action"]).size().to_string())
    print(comparison.loc[comparison.excluded_rows.gt(0)].to_string(index=False))


if __name__ == "__main__":
    main()
