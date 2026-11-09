from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_items


DEFAULT_PRICE_PATH = Path(__file__).resolve().parents[2] / "data/raw/price/settlement_daily.parquet"
REQUIRED_COLUMNS = {"date", "item", "amount", "volume_kg", "n_rows", "price_per_kg"}


def validate_prices(prices: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(sorted(missing))}")
    if prices.empty:
        raise ValueError("Price data must not be empty")
    if prices[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Price data contains missing values")
    if not pd.api.types.is_datetime64_any_dtype(prices["date"]):
        raise ValueError("Price dates must use a datetime dtype")
    if (prices["date"] != prices["date"].dt.normalize()).any():
        raise ValueError("Price dates must not contain time components")
    if set(prices["item"]) != set(load_items()["items"]):
        raise ValueError("Price data must contain exactly the five configured items")
    if prices.duplicated(["date", "item"]).any():
        raise ValueError("Duplicate date/item pairs in price data")
    for column in ("amount", "volume_kg", "n_rows", "price_per_kg"):
        values = prices[column]
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise ValueError(f"Price column must be numeric: {column}")
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"Price column must contain finite positive values: {column}")
    if (prices["n_rows"] % 1 != 0).any():
        raise ValueError("n_rows must contain whole numbers")
    if not np.allclose(prices["price_per_kg"], prices["amount"] / prices["volume_kg"], rtol=1e-6, atol=1e-6):
        raise ValueError("price_per_kg does not match amount / volume_kg")


def load_prices(path: str | Path = DEFAULT_PRICE_PATH) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        prices = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        prices = pd.read_csv(path, parse_dates=["date"])
    else:
        raise ValueError("Price file must be Parquet or CSV")
    validate_prices(prices)
    return prices
