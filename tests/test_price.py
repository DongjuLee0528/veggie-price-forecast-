import numpy as np
import pandas as pd
import pytest

from src.collect.price import load_prices, validate_prices
from src.config import load_items


@pytest.fixture
def prices():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02"] * 5),
        "item": list(load_items()["items"]),
        "amount": [1000.] * 5,
        "volume_kg": [10.] * 5,
        "n_rows": [2] * 5,
        "price_per_kg": [100.] * 5,
    })


@pytest.mark.parametrize("suffix", ["parquet", "csv"])
def test_load_formats(prices, tmp_path, monkeypatch, suffix):
    path = tmp_path / f"prices.{suffix}"
    if suffix == "parquet":
        prices.to_parquet(path, index=False)
    else:
        prices.to_csv(path, index=False)
    monkeypatch.chdir(tmp_path)
    pd.testing.assert_frame_equal(load_prices(path), prices)


@pytest.mark.parametrize("column,value", [
    ("price_per_kg", 0), ("volume_kg", -1), ("amount", np.inf),
    ("price_per_kg", np.nan), ("n_rows", 1.5), ("item", "unknown"),
    ("price_per_kg", 200), ("date", pd.Timestamp("2026-01-02 12:00")),
])
def test_invalid_values(prices, column, value):
    if column == "n_rows":
        prices[column] = prices[column].astype(float)
    prices.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_prices(prices)


def test_duplicates(prices):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prices(pd.concat([prices, prices.iloc[:1]]))


def test_missing_column(prices):
    with pytest.raises(ValueError, match="Missing price columns"):
        validate_prices(prices.drop(columns="amount"))


def test_empty(prices):
    with pytest.raises(ValueError, match="empty"):
        validate_prices(prices.iloc[:0])


def test_no_mutation(prices):
    original = prices.copy(deep=True)
    validate_prices(prices)
    pd.testing.assert_frame_equal(prices, original)
