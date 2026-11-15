import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.compare_all_items import ITEMS

HORIZONS = (1, 5, 10)
RADISH_TRADE_FEATURES = {
    "import_onion_ratio_12m",
    "import_fresh_chilled_ratio_12m",
    "import_frozen_ratio_12m",
    "import_dried_ratio_12m",
    "import_powder_ratio_12m",
}


def export_short(inputs, output):
    base = pd.read_csv(inputs / "short_forecast.csv").query("status == 'predicted'")
    supplementary = pd.read_csv(inputs / "short_forecast_pending_preview.csv")
    radish = supplementary.query("item == '무' and horizon in [5, 10]").copy()
    if len(radish) != 2 or not radish.missing_features.fillna("").map(
        lambda value: set(value.split(", ")) == RADISH_TRADE_FEATURES
    ).all():
        raise ValueError("Radish must have only the expected missing trade features")
    potato_backup = supplementary.query(
        "item == '감자 수미' and horizon == 5 and status == 'preview_saved_fallback'"
    )
    tuned = pd.read_csv(inputs / "potato_tuned/predictions.csv")
    if set(zip(tuned.item, tuned.horizon)) != {("감자 수미", 1), ("감자 수미", 10)}:
        raise ValueError("Expected tuned potato forecasts at horizons 1 and 10")
    rows = pd.concat([
        base.assign(model_route="saved_production"),
        radish.assign(model_route="saved_production"),
        potato_backup.assign(model_route="price_only_fallback"),
        tuned.assign(model_route="tuned"),
    ], ignore_index=True)
    expected = {(item, horizon) for item in ITEMS for horizon in HORIZONS}
    if rows.duplicated(["item", "horizon"]).any() or set(zip(rows.item, rows.horizon)) != expected:
        raise ValueError("Expected one forecast per item and horizon")
    if (rows.base_date.isna().any() or rows.base_date.nunique() != 1
            or not rows.source.eq("mafra_api_provisional").all()
            or not np.isfinite(rows.prediction_krw_per_kg).all()
            or not rows.prediction_krw_per_kg.gt(0).all()):
        raise ValueError("Inconsistent forecast date, source, or price")
    rows = rows.assign(horizon_unit="observed_trading_days", target_date=None)
    columns = ["item", "horizon", "horizon_unit", "base_date", "target_date",
               "prediction_krw_per_kg", "model", "model_route", "source"]
    rows = rows[columns].sort_values(["item", "horizon"])
    report = {
        "base_date": rows.base_date.iloc[0],
        "unit": "KRW/kg",
        "source": "mafra_api_provisional",
        "target_date_status": "trading_dates_unconfirmed",
        "forecasts": rows.to_dict(orient="records"),
    }
    output.mkdir(parents=True, exist_ok=False)
    rows.to_csv(output / "predictions.csv", index=False)
    (output / "forecast.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Export the combined short-term forecasts.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_short(args.inputs, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
