import json

import pandas as pd

from src.collect.price import validate_prices
from src.preprocess.price_review import ROOT, review_prices


def main():
    source = ROOT / "data/raw/price/settlement_rows.parquet"
    prices, flagged, comparison = review_prices(pd.read_parquet(source))
    validate_prices(prices)
    excluded = flagged.loc[flagged.action.eq("exclude_trial")].copy()
    excluded["action"] = "exclude"
    review_only = flagged.loc[flagged.action.eq("review_only")].copy()
    summary = {
        "source": str(source.relative_to(ROOT)),
        "policy": "reviewed_trades_only_v1",
        "excluded_rows": len(excluded),
        "excluded_by_item": excluded.groupby("item").size().to_dict(),
        "review_only_rows": len(review_only),
        "affected_item_days": int(comparison.excluded_rows.gt(0).sum()),
        "daily_rows": len(prices),
        "items": {
            item: {"start": str(group.date.min().date()), "end": str(group.date.max().date()), "rows": len(group)}
            for item, group in prices.groupby("item")
        },
        "validation": "passed",
        "limitation": "Exclusions were selected retrospectively; account for this in historical evaluation.",
    }
    output = ROOT / "data/interim/price"
    output.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output / "settlement_daily_clean.parquet", index=False)
    excluded.to_csv(output / "excluded_rows.csv", index=False)
    review_only.to_csv(output / "review_only_rows.csv", index=False)
    comparison.to_csv(output / "daily_comparison.csv", index=False)
    (output / "validation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
