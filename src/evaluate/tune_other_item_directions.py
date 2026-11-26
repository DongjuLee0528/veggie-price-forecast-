"""Full-coverage logistic next-trade directions. Select C using 2024 only."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate.freeze_next_day_direction import FEATURES, ROOT, peer_inputs, price_inputs


def main():
    daily = pd.read_parquet(ROOT / "data/interim/price/settlement_daily_clean.parquet")
    potato = pd.read_parquet(ROOT / "data/processed/potato_sumi/prices.parquet")
    prices = pd.concat([daily, potato], ignore_index=True)
    metrics, trials, predictions = [], [], []
    for item in ("배추", "양파", "감자 수미", "건고추"):
        f = peer_inputs(price_inputs(item, prices, 1), prices)
        columns = FEATURES + list(f.filter(regex="^peer_").columns)
        x, y = f[columns], f.actual_direction
        boundary = f.loc[f.date.lt("2024-01-01"), "date"].iloc[-10]
        train = f.target_date.lt(boundary)
        valid = f.date.dt.year.eq(2024) & f.target_date.dt.year.eq(2024)
        test = (f.date.dt.year.isin([2025, 2026]) & f.target_date.dt.year.eq(f.date.dt.year)
                & f.target_date.dt.month.le(9))
        # Equal prices cannot be correctly called by an up/down-only forecast.
        fit = train & y.ne(0)
        transforms = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        z = transforms.fit(x[fit]).transform(x)
        best = -1
        for c in (.01, .0001, .0005, .001, .003, .005, .02, .05, .1, 1., 10.):
            model = LogisticRegression(C=c, max_iter=2000).fit(z[fit], y[fit])
            score = int((model.predict(z[valid]) == y[valid]).sum())
            trials.append(dict(item=item, C=c, correct=score, n=int(valid.sum())))
            if c == .01:
                fixed = model
            if score > best:
                best, chosen, selected = score, c, model
        result = f.loc[test, ["date", "target_date", "actual_direction"]].copy()
        result["item"] = item
        result["tuned"] = selected.predict(z[test])
        result["fixed"] = fixed.predict(z[test])
        result["previous"] = np.sign(f.loc[test, "ret1"])
        assert result.previous.notna().all() and not result.date.duplicated().any()
        predictions.append(result)
        for year, g in result.groupby(result.date.dt.year):
            metrics.append(dict(item=item, year=year, C=chosen, n=len(g),
                                **{key:int(g[key].eq(g.actual_direction).sum())
                                   for key in ("tuned", "fixed", "previous")}))
    dest = ROOT / "results/other_item_direction_tuning"
    dest.mkdir(exist_ok=True)
    pd.DataFrame(trials).to_csv(dest / "validation.csv", index=False)
    pd.concat(predictions).to_csv(dest / "predictions.csv", index=False)
    summary = pd.DataFrame(metrics)
    summary.to_csv(dest / "metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
