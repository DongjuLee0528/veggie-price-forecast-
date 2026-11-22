import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.collect.wholesale_codes import fetch_unit
from src.models.compare_all_items import ITEMS
from src.pipeline import selected_run
from src.pipeline.daily_refresh import ready_items, origin_ready


@pytest.mark.parametrize("key", ["example%2Bkey%3D", "example+key="])
def test_unit_code_decodes_key_once(key):
    response = Mock(status_code=200)
    response.json.return_value = {"response": {"header": {"resultCode": "0"}, "body": {
        "items": {"item": [{"unit_cd": "10", "unit_nm": "."}]}}}}
    session = Mock()
    session.get.return_value = response
    assert fetch_unit(session, key, "10") == [{"unit_cd": "10", "unit_nm": "."}]
    assert session.get.call_args.kwargs["params"]["serviceKey"] == "example+key="


def test_one_incomplete_item_does_not_block_other_items(monkeypatch, tmp_path):
    origin = pd.Timestamp("2026-11-20")
    directory = tmp_path / "mafra_settlement"
    directory.mkdir()
    daily = pd.DataFrame([dict(date=str(origin.date()), item=item, observed_markets=28,
        review_markets=0, unconfirmed_markets=0, provisional_price_per_kg=100., volume_kg=1000.)
        for item in sorted(ITEMS)])
    path = directory / "daily_20261120.csv"
    daily.to_csv(path, index=False)
    assert origin_ready(path, origin)
    prices = daily.rename(columns={"provisional_price_per_kg": "price_per_kg"})
    prices.loc[prices.item.eq("무"), "price_per_kg"] = np.nan
    features = daily[["date", "item"]].assign(base_price=100., lag_1=100., yoy=np.nan)
    features.loc[features.item.eq("무"), "lag_1"] = np.nan
    monkeypatch.setattr(selected_run, "WORK", tmp_path)
    monkeypatch.setattr(selected_run, "prepare_inputs", lambda *_: (prices, features))
    _, _, pending = selected_run.operational_inputs(origin, tmp_path)
    assert set(pending) == {"무"}
    daily.loc[daily.item.eq("무"), "review_markets"] = 1
    daily.to_csv(path, index=False)
    assert ready_items(path, origin) == ITEMS - {"무"}
    assert not origin_ready(path, origin)


def test_refresh_publishes_latest_partial_date_and_does_not_repeat(monkeypatch, tmp_path):
    from src.pipeline import daily_refresh

    today = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    latest = today - pd.Timedelta(days=1)
    work = tmp_path / "outputs" / "wholesale_verify"
    cache = work / "mafra_settlement"
    cache.mkdir(parents=True)
    for day in pd.date_range("2026-10-01", today):
        report = pd.DataFrame([dict(date=str(day.date()), item=item,
            observed_markets=int(day == latest), review_markets=0, unconfirmed_markets=0,
            provisional_price_per_kg=100. if day == latest else np.nan, volume_kg=1000.)
            for item in sorted(ITEMS)])
        report.to_csv(cache / f"daily_{day:%Y%m%d}.csv", index=False)
    monkeypatch.setattr(daily_refresh, "ROOT", tmp_path)
    monkeypatch.setattr(daily_refresh, "WORK", work)
    monkeypatch.setattr(daily_refresh, "operational_inputs", lambda *_: (None, None, {"무": "Incomplete"}))
    execute = Mock()
    monkeypatch.setattr(daily_refresh, "execute", execute)
    daily_refresh.main()
    execute.assert_any_call("src.pipeline.selected_run", "--date", str(latest.date()))
    execute.assert_any_call("src.db.sync_mysql")
    folder = tmp_path / "outputs" / f"selected_forecasts_{latest:%Y%m%d}" / "short"
    folder.mkdir(parents=True)
    (folder / "forecast.json").write_text(json.dumps({"forecasts": [
        {"item": item} for item in sorted(ITEMS - {"무"})], "pending_items": {"무": "Incomplete"}}))
    execute.reset_mock()
    daily_refresh.main()
    assert not any(call.args[0] == "src.pipeline.selected_run" for call in execute.call_args_list)
