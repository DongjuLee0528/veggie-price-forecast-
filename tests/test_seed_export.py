import json

import pandas as pd
import pytest

from src.config import load_items
from src.db.export_seed import load_forecast


def test_seed_export_checks_csv_matches_json(tmp_path):
    date = '2026-09-30'
    items = load_items()['items']
    rows = [{'item': item, 'base_date': date, 'horizon': horizon, 'target_date': None,
             'prediction_krw_per_kg': 900., 'base_price_7day_krw_per_kg': 850.}
            for item in items for horizon in [1, 5, 10]]
    (tmp_path / 'forecast.json').write_text(json.dumps({'base_date': date, 'trained_through': date,
        'status': 'complete', 'unit': 'KRW/kg', 'forecasts': rows}))
    pd.DataFrame(rows).to_csv(tmp_path / 'predictions.csv', index=False)
    _, loaded, _ = load_forecast(tmp_path, items)
    assert len(loaded) == 15
    rows[0]['prediction_krw_per_kg'] = 2000.
    pd.DataFrame(rows).to_csv(tmp_path / 'predictions.csv', index=False)
    with pytest.raises(AssertionError):
        load_forecast(tmp_path, items)
