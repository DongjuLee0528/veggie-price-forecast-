INSERT INTO item (id, name, type, unit) VALUES (1, '배추', 'FRESH', 'KRW/kg');
INSERT INTO price_daily (item_id, trade_date, price_per_kg, volume_kg) VALUES
    (1, '2025-09-30', 790, 1000), (1, '2026-09-29', 810, 1000), (1, '2026-09-30', 850, 1000);
INSERT INTO forecast_run (id, base_date, source, model_version, created_at) VALUES
    (1, '2026-09-30', 'LIVE', 'fixture', '2026-09-30 12:00:00'),
    (2, '2026-09-29', 'BACKTEST', 'fixture', '2026-09-30 12:00:00');
INSERT INTO forecast_short (run_id, item_id, horizon, target_date, p50, base_price, model) VALUES
    (1, 1, 1, NULL, 875, 830, 'adaptive'), (1, 1, 5, NULL, 900, 830, 'adaptive'),
    (1, 1, 10, NULL, 840, 830, 'adaptive'), (2, 1, 1, '2026-09-30', 845, 820, 'adaptive');
INSERT INTO forecast_metric (item_id, horizon, period_start, period_end, nmae, direction_acc) VALUES
    (1, 1, '2026-09-30', '2026-09-30', 0.00588235, 1);
