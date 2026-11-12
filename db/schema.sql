CREATE TABLE item (
    id BIGINT PRIMARY KEY,
    name VARCHAR(40) NOT NULL UNIQUE,
    type VARCHAR(12) NOT NULL CHECK (type IN ('FRESH', 'STORAGE')),
    unit VARCHAR(12) NOT NULL
);

CREATE TABLE price_daily (
    item_id BIGINT NOT NULL,
    trade_date DATE NOT NULL,
    price_per_kg DECIMAL(18,6) NOT NULL CHECK (price_per_kg > 0),
    volume_kg DECIMAL(20,6) CHECK (volume_kg >= 0),
    FOREIGN KEY (item_id) REFERENCES item(id),
    PRIMARY KEY (item_id, trade_date)
);
CREATE INDEX price_daily_date_idx ON price_daily(trade_date);

CREATE TABLE forecast_run (
    id BIGINT PRIMARY KEY,
    base_date DATE NOT NULL,
    source VARCHAR(12) NOT NULL CHECK (source IN ('LIVE', 'BACKTEST')),
    model_version VARCHAR(80) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    UNIQUE (base_date, source, model_version)
);

CREATE TABLE forecast_short (
    run_id BIGINT NOT NULL,
    item_id BIGINT NOT NULL,
    horizon INT NOT NULL CHECK (horizon BETWEEN 1 AND 10),
    target_date DATE,
    p10 DECIMAL(18,6),
    p50 DECIMAL(18,6) NOT NULL CHECK (p50 > 0),
    p90 DECIMAL(18,6),
    base_price DECIMAL(18,6) NOT NULL CHECK (base_price > 0),
    model VARCHAR(100) NOT NULL,
    FOREIGN KEY (run_id) REFERENCES forecast_run(id),
    FOREIGN KEY (item_id) REFERENCES item(id),
    PRIMARY KEY (run_id, item_id, horizon),
    CHECK (p10 IS NULL OR p10 <= p50),
    CHECK (p90 IS NULL OR p90 >= p50)
);
CREATE INDEX forecast_short_target_idx ON forecast_short(item_id, target_date);

CREATE TABLE forecast_mid (
    run_id BIGINT NOT NULL,
    item_id BIGINT NOT NULL,
    k INT NOT NULL CHECK (k IN (2, 3)),
    period_label VARCHAR(40) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    p10 DECIMAL(18,6),
    p50 DECIMAL(18,6) NOT NULL CHECK (p50 > 0),
    p90 DECIMAL(18,6),
    FOREIGN KEY (run_id) REFERENCES forecast_run(id),
    FOREIGN KEY (item_id) REFERENCES item(id),
    PRIMARY KEY (run_id, item_id, k)
);

CREATE TABLE forecast_meta (
    run_id BIGINT NOT NULL,
    item_id BIGINT NOT NULL,
    signal VARCHAR(40),
    alert BOOLEAN NOT NULL DEFAULT FALSE,
    reasons TEXT,
    FOREIGN KEY (run_id) REFERENCES forecast_run(id),
    FOREIGN KEY (item_id) REFERENCES item(id),
    PRIMARY KEY (run_id, item_id)
);

CREATE TABLE forecast_metric (
    item_id BIGINT NOT NULL,
    horizon INT NOT NULL CHECK (horizon BETWEEN 1 AND 10),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    nmae DECIMAL(12,8) NOT NULL,
    direction_acc DECIMAL(12,8) NOT NULL,
    coverage DECIMAL(12,8),
    FOREIGN KEY (item_id) REFERENCES item(id),
    PRIMARY KEY (item_id, horizon, period_start)
);
