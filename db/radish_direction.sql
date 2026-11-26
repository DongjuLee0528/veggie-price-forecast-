CREATE TABLE IF NOT EXISTS radish_direction_logistic (
    run_id BIGINT PRIMARY KEY,
    probability_up DOUBLE NOT NULL CHECK (probability_up BETWEEN 0 AND 1),
    reconstructed BOOLEAN NOT NULL,
    model_hash CHAR(64) NOT NULL,
    recorded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES forecast_run(id)
);
