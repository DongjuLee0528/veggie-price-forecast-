package com.veggie.forecast;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

@Component
@Profile("h2")
class SeedLoader implements ApplicationRunner {
    private final JdbcTemplate db;
    private final Path directory;

    SeedLoader(JdbcTemplate db, @Value("${forecast.seed-dir}") String directory) {
        this.db = db;
        this.directory = Path.of(directory).toAbsolutePath().normalize();
    }

    @Override
    @Transactional
    public void run(ApplicationArguments args) {
        List<SeedTable> tables = List.of(
            new SeedTable("item", "id, name, type, unit", true),
            new SeedTable("price_daily", "item_id, trade_date, price_per_kg, volume_kg", true),
            new SeedTable("forecast_run", "id, base_date, source, model_version, created_at", true),
            new SeedTable("forecast_short", "run_id, item_id, horizon, target_date, p10, p50, p90, base_price, model", true),
            new SeedTable("forecast_mid", "run_id, item_id, k, period_label, period_start, period_end, p10, p50, p90", false),
            new SeedTable("forecast_meta", "run_id, item_id, signal, alert, reasons", false),
            new SeedTable("forecast_metric", "item_id, horizon, period_start, period_end, nmae, direction_acc, coverage", false)
        );
        for (SeedTable table : tables) {
            Path path = directory.resolve(table.name() + ".csv");
            if (!Files.isRegularFile(path)) {
                if (table.required()) {
                    throw new IllegalStateException("Missing seed CSV: " + path + ". Run src.db.export_seed first.");
                }
                continue;
            }
            String file = path.toString().replace("'", "''");
            int rows = db.update("INSERT INTO " + table.name() + " (" + table.columns() + ") SELECT "
                + table.columns() + " FROM CSVREAD('" + file + "', NULL, 'charset=UTF-8')");
            if (table.required() && rows == 0) {
                throw new IllegalStateException("Empty seed CSV: " + path);
            }
        }
    }

    private record SeedTable(String name, String columns, boolean required) {}
}
