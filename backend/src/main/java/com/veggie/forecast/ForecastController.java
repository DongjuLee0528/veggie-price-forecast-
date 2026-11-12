package com.veggie.forecast;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

@RestController
@RequestMapping("/api")
class ForecastController {
    private final JdbcClient db;
    private final ItemRepository items;
    private final ObjectMapper mapper;

    ForecastController(JdbcClient db, ItemRepository items, ObjectMapper mapper) {
        this.db = db;
        this.items = items;
        this.mapper = mapper;
    }

    @GetMapping("/items")
    List<ItemView> items() {
        return items.findAllByOrderById().stream()
            .map(item -> new ItemView(item.name(), item.type(), item.unit())).toList();
    }

    @GetMapping("/forecast")
    ForecastView forecast(@RequestParam String item) throws JsonProcessingException {
        Item selected = findItem(item);
        Run run = db.sql("""
            SELECT r.id, r.base_date, r.model_version FROM forecast_run r
            WHERE r.source = 'LIVE' AND EXISTS (
                SELECT 1 FROM forecast_short s WHERE s.run_id = r.id AND s.item_id = :item
            ) ORDER BY r.base_date DESC, r.created_at DESC, r.id DESC LIMIT 1
            """).param("item", selected.id()).query(Run.class).optional()
            .orElseThrow(() -> new ApiError(HttpStatus.NOT_FOUND, "FORECAST_NOT_FOUND", "No forecast is available for this item."));
        List<Point> recent = db.sql("""
            SELECT trade_date AS date, price_per_kg AS price FROM (
                SELECT trade_date, price_per_kg FROM price_daily
                WHERE item_id = :item AND trade_date <= :date ORDER BY trade_date DESC LIMIT 30
            ) recent ORDER BY trade_date
            """).param("item", selected.id()).param("date", run.baseDate()).query(Point.class).list();
        List<ShortTerm> shortTerm = db.sql("""
            SELECT horizon, target_date AS date, p10, p50, p90, base_price, model
            FROM forecast_short WHERE run_id = :run AND item_id = :item ORDER BY horizon
            """).param("run", run.id()).param("item", selected.id()).query(ShortTerm.class).list();
        List<MidTerm> midTerm = db.sql("""
            SELECT k, period_label AS period, period_start, period_end, p10, p50, p90
            FROM forecast_mid WHERE run_id = :run AND item_id = :item ORDER BY k
            """).param("run", run.id()).param("item", selected.id()).query(MidTerm.class).list();
        Meta meta = db.sql("SELECT signal, alert, reasons FROM forecast_meta WHERE run_id = :run AND item_id = :item")
            .param("run", run.id()).param("item", selected.id()).query(Meta.class).optional()
            .orElse(new Meta(null, false, null));
        List<String> reasons = meta.reasons() == null ? List.of() : mapper.readValue(meta.reasons(), new TypeReference<>() {});
        return new ForecastView(item, selected.type(), run.baseDate(), selected.unit(), run.modelVersion(),
            recent, shortTerm, midTerm, meta.signal(), reasons, meta.alert());
    }

    @GetMapping("/history")
    HistoryView history(@RequestParam String item, @RequestParam(defaultValue = "3m") String range) {
        Item selected = findItem(item);
        LocalDate end = db.sql("SELECT MAX(trade_date) FROM price_daily WHERE item_id = :item")
            .param("item", selected.id()).query(LocalDate.class).optional()
            .orElseThrow(() -> new ApiError(HttpStatus.NOT_FOUND, "HISTORY_NOT_FOUND", "No price history is available."));
        LocalDate start = switch (range) {
            case "1m" -> end.minusMonths(1);
            case "3m" -> end.minusMonths(3);
            case "1y" -> end.minusYears(1);
            case "all" -> LocalDate.of(1001, 1, 1);
            default -> throw new ApiError(HttpStatus.BAD_REQUEST, "INVALID_RANGE", "Use 1m, 3m, 1y or all.");
        };
        List<Point> actual = points(selected.id(), start, end);
        Map<LocalDate, Point> previous = new LinkedHashMap<>();
        for (Point point : points(selected.id(), start.minusYears(1), end.minusYears(1))) {
            LocalDate date = point.date().plusYears(1);
            previous.putIfAbsent(date, new Point(date, point.price()));
        }
        return new HistoryView(item, selected.unit(), range, actual, List.copyOf(previous.values()));
    }

    @GetMapping("/forecast-log")
    LogView log(@RequestParam String item, @RequestParam(defaultValue = "10") int h) {
        if (h != 1 && h != 5 && h != 10) {
            throw new ApiError(HttpStatus.BAD_REQUEST, "INVALID_HORIZON", "Use 1, 5 or 10 trading days.");
        }
        Item selected = findItem(item);
        List<LogPoint> points = db.sql("""
            SELECT * FROM (
                SELECT s.target_date, r.base_date AS made_on, s.p10, s.p50, s.p90,
                    p.price_per_kg AS actual, LOWER(r.source) AS source
                FROM forecast_short s JOIN forecast_run r ON s.run_id = r.id
                LEFT JOIN price_daily p ON p.item_id = s.item_id AND p.trade_date = s.target_date
                WHERE s.item_id = :item AND s.horizon = :h
                ORDER BY r.base_date DESC, r.id DESC LIMIT 90
            ) recent ORDER BY made_on
            """).param("item", selected.id()).param("h", h).query(LogPoint.class).list();
        Metric metric = db.sql("""
            SELECT period_start, period_end, nmae, direction_acc, coverage FROM forecast_metric
            WHERE item_id = :item AND horizon = :h ORDER BY period_end DESC LIMIT 1
            """).param("item", selected.id()).param("h", h).query(Metric.class).optional()
            .orElse(new Metric(null, null, null, null, null));
        return new LogView(item, h, points, metric.nmae(), metric.directionAcc(), metric.coverage(),
            metric.periodStart(), metric.periodEnd());
    }

    private Item findItem(String name) {
        return items.findByName(name).orElseThrow(() -> new ApiError(HttpStatus.NOT_FOUND, "ITEM_NOT_FOUND", "Item not found."));
    }

    private List<Point> points(long id, LocalDate start, LocalDate end) {
        return db.sql("""
            SELECT trade_date AS date, price_per_kg AS price FROM price_daily
            WHERE item_id = :item AND trade_date BETWEEN :start AND :end ORDER BY trade_date
            """).param("item", id).param("start", start).param("end", end).query(Point.class).list();
    }

    @ExceptionHandler(ApiError.class)
    ResponseEntity<ErrorView> handleApiError(ApiError error) {
        return ResponseEntity.status(error.status).body(new ErrorView(error.code, error.getMessage()));
    }

    @ExceptionHandler({MissingServletRequestParameterException.class, MethodArgumentTypeMismatchException.class})
    ResponseEntity<ErrorView> handleInvalidParameter(Exception error) {
        return ResponseEntity.badRequest().body(new ErrorView("INVALID_PARAMETER", "Check the request parameters."));
    }

    private static class ApiError extends RuntimeException {
        private final HttpStatus status;
        private final String code;

        ApiError(HttpStatus status, String code, String message) {
            super(message);
            this.status = status;
            this.code = code;
        }
    }

    record ErrorView(String code, String message) {}
    record ItemView(String name, String type, String unit) {}
    record Run(long id, LocalDate baseDate, String modelVersion) {}
    record Point(LocalDate date, BigDecimal price) {}
    record ShortTerm(int horizon, LocalDate date, BigDecimal p10, BigDecimal p50, BigDecimal p90,
                     BigDecimal basePrice, String model) {}
    record MidTerm(int k, String period, LocalDate periodStart, LocalDate periodEnd,
                   BigDecimal p10, BigDecimal p50, BigDecimal p90) {}
    record Meta(String signal, boolean alert, String reasons) {}
    record ForecastView(String item, String type, LocalDate baseDate, String unit, String modelVersion,
                        List<Point> recentActual, List<ShortTerm> shortTerm, List<MidTerm> midTerm,
                        String signal, List<String> reasons, boolean alert) {}
    record HistoryView(String item, String unit, String range, List<Point> actual, List<Point> lastYear) {}
    record LogPoint(LocalDate targetDate, LocalDate madeOn, BigDecimal p10, BigDecimal p50, BigDecimal p90,
                    BigDecimal actual, String source) {}
    record Metric(LocalDate periodStart, LocalDate periodEnd, BigDecimal nmae, BigDecimal directionAcc, BigDecimal coverage) {}
    record LogView(String item, int h, List<LogPoint> points, BigDecimal nmae, BigDecimal directionAcc, BigDecimal coverage,
                   LocalDate periodStart, LocalDate periodEnd) {}
}
