package com.veggie.forecast;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.jdbc.Sql;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@Transactional
@Sql("/forecast-fixture.sql")
class ForecastApiTest {
    @Autowired
    private MockMvc client;

    @Test
    void returnsPointForecastFromLatestLiveRun() throws Exception {
        client.perform(get("/api/items"))
            .andExpect(status().isOk()).andExpect(jsonPath("$[0].name").value("배추"));
        client.perform(get("/api/forecast").param("item", "배추"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.base_date").value("2026-09-30"))
            .andExpect(jsonPath("$.short_term.length()").value(3))
            .andExpect(jsonPath("$.short_term[0].p50").value(875))
            .andExpect(jsonPath("$.short_term[0].date").doesNotExist())
            .andExpect(jsonPath("$.short_term[0].p10").doesNotExist())
            .andExpect(jsonPath("$.recent_actual[2].price").value(850));
    }

    @Test
    void returnsPreviousYearAlignedToTheSelectedPeriod() throws Exception {
        client.perform(get("/api/history").param("item", "배추").param("range", "1m"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.actual.length()").value(2))
            .andExpect(jsonPath("$.last_year[0].date").value("2026-09-30"))
            .andExpect(jsonPath("$.last_year[0].price").value(790));
    }

    @Test
    void distinguishesBacktestFromUnscoredLivePredictions() throws Exception {
        client.perform(get("/api/forecast-log").param("item", "배추").param("h", "1"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.points[0].source").value("backtest"))
            .andExpect(jsonPath("$.points[0].actual").value(850))
            .andExpect(jsonPath("$.points[1].source").value("live"))
            .andExpect(jsonPath("$.points[1].actual").doesNotExist())
            .andExpect(jsonPath("$.direction_acc").value(1));
    }

    @Test
    void rejectsUnknownItemsAndInvalidParameters() throws Exception {
        client.perform(get("/api/forecast").param("item", "unknown"))
            .andExpect(status().isNotFound()).andExpect(jsonPath("$.code").value("ITEM_NOT_FOUND"));
        client.perform(get("/api/history").param("item", "배추").param("range", "2d"))
            .andExpect(status().isBadRequest()).andExpect(jsonPath("$.code").value("INVALID_RANGE"));
        client.perform(get("/api/forecast-log").param("item", "배추").param("h", "2"))
            .andExpect(status().isBadRequest()).andExpect(jsonPath("$.code").value("INVALID_HORIZON"));
        client.perform(get("/api/forecast"))
            .andExpect(status().isBadRequest()).andExpect(jsonPath("$.code").value("INVALID_PARAMETER"));
    }
}
