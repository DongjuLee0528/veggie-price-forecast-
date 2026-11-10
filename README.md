# veggie-price-forecast

배추·무·양파·깐마늘·건고추 5개 품목의 도매 경락가를 단기(10거래일 일별)·중기(2·3순 후)로 예측하고,
가격예보 웹 화면으로 보여 주는 인공지능 수업 프로젝트.

- 2×2 구조: 신선형(배추·무)/저장형(양파·깐마늘·건고추) × 단기/중기
- 모델: LightGBM · SARIMAX · TFT/TCN 칸별 앙상블, 가격 변화율 + 분위수 예측
- 상세 명세: 노션 "인공지능 프로젝트" 페이지

## 시작하기

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 인증키 입력
```

## 폴더

| 경로 | 내용 |
| --- | --- |
| `configs/` | 품목·피처 설정 |
| `data/raw` → `interim` → `processed` | 원본 → 정제 → 학습용 (Git 제외) |
| `data/manual/` | 관측월보·수급 대책 수기 CSV |
| `src/collect` | API·CSV 수집 |
| `src/preprocess`, `src/features` | 정합·공개 시차·피처·타깃 |
| `src/models`, `src/ensemble` | 베이스라인·LightGBM·SARIMAX·딥러닝, 칸별 앙상블 |
| `src/evaluate`, `src/explain` | 롤링 백테스트, 변동 근거·신호 |
| `src/pipeline` | 수집→예측→저장 실행 |
| `api/` | FastAPI |
| `web/` | 가격예보 화면 (React + TypeScript) |
| `results/`, `outputs/`, `snapshots/` | 백테스트 결과, 예측 JSON, 회차별 스냅샷 (Git 제외) |

## Price outlier policy

- Preserve raw settlement records; do not guess corrected prices or weights.
- Exclude only the 19 reviewed suspect trades: 12 dried pepper trades in `REVIEWED_PEPPER_TRADES` (Jinju, variety 건고추(일반), 0.6 kg) and seven peeled garlic trades (2024-04-11, Busan Banyeo, variety 깐마늘 남도, 500 kg).
- `src/preprocess/price_review.py` identifies these trades and recomputes daily prices as total transaction amount divided by total weight. New matching suspects remain `review_only` until reviewed.
- Do not apply item/year 1st–99th percentile filtering. Any future statistical thresholds must be evaluated using training-period information only.
- Keep excluded records, reasons and before/after comparisons in `data/interim/price_review/`. The current `exclude_trial` label and trial filename remain until the final cleaned dataset is generated.
- Reproduce the comparison with `python -m src.preprocess.price_review`. This does not overwrite raw data.
- These exclusions were selected retrospectively; report this limitation when evaluating historical forecasts.

Generate the cleaned daily dataset with `python -m src.preprocess.clean_prices`.
The output is `data/interim/price/settlement_daily_clean.parquet`, accompanied by
`excluded_rows.csv`, `review_only_rows.csv`, `daily_comparison.csv` and
`validation.json`. Load it with
`load_prices("data/interim/price/settlement_daily_clean.parquet")` from
`src.collect.price`; the loader's default still points to raw data.

## Short-term baseline evaluation

Run `python -m src.evaluate.backtest` to evaluate the three arithmetic baselines
at origins spaced ten observed trading days apart per item. Horizons are the next
1–10 observed trading days. Validation origins and targets stay in 2024; test
origins start in 2025. Earlier observations remain available as history. No
parameters are fitted, so a training-label gap is not applicable to these rules.
Results are in `results/baseline/`: `backtest.parquet`, split metric CSVs and
`common_sample_metrics.csv` (only samples predicted by all three rules).
NMAE is mean absolute error divided by each actual price; direction uses the
seven-trading-day base price, with unchanged counted as a separate direction.
`coverage` is the fraction of available point predictions, not interval coverage.
Observed future dates are used for retrospective scoring only, not as a live
trading-calendar forecast. Missing prior-year prices are not imputed. Scores
averaged across item/horizon cells are macro averages. Retrospectively selected
cleaning exclusions remain a limitation. No learned model or interval is fitted.

## Garak pre-auction arrivals (F-02)

The command loads the project `.env` at runtime without printing credentials.
Existing process environment variables take precedence. Run a one-day check before a historical collection:
`python -m src.collect.garak --start 2024-04-11 --end 2024-04-11`.
After checking the real response, collect from 2021-01-01 with an explicit end date.
Daily `data/raw/volume/garak_YYYYMMDD.parquet` files support resuming; existing
files are read, not overwritten. Missing items/days remain NaN with a status;
absence is not assumed to mean zero or market closure. A reported zero is kept.
Garlic arrivals include both peeled and unpeeled garlic; `source_item` records
this distinction from the peeled-garlic price target. Exact category/item matching
excludes related vegetables and category totals. API/network errors stop the run.
No data before 2021 is fabricated; the later merge must retain 2020 as NaN.
The implementation follows the Notion F-02 response contract. Live response
schema, HTTPS endpoint and authentication still require the one-day check.

Garak validation (2021-01-01 through 2026-11-08): 2,138 daily files and 10,690
rows, no missing date files or duplicate date/item keys. Cabbage, radish and
onion are entirely missing in 2021–2022; two sampled original API responses
contained no corresponding items in their categories. Keep these values NaN;
the API's full historical provision policy is not confirmed. The first observed
date for these three items is 2023-01-03.

## ASOS daily weather (F-03)

Run a one-station smoke check with
`python -m src.collect.asos --start 2024-01-01 --end 2024-01-02 --station 100`.
After validating the live response, run
`python -m src.collect.asos --start 2020-01-01 --end 2026-11-08`.
The command loads `DATA_GO_KR_KEY` at runtime without printing it. Fourteen unique
configured stations are collected by month, with resumable files in
`data/raw/weather/`. Records retain station-level daily mean/max/min temperature
(Celsius), precipitation (mm) and sunshine (hours). Blank observations remain NaN,
not zero. Dates absent from the response are represented with `observed=False`;
this is coverage tracking, not a claim that the station supplied data. Seasonal
station selection, regional averaging and t-1 publication lag belong to the later
merge stage. Source: Korea Meteorological Administration, ASOS daily data.

Consolidate validated station-level weather with `python -m src.preprocess.weather`.
Outputs: `data/interim/weather/asos_daily.parquet` and `missing_summary.csv`.
The loader removes identical overlapping records but rejects conflicting values
or missing station/date rows. Source-missing rows and all blank fields remain
NaN; no blanket zero-fill is applied to precipitation because the API blank-value
semantics have not been established. Official ASOS information:
https://data.kma.go.kr/data/grnd/selectAsosList.do?pgmNo=36
The current default validation range is 2020-01-01 through 2026-11-08; pass explicit
start/end dates to `load_weather` for other ranges.

## Demand calendar (F-06)

Run `python -m src.collect.calendar --start-year 2019 --end-year 2027`.
Yearly holiday source records are stored in `data/raw/calendar/`; daily features
are in `data/interim/calendar/calendar_daily.parquet`. Exact three-day Seollal
and Chuseok blocks are required; their middle date anchors the countdown, with
zero on the festival and NaN beyond the last known festival. Substitute holidays
are not festival anchors. Kimjang is a modeling interval, November 1–December 10,
configured in `configs/features.yaml`, not a regional observation.
Historical temporary/substitute holiday announcements must be checked before
using today's holiday list in retrospective forecasts. Future lists may change.

## Customs imports (F-04)

Run `python -m src.collect.customs --start 2020-01 --end 2026-09`.
The HTTPS collector queries parent headings for onions and fresh/chilled garlic,
and exact codes for frozen garlic, dried pepper and powder. Only allowed onion
leaf codes (0703101000/0703101010/0703101090) are summed; shallots and green onions
are excluded. Parent detail records are checked against the API grand total.
Code-level absence within a returned parent month is not converted into a missing
whole-item total. Entirely absent selected months remain NaN.
Original yearly code-level files remain unchanged; new source XML and leaf-level
Parquet files are in `data/raw/trade/parent_v1/`. Raw XML is checked for embedded
credentials before saving. `data/interim/trade/imports_monthly.parquet` contains
physical-weight totals; `imports_by_form.parquet` keeps fresh/chilled garlic and
frozen garlic separate, as well as dried pepper and powder. No fresh-equivalent
yield or causal price-impact weights are applied. Use separate form features
before considering weights, chosen on validation data only.
`parent_reconciliation.csv` records exact matches and source-total differences.
Differences no larger than (detail row count + 1)/2 units are retained and flagged
as potentially compatible with integer rounding; this is an explicit review
assumption, not verified source rounding policy. Larger differences stop the run.
The 2020-01..2026-09 run has 243 complete item-month totals. Ten of 35 source
batches have 1–2 kg total/detail discrepancies; USD totals match. The discrepancy
cause is unresolved and values are not adjusted. Publication lag and historical
revisions must still be addressed in integration.
Source: https://www.data.go.kr/data/15100475/openapi.do

## ECOS exchange rates (F-05)

Run `python -m src.collect.ecos --start 2020-01-01 --end 2026-11-09`.
The collector uses configured 731Y001 daily USD/CNY series and stores yearly
observations under `data/raw/fx/`, with unfilled calendar-day data under
`data/interim/fx/fx_daily.parquet`. Values are KRW per one USD/CNY; series name,
code and unit are retained in raw files. Existing yearly ranges are resumed.
Missing dates are not filled here; t-1 availability and backward-only carry
forward, if used, belong to integration. No exchange-rate forecasting is done.

Customs QA conclusion: all 35 source batches were checked; ten differ by 1–2 kg
(maximum relative difference below 0.000064%), while USD totals match. Use the
unaltered monthly detail sums, retain source totals and audit differences. The
underlying rounding/revision cause is not established; this is a documented
source limitation, not a correction or a claim of exact reconciliation.

## First-pass integration (F-11)

Run `python -m src.preprocess.merge`. Output:
`data/interim/integrated/daily.parquet`, `missing_by_item.csv`, `validation.json`.
The table retains only actual price trading dates and preserves price/volume
values. Arrivals join on the same day; weather uses the prior calendar day and
seasonal station rules for that source day. Regional averages require every
selected station value; counts remain available. FX uses the prior calendar day
without filling unpublished values. Trade forms join the month two months before
the price date. Calendar joins exclude historical is_holiday because announcement
vintages are not verified. Empty manual outlook/policy inputs are omitted for the
first pass, without dummy values; nonempty manual files cause a review error.
Date cutoffs are tested, but trade release timing is an assumption and retrieved
historical revisions are not point-in-time vintages. This is an integration table,
not final model features/targets. Complete rainfall windows cannot yet be assumed
because blank rain values remain unknown. Separate form imports have no arbitrary
weights. Full F-11 manual-source support and release-date QA remain outstanding.

Integration availability update:
- FX now selects each currency's latest nonmissing observation at or before t-1,
  retaining its actual source date and age. Raw FX gaps remain unchanged. No
  future fill is used; long gaps are visible through age and are not certified
  as healthy market closures.
- `configs/trade_releases.csv` lists all 81 source months. Official release dates
  verified for 2026-08 and 2026-09 are used from the next calendar day because
  intraday availability is unknown. The other 79 months explicitly retain the
  two-month availability assumption; these are not verified release dates.
  Report release timing does not establish the original vintage of today's
  detailed HSK API values. Source date, availability date and basis are retained
  per import form. No release date is inferred by assuming every month uses day 15.
- Missing rainfall remains NaN in both station and regional tables, with station
  observation counts. It is not converted to zero or silently averaged away.
