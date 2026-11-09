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
