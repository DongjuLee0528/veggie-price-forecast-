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
