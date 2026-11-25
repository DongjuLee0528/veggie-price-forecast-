import argparse
import json
import os
import runpy
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

import pandas as pd
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--date", default="2026-09-15")
parser.add_argument("--gumi-pepper", action="store_true")
parser.add_argument("--gumi-pepper-direct", action="store_true")
parser.add_argument("--collect", nargs=2, metavar=("START", "END"))
parser.add_argument("--cached-only", action="store_true", help="Rebuild collection reports from saved market responses.")
parser.add_argument("--refresh-all", action="store_true", help="Fetch all markets again, including cached successes.")
parser.add_argument("--refresh-empty", action="store_true", help="Retry markets with empty cached responses.")
args = parser.parse_args()
if (args.refresh_empty or args.refresh_all) and (not args.collect or args.cached_only):
    parser.error("--refresh-empty requires live collection")
if args.cached_only and not args.collect:
    parser.error("--cached-only requires --collect")
if args.collect:
    start_date, end_date = map(pd.Timestamp, args.collect)
    if start_date < pd.Timestamp("2026-10-01") or end_date < start_date:
        parser.error("--collect requires an ordered range starting on or after 2026-10-01")
root = Path.cwd()
date = pd.Timestamp(args.date).normalize()
codes = {"0501": "감자 수미", "1001": "배추", "1101": "무",
         "1201": "양파", "1207": "건고추"}
classify = runpy.run_path(str(root / "scripts/filter_settlement.py"))["classify"]

existing = pd.read_parquet(
    root / "data/raw/price/settlement_rows.parquet",
    columns=["date", "market_code", "item"],
    filters=[("date", ">=", pd.Timestamp("2026-09-01")),
             ("date", "<", pd.Timestamp("2026-10-01"))],
)
markets = sorted(existing.market_code.unique())
baseline = pd.concat([
    pd.read_parquet(root / "data/interim/price/settlement_daily_clean.parquet"),
    pd.read_parquet(root / "data/processed/potato_sumi/prices.parquet"),
])
baseline = baseline[baseline.date.eq(date) & baseline.item.isin(codes.values())]
if baseline.empty and not (args.collect or args.gumi_pepper or args.gumi_pepper_direct):
    raise SystemExit("선택한 날짜의 기존 모델 가격이 없습니다.")

key = os.getenv("MAFRA_WHOLESALE_API_KEY", "").strip()
if not key and not args.cached_only:
    raise SystemExit("MAFRA_WHOLESALE_API_KEY 환경변수가 비어 있습니다.")
base = (f"http://211.237.50.150:7080/openapi/{quote(key, safe='')}"
        "/xml/Grid_20240625000000000653_1")


def fetch(session, market, day, filters=None):
    start, total, records = 1, None, []
    params = {"SALEDATE": day.strftime("%Y%m%d"), "WHSALCD": market}
    params.update(filters or {})
    while total is None or start <= total:
        end = start + 999 if total is None else min(start + 999, total)
        try:
            response = session.get(
                f"{base}/{start}/{end}",
                params=params,
                timeout=30, allow_redirects=False,
            )
        except requests.RequestException as error:
            raise ValueError(type(error).__name__) from None
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}")
        xml = ET.fromstring(response.content)
        code = xml.findtext(".//code", "")
        if code == "INFO-200" and start == 1:
            return None
        if code != "INFO-000":
            raise ValueError(code.replace(key, "***") or "결과 코드 없음")
        count = int(xml.findtext("totalCnt", "-1"))
        page = xml.findall(".//row")
        if count < 0 or (total is not None and count != total):
            raise ValueError("전체 행 수 불일치")
        total = count
        if len(page) != min(end, total) - start + 1:
            raise ValueError("페이지 행 수 불일치")
        for row in page:
            if (row.findtext("WHSALCD", "").strip() != market
                    or row.findtext("SALEDATE", "").strip() != day.strftime("%Y%m%d")):
                raise ValueError("응답 시장 또는 날짜 불일치")
            item_code = "".join(row.findtext(field, "").strip() for field in ("LARGE", "MID"))
            item_name = row.findtext("MIDNAME", "").strip()
            variety = row.findtext("SMALLNAME", "").strip()
            pepper_candidate = market == "371501" and (
                item_code == "1207" or any(word in item_name + variety for word in ("건고추", "마른고추"))
            )
            if item_code not in codes and not pepper_candidate:
                continue
            small = row.findtext("SMALL", "").strip()
            if item_code == "0501" and small != "01":
                continue
            records.append({
                "market": market,
                "PDLT_NM": codes.get(item_code, ""),
                "SPCS_NM": variety,
                "api_item_code": item_code,
                "api_variety_code": item_code + small,
                "api_item_name": item_name,
                "corporation": row.findtext("CMPCD", "").strip(),
                "package": row.findtext("STD", "").strip(),
                "pepper_candidate": pepper_candidate,
                "unit": row.findtext("DANCD", "").strip(),
                "api_won": row.findtext("TOTAMT", "").strip(),
                "api_kg": row.findtext("TOTQTY", "").strip(),
            })
        start = end + 1
    return records


if args.collect:
    output = Path(__file__).resolve().parents[2] / "outputs" / "wholesale_verify" / "mafra_settlement"
    output.mkdir(exist_ok=True)

    def save(frame_or_text, target):
        pending = target.with_suffix(".tmp")
        pending.write_text(frame_or_text, encoding="utf-8")
        pending.replace(target)

    with requests.Session() as session:
        for day in pd.date_range(start_date, end_date):
            cache = output / f"markets_{day:%Y%m%d}.json"
            received = json.loads(cache.read_text()) if cache.exists() else {}
            states = {}
            for market in markets:
                if args.refresh_all or market not in received or (args.refresh_empty and not received[market]):
                    if args.cached_only:
                        states[market] = "not_cached"
                        continue
                    try:
                        rows = fetch(session, market, day)
                    except (ValueError, ET.ParseError) as error:
                        states[market] = f"request_failed: {error}"
                        print(f"{day:%Y-%m-%d} {market}: {states[market]}", flush=True)
                        continue
                    if rows is None:
                        states[market] = "no_data_or_unpublished"
                        continue
                    received[market] = rows
                    save(json.dumps(received, ensure_ascii=False), cache)
                states[market] = "received"

            if args.refresh_all and any(state.startswith("request_failed") for state in states.values()):
                raise SystemExit("Incomplete refresh; retaining previous daily report.")
            records = [row for market in markets if states[market] == "received"
                       for row in received[market]]
            api = pd.DataFrame(records, columns=["market", "PDLT_NM", "SPCS_NM",
                                                "api_variety_code", "unit", "api_won", "api_kg"])
            missing_names = api.SPCS_NM.eq("")
            verified_names = {"100105": "고냉지배추", "110103": "고냉지무"}
            mapped_names = api.loc[missing_names, "api_variety_code"].map(verified_names).dropna()
            api.loc[mapped_names.index, "SPCS_NM"] = mapped_names
            api["item"] = classify(api)
            api.loc[api.PDLT_NM.eq("감자 수미"), "item"] = "감자 수미"
            for field in ("api_won", "api_kg"):
                api[field] = pd.to_numeric(api[field].astype("string").str.replace(",", "", regex=False),
                                           errors="coerce")
            excluded = (api.api_won.notna() & api.api_kg.notna()
                        & (api.api_won.le(0) | api.api_kg.le(0)))
            valid = (api.unit.eq("12") & api.SPCS_NM.ne("")
                     & api.api_won.gt(0) & api.api_kg.gt(0)).fillna(False)
            report = []
            for market in markets:
                for item in codes.values():
                    mask = api.market.eq(market) & api.item.eq(item) & ~excluded
                    usable = api[mask & valid]
                    unchecked = int((mask & ~valid).sum())
                    state = states[market]
                    report.append({
                        "date": day.date(), "market_code": market, "item": item,
                        "market_status": state,
                        "item_status": ("unconfirmed" if state != "received" else
                                        "needs_review" if unchecked else
                                        "no_item_rows" if usable.empty else "observed"),
                        "api_groups": len(usable), "unchecked_rows": unchecked,
                        "amount": usable.api_won.sum() if not usable.empty else float("nan"),
                        "volume_kg": usable.api_kg.sum() if not usable.empty else float("nan"),
                    })
            market_report = pd.DataFrame(report)
            save(market_report.to_csv(index=False), output / f"coverage_{day:%Y%m%d}.csv")
            daily = market_report.groupby("item", as_index=False).agg(
                amount=("amount", "sum"), volume_kg=("volume_kg", "sum"),
                observed_markets=("item_status", lambda status: status.eq("observed").sum()),
                review_markets=("item_status", lambda status: status.eq("needs_review").sum()),
                no_item_markets=("item_status", lambda status: status.eq("no_item_rows").sum()),
                unconfirmed_markets=("item_status", lambda status: status.eq("unconfirmed").sum()),
            )
            daily.insert(0, "date", day.date())
            price_available = (
                daily.volume_kg.gt(0) & daily.review_markets.eq(0) & daily.unconfirmed_markets.eq(0)
            )
            daily["provisional_price_per_kg"] = daily.amount / daily.volume_kg.where(price_available)
            save(daily.to_csv(index=False), output / f"daily_{day:%Y%m%d}.csv")
            print(f"{day:%Y-%m-%d} 조회 확인 {sum(state == 'received' for state in states.values())}/{len(markets)}시장", flush=True)
    raise SystemExit(0)


if args.gumi_pepper or args.gumi_pepper_direct:
    market = "371501"
    if args.gumi_pepper_direct:
        days = [date]
        filters = {"LARGE": "12", "MID": "07"}
    else:
        days = pd.date_range("2026-09-14", "2026-09-16")
        filters = None
    local = pd.read_parquet(
        root / "data/raw/price/settlement_rows.parquet",
        filters=[("market_code", "=", market), ("item", "=", "건고추"),
                 ("date", "in", list(days))],
    )
    with requests.Session() as session:
        for day in days:
            rows = local[local.date.eq(day)]
            print(f"\n{day:%Y-%m-%d} CSV {len(rows)}행, "
                  f"{(rows.unit_qty * rows.kg_per_unit).sum():,.2f}kg, "
                  f"{(rows.unit_qty * rows.price_per_unit).sum():,.0f}원", flush=True)
            try:
                received = fetch(session, market, day, filters)
            except (ValueError, ET.ParseError) as error:
                print(f"API 실패 {error}")
                continue
            if received is None:
                print("API INFO-200 (데이터 없음 또는 비공개)")
                continue
            pepper_rows = [row for row in received if row["pepper_candidate"]]
            print(f"API 건고추 후보 {len(pepper_rows)}행")
            if not pepper_rows:
                continue
            frame = pd.DataFrame(pepper_rows).drop(columns=["pepper_candidate"])
            frame["included"] = classify(frame).eq("건고추")
            print(frame.to_string(index=False))
            included = frame[frame.included]
            amounts = pd.to_numeric(included.api_won.str.replace(",", ""), errors="coerce")
            quantities = pd.to_numeric(included.api_kg.str.replace(",", ""), errors="coerce")
            if included.empty:
                print("API 집계 포함 행 없음")
            elif amounts.isna().any() or quantities.isna().any() or not included.unit.eq("12").all():
                print("API 금액·물량 또는 단위 확인 필요")
            else:
                print(f"API 집계 {quantities.sum():,.2f}kg (잠정), {amounts.sum():,.0f}원")
    raise SystemExit(0)


records, unconfirmed = [], []
with requests.Session() as session:
    for index, market in enumerate(markets, 1):
        try:
            rows = fetch(session, market, date)
            if rows is None:
                unconfirmed.append(market)
                detail = "INFO-200 (데이터 없음 또는 비공개)"
            else:
                records.extend(rows)
                detail = f"후보 {len(rows)}행"
        except (ValueError, ET.ParseError) as error:
            unconfirmed.append(market)
            detail = f"실패 {error}"
        print(f"[{index}/{len(markets)}] {market} {detail}", flush=True)

if not records:
    raise SystemExit("대조할 API 응답 행이 없습니다.")

api = pd.DataFrame(records)
api["item"] = classify(api)
api.loc[api.PDLT_NM.eq("감자 수미"), "item"] = "감자 수미"
fields = ["market", "api_item_code", "api_variety_code", "api_item_name", "SPCS_NM",
          "corporation", "package", "unit", "api_won", "api_kg"]
pepper = api.loc[api.pepper_candidate, fields + ["item"]].copy()
pepper["status"] = pepper.item.fillna("").eq("건고추").map(
    {True: "집계 포함", False: "집계 제외"}
)
api = api[api.item.notna()].copy()
raw = api[fields].copy()
for field in ("api_won", "api_kg"):
    api[field] = pd.to_numeric(api[field].str.replace(",", "", regex=False), errors="coerce")
zero = api.api_won.eq(0) & api.api_kg.eq(0)
zero_rows = raw.loc[zero].assign(item=api.loc[zero, "item"])
api = api.loc[~zero].copy()
raw = raw.loc[api.index]
ik = (
    (date == pd.Timestamp("2026-09-14"))
    & api.market.eq("350301") & api.corporation.eq("35030102")
    & api.api_variety_code.eq("120101") & api.package.eq("15.000..")
    & api.unit.eq("10") & api.api_won.eq(5520000) & api.api_kg.eq(5400)
)
ik &= ik.sum() == 1
checks = {
    "kg 환산 미확인": api.unit.ne("12") & ~ik,
    "품종명 없음": api.SPCS_NM.eq(""),
    "거래액이 비었거나 양수가 아님": ~api.api_won.gt(0),
    "물량이 비었거나 양수가 아님": ~api.api_kg.gt(0),
}
failed = pd.DataFrame(checks)
api["invalid"] = failed.any(axis=1)
unchecked = raw.loc[api.invalid].assign(item=api.loc[api.invalid, "item"])
unchecked["status"] = failed.loc[api.invalid].apply(
    lambda row: ", ".join(row.index[row]), axis=1
) if not unchecked.empty else ""
nonpositive = (
    api.api_won.notna() & api.api_kg.notna()
    & (api.api_won.le(0) | api.api_kg.le(0))
)
positive = api.loc[~nonpositive].groupby("item").agg(
    positive_won=("api_won", "sum"), positive_kg=("api_kg", "sum"),
    positive_unchecked_rows=("invalid", "sum"),
)
totals = api.groupby("item").agg(
    api_won=("api_won", "sum"), api_kg=("api_kg", "sum"),
    markets=("market", "nunique"), unchecked_rows=("invalid", "sum"),
)
report = baseline[["item", "amount", "volume_kg", "price_per_kg"]].merge(
    totals, on="item", how="left", validate="one_to_one",
).join(positive, on="item", validate="one_to_one")
report["coverage_complete"] = not unconfirmed
report["api_won_per_kg_provisional"] = (report.api_won / report.api_kg).where(
    report.unchecked_rows.eq(0) & report.coverage_complete
)
report["price_diff_pct"] = (report.api_won_per_kg_provisional / report.price_per_kg - 1) * 100
report["positive_api_won_per_kg"] = (report.positive_won / report.positive_kg).where(
    report.positive_unchecked_rows.eq(0) & report.coverage_complete
)
report["positive_price_diff_pct"] = (report.positive_api_won_per_kg / report.price_per_kg - 1) * 100
report["known_gap"] = (date == pd.Timestamp("2026-09-15")) & report.item.eq("건고추")
target = root / "outputs" / "wholesale_verify" / f"national_{date:%Y%m%d}.csv"
report.to_csv(target, index=False, encoding="utf-8-sig")
print(report.round(4).to_string(index=False))
print("미확인 시장", ", ".join(unconfirmed) or "없음")
print("저장", target)
for label, frame in [("구미 건고추 후보", pepper), ("거래액·물량이 모두 0인 제외 행", zero_rows),
                     ("양수 비교에서 제외한 행", raw.loc[nonpositive]),
                     ("단가 계산 보류 행", unchecked)]:
    print(f"\n{label} ({len(frame)}행)")
    if not frame.empty:
        print(frame.fillna("").to_string(index=False))
