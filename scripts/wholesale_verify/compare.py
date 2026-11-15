import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

import pandas as pd
import requests

folder = Path(__file__).resolve().parents[2] / "outputs" / "wholesale_verify"
baseline = pd.read_csv(folder / "wholesale_compare_baseline.csv", dtype=str)
key = os.getenv("MAFRA_WHOLESALE_API_KEY", "").strip()
if not key:
    raise SystemExit("MAFRA_WHOLESALE_API_KEY 환경변수가 비어 있습니다.")
base = (
    f"http://211.237.50.150:7080/openapi/{quote(key, safe='')}"
    "/xml/Grid_20240625000000000653_1"
)


def fetch(session, date, market):
    rows, start, total = [], 1, None
    while total is None or start <= total:
        end = start + 999 if total is None else min(start + 999, total)
        try:
            response = session.get(
                f"{base}/{start}/{end}",
                params={"SALEDATE": date, "WHSALCD": market},
                timeout=30,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise ValueError(type(error).__name__) from None
        if response.status_code != 200:
            raise ValueError(f"HTTP {response.status_code}")
        root = ET.fromstring(response.content)
        if root.findtext(".//code") != "INFO-000":
            raise ValueError("정상 처리 코드가 반환되지 않았습니다.")
        count = int(root.findtext("totalCnt", "-1"))
        page = root.findall(".//row")
        if count < 0 or (total is not None and total != count):
            raise ValueError("전체 행 수 불일치")
        total = count
        if len(page) != min(end, total) - start + 1:
            raise ValueError(f"페이지 행 수 불일치 ({start}~{end})")
        rows.extend(page)
        start = end + 1
    return rows


results = []
with requests.Session() as session:
    for (date, market), group in baseline.groupby(["AUC_YMD", "WHSL_MRKT_CODE"]):
        print(f"{date} / {market} 조회 중", flush=True)
        try:
            rows = fetch(session, date, market)
        except (ValueError, ET.ParseError) as error:
            results.extend(dict(row, status=f"조회 실패: {error}") for row in group.to_dict("records"))
            continue
        indexed = {}
        for row in rows:
            code = "".join(row.findtext(field, "").strip() for field in ("LARGE", "MID", "SMALL"))
            indexed.setdefault(code, []).append(row)
        for record in group.to_dict("records"):
            matches = indexed.get(record["SPCS_CODE"], [])
            record["api_rows"] = len(matches)
            if not matches:
                results.append(dict(record, status="API 일치 코드 없음"))
                continue
            record["api_names"] = " / ".join(sorted({
                row.findtext("SMALLNAME", "").strip() for row in matches
            }))
            try:
                amount, quantity = (
                    sum(Decimal(row.findtext(field, "").strip().replace(",", "")) for row in matches)
                    for field in ("TOTAMT", "TOTQTY")
                )
            except InvalidOperation:
                results.append(dict(record, status="금액·중량 확인 필요"))
                continue
            record.update(api_won=amount, api_totqty=quantity)
            if not all(row.findtext("DANCD", "").strip() == "12" for row in matches):
                results.append(dict(record, status="단위 확인 필요"))
                continue
            csv_amount, csv_kg = Decimal(record["csv_won"]), Decimal(record["csv_kg"])
            if min(amount, quantity, csv_amount, csv_kg) <= 0:
                results.append(dict(record, status="금액·중량 확인 필요"))
                continue
            won_diff, kg_diff = amount - csv_amount, quantity - csv_kg
            record.update(
                won_diff=won_diff,
                kg_diff=kg_diff,
                csv_won_per_kg=round(csv_amount / csv_kg, 2),
                api_won_per_kg=round(amount / quantity, 2),
                price_diff_pct=round((amount / quantity / (csv_amount / csv_kg) - 1) * 100, 2),
                status="일치" if abs(won_diff) < Decimal("0.01") and abs(kg_diff) < Decimal("0.001") else "차이 있음",
            )
            results.append(record)

report = pd.DataFrame(results)
target = folder / "wholesale_compare_result.csv"
report.to_csv(target, index=False, encoding="utf-8-sig")
print(pd.crosstab(report["item"], report["status"]).to_string())
print(f"저장 {target}")
