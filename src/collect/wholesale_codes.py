import argparse
import json
from urllib.parse import unquote

import requests

from src.config import require_api_key


def fetch_unit(session, key, code):
    try:
        response = session.get(
            "https://apis.data.go.kr/B552845/katCode/units",
            params={"serviceKey": unquote(key.strip()), "pageNo": 1, "numOfRows": 100,
                    "returnType": "json", "cond[unit_cd::EQ]": str(code)},
            timeout=(5, 20),
        )
    except requests.RequestException:
        raise RuntimeError("Unit code request failed; serviceKey=***") from None
    if response.status_code != 200:
        raise RuntimeError(f"Unit code HTTP {response.status_code}; serviceKey=***")
    try:
        report = response.json()["response"]
        if str(report["header"]["resultCode"]) != "0":
            raise ValueError("Unsuccessful unit code response")
        rows = report["body"]["items"]["item"]
        if not isinstance(rows, list) or any(str(row["unit_cd"]) != str(code) for row in rows):
            raise ValueError("Unexpected unit codes")
        return [{"unit_cd": str(row["unit_cd"]), "unit_nm": row["unit_nm"]} for row in rows]
    except (ValueError, KeyError, TypeError):
        raise RuntimeError("Invalid unit code response; serviceKey=***") from None


def main():
    parser = argparse.ArgumentParser(description="Look up official wholesale unit codes.")
    parser.add_argument("--unit", default="10")
    args = parser.parse_args()
    try:
        with requests.Session() as session:
            rows = fetch_unit(session, require_api_key("DATA_GO_KR_KEY"), args.unit)
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(rows, ensure_ascii=False))


if __name__ == "__main__":
    main()
