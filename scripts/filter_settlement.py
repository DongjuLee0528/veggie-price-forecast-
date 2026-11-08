import argparse
import glob
import os
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

USECOLS = [
    "AUC_YMD",
    "WHSL_MRKT_CODE",
    "WHSL_MRKT_NM",
    "PDLT_NM",
    "SPCS_NM",
    "UNIT_QYT",
    "PRCE",
    "KG_UNIT_CNVR_QYT",
]

RULES = {
    "배추": {"pdlt": "배추", "include": None, "exclude": ["쌈", "뿌리", "우거지", "절임", "배양채", "수입"]},
    "무": {"pdlt": "무", "include": None, "exclude": ["말랭이", "건무", "단무지", "달랑", "다발", "솎음", "소궁기", "자색", "수입"]},
    "양파": {"pdlt": "양파", "include": None, "exclude": ["깐양파", "자주", "수입"]},
    "깐마늘": {"pdlt": "마늘", "include": "깐마늘", "exclude": ["수입"]},
    "건고추": {"pdlt": "건고추", "include": None, "exclude": ["수입"]},
}

TARGET_PDLT = {rule["pdlt"] for rule in RULES.values()}

SCHEMA = pa.schema([
    ("date", pa.timestamp("ns")),
    ("market_code", pa.string()),
    ("market_name", pa.string()),
    ("item", pa.string()),
    ("variety", pa.string()),
    ("unit_qty", pa.float64()),
    ("price_per_unit", pa.float64()),
    ("kg_per_unit", pa.float64()),
])


def classify(df):
    item = pd.Series(pd.NA, index=df.index, dtype="object")
    spcs = df["SPCS_NM"].fillna("")
    for name, rule in RULES.items():
        mask = df["PDLT_NM"] == rule["pdlt"]
        if rule["include"]:
            mask &= spcs.str.contains(rule["include"], regex=False)
        for word in rule["exclude"]:
            mask &= ~spcs.str.contains(word, regex=False)
        item[mask] = name
    return item


def process_file(path, chunksize, excluded_counts):
    parts = []
    reader = pd.read_csv(
        path,
        encoding="cp949",
        encoding_errors="replace",
        usecols=USECOLS,
        dtype=str,
        chunksize=chunksize,
    )
    for chunk in reader:
        chunk = chunk[chunk["PDLT_NM"].isin(TARGET_PDLT)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk["item"] = classify(chunk)

        dropped = chunk[chunk["item"].isna()]
        for key, n in dropped.groupby(["PDLT_NM", "SPCS_NM"], dropna=False).size().items():
            excluded_counts[key] = excluded_counts.get(key, 0) + int(n)

        chunk = chunk[chunk["item"].notna()]
        for col in ["UNIT_QYT", "PRCE", "KG_UNIT_CNVR_QYT"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk[(chunk["UNIT_QYT"] > 0) & (chunk["PRCE"] > 0) & (chunk["KG_UNIT_CNVR_QYT"] > 0)]
        if chunk.empty:
            continue

        out = pd.DataFrame({
            "date": pd.to_datetime(chunk["AUC_YMD"], format="%Y%m%d", errors="coerce"),
            "market_code": chunk["WHSL_MRKT_CODE"],
            "market_name": chunk["WHSL_MRKT_NM"],
            "item": chunk["item"],
            "variety": chunk["SPCS_NM"],
            "unit_qty": chunk["UNIT_QYT"],
            "price_per_unit": chunk["PRCE"],
            "kg_per_unit": chunk["KG_UNIT_CNVR_QYT"],
        })
        parts.append(out[out["date"].notna()])
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="/work/raw/at-settlement")
    parser.add_argument("--out", default="/work/output/at-settlement")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.src, "*.csv")))
    if not files:
        sys.exit(f"No CSV files in {args.src}")

    rows_path = os.path.join(args.out, "settlement_rows.parquet")
    writer = None
    skipped = []
    excluded_counts = {}
    total_rows = 0

    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        if os.path.getsize(path) == 0:
            skipped.append(name)
            print(f"[{i}/{len(files)}] SKIP empty file: {name}", flush=True)
            continue
        df = process_file(path, args.chunksize, excluded_counts)
        if df is None:
            print(f"[{i}/{len(files)}] no target rows: {name}", flush=True)
            continue
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(rows_path, SCHEMA, compression="zstd")
        writer.write_table(table)
        total_rows += len(df)
        print(f"[{i}/{len(files)}] {len(df):>8,} rows  {name}", flush=True)

    if writer is None:
        sys.exit("No target rows found")
    writer.close()

    rows = pd.read_parquet(rows_path, columns=["date", "item", "unit_qty", "price_per_unit", "kg_per_unit"])
    rows["amount"] = rows["unit_qty"] * rows["price_per_unit"]
    rows["volume_kg"] = rows["unit_qty"] * rows["kg_per_unit"]
    daily = (
        rows.groupby(["date", "item"], as_index=False)
        .agg(amount=("amount", "sum"), volume_kg=("volume_kg", "sum"), n_rows=("amount", "size"))
    )
    daily["price_per_kg"] = daily["amount"] / daily["volume_kg"]
    daily = daily.sort_values(["item", "date"])
    daily.to_parquet(os.path.join(args.out, "settlement_daily.parquet"), index=False)
    daily.to_csv(os.path.join(args.out, "settlement_daily.csv"), index=False, encoding="utf-8")

    excluded = pd.DataFrame(
        [(k[0], k[1], v) for k, v in excluded_counts.items()],
        columns=["PDLT_NM", "SPCS_NM", "rows"],
    ).sort_values("rows", ascending=False)
    excluded.to_csv(os.path.join(args.out, "excluded_varieties.csv"), index=False, encoding="utf-8")

    print("\n== Summary ==")
    print(f"target rows: {total_rows:,}")
    print(f"skipped empty files ({len(skipped)}): {skipped}")
    print(daily.groupby("item")["date"].agg(["min", "max", "count"]).to_string())
    print("\nprice_per_kg by item (median):")
    print(daily.groupby("item")["price_per_kg"].median().round(0).to_string())


if __name__ == "__main__":
    main()
