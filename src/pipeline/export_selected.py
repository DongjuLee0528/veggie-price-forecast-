import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.evaluate.compare_candidates import ROUTES
from src.models.compare_all_items import ITEMS

EXPECTED = {("short", item, horizon): model for item, horizon, model, _ in ROUTES}
EXPECTED["mid", "건고추", 2] = "sarimax"


def export_selected(short_dir, mid_dir, candidate_dir, output):
    selection = json.loads((candidate_dir / "comparison.json").read_text(encoding="utf-8"))
    candidates = selection["forecasts"]
    selected = {(row["term"], row["item"], row["horizon"]): row for row in candidates}
    if (len(candidates) != len(EXPECTED) or set(selected) != set(EXPECTED)
            or any(row["status"] != "candidate" or row["candidate_model"] != EXPECTED[key]
                   or not math.isfinite(row["candidate_price"]) or row["candidate_price"] <= 0
                   for key, row in selected.items())):
        raise ValueError("Expected all five verified candidate routes")
    hashes = selection["model_sha256"]
    if set(hashes) != {model for (term, _, _), model in EXPECTED.items()
                       if term == "short"}:
        raise ValueError("Short-term model hashes are missing")
    for name, expected in hashes.items():
        actual = hashlib.sha256((candidate_dir / f"{name}.txt").read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Model checksum mismatch: {name}")

    reports = {}
    for term, directory, horizons in (("short", short_dir, (1, 5, 10)), ("mid", mid_dir, (2, 3))):
        report = json.loads((directory / "forecast.json").read_text(encoding="utf-8"))
        rows = report["forecasts"]
        expected = {(item, horizon) for item in ITEMS for horizon in horizons}
        if (len(rows) != len(expected) or {(row["item"], row["horizon"]) for row in rows} != expected
                or report["base_date"] != selection["base_date"] or report["source"] != selection["source"]):
            raise ValueError(f"Incompatible {term} forecast export")
        for row in rows:
            key = (term, row["item"], row["horizon"])
            if key not in selected:
                continue
            replacement = selected[key]
            if (not math.isclose(row["prediction_krw_per_kg"], replacement["existing_price"], rel_tol=1e-12)
                    or any(row.get(field) != replacement[field] for field in ("target_start", "target_end"))):
                raise ValueError(f"Saved baseline or target does not match: {key}")
            row["prediction_krw_per_kg"] = replacement["candidate_price"]
            row["model"] = replacement["candidate_model"]
            if "model_route" in row:
                row["model_route"] = "selected_candidate"
        report["candidate_generated_at"] = selection["generated_at"]
        report["candidate_run_type"] = selection["run_type"]
        reports[term] = report

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "export"
        stage.mkdir()
        for term, report in reports.items():
            folder = stage / term
            folder.mkdir()
            pd.DataFrame(report["forecasts"]).to_csv(folder / "predictions.csv", index=False)
            (folder / "forecast.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description="Export selected short- and mid-term forecast routes together.")
    for option in ("short-dir", "mid-dir", "candidate-dir", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    export_selected(args.short_dir, args.mid_dir, args.candidate_dir, args.output)
    print(f"Saved 15 short-term and 10 mid-term forecasts: {args.output}")


if __name__ == "__main__":
    main()
