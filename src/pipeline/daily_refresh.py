import json
import subprocess
import sys

import pandas as pd

from scripts.wholesale_verify.prepare_inputs import WORK
from src.evaluate.prospective_forecasts import ROOT
from src.models.compare_all_items import ITEMS
from src.pipeline.selected_run import input_signature, operational_inputs


def execute(module, *args):
    print(f"Starting {module} {args}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True, timeout=1200)


def ready_items(path, origin):
    if not path.is_file():
        return set()
    report = pd.read_csv(path, parse_dates=["date"])
    if len(report) != len(ITEMS) or set(report.item) != ITEMS or not report.date.eq(origin).all():
        raise ValueError("Invalid daily settlement report")
    usable = (report.observed_markets.gt(0) & report.review_markets.eq(0)
              & report.unconfirmed_markets.eq(0) & report.provisional_price_per_kg.gt(0)
              & report.volume_kg.gt(0))
    return set(report.loc[usable, "item"])


def origin_ready(path, origin):
    return ready_items(path, origin) == ITEMS


def main():
    origin = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    cache = WORK / "mafra_settlement"
    first = pd.Timestamp("2026-10-01")
    required = list(pd.date_range(first, origin))
    recent = list(pd.date_range(max(first, origin - pd.Timedelta(days=7)), origin))[::-1]
    missing = [day for day in required if not (cache / f"daily_{day:%Y%m%d}.csv").is_file()]
    # Revisit one older unresolved report per cycle, oldest check first.
    older = [day for day in required if day < recent[-1] and day not in missing
             and not origin_ready(cache / f"daily_{day:%Y%m%d}.csv", day)]
    overdue = sorted(older, key=lambda day: (cache / f"daily_{day:%Y%m%d}.csv").stat().st_mtime)[:1]
    failures = []
    for day in dict.fromkeys([*recent, *missing[::-1], *overdue]):
        date = str(day.date())
        try:
            execute("scripts.wholesale_verify.compare_national", "--collect", date, date, "--refresh-all")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            failures.append(date)
            print(f"Collection failed for {date}; continuing.", flush=True)
    reports = ROOT / "outputs"
    ready = [day for day in required if ready_items(cache / f"daily_{day:%Y%m%d}.csv", day)]
    candidates = set(recent + missing + overdue)
    if ready:
        candidates.add(max(ready))
    for latest in sorted(candidates.intersection(ready), reverse=True):
        exports = sorted(reports.glob(f"selected_forecasts_{latest:%Y%m%d}*/short/forecast.json"))
        prices, _, pending = operational_inputs(latest, WORK / f"forecast_inputs_{latest:%Y%m%d}")
        available = ITEMS - pending.keys()
        if pending:
            print(f"Items awaiting complete inputs: {pending}", flush=True)
        signature = input_signature(prices, cache, latest)
        previous = json.loads(exports[-1].read_text()) if exports else {}
        published = (previous.get("input_signature") == signature
                     and {row["item"] for row in previous.get("forecasts", [])} == available)
        if available and not published:
            try:
                execute("src.pipeline.selected_run", "--date", str(latest.date()))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                failures.append(str(latest.date()))
                print(f"Forecast generation failed for {latest.date()}; continuing.", flush=True)
        print(f"Latest usable settlement date: {latest.date()}", flush=True)
    if not ready:
        print("Settlement pending; retaining existing forecast.", flush=True)
    execute("src.db.sync_mysql")
    if failures:
        raise RuntimeError("Refresh failed for dates: " + ", ".join(failures))


if __name__ == "__main__":
    main()
