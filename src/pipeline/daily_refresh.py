import json
import subprocess
import sys

import pandas as pd

from scripts.wholesale_verify.prepare_inputs import WORK
from src.evaluate.prospective_forecasts import ROOT
from src.models.compare_all_items import ITEMS
from src.pipeline.selected_run import operational_inputs


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
    recent = list(pd.date_range(max(first, origin - pd.Timedelta(days=3)), origin))[::-1]
    missing = [day for day in required if not (cache / f"daily_{day:%Y%m%d}.csv").is_file()]
    failures = []
    for day in dict.fromkeys([*recent, *missing[::-1]]):
        path = cache / f"daily_{day:%Y%m%d}.csv"
        if origin_ready(path, day):
            continue
        date = str(day.date())
        try:
            execute("scripts.wholesale_verify.compare_national", "--collect", date, date, "--refresh-empty")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            failures.append(date)
            print(f"Collection failed for {date}; continuing.", flush=True)
    reports = ROOT / "outputs"
    ready = [day for day in required if ready_items(cache / f"daily_{day:%Y%m%d}.csv", day)]
    if ready:
        latest = max(ready)
        exports = sorted(reports.glob(f"selected_forecasts_{latest:%Y%m%d}*/short/forecast.json"))
        _, _, pending = operational_inputs(latest, WORK / f"forecast_inputs_{latest:%Y%m%d}")
        available = ITEMS - pending.keys()
        if pending:
            print(f"Items awaiting complete inputs: {pending}", flush=True)
        published = any({row["item"] for row in json.loads(path.read_text())["forecasts"]} == available
                        for path in exports)
        if available and not published:
            try:
                execute("src.pipeline.selected_run", "--date", str(latest.date()))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                execute("src.db.sync_mysql", "--prices-only")
                raise
        print(f"Latest usable settlement date: {latest.date()}", flush=True)
    else:
        print("Settlement pending; retaining existing forecast.", flush=True)
    execute("src.db.sync_mysql")
    if failures:
        raise RuntimeError("Collection failed for dates: " + ", ".join(failures))


if __name__ == "__main__":
    main()
