import subprocess
import sys

import pandas as pd

from scripts.wholesale_verify.prepare_inputs import WORK
from src.evaluate.prospective_forecasts import ROOT
from src.models.compare_all_items import ITEMS


def execute(module, *args):
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True)


def origin_ready(path, origin):
    if not path.is_file():
        return False
    report = pd.read_csv(path, parse_dates=["date"])
    if len(report) != len(ITEMS) or set(report.item) != ITEMS or not report.date.eq(origin).all():
        raise ValueError("Invalid daily settlement report")
    return bool((report.observed_markets.gt(0) & report.review_markets.eq(0)
                 & report.unconfirmed_markets.eq(0) & report.provisional_price_per_kg.gt(0)
                 & report.volume_kg.gt(0)).all())


def main():
    origin = pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None)
    cache = WORK / "mafra_settlement"
    first = pd.Timestamp("2026-10-01")
    for day in pd.date_range(first, origin):
        path = cache / f"daily_{day:%Y%m%d}.csv"
        missing = not path.is_file()
        recent_pending = day >= origin - pd.Timedelta(days=3) and not origin_ready(path, day)
        if missing or recent_pending:
            date = str(day.date())
            execute("scripts.wholesale_verify.compare_national", "--collect", date, date, "--refresh-empty")
    reports = ROOT / "outputs"
    selected = reports / f"selected_forecasts_{origin:%Y%m%d}"
    if not selected.is_dir():
        if origin_ready(cache / f"daily_{origin:%Y%m%d}.csv", origin):
            try:
                execute("src.pipeline.selected_run")
            except subprocess.CalledProcessError:
                execute("src.db.sync_mysql", "--prices-only")
                raise
        else:
            print(f"Settlement pending for {origin.date()}. The timer will retry.", flush=True)
    execute("src.db.sync_mysql")


if __name__ == "__main__":
    main()
