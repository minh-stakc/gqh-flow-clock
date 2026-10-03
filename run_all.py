"""Reproduce every number in the quant note.

    python data/download.py                    # free data (Yahoo, FRED, Ken French, Treasury auctions)
    python data/download_databento.py          # optional: CME futures replication (needs DATABENTO_API_KEY)
    python run_all.py                          # in-sample research: all strategies, ensemble search, selection
    GQH_OOS_UNLOCK=1 python run_all.py --final # out-of-sample evaluation (the team ran it once; judges may rerun)

Outputs: results/is/*.json|csv (in-sample), results/oos/*.json|csv (out-of-sample), results/figures/,
results/trials.csv (every backtest), results/oos_log.csv (every out-of-sample evaluation).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import analysis as AN  # noqa: E402
from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402

MODULES = ["ifc_rebalance", "ifc_dash", "ifc_treasury", "ifc_fx", "ifc_auction", "pct"]


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def in_sample(skip_modules: bool = False) -> None:
    for m in ([] if skip_modules else MODULES):
        print(f"== {m}")
        importlib.import_module(f"scripts.run_{m}").main()
    from src.strategies import ifc

    print("== IFC composite")
    out = ifc.run("IS")
    res = out.pop("_results")
    (C.RESULTS_DIR / "is" / "ifc.json").write_text(json.dumps(_jsonable(out), indent=2))
    pd.DataFrame({k: v.returns for k, v in res.items()}).to_csv(C.RESULTS_DIR / "is" / "ifc_returns.csv")
    if (C.DATA_DIR / "futures_1600.parquet").exists():
        print("== IFC-F (Databento futures replication)")
        outf = ifc.run("IS", futures=True)
        outf.pop("_results")
        (C.RESULTS_DIR / "is" / "ifc_f.json").write_text(json.dumps(_jsonable(outf), indent=2))
    print("== PCT-F (Databento futures replication)")
    if (C.DATA_DIR / "futures_daily.parquet").exists():
        importlib.import_module("scripts.run_pct_f").main()
    print("== FTE ensemble search")
    importlib.import_module("scripts.run_fte").main()
    print("== selection (committed rule) -> results/selection.json")
    importlib.import_module("scripts.select").main()


def final_oos() -> None:
    if not E.oos_unlocked():
        raise SystemExit("set GQH_OOS_UNLOCK=1 to run the out-of-sample evaluation")
    from scripts import diagnostics, run_oos

    run_oos.main()
    diagnostics.main()   # post-hoc explanations only (event-time effects, rolling Sharpe, capacity)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="run the out-of-sample evaluation")
    ap.add_argument("--skip-modules", action="store_true", help="in-sample: reuse the per-module runs")
    args = ap.parse_args()
    final_oos() if args.final else in_sample(args.skip_modules)
