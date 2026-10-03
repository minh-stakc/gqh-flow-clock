"""PCT-F: the PCT rules on 20 CME futures from Databento (HYPOTHESES.md section 4), in-sample.

Same code as src/strategies/pct.py with the futures universe, next_close execution and the
pre-registered futures costs. Reported as a replication; not used for selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import analysis as AN  # noqa: E402
from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402
from src.strategies import pct  # noqa: E402

ROOTS = ["ES", "NQ", "RTY", "YM", "ZT", "ZF", "ZN", "ZB", "CL", "NG", "GC", "SI", "HG",
         "6E", "6J", "6B", "6A", "6C", "ZC", "ZS"]
TICKERS = [f"F_{r}" for r in ROOTS]
COST_BPS = {t: (5.0 if t[2:] in {"NG", "RTY", "ZC", "ZS", "6A", "6C"} else 1.5) for t in TICKERS}
EXEC = "next_close"
FAMILY = "PCT_F"


def weights(period: str, **params):
    p = {**pct.BASE_PARAMS, **params}
    p.pop("cost_mult", None)
    return pct.decision_weights(period=period, tickers=TICKERS, **p)


def run(period: str = "IS") -> dict:
    ohlc = E.load_ohlc(TICKERS, period)
    rf = E.load_rf(period)
    out = {}
    for name, ov in [("base", {}), ("tsmom_benchmark", {"measure": "none"}), ("id_variant", {"measure": "ID"})]:
        w = weights(period, **ov)
        r = E.run_backtest(f"PCTF_{name}", FAMILY, w, ohlc, rf, period=period, params={**pct.BASE_PARAMS, **ov, "universe": "futures20"},
                           exec=EXEC, cost_bps=COST_BPS)
        r2 = E.run_backtest(f"PCTF_{name}", FAMILY, w, ohlc, rf, period=period, params={**pct.BASE_PARAMS, **ov, "universe": "futures20"},
                            exec=EXEC, cost_bps=COST_BPS, cost_mult=2.0)
        out[name] = {"stats": r.stats, "stats_2x": r2.stats}
        if name == "base":
            out[name]["subperiods"] = AN.subperiod_table(r).to_dict("records")
            out[name]["factor"] = AN.factor_table(r)
            out[name]["capacity"] = AN.capacity_curve(r, ohlc).to_dict("records")
    return out


def main() -> None:
    out = run("IS")
    (C.RESULTS_DIR / "is" / "pct_f.json").write_text(json.dumps(out, indent=2, default=str))
    for k, v in out.items():
        print(k, round(v["stats"]["sharpe"], 3), "2x:", round(v["stats_2x"]["sharpe"], 3))


if __name__ == "__main__":
    main()
