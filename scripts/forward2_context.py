"""Historical context of the frozen forward test 2 strategies (FORWARD_TEST_2.md), computed after the freeze.

    python scripts/forward2_context.py      # writes forward/historical_context2.json

Context only: nothing here chooses or changes anything in the frozen test. The 2024-10 .. 2026-10 window was seen
before the freeze (it is the competition's out-of-sample window), so its numbers are descriptive, not a validation.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402
from src import forward2 as F2  # noqa: E402

KEYS = ("start", "end", "ann_return", "ann_vol", "sharpe", "max_drawdown", "worst_month")


def _stats(net: pd.Series, rf: pd.Series) -> dict:
    ex = net - rf.reindex(net.index).ffill().fillna(0.0)
    st = E.perf_stats(net, ex, pd.Series(0.0, index=net.index))
    return {k: st[k] for k in KEYS}


def _weekly_corr(a: pd.Series, ticker: str) -> dict:
    try:
        import yfinance as yf

        px = yf.download(ticker, start=str(a.index[0].date()), auto_adjust=True, progress=False)["Close"].squeeze()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    b = px.pct_change().dropna()
    j = pd.concat([(1 + a).resample("W-FRI").prod() - 1, (1 + b).resample("W-FRI").prod() - 1],
                  axis=1, join="inner").dropna()
    out = {}
    for label, lo, hi in (("in_sample", None, C.IS_END), ("out_of_sample", C.OOS_START, C.OOS_END)):
        blk = j.loc[lo:hi]
        if len(blk) >= 8:
            out[label] = {"start": str(blk.index[0].date()), "n_weeks": len(blk),
                          "corr": float(blk.iloc[:, 0].corr(blk.iloc[:, 1]))}
    return out


def main() -> None:
    head = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--short", "forward-test-2-2026-10-03"],
                          cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rf = E.load_rf("FWD")
    out = {"computed_after_freeze_commit": f"{head} (tag forward-test-2-2026-10-03)",
           "note": "Context only: history of the frozen forward test 2 strategies and benchmarks, from each one's "
                   "first traded day. The out-of-sample window was seen before the freeze. Not used to choose or "
                   "change anything.",
           "strategies": {}}
    for name in F2.STRATEGIES + F2.BENCHMARKS:
        net = F2.returns(name, "FWD").loc[:C.OOS_END]
        ex = net - rf.reindex(net.index).ffill().fillna(0.0)
        traded = ex[ex.abs() > 1e-12]
        if traded.empty:
            continue
        net = net.loc[traded.index[0]:]
        out["strategies"][name] = {"in_sample": _stats(net.loc[:C.IS_END], rf),
                                   "out_of_sample": _stats(net.loc[C.OOS_START:C.OOS_END], rf)}
        if name == "S1":
            out["strategies"][name]["weekly_corr"] = {f: _weekly_corr(net, f) for f in ("DBMF", "AQMIX")}
        s = out["strategies"][name]
        print(f"{name:12s} IS {s['in_sample']['start']} SR {s['in_sample']['sharpe']:+.2f} "
              f"MDD {s['in_sample']['max_drawdown']:.1%} | OOS SR {s['out_of_sample']['sharpe']:+.2f} "
              f"MDD {s['out_of_sample']['max_drawdown']:.1%}")
    path = C.FWD_DIR / "historical_context2.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
