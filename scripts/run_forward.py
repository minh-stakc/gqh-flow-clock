"""Forward test runner (FORWARD_TEST.md).

    python scripts/run_forward.py positions   # target weights for the next session, from the latest data
    python scripts/run_forward.py evaluate    # performance on return days >= FWD_START (needs fresh data)

Fresh data: re-run `python data/download.py` and, for F2/F3, `GQH_DATA_END=<date> python
data/download_databento.py` into an empty GQH_DATA_DIR. Every evaluation is appended to
forward/forward_log.csv and written to forward/report_<last date>.json, whatever it shows.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import analysis as AN  # noqa: E402
from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402
from src import forward as F  # noqa: E402


def positions() -> None:
    last = pd.Timestamp(E.data_end("FWD"))
    pad = list(pd.bdate_range(last + pd.Timedelta(days=1), periods=2))
    rows = []
    for name in F.STRATEGIES:
        pos = F.positions_for_next_session(name, "FWD", pad)
        for tk, w in pos.items():
            rows.append({"strategy": name, "instrument": tk, "weight_of_nav": round(float(w), 6),
                         "held_over_return_day": str(pad[-1].date()), "data_through": str(last.date())})
    C.FWD_DIR.mkdir(parents=True, exist_ok=True)
    out = C.FWD_DIR / f"positions_for_{pad[-1].date()}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    print(pd.DataFrame(rows).to_string(index=False))


def evaluate() -> None:
    last = E.data_end("FWD")
    if pd.Timestamp(last) < pd.Timestamp(C.FWD_START):
        print(f"no forward data yet: latest cached session is {last}, forward window starts {C.FWD_START}")
        return
    rf = E.load_rf("FWD")
    report = {"as_of": last, "forward_start": C.FWD_START, "primary": F.PRIMARY, "strategies": {}}
    for name in F.STRATEGIES + F.BENCHMARKS:
        net = F.returns(name, "FWD").loc[C.FWD_START:]
        if len(net) == 0:
            continue
        ex = net - rf.reindex(net.index).ffill().fillna(0.0)
        to = pd.Series(0.0, index=net.index)
        res = E.BacktestResult(f"FWD_{name}", "FWD", net, net, ex, to, pd.DataFrame(index=net.index),
                               pd.Series(0.0, index=net.index), {"strategy": name})
        res.stats = E.perf_stats(net, ex, to) if len(net) > 2 else {"n_days": len(net)}
        st = dict(res.stats)
        st["n_days"] = len(net)
        st["cumulative_return"] = float((1 + net).prod() - 1)
        if len(ex) >= 63:
            st["bootstrap_sharpe_90"] = AN.bootstrap_sharpe_ci(ex)
        st["sharpe_standard_error"] = float(math.sqrt(252 / max(len(ex), 1)))
        report["strategies"][name] = st
        if len(net) > 2:
            E.log_trial(res, "FORWARD", 1.0, note="forward test evaluation")
    C.FWD_DIR.mkdir(parents=True, exist_ok=True)
    (C.FWD_DIR / f"report_{last}.json").write_text(json.dumps(report, indent=2, default=str))
    for k, v in report["strategies"].items():
        print(k, {x: v.get(x) for x in ("n_days", "sharpe", "ann_return", "max_drawdown", "cumulative_return")})


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "evaluate"
    {"positions": positions, "evaluate": evaluate}[cmd]()
