"""Post-hoc diagnostics (after the out-of-sample run; nothing here changes the strategy):

1. Sharpe ratio of the submitted strategy by calendar year, and each stream's excess-return contribution
   by year (the stream construction is src/forward.py F1, verified identical to the submission);
2. the same strategy re-sliced as if the data started in 2017 (the brief's 20 %-or-2-years rule then
   binds at 20 %, so the out-of-sample window starts two weeks later).

Writes results/oos_by_year.json. Uses period "FWD" (all cached data), so no unlock flag is needed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402
from src import ensemble as EN  # noqa: E402
from src import forward as F  # noqa: E402


def _sr(x: pd.Series) -> float:
    return float(x.mean() / x.std() * math.sqrt(C.TRADING_DAYS)) if len(x) > 2 and x.std() > 0 else float("nan")


def main() -> None:
    fr = F.stream_frames("F1", "FWD")
    labels = list(fr["ex"].columns)
    out = EN.combine(fr["ex"], fr["gr"], fr["rf"], labels, "minvar_lw", 0.08, False)
    ex = out["excess"].loc[: C.OOS_END]
    contrib = (out["lam"][labels] * out["k"].values[:, None] * fr["ex"][labels].reindex(out["lam"].index)).loc[: C.OOS_END]
    periods = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(2007, 2024)]
    periods += [("2024_IS", "2024-01-01", C.IS_END), ("2024Q4_OOS", C.OOS_START, "2024-12-31"),
                ("2025_OOS", "2025-01-01", "2025-12-31"), ("2026_OOS", "2026-01-01", C.OOS_END)]
    by_year = {}
    for lab, a, b in periods:
        e = ex.loc[a:b]
        by_year[lab] = {"sharpe": _sr(e), "excess_sum": float(e.sum()),
                        "contrib": {s: float(contrib[s].loc[a:b].sum()) for s in labels}}
    full_is = [by_year[str(y)]["sharpe"] for y in range(2007, 2024)]
    # 2017-start re-slice under the 20 %-or-2-years rule
    cal = ex.loc["2017-01-01":].index
    k = min(int(round(0.2 * len(cal))), len(ex.loc[C.OOS_START:]))
    oos17 = cal[-k]
    res = {"by_year": by_year, "worst_full_is_year_sharpe": float(min(full_is)),
           "start2017": {"oos_start": str(oos17.date()), "oos_days": int(k),
                         "is_sharpe": _sr(ex.loc["2017-01-03": cal[-k - 1]]), "oos_sharpe": _sr(ex.loc[oos17:])}}
    (C.RESULTS_DIR / "oos_by_year.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res["by_year"][k] for k in ("2024Q4_OOS", "2025_OOS", "2026_OOS")}, indent=1))
    print("worst full IS year:", round(res["worst_full_is_year_sharpe"], 2), "| 2017 start:", res["start2017"])


if __name__ == "__main__":
    main()
