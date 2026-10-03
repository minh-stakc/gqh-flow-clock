"""Export the submitted strategy's (F1) daily held ETF weights and reference returns for replay_weights.py.

The weights are those of validation/backtrader_crosscheck.py section 4 (submitted_strategy()): W_t = sum_s
lam_s,t k_t held_s,t, the ETF weights held over return day t. Writes to --out:

  f1_weights.csv        W, one column per ETF, on the NYSE calendar
  f1_engine_simple.csv  our engine on W: rf = 0, per-ETF costs on the netted trades, no borrow fee (section 4's
                        "engine_simple"; compare with --reference-rf zero)
  f1_official.csv       the official F1 daily net returns, cash earning the T-bill (forward.returns("F1", "FWD");
                        compare with --reference-rf tbill)

Run from the repo root:
    GQH_DATA_DIR=<cache> python validation/starter_kit/export_f1.py --out <dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import config as C                          # noqa: E402
from src import engine as E                          # noqa: E402
from validation import backtrader_crosscheck as X    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output folder")
    args = ap.parse_args()
    if os.environ.get("GQH_OOS_UNLOCK"):
        raise SystemExit("unset GQH_OOS_UNLOCK: only the FWD period is needed")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sub = X.submitted_strategy()
    W, idx, ohlc = sub["W"], sub["idx"], sub["ohlc"]
    cost = {t: C.cost_bps(t) for t in W.columns}
    net, _, _, held, _ = E.simulate(W.shift(-2), ohlc, X.zero_rf(W.index), exec="next_close", cost_bps=cost)
    borrow = (held.clip(upper=0).abs() * C.SHORT_BORROW_BPS_PER_YEAR / 1e4 / C.TRADING_DAYS).sum(axis=1)
    W.rename_axis("date").to_csv(out / "f1_weights.csv")
    (net + borrow).reindex(idx).rename("ret").rename_axis("date").to_csv(out / "f1_engine_simple.csv")
    sub["out"]["net"].rename("ret").rename_axis("date").to_csv(out / "f1_official.csv")
    print(f"wrote {len(W.columns)} ETF weights over {len(W)} sessions and two reference series to {out}")


if __name__ == "__main__":
    main()
