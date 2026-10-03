"""Single source of truth for dates, costs and paths.

The out-of-sample (OOS) rule of the Systematic Trading brief: hold out the most
recent 20% of history or the most recent 2 years, whichever is shorter. Our
history starts in 2005, so the 2-year cap binds.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# raw downloads, git-ignored; override with GQH_DATA_DIR (e.g. to keep caches off a slow drive)
DATA_DIR = Path(os.environ.get("GQH_DATA_DIR", ROOT / "data" / "cache"))
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"
TRIALS_LOG = RESULTS_DIR / "trials.csv"      # every backtest ever run (appended)
OOS_LOG = RESULTS_DIR / "oos_log.csv"        # every OOS evaluation (should be one line per strategy)

HISTORY_START = "2005-01-03"
IS_END = "2024-10-02"        # last in-sample session
OOS_START = "2024-10-03"     # first out-of-sample session
OOS_END = "2026-10-02"       # last session available at the event (Fri Oct 2, 2026)

TRADING_DAYS = 252

# One-way transaction cost in basis points per unit of turnover (commission +
# half-spread + slippage). Tier-1 ETFs quote 1-cent spreads on $100-$600 prices
# (0.08-0.5 bp half-spread); we charge several times that to cover slippage and
# market impact at the sizes discussed in the capacity section.
COST_BPS_TIER1 = 3.0
COST_BPS_TIER2 = 5.0
COST_BPS_DEFAULT = 10.0
TIER1 = {
    "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "SHY", "IEI", "TLH", "LQD", "HYG",
    "GLD", "AGG", "VTI", "BND", "DIA",
}
TIER2 = {
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "VNQ", "SLV", "USO",
    "DBC", "UUP", "FXE", "FXY", "TIP", "EWJ", "VGK", "MDY", "EMB", "BIL",
}

# Annual borrow fee charged on short ETF positions (general-collateral ETFs).
SHORT_BORROW_BPS_PER_YEAR = 30.0


COST_BPS_FUTURES = 1.5   # default for CME futures; strategies pass their pre-registered map


def is_future(ticker: str) -> bool:
    """Databento futures: F_<root> (daily bars) or <root>16 (sampled at 16:00 ET)."""
    return ticker.startswith("F_") or ticker.endswith("16")


def cost_bps(ticker: str) -> float:
    if is_future(ticker):
        return COST_BPS_FUTURES
    if ticker in TIER1:
        return COST_BPS_TIER1
    if ticker in TIER2:
        return COST_BPS_TIER2
    return COST_BPS_DEFAULT
