"""Month-end calendar helpers shared by every calendar strategy.

Terminology (see HYPOTHESES.md section 0): for month m, T is its last trading day; T-j is the j-th
trading day before T; T+j is the j-th trading day of the following month. "Return day t" is
close(t-1) -> close(t). A hold window [a, b] means the position is established at the close of a-1
and closed at the close of b.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SETTLEMENT_T2 = pd.Timestamp("2017-09-05")   # US equities T+3 -> T+2
SETTLEMENT_T1 = pd.Timestamp("2024-05-28")   # US equities T+2 -> T+1


def month_offsets(cal: pd.DatetimeIndex) -> pd.DataFrame:
    """Per trading day: its month's last trading day T, offset to own T (<= 0), offset from the
    previous month's T (>= 1), and the previous month's T.

    The last month in ``cal`` may be incomplete; its T is the last date in ``cal`` only if the
    calendar actually ends on a month end, otherwise offsets to own T are NaN for that month.
    """
    cal = pd.DatetimeIndex(cal).sort_values()
    s = pd.Series(np.arange(len(cal)), index=cal)
    per = cal.to_period("M")
    last_idx = s.groupby(per).transform("max")
    df = pd.DataFrame(index=cal)
    df["month"] = per
    df["i"] = s.values
    df["T_i"] = last_idx.values
    # the final month is complete only if the next calendar day after its last date is a new month
    final_month = per[-1]
    complete = cal[-1] + pd.offsets.BDay(1)
    if complete.to_period("M") == final_month:   # the data stops mid-month: T unknown
        df.loc[df["month"] == final_month, "T_i"] = np.nan
    df["off_own"] = df["i"] - df["T_i"]                       # 0 on T, -j on T-j
    prev_T = s.groupby(per).min() - 1                           # index of previous month's T
    df["prevT_i"] = per.map(prev_T).values
    df["off_prev"] = df["i"] - df["prevT_i"]                   # 1 on first trading day of month
    df.loc[df["prevT_i"] < 0, "off_prev"] = np.nan
    df["T"] = [cal[int(x)] if not np.isnan(x) else pd.NaT for x in df["T_i"]]
    return df


def window_mask(cal: pd.DatetimeIndex, a: int, b: int) -> pd.Series:
    """True on return days in [T+a, T+b] for any month (a, b relative to T; a <= b).

    Days before/at T are matched against their own month's T, days after it against the previous
    month's T, so windows like [T-3, T+3] that span the turn of the month work.
    """
    mo = month_offsets(cal)
    own = mo["off_own"]
    prev = mo["off_prev"]
    m = pd.Series(False, index=mo.index)
    if a <= 0:
        m |= own.between(a, min(b, 0))
    if b >= 1:
        m |= prev.between(max(a, 1), b)
    return m.fillna(False).astype(bool)


def month_T_for_window(cal: pd.DatetimeIndex, a: int, b: int) -> pd.Series:
    """For days inside window [T+a, T+b], the T that the window belongs to (NaT elsewhere)."""
    mo = month_offsets(cal)
    out = pd.Series(pd.NaT, index=mo.index, dtype="datetime64[ns]")
    if a <= 0:
        sel = mo["off_own"].between(a, min(b, 0))
        out[sel] = mo.loc[sel, "T"]
    if b >= 1:
        sel = mo["off_prev"].between(max(a, 1), b)
        out[sel] = [cal[int(i)] for i in mo.loc[sel, "prevT_i"]]
    return out


def equity_settlement_lag(T: pd.Timestamp) -> int:
    """US equity settlement lag k in force for trades around month end T."""
    if T < SETTLEMENT_T2:
        return 3
    if T < SETTLEMENT_T1:
        return 2
    return 1


def hold_to_decision(hold: pd.DataFrame, exec: str = "next_close") -> pd.DataFrame:
    """Convert desired holdings per return day into engine decision weights.

    With ``next_close`` the engine holds over return day t the weight decided at the close of t-2;
    with ``next_open`` the one decided at t-1. Any data-dependent signal written into ``hold`` for
    return day t must therefore use information up to the close of t-2 (t-1 for next_open).
    """
    lag = {"next_close": 2, "next_open": 1}[exec]
    return hold.shift(-lag).fillna(0.0)
