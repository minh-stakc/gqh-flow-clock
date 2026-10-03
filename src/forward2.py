"""Forward test 2 (FORWARD_TEST_2.md), frozen before any data after 2026-10-02. Do not edit after the tag.

Chosen from the literature review (strategies with the best live / post-publication evidence that daily liquid
data can reproduce), not from any backtest of ours:

S1  broad CME trend: 30 futures, signal = mean of the signs of the trailing 21/63/252-day excess returns
    (Hurst, Ooi & Pedersen 2013 style 1/3/12-month blend), sized 0.40/sigma_i/N with EWMA volatility
    (centre of mass 60 days, Moskowitz-Ooi-Pedersen 2012), book scaled to 10 % ex-ante volatility with the
    trailing 252-day covariance, gross <= 3, monthly decisions, next-close fills.
S2  equity + trend, equal risk: 0.5 x (ES at 10 % ex-ante vol) + 0.5 x S1, the sum rescaled to 10 % ex-ante
    volatility with the trailing 252-day covariance of the instruments, gross <= 3, monthly, next-close fills.
S3  Faber (2007) GTAA: 20 % each in SPY, EFA, IEF, VNQ, DBC when the month-end close is above the average of
    the last 10 month-end closes, otherwise T-bills; monthly, next-open fills, project ETF costs.
Benchmarks (yardsticks, not strategies): TSMOM_F (forward test 1's frozen code), LONG_ONLY_F (S1's construction
with every signal set to +1: does trend add anything over being long?), ES_10VOL (the S2 equity sleeve alone),
BH5 (constant 20 % weights in the S3 ETFs once their 10-month warm-up is over; the engine rebalances to target daily).

"Monthly" means monthly target decisions; the engine trades back to the targets every session (drift costs included).
Eligibility of a futures contract at a decision: >= MIN_HISTORY sessions of data, a positive EWMA volatility, and at
least one traded session (volume > 0) in the last ACTIVE_WINDOW sessions; a contract that stops trading (delisting,
vendor coverage loss) therefore leaves the book at the next month-end. ETFs without a close drop out the same way.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import calendar_utils as CU
from . import engine as E

STRATEGIES = ["S1", "S2", "S3"]
BENCHMARKS = ["TSMOM_F", "LONG_ONLY_F", "ES_10VOL", "BH5"]
PRIMARY = "S2"

S1_ROOTS = ["ES", "NQ", "RTY", "YM", "ZT", "ZF", "ZN", "ZB", "UB", "CL", "HO", "RB", "NG", "GC", "SI", "PL",
            "HG", "ZC", "ZS", "ZW", "ZL", "ZM", "LE", "HE", "6E", "6J", "6B", "6A", "6C", "6S"]
S1_TICKERS = [f"F_{r}" for r in S1_ROOTS]
_ORIGINAL_CHEAP = {"ES", "NQ", "YM", "ZT", "ZF", "ZN", "ZB", "CL", "GC", "SI", "HG", "6E", "6J", "6B"}
S1_COST_BPS = {f"F_{r}": (1.5 if (r in _ORIGINAL_CHEAP or r in {"UB", "6S"}) else 5.0) for r in S1_ROOTS}
S3_TICKERS = ["SPY", "EFA", "IEF", "VNQ", "DBC"]

LOOKBACKS = (21, 63, 252)
EWMA_COM = 60
COV_WINDOW = 252
MIN_HISTORY = 300
ACTIVE_WINDOW = 10      # sessions; longer than any past vendor gap (5 sessions, LE/HE Feb 2012)
RAW_SCALE = 0.40
TARGET_VOL = 0.10
GROSS_CAP = 3.0


def _month_ends(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    mo = CU.month_offsets(cal)
    return mo.index[mo["off_own"] == 0]


def _futures_panel(tickers: list[str], period: str):
    ohlc = E.load_ohlc(tickers, period)
    rf = E.load_rf(period).reindex(ohlc["close"].index).ffill().fillna(0.0)
    close = ohlc["close"]
    ret = close.pct_change(fill_method=None)
    ex = ret.sub(rf, axis=0)                      # futures excess return (index includes T-bill accrual)
    return ohlc, rf, ex


def _active(ohlc) -> pd.DataFrame:
    """True where the contract traded (volume > 0) in at least one of the last ACTIVE_WINDOW sessions; the data
    builder pads sessions without vendor bars with zero volume and zero excess return."""
    traded = (ohlc["volume"].fillna(0.0) > 0).astype(float)
    return traded.rolling(ACTIVE_WINDOW, min_periods=1).max().astype(bool)


def _scale_to_target(w: pd.Series, ex_hist: pd.DataFrame) -> pd.Series:
    cols = [c for c in w.index if w[c] != 0]
    if not cols:
        return w * 0.0
    cov = ex_hist[cols].tail(COV_WINDOW).cov(min_periods=60).fillna(0.0).to_numpy()
    v = w[cols].to_numpy()
    var = float(v @ cov @ v) * 252
    if not np.isfinite(var) or var <= 0:
        return w * 0.0
    w = w * (TARGET_VOL / math.sqrt(var))
    g = w.abs().sum()
    return w * (GROSS_CAP / g) if g > GROSS_CAP else w


def s1_decisions(period: str, long_only: bool = False) -> pd.DataFrame:
    """S1 decision weights (value at d uses data up to the close of d), on the NYSE calendar.
    long_only=True gives the LONG_ONLY_F benchmark (every signal +1, otherwise identical)."""
    ohlc, _, ex = _futures_panel(S1_TICKERS, period)
    active = _active(ohlc)
    vol = np.sqrt(ex.pow(2).ewm(com=EWMA_COM, min_periods=60).mean() * 252)
    counts = ex.notna().cumsum()
    w_dec = pd.DataFrame(np.nan, index=ex.index, columns=S1_TICKERS)
    for t in _month_ends(ex.index):
        hist = ex.loc[:t]
        elig = [c for c in S1_TICKERS if counts.loc[t, c] >= MIN_HISTORY and np.isfinite(vol.loc[t, c])
                and vol.loc[t, c] > 0 and active.loc[t, c]]
        w = pd.Series(0.0, index=S1_TICKERS)
        if elig:
            sig = pd.Series(0.0, index=elig)
            for k in LOOKBACKS:
                cum = (1 + hist[elig].tail(k).fillna(0.0)).prod() - 1
                sig += np.sign(cum)
            sig /= len(LOOKBACKS)
            if long_only:
                sig[:] = 1.0
            w[elig] = sig * (RAW_SCALE / vol.loc[t, elig]) / len(elig)
            w = _scale_to_target(w, hist)
        w_dec.loc[t] = w.values
    return w_dec.ffill().fillna(0.0)


def s2_decisions(period: str) -> pd.DataFrame:
    """S2 = 0.5 x ES sleeve (10 % ex-ante vol) + 0.5 x S1, rescaled to 10 % ex-ante vol, gross <= 3."""
    ohlc, _, ex = _futures_panel(S1_TICKERS, period)
    active = _active(ohlc)
    vol = np.sqrt(ex.pow(2).ewm(com=EWMA_COM, min_periods=60).mean() * 252)
    s1 = s1_decisions(period)
    counts = ex.notna().cumsum()
    w_dec = pd.DataFrame(np.nan, index=ex.index, columns=S1_TICKERS)
    for t in _month_ends(ex.index):
        es = pd.Series(0.0, index=S1_TICKERS)
        es_ok = (counts.loc[t, "F_ES"] >= MIN_HISTORY and np.isfinite(vol.loc[t, "F_ES"])
                 and vol.loc[t, "F_ES"] > 0 and active.loc[t, "F_ES"])
        if es_ok:                                  # an untradeable ES leaves only the trend half
            es["F_ES"] = TARGET_VOL / vol.loc[t, "F_ES"]
        combo = 0.5 * es + 0.5 * s1.loc[t]
        w_dec.loc[t] = _scale_to_target(combo, ex.loc[:t]).values
    return w_dec.ffill().fillna(0.0)


def es_sleeve_decisions(period: str) -> pd.DataFrame:
    """Benchmark ES_10VOL: the S2 equity sleeve alone (long ES at 10 % ex-ante vol, monthly)."""
    ohlc, _, ex = _futures_panel(["F_ES"], period)
    active = _active(ohlc)
    vol = np.sqrt(ex.pow(2).ewm(com=EWMA_COM, min_periods=60).mean() * 252)
    counts = ex.notna().cumsum()
    w_dec = pd.DataFrame(np.nan, index=ex.index, columns=["F_ES"])
    for t in _month_ends(ex.index):
        ok = (counts.loc[t, "F_ES"] >= MIN_HISTORY and np.isfinite(vol.loc[t, "F_ES"])
              and vol.loc[t, "F_ES"] > 0 and active.loc[t, "F_ES"])
        w_dec.loc[t, "F_ES"] = min(TARGET_VOL / vol.loc[t, "F_ES"], GROSS_CAP) if ok else 0.0
    return w_dec.ffill().fillna(0.0)


def s3_decisions(period: str, buy_and_hold: bool = False) -> pd.DataFrame:
    """Faber GTAA (or the BH5 benchmark): 20 % per ETF if month-end close > 10-month SMA of month-end closes."""
    ohlc = E.load_ohlc(S3_TICKERS, period)
    close = ohlc["close"]
    me = _month_ends(close.index)
    mclose = close.loc[me]
    sma = mclose.rolling(10, min_periods=10).mean()
    w_dec = pd.DataFrame(np.nan, index=close.index, columns=S3_TICKERS)
    for t in me:
        if buy_and_hold:
            row = (mclose.loc[t].notna() & sma.loc[t].notna()).astype(float) * 0.20
        else:
            row = ((mclose.loc[t] > sma.loc[t]) & sma.loc[t].notna()).astype(float) * 0.20
        w_dec.loc[t] = row.values
    return w_dec.ffill().fillna(0.0)


SPECS = {   # name -> (decision function, tickers, exec, cost map)
    "S1": (s1_decisions, S1_TICKERS, "next_close", S1_COST_BPS),
    "S2": (s2_decisions, S1_TICKERS, "next_close", S1_COST_BPS),
    "S3": (s3_decisions, S3_TICKERS, "next_open", None),
    "LONG_ONLY_F": (lambda p: s1_decisions(p, long_only=True), S1_TICKERS, "next_close", S1_COST_BPS),
    "ES_10VOL": (es_sleeve_decisions, ["F_ES"], "next_close", S1_COST_BPS),
    "BH5": (lambda p: s3_decisions(p, buy_and_hold=True), S3_TICKERS, "next_open", None),
}


def returns(name: str, period: str) -> pd.Series:
    if name == "TSMOM_F":                          # forward test 1's frozen benchmark, unchanged
        from . import forward

        return forward.returns("TSMOM_F", period)
    fn, tickers, ex_conv, cost = SPECS[name]
    w = fn(period)
    ohlc = E.load_ohlc(tickers, period)
    rf = E.load_rf(period)
    net, *_ = E.simulate(w, ohlc, rf, exec=ex_conv, cost_bps=cost)
    return net


def positions_for_next_session(name: str, period: str) -> pd.Series:
    """Weights held over the second session after the last data date (next_close) / the next session
    (next_open): in both cases the decision at the last available close."""
    fn, *_ = SPECS[name]
    w = fn(period)
    last = w.iloc[-1]
    return last[last.abs() > 1e-12]
