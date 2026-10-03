"""Forward test 2's S2 (equities + trend) and S1 (broad CME trend), native in backtrader (src/forward2.py).

The signals are computed inside the strategy from the 30 CME futures feeds (F_*: fully collateralised total-return
index levels, priced like ETFs with no multiplier) and two pieces of side data loaded in __init__ and read only up
to the current bar's date: the daily T-bill accrual (rf_daily) and the published NYSE session calendar, whose last
session of each month is a decision date (base.planned_month_ends; equal to forward2._month_ends).

At each month-end close T, from bars up to T only (excess return = index return - T-bill):

* eligible: >= MIN_HISTORY excess returns, a positive EWMA volatility (centre of mass EWMA_COM, >= 60 returns,
  pandas' adjust=True weights) and a traded session (volume > 0) among the last ACTIVE_WINDOW;
* S1: mean sign of the trailing 21/63/252-day compounded excess returns x RAW_SCALE / vol / N eligible, scaled to
  TARGET_VOL ex ante with the trailing COV_WINDOW-day covariance (pairwise, >= 60 common returns), gross <= 3;
* S2: 0.5 x F_ES at TARGET_VOL / vol (when eligible) + 0.5 x S1, rescaled the same way.

Between decisions the book is traded back to the targets every session (the engine rebalances daily).

Fills (``fill``):

* ``next_close`` (default, native): a ``bt.Order.Close`` submitted in next() of bar d fills at the close of d+1. It
  is sized at the close of d, so each weight drifts for one session before the fill; the engine assumes the exact
  target at that close.
* ``coc`` (diagnostic): the target in force at d-1 is ordered at bar d with cheat-on-close and fills at the close
  of d at exact weights. This is the engine's book, which checks the signals apart from the fill timing.

    --strategy gqh_equity_trend [--params book=S2|S1,fill=next_close|coc,decisions=<csv>]

``decisions`` writes the month-end target weights to a CSV when the run stops. Run from the first print (the
harness default, 2010-06-07): a later --start would shorten the history the signals see, so the strategy raises.

Checked against forward2 (data through 2026-10-02, project costs): the 196 month-end decisions equal
s1_decisions / s2_decisions to 4e-15; with ``coc`` the excess returns match forward2.returns to 0.0003 bp a day
(correlation 1.0). With ``next_close`` S2 correlates at 0.9996 (0.8 bp a day): about 0.5 bp is the one-session
sizing drift, the rest is the day the trade cost lands (backtrader books it on the fill bar d+1, the engine on
d+2, the first day held).
"""
from __future__ import annotations

import math
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from src import engine as E
from src.forward2 import (ACTIVE_WINDOW, COV_WINDOW, EWMA_COM, GROSS_CAP, LOOKBACKS, MIN_HISTORY, RAW_SCALE,
                          S1_TICKERS, TARGET_VOL)
from validation.starter_kit.base import WeightStrategy, planned_month_ends

SYMBOLS = list(S1_TICKERS)
NEEDS_MARGIN = True     # gross up to 3x: cash goes negative
IS_START = "2011-09-02"  # first session held after the first decision (MIN_HISTORY returns from 2010-06-08)
VOL_MIN_OBS = 60         # forward2: ewm(min_periods=60) and cov(min_periods=60)
BOOKS = ("S2", "S1")
FILLS = ("next_close", "coc")


def EXEC(params: dict) -> str:
    """The harness turns on cheat-on-close for the diagnostic fill."""
    return "coc" if params.get("fill") == "coc" else "next_close"


class EquityTrend(WeightStrategy):
    params = dict(book="S2", fill="next_close", decisions="")

    def __init__(self):
        super().__init__()
        if self.p.book not in BOOKS or self.p.fill not in FILLS:
            raise ValueError(f"book must be one of {BOOKS} and fill one of {FILLS}")
        missing = sorted(set(SYMBOLS) - {d._name for d in self.datas})
        if missing:
            raise ValueError(f"feeds missing: {missing}")
        self.feeds = [self.getdatabyname(t) for t in SYMBOLS]
        self.i_es = SYMBOLS.index("F_ES")
        rf = E.load_rf("FWD")
        self.rf_day = rf.index.values.astype("datetime64[D]")
        self.rf = rf.to_numpy(float)
        self.month_ends = planned_month_ends()
        self.first_print = E.load_ohlc(SYMBOLS, "FWD")["close"].apply(pd.Series.first_valid_index).min().date()
        n = len(SYMBOLS)
        self.decay = 1.0 - 1.0 / (1.0 + EWMA_COM)
        self.ewm_num, self.ewm_den = np.zeros(n), np.zeros(n)    # sum of decayed ex^2 and of decayed weights
        self.n_ret = np.zeros(n, dtype=int)                       # excess returns seen (MIN_HISTORY, ewm minp)
        self.ex_hist = []                                         # one row of excess returns per session
        self.traded = deque(maxlen=ACTIVE_WINDOW)
        self.target = None
        self.decisions = OrderedDict()

    def tbill(self, day) -> float:
        """rf_daily at the last date up to ``day``."""
        i = int(np.searchsorted(self.rf_day, np.datetime64(day, "D"), side="right")) - 1
        return float(self.rf[i]) if i >= 0 else 0.0

    def observe(self, day) -> None:
        """Excess returns, EWMA and activity from this bar; a return needs real prints at this bar and the last."""
        rf = self.tbill(day)
        ex = np.full(len(self.feeds), np.nan)
        for i, d in enumerate(self.feeds):
            if len(d) > 1 and self.has_print(d) and self.has_print(d, -1):
                ex[i] = d.close[0] / d.close[-1] - 1.0 - rf
        seen = np.isfinite(ex)
        self.ewm_num = self.ewm_num * self.decay + np.where(seen, ex * ex, 0.0)   # missing returns still decay
        self.ewm_den = self.ewm_den * self.decay + seen
        self.n_ret += seen
        self.ex_hist.append(ex)
        self.traded.append(np.array([d.volume[0] > 0 for d in self.feeds]))

    def vol(self) -> np.ndarray:
        """Annualised EWMA volatility of excess returns (NaN before VOL_MIN_OBS returns)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.sqrt(self.ewm_num / self.ewm_den * 252)
        return np.where(self.n_ret >= VOL_MIN_OBS, v, np.nan)

    def scale_to_target(self, w: np.ndarray, hist: np.ndarray) -> np.ndarray:
        """TARGET_VOL ex ante with the trailing covariance of the instruments held, then gross <= GROSS_CAP."""
        cols = np.flatnonzero(w != 0)
        if not cols.size:
            return w * 0.0
        cov = pd.DataFrame(hist[-COV_WINDOW:, cols]).cov(min_periods=VOL_MIN_OBS).fillna(0.0).to_numpy()
        v = w[cols]
        var = float(v @ cov @ v) * 252
        if not np.isfinite(var) or var <= 0:
            return w * 0.0
        w = w * (TARGET_VOL / math.sqrt(var))
        g = np.abs(w).sum()
        return w * (GROSS_CAP / g) if g > GROSS_CAP else w

    def decide(self) -> np.ndarray:
        """Target weights at this month-end close."""
        hist = np.array(self.ex_hist[-max(COV_WINDOW, *LOOKBACKS):])
        vol = self.vol()
        with np.errstate(invalid="ignore"):
            ok = (self.n_ret >= MIN_HISTORY) & np.isfinite(vol) & (vol > 0) & np.any(self.traded, axis=0)
        s1 = np.zeros(len(SYMBOLS))
        if ok.any():
            past = np.nan_to_num(hist[:, ok], nan=0.0)
            sig = sum(np.sign(np.prod(1 + past[-k:], axis=0) - 1) for k in LOOKBACKS) / len(LOOKBACKS)
            s1[ok] = sig * (RAW_SCALE / vol[ok]) / ok.sum()
            s1 = self.scale_to_target(s1, hist)
        if self.p.book == "S1":
            return s1
        es = np.zeros(len(SYMBOLS))
        if ok[self.i_es]:                          # an untradeable ES leaves only the trend half
            es[self.i_es] = TARGET_VOL / vol[self.i_es]
        return self.scale_to_target(0.5 * es + 0.5 * s1, hist)

    def next(self):
        day = self.feeds[0].datetime.date(0)
        if len(self) == 1 and day > self.first_print:
            raise ValueError(f"feeds start {day}, after the first print {self.first_print}: the signals need the "
                             "whole history (keep the harness's default --start)")
        self.observe(day)
        if self.p.fill == "coc" and self.target is not None:      # the target in force at d-1, filled at close d
            self.rebalance(self.target, "coc", SYMBOLS)
        if day in self.month_ends:
            w = self.decide()
            self.decisions[day] = w
            self.target = dict(zip(SYMBOLS, w))
        if self.p.fill == "next_close" and self.target is not None:
            self.rebalance(self.target, "next_close", SYMBOLS)

    def stop(self):
        if self.p.decisions and self.decisions:
            out = pd.DataFrame(list(self.decisions.values()), index=pd.to_datetime(list(self.decisions)),
                               columns=SYMBOLS)
            Path(self.p.decisions).parent.mkdir(parents=True, exist_ok=True)
            out.rename_axis("date").to_csv(self.p.decisions)


STRATEGY_CLASS = EquityTrend
