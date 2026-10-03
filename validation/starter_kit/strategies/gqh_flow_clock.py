"""F1, the submitted "flow clock" strategy, native in backtrader (src/forward.py: A + C + TSMOM, src/ensemble.py).

The three streams and the ensemble are computed inside the strategy from the 15 ETF feeds and three pieces of side
data loaded in __init__ and read only up to the current bar's date: the daily T-bill accrual (rf_daily), the
Treasury auction announcements (ifc_treasury.load_nominal_auctions; an auction counts from its announcement date)
and the published NYSE schedule (realised sessions plus the unscheduled closures, calendar_utils.planned_calendar),
on which every T-j is counted, as in the engine.

* A (ifc_rebalance): at the close of T-6, the drift of a 60/40 SPY/IEF shadow portfolio since the previous month
  end over the root-mean-square of all earlier months' drifts (>= 12 months), clipped to +/-2 = q; hold
  -q 0.05 / vol SPY and +q 0.05 / vol IEF over return days [T-4, T]; next_close.
* C (ifc_treasury): at the close of T-4, the month's coupon issuance (offering x duration of the nominal Note/Bond
  auctions announced by then) over the median of the previous 12 months, clipped to [0.5, 2] = q; hold
  min(q 0.10 / vol IEF, 2) over [T-2, T]; next_close.
* TSMOM (pct, measure "none"): at each month-end close, sign of the 252-day return over the T-bill x 0.40 / vol /
  N eligible, scaled to 10 % ex ante with the 252-day covariance, gross <= 3; next_open, rebalanced every open.

Each stream's paper return is engine.simulate's (its own trades at its own costs, short borrow, cash at the
T-bill). They feed ensemble.combine's rules: minimum-variance Ledoit-Wolf weights lam (expanding, re-estimated
every 21 sessions, >= 252 live days, applied two sessions later), the 8 % volatility target k from the EWMA
(span 63) of the lam-weighted excess return (also two sessions late), and the 4x gross cap. The book holds
sum_s lam_s,t k_t held_s,t of each ETF over return day t.

Fills (``fill``):

* ``native`` (default): per ETF at bar d, one ``bt.Order.Close`` (fills at the close of d+1) for A, C and the
  re-scaling of the TSMOM book, and one market order (fills at the open of d+1) for TSMOM's own trades, both sized
  at the close of d. The ensemble applies scale_t to the whole of return day t (close t-1 to close t), including
  TSMOM's overnight leg, so the TSMOM re-scale goes with the close orders; ``tsmom_scale=open`` moves it into the
  open order instead (one TSMOM order a day).
* ``coc`` (diagnostic): the close part for return day d+1 is ordered at bar d with cheat-on-close and fills at the
  close of d at exact weights (the engine's book); the TSMOM open orders are unchanged (``coc=False``).

How ``native`` differs from the engine (execution only; the signals are the engine's):

* Sizing at the close of d: the A/C weights drift for one session before their close fill, TSMOM's overnight.
* k_{d+2} for the A/C close order is capped with TSMOM's gross over d+2, decided at d+1; the gross known at d is
  used. TSMOM's gross changes only from the session after a month end, when A and C hold nothing.
* Unscheduled closures are not known at d: a close order meant for the next planned session fills at the next real
  close (around 2012-10-29/30, A and C hold one session longer than the engine, which re-plans at that close).
* Both of the first two points let the realised gross exceed the 4x cap slightly: on 36 sessions to 2026-10-02,
  14 of them above 4.02x, at most 4.086x (2021-02-26).

    --strategy gqh_flow_clock [--params fill=native|coc,tsmom_scale=close|open,dump=<csv>]

``dump`` writes the daily internals (stream holdings, paper returns, lam, k, the paper ensemble return, fills) to a
CSV. The feeds must start at the first cached session (the harness default) for the engine's warm-ups; the
strategy raises otherwise.

Verified on 2026-10-03 (validation/starter_kit/check_flow_clock.py): the stream holdings, paper returns, lam, k and
the paper ensemble return equal the engine's to 1e-16 (forward.returns("F1", "FWD") exactly). At project costs the
native book's daily excess returns correlate with the official F1 at 0.9962 in-sample and 0.9939 out-of-sample
(2.0 / 2.3 bp mean absolute difference). Almost all of that difference comes from when costs are booked:
backtrader charges a close fill's commission on the fill session, the engine on the next return day. With those
costs moved one session later the figures are 0.9997 / 0.9998 (0.29 bp).
"""
from __future__ import annotations

import math
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src import calendar_utils as CU
from src import config as C
from src import engine as E
from src import ensemble as EN
from src import forward as F
from src.strategies import ifc_treasury
from validation.starter_kit.base import WeightStrategy

SPEC = F._stream_specs("F1")                # label -> (module, params, exec, cost map), as forward.stream_frames
STREAMS = list(SPEC)                        # A, C, TSMOM
PA, PC, PT = (SPEC[s][1] for s in STREAMS)
SYMBOLS = list(SPEC["TSMOM"][0].TICKERS)    # pct.TICKERS: includes A's SPY/IEF and C's IEF
NEEDS_MARGIN = True                         # up to 4x gross
IS_START = "2006-05-08"                     # first live day of the ensemble
FILLS = ("native", "coc")
TSMOM_SCALES = ("close", "open")
RECOMPUTE = 21                              # ensemble._weights: minvar_lw re-estimated every 21 sessions
LAG = 2                                     # lam and k use stream returns up to t-2
EWM_ALPHA = 2.0 / (63 + 1)                  # ensemble.combine: ewm(span=63, min_periods=63), adjust=True
FFILL_LIMIT = 5                             # engine.simulate bridges isolated missing prints only


def EXEC(params: dict) -> str:
    """The harness turns on cheat-on-close for the diagnostic fill."""
    return "coc" if params.get("fill") == "coc" else "mixed"


def _check_spec() -> None:
    """The rules implemented below are those of F1's specification; fail loudly if it changes."""
    execs = {s: SPEC[s][2] for s in STREAMS}
    if STREAMS != ["A", "C", "TSMOM"] or execs != {"A": "next_close", "C": "next_close", "TSMOM": "next_open"}:
        raise ValueError(f"unexpected F1 streams {execs}")
    if any(SPEC[s][3] is not None for s in STREAMS) or F.ENSEMBLE != {"scheme": "minvar_lw", "vol_target": 0.08,
                                                                       "brake": False}:
        raise ValueError("unexpected F1 costs or ensemble")
    if PA["mode"] != "monthly" or PA["extra_delay"] or PC["extra_delay"] or not PC["issuance_sizing"]:
        raise ValueError("unexpected A/C parameters")
    if PC["calendar"] != "ex_ante" or PT["measure"] != "none" or PT["extra_delay"] or "tickers" in PT:
        raise ValueError("unexpected C/TSMOM parameters")


def _roll_vol(x: np.ndarray, window: int) -> float:
    """Annualised rolling std of daily returns at the last point (pandas' rolling, as the engine computes it)."""
    r = pd.Series(x).pct_change(fill_method=None)
    return float(r.rolling(window, min_periods=window).std().iloc[-1] * math.sqrt(C.TRADING_DAYS))


def _month_key(ts) -> int:
    return ts.year * 12 + ts.month


class FlowClock(WeightStrategy):
    params = dict(fill="native", tsmom_scale="close", dump="")

    def __init__(self):
        super().__init__()
        _check_spec()
        if self.p.fill not in FILLS or self.p.tsmom_scale not in TSMOM_SCALES:
            raise ValueError(f"fill must be one of {FILLS} and tsmom_scale one of {TSMOM_SCALES}")
        missing = sorted(set(SYMBOLS) - {d._name for d in self.datas})
        if missing:
            raise ValueError(f"feeds missing: {missing}")
        self.feeds = [self.getdatabyname(t) for t in SYMBOLS]
        self.jA = [SYMBOLS.index(PA["equity"]), SYMBOLS.index(PA["bond"])]
        self.jC = SYMBOLS.index(PC["ticker"])
        n = len(SYMBOLS)

        # published calendar: realised sessions plus unscheduled closures (expected sessions when decisions are made)
        real = E.trading_calendar("FWD")
        planned = CU.planned_calendar(real)
        if not planned.equals(ifc_treasury.expected_calendar(real, PC["calendar"])):
            raise ValueError("A's and C's calendars differ")
        self.real = real
        self.real_pos = {ts.date(): i for i, ts in enumerate(real)}
        self.ppos = planned.get_indexer(real)
        nr, npl = len(real), len(planned)
        mo = CU.month_offsets(planned)
        self.month_end = (mo["off_own"] == 0).to_numpy()            # TSMOM decides at each complete month's T
        if not set(planned[self.month_end]) <= set(real):
            raise ValueError("a month end falls on an unscheduled closure")

        # side data, read at index <= the current bar
        rf = E.load_rf("FWD").reindex(real).ffill().fillna(0.0).to_numpy()
        self.rf = rf
        self.cum_lrf = np.concatenate([[0.0], np.cumsum(np.log1p(rf))])     # pct.signal_panel
        self.cb = np.array([C.cost_bps(t) for t in SYMBOLS]) / 1e4        # F1 streams: engine default costs
        self.borrow = np.array([0.0 if C.is_future(t) else C.SHORT_BORROW_BPS_PER_YEAR
                                for t in SYMBOLS]) / 1e4 / C.TRADING_DAYS
        self.cb_overlay = np.array([EN.STREAM_COST_BPS[s] for s in STREAMS]) / 1e4

        # A: one row per month with a T (ifc_rebalance.month_table): observation, previous T, hold window
        first = pd.Series(np.arange(npl), index=planned).groupby(mo["month"].values).min()
        obs, (a, b) = int(PA["obs"]), (int(PA["window"][0]), int(PA["window"][1]))
        self.a_months = []
        for month, ti in mo.groupby("month")["T_i"].first().items():
            if pd.isna(ti):
                continue
            ti, prev = int(ti), int(first[month]) - 1
            o, ia, ib = ti + obs, ti + a, ti + b
            if prev < 0 or o <= prev or ib >= npl or ia - 1 < 0:
                continue
            self.a_months.append((o, prev, ia, ib))
        self.a_next, self.a_d2, self.a_n = 0, 0.0, 0

        # C: decision session (last real session <= T+a-2) -> hold window on the planned calendar
        ac, bc = PC["window"]
        self.c_by_day = {}
        for ti in mo["T_i"].dropna().astype(int).unique():
            if ti + ac - 2 < 0:
                continue
            d = int(real.searchsorted(planned[ti + ac - 2], side="right")) - 1
            self.c_by_day.setdefault(d, []).append((ti + ac, min(ti + bc, npl - 1), _month_key(planned[ti])))
        # C issuance: every month's cut-off on the 2004-extended calendar (ifc_treasury.monthly_issuance)
        ext = ifc_treasury.extended_calendar("FWD")
        calx = ifc_treasury.expected_calendar(ext, PC["calendar"])
        mox = CU.month_offsets(calx)
        self.iss_months, self.iss_cut = [], []
        for ti in mox["T_i"].dropna().astype(int).unique():
            if ti + ac - 2 < 0:
                continue
            self.iss_months.append(_month_key(calx[ti]))
            self.iss_cut.append(ext[ext.searchsorted(calx[ti + ac - 2], side="right") - 1])
        self.iss_row = {m: k for k, m in enumerate(self.iss_months)}
        self.iss_S = {}
        auc = ifc_treasury.load_nominal_auctions("FWD")
        self.auc_month = (auc["issue_date"].dt.year * 12 + auc["issue_date"].dt.month).to_numpy()
        self.auc_ann = auc["announcemt_date"].to_numpy()
        self.auc_dv01 = auc["dv01"].to_numpy()

        # price history (session index from the calendar start) and planned SPY/IEF closes for A
        self.close = np.full((nr, n), np.nan)       # real prints only (pct uses unfilled closes)
        self.cff = np.full((nr, n), np.nan)         # engine.simulate: ffill(limit=5)
        self.off = np.full((nr, n), np.nan)
        self.last_c, self.last_o = np.full(n, np.nan), np.full(n, np.nan)
        self.gap = np.zeros(n, dtype=int)
        self.nvalid = np.zeros(n, dtype=int)
        self.pclose = np.full((npl, 2), np.nan)
        self.last_p = -1

        # streams: planned holdings (A, C), TSMOM decisions, paper holdings over each return day
        self.holdA = np.zeros((npl + 3, 2))
        self.holdC = np.zeros(npl + 3)
        self.wT = np.zeros((nr + 1, n))
        self.heldA = np.zeros((nr + 1, 2))
        self.heldC = np.zeros(nr + 1)
        self.ordA = np.zeros((nr, 2))               # the holdings each bar's close orders target
        self.ordC = np.zeros(nr)

        # ensemble
        self.ex = np.zeros((nr, 3))
        self.gr = np.zeros((nr + 1, 3))
        self.x = np.full((nr, 3), np.nan)
        self.live = np.zeros(3, dtype=bool)
        self.nlive = np.zeros(3, dtype=int)
        self.w_last = None
        self.lam = np.zeros((nr + LAG + 1, 3))
        self.sig = np.full(nr + LAG + 1, np.nan)
        self.ewm_avg, self.ewm_wt, self.ewm_n = math.nan, 1.0, 0
        self.k = np.zeros(nr + 2)                   # k_t, set at bar t-1 (all its inputs known then)
        self.kproxy = np.full(nr + 3, np.nan)       # k_t as sized by the close orders of bar t-2
        self.started = False
        self.prev_scale, self.prev_g = np.zeros(3), np.zeros(3)
        self.paper = np.full((nr, 3), np.nan)       # paper ensemble: net, excess, overlay turnover

        # book: shares per part (A/C close part, TSMOM part) once the submitted orders have filled
        self.shAC, self.shT = np.zeros(n), np.zeros(n)
        self.desync = 0
        self.seen = []
        self.fills = np.zeros((nr, 4))              # per fill session: commission and notional, close / other fills
        self.nav = np.full(nr, np.nan)

    def start(self):
        self.broker.set_coc(self.p.fill == "coc")

    def notify_order(self, order):
        super().notify_order(order)
        if order.status == order.Completed:          # booked in the NAV of the session being processed
            i = self.real_pos[self.feeds[0].datetime.date(0)]
            c = 0 if order.exectype == bt.Order.Close else 2
            self.fills[i, c] += order.executed.comm
            self.fills[i, c + 1] += abs(order.executed.size) * order.executed.price

    # ------------------------------------------------------------------ data

    def observe(self, i: int, p: int) -> None:
        for j, d in enumerate(self.feeds):
            if d.valid[0]:
                self.close[i, j] = d.close[0]
                self.last_c[j], self.last_o[j], self.gap[j] = d.close[0], d.open[0], 0
                self.nvalid[j] += 1
            else:
                self.gap[j] += 1
            if self.gap[j] <= FFILL_LIMIT:
                self.cff[i, j], self.off[i, j] = self.last_c[j], self.last_o[j]
        for q in range(max(self.last_p + 1, 1), p):         # closures since the last bar keep the last close
            self.pclose[q] = self.pclose[q - 1]
        self.pclose[p] = self.cff[i, self.jA]
        self.last_p = p

    # ------------------------------------------------------------------ paper returns (engine.simulate)

    def _next_close_ex(self, i: int, cols: list[int], held: np.ndarray, prev: np.ndarray) -> float:
        rf = self.rf[i]
        r = np.nan_to_num(self.cff[i, cols] / self.cff[i - 1, cols] - 1.0) if i >= 1 else np.zeros(len(cols))
        rp = np.nan_to_num(self.cff[i - 1, cols] / self.cff[i - 2, cols] - 1.0) if i >= 2 else np.zeros(len(cols))
        den = 1 + float((prev * rp).sum())
        drifted = prev * (1 + rp) / den if den != 0 else np.zeros(len(cols))
        trade = np.abs(held - drifted)
        gross = float((held * r).sum()) + (1.0 - held.sum()) * rf
        costs = float((trade * self.cb[cols]).sum()) + float((np.abs(np.minimum(held, 0)) * self.borrow[cols]).sum())
        return (gross - costs) - rf

    def _next_open_ex(self, i: int) -> float:
        n = len(SYMBOLS)
        rf = self.rf[i]
        wn = self.wT[i - 1] if i >= 1 else np.zeros(n)
        wo = self.wT[i - 2] if i >= 2 else np.zeros(n)
        if i >= 1:
            ro = np.nan_to_num(self.off[i] / self.cff[i - 1] - 1.0)
            rc = np.nan_to_num(self.cff[i] / self.off[i] - 1.0)
        else:
            ro = rc = np.zeros(n)
        on = float((wo * ro).sum())
        gross_risky = on + (1 + on) * float((wn * rc).sum())
        drifted = wo * (1 + ro) / (1 + on) if 1 + on != 0 else np.zeros(n)
        trade = np.abs(wn - drifted)
        gross = gross_risky + (1.0 - wn.sum()) * rf
        costs = float((trade * self.cb).sum()) + float((np.abs(np.minimum(wn, 0)) * self.borrow).sum())
        return (gross - costs) - rf

    def stream_returns(self, i: int) -> None:
        self.ex[i, 0] = self._next_close_ex(i, self.jA, self.heldA[i], self.heldA[i - 1] if i else np.zeros(2))
        self.ex[i, 1] = self._next_close_ex(i, [self.jC], self.heldC[i:i + 1],
                                            self.heldC[i - 1:i] if i else np.zeros(1))
        self.ex[i, 2] = self._next_open_ex(i)
        self.gr[i] = self.gross_over(self.heldA[i], self.heldC[i], self.wT[i - 1] if i else 0.0)

    @staticmethod
    def gross_over(held_a, held_c, w_t) -> np.ndarray:
        """Gross exposure of each stream at its own scale (engine: held.abs().sum(axis=1))."""
        return np.array([np.abs(held_a).sum(), abs(float(held_c)), np.abs(w_t).sum()])

    # ------------------------------------------------------------------ ensemble (ensemble.combine)

    def update_ensemble(self, i: int) -> None:
        ex = self.ex[i]
        self.live |= self.gr[i] > 0
        self.nlive += self.live
        self.x[i] = np.where(self.live, ex, np.nan)
        if i % RECOMPUTE == 0:
            ok = np.flatnonzero(self.nlive >= EN.MIN_HISTORY)
            if ok.size:
                hist = self.x[:i + 1][:, ok]
                hist = hist[~np.isnan(hist).any(axis=1)]
                if len(hist) >= EN.MIN_HISTORY:
                    cov = LedoitWolf().fit(hist).covariance_
                    raw = np.clip(np.linalg.solve(cov, np.ones(ok.size)), 0, None)
                    if raw.sum() > 0:
                        w = np.zeros(3)
                        w[ok] = raw / raw.sum() / np.sqrt(np.diag(cov)).mean()
                        self.w_last = w
        if self.w_last is not None:
            self.lam[i + LAG] = self.w_last
        c2 = float((self.lam[i] * ex).sum()) ** 2
        self.ewm_n += 1
        if self.ewm_n == 1:                                  # pandas' ewm(adjust=True) recursion
            self.ewm_avg = c2
        else:
            self.ewm_wt *= 1.0 - EWM_ALPHA
            if self.ewm_avg != c2:
                self.ewm_avg = (self.ewm_wt * self.ewm_avg + c2) / (self.ewm_wt + 1.0)
            self.ewm_wt += 1.0
        if self.ewm_n >= 63:
            self.sig[i + LAG] = math.sqrt(self.ewm_avg * C.TRADING_DAYS)
        if not self.started and np.abs(self.lam[i]).sum() > 0 and not math.isnan(self.sig[i]):
            self.started = True
        if self.started:                                     # combine's loop: the official F1 return
            scale = self.k[i] * 1.0 * self.lam[i]
            trade = np.abs(scale - self.prev_scale) * np.minimum(self.gr[i], self.prev_g)
            r = scale @ ex - trade @ self.cb_overlay
            self.paper[i] = (r + self.rf[i], r, trade.sum())
            self.prev_scale, self.prev_g = scale, self.gr[i]

    def k_at(self, t: int, gr_t: np.ndarray, gr_prev: np.ndarray) -> float:
        """k_t: vol target on sig_t, then the gross cap on lam_t x max(gross over t, gross over t-1)."""
        sig = self.sig[t]
        if math.isnan(sig):
            return 0.0
        k = min(F.ENSEMBLE["vol_target"] / sig, EN.LEV_CAP) if sig > 0 else EN.LEV_CAP
        lev = k * float((self.lam[t] * np.maximum(gr_t, gr_prev)).sum())
        return k * (min(EN.LEV_CAP / lev, 1.0) if lev > 0 else 1.0)

    # ------------------------------------------------------------------ decisions

    def decide_a(self, o: int, prev: int, ia: int, ib: int) -> None:
        ce, cb = self.pclose[:, 0], self.pclose[:, 1]
        ge, gb = ce[o] / ce[prev], cb[o] / cb[prev]
        w = PA["w_equity"]
        D = w * ge / (w * ge + (1 - w) * gb) - w
        if not np.isfinite(D):
            return                                           # a month without a drift is not in the RMS history
        n_hist = self.a_n
        s = math.sqrt(self.a_d2 / n_hist) if n_hist else math.nan
        vol_eq = _roll_vol(ce[:o + 1], int(PA["vol_window"]))
        vol_bd = _roll_vol(cb[:o + 1], int(PA["vol_window"]))
        if n_hist >= int(PA["min_hist"]) and np.isfinite(vol_eq) and np.isfinite(vol_bd):
            q = min(max(D / s, -PA["clip"]), PA["clip"])
            self.holdA[ia:ib + 1, 0] += -q * PA["target_vol"] / vol_eq
            self.holdA[ia:ib + 1, 1] += q * PA["target_vol"] / vol_bd
        self.a_d2 += D ** 2
        self.a_n += 1

    def issuance(self, row: int, day) -> float:
        """S_m: offering x duration of the month's issues announced by its cut-off (already passed)."""
        if row not in self.iss_S:
            cut = self.iss_cut[row]
            assert cut.date() <= day, "issuance read before its cut-off"
            sel = (self.auc_month == self.iss_months[row]) & (self.auc_ann <= np.datetime64(cut))
            self.iss_S[row] = float(self.auc_dv01[sel].sum())
        return self.iss_S[row]

    def decide_c(self, i: int, lo: int, hi: int, month: int, day) -> None:
        row = self.iss_row.get(month)
        k = int(PC["median_months"])
        if row is None or row < k:
            return
        med = float(np.median([self.issuance(r, day) for r in range(row - k, row)]))
        with np.errstate(divide="ignore", invalid="ignore"):
            q = float(np.clip(np.float64(self.issuance(row, day)) / med, *PC["q_clip"]))
        v = _roll_vol(self.close[:i + 1, self.jC], int(PC["vol_window"]))
        if not np.isfinite(q) or not np.isfinite(v) or v <= 0:
            return
        self.holdC[lo:hi + 1] = min(q * PC["scale"] / v, PC["cap"])

    def decide_tsmom(self, i: int) -> np.ndarray | None:
        """pct.signal_panel / decision_weights with measure "none"; None when no ETF is eligible (no decision row)."""
        lb, vw, cw = int(PT["trend_lb"]), int(PT["vol_window"]), int(PT["cov_window"])
        need = max(int(PT["W"]) * int(PT["legs"]), lb, vw, cw, int(PT["id_window"]))
        cols, s_, vol_ = [], [], []
        for j in range(len(SYMBOLS)):
            if self.nvalid[j] < PT["min_history"] or np.isnan(self.close[i, j]):
                continue
            if i - need < 0 or np.isnan(self.close[i - need:i + 1, j]).any():
                continue
            lp = np.log(self.close[[i, i - lb], j])
            tr = math.exp(lp[0] - lp[1]) - 1.0
            rf_win = math.exp(self.cum_lrf[i + 1] - self.cum_lrf[i - lb + 1]) - 1.0
            c = self.close[i - vw:i + 1, j]
            cols.append(j)
            s_.append(float(np.sign(tr - rf_win)))
            vol_.append(float(np.std(c[1:] / c[:-1] - 1.0, ddof=1) * math.sqrt(C.TRADING_DAYS)))
        if not cols:
            return None
        names = [SYMBOLS[j] for j in cols]
        raw = pd.Series(s_, index=names) * 1.0 * (PT["raw_scale"] / pd.Series(vol_, index=names)) / len(cols)
        c = self.close[i - cw:i + 1][:, cols]
        sigma = pd.DataFrame(c[1:] / c[:-1] - 1.0, columns=names).cov(min_periods=20).to_numpy() * C.TRADING_DAYS
        wv = raw.to_numpy()
        var = float(wv @ np.nan_to_num(sigma) @ wv)
        row = np.zeros(len(SYMBOLS))
        if var > 0:
            w = raw * (PT["target_vol"] / math.sqrt(var))
            gross = float(w.abs().sum())
            if gross > PT["gross_cap"]:
                w = w * PT["gross_cap"] / gross
            row[cols] = w.to_numpy()
        return row

    def decide(self, i: int, p: int, day) -> None:
        while self.a_next < len(self.a_months) and self.a_months[self.a_next][0] <= p:
            self.decide_a(*self.a_months[self.a_next])
            self.a_next += 1
        for lo, hi, month in self.c_by_day.get(i, ()):
            self.decide_c(i, lo, hi, month, day)
        row = self.decide_tsmom(i) if self.month_end[p] else None
        self.wT[i] = row if row is not None else (self.wT[i - 1] if i else 0.0)
        # the engine's holdings over the next return day (first planned day after this session) ...
        self.heldA[i + 1], self.heldC[i + 1] = self.holdA[p + 1], self.holdC[p + 1]
        # ... and the ones a close order placed now targets (the day after the next planned session)
        self.ordA[i], self.ordC[i] = self.holdA[p + 2], self.holdC[p + 2]
        gr_next = self.gross_over(self.heldA[i + 1], self.heldC[i + 1], self.wT[i])
        self.gr[i + 1] = gr_next
        self.k[i + 1] = self.k_at(i + 1, gr_next, self.gr[i])
        gr_after = self.gross_over(self.ordA[i], self.ordC[i], self.wT[i])   # TSMOM: gross known now
        self.kproxy[i + 2] = self.k_at(i + 2, gr_after, gr_next)

    # ------------------------------------------------------------------ orders

    def submit(self, delta: np.ndarray, exectype, **info) -> None:
        """One netted order per ETF (sells first) with the given execution type."""
        for j in sorted(np.flatnonzero(delta), key=lambda j: delta[j] > 0):
            size = float(delta[j])
            if size > 0:
                self.buy(data=self.feeds[j], size=size, exectype=exectype, **info)
            else:
                self.sell(data=self.feeds[j], size=-size, exectype=exectype, **info)

    def trade(self, i: int) -> None:
        held = np.array([self.getposition(d).size for d in self.feeds])
        if not np.array_equal(held, self.shAC + self.shT):      # a refused fill: re-base the TSMOM part
            self.desync += 1
            self.shT = held - self.shAC
        nav = self.broker.getvalue()
        px = np.array([d.close[0] for d in self.feeds])
        s1 = self.k[i + 1] * self.lam[i + 1]                    # scale over return day i+1 (exact)
        t_in = np.round(s1[2] * self.wT[i] * nav / px)           # TSMOM from the open of i+1
        if self.p.fill == "native":
            s2 = self.kproxy[i + 2] * self.lam[i + 2]            # scale over return day i+2, at its close fill
            w = np.zeros(len(SYMBOLS))
            w[self.jA] += s2[0] * self.ordA[i]
            w[self.jC] += s2[1] * self.ordC[i]
            ac = np.round(w * nav / px)
            t_ov = np.round(s2[2] * self.wT[i] * nav / px) if self.p.tsmom_scale == "close" else t_in
            self.submit(t_in - self.shT, bt.Order.Market)
            self.submit((ac - self.shAC) + (t_ov - t_in), bt.Order.Close)
            self.shAC, self.shT = ac, t_ov
        else:
            w = np.zeros(len(SYMBOLS))
            w[self.jA] += s1[0] * self.heldA[i + 1]
            w[self.jC] += s1[1] * self.heldC[i + 1]
            ac = np.round(w * nav / px)
            prev_t = self.wT[i - 1] if i else np.zeros(len(SYMBOLS))
            t_ov = np.round(s1[2] * prev_t * nav / px) if self.p.tsmom_scale == "close" else self.shT
            self.submit((ac - self.shAC) + (t_ov - self.shT), bt.Order.Market)        # cheat-on-close: close of i
            self.submit(t_in - t_ov, bt.Order.Market, coc=False)                     # open of i+1
            self.shAC, self.shT = ac, t_in

    # ------------------------------------------------------------------ loop

    def next(self):
        day = self.feeds[0].datetime.date(0)
        i = self.real_pos[day]
        if not self.seen and i != 0:
            raise ValueError(f"feeds start {day}, not at the first cached session {self.real[0].date()}: the "
                             "warm-ups need the whole history (keep the harness's default --start)")
        p = int(self.ppos[i])
        self.observe(i, p)
        self.stream_returns(i)
        self.update_ensemble(i)
        self.decide(i, p, day)
        self.nav[i] = self.broker.getvalue()
        self.trade(i)
        self.seen.append(i)

    def stop(self):
        if not self.p.dump or not self.seen:
            return
        ix = np.array(self.seen)
        cols = {"heldA_SPY": self.heldA[ix, 0], "heldA_IEF": self.heldA[ix, 1], "heldC_IEF": self.heldC[ix],
                "ordA_SPY": self.ordA[ix, 0], "ordA_IEF": self.ordA[ix, 1], "ordC_IEF": self.ordC[ix]}
        cols.update({f"wT_{t}": self.wT[ix, j] for j, t in enumerate(SYMBOLS)})
        for s, lab in enumerate(STREAMS):
            cols[f"ex_{lab}"], cols[f"gr_{lab}"], cols[f"lam_{lab}"] = self.ex[ix, s], self.gr[ix, s], self.lam[ix, s]
        cols.update({"sig": self.sig[ix], "k": self.k[ix], "kproxy": self.kproxy[ix], "paper_net": self.paper[ix, 0],
                     "paper_excess": self.paper[ix, 1], "overlay_turnover": self.paper[ix, 2], "nav": self.nav[ix],
                     "comm_close_fills": self.fills[ix, 0], "notional_close_fills": self.fills[ix, 1],
                     "comm_other_fills": self.fills[ix, 2], "notional_other_fills": self.fills[ix, 3],
                     "desync": self.desync})
        out = pd.DataFrame(cols, index=self.real[ix])
        Path(self.p.dump).parent.mkdir(parents=True, exist_ok=True)
        out.rename_axis("date").to_csv(self.p.dump)


STRATEGY_CLASS = FlowClock
