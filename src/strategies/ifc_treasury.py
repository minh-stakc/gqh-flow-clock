"""IFC sleeve C: Treasury month-end index extension, sized by issuance (HYPOTHESES.md section 1).

Rule as pre-registered
----------------------
``S_m`` = sum over Note and Bond auctions with issue date in month m and announcement date on or
before ``T-4`` of offering amount x approximate modified duration of the original term (2y 1.9,
3y 2.8, 5y 4.5, 7y 6.2, 10y 8.3, 20y 13.5, 30y 18.0). ``q = clip(S_m / median(S over the previous
12 months), 0.5, 2.0)``. Hold over return days [``T-2``, ``T``]: ``w_IEF = q * 0.10 / vol_IEF``,
capped at 2.0. Execution ``next_close``; vol = 63-day realised volatility observed at the
decision date. Kill condition: the IEF excess return over [T-2, T] is not positive on average.

Interpretation decisions (fixed before the first backtest of this module; no alternative was run)
-------------------------------------------------------------------------------------------------
1. Calendar: ``T-j`` / ``T+j`` are integer offsets on the NYSE (SPY) trading calendar from the
   month's last trading day (``calendar_utils.month_offsets``). The incomplete final month of a
   period (October 2024 in-sample) has no ``T`` and gets no window.
2. Decision date: for a hold window [``T+a``, ``T+b``] the decision date is ``T+a-2`` (section 0:
   "observed at the close of a-2"), i.e. ``T-4`` for the base window. Both the issuance cut-off
   ("announced on or before") and the volatility are observed at that date, and the weight is held
   constant over the whole window (the engine rebalances to it daily). This matches sleeves A and
   B. In the neighbourhood variants the cut-off moves with the window start:
   window2 = [T-1, T] (cut-off T-3), window4 = [T-3, T] (T-5), shift_early = [T-3, T-1] (T-5),
   shift_late = [T-1, T+1] (T-3). The S of month m always counts issues dated in month m.
3. ``median(S over the previous 12 months)`` = median of S_{m-1} .. S_{m-12}, each computed with
   ITS OWN month's cut-off (the same statistic as it was available at each earlier decision).
   The 2004 months needed for the 2005 medians lie before the engine's calendar (history starts
   2005-01-03); their T-j dates come from a 2004 NYSE business-day calendar built here (weekdays
   minus the ten 2004 NYSE closures, including the 2004-06-11 Reagan day of mourning).
4. Point-in-time auction data: an auction counts only if ``announcemt_date`` <= the cut-off date
   (announcements are made during the day, before the close), and auctions announced after the
   period's last visible date (2024-10-02 in-sample) are dropped before anything else. The cache is
   read with ``engine._read_parquet`` because it is not date-indexed; it holds no prices.
5. Security selection: ``security_type`` in {Note, Bond}. The cached Fiscal Data extract has no
   TIPS / FRN flag column (``data/download.py`` did not request ``inflation_index_security`` or
   ``floating_rate``), and TIPS and FRNs appear under Note/Bond. They are identified from the data:
   * FRN = Note/Bond row with no ``high_yield`` (FRNs are priced on a discount margin): exactly
     the 2-year rows from 2014 on, 12-13 a year, which is the FRN auction calendar.
   * TIPS = CUSIP whose largest gap (FRED nominal yield interpolated linearly in maturity on the
     2y/10y/30y points at the auction date, minus the auction high yield) exceeds 0.7 pp: a TIPS
     high yield is a real yield. The separation is clean in-sample: every nominal CUSIP has a max
     gap <= 0.36 pp, every TIPS CUSIP >= 1.07 pp, nothing lies between. TIPS status is a static
     attribute printed in the auction announcement, so using a CUSIP's later reopenings to recover
     the label is not look-ahead in the trading sense. Counts are reported in the summary.
6. Duration of an auction = the pre-registered duration of the nearest standard original term
   (``original_security_term`` parsed as years + months/12, e.g. "29-Year 9-Month" -> 30y -> 18.0),
   also for reopenings (the spec says "of the original term").
7. vol_IEF = 63-day realised volatility of daily close-to-close total-return returns, sample std x
   sqrt(252), all 63 returns required (the sleeve starts in April 2005 because the engine's data
   begin on 2005-01-03). The cap 2.0 applies to the weight after vol scaling.
8. Variants: ``q1`` = q fixed at 1 (no issuance sizing). ``delay_1d`` = decision weights shifted
   one more trading day (robustness, not part of the G3 neighbourhood). ``cost_2x`` = base at 2x
   costs. The ticker is a parameter only so the pre-registered Databento replication (ZN sampled at
   16:00 ET, HYPOTHESES.md section 4) can reuse the identical rule; nothing here ran on futures.
9. Prediction tests use IEF daily excess returns (close-to-close return minus that day's T-bill).
   A window's return is the SUM of its daily excess returns. (P1) daily mean inside [T-2, T] vs all
   other in-sample days: OLS of the daily excess return on a window dummy with Newey-West (5 lags)
   and Welch's t; plus the per-window mean with an i.i.d. t (the kill condition). (P2) OLS slope of
   the per-window return on the base q (HC1 t), also on the unclipped ratio, and tercile means.
   (P3) everything split at 2015-01-01 by the window's T.
10. (Reviewer fix, 2026-10-03.) Section 0 allows calendar facts only from the date they were public.
   Unscheduled NYSE closures (``UNSCHEDULED_CLOSURES``: Ford, Hurricane Sandy, G.H.W. Bush, Carter
   days) were each announced at most a few days ahead, after the decision that could depend on them.
   With ``calendar="ex_ante"`` (the default) the T-j offsets of the decision date, the issuance
   cut-off, the entry and the exit are counted on the calendar a trader expected at the decision
   date (realised sessions plus those closures); a decision/cut-off that falls on such a day uses the
   last real session before it, and an entry or exit planned for it is filled at the next real
   session's close. The only in-sample month this changes is October 2012 (Sandy closed 10-29 and
   10-30): the realised-calendar code held IEF over 10-25 and 10-26 on a decision taken 10-23, which
   used the closures before they were announced. ``calendar="realised"`` reproduces the earlier
   numbers.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import stats

from .. import analysis as A
from .. import calendar_utils as CU
from .. import config as C
from .. import engine as E

FAMILY = "IFC"
SLEEVE = "C"
EXEC = "next_close"
TICKERS = ["IEF"]
COST_BPS = None          # engine defaults (IEF tier 1: 3 bp one-way)

BASE_PARAMS: dict = {
    "ticker": "IEF",
    "window": (-2, 0),        # hold over return days [T+a, T+b]
    "issuance_sizing": True,  # False -> q = 1
    "scale": 0.10,            # w = q * scale / vol
    "cap": 2.0,
    "vol_window": 63,
    "q_clip": (0.5, 2.0),
    "median_months": 12,
    "extra_delay": 0,
    "calendar": "ex_ante",    # T-j counted on the calendar known at the decision date (note 10)
}

VARIANTS: dict[str, dict] = {
    "window2": {"window": (-1, 0)},          # pre-registered neighbourhood (G3)
    "window4": {"window": (-3, 0)},          # pre-registered neighbourhood (G3)
    "shift_early": {"window": (-3, -1)},     # pre-registered neighbourhood (G3)
    "shift_late": {"window": (-1, 1)},       # pre-registered neighbourhood (G3)
    "q1": {"issuance_sizing": False},        # pre-registered neighbourhood (G3)
    "delay_1d": {"extra_delay": 1},          # robustness: one extra day of execution delay
    "cost_2x": {"cost_mult": 2.0},           # robustness: base specification at 2x costs
}
NEIGHBOURHOOD = ["window2", "window4", "shift_early", "shift_late", "q1"]

DURATION = {2: 1.9, 3: 2.8, 5: 4.5, 7: 6.2, 10: 8.3, 20: 13.5, 30: 18.0}
TIPS_GAP_PP = 0.7
SPLIT_DATE = pd.Timestamp("2015-01-01")
PROFILE_OFFSETS = list(range(-10, 6))

# NYSE full-day closures in 2004 (only used to date the 2004 months' cut-offs, see note 3)
NYSE_HOLIDAYS_2004 = [
    "2004-01-01", "2004-01-19", "2004-02-16", "2004-04-09", "2004-05-31",
    "2004-06-11", "2004-07-05", "2004-09-06", "2004-11-25", "2004-12-24",
]
# Unscheduled NYSE closures announced only days ahead (note 10)
UNSCHEDULED_CLOSURES = pd.DatetimeIndex(["2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])


# --------------------------------------------------------------------------- auction data

def term_years(term: str) -> float:
    m = re.match(r"\s*(\d+)-Year(?:\s+(\d+)-Month)?", str(term))
    if not m:
        return float("nan")
    return int(m.group(1)) + (int(m.group(2)) / 12 if m.group(2) else 0.0)


def term_duration(term: str) -> tuple[int, float]:
    y = term_years(term)
    if not np.isfinite(y):
        return 0, float("nan")
    k = min(DURATION, key=lambda t: abs(t - y))
    return k, DURATION[k]


def load_nominal_auctions(period: str = "IS", return_report: bool = False):
    """Nominal (non-TIPS, non-FRN) Note/Bond auctions announced by the period's last visible date,
    with ``std_term``, ``duration`` and ``dv01 = offering_amt * duration`` columns."""
    end = pd.Timestamp(E.data_end(period))
    raw = E._read_parquet("treasury_auctions.parquet")
    a = raw[(raw["announcemt_date"] <= end) & raw["security_type"].isin(["Note", "Bond"])].copy()
    n_nb = len(a)
    # FRNs: no high yield (discount-margin priced)
    frn = a["high_yield"].isna()
    frn_terms = sorted(a.loc[frn, "original_security_term"].unique().tolist())
    frn_years = a.loc[frn, "auction_date"].dt.year
    a = a[~frn].copy()
    # TIPS: real-yield gap vs the FRED nominal curve, maximised per CUSIP
    fred = E.load_series("fred_daily.parquet", period)[["DGS2", "DGS10", "DGS30"]].ffill().dropna()
    pos = fred.index.searchsorted(a["auction_date"].to_numpy(), side="right") - 1
    curve = fred.to_numpy()[np.clip(pos, 0, None)]
    mat = (a["maturity_date"] - a["auction_date"]).dt.days.to_numpy() / 365.25
    nom = np.array([np.interp(m, [2.0, 10.0, 30.0], c) for m, c in zip(mat, curve)])
    a["nominal_curve_yield"] = nom
    a["real_gap"] = a["nominal_curve_yield"] - a["high_yield"]
    maxgap = a.groupby("cusip")["real_gap"].transform("max")
    tips = maxgap > TIPS_GAP_PP
    cus_gap = a.groupby("cusip")["real_gap"].max()
    is_tips_cusip = cus_gap > TIPS_GAP_PP
    report = {
        "note_bond_rows_announced_in_period": int(n_nb),
        "frn_rows_excluded": int(frn.sum()),
        "frn_original_terms": frn_terms,
        "frn_first_year": int(frn_years.min()) if len(frn_years) else None,
        "tips_rows_excluded": int(tips.sum()),
        "tips_cusips": int(is_tips_cusip.sum()),
        "nominal_cusips": int((~is_tips_cusip).sum()),
        "max_gap_nominal_cusips_pp": float(cus_gap[~is_tips_cusip].max()),
        "min_gap_tips_cusips_pp": float(cus_gap[is_tips_cusip].min()) if is_tips_cusip.any() else None,
        "tips_rows_by_term": a.loc[tips].groupby("original_security_term").size().to_dict(),
    }
    a = a[~tips].copy()
    st = a["original_security_term"].map(term_duration)
    a["std_term"] = [s[0] for s in st]
    a["duration"] = [s[1] for s in st]
    a["dv01"] = a["offering_amt"].astype(float) * a["duration"]
    report["nominal_rows"] = int(len(a))
    report["nominal_rows_by_std_term"] = a.groupby("std_term").size().to_dict()
    odd = a.loc[a["original_security_term"].map(term_years) != a["std_term"], "original_security_term"]
    report["odd_original_terms_mapped"] = {t: int(term_duration(t)[0]) for t in sorted(odd.unique())}
    a = a.sort_values(["auction_date", "cusip"]).reset_index(drop=True)
    cols = ["cusip", "security_type", "security_term", "original_security_term", "auction_date",
            "issue_date", "maturity_date", "announcemt_date", "offering_amt", "reopening", "std_term",
            "duration", "dv01"]
    return (a[cols], report) if return_report else a[cols]


# --------------------------------------------------------------------------- monthly issuance

def _params(params: dict) -> dict:
    p = {**BASE_PARAMS, **{k: v for k, v in params.items() if k != "cost_mult"}}
    unknown = set(p) - set(BASE_PARAMS)
    if unknown:
        raise KeyError(f"unknown params {sorted(unknown)}")
    p["window"] = tuple(int(x) for x in p["window"])
    p["q_clip"] = tuple(float(x) for x in p["q_clip"])
    if p["calendar"] not in ("ex_ante", "realised"):
        raise ValueError(f"calendar must be 'ex_ante' or 'realised', got {p['calendar']!r}")
    return p


def expected_calendar(real: pd.DatetimeIndex, calendar: str = "ex_ante") -> pd.DatetimeIndex:
    """Calendar on which T-j is counted (note 10): the realised sessions plus, for ``ex_ante``, the
    unscheduled closures strictly inside their span (sessions a trader still expected at the decision)."""
    if calendar == "realised":
        return real
    u = UNSCHEDULED_CLOSURES[(UNSCHEDULED_CLOSURES > real[0]) & (UNSCHEDULED_CLOSURES < real[-1])]
    return real.union(u.astype(real.dtype))


def extended_calendar(period: str = "IS") -> pd.DatetimeIndex:
    """The engine's NYSE calendar preceded by 2004 NYSE trading days (for the 2004 medians)."""
    cal = E.trading_calendar(period)
    d04 = pd.bdate_range("2004-01-01", "2004-12-31")
    d04 = d04[~d04.isin(pd.DatetimeIndex(NYSE_HOLIDAYS_2004))]
    d04 = d04[d04 < cal[0]]
    return pd.DatetimeIndex(np.concatenate([d04.values.astype("datetime64[ns]"),
                                            cal.values.astype("datetime64[ns]")]))


def monthly_issuance(period: str = "IS", auctions: pd.DataFrame | None = None, **params) -> pd.DataFrame:
    """One row per complete month: T, cut-off date (T + a - 2), S_m, the trailing median of the
    previous ``median_months`` S values, the raw ratio and q."""
    p = _params(params)
    a_off = p["window"][0]
    if auctions is None:
        auctions = load_nominal_auctions(period)
    real = extended_calendar(period)
    cal = expected_calendar(real, p["calendar"])
    mo = CU.month_offsets(cal)
    t_idx = mo["T_i"].dropna().astype(int).unique()
    iss_month = auctions["issue_date"].dt.to_period("M")
    ann = auctions["announcemt_date"].to_numpy()
    dv01 = auctions["dv01"].to_numpy()
    rows = []
    for ti in t_idx:
        ci = ti + a_off - 2
        if ci < 0:
            continue
        T, cut = cal[ti], cal[ci]
        cut = real[real.searchsorted(cut, side="right") - 1]   # last real session on or before it
        m = T.to_period("M")
        sel = (iss_month == m).to_numpy() & (ann <= np.datetime64(cut))
        rows.append({"month": m, "T": T, "cutoff": cut, "S": float(dv01[sel].sum()),
                     "n_auctions": int(sel.sum()),
                     "n_issued_total": int((iss_month == m).sum())})
    mt = pd.DataFrame(rows).set_index("month")
    k = int(p["median_months"])
    mt["median_prev"] = mt["S"].shift(1).rolling(k, min_periods=k).median()
    mt["ratio"] = mt["S"] / mt["median_prev"]
    lo, hi = p["q_clip"]
    mt["q"] = mt["ratio"].clip(lo, hi) if p["issuance_sizing"] else np.where(mt["ratio"].notna(), 1.0, np.nan)
    return mt


# --------------------------------------------------------------------------- weights

def realised_vol(close: pd.Series, window: int = 63) -> pd.Series:
    r = close.pct_change(fill_method=None)
    return r.rolling(window, min_periods=window).std() * math.sqrt(C.TRADING_DAYS)


def hold_weights(period: str = "IS", ohlc: dict | None = None, auctions: pd.DataFrame | None = None,
                 **params) -> pd.DataFrame:
    """Desired holdings per return day (index = trading days, one column = the ticker)."""
    p = _params(params)
    tk = p["ticker"]
    if ohlc is None:
        ohlc = E.load_ohlc([tk], period)
    close = ohlc["close"][tk]
    cal = close.index
    n = len(cal)
    vol = realised_vol(close, p["vol_window"]).to_numpy()
    mt = monthly_issuance(period, auctions=auctions, **p)
    q_by_month = mt["q"].to_dict()
    a, b = p["window"]
    calx = expected_calendar(cal, p["calendar"])   # T-j counted as known at the decision (note 10)
    nx = len(calx)
    mo = CU.month_offsets(calx)
    hold = np.zeros(n)
    used = np.zeros(n, dtype=bool)
    for ti in mo["T_i"].dropna().astype(int).unique():
        q = q_by_month.get(calx[ti].to_period("M"), np.nan)
        if not np.isfinite(q) or ti + a - 2 < 0:
            continue
        d = cal.searchsorted(calx[ti + a - 2], side="right") - 1   # decision: last real session <= T+a-2
        v = vol[d]
        if not np.isfinite(v) or v <= 0:
            continue
        e = cal.searchsorted(calx[ti + a - 1], side="left")             # entry filled at this close
        x = cal.searchsorted(calx[min(ti + b, nx - 1)], side="left")    # exit filled at this close
        lo, hi = e + 1, min(x, n - 1)
        if lo > hi:
            continue
        assert not used[lo:hi + 1].any(), f"overlapping windows around {calx[ti]}"
        hold[lo:hi + 1] = min(q * p["scale"] / v, p["cap"])
        used[lo:hi + 1] = True
    return pd.DataFrame({tk: hold}, index=cal)


def decision_weights(period: str = "IS", ohlc: dict | None = None, auctions: pd.DataFrame | None = None,
                     **params) -> pd.DataFrame:
    """Engine decision weights (value at d uses information up to the close of d)."""
    p = _params(params)
    hold = hold_weights(period, ohlc=ohlc, auctions=auctions, **p)
    w = CU.hold_to_decision(hold, EXEC)
    if p["extra_delay"]:
        w = w.shift(int(p["extra_delay"])).fillna(0.0)
    return w


# --------------------------------------------------------------------------- prediction tests

def _f(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else x


def _tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def excess_returns(ohlc: dict, rf: pd.Series, ticker: str) -> pd.Series:
    close = ohlc["close"][ticker]
    return close.pct_change(fill_method=None) - rf.reindex(close.index).ffill().fillna(0.0)


def window_returns(period: str, ex: pd.Series, mt: pd.DataFrame, window=(-2, 0)) -> pd.DataFrame:
    """Per complete month: the sum of daily excess returns over the window, with q and ratio."""
    cal = ex.index
    exv = ex.to_numpy()
    n = len(cal)
    start, end = (pd.Timestamp(x) for x in E.eval_window(period))
    mo = CU.month_offsets(cal)
    a, b = window
    rows = []
    for ti in mo["T_i"].dropna().astype(int).unique():
        lo, hi = ti + a, ti + b
        if lo < 1 or hi >= n or cal[lo] < start or cal[hi] > end:
            continue
        m = cal[ti].to_period("M")
        if m not in mt.index:
            continue
        rows.append({"T": cal[ti], "ret": float(np.nansum(exv[lo:hi + 1])),
                     "q": mt.at[m, "q"], "ratio": mt.at[m, "ratio"], "S": mt.at[m, "S"]})
    return pd.DataFrame(rows)


def _dummy_test(ex: pd.Series, mask: pd.Series) -> dict:
    import statsmodels.api as sm

    df = pd.DataFrame({"y": ex, "d": mask.astype(float)}).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df["d"])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    inw, outw = df.loc[df["d"] == 1, "y"], df.loc[df["d"] == 0, "y"]
    w = stats.ttest_ind(inw, outw, equal_var=False)
    return {
        "n_window_days": int(len(inw)), "n_other_days": int(len(outw)),
        "mean_window_day_bp": _f(inw.mean() * 1e4), "mean_other_day_bp": _f(outw.mean() * 1e4),
        "diff_bp": _f(fit.params["d"] * 1e4), "diff_t_newey_west": _f(fit.tvalues["d"]),
        "diff_t_welch": _f(w.statistic), "diff_p_welch": _f(w.pvalue),
    }


def _window_mean(wr: pd.DataFrame) -> dict:
    x = wr["ret"].astype(float)
    return {"n_months": int(len(x)), "mean_window_ret_bp": _f(x.mean() * 1e4), "t": _f(_tstat(x)),
            "hit_rate": _f((x > 0).mean()), "median_bp": _f(x.median() * 1e4)}


def _slope(wr: pd.DataFrame, xcol: str) -> dict:
    import statsmodels.api as sm

    df = wr[["ret", xcol]].dropna().astype(float)
    if len(df) < 10:
        return {"n": int(len(df))}
    fit = sm.OLS(df["ret"], sm.add_constant(df[xcol])).fit(cov_type="HC1")
    return {"n": int(len(df)), "slope_bp_per_unit": _f(fit.params[xcol] * 1e4),
            "t_hc1": _f(fit.tvalues[xcol]), "intercept_bp": _f(fit.params["const"] * 1e4),
            "r2": _f(fit.rsquared), "spearman_rho": _f(stats.spearmanr(df[xcol], df["ret"]).statistic)}


def prediction_tests(period: str, ohlc: dict, rf: pd.Series, base: E.BacktestResult,
                     auctions: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    p = _params({})
    tk = p["ticker"]
    ex = excess_returns(ohlc, rf, tk)
    start, end = E.eval_window(period)
    ex = ex.loc[start:end].iloc[1:]           # first day has no return
    mt = monthly_issuance(period, auctions=auctions, **p)
    wr = window_returns(period, ex, mt, p["window"])
    wr = wr[wr["q"].notna()].reset_index(drop=True)   # months with a q (all IS months)
    mask = CU.window_mask(ex.index, *p["window"])
    out: dict = {}
    out["P1_window_vs_other_days"] = _dummy_test(ex, mask)
    out["P1_window_mean_kill_condition"] = _window_mean(wr)
    out["P1_kill_condition_triggered"] = bool(wr["ret"].mean() <= 0)
    out["P2_slope_on_q"] = _slope(wr, "q")
    out["P2_slope_on_unclipped_ratio"] = _slope(wr, "ratio")
    terc = pd.qcut(wr["q"].rank(method="first"), 3, labels=["low", "mid", "high"])
    out["P2_q_terciles"] = {
        str(k): {"n": int(len(g)), "mean_q": _f(g["q"].mean()), "mean_window_ret_bp": _f(g["ret"].mean() * 1e4),
                 "t": _f(_tstat(g["ret"]))} for k, g in wr.groupby(terc, observed=True)}
    out["P2_high_minus_low_tercile_bp"] = _f(out["P2_q_terciles"]["high"]["mean_window_ret_bp"]
                                            - out["P2_q_terciles"]["low"]["mean_window_ret_bp"])
    out["P2_share_months_q_at_clip"] = {"at_0.5": _f((wr["q"] <= 0.5).mean()), "at_2.0": _f((wr["q"] >= 2.0).mean())}
    # P3: pre / post 2015
    split = {}
    for lab, sel_m, sel_d in (("pre_2015", wr["T"] < SPLIT_DATE, ex.index < SPLIT_DATE),
                              ("post_2015", wr["T"] >= SPLIT_DATE, ex.index >= SPLIT_DATE)):
        w_ = wr[sel_m]
        exb = base.excess[(base.excess.index < SPLIT_DATE) if lab == "pre_2015" else (base.excess.index >= SPLIT_DATE)]
        split[lab] = {
            "window_mean": _window_mean(w_),
            "window_vs_other_days": _dummy_test(ex[sel_d], mask[sel_d]),
            "slope_on_q": _slope(w_, "q"),
            "strategy_sharpe": _f(exb.mean() / exb.std() * math.sqrt(C.TRADING_DAYS)) if exb.std() > 0 else None,
            "strategy_nw_t": _f(E.newey_west_tstat(exb)),
        }
    pre, post = wr.loc[wr["T"] < SPLIT_DATE, "ret"], wr.loc[wr["T"] >= SPLIT_DATE, "ret"]
    wt = stats.ttest_ind(pre, post, equal_var=False)
    split["post_minus_pre_window_mean_bp"] = _f((post.mean() - pre.mean()) * 1e4)
    split["welch_t"] = _f(wt.statistic)
    out["P3_pre_post_2015"] = split
    # event-time profile of daily IEF excess returns around month end (diagnostic for the figure)
    cal = ex.index
    mo = CU.month_offsets(cal)
    exv = ex.to_numpy()
    prof_rows = []
    for ti in mo["T_i"].dropna().astype(int).unique():
        row = {"T": cal[ti]}
        for o in PROFILE_OFFSETS:
            j = ti + o
            row[o] = exv[j] if 0 <= j < len(cal) else np.nan
        prof_rows.append(row)
    prof = pd.DataFrame(prof_rows)
    out["profile_mean_bp_by_offset"] = {f"{o:+d}" if o else "T": _f(prof[o].mean() * 1e4) for o in PROFILE_OFFSETS}
    out["profile_t_by_offset"] = {f"{o:+d}" if o else "T": _f(_tstat(prof[o])) for o in PROFILE_OFFSETS}
    return out, wr, prof


# --------------------------------------------------------------------------- run

def _stats_js(s: dict) -> dict:
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()}


def run(period: str = "IS", log: bool = True, return_results: bool = False):
    """Run the base specification and every variant; return a JSON-serialisable summary
    (and, with ``return_results=True``, also the BacktestResults and test tables)."""
    tk = BASE_PARAMS["ticker"]
    ohlc = E.load_ohlc([tk], period)
    rf = E.load_rf(period)
    auctions, report = load_nominal_auctions(period, return_report=True)

    def bt(name: str, overrides: dict) -> E.BacktestResult:
        cost_mult = float(overrides.get("cost_mult", 1.0))
        p = _params(overrides)
        w = decision_weights(period, ohlc=ohlc, auctions=auctions, **p)
        params = {"sleeve": SLEEVE, "variant": name, **p}
        return E.run_backtest(f"IFC_C_{name}", FAMILY, w, ohlc, rf, period=period, params=params,
                              exec=EXEC, cost_mult=cost_mult, log=log)

    results = {"base": bt("base", {})}
    for vname, ov in VARIANTS.items():
        results[vname] = bt(vname, ov)
    base = results["base"]

    preds, wr, prof = prediction_tests(period, ohlc, rf, base, auctions)
    mt = monthly_issuance(period, auctions=auctions)
    lo, hi = A.bootstrap_sharpe_ci(base.excess)
    nb_sr = [results[v].stats["sharpe"] for v in NEIGHBOURHOOD]
    nb_med = float(np.median(nb_sr))
    held = base.weights[tk]
    active = held[held != 0]
    conc = A.pnl_concentration(base)
    gates = {   # the selection gates are defined for the IFC composite; shown for this sleeve only
        "G1_sharpe_2x_costs_gt_0": bool(results["cost_2x"].stats["sharpe"] > 0),
        "G2_best_year_share_le_40pct": bool(conc["best_year_share"] <= 0.40),
        "G3_neighbourhood_median_ge_half_base": bool(nb_med >= 0.5 * base.stats["sharpe"]),
        "neighbourhood_median_sharpe": nb_med,
    }
    mt_live = mt[mt.index >= pd.Period(base.returns.index[0], "M")]
    summary = {
        "module": "ifc_treasury", "family": FAMILY, "sleeve": SLEEVE, "period": period, "exec": EXEC,
        "base_params": BASE_PARAMS, "variants_spec": VARIANTS, "neighbourhood": NEIGHBOURHOOD,
        "auction_data": report,
        "base": _stats_js(base.stats),
        "base_cost_2x": _stats_js(results["cost_2x"].stats),
        "variants": {v: {"sharpe": _f(r.stats["sharpe"]), "ann_return": _f(r.stats["ann_return"]),
                         "ann_vol": _f(r.stats["ann_vol"]), "max_drawdown": _f(r.stats["max_drawdown"]),
                         "turnover_per_year": _f(r.stats["turnover_per_year"]),
                         "nw_t": _f(r.stats["nw_tstat_mean_excess"])}
                     for v, r in results.items() if v != "base"},
        "subperiods": A.subperiod_table(base).to_dict(orient="records"),
        "pnl_concentration": {k: (_f(v) if isinstance(v, float) else v) for k, v in conc.items()},
        "bootstrap_sharpe_90ci": [lo, hi],
        "yearly_returns": {int(k): _f(v) for k, v in E.yearly_returns(base.returns).items()},
        "yearly_excess_sum": {int(k): _f(v) for k, v in base.excess.groupby(base.excess.index.year).sum().items()},
        "position": {"mean_weight_when_held": _f(active.mean()), "max_weight": _f(active.max()),
                     "share_days_held": _f((held != 0).mean()),
                     "share_windows_at_cap": _f((active >= BASE_PARAMS["cap"] - 1e-12).mean()),
                     "costs_and_borrow_ann": _f(base.costs.sum() / (len(held) / C.TRADING_DAYS))},
        "issuance": {"months": int(mt_live["q"].notna().sum()),
                     "q_mean": _f(mt_live["q"].mean()), "q_median": _f(mt_live["q"].median()),
                     "ratio_quantiles": {str(k): _f(v) for k, v in mt_live["ratio"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()},
                     "auctions_counted_share": _f(mt_live["n_auctions"].sum() / max(mt_live["n_issued_total"].sum(), 1))},
        "predictions": preds,
        "gates_sleeve_only": gates,
    }
    try:
        summary["factor_table"] = A.factor_table(base)
    except Exception as exc:  # pragma: no cover - reported, not hidden
        summary["factor_table"] = {"error": repr(exc)}
    if return_results:
        return summary, {"results": results, "window_returns": wr, "profile": prof, "monthly": mt}
    return summary
