"""IFC exploratory sleeve E: Treasury auction cycle (HYPOTHESES.md section 1, "Exploratory sleeves").

Reported only, never part of the IFC composite.

Rule as pre-registered
----------------------
For each Note/Bond auction on day ``A`` (announced by the decision date), short IEF over return
days [``A-4``, ``A``] and long over [``A+1``, ``A+4``], sized by the auction's DV01 relative to the
trailing 12-month mean auction DV01, 0.10 / vol_IEF per unit, capped at 2. (Lou, Yan & Zhang 2013;
reported decayed after 2014 by Fleming, Liu & Nguyen 2026.)

Interpretation decisions (fixed before the first backtest of this module; no alternative was run)
-------------------------------------------------------------------------------------------------
1. Auctions: the same nominal Note/Bond set as sleeve C (``ifc_treasury.load_nominal_auctions``):
   ``security_type`` in {Note, Bond}, TIPS and FRNs removed (the cache has no flag; FRN = no high
   yield, TIPS = real-yield gap > 0.7 pp vs the FRED nominal curve, see ifc_treasury note 5), only
   auctions announced by the period's last visible date (2024-10-02 in-sample). ``A`` = the auction
   date on the NYSE calendar (every in-sample auction date is an NYSE trading day). ``A+j`` are
   integer trading-day offsets; dates before 2005-01-03 use the 2004 NYSE calendar of sleeve C and
   dates after the period end use weekdays (only needed to index late auctions; no position is
   ever taken outside the period).
2. DV01 of an auction = offering amount x the pre-registered duration of its (nearest standard)
   original term, as in sleeve C. The trailing 12-month mean is taken over the nominal auctions
   whose AUCTION date lies in [announcement date - 12 months, announcement date): all of them are
   completed and public when the auction is announced. ``u_A = DV01_A / that mean``.
3. "Announced by the decision date" is applied per return day: under ``next_close`` the holding on
   return day t is decided at the close of t-2, so auction A contributes to day t only if its
   announcement date is <= t-2. Treasury announces 2-6 trading days before the auction (mostly
   3-5 in-sample), so the short window [A-4, A] is traded only on its return days t with
   t-2 >= the announcement day (54 % of short-window days in-sample; a strict per-window reading,
   announcement <= A-6, would drop ~95 % of the short legs and would no longer test the
   hypothesis). The long window [A+1, A+4] is always tradable. Traded shares are reported.
4. Overlapping auctions ADD (each auction is a separate supply event), with the short leg -u_A and
   the long leg +u_A per day; the net per-day sum is multiplied by 0.10 / vol_IEF and the result
   is capped at |w| <= 2. vol_IEF = 63-day realised volatility (sample std x sqrt(252), all 63
   returns required) observed at the decision date t-2 of each return day (positions change daily
   anyway because auctions overlap).
5. Variants: ``delay_1d`` = decision weights shifted one more trading day; ``cost_2x`` = base at 2x
   costs. Sleeve E has no pre-registered neighbourhood. The ticker is a parameter only so the
   Databento replication (ZN sampled at 16:00 ET) could reuse it; nothing here ran on futures.
6. Prediction tests use IEF daily excess returns (close-to-close minus that day's T-bill) around
   every nominal auction with A-5 .. A+5 inside the period: the mean by event day A-5..A+5 (i.i.d.
   t across auctions), and per-auction window sums for [A-4, A] and [A+1, A+4] with a Newey-West t
   over the date-ordered auction sequence (5 lags, because windows of auctions in the same or
   adjacent weeks overlap). Everything is split at 2014-01-01 by the auction date.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .. import analysis as A
from .. import config as C
from .. import engine as E
from . import ifc_treasury as TC

FAMILY = "IFC_EXPLORATORY"
SLEEVE = "E"
EXEC = "next_close"
TICKERS = ["IEF"]
COST_BPS = None          # engine defaults (IEF tier 1: 3 bp one-way)

BASE_PARAMS: dict = {
    "ticker": "IEF",
    "pre": (-4, 0),          # short over return days [A-4, A]
    "post": (1, 4),          # long over return days [A+1, A+4]
    "scale": 0.10,
    "cap": 2.0,
    "vol_window": 63,
    "trailing_months": 12,
    "extra_delay": 0,
}

VARIANTS: dict[str, dict] = {
    "delay_1d": {"extra_delay": 1},          # robustness: one extra day of execution delay
    "cost_2x": {"cost_mult": 2.0},           # robustness: base specification at 2x costs
}
NEIGHBOURHOOD: list[str] = []

SPLIT_DATE = pd.Timestamp("2014-01-01")
EVENT_OFFSETS = list(range(-5, 6))


def _params(params: dict) -> dict:
    p = {**BASE_PARAMS, **{k: v for k, v in params.items() if k != "cost_mult"}}
    unknown = set(p) - set(BASE_PARAMS)
    if unknown:
        raise KeyError(f"unknown params {sorted(unknown)}")
    p["pre"] = tuple(int(x) for x in p["pre"])
    p["post"] = tuple(int(x) for x in p["post"])
    return p


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


# --------------------------------------------------------------------------- events

def index_calendar(period: str = "IS") -> pd.DatetimeIndex:
    """2004 NYSE days + the period's NYSE calendar + 20 weekdays after its end (indexing only)."""
    from .. import calendar_utils as CU

    cal = CU.planned_calendar(TC.extended_calendar(period))   # unscheduled closures = expected sessions
    tail = pd.bdate_range(cal[-1] + pd.Timedelta(days=1), periods=20)
    return pd.DatetimeIndex(np.concatenate([cal.values, tail.values.astype("datetime64[ns]")]))


def auction_events(period: str = "IS", auctions: pd.DataFrame | None = None, **params) -> pd.DataFrame:
    """Nominal auctions with their calendar index, announcement index and relative DV01 u_A."""
    p = _params(params)
    if auctions is None:
        auctions = TC.load_nominal_auctions(period)
    ev = auctions.copy()
    cal = index_calendar(period)
    ev = ev[ev["auction_date"] >= cal[0]].reset_index(drop=True)
    ev["A_i"] = cal.searchsorted(ev["auction_date"].to_numpy(), side="left")
    ev["A_on_cal"] = cal[np.clip(ev["A_i"], 0, len(cal) - 1)] == ev["auction_date"].to_numpy()
    ev["ann_i"] = cal.searchsorted(ev["announcemt_date"].to_numpy(), side="right") - 1  # last day <= ann
    adate = auctions["auction_date"].to_numpy()
    dv = auctions["dv01"].to_numpy()
    k = int(p["trailing_months"])
    ref = []
    for ann in ev["announcemt_date"]:
        lo = np.datetime64(ann - pd.DateOffset(months=k))
        sel = (adate >= lo) & (adate < np.datetime64(ann))
        ref.append(dv[sel].mean() if sel.any() else np.nan)
    ev["dv01_ref"] = ref
    ev["u"] = ev["dv01"] / ev["dv01_ref"]
    ev["lead_days"] = ev["A_i"] - ev["ann_i"]
    return ev


# --------------------------------------------------------------------------- weights

def hold_weights(period: str = "IS", ohlc: dict | None = None, auctions: pd.DataFrame | None = None,
                 return_units: bool = False, **params):
    p = _params(params)
    tk = p["ticker"]
    if ohlc is None:
        ohlc = E.load_ohlc([tk], period)
    from .. import calendar_utils as CU

    close = CU.to_planned(ohlc["close"][tk])     # planned calendar (calendar_utils)
    cal = close.index
    icx = index_calendar(period)
    pos_in_cal = pd.Series(np.arange(len(cal)), index=cal)
    ev = auction_events(period, auctions=auctions, **p)
    units = np.zeros(len(cal))
    legs = {"pre_days_total": 0, "pre_days_traded": 0, "post_days_total": 0, "post_days_traded": 0}
    for row in ev.itertuples(index=False):
        if not np.isfinite(row.u):
            continue
        for (a, b), sign, key in ((p["pre"], -1.0, "pre"), (p["post"], 1.0, "post")):
            for j in range(row.A_i + a, row.A_i + b + 1):
                if j - 2 < 0 or j >= len(icx):
                    continue
                d = icx[j]
                if d not in pos_in_cal.index:
                    continue
                legs[f"{key}_days_total"] += 1
                if icx[j - 2] < row.announcemt_date:      # not yet announced at the decision date
                    continue
                legs[f"{key}_days_traded"] += 1
                units[pos_in_cal[d]] += sign * row.u
    vol = TC.realised_vol(close, p["vol_window"]).shift(2)   # observed at the close of t-2
    w = (pd.Series(units, index=cal) * p["scale"] / vol).clip(-p["cap"], p["cap"]).fillna(0.0)
    hold = pd.DataFrame({tk: w}, index=cal)
    if return_units:
        return hold, pd.Series(units, index=cal), legs
    return hold


def decision_weights(period: str = "IS", ohlc: dict | None = None, auctions: pd.DataFrame | None = None,
                     **params) -> pd.DataFrame:
    """Engine decision weights (value at d uses information up to the close of d)."""
    from .. import calendar_utils as CU

    p = _params(params)
    if ohlc is None:
        ohlc = E.load_ohlc([p["ticker"]], period)
    hold = hold_weights(period, ohlc=ohlc, auctions=auctions, **p)
    hold = CU.planned_to_realized(hold, ohlc["close"].index)
    w = CU.hold_to_decision(hold, EXEC)
    if p["extra_delay"]:
        w = w.shift(int(p["extra_delay"])).fillna(0.0)
    return w


# --------------------------------------------------------------------------- prediction tests

def event_table(period: str, ex: pd.Series, ev: pd.DataFrame) -> pd.DataFrame:
    """One row per auction with A-5..A+5 inside the period: daily excess returns by event day and
    the [A-4, A] and [A+1, A+4] window sums."""
    cal = ex.index
    exv = ex.to_numpy()
    pos = pd.Series(np.arange(len(cal)), index=cal)
    rows = []
    for row in ev.itertuples(index=False):
        if row.auction_date not in pos.index:
            continue
        i = int(pos[row.auction_date])
        if i - 5 < 0 or i + 5 >= len(cal):
            continue
        r = {"A": row.auction_date, "std_term": row.std_term, "u": row.u}
        for k in EVENT_OFFSETS:
            r[k] = exv[i + k]
        r["pre"] = float(np.nansum(exv[i - 4:i + 1]))
        r["post"] = float(np.nansum(exv[i + 1:i + 5]))
        rows.append(r)
    t = pd.DataFrame(rows)
    t["post_minus_pre"] = t["post"] - t["pre"]
    return t


def _seq_stats(x: pd.Series) -> dict:
    x = x.astype(float)
    return {"n": int(len(x)), "mean_bp": _f(x.mean() * 1e4), "t_iid": _f(_tstat(x)),
            "t_newey_west_5": _f(E.newey_west_tstat(x, lags=5))}


def _event_block(t: pd.DataFrame) -> dict:
    out = {"n_auctions": int(len(t))}
    out["by_event_day"] = {f"A{k:+d}" if k else "A": {"mean_bp": _f(t[k].mean() * 1e4), "t": _f(_tstat(t[k]))}
                           for k in EVENT_OFFSETS}
    out["pre_window_A-4_A"] = _seq_stats(t["pre"])
    out["post_window_A+1_A+4"] = _seq_stats(t["post"])
    out["post_minus_pre"] = _seq_stats(t["post_minus_pre"])
    return out


def prediction_tests(period: str, ohlc: dict, rf: pd.Series, base: E.BacktestResult,
                     ev: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    tk = BASE_PARAMS["ticker"]
    close = ohlc["close"][tk]
    ex = close.pct_change(fill_method=None) - rf.reindex(close.index).ffill().fillna(0.0)
    start, end = E.eval_window(period)
    ex = ex.loc[start:end].iloc[1:]
    t = event_table(period, ex, ev)
    out = {"all": _event_block(t)}
    pre14, post14 = t[t["A"] < SPLIT_DATE], t[t["A"] >= SPLIT_DATE]
    out["pre_2014"] = _event_block(pre14)
    out["post_2014"] = _event_block(post14)
    w = stats.ttest_ind(post14["post_minus_pre"], pre14["post_minus_pre"], equal_var=False)
    out["post2014_minus_pre2014_spread_bp"] = _f((post14["post_minus_pre"].mean() - pre14["post_minus_pre"].mean()) * 1e4)
    out["post2014_minus_pre2014_welch_t"] = _f(w.statistic)
    for lab, sel in (("pre_2014", base.excess.index < SPLIT_DATE), ("post_2014", base.excess.index >= SPLIT_DATE)):
        x = base.excess[sel]
        out[lab]["strategy_sharpe"] = _f(x.mean() / x.std() * math.sqrt(C.TRADING_DAYS)) if x.std() > 0 else None
        out[lab]["strategy_nw_t"] = _f(E.newey_west_tstat(x))
    out["by_term_post_minus_pre_bp"] = {int(k): {"n": int(len(g)), "mean_bp": _f(g["post_minus_pre"].mean() * 1e4),
                                                 "t_iid": _f(_tstat(g["post_minus_pre"]))}
                                        for k, g in t.groupby("std_term")}
    return out, t


# --------------------------------------------------------------------------- run

def _stats_js(s: dict) -> dict:
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()}


def run(period: str = "IS", log: bool = True, return_results: bool = False):
    """Run the base specification and the robustness variants; return a JSON-serialisable summary
    (and, with ``return_results=True``, also the BacktestResults and the event table)."""
    tk = BASE_PARAMS["ticker"]
    ohlc = E.load_ohlc([tk], period)
    rf = E.load_rf(period)
    auctions, report = TC.load_nominal_auctions(period, return_report=True)

    def bt(name: str, overrides: dict) -> E.BacktestResult:
        cost_mult = float(overrides.get("cost_mult", 1.0))
        p = _params(overrides)
        w = decision_weights(period, ohlc=ohlc, auctions=auctions, **p)
        params = {"sleeve": SLEEVE, "variant": name, **p}
        return E.run_backtest(f"IFC_E_{name}", FAMILY, w, ohlc, rf, period=period, params=params,
                              exec=EXEC, cost_mult=cost_mult, log=log)

    results = {"base": bt("base", {})}
    for vname, ov in VARIANTS.items():
        results[vname] = bt(vname, ov)
    base = results["base"]

    ev = auction_events(period, auctions=auctions)
    hold, units, legs = hold_weights(period, ohlc=ohlc, auctions=auctions, return_units=True)
    preds, evt = prediction_tests(period, ohlc, rf, base, ev)
    lo, hi = A.bootstrap_sharpe_ci(base.excess)
    held = base.weights[tk]
    active = held[held != 0]
    conc = A.pnl_concentration(base)
    ev_is = ev[(ev["auction_date"] >= pd.Timestamp(E.eval_window(period)[0]))]
    summary = {
        "module": "ifc_auction", "family": FAMILY, "sleeve": SLEEVE, "period": period, "exec": EXEC,
        "base_params": BASE_PARAMS, "variants_spec": VARIANTS,
        "auction_data": report,
        "events": {
            "n_auctions_in_period": int(len(ev_is)),
            "announcement_lead_trading_days": {int(k): int(v) for k, v in ev_is["lead_days"].value_counts().sort_index().items()},
            "auction_dates_not_trading_days": int((~ev_is["A_on_cal"]).sum()),
            "u_quantiles": {str(k): _f(v) for k, v in ev_is["u"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).items()},
            **{k: int(v) for k, v in legs.items()},
            "pre_window_traded_share": _f(legs["pre_days_traded"] / max(legs["pre_days_total"], 1)),
            "post_window_traded_share": _f(legs["post_days_traded"] / max(legs["post_days_total"], 1)),
        },
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
        "position": {"mean_abs_weight_when_held": _f(active.abs().mean()), "share_days_held": _f((held != 0).mean()),
                     "share_days_long": _f((held > 0).mean()), "share_days_short": _f((held < 0).mean()),
                     "share_held_days_at_cap": _f((active.abs() >= BASE_PARAMS["cap"] - 1e-12).mean()),
                     "mean_weight": _f(held.mean()),
                     "costs_and_borrow_ann": _f(base.costs.sum() / (len(held) / C.TRADING_DAYS))},
        "predictions": preds,
    }
    try:
        summary["factor_table"] = A.factor_table(base)
    except Exception as exc:  # pragma: no cover - reported, not hidden
        summary["factor_table"] = {"error": repr(exc)}
    if return_results:
        return summary, {"results": results, "events": evt, "units": units}
    return summary
