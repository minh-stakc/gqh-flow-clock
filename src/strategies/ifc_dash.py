"""IFC sleeve B: settlement-cycle "dash for cash" (HYPOTHESES.md section 1, Sleeve B).

Rule (as pre-registered). For month m with last trading day T and U.S. equity settlement lag
k = calendar_utils.equity_settlement_lag(T) (3 if T < 2017-09-05, 2 until 2024-05-27, 1 from
2024-05-28):

* pressure window  = return days [T-k-5, T-k-1]: hold w_SPY = -0.10 / vol_SPY
* liquidity window = return days [T-k,   T+3]:   hold w_SPY = +0.10 / vol_SPY
* 0 otherwise.  Execution: next_close (decision at close d, filled at close d+1).

Interpretation decisions (the text is silent or ambiguous on these; each is the most natural
reading and no alternative was tried for performance):

1. vol_SPY is the 63-day realised volatility of SPY daily close-to-close (total-return-adjusted)
   returns, sample std (ddof=1) x sqrt(252), requiring all 63 returns.
2. "Observed at the decision date" is read per window, following section 0 ("a data-dependent
   signal used for a hold window [a, b] under next_close is observed at the close of a-2"): each
   window's size is fixed for the whole window using the vol observed at the close of a-2, where a
   is that window's first return day. The pressure and liquidity windows are separate windows, so
   each has its own decision date (T-k-7 and T-k-2 respectively). Windows whose decision-date vol
   is unavailable (warm-up) are skipped.
3. T+j for j >= 1 is the j-th trading day of the following month (index T_i + j on the NYSE/SPY
   calendar); T-j is index T_i - j. Months are taken from calendar_utils.month_offsets; the final,
   incomplete month of the data (whose T is unknown) gets no windows. A window truncated by the end
   of the data (Sep 2024: T+3 = 2024-10-03 is out of sample) is traded on the days that exist.
4. The regime of a month (and k) is set by that month's T, so the liquidity days T+1..T+3 that fall
   in the next calendar month belong to the previous month's window.
5. No position cap inside the sleeve (the composite applies |w| <= 3 and gross <= 4).
6. Windows never overlap for any pre-registered variant (checked by an assertion).
7. Neighbourhood variants (section 3, IFC points that touch sleeve B): "all windows shifted one day
   earlier / later" = both windows moved by -1 / +1 trading day, with k unchanged. Counterfactual
   "fixed T+3 windows throughout" = k = 3 for every month. "One extra day of execution delay" =
   decision weights shifted one more trading day (w_dec.shift(1)). "2x costs" = cost_mult 2.
8. Prediction tests use SPY daily excess returns (close-to-close minus the T-bill of that day);
   a window's return is the SUM of its daily excess returns (this matches the P&L of a constant
   weight held over the window; the difference to compounding is negligible). A month enters the
   spread tests only if both windows are complete inside the evaluation period, so Sep 2024 (T+3
   missing) is excluded and the T+1 regime has 4 in-sample months.
9. t-statistics on monthly spreads are plain i.i.d. t (mean / (sd / sqrt(n))); a Newey-West t is
   also reported. B3 uses Welch's unequal-variance t over all complete months (all regimes).
10. B2's "Sharpe over T+2 months": daily excess returns of the base and fixed-T+3 backtests (1x
   costs) over the date range from the day after the last T+3 month's liquidity window ends to the
   last day of the last T+2 month's liquidity window; inside that range both strategies trade only
   T+2-regime months' windows.
11. The liquidity window is longer than the pressure window (7 vs 5 days under T+3, 6 vs 5 under
   T+2, 5 vs 5 under T+1), so the sleeve is net long on average and the raw spread contains an
   equity-premium component of about (L_liq - L_press) x mean daily SPY excess return. That
   baseline is reported next to B1 (diagnostic only; the traded rule is unchanged).
12. "sharpe_by_regime_range" splits each logged run's daily excess returns by the base windows'
   regime date ranges (regime_ranges); it is a date split of existing runs, not a new trial.

The module is ticker-parametrised (BASE_PARAMS["ticker"]) so the pre-registered Databento
replication (IFC-F, ES sampled at 16:00 ET, ticker "ES16") can reuse the identical windows once
futures_1600.parquet is in the cache.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .. import analysis as A
from .. import calendar_utils as CU
from .. import config as C
from .. import engine as E

FAMILY = "IFC"
SLEEVE = "B"
EXEC = "next_close"
TICKERS = ["SPY"]
COST_BPS = None          # engine defaults (SPY tier 1: 3 bp one-way)

BASE_PARAMS: dict = {
    "ticker": "SPY",
    "scale": 0.10,          # target-vol multiplier: w = +/- scale / vol
    "vol_window": 63,
    "pressure_len": 5,      # pressure window [T-k-5, T-k-1]
    "liq_after": 3,         # liquidity window [T-k, T+3]
    "shift": 0,             # neighbourhood: move both windows by this many trading days
    "k_fixed": None,        # counterfactual: force k for every month (None = settlement-aware)
    "extra_delay": 0,       # robustness: extra days of execution delay
}

VARIANTS: dict[str, dict] = {
    "shift_earlier": {"shift": -1},      # pre-registered neighbourhood (G3)
    "shift_later": {"shift": 1},         # pre-registered neighbourhood (G3)
    "fixed_T3": {"k_fixed": 3},          # counterfactual for B2: old T+3 windows throughout
    "delay_1d": {"extra_delay": 1},      # robustness: one extra day of execution delay
    "cost_2x": {"cost_mult": 2.0},       # robustness: base specification at 2x costs
}
NEIGHBOURHOOD = ["shift_earlier", "shift_later"]

REGIME_ORDER = ["T+3", "T+2", "T+1"]
PROFILE_OFFSETS = list(range(-10, 6))


# --------------------------------------------------------------------------- windows

def _params(params: dict) -> dict:
    p = {**BASE_PARAMS, **{k: v for k, v in params.items() if k != "cost_mult"}}
    unknown = set(p) - set(BASE_PARAMS)
    if unknown:
        raise KeyError(f"unknown params {sorted(unknown)}")
    return p


def month_table(cal: pd.DatetimeIndex, shift: int = 0, k_fixed: int | None = None,
                pressure_len: int = 5, liq_after: int = 3) -> pd.DataFrame:
    """One row per complete month: T, its calendar index, k used, the settlement regime and the
    calendar indices of the pressure [pa, pb] and liquidity [la, lb] windows (return days)."""
    mo = CU.month_offsets(cal)
    t_idx = mo["T_i"].dropna().astype(int).unique()
    rows = []
    for ti in t_idx:
        T = cal[ti]
        k_set = CU.equity_settlement_lag(T)
        k = k_set if k_fixed is None else int(k_fixed)
        rows.append({
            "T": T, "T_i": int(ti), "k": k, "k_settle": k_set, "regime": f"T+{k_set}",
            "friday": bool(T.dayofweek == 4),
            "pa": ti - k - pressure_len + shift, "pb": ti - k - 1 + shift,
            "la": ti - k + shift, "lb": ti + liq_after + shift,
        })
    return pd.DataFrame(rows)


def realised_vol(close: pd.Series, window: int = 63) -> pd.Series:
    r = close.pct_change(fill_method=None)
    return r.rolling(window, min_periods=window).std() * math.sqrt(C.TRADING_DAYS)


def hold_weights(period: str = "IS", ohlc: dict | None = None, **params) -> pd.DataFrame:
    """Desired holdings per return day (index = trading days, one column = the ticker)."""
    p = _params(params)
    tk = p["ticker"]
    if ohlc is None:
        ohlc = E.load_ohlc([tk], period)
    close = ohlc["close"][tk]
    cal = close.index
    n = len(cal)
    vol = realised_vol(close, p["vol_window"]).to_numpy()
    mt = month_table(cal, p["shift"], p["k_fixed"], p["pressure_len"], p["liq_after"])
    hold = np.zeros(n)
    used = np.zeros(n, dtype=bool)
    for row in mt.itertuples(index=False):
        for a, b, sign in ((row.pa, row.pb, -1.0), (row.la, row.lb, +1.0)):
            d = a - 2                       # decision date for a window starting on return day a
            if d < 0 or d >= n:
                continue
            v = vol[d]
            if not np.isfinite(v) or v <= 0:
                continue
            lo, hi = max(a, 1), min(b, n - 1)
            if lo > hi:
                continue
            assert not used[lo:hi + 1].any(), f"overlapping windows around {row.T}"
            hold[lo:hi + 1] = sign * p["scale"] / v
            used[lo:hi + 1] = True
    return pd.DataFrame({tk: hold}, index=cal)


def decision_weights(period: str = "IS", ohlc: dict | None = None, **params) -> pd.DataFrame:
    """Engine decision weights (value at d uses information up to the close of d)."""
    p = _params(params)
    if ohlc is None:
        ohlc = E.load_ohlc([p["ticker"]], period)
    real = ohlc["close"].index
    # offsets on the planned calendar (unscheduled closures count as expected sessions), then
    # holdings mapped back to real sessions (calendar_utils.planned_to_realized)
    planned = {"close": CU.to_planned(ohlc["close"][[p["ticker"]]])}
    hold = CU.planned_to_realized(hold_weights(period, ohlc=planned, **p), real)
    w = CU.hold_to_decision(hold, EXEC)
    if p["extra_delay"]:
        w = w.shift(int(p["extra_delay"])).fillna(0.0)
    return w


# --------------------------------------------------------------------------- prediction tests

def _excess_returns(ohlc: dict, rf: pd.Series, ticker: str) -> pd.Series:
    close = ohlc["close"][ticker]
    return close.pct_change(fill_method=None) - rf.reindex(close.index).ffill().fillna(0.0)


def _tstat(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x))))


def _f(x) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else x


def window_spreads(period: str, ohlc: dict, rf: pd.Series, **params) -> pd.DataFrame:
    """Per complete month: pressure- and liquidity-window excess returns (sums of daily excess
    returns of the ticker) and their spread, for the windows defined by ``params``."""
    p = _params(params)
    ex = _excess_returns(ohlc, rf, p["ticker"])
    cal = ex.index
    exv = ex.to_numpy()
    n = len(cal)
    start, end = (pd.Timestamp(x) for x in E.eval_window(period))
    mt = month_table(cal, p["shift"], p["k_fixed"], p["pressure_len"], p["liq_after"])
    mt = mt[(mt["pa"] >= 1) & (mt["lb"] <= n - 1)].copy()
    mt = mt[[cal[a] >= start and cal[b] <= end for a, b in zip(mt["pa"], mt["lb"])]].copy()
    mt["r_press"] = [exv[a:b + 1].sum() for a, b in zip(mt["pa"], mt["pb"])]
    mt["r_liq"] = [exv[a:b + 1].sum() for a, b in zip(mt["la"], mt["lb"])]
    mt["len_press"] = mt["pb"] - mt["pa"] + 1
    mt["len_liq"] = mt["lb"] - mt["la"] + 1
    mt["spread"] = mt["r_liq"] - mt["r_press"]
    mt["spread_perday"] = mt["r_liq"] / mt["len_liq"] - mt["r_press"] / mt["len_press"]
    return mt.reset_index(drop=True)


def _spread_summary(df: pd.DataFrame, mean_daily_ex: float) -> dict:
    x = df["spread"].to_numpy()
    out = {
        "n_months": int(len(df)),
        "mean_spread": _f(x.mean()) if len(x) else None,
        "sd_spread": _f(x.std(ddof=1)) if len(x) > 1 else None,
        "t_spread": _f(_tstat(x)),
        "nw_t_spread": _f(E.newey_west_tstat(pd.Series(x))) if len(x) >= 10 else None,
        "hit_rate": _f((x > 0).mean()) if len(x) else None,
        "mean_r_press": _f(df["r_press"].mean()) if len(x) else None,
        "t_r_press": _f(_tstat(df["r_press"])),
        "mean_r_liq": _f(df["r_liq"].mean()) if len(x) else None,
        "t_r_liq": _f(_tstat(df["r_liq"])),
        "mean_spread_perday": _f(df["spread_perday"].mean()) if len(x) else None,
        "t_spread_perday": _f(_tstat(df["spread_perday"])),
    }
    if len(x):
        extra_days = float((df["len_liq"] - df["len_press"]).mean())
        out["net_long_days"] = extra_days
        out["equity_premium_baseline"] = _f(extra_days * mean_daily_ex)
    return out


def test_b1(spreads: pd.DataFrame, mean_daily_ex: float) -> dict:
    """B1: liquidity-minus-pressure spread per month, by settlement regime (and pooled)."""
    out = {"all": _spread_summary(spreads, mean_daily_ex)}
    for reg in REGIME_ORDER:
        sub = spreads[spreads["regime"] == reg]
        out[reg] = _spread_summary(sub, mean_daily_ex)
    out["T+1"]["note"] = "only ~4 complete in-sample months: reported, not interpreted"
    return out


def event_profile(period: str, ohlc: dict, rf: pd.Series, ticker: str = "SPY",
                  offsets: list[int] = PROFILE_OFFSETS) -> pd.DataFrame:
    """Rows = months with all offsets inside the evaluation period; columns = offsets relative
    to T (excess return of that return day), plus T and regime."""
    ex = _excess_returns(ohlc, rf, ticker)
    cal = ex.index
    exv = ex.to_numpy()
    n = len(cal)
    start, end = (pd.Timestamp(x) for x in E.eval_window(period))
    mt = month_table(cal)
    lo, hi = min(offsets), max(offsets)
    rows = []
    for row in mt.itertuples(index=False):
        a, b = row.T_i + lo, row.T_i + hi
        if a < 1 or b > n - 1 or cal[a] < start or cal[b] > end:
            continue
        rec = {o: exv[row.T_i + o] for o in offsets}
        rec.update({"T": row.T, "regime": row.regime})
        rows.append(rec)
    return pd.DataFrame(rows)


def _profile_stats(prof: pd.DataFrame, regime: str, offsets: list[int]) -> pd.DataFrame:
    sub = prof[prof["regime"] == regime][offsets].astype(float)
    m = sub.mean()
    se = sub.std(ddof=1) / math.sqrt(len(sub))
    return pd.DataFrame({"mean": m, "se": se, "t": m / se, "cum_mean": m.cumsum(), "n": len(sub)})


def test_b2(period: str, ohlc: dict, rf: pd.Series, res_base: E.BacktestResult,
            res_fixed: E.BacktestResult, mean_daily_ex: float) -> tuple[dict, pd.DataFrame]:
    """B2: in the T+2 regime, settlement-aware (k=2) windows vs the old T+3 windows."""
    tk = BASE_PARAMS["ticker"]
    sp_aware = window_spreads(period, ohlc, rf)
    sp_old = window_spreads(period, ohlc, rf, k_fixed=3)
    a = sp_aware[sp_aware["regime"] == "T+2"].set_index("T")
    b = sp_old[sp_old["regime"] == "T+2"].set_index("T")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    diff = a["spread"] - b["spread"]
    out = {
        "n_months": int(len(common)),
        "aware_k2": _spread_summary(a.reset_index(), mean_daily_ex),
        "old_k3": _spread_summary(b.reset_index(), mean_daily_ex),
        "paired_diff_mean": _f(diff.mean()) if len(diff) else None,
        "paired_diff_t": _f(_tstat(diff)),
        "paired_diff_hit_rate": _f((diff > 0).mean()) if len(diff) else None,
    }
    # date range in which both backtests trade only T+2-regime windows
    cal = ohlc["close"].index
    mt = month_table(cal)
    last_t3 = mt[mt["regime"] == "T+3"]
    last_t2 = mt[mt["regime"] == "T+2"]
    if len(last_t3) and len(last_t2):
        d0 = cal[int(last_t3["lb"].iloc[-1]) + 1]
        d1 = cal[min(int(last_t2["lb"].iloc[-1]), len(cal) - 1)]
        rng = {}
        for name, res in (("aware_k2", res_base), ("old_k3", res_fixed)):
            ex = res.excess.loc[d0:d1]
            rng[name] = {
                "sharpe": _f(ex.mean() / ex.std() * math.sqrt(C.TRADING_DAYS)) if ex.std() > 0 else None,
                "ann_excess_return": _f(ex.mean() * C.TRADING_DAYS),
                "n_days": int(len(ex)),
            }
        d = res_base.excess.loc[d0:d1] - res_fixed.excess.loc[d0:d1]
        rng["diff_series_nw_t"] = _f(E.newey_west_tstat(d))
        out["daily_backtest_T2_range"] = {"start": str(d0.date()), "end": str(d1.date()), **rng}
    # event-time profile T-10..T+5 by regime
    prof = event_profile(period, ohlc, rf, tk)
    profiles = {}
    for reg in ("T+3", "T+2"):
        ps = _profile_stats(prof, reg, PROFILE_OFFSETS)
        cum = ps["cum_mean"]
        pre = cum.loc[:0]                        # trough of the cumulative profile before/at T
        profiles[reg] = {
            "n_months": int(ps["n"].iloc[0]),
            "mean_by_offset": {int(o): _f(v) for o, v in ps["mean"].items()},
            "t_by_offset": {int(o): _f(v) for o, v in ps["t"].items()},
            "cum_trough_offset_up_to_T": int(pre.idxmin()),
            "predicted_trough_offset": -int(reg[-1]) - 1,   # last pressure day T-k-1
        }
    out["event_profile"] = profiles
    # the one offset whose predicted sign flips between regimes: T-3 (first liquidity day under
    # T+3, last pressure day under T+2); also T-8 (first pressure day under T+3, outside under T+2)
    flips = {}
    for o in (-3, -8):
        x3 = prof.loc[prof["regime"] == "T+3", o].astype(float)
        x2 = prof.loc[prof["regime"] == "T+2", o].astype(float)
        w = stats.ttest_ind(x3, x2, equal_var=False)
        flips[f"T{o}"] = {"mean_T+3": _f(x3.mean()), "mean_T+2": _f(x2.mean()),
                          "diff_T3_minus_T2": _f(x3.mean() - x2.mean()), "welch_t": _f(w.statistic),
                          "p": _f(w.pvalue)}
    out["boundary_offsets"] = flips
    return out, prof


def test_b3(spreads: pd.DataFrame) -> dict:
    """B3: spread when T is a Friday vs not (all complete months, settlement-aware windows)."""
    fri = spreads.loc[spreads["friday"], "spread"].astype(float)
    oth = spreads.loc[~spreads["friday"], "spread"].astype(float)
    w = stats.ttest_ind(fri, oth, equal_var=False)
    out = {
        "n_friday": int(len(fri)), "n_other": int(len(oth)),
        "mean_friday": _f(fri.mean()), "mean_other": _f(oth.mean()),
        "t_friday": _f(_tstat(fri)), "t_other": _f(_tstat(oth)),
        "diff": _f(fri.mean() - oth.mean()), "welch_t": _f(w.statistic), "p": _f(w.pvalue),
        "by_regime": {},
    }
    for reg in REGIME_ORDER:
        s = spreads[spreads["regime"] == reg]
        f_, o_ = s.loc[s["friday"], "spread"], s.loc[~s["friday"], "spread"]
        out["by_regime"][reg] = {"n_friday": int(len(f_)), "n_other": int(len(o_)),
                                 "diff": _f(f_.mean() - o_.mean()) if len(f_) and len(o_) else None}
    return out


# --------------------------------------------------------------------------- run

def _stats_js(s: dict) -> dict:
    return {k: (_f(v) if isinstance(v, (float, np.floating)) else v) for k, v in s.items()}


def regime_ranges(cal: pd.DatetimeIndex) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Date range of each settlement regime for the base windows: from the day after the previous
    regime's last liquidity window ends (or the first date) to the end of the regime's own last
    liquidity window (or the last date). Inside a range the base trades only that regime's months."""
    mt = month_table(cal)
    present = [reg for reg in REGIME_ORDER if (mt["regime"] == reg).any()]
    out, start_i = {}, 0
    for j, reg in enumerate(present):
        last_lb = int(mt.loc[mt["regime"] == reg, "lb"].iloc[-1])
        end_i = len(cal) - 1 if j == len(present) - 1 else min(last_lb, len(cal) - 1)
        out[reg] = (cal[start_i], cal[end_i])
        start_i = end_i + 1
    return out


def run(period: str = "IS", log: bool = True, return_results: bool = False):
    """Run the base specification and every variant; return a JSON-serialisable summary
    (and, with ``return_results=True``, also the BacktestResults and test tables)."""
    tk = BASE_PARAMS["ticker"]
    ohlc = E.load_ohlc([tk], period)
    rf = E.load_rf(period)

    def bt(name: str, overrides: dict) -> E.BacktestResult:
        cost_mult = float(overrides.get("cost_mult", 1.0))
        p = _params(overrides)
        w = decision_weights(period, ohlc=ohlc, **p)
        params = {"sleeve": SLEEVE, "variant": name, **p}
        return E.run_backtest(f"IFC_B_{name}", FAMILY, w, ohlc, rf, period=period, params=params,
                              exec=EXEC, cost_mult=cost_mult, log=log)

    results = {"base": bt("base", {})}
    for vname, ov in VARIANTS.items():
        results[vname] = bt(vname, ov)
    base = results["base"]

    ex_all = _excess_returns(ohlc, rf, tk)
    start, end = E.eval_window(period)
    mean_daily_ex = float(ex_all.loc[start:end].mean())

    spreads = window_spreads(period, ohlc, rf)
    b1 = test_b1(spreads, mean_daily_ex)
    b2, prof = test_b2(period, ohlc, rf, base, results["fixed_T3"], mean_daily_ex)
    b3 = test_b3(spreads)

    lo, hi = A.bootstrap_sharpe_ci(base.excess)
    nb_sr = [results[v].stats["sharpe"] for v in NEIGHBOURHOOD]

    # decomposition of the base P&L (no new trial): gross excess P&L of the long (liquidity) and
    # short (pressure) legs, and the cost + borrow drag, annualised
    held = base.weights[tk]
    exb = ex_all.reindex(held.index).fillna(0.0)
    yrs = len(held) / C.TRADING_DAYS
    legs = {
        "long_leg_gross_excess_ann": _f((held.clip(lower=0) * exb).sum() / yrs),
        "short_leg_gross_excess_ann": _f((held.clip(upper=0) * exb).sum() / yrs),
        "costs_and_borrow_ann": _f(base.costs.sum() / yrs),
        "mean_abs_weight_when_held": _f(held[held != 0].abs().mean()),
    }
    # net Sharpe of every logged run inside each regime's date range (no new trial: the logged
    # return series are only split by date; ranges are those of the base windows)
    by_regime_range = {}
    for reg, (d0, d1) in regime_ranges(ohlc["close"].index).items():
        rec = {"start": str(d0.date()), "end": str(d1.date())}
        for v, r in results.items():
            ex = r.excess.loc[d0:d1]
            rec[v] = _f(ex.mean() / ex.std() * math.sqrt(C.TRADING_DAYS)) if ex.std() > 0 else None
        by_regime_range[reg] = rec
    nb_med = float(np.median(nb_sr))
    gates = {   # selection gates are defined for the composite candidate; shown for this sleeve only
        "G1_sharpe_2x_costs_gt_0": bool(results["cost_2x"].stats["sharpe"] > 0),
        "G2_best_year_share_le_40pct": bool(A.pnl_concentration(base)["best_year_share"] <= 0.40),
        "G3_neighbourhood_median_ge_half_base": bool(nb_med >= 0.5 * base.stats["sharpe"]),
        "neighbourhood_median_sharpe": nb_med,
    }
    summary = {
        "module": "ifc_dash", "family": FAMILY, "sleeve": SLEEVE, "period": period, "exec": EXEC,
        "base_params": BASE_PARAMS, "variants_spec": VARIANTS,
        "base": _stats_js(base.stats),
        "base_cost_2x": _stats_js(results["cost_2x"].stats),
        "variants": {v: {"sharpe": _f(r.stats["sharpe"]), "ann_return": _f(r.stats["ann_return"]),
                         "ann_vol": _f(r.stats["ann_vol"]), "max_drawdown": _f(r.stats["max_drawdown"]),
                         "turnover_per_year": _f(r.stats["turnover_per_year"])}
                     for v, r in results.items() if v != "base"},
        "sharpe_by_regime_range": by_regime_range,
        "neighbourhood_median_sharpe_over_base": _f(np.median(nb_sr) / base.stats["sharpe"])
        if base.stats["sharpe"] != 0 else None,
        "subperiods": A.subperiod_table(base).to_dict(orient="records"),
        "pnl_concentration": _stats_js(A.pnl_concentration(base)),
        "bootstrap_sharpe_ci90": [lo, hi],
        "yearly_returns": {int(k): _f(v) for k, v in E.yearly_returns(base.returns).items()},
        "yearly_excess_sum": {int(k): _f(v) for k, v in base.excess.groupby(base.excess.index.year).sum().items()},
        "mean_daily_excess_spy": mean_daily_ex,
        "pnl_by_leg": legs,
        "sleeve_gates_informational": gates,
        "B1": b1, "B2": b2, "B3": b3,
        "kill_condition_spread_positive": bool((b1["all"]["mean_spread"] or 0) > 0),
    }
    try:
        summary["factor_table"] = A.factor_table(base)
    except Exception as exc:          # factor data missing: report rather than fail the run
        summary["factor_table"] = {"error": repr(exc)}
    if return_results:
        return summary, {"results": results, "spreads": spreads, "profile": prof}
    return summary
