"""IFC exploratory sleeve D: FX hedge rebalancing (Melvin & Prins 2015). Reported, never in the composite.

Pre-registered rule (HYPOTHESES.md section 1, "Exploratory sleeves"):
    Signal = SPY month-to-date return minus the local-currency month-to-date return of the foreign
    index (Euro Stoxx 50 for FXE, Nikkei 225 for FXY), observed at T-3. Hold the currency ETF in the
    signal's direction over return days [T-1, T] at 0.05 / vol.

Economic reading. Investors who hold foreign equities with a currency hedge re-size the hedge at the
month-end fix. When US equities have outperformed the foreign market month to date, foreign holders of
US equities must sell more USD forward and US holders of foreign equities buy back part of their short
foreign-currency hedge, so the foreign currency should rise into the month end: a positive signal means
long the currency ETF (FXE / FXY are USD prices of EUR / JPY deposits, so they rise when EUR / JPY
appreciate against the dollar).

Interpretation decisions (fixed before the first run; no alternatives were tried):
  * Month-to-date returns run from the close of the previous month's T to the close of T-3, both on the
    NYSE calendar (calendar_utils.month_offsets). SPY uses the engine's total-return-adjusted close
    (engine.load_ohlc); the foreign indices are price-index levels (Yahoo ^STOXX50E, ^N225, no
    dividends), the only local-currency series available for free. The small dividend mismatch is
    disclosed, not corrected.
  * Local index levels come from GQH_DATA_DIR/local_indices.parquet (data/download.py
    download_local_indices). They are read with engine._read_parquet because the generic loaders only
    know etf_daily, and are TRUNCATED HERE to dates <= engine.data_end(period) (2024-10-02 in-sample),
    so this module obeys the same OOS lock as the engine loaders.
  * Foreign dates are aligned to the NYSE calendar with the last available foreign close on or before
    each NYSE date (Tokyo and the Euro Stoxx close both happen before the New York close on the same
    date, so the value at NYSE date d is known at the close of d: no lookahead). The staleness of the
    aligned value at every date used is recorded in the summary.
  * Size: w = sign(signal) * 0.05 / vol, where vol is the 63-day realised volatility (std of daily
    close-to-close adjusted returns, annualised with sqrt(252), at least 63 returns) of that currency
    ETF, observed at the same close of T-3 as the signal (IFC convention "63-day realised volatility
    observed at the decision date"). No cap is applied (none is pre-registered).
  * One position per month and currency: the signal and vol observed at T-3 size both hold rows T-1
    and T, i.e. the ETF is bought at the close of T-2 and sold at the close of T (next_close: the hold
    row t may use data up to the close of t-2; T-3 <= t-2 for both rows, asserted in code).
  * Both currencies are traded at the same time, each at the full 0.05 / vol (the sleeve is not divided
    by the number of currencies). A currency is active only once its ETF, its index and its 63-day vol
    exist; Yahoo's ^STOXX50E history starts 2007-03-30, so FXE starts April 2007 and FXY (listed
    2007-02-13) once 63 returns exist.
  * A zero or missing signal or vol means no position in that currency that month.
  * Robustness variants (the only ones for this module): 2x costs (same weights, cost_mult=2, logged under
    the base name) and one extra day of execution delay (decision weights shifted one more trading day,
    so the position covers return days [T, T+1]).
  * Prediction test (per currency and pooled): for every month with a position, the ETF's raw return over
    the hold window, close(T-2) -> close(T). Sign hit rate = share of months with sign(window return) ==
    sign(signal) among months with a non-zero window return (two-sided binomial test vs 0.5); mean
    signed return = mean of sign(signal) * window return per unit notional (one-sample t-test), also
    shown after a 10 bp round trip (2 x the 5 bp tier-2 one-way cost), plus mean window returns after
    positive and negative signals and the Spearman correlation of signal and window return.
  * Yearly returns use engine.yearly_returns (analysis.py has no yearly_returns function).
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

FAMILY = "IFC_EXPLORATORY"
EXEC = "next_close"
TICKERS = ["FXE", "FXY"]
COST_BPS = None                      # engine defaults: FXE, FXY are tier-2 (5 bp one-way)
FOREIGN = {"FXE": "^STOXX50E", "FXY": "^N225"}
LOCAL_FILE = "local_indices.parquet"
NAME = "D_fx_hedge"

BASE_PARAMS = {
    "sleeve": "D_fx_hedge_rebalancing",
    "signal_offset": -3,     # signal and vol observed at the close of T-3
    "hold_start": -1,        # hold over return days [T-1, T]
    "hold_end": 0,
    "target_vol": 0.05,      # w = sign(signal) * target_vol / vol
    "vol_window": 63,
    "extra_lag": 0,          # extra trading days of execution delay
}
VARIANTS = {
    "cost2x": {"cost_mult": 2.0},
    "extra_lag": {"extra_lag": 1},
}


# --------------------------------------------------------------------------- data

def load_local_indices(period: str = "IS", cal: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Local-currency index closes aligned to the NYSE calendar, plus the staleness (calendar days)
    of each aligned value. Truncated to engine.data_end(period) before anything else."""
    df = E._read_parquet(LOCAL_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp(E.data_end(period))]
    cal = E.trading_calendar(period) if cal is None else cal
    levels, stale = {}, {}
    for etf, idx in FOREIGN.items():
        s = df.loc[df["ticker"] == idx, ["date", "close"]].dropna().drop_duplicates("date", keep="last")
        s = s[s["close"] > 0].set_index("date")["close"].sort_index()
        s.index = pd.DatetimeIndex(s.index).as_unit("ns")
        cal_ns = pd.DatetimeIndex(cal).as_unit("ns")
        levels[idx] = s.reindex(cal_ns, method="ffill")
        last_date = pd.Series(s.index, index=s.index).reindex(cal_ns, method="ffill")
        stale[idx] = (cal_ns.to_series() - last_date).dt.days
    lv = pd.DataFrame(levels)
    st = pd.DataFrame(stale)
    lv.index = st.index = pd.DatetimeIndex(cal)
    return lv, st


def _mtd(close: pd.DataFrame, mo: pd.DataFrame) -> pd.DataFrame:
    """Return from the previous month's T close to each day's close."""
    prev_i = mo["prevT_i"].to_numpy()
    ok = prev_i >= 0
    arr = close.to_numpy(dtype=float)
    ref = np.full_like(arr, np.nan)
    ref[ok] = arr[prev_i[ok].astype(int)]
    return pd.DataFrame(arr / ref - 1.0, index=close.index, columns=close.columns)


def monthly_table(period: str = "IS", **params) -> pd.DataFrame:
    """One row per (month, currency) with the T-3 inputs and the realised window return (the latter is
    for the prediction test only and is never used in the weights)."""
    p = {**BASE_PARAMS, **params}
    p.pop("cost_mult", None)
    # planned calendar: unscheduled closures count as expected sessions (calendar_utils)
    cal = CU.planned_calendar(E.trading_calendar(period))
    close = CU.to_planned(E.load_ohlc(["SPY"] + TICKERS, period)["close"])
    lv, stale = load_local_indices(period, cal)
    mo = CU.month_offsets(cal)
    mtd_etf = _mtd(close[["SPY"]], mo)["SPY"]
    mtd_loc = _mtd(lv, mo)
    rets = close[TICKERS].pct_change(fill_method=None)
    vol = rets.rolling(p["vol_window"], min_periods=p["vol_window"]).std() * math.sqrt(C.TRADING_DAYS)

    sig_rows = np.flatnonzero(mo["off_own"].to_numpy() == p["signal_offset"])
    rows = []
    for i in sig_rows:
        T_i = int(mo["T_i"].iloc[i])
        a_i, b_i = T_i + p["hold_start"], T_i + p["hold_end"]
        assert i <= a_i - 2, "signal must be observed by the close of a-2 (next_close)"
        prev_i = int(mo["prevT_i"].iloc[i])
        for etf, idx in FOREIGN.items():
            c = close[etf]
            win = c.iloc[b_i] / c.iloc[a_i - 1] - 1 if b_i < len(cal) else np.nan
            rows.append({
                "T": cal[T_i], "signal_date": cal[i], "ticker": etf, "index": idx,
                "spy_mtd": mtd_etf.iloc[i], "local_mtd": mtd_loc[idx].iloc[i],
                "signal": mtd_etf.iloc[i] - mtd_loc[idx].iloc[i],
                "vol": vol[etf].iloc[i],
                "stale_days_signal": stale[idx].iloc[i],
                "stale_days_ref": stale[idx].iloc[prev_i] if prev_i >= 0 else np.nan,
                "window_ret": win,
            })
    tab = pd.DataFrame(rows)
    ok = tab["signal"].notna() & tab["vol"].notna() & (tab["vol"] > 0) & (tab["signal"] != 0)
    tab["active"] = ok
    tab["weight"] = np.where(ok, np.sign(tab["signal"]) * p["target_vol"] / tab["vol"], 0.0)
    return tab


# --------------------------------------------------------------------------- weights

def decision_weights(period: str = "IS", **params) -> pd.DataFrame:
    """Engine DECISION weights (next_close) for FXE and FXY."""
    p = {**BASE_PARAMS, **params}
    p.pop("cost_mult", None)
    real = E.trading_calendar(period)
    cal = CU.planned_calendar(real)
    tab = monthly_table(period, **p)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    hold = pd.DataFrame(0.0, index=cal, columns=TICKERS)
    for _, r in tab[tab["active"]].iterrows():
        T_i = pos[r["T"]]
        for t_i in range(T_i + p["hold_start"], T_i + p["hold_end"] + 1):
            assert pos[r["signal_date"]] <= t_i - 2
            hold.iloc[t_i, TICKERS.index(r["ticker"])] = r["weight"]
    hold = CU.planned_to_realized(hold, real)
    w = CU.hold_to_decision(hold, EXEC)
    if p["extra_lag"]:
        w = w.shift(int(p["extra_lag"])).fillna(0.0)
    return w


# --------------------------------------------------------------------------- prediction tests

def prediction_tests(tab: pd.DataFrame) -> dict:
    out = {}
    act = tab[tab["active"] & tab["window_ret"].notna()]
    groups = {t: act[act["ticker"] == t] for t in TICKERS}
    groups["pooled"] = act
    rt_cost = 2 * C.cost_bps("FXE") / 1e4
    for name, g in groups.items():
        sgn = np.sign(g["signal"])
        signed = sgn * g["window_ret"]
        nz = g["window_ret"] != 0
        hits = int((signed[nz] > 0).sum())
        n_nz = int(nz.sum())
        tt = stats.ttest_1samp(signed, 0.0)
        pos_, neg_ = g.loc[sgn > 0, "window_ret"], g.loc[sgn < 0, "window_ret"]
        rho, rho_p = stats.spearmanr(g["signal"], g["window_ret"])
        out[name] = {
            "n_months": int(len(g)),
            "first_T": str(g["T"].min().date()),
            "n_zero_window_ret": int((~nz).sum()),
            "hit_rate": hits / n_nz if n_nz else float("nan"),
            "hit_binom_p_two_sided": float(stats.binomtest(hits, n_nz, 0.5).pvalue) if n_nz else float("nan"),
            "mean_signed_ret_bp": float(signed.mean() * 1e4),
            "mean_signed_ret_t": float(tt.statistic),
            "mean_signed_ret_p": float(tt.pvalue),
            "mean_signed_ret_after_10bp_rt_bp": float((signed.mean() - rt_cost) * 1e4),
            "mean_window_ret_bp_unconditional": float(g["window_ret"].mean() * 1e4),
            "mean_window_ret_bp_signal_pos": float(pos_.mean() * 1e4),
            "n_signal_pos": int(len(pos_)),
            "mean_window_ret_bp_signal_neg": float(neg_.mean() * 1e4),
            "n_signal_neg": int(len(neg_)),
            "spearman_signal_vs_ret": float(rho),
            "spearman_p": float(rho_p),
        }
    return out


# --------------------------------------------------------------------------- run

def _report(res: E.BacktestResult) -> dict:
    lo, hi = A.bootstrap_sharpe_ci(res.excess)
    return {
        "stats": res.stats,
        "subperiods": A.subperiod_table(res).to_dict(orient="records"),
        "pnl_concentration": A.pnl_concentration(res),
        "factor_table": A.factor_table(res),
        "bootstrap_sharpe_ci90": [lo, hi],
        "yearly_returns": {int(k): float(v) for k, v in E.yearly_returns(res.returns).items()},
        "active_days_frac": float((res.weights.abs().sum(axis=1) > 0).mean()),
        "mean_gross_exposure_when_active": float(
            res.weights.abs().sum(axis=1)[res.weights.abs().sum(axis=1) > 0].mean()),
    }


def run(period: str = "IS", log: bool = True) -> dict:
    """Base + variants. Keys starting with '_' hold in-memory pandas objects (not JSON)."""
    ohlc = E.load_ohlc(TICKERS, period)
    rf = E.load_rf(period)
    tab = monthly_table(period)
    w_base = decision_weights(period)
    results = {"base": E.run_backtest(f"{NAME}_base", FAMILY, w_base, ohlc, rf, period=period,
                                      params=dict(BASE_PARAMS), exec=EXEC, log=log, note="sleeve D base")}
    for vname, ov in VARIANTS.items():
        p = {**BASE_PARAMS, **ov}
        cm = p.pop("cost_mult", 1.0)
        w = decision_weights(period, **p)
        run_name = f"{NAME}_base" if vname == "cost2x" else f"{NAME}_{vname}"
        results[vname] = E.run_backtest(run_name, FAMILY, w, ohlc, rf, period=period, params=p, exec=EXEC,
                                        cost_mult=cm, log=log, note=f"sleeve D variant {vname}")
    base = results["base"]
    summary = {
        "module": "ifc_fx",
        "family": FAMILY,
        "exec": EXEC,
        "tickers": TICKERS,
        "foreign_indices": FOREIGN,
        "period": period,
        "base_params": BASE_PARAMS,
        "variants": VARIANTS,
        "base": _report(base),
        "base_2x_costs": {"stats": results["cost2x"].stats},
        "variant_sharpe": {k: r.stats["sharpe"] for k, r in results.items()},
        "variant_stats": {k: r.stats for k, r in results.items() if k != "base"},
        "prediction_tests": prediction_tests(tab),
        "data_notes": {
            "active_months": {t: int(tab[(tab["ticker"] == t) & tab["active"]].shape[0]) for t in TICKERS},
            "first_active_T": {t: str(tab[(tab["ticker"] == t) & tab["active"]]["T"].min().date()) for t in TICKERS},
            "max_stale_days_at_signal": {t: float(tab.loc[(tab["ticker"] == t) & tab["active"], "stale_days_signal"].max()) for t in TICKERS},
            "max_stale_days_at_reference": {t: float(tab.loc[(tab["ticker"] == t) & tab["active"], "stale_days_ref"].max()) for t in TICKERS},
            "weight_range": {t: [float(tab.loc[(tab["ticker"] == t) & tab["active"], "weight"].abs().min()),
                                 float(tab.loc[(tab["ticker"] == t) & tab["active"], "weight"].abs().max())] for t in TICKERS},
            "share_signal_positive": {t: float((tab.loc[(tab["ticker"] == t) & tab["active"], "signal"] > 0).mean()) for t in TICKERS},
        },
        "sanity_sharpe_above_3": bool(any(abs(r.stats["sharpe"]) > 3 for r in results.values())),
    }
    summary["_returns"] = pd.DataFrame({k: r.returns for k, r in results.items()})
    summary["_results"] = results
    summary["_monthly"] = tab
    return summary
