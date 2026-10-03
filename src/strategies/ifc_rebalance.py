"""IFC sleeve A: stock/bond month-end rebalancing pressure (HYPOTHESES.md section 1, Sleeve A).

Rule as pre-registered
----------------------
Shadow portfolio 60 % SPY / 40 % IEF, reset to 60/40 at the close of every month-end ``T``.
Drift ``D_d`` = shadow equity weight at the close of d minus 0.60. Observe ``D`` at the close of
``T-6``; ``z = D / s`` with ``s`` the root-mean-square of all earlier months' ``D`` at ``T-6``
(expanding, >= 12 months, sleeve inactive before); ``q = clip(z, -2, 2)``. Hold over return days
[``T-4``, ``T``]: ``w_SPY = -q * 0.05 / vol_SPY``, ``w_IEF = +q * 0.05 / vol_IEF``; ``next_close``.

Interpretation decisions (fixed before the first backtest of this module; not tuned)
-----------------------------------------------------------------------------------
1. Calendar: ``T-j`` / ``T+j`` are taken as integer offsets on the NYSE (SPY) trading calendar from
   the month's last trading day (``calendar_utils.month_offsets``). The incomplete final month of a
   period (October 2024 in-sample) has no ``T`` and is skipped.
2. Shadow drift: with ``G_x = close_x(d) / close_x(previous T)`` on total-return-adjusted closes,
   ``D_d = 0.6 G_SPY / (0.6 G_SPY + 0.4 G_IEF) - 0.6``. January 2005 has no previous ``T`` in the
   data (history starts 2005-01-03), so it has no ``D`` and is not part of the RMS history.
3. ``s`` for month m uses the ``D`` values of strictly earlier months only (all of them, including
   the warm-up months); the sleeve trades from the 13th month that has a ``D`` (Feb 2006 window).
4. Volatility: 63-day realised volatility of daily close-to-close returns (sample std x sqrt(252)),
   window ending at the close of the observation day (the decision date), full 63 days required.
   The leg weights are computed once per month at the decision date and held constant (the engine
   rebalances to them daily) over the whole window.
5. Variants (HYPOTHESES.md section 3): the observation day always moves with the window start,
   ``obs = a - 2`` for window [``T+a``, ``T+b``] (the protocol's "observed at the close of a-2"):
   shift_early = window [T-5, T-1], obs T-7; shift_late = [T-3, T+1], obs T-5;
   win4 = [T-3, T], obs T-5; win6 = [T-5, T], obs T-7. The RMS ``s`` is built from earlier months'
   ``D`` at the variant's own observation day, and the 63-day vol is observed on that day too.
   ``delay1`` shifts the base decision weights by one more trading day (robustness, not part of
   the G3 neighbourhood).
6. Threshold-rebalancing variant (robustness): for each band b in {0.5, 1, 1.5, 2} % a separate
   shadow 60/40 portfolio starts at the first close of the data (2005-01-03) and drifts with
   close-to-close returns; at any close t where ``|D_t| >= b`` it is reset to 60/40 (a "trigger").
   For each trigger, ``q = clip(D_t / s_b, -2, 2)`` where ``s_b`` is the RMS of that band's
   earlier trigger drifts (>= 12 earlier triggers, inactive before); the position
   ``w_SPY = -q 0.05 / vol_SPY(t)``, ``w_IEF = +q 0.05 / vol_IEF(t)`` is held over return days
   [t+2, t+6] (decided at close t, established at close t+1 under ``next_close``). Overlapping
   trigger positions within a band ADD (each trigger is a separate flow event); the variant's
   weight is the simple average of the four band positions (an inactive band contributes 0).
7. Prediction test: per month, window spread = compound SPY return minus compound IEF return over
   the hold window (close of a-1 to close of b). Primary test = OLS slope of the spread on ``D`` at
   the observation day over ALL months that have ``D`` and whose window lies in the evaluation
   period (heteroskedasticity-robust HC1 t-stat; windows do not overlap) plus a one-sided sign test
   of P(sign(spread) = -sign(D)) > 0.5. The same is reported on traded (active) months with ``z``.
8. Correlation regime split (diagnostic): trailing 252-day correlation of SPY and IEF daily
   returns ending at the observation day (full 252 days required); > 0 vs <= 0. Reported per
   regime: per-window net excess P&L of the base backtest (return days a .. b+1, so the exit cost
   is included), its t-stat and hit rate, and the spread-on-D slope.
9. ``equity`` / ``bond`` tickers are parameters only so the same code can be reused for the
   pre-registered Databento replication (ES16 / ZN16, HYPOTHESES.md section 4); the base is
   SPY / IEF and nothing here was run on futures.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import stats

from .. import analysis as A
from .. import calendar_utils as CU
from .. import config as C
from .. import engine as E

FAMILY = "IFC"
SLEEVE = "A"
EXEC = "next_close"
TICKERS = ["SPY", "IEF"]
COST_BPS = None          # engine defaults: SPY and IEF are tier-1 (3 bp one-way)

BASE_PARAMS: dict = {
    "sleeve": "A",
    "mode": "monthly",             # "monthly" (pre-registered rule) or "threshold" (robustness)
    "equity": "SPY",
    "bond": "IEF",
    "w_equity": 0.60,
    "obs": -6,                     # observation day relative to T
    "window": [-4, 0],             # hold over return days [T+a, T+b]
    "clip": 2.0,
    "min_hist": 12,
    "target_vol": 0.05,
    "vol_window": 63,
    "bands": None,
    "hold": None,                  # threshold mode: hold over [t+hold[0], t+hold[1]]
    "extra_delay": 0,
}

VARIANTS: dict[str, dict] = {
    # pre-registered neighbourhood (section 3, sleeve A part)
    "shift_early": {"obs": -7, "window": [-5, -1]},
    "shift_late": {"obs": -5, "window": [-3, 1]},
    "win4": {"obs": -5, "window": [-3, 0]},
    "win6": {"obs": -7, "window": [-5, 0]},
    # robustness runs (section 3)
    "threshold": {"mode": "threshold", "bands": [0.005, 0.01, 0.015, 0.02], "hold": [2, 6]},
    "delay1": {"extra_delay": 1},
}
NEIGHBOURHOOD = ["shift_early", "shift_late", "win4", "win6"]
ROBUSTNESS = ["threshold", "delay1"]


# --------------------------------------------------------------------------- helpers

def _params(**overrides) -> dict:
    p = json.loads(json.dumps(BASE_PARAMS))
    for k, v in overrides.items():
        if k not in p:
            raise KeyError(f"unknown parameter {k}")
        p[k] = v
    return p


def _closes(period: str, p: dict) -> pd.DataFrame:
    ohlc = E.load_ohlc([p["equity"], p["bond"]], period)
    return ohlc["close"][[p["equity"], p["bond"]]].ffill(limit=5)


def _ann_vol(close: pd.DataFrame, window: int) -> pd.DataFrame:
    r = close.pct_change(fill_method=None)
    return r.rolling(window, min_periods=window).std() * math.sqrt(C.TRADING_DAYS)


def month_table(close: pd.DataFrame, p: dict) -> pd.DataFrame:
    """One row per month with a known T: drift at the observation day, its standardisation, the
    leg weights, and the realised window returns (for the prediction tests)."""
    eq, bd = p["equity"], p["bond"]
    obs, (a, b) = int(p["obs"]), (int(p["window"][0]), int(p["window"][1]))
    if obs > a - 2:
        raise ValueError(f"lookahead: observation T{obs:+d} is later than window start T{a:+d} minus 2")
    cal = close.index
    n = len(cal)
    mo = CU.month_offsets(cal)
    first_idx = pd.Series(np.arange(n), index=cal).groupby(mo["month"].values).min()
    vol = _ann_vol(close, int(p["vol_window"]))
    r = close.pct_change(fill_method=None)
    corr = r[eq].rolling(252, min_periods=252).corr(r[bd])
    ce, cb = close[eq].to_numpy(), close[bd].to_numpy()
    rows = []
    for month, T_i in mo.groupby("month")["T_i"].first().items():
        if pd.isna(T_i):
            continue
        T_i = int(T_i)
        prevT = int(first_idx[month]) - 1
        o, ia, ib = T_i + obs, T_i + a, T_i + b
        if prevT < 0 or o <= prevT or ib >= n or ia - 1 < 0:
            continue
        ge, gb = ce[o] / ce[prevT], cb[o] / cb[prevT]
        w = p["w_equity"]
        D = w * ge / (w * ge + (1 - w) * gb) - w
        rows.append({
            "T": cal[T_i], "obs_date": cal[o], "start": cal[ia], "end": cal[ib],
            "i_obs": o, "i_a": ia, "i_b": ib, "D": D,
            "vol_eq": vol[eq].iloc[o], "vol_bd": vol[bd].iloc[o], "corr252": corr.iloc[o],
            "ret_eq": ce[ib] / ce[ia - 1] - 1, "ret_bd": cb[ib] / cb[ia - 1] - 1,
        })
    mt = pd.DataFrame(rows).set_index("T")
    mt = mt[mt["D"].notna()]
    d2 = mt["D"] ** 2
    mt["n_hist"] = np.arange(len(mt))                          # earlier months with a D
    mt["s"] = np.sqrt(d2.cumsum().shift(1) / mt["n_hist"].replace(0, np.nan))
    mt["active"] = (mt["n_hist"] >= int(p["min_hist"])) & mt["vol_eq"].notna() & mt["vol_bd"].notna()
    mt["z"] = mt["D"] / mt["s"]
    mt["q"] = mt["z"].clip(-p["clip"], p["clip"]).where(mt["active"], 0.0)
    mt["w_eq"] = (-mt["q"] * p["target_vol"] / mt["vol_eq"]).where(mt["active"], 0.0)
    mt["w_bd"] = (mt["q"] * p["target_vol"] / mt["vol_bd"]).where(mt["active"], 0.0)
    mt["spread"] = mt["ret_eq"] - mt["ret_bd"]
    return mt


def threshold_events(close: pd.DataFrame, p: dict, band: float) -> pd.DataFrame:
    """Reset events of a 60/40 shadow portfolio rebalanced only when |drift| >= band."""
    eq, bd = p["equity"], p["bond"]
    r = close.pct_change(fill_method=None).fillna(0.0)
    re, rb = r[eq].to_numpy(), r[bd].to_numpy()
    w0 = p["w_equity"]
    we = w0
    ev = []
    for i in range(1, len(close)):
        ve, vb = we * (1 + re[i]), (1 - we) * (1 + rb[i])
        we = ve / (ve + vb)
        D = we - w0
        if abs(D) >= band:
            ev.append((i, D))
            we = w0
    ev = pd.DataFrame(ev, columns=["i", "D"])
    vol = _ann_vol(close, int(p["vol_window"]))
    ev["date"] = close.index[ev["i"].to_numpy()]
    ev["vol_eq"] = vol[eq].to_numpy()[ev["i"].to_numpy()]
    ev["vol_bd"] = vol[bd].to_numpy()[ev["i"].to_numpy()]
    ev["n_hist"] = np.arange(len(ev))
    ev["s"] = np.sqrt((ev["D"] ** 2).cumsum().shift(1) / ev["n_hist"].replace(0, np.nan))
    ev["active"] = (ev["n_hist"] >= int(p["min_hist"])) & ev["vol_eq"].notna() & ev["vol_bd"].notna()
    ev["q"] = (ev["D"] / ev["s"]).clip(-p["clip"], p["clip"]).where(ev["active"], 0.0)
    return ev


# --------------------------------------------------------------------------- weights

def hold_weights(close: pd.DataFrame, p: dict) -> pd.DataFrame:
    """Desired holdings per return day (before conversion to decision weights)."""
    eq, bd = p["equity"], p["bond"]
    n = len(close)
    hold = np.zeros((n, 2))
    if p["mode"] == "monthly":
        mt = month_table(close, p)
        for _, row in mt[mt["active"]].iterrows():
            hold[int(row["i_a"]): int(row["i_b"]) + 1, 0] += row["w_eq"]
            hold[int(row["i_a"]): int(row["i_b"]) + 1, 1] += row["w_bd"]
    elif p["mode"] == "threshold":
        h0, h1 = int(p["hold"][0]), int(p["hold"][1])
        if h0 < 2:
            raise ValueError("lookahead: a trigger at close t can only be held from return day t+2")
        bands = list(p["bands"])
        for band in bands:
            ev = threshold_events(close, p, band)
            for _, e in ev[ev["active"]].iterrows():
                i = int(e["i"])
                lo, hi = i + h0, min(i + h1, n - 1)
                if lo > n - 1:
                    continue
                hold[lo: hi + 1, 0] += -e["q"] * p["target_vol"] / e["vol_eq"] / len(bands)
                hold[lo: hi + 1, 1] += e["q"] * p["target_vol"] / e["vol_bd"] / len(bands)
    else:
        raise ValueError(p["mode"])
    return pd.DataFrame(hold, index=close.index, columns=[eq, bd])


def decision_weights(period: str = "IS", **params) -> pd.DataFrame:
    p = _params(**params)
    close = _closes(period, p)
    # offsets on the planned calendar (unscheduled closures count as expected sessions), then
    # holdings mapped back to real sessions (calendar_utils.planned_to_realized)
    hold = CU.planned_to_realized(hold_weights(CU.to_planned(close), p), close.index)
    w_dec = CU.hold_to_decision(hold, EXEC)
    if int(p["extra_delay"]):
        w_dec = w_dec.shift(int(p["extra_delay"])).fillna(0.0)
    return w_dec


# --------------------------------------------------------------------------- tests

def _slope_test(y: pd.Series, x: pd.Series) -> dict:
    import statsmodels.api as sm

    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 10:
        return {"n": int(len(df))}
    fit = sm.OLS(df["y"], sm.add_constant(df["x"])).fit(cov_type="HC1")
    nz = df[df["x"] != 0]
    opp = int((np.sign(nz["y"]) == -np.sign(nz["x"])).sum())
    bt = stats.binomtest(opp, len(nz), 0.5, alternative="greater")
    return {
        "n": int(len(df)),
        "slope": float(fit.params["x"]),
        "slope_t_hc1": float(fit.tvalues["x"]),
        "slope_p_two_sided": float(fit.pvalues["x"]),
        "intercept": float(fit.params["const"]),
        "r2": float(fit.rsquared),
        "spearman_rho": float(stats.spearmanr(df["x"], df["y"]).statistic),
        "sign_opposes_frac": float(opp / len(nz)),
        "sign_test_p_one_sided": float(bt.pvalue),
    }


def window_pnl(res: E.BacktestResult, mt: pd.DataFrame) -> pd.Series:
    """Net excess P&L per traded window: return days a .. b+1 (the exit is costed on b+1)."""
    ex = res.excess
    cal = ex.index
    out = {}
    for T, row in mt[mt["active"]].iterrows():
        sl = ex.loc[row["start"]:]
        k = int(row["i_b"]) - int(row["i_a"]) + 2
        if len(sl) == 0:
            continue
        out[T] = float(sl.iloc[:k].sum())
    return pd.Series(out, dtype=float)


def prediction_tests(close: pd.DataFrame, p: dict, period: str, res: E.BacktestResult | None = None) -> dict:
    start, end = E.eval_window(period)
    mt = month_table(close, p)
    mt = mt[(mt["start"] >= pd.Timestamp(start)) & (mt["end"] <= pd.Timestamp(end))]
    out = {
        "all_months_spread_on_D": _slope_test(mt["spread"], mt["D"]),
        "active_months_spread_on_z": _slope_test(mt.loc[mt["active"], "spread"], mt.loc[mt["active"], "z"]),
        "mean_spread_D_pos": float(mt.loc[mt["D"] > 0, "spread"].mean()),
        "mean_spread_D_neg": float(mt.loc[mt["D"] < 0, "spread"].mean()),
        "n_D_pos": int((mt["D"] > 0).sum()),
        "n_D_neg": int((mt["D"] < 0).sum()),
    }
    # quintiles of D (all months) for the diagnostic figure / table
    qd = pd.qcut(mt["D"], 5, labels=False)
    out["spread_by_D_quintile"] = {int(k): float(v) for k, v in mt.groupby(qd)["spread"].mean().items()}
    if res is not None:
        act = mt[mt["active"]]
        pnl = window_pnl(res, act)
        reg = {}
        for name, sel in (("corr_pos", act["corr252"] > 0), ("corr_nonpos", act["corr252"] <= 0)):
            sub = act[sel]
            x = pnl.reindex(sub.index).dropna()
            reg[name] = {
                "n_windows": int(len(sub)),
                "mean_window_pnl_bp": float(x.mean() * 1e4) if len(x) else float("nan"),
                "t_window_pnl": float(x.mean() / x.std() * math.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else float("nan"),
                "hit_rate": float((x > 0).mean()) if len(x) else float("nan"),
                "spread_on_D": _slope_test(sub["spread"], sub["D"]),
            }
        reg["n_no_corr"] = int(act["corr252"].isna().sum())
        out["corr_regime_split"] = reg
        out["window_pnl_all"] = {
            "n": int(len(pnl)), "mean_bp": float(pnl.mean() * 1e4),
            "t": float(pnl.mean() / pnl.std() * math.sqrt(len(pnl))) if len(pnl) > 2 else float("nan"),
            "hit_rate": float((pnl > 0).mean()),
        }
    return out


# --------------------------------------------------------------------------- run

def _bt(name: str, p: dict, period: str, ohlc: dict, rf: pd.Series, log: bool, cost_mult: float = 1.0,
        note: str = "") -> E.BacktestResult:
    close = ohlc["close"][[p["equity"], p["bond"]]].ffill(limit=5)
    w_dec = CU.hold_to_decision(hold_weights(close, p), EXEC)
    if int(p["extra_delay"]):
        w_dec = w_dec.shift(int(p["extra_delay"])).fillna(0.0)
    return E.run_backtest(name, FAMILY, w_dec, ohlc, rf, period=period, params=p, exec=EXEC,
                          cost_mult=cost_mult, log=log, note=note)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, pd.Timestamp):
        return str(x.date())
    return x


def to_jsonable(summary: dict) -> dict:
    """Drop the private (underscore) keys holding results/series and make numbers JSON-safe."""
    return _clean(summary)


def run(period: str = "IS", log: bool = True) -> dict:
    """Base + 2x costs + every variant. Returns a summary dict; private keys ``_results`` (name ->
    BacktestResult) and ``_returns`` (daily net returns, one column per run) are for the caller."""
    p0 = _params()
    ohlc = E.load_ohlc(TICKERS if (p0["equity"], p0["bond"]) == tuple(TICKERS) else [p0["equity"], p0["bond"]], period)
    rf = E.load_rf(period)
    close = ohlc["close"][[p0["equity"], p0["bond"]]].ffill(limit=5)

    results: dict[str, E.BacktestResult] = {}
    results["base"] = _bt("IFC_A_base", p0, period, ohlc, rf, log, note="sleeve A base")
    results["base_2x"] = _bt("IFC_A_base", p0, period, ohlc, rf, log, cost_mult=2.0, note="sleeve A base, 2x costs")
    for vname, ov in VARIANTS.items():
        pv = _params(**ov)
        kind = "neighbourhood" if vname in NEIGHBOURHOOD else "robustness"
        results[vname] = _bt(f"IFC_A_{vname}", pv, period, ohlc, rf, log, note=f"sleeve A {kind} variant")

    base = results["base"]
    yearly = E.yearly_returns(base.returns)
    yearly_ex = base.excess.groupby(base.excess.index.year).sum()
    nb_sharpes = [results[v].stats["sharpe"] for v in NEIGHBOURHOOD]
    summary = {
        "module": "ifc_rebalance",
        "family": FAMILY,
        "sleeve": SLEEVE,
        "period": period,
        "exec": EXEC,
        "base_params": p0,
        "variants": VARIANTS,
        "stats": {k: r.stats for k, r in results.items()},
        "sharpe": {k: r.stats["sharpe"] for k, r in results.items()},
        "neighbourhood_median_sharpe": float(np.median(nb_sharpes)),
        "neighbourhood_median_over_base": float(np.median(nb_sharpes) / base.stats["sharpe"]) if base.stats["sharpe"] else None,
        "subperiods": A.subperiod_table(base).to_dict(orient="records"),
        "subperiods_2x": A.subperiod_table(results["base_2x"]).to_dict(orient="records"),
        "pnl_concentration": A.pnl_concentration(base),
        "factor_table": A.factor_table(base),
        "bootstrap_sharpe_ci90": list(A.bootstrap_sharpe_ci(base.excess)),
        "bootstrap_sharpe_ci90_2x": list(A.bootstrap_sharpe_ci(results["base_2x"].excess)),
        "yearly_net_returns": {int(k): float(v) for k, v in yearly.items()},
        "yearly_excess_sum": {int(k): float(v) for k, v in yearly_ex.items()},
        "exposure": {
            "mean_gross_when_active": float(base.weights.abs().sum(axis=1)[base.weights.abs().sum(axis=1) > 0].mean()),
            "max_gross": float(base.weights.abs().sum(axis=1).max()),
            "frac_days_invested": float((base.weights.abs().sum(axis=1) > 0).mean()),
            "threshold_max_gross": float(results["threshold"].weights.abs().sum(axis=1).max()),
            "threshold_frac_days_invested": float((results["threshold"].weights.abs().sum(axis=1) > 0).mean()),
        },
        "predictions": {"base": prediction_tests(close, p0, period, base)},
    }
    for v in NEIGHBOURHOOD:
        summary["predictions"][v] = prediction_tests(close, _params(**VARIANTS[v]), period)
    thr = {}
    pt = _params(**VARIANTS["threshold"])
    for band in pt["bands"]:
        ev = threshold_events(close, pt, band)
        ev = ev[ev["date"] <= pd.Timestamp(E.eval_window(period)[1])]
        thr[str(band)] = {"n_triggers": int(len(ev)), "n_active": int(ev["active"].sum()),
                          "first_active": str(ev.loc[ev["active"], "date"].min().date()) if ev["active"].any() else None,
                          "mean_abs_D": float(ev["D"].abs().mean())}
    summary["threshold_events"] = thr
    summary["_results"] = results
    summary["_returns"] = pd.DataFrame({k: r.returns for k, r in results.items()})
    summary["_month_table"] = month_table(close, p0)
    return summary
