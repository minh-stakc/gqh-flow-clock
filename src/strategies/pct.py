"""Candidate 2: Persistence-Conditioned Trend (PCT), exactly as pre-registered in HYPOTHESES.md s.2.

Family "PCT", execution ``next_open``: the decision made at the close of each month's last trading
day T (using closes up to and including T) is traded at the open of T+1 and then held at constant
target weights (the engine forward-fills the decision and rebalances drift at every open, charging
costs for it) until the next decision.

Interpretation decisions (the pre-registration was ambiguous on these points; the most natural
reading was chosen once and no alternative was tried):

1. Legs. "4 non-overlapping 63-day legs covering the past 252 days" = the last 252 daily log
   returns ending at T split into 4 consecutive blocks of W = 63 returns. Each leg therefore has
   W + 1 = 64 log prices; consecutive legs share their endpoint price (non-overlapping in returns).
   The same construction is used for W = 42 (6 legs) and W = 126 (2 legs).
2. Statistic. Log total-return prices p_0..p_W (the adjusted close). Chord c_j = p_0 + (j/W)(p_W - p_0).
   A = mean over all W + 1 points (endpoints included, where the deviation is 0) of |p_j - c_j|.
   L = mean of |p_j - p_{j-1}| over the W returns. R = A / (L * sqrt(W)); Rbar = mean of R over legs.
3. Null. mu0, sd0 = mean and sample standard deviation (ddof=1) of Rbar over MC_PATHS = 50 000
   zero-drift, unit-variance Gaussian random walks of length W * legs, seed MC_SEED, computed once
   per (W, legs) and cached (functools.lru_cache). R is invariant to the increment scale and the
   chord absorbs a linear drift in A, so sigma = 1 and zero drift is the natural null.
4. Trend sign. s_i = sign(close_T / close_{T-lb} - 1 - (prod over the same lb return days of
   (1 + rf_t) - 1)), with rf the engine's daily T-bill return on the NYSE calendar (forward-filled
   exactly like the engine does). A sign of exactly 0 gives no position.
5. Eligibility. An ETF is eligible at T when it has at least 300 valid closes up to and including T
   (and, as a safeguard, complete data inside every window used). N = number of eligible ETFs at T.
6. vol_i = sample standard deviation (ddof=1) of the last ``vol_window`` daily simple close-to-close
   returns ending at T, times sqrt(252). The vol-window-126 neighbourhood point changes only vol_i;
   the ex-ante covariance window stays at 252 days.
7. Scaling. Sigma = trailing 252-day covariance of daily simple returns of the eligible ETFs
   (pandas pairwise-complete covariance, annualised x252). w <- w * 0.10 / sqrt(w' Sigma w), then if
   sum|w| > 3, w <- w * 3 / sum|w|. Weights do not sum to one; the remainder is cash at the T-bill
   rate (negative cash = financing at the T-bill rate, as the engine models it).
8. Multiplier floor. m = max(0, 1 + clip(z, lo, hi)). The floor never binds for the base (+/-1)
   or the +/-0.5 point; for the "+/-2" point it implements the pre-registered "m in [0, 3]"
   (i.e. z is effectively clipped to [-1, 2]; the trend direction is never reversed).
9. ID variant (Da, Gurun & Warachka 2014). PRET = raw total return over the past 252 days (the
   paper's cumulative formation return, not the excess return); %pos / %neg = fraction of the 252
   daily returns that are > 0 / < 0 (zero days count in neither). ID = sign(PRET) * (%neg - %pos),
   z = -ID * sqrt(252), m = 1 + clip(z, -1, 1). Trend sign, sizing and scaling are unchanged.
10. TSMOM benchmark: identical code with m = 1 for every asset.
11. Extra execution delay (robustness): the forward-filled daily decision weights are shifted one
   more trading day, so each monthly decision is traded at the open of T+2.
12. P2 forward return. For a decision at T, r_next is the asset's open(T+1) -> open(T_next+1)
   return (exactly the period the strategy holds that decision) minus the compounded T-bill over
   return days T+1..T_next; y = s_i * r_next / vol_i (vol_i annualised, base 63-day window).
   Terciles are pooled over all asset-months (rank-based so ties split evenly); the top-minus-bottom
   difference is tested with month-clustered OLS standard errors and, as a check, a Newey-West
   t-stat on the monthly series of (mean top - mean bottom). The final month whose holding period
   ends outside the evaluation window is dropped.

The module is universe-agnostic (``tickers`` param) so the PCT-F futures replication (HYPOTHESES.md
s.4, Databento) can reuse it once the futures cache exists; that version's ``next_close`` execution
and cost map are not implemented here.
"""
from __future__ import annotations

import functools
import json
import math

import numpy as np
import pandas as pd

from .. import analysis as A
from .. import calendar_utils as CU
from .. import config as C
from .. import engine as E

FAMILY = "PCT"
EXEC = "next_open"
TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "LQD", "HYG", "TIP", "GLD", "SLV", "DBC", "UUP", "VNQ"]
COST_BPS = None   # engine defaults: 3 bp tier-1, 5 bp tier-2 (TIP, SLV, DBC, UUP, VNQ), 30 bp/yr borrow

MC_PATHS = 50_000
MC_SEED = 20261003

BASE_PARAMS = {
    "measure": "R",          # "R" (PCT), "ID" (information discreteness) or "none" (TSMOM, m = 1)
    "W": 63,                 # leg length in returns
    "legs": 4,               # number of legs (W * legs = 252)
    "trend_lb": 252,         # trend lookback for the sign
    "clip_lo": -1.0,
    "clip_hi": 1.0,
    "vol_window": 63,
    "cov_window": 252,
    "target_vol": 0.10,
    "gross_cap": 3.0,
    "raw_scale": 0.40,
    "min_history": 300,
    "id_window": 252,
    "extra_delay": 0,
}

# Every pre-registered variant that applies to this module. "cost_mult" is consumed by run(), not by
# decision_weights(). NEIGHBOURHOOD lists the gate-G3 points (HYPOTHESES.md s.3).
VARIANTS = {
    "tsmom_benchmark": {"measure": "none"},
    "id_variant": {"measure": "ID"},
    "W42_6legs": {"W": 42, "legs": 6},
    "W126_2legs": {"W": 126, "legs": 2},
    "trend126": {"trend_lb": 126},
    "clip05": {"clip_lo": -0.5, "clip_hi": 0.5},
    "clip2": {"clip_lo": -2.0, "clip_hi": 2.0},
    "vol126": {"vol_window": 126},
    "delay1": {"extra_delay": 1},
    "cost2x": {"cost_mult": 2.0},
}
NEIGHBOURHOOD = ["W42_6legs", "W126_2legs", "trend126", "clip05", "clip2", "vol126"]


# --------------------------------------------------------------------------- statistic

def bridge_excursion(logp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R, A, L for legs of log prices along the last axis (shape (..., W+1))."""
    logp = np.asarray(logp, dtype=float)
    W = logp.shape[-1] - 1
    frac = np.arange(W + 1) / W
    chord = logp[..., :1] + frac * (logp[..., -1:] - logp[..., :1])
    a = np.abs(logp - chord).mean(axis=-1)
    l = np.abs(np.diff(logp, axis=-1)).mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / (l * math.sqrt(W))
    return r, a, l


def leg_stats(logp_window: np.ndarray, W: int, legs: int) -> tuple[float, float, float]:
    """Average R, A, L over ``legs`` consecutive legs of W returns; logp_window has W*legs+1 points."""
    assert len(logp_window) == W * legs + 1
    idx = np.arange(legs)[:, None] * W + np.arange(W + 1)[None, :]
    r, a, l = bridge_excursion(logp_window[idx])
    return float(r.mean()), float(a.mean()), float(l.mean())


@functools.lru_cache(maxsize=None)
def null_moments(W: int, legs: int, n_paths: int = MC_PATHS, seed: int = MC_SEED) -> tuple[float, float]:
    """(mu0, sd0) of Rbar under a zero-drift Gaussian random walk (Monte Carlo, fixed seed)."""
    rng = np.random.default_rng(seed)
    out = []
    chunk = max(1, 2_000_000 // (legs * W))      # bound memory for the large-W check
    for start in range(0, n_paths, chunk):
        n = min(chunk, n_paths - start)
        inc = rng.standard_normal((n, legs, W))
        p = np.concatenate([np.zeros((n, legs, 1)), np.cumsum(inc, axis=-1)], axis=-1)
        r, _, _ = bridge_excursion(p)
        out.append(r.mean(axis=-1))
    rbar = np.concatenate(out)
    return float(rbar.mean()), float(rbar.std(ddof=1))


def large_w_check(seed: int = MC_SEED) -> dict:
    """Mean single-leg R (with its Monte Carlo standard error) for long Gaussian random walks;
    it should approach pi/8 = 0.3927 as W grows."""
    out = {}
    for W, n in ((2_000, 50_000), (10_000, 10_000)):
        mu, sd = null_moments(W, 1, n, seed)
        out[f"W{W}"] = {"mean_R": mu, "mc_se": sd / math.sqrt(n), "n_paths": n,
                        "z_vs_pi_over_8": (mu - math.pi / 8) / (sd / math.sqrt(n))}
    return out


# --------------------------------------------------------------------------- data / signals

@functools.lru_cache(maxsize=4)
def _data(period: str, tickers: tuple[str, ...]):
    ohlc = E.load_ohlc(list(tickers), period)
    rf = E.load_rf(period)
    return ohlc, rf


def decision_dates(cal: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Each complete month's last trading day T (an incomplete final month has no T)."""
    mo = CU.month_offsets(cal)
    return pd.DatetimeIndex(mo.index[mo["off_own"] == 0])


def _params(**overrides) -> dict:
    p = dict(BASE_PARAMS)
    unknown = set(overrides) - set(p) - {"tickers"}
    if unknown:
        raise KeyError(f"unknown PCT params: {sorted(unknown)}")
    p.update(overrides)
    return p


def signal_panel(period: str = "IS", tickers: list[str] | None = None, **params) -> pd.DataFrame:
    """Long table (date, ticker) of every signal ingredient at each decision date for eligible assets."""
    p = _params(**params)
    tickers = list(tickers or TICKERS)
    ohlc, rf = _data(period, tuple(tickers))
    close = ohlc["close"][tickers]
    cal = close.index
    rf_d = rf.reindex(cal).ffill().fillna(0.0)
    logp = np.log(close.to_numpy())
    ret = close.pct_change(fill_method=None).to_numpy()
    lrf = np.log1p(rf_d.to_numpy())
    cum_lrf = np.concatenate([[0.0], np.cumsum(lrf)])       # cum_lrf[i+1] = sum lrf[0..i]
    nvalid = np.cumsum(~np.isnan(close.to_numpy()), axis=0)
    W, legs = int(p["W"]), int(p["legs"])
    span = W * legs
    if p["measure"] == "R":
        mu0, sd0 = null_moments(W, legs)
    rows = []
    pos = {d: i for i, d in enumerate(cal)}
    for d in decision_dates(cal):
        i = pos[d]
        for j, t in enumerate(tickers):
            if nvalid[i, j] < p["min_history"] or np.isnan(logp[i, j]):
                continue
            need = max(span, int(p["trend_lb"]), int(p["vol_window"]), int(p["cov_window"]), int(p["id_window"]))
            if i - need < 0 or np.isnan(logp[i - need: i + 1, j]).any():
                continue
            lb = int(p["trend_lb"])
            tr = math.exp(logp[i, j] - logp[i - lb, j]) - 1.0
            rf_win = math.exp(cum_lrf[i + 1] - cum_lrf[i - lb + 1]) - 1.0
            s = float(np.sign(tr - rf_win))
            vw = int(p["vol_window"])
            vol = float(np.std(ret[i - vw + 1: i + 1, j], ddof=1) * math.sqrt(C.TRADING_DAYS))
            rbar, abar, lbar = leg_stats(logp[i - span: i + 1, j], W, legs)
            idw = int(p["id_window"])
            rw = ret[i - idw + 1: i + 1, j]
            pret = math.exp(logp[i, j] - logp[i - idw, j]) - 1.0
            info_d = float(np.sign(pret) * ((rw < 0).mean() - (rw > 0).mean()))
            z_id = -info_d * math.sqrt(idw)
            if p["measure"] == "R":
                z = (rbar - mu0) / sd0
            elif p["measure"] == "ID":
                z = z_id
            else:
                z = 0.0
            if p["measure"] == "none":
                m = 1.0
            else:
                m = max(0.0, 1.0 + min(max(z, p["clip_lo"]), p["clip_hi"]))
            rows.append({"date": d, "ticker": t, "s": s, "trend_ret": tr, "rf_win": rf_win, "vol": vol,
                         "Rbar": rbar, "Abar": abar, "Lbar": lbar, "z": z, "z_id": z_id, "ID": info_d, "m": m})
    return pd.DataFrame(rows)


def decision_weights(period: str = "IS", tickers: list[str] | None = None, return_panel: bool = False, **params):
    """Daily DECISION weights (forward-filled between monthly decisions, 0 before the first one)."""
    p = _params(**params)
    tickers = list(tickers or TICKERS)
    ohlc, _ = _data(period, tuple(tickers))
    close = ohlc["close"][tickers]
    cal = close.index
    ret = close.pct_change(fill_method=None)
    panel = signal_panel(period, tickers, **{k: v for k, v in p.items() if k != "extra_delay"})
    w_dec = pd.DataFrame(np.nan, index=cal, columns=tickers)
    diag = []
    for d, g in panel.groupby("date"):
        g = g.set_index("ticker")
        n = len(g)
        raw = g["s"] * g["m"] * (p["raw_scale"] / g["vol"]) / n
        i = cal.get_loc(d)
        cw = int(p["cov_window"])
        win = ret.iloc[i - cw + 1: i + 1][list(g.index)]
        sigma = win.cov(min_periods=20).to_numpy() * C.TRADING_DAYS
        wv = raw.to_numpy()
        var = float(wv @ np.nan_to_num(sigma) @ wv)
        row = pd.Series(0.0, index=tickers)
        scale, capped = 0.0, False
        if var > 0:
            scale = p["target_vol"] / math.sqrt(var)
            w = raw * scale
            gross = float(w.abs().sum())
            if gross > p["gross_cap"]:
                w = w * p["gross_cap"] / gross
                capped = True
            row[w.index] = w.values
        w_dec.loc[d] = row.values
        we = row[list(g.index)].to_numpy()
        diag.append({"date": d, "n_eligible": n, "gross": float(row.abs().sum()), "net": float(row.sum()),
                     "capped": capped, "scale": scale,
                     "exante_vol": float(math.sqrt(max(we @ np.nan_to_num(sigma) @ we, 0.0)))})
    w_dec = w_dec.ffill().fillna(0.0)
    if p["extra_delay"]:
        w_dec = w_dec.shift(int(p["extra_delay"])).fillna(0.0)
    if return_panel:
        return w_dec, panel, pd.DataFrame(diag).set_index("date")
    return w_dec


# --------------------------------------------------------------------------- tests

def _clean(obj):
    """JSON-safe conversion (numpy scalars, NaN -> None, DataFrames -> records)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _clean(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _clean({str(k): v for k, v in obj.items()})
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def sharpe_diff_bootstrap(ex_a: pd.Series, ex_b: pd.Series, block: int = 21, n_boot: int = 2000,
                          seed: int = 7) -> dict:
    """Paired moving-block bootstrap of Sharpe(a) - Sharpe(b) (same resampled days for both)."""
    df = pd.concat([ex_a.rename("a"), ex_b.rename("b")], axis=1, join="inner").dropna()
    x = df.to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    k = math.sqrt(C.TRADING_DAYS)

    def sr(v):
        return v.mean(axis=0) / v.std(axis=0, ddof=1) * k

    point = sr(x)
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        s = sr(x[idx])
        out[b] = s[0] - s[1]
    return {"sharpe_a": float(point[0]), "sharpe_b": float(point[1]), "diff": float(point[0] - point[1]),
            "ci90": [float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95))],
            "frac_boot_diff_gt_0": float((out > 0).mean()), "n_days": n, "block": block, "n_boot": n_boot,
            "seed": seed}


def forward_returns(panel: pd.DataFrame, period: str, tickers: list[str]) -> pd.DataFrame:
    """Attach r_next (open(T+1) -> open(T_next+1), minus T-bill) and y = s * r_next / vol."""
    ohlc, rf = _data(period, tuple(tickers))
    op = ohlc["open"][tickers]
    cal = op.index
    rf_d = rf.reindex(cal).ffill().fillna(0.0)
    cum_lrf = pd.Series(np.cumsum(np.log1p(rf_d.to_numpy())), index=cal)
    dates = decision_dates(cal)
    nxt = {d: dates[k + 1] for k, d in enumerate(dates[:-1])}
    pos = {d: i for i, d in enumerate(cal)}
    out = []
    for _, r in panel.iterrows():
        d = r["date"]
        if d not in nxt:
            continue
        i0, i1 = pos[d] + 1, pos[nxt[d]] + 1
        if i1 >= len(cal):
            continue
        o0, o1 = op.iat[i0, tickers.index(r["ticker"])], op.iat[i1, tickers.index(r["ticker"])]
        if not (np.isfinite(o0) and np.isfinite(o1)):
            continue
        rf_hold = math.exp(cum_lrf.iat[i1 - 1] - cum_lrf.iat[i0 - 1]) - 1.0   # return days T+1..T_next
        rn = o1 / o0 - 1.0 - rf_hold
        out.append({**r.to_dict(), "r_next": rn, "y": r["s"] * rn / r["vol"]})
    return pd.DataFrame(out)


def tercile_test(df: pd.DataFrame, zcol: str) -> dict:
    """Pooled terciles of zcol; top-minus-bottom mean of y with month-clustered and NW t-stats."""
    import statsmodels.api as sm

    d = df[["date", zcol, "y"]].dropna().copy()
    d["terc"] = pd.qcut(d[zcol].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
    means = d.groupby("terc")["y"].agg(["mean", "count", "std"])
    sub = d[d["terc"].isin([1, 3])].copy()
    sub["top"] = (sub["terc"] == 3).astype(float)
    X = sm.add_constant(sub[["top"]])
    groups = pd.factorize(sub["date"])[0]
    fit = sm.OLS(sub["y"], X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    monthly = sub.groupby(["date", "terc"])["y"].mean().unstack()
    mdiff = (monthly[3] - monthly[1]).dropna() if {1, 3} <= set(monthly.columns) else pd.Series(dtype=float)
    return {
        "zcol": zcol,
        "tercile_means": {int(k): float(v) for k, v in means["mean"].items()},
        "tercile_counts": {int(k): int(v) for k, v in means["count"].items()},
        "tercile_z_ranges": {int(k): [float(g[zcol].min()), float(g[zcol].max())] for k, g in d.groupby("terc")},
        "top_minus_bottom": float(fit.params["top"]),
        "t_cluster_month": float(fit.tvalues["top"]),
        "n_obs": int(len(d)),
        "n_months": int(d["date"].nunique()),
        "monthly_diff_mean": float(mdiff.mean()) if len(mdiff) else float("nan"),
        "monthly_diff_nw_t": E.newey_west_tstat(mdiff) if len(mdiff) else float("nan"),
        "n_months_both": int(len(mdiff)),
    }


def vol_neutrality(panel: pd.DataFrame) -> dict:
    """P3: correlation of z (and, for contrast, of the scale-dependent Abar) with the asset's vol."""
    d = panel.dropna(subset=["z", "vol"])
    within = d.groupby("date").apply(
        lambda g: g["z"].corr(g["vol"], method="spearman") if len(g) >= 5 else np.nan, include_groups=False
    )
    return {
        "pearson_z_vol": float(d["z"].corr(d["vol"])),
        "spearman_z_vol": float(d["z"].corr(d["vol"], method="spearman")),
        "mean_within_month_spearman_z_vol": float(within.mean()),
        "spearman_zid_vol": float(d["z_id"].corr(d["vol"], method="spearman")),
        "contrast_spearman_Abar_vol": float(d["Abar"].corr(d["vol"], method="spearman")),
        "contrast_spearman_Lbar_vol": float(d["Lbar"].corr(d["vol"], method="spearman")),
        "z_mean": float(d["z"].mean()),
        "z_sd": float(d["z"].std()),
        "n_obs": int(len(d)),
    }


# --------------------------------------------------------------------------- run

def _variant_params(name: str) -> tuple[dict, float]:
    over = dict(VARIANTS.get(name, {})) if name != "base" else {}
    cost_mult = float(over.pop("cost_mult", 1.0))
    return _params(**over), cost_mult


def run(period: str = "IS", log: bool = True) -> dict:
    tickers = list(TICKERS)
    ohlc, rf = _data(period, tuple(tickers))
    mc = {f"W{W}_legs{L}": dict(zip(("mu0", "sd0"), null_moments(W, L)))
          for W, L in ((63, 4), (42, 6), (126, 2))}
    mc["large_W_single_leg"] = large_w_check()
    mc["pi_over_8"] = math.pi / 8
    mc["n_paths"], mc["seed"] = MC_PATHS, MC_SEED
    print("MC null:", json.dumps(mc, indent=1))

    results: dict[str, E.BacktestResult] = {}
    panels, diags = {}, {}
    for name in ["base"] + list(VARIANTS):
        p, cost_mult = _variant_params(name)
        w, panel, diag = decision_weights(period, tickers, return_panel=True, **p)
        panels[name], diags[name] = panel, diag
        params = {**p, "tickers": tickers, "mc_paths": MC_PATHS, "mc_seed": MC_SEED, "variant": name}
        results[name] = E.run_backtest(f"PCT_{name}", FAMILY, w, ohlc, rf, period=period, params=params,
                                       exec=EXEC, cost_mult=cost_mult, log=log)
        print(f"{name:16s} SR={results[name].stats['sharpe']:.3f}  ann={results[name].stats['ann_return']:.4f}"
              f"  vol={results[name].stats['ann_vol']:.4f}  TO={results[name].stats['turnover_per_year']:.2f}")

    base = results["base"]
    summary: dict = {"family": FAMILY, "exec": EXEC, "tickers": tickers, "period": period,
                     "base_params": BASE_PARAMS, "variants_def": VARIANTS, "mc_null": mc}
    summary["base_stats"] = base.stats
    summary["base_stats_cost2x"] = results["cost2x"].stats
    summary["base_stats_delay1"] = results["delay1"].stats
    summary["variant_sharpe"] = {k: r.stats["sharpe"] for k, r in results.items()}
    summary["variant_stats"] = {k: r.stats for k, r in results.items()}
    summary["subperiods"] = A.subperiod_table(base)
    summary["subperiods_cost2x"] = A.subperiod_table(results["cost2x"])
    summary["pnl_concentration"] = A.pnl_concentration(base)
    summary["factor_table"] = A.factor_table(base)
    summary["factor_table_tsmom"] = A.factor_table(results["tsmom_benchmark"])
    summary["bootstrap_sharpe_ci90"] = A.bootstrap_sharpe_ci(base.excess)
    summary["bootstrap_sharpe_ci90_cost2x"] = A.bootstrap_sharpe_ci(results["cost2x"].excess)
    # pre-registered robustness run (HYPOTHESES.md s.3): square-root-impact capacity curve of the base
    summary["capacity_curve"] = A.capacity_curve(base, ohlc).to_dict("records")
    summary["yearly_returns"] = E.yearly_returns(base.returns)
    summary["yearly_returns_tsmom"] = E.yearly_returns(results["tsmom_benchmark"].returns)
    nb = [results[k].stats["sharpe"] for k in NEIGHBOURHOOD]
    pc = summary["pnl_concentration"]
    summary["gates"] = {
        "G1_sharpe_2x_costs": results["cost2x"].stats["sharpe"],
        "G1_pass": results["cost2x"].stats["sharpe"] > 0,
        "G2_best_year": pc["best_year"],
        "G2_best_year_share": pc["best_year_share"],
        "G2_pass": bool(pc["best_year_share"] is not None and np.isfinite(pc["best_year_share"])
                        and pc["best_year_share"] <= 0.40),
        "G3_neighbourhood_sharpes": dict(zip(NEIGHBOURHOOD, nb)),
        "G3_median": float(np.median(nb)),
        "G3_threshold_half_base": base.stats["sharpe"] / 2,
        "G3_pass": float(np.median(nb)) >= base.stats["sharpe"] / 2,
    }
    d = diags["base"]
    summary["portfolio_diagnostics"] = {
        "first_decision": d.index[0], "n_decisions": int(len(d)),
        "mean_n_eligible": float(d["n_eligible"].mean()), "mean_gross": float(d["gross"].mean()),
        "max_gross": float(d["gross"].max()), "frac_gross_capped": float(d["capped"].mean()),
        "mean_net_exposure": float(d["net"].mean()),
        "mean_m_base": float(panels["base"]["m"].mean()),
        "frac_m_at_0": float((panels["base"]["m"] <= 1e-12).mean()),
        "frac_m_at_2": float((panels["base"]["m"] >= 2 - 1e-12).mean()),
        "mean_turnover_per_year_tsmom": results["tsmom_benchmark"].stats["turnover_per_year"],
    }

    # ---- predictions
    p1 = sharpe_diff_bootstrap(base.excess, results["tsmom_benchmark"].excess)
    p1_id = sharpe_diff_bootstrap(results["id_variant"].excess, results["tsmom_benchmark"].excess)
    start, end = E.eval_window(period)
    pan = panels["base"]
    pan = pan[(pan["date"] >= pd.Timestamp(start)) & (pan["date"] <= pd.Timestamp(end))]
    fwd = forward_returns(pan, period, tickers)
    fwd = fwd[fwd["s"] != 0]
    p2_r = tercile_test(fwd, "z")
    p2_id = tercile_test(fwd, "z_id")
    p3 = vol_neutrality(pan)
    summary["predictions"] = {
        "P1_pct_minus_tsmom": p1,
        "P1_pass": p1["diff"] > 0,
        "P1_id_minus_tsmom": p1_id,
        "P2_R": p2_r,
        "P2_pass": p2_r["top_minus_bottom"] > 0,
        "P2_ID": p2_id,
        "P3_vol_neutrality": p3,
        "kill_condition_triggered": (p1["diff"] <= 0) and (p2_r["top_minus_bottom"] <= 0),
    }
    rets = pd.DataFrame({k: r.returns for k, r in results.items()})
    summary = _clean(summary)
    summary["_returns"] = rets
    summary["_results"] = results
    summary["_fwd_panel"] = fwd
    summary["_panel"] = pan
    return summary
