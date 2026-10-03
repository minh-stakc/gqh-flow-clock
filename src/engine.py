"""Backtest engine shared by every candidate strategy.

Conventions (identical for every strategy, so results are comparable):

* A strategy returns DECISION weights: ``w_dec.loc[d]`` may use only information
  available at the close of session ``d``.
* Execution is at the NEXT session's open (``exec="next_open"``, default) or the
  next session's close (``exec="next_close"``). The decision is never filled on
  the bar that produced it, which removes same-bar lookahead by construction.
* Every reported return is net of transaction costs (one-way bps per unit of
  turnover, per instrument) and of a borrow fee on short positions. Cash earns
  the 3-month T-bill rate. Sharpe ratios use returns in excess of the T-bill.
* In-sample runs physically cannot see out-of-sample data: the loaders truncate
  at ``IS_END`` unless the OOS lock is explicitly released with the environment
  variable ``GQH_OOS_UNLOCK=1``. Every OOS evaluation is appended to
  ``results/oos_log.csv`` and every backtest of any kind to ``results/trials.csv``
  so the number of variants tried is auditable.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

EULER_GAMMA = 0.5772156649015329


class OOSLocked(RuntimeError):
    pass


# --------------------------------------------------------------------------- periods

def oos_unlocked() -> bool:
    return os.environ.get("GQH_OOS_UNLOCK") == "1"


def data_end(period: str) -> str:
    """Last date a strategy is allowed to SEE for the given period."""
    if period == "IS":
        return C.IS_END
    if period in ("OOS", "FULL"):
        if not oos_unlocked():
            raise OOSLocked(
                "Out-of-sample data is locked. It is evaluated once, at the end, via "
                "`GQH_OOS_UNLOCK=1 python run_all.py --final`."
            )
        return C.OOS_END
    raise ValueError(period)


def eval_window(period: str) -> tuple[str, str]:
    if period == "IS":
        return C.HISTORY_START, C.IS_END
    if period == "OOS":
        data_end(period)
        return C.OOS_START, C.OOS_END
    if period == "FULL":
        data_end(period)
        return C.HISTORY_START, C.OOS_END
    raise ValueError(period)


# --------------------------------------------------------------------------- data

def _read_parquet(name: str) -> pd.DataFrame:
    path = C.DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python data/download.py` first")
    return pd.read_parquet(path)


def load_ohlc(tickers: list[str] | None = None, period: str = "IS") -> dict[str, pd.DataFrame]:
    """Total-return-adjusted daily OHLCV panels (dict of wide DataFrames), truncated to the period."""
    df = _read_parquet("etf_daily.parquet")
    if tickers is not None:
        missing = sorted(set(tickers) - set(df["ticker"].unique()))
        if missing:
            raise KeyError(f"tickers not in cache: {missing}")
        df = df[df["ticker"].isin(tickers)]
    end = pd.Timestamp(data_end(period))
    df = df[(df["date"] >= pd.Timestamp(C.HISTORY_START)) & (df["date"] <= end)]
    out = {}
    for col in ("open", "high", "low", "close", "volume"):
        wide = df.pivot(index="date", columns="ticker", values=col).sort_index()
        out[col] = wide if tickers is None else wide.reindex(columns=tickers)
    return out


def load_series(name: str, period: str = "IS") -> pd.DataFrame:
    """Any auxiliary cached table indexed by date (rf, factors, macro), truncated to the period."""
    df = _read_parquet(name)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.loc[: pd.Timestamp(data_end(period))].sort_index()


def load_rf(period: str = "IS") -> pd.Series:
    return load_series("rf_daily.parquet", period)["rf"]


# --------------------------------------------------------------------------- simulation

@dataclass
class BacktestResult:
    name: str
    period: str
    returns: pd.Series            # daily net returns (total, incl. cash)
    gross_returns: pd.Series      # daily returns before costs
    excess: pd.Series             # net returns minus T-bill
    turnover: pd.Series           # sum |trade| as a fraction of NAV, per day
    weights: pd.DataFrame         # weights held after the day's trade (at the open)
    costs: pd.Series
    params: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


def simulate(
    w_dec: pd.DataFrame,
    ohlc: dict[str, pd.DataFrame],
    rf: pd.Series,
    exec: str = "next_open",
    cost_mult: float = 1.0,
    cost_bps: dict[str, float] | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    """Return (net, gross, turnover, held_weights, costs) daily series.

    next_open: the target decided at close d is traded at the open of d+1.
      r_t = sum(w_old * r_co) + (1 + sum(w_old * r_co)) * sum(w_new * r_oc)
      with r_co = open_t / close_{t-1} - 1 and r_oc = close_t / open_t - 1.
      This is exact for rebalancing to target at the open.
    next_close: the target decided at close d is traded at the close of d+1 and
      earns close-to-close returns from d+2 on.
    """
    close = ohlc["close"]
    tickers = list(w_dec.columns)
    close = close[tickers]
    idx = close.index
    w_dec = w_dec.reindex(idx).ffill().fillna(0.0)
    rf = rf.reindex(idx).ffill().fillna(0.0)
    cb = pd.Series({t: (cost_bps or {}).get(t, C.cost_bps(t)) for t in tickers}) * cost_mult / 1e4
    borrow_daily = C.SHORT_BORROW_BPS_PER_YEAR / 1e4 / C.TRADING_DAYS

    r_cc = close.pct_change(fill_method=None).fillna(0.0)
    if exec == "next_open":
        open_ = ohlc["open"][tickers]
        r_co = (open_ / close.shift(1) - 1).fillna(0.0)
        r_oc = (close / open_ - 1).fillna(0.0)
        w_new = w_dec.shift(1).fillna(0.0)     # traded at today's open
        w_old = w_dec.shift(2).fillna(0.0)     # held overnight into today's open
        on = (w_old * r_co).sum(axis=1)
        gross_risky = on + (1 + on) * (w_new * r_oc).sum(axis=1)
        drifted = w_old.mul(1 + r_co).div((1 + on).replace(0, np.nan), axis=0).fillna(0.0)
        trade = (w_new - drifted).abs()
        held = w_new
    elif exec == "next_close":
        held = w_dec.shift(2).fillna(0.0)
        prev = w_dec.shift(3).fillna(0.0)
        port_prev = (prev * r_cc.shift(1).fillna(0.0)).sum(axis=1)
        drifted = prev.mul(1 + r_cc.shift(1).fillna(0.0)).div((1 + port_prev).replace(0, np.nan), axis=0).fillna(0.0)
        trade = (held - drifted).abs()
        gross_risky = (held * r_cc).sum(axis=1)
    else:
        raise ValueError(exec)

    cash_w = 1.0 - held.sum(axis=1)
    gross = gross_risky + cash_w * rf
    costs = (trade * cb).sum(axis=1) + held.clip(upper=0).abs().sum(axis=1) * borrow_daily
    net = gross - costs
    turnover = trade.sum(axis=1)
    return net, gross, turnover, held, costs


# --------------------------------------------------------------------------- statistics

def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def newey_west_tstat(x: pd.Series, lags: int | None = None) -> float:
    x = pd.Series(x).dropna().to_numpy()
    n = len(x)
    if n < 10:
        return float("nan")
    lags = lags if lags is not None else int(4 * (n / 100) ** (2 / 9))
    e = x - x.mean()
    s = e @ e / n
    for k in range(1, lags + 1):
        wk = 1 - k / (lags + 1)
        s += 2 * wk * (e[k:] @ e[:-k]) / n
    return float(x.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def probabilistic_sharpe(sr_d: float, n: int, skew: float, kurt: float, sr_star_d: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012). Sharpe ratios in per-period (daily) units; kurt is raw (normal=3)."""
    denom = 1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2
    if denom <= 0 or n < 3:
        return float("nan")
    return float(stats.norm.cdf((sr_d - sr_star_d) * math.sqrt(n - 1) / math.sqrt(denom)))


def expected_max_sharpe(n_trials: int, var_sr_d: float) -> float:
    """E[max SR] of n_trials unskilled strategies (Bailey & Lopez de Prado 2014), daily units."""
    if n_trials <= 1 or var_sr_d <= 0:
        return 0.0
    a = (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
    b = EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e))
    return float(math.sqrt(var_sr_d) * (a + b))


def deflated_sharpe(excess: pd.Series, n_trials: int, var_sr_ann: float) -> dict:
    x = excess.dropna()
    n = len(x)
    sr_d = x.mean() / x.std() if x.std() > 0 else 0.0
    skew = float(stats.skew(x))
    kurt = float(stats.kurtosis(x, fisher=False))
    sr0_d = expected_max_sharpe(n_trials, var_sr_ann / C.TRADING_DAYS)
    return {
        "dsr": probabilistic_sharpe(sr_d, n, skew, kurt, sr0_d),
        "sr0_ann": sr0_d * math.sqrt(C.TRADING_DAYS),
        "n_trials": n_trials,
    }


def perf_stats(net: pd.Series, excess: pd.Series, turnover: pd.Series) -> dict:
    n = len(net)
    years = n / C.TRADING_DAYS
    eq = (1 + net).cumprod()
    ann_ret = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")
    vol = float(net.std() * math.sqrt(C.TRADING_DAYS))
    sr = float(excess.mean() / excess.std() * math.sqrt(C.TRADING_DAYS)) if excess.std() > 0 else 0.0
    monthly = (1 + net).resample("ME").prod() - 1
    sr_d = sr / math.sqrt(C.TRADING_DAYS)
    skew = float(stats.skew(excess))
    kurt = float(stats.kurtosis(excess, fisher=False))
    active = (turnover > 0).mean()
    return {
        "start": str(net.index[0].date()),
        "end": str(net.index[-1].date()),
        "years": round(years, 2),
        "ann_return": ann_ret,
        "ann_vol": vol,
        "sharpe": sr,
        "max_drawdown": max_drawdown(net),
        "turnover_per_year": float(turnover.sum() / years) if years > 0 else float("nan"),
        "worst_month": float(monthly.min()),
        "best_month": float(monthly.max()),
        "skew_daily": skew,
        "kurtosis_daily": kurt,
        "hit_rate_daily": float((excess > 0).mean()),
        "nw_tstat_mean_excess": newey_west_tstat(excess),
        "psr_vs_0": probabilistic_sharpe(sr_d, n, skew, kurt, 0.0),
        "trading_days_frac": float(active),
        "calmar": float(ann_ret / abs(max_drawdown(net))) if max_drawdown(net) < 0 else float("nan"),
    }


def yearly_returns(net: pd.Series) -> pd.Series:
    return (1 + net).groupby(net.index.year).prod() - 1


def factor_regression(excess: pd.Series, factors: pd.DataFrame, lags: int = 5) -> dict:
    """OLS of daily strategy excess returns on factor returns with Newey-West (HAC) errors."""
    import statsmodels.api as sm

    df = pd.concat([excess.rename("y"), factors], axis=1, join="inner").dropna()
    X = sm.add_constant(df.drop(columns="y"))
    fit = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "alpha_ann": float(fit.params["const"] * C.TRADING_DAYS),
        "alpha_t": float(fit.tvalues["const"]),
        "betas": {k: float(v) for k, v in fit.params.drop("const").items()},
        "t": {k: float(v) for k, v in fit.tvalues.drop("const").items()},
        "r2": float(fit.rsquared),
        "n": int(fit.nobs),
    }


# --------------------------------------------------------------------------- logging

def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def _append_csv(path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def log_trial(res: BacktestResult, family: str, cost_mult: float, note: str = "") -> None:
    params = json.dumps(res.params, sort_keys=True, default=str)
    row = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "family": family,
        "name": res.name,
        "period": res.period,
        "cost_mult": cost_mult,
        "params_hash": hashlib.sha1(params.encode()).hexdigest()[:10],
        "params": params,
        "sharpe": round(res.stats.get("sharpe", float("nan")), 4),
        "ann_return": round(res.stats.get("ann_return", float("nan")), 5),
        "max_drawdown": round(res.stats.get("max_drawdown", float("nan")), 4),
        "turnover_per_year": round(res.stats.get("turnover_per_year", float("nan")), 2),
        "n_days": len(res.returns),
        "git": _git_head(),
        "note": note,
    }
    _append_csv(C.TRIALS_LOG, row)
    if res.period in ("OOS", "FULL"):
        _append_csv(C.OOS_LOG, row)


def trial_count(family: str | None = None, period: str = "IS", cost_mult: float = 1.0) -> tuple[int, float]:
    """Number of distinct in-sample variants tried (by params hash) and the variance of their annual Sharpe."""
    if not C.TRIALS_LOG.exists():
        return 1, 0.0
    t = pd.read_csv(C.TRIALS_LOG)
    t = t[(t["period"] == period) & (t["cost_mult"] == cost_mult)]
    if family is not None:
        t = t[t["family"] == family]
    t = t.drop_duplicates(subset=["family", "name", "params_hash"], keep="last")
    n = max(len(t), 1)
    var = float(t["sharpe"].var(ddof=1)) if len(t) > 1 else 0.0
    return n, var


# --------------------------------------------------------------------------- one-call API

def run_backtest(
    name: str,
    family: str,
    w_dec: pd.DataFrame,
    ohlc: dict[str, pd.DataFrame],
    rf: pd.Series,
    period: str = "IS",
    params: dict | None = None,
    exec: str = "next_open",
    cost_mult: float = 1.0,
    log: bool = True,
    note: str = "",
) -> BacktestResult:
    start, end = eval_window(period)
    net, gross, turnover, held, costs = simulate(w_dec, ohlc, rf, exec=exec, cost_mult=cost_mult)
    sl = slice(pd.Timestamp(start), pd.Timestamp(end))
    net, gross, turnover, held, costs = net.loc[sl], gross.loc[sl], turnover.loc[sl], held.loc[sl], costs.loc[sl]
    # drop the warm-up before the strategy first takes a position
    live = held.abs().sum(axis=1) > 0
    if period == "IS" and live.any():
        first = live.idxmax()
        net, gross, turnover, held, costs = (s.loc[first:] for s in (net, gross, turnover, held, costs))
    excess = net - rf.reindex(net.index).ffill().fillna(0.0)
    res = BacktestResult(name, period, net, gross, excess, turnover, held, costs, dict(params or {}))
    res.params.update({"exec": exec, "cost_mult": cost_mult})
    res.stats = perf_stats(net, excess, turnover)
    if log:
        log_trial(res, family, cost_mult, note)
    return res
