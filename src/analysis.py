"""Robustness, factor, capacity and plotting helpers shared by all strategies."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as C
from . import engine as E


# --------------------------------------------------------------------------- factors

def factor_panel(period: str = "IS") -> pd.DataFrame:
    """Daily factor returns: Fama-French 5 + momentum, plus a Treasury duration factor
    (IEF minus T-bill) and a commodity factor (DBC minus T-bill) because our strategies
    trade across asset classes."""
    ff = E.load_series("ff_daily.parquet", period)
    ohlc = E.load_ohlc(["IEF", "DBC"], period)
    rf = E.load_rf(period)
    r = ohlc["close"].pct_change(fill_method=None)
    out = ff[["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"]].copy()
    out["Bond"] = (r["IEF"] - rf.reindex(r.index).ffill()).reindex(out.index)
    out["Cmdty"] = (r["DBC"] - rf.reindex(r.index).ffill()).reindex(out.index)
    return out


def factor_table(res: E.BacktestResult, period: str | None = None) -> dict:
    fac = factor_panel(period or res.period)
    return E.factor_regression(res.excess, fac)


# --------------------------------------------------------------------------- robustness

def subperiod_table(res: E.BacktestResult, splits: list[str] | None = None) -> pd.DataFrame:
    splits = splits or ["2005-01-01", "2009-01-01", "2013-01-01", "2017-01-01", "2021-01-01", "2030-01-01"]
    rows = []
    for a, b in zip(splits[:-1], splits[1:]):
        sl = slice(pd.Timestamp(a), pd.Timestamp(b) - pd.Timedelta(days=1))
        net, ex, to = res.returns.loc[sl], res.excess.loc[sl], res.turnover.loc[sl]
        if len(net) < 60:
            continue
        s = E.perf_stats(net, ex, to)
        rows.append({"period": f"{a[:4]}-{int(b[:4]) - 1}", "ann_return": s["ann_return"], "ann_vol": s["ann_vol"],
                     "sharpe": s["sharpe"], "max_drawdown": s["max_drawdown"]})
    return pd.DataFrame(rows)


def pnl_concentration(res: E.BacktestResult) -> dict:
    """Share of cumulative (arithmetic) excess P&L from the single best year and best 1% of days."""
    ex = res.excess
    total = ex.sum()
    by_year = ex.groupby(ex.index.year).sum()
    top_days = ex.sort_values(ascending=False)
    k = max(1, int(len(ex) * 0.01))
    return {
        "best_year": int(by_year.idxmax()),
        "best_year_share": float(by_year.max() / total) if total > 0 else float("nan"),
        "top1pct_days_share": float(top_days.iloc[:k].sum() / total) if total > 0 else float("nan"),
        "ex_best_year_sharpe": float(
            ex[ex.index.year != by_year.idxmax()].mean() / ex[ex.index.year != by_year.idxmax()].std() * math.sqrt(C.TRADING_DAYS)
        ),
    }


def bootstrap_sharpe_ci(excess: pd.Series, block: int = 21, n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    """Stationary-block bootstrap 90% interval for the annualized Sharpe ratio."""
    x = excess.dropna().to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    nb = int(math.ceil(n / block))
    for i in range(n_boot):
        starts = rng.integers(0, n - block, size=nb)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        s = x[idx]
        out[i] = s.mean() / s.std() * math.sqrt(C.TRADING_DAYS) if s.std() > 0 else 0.0
    return float(np.quantile(out, 0.05)), float(np.quantile(out, 0.95))


# --------------------------------------------------------------------------- capacity

def capacity_curve(
    res: E.BacktestResult,
    ohlc: dict[str, pd.DataFrame],
    aum_grid: list[float] | None = None,
    impact_coef: float = 1.0,
    adv_window: int = 63,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Square-root impact model: cost_bps(trade) = fixed + impact_coef * sigma_daily * sqrt(trade$/ADV$) * 1e4.

    Uses the strategy's actual daily trades (|delta w| x AUM) against each ETF's trailing
    63-day median dollar volume and 63-day daily volatility. Returns the net Sharpe at each AUM.
    """
    aum_grid = aum_grid or [1e3, 1e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10]
    close, vol = ohlc["close"], ohlc["volume"]
    tickers = list(res.weights.columns)
    dollar_vol = (close[tickers] * vol[tickers]).rolling(adv_window, min_periods=20).median()
    sigma = close[tickers].pct_change(fill_method=None).rolling(adv_window, min_periods=20).std()
    idx = res.weights.index
    trades = res.weights.diff().abs().fillna(res.weights.abs())   # approx. |delta w| per name
    dollar_vol, sigma = dollar_vol.reindex(idx).ffill(), sigma.reindex(idx).ffill()
    fixed = pd.Series({t: C.cost_bps(t) for t in tickers}) / 1e4
    gross_ex = res.excess + res.costs                             # excess return before costs
    if start is not None or end is not None:                      # e.g. capacity at today's liquidity
        keep = (idx >= pd.Timestamp(start or idx[0])) & (idx <= pd.Timestamp(end or idx[-1]))
        trades, dollar_vol, sigma, gross_ex = trades[keep], dollar_vol[keep], sigma[keep], gross_ex[keep]
    rows = []
    for aum in aum_grid:
        part = (trades * aum / dollar_vol).clip(lower=0)
        impact = impact_coef * sigma * np.sqrt(part)
        cost = (trades * (impact + fixed)).sum(axis=1)
        net_ex = gross_ex - cost
        sr = float(net_ex.mean() / net_ex.std() * math.sqrt(C.TRADING_DAYS))
        max_part = float((part.where(trades > 0)).max().max())
        rows.append({"aum": aum, "net_sharpe": sr, "cost_drag_ann": float(cost.mean() * C.TRADING_DAYS),
                     "max_participation": max_part})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- plots

def plot_equity(results: dict[str, E.BacktestResult], path, title: str = "", log: bool = True) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=150)
    for name, r in results.items():
        ax.plot((1 + r.returns).cumprod(), label=name, lw=1.1)
    if log:
        ax.set_yscale("log")
    ax.axvline(pd.Timestamp(C.OOS_START), color="grey", ls="--", lw=0.8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
