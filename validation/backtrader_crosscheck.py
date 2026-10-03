"""Cross-engine replication: our backtest engine (src/engine.py) against backtrader.

The organisers' starter kit (gqh-webull-backtrader-starter) runs strategies in backtrader on Webull
data. We have no Webull credentials, so every backtrader run here reads our own cached total-return
OHLCV through ``bt.feeds.PandasData``. The script answers two questions: does backtrader reproduce
our numbers once the conventions match, and how do the kit's defaults change the reported figures?

1. As shipped: the kit's DualMovingAverageStrategy (imported from the kit) on SPY 2005-01-03 ..
   2024-10-02 with the kit's cerebro setup (100k cash, FixedSize stake 10, no commission, the kit's
   DrawDown / TradeAnalyzer / SharpeRatio analyzers, runonce=False).
2. Matched dual MA: the same 5/20 SMA rule with a 99 % target in backtrader (fill at the next open)
   against the same rule in our engine (exec="next_open"), rf = 0, at 0 bp and at 5 bp per side.
3. Sleeve C replay: our sleeve C decision weights executed in backtrader with cheat-on-close fills one
   session after the decision (our next_close convention), 3 bp per side, rf = 0.
4. Submitted strategy replay: the F1 ensemble's daily ETF-level held weights executed in backtrader,
   against our engine on the same weights (rf = 0) and against the official strategy returns. The
   gap to the official returns is split exactly into its components.
5. Sharpe definitions: the kit's (daily total returns, rf = 0, population std) against ours (daily
   excess over the T-bill, sample std) for the submitted strategy, in and out of sample.
6. The hacker guide's example costs, 5 bp commission plus 5 bp slippage per side, on the section 4
   replay (one netted book). The slippage is charged as part of the commission (10 bp per side):
   backtrader's set_slippage_perc does not slip bt.Order.Close fills at all, and with cheat-on-close
   it caps the slip at the NEXT bar's high/low, which on gap days fills at a better price than the
   signal close. An earlier version used the setter and overstated the result (kit Sharpe 0.95 / 0.19).

Side effects: none outside results/validation/. We call ``engine.simulate`` and never
``engine.run_backtest``, so nothing is appended to results/trials.csv. Only the periods "IS" and "FWD"
are read; the out-of-sample lock is never released. FWD holds all cached data (through 2026-10-02).

Run from the repo root:
    GQH_DATA_DIR=<data cache> python validation/backtrader_crosscheck.py --kit <starter kit root>
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
import types
from collections import OrderedDict
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C            # noqa: E402
from src import engine as E            # noqa: E402
from src import ensemble as EN         # noqa: E402
from src import forward as F           # noqa: E402
from src.strategies import ifc_treasury  # noqa: E402

OUT = ROOT / "results" / "validation" / "backtrader_crosscheck.json"
TD = C.TRADING_DAYS
IS_WINDOW = (pd.Timestamp("2006-05-08"), pd.Timestamp(C.IS_END))      # first live day of the ensemble
OOS_WINDOW = (pd.Timestamp(C.OOS_START), pd.Timestamp(C.OOS_END))
REPLAY_CASH = 1e9          # large enough that whole-share rounding is negligible
MARGIN_LEVERAGE = 10.0     # backtrader margin account for the leveraged replays (see replay())


# --------------------------------------------------------------------------- statistics

def ann_return(r: pd.Series) -> float:
    """Compound annual growth rate over the sample (252 return days a year), as in engine.perf_stats."""
    return float((1 + r).prod() ** (TD / len(r)) - 1)


def sharpe(r: pd.Series) -> float:
    """Our convention: mean / sample std of daily returns, x sqrt(252). Pass excess returns for our
    official figure, total returns for a rf = 0 figure."""
    return float(r.mean() / r.std(ddof=1) * math.sqrt(TD))


def starter_sharpe(r: pd.Series) -> float:
    """The kit's figure: backtrader SharpeRatio(timeframe=Days, riskfreerate=0, annualize=True) is the
    mean over the population std (stddev_sample=False) of daily total returns, x sqrt(252)."""
    return float(r.mean() / r.std(ddof=0) * math.sqrt(TD))


def perf(r: pd.Series) -> dict:
    return {"ann_return": ann_return(r), "ann_vol": float(r.std(ddof=1) * math.sqrt(TD)),
            "sharpe_rf0": sharpe(r), "max_drawdown": E.max_drawdown(r), "n_days": int(len(r))}


def agreement(a: pd.Series, b: pd.Series) -> dict:
    """How closely two daily return series agree (a minus b)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    d = j["a"] - j["b"]
    return {"corr": float(j["a"].corr(j["b"])),
            "mean_abs_diff_bp": float(d.abs().mean() * 1e4),
            "max_abs_diff_bp": float(d.abs().max() * 1e4),
            "tracking_diff_ann": float(d.mean() * TD),           # annualised mean of a - b
            "tracking_error_ann": float(d.std(ddof=1) * math.sqrt(TD)),
            "days_diff_over_1bp": int((d.abs() > 1e-4).sum()),
            "n_days": int(len(d))}


def window(s: pd.Series, w: tuple) -> pd.Series:
    return s.loc[w[0]: w[1]]


# --------------------------------------------------------------------------- backtrader plumbing

class CachedData(bt.feeds.PandasData):
    """PandasData over our cached daily total-return OHLCV, plus the extra ``trading_session`` line of
    the kit's WebullData (the kit's DualMovingAverageStrategy reads it; daily bars carry code 0)."""
    lines = ("trading_session",)
    params = (("trading_session", -1), ("openinterest", None))


def make_feed(ohlc: dict, ticker: str) -> CachedData:
    """One backtrader feed on the engine's NYSE calendar. Days before a fund's first print carry its
    first close (zero return; every strategy holds 0 there), isolated missing prints the last close."""
    close = ohlc["close"][ticker].ffill().bfill()
    df = pd.DataFrame({c: ohlc[c][ticker].fillna(close) for c in ("open", "high", "low")})
    df["close"] = close
    df["volume"] = ohlc["volume"][ticker].fillna(0.0)
    df["trading_session"] = 0.0
    return CachedData(dataname=df[["open", "high", "low", "close", "volume", "trading_session"]])


def add_kit_analyzers(cerebro: bt.Cerebro) -> None:
    """The analyzers of the kit's examples/backtest/main.py (minus its plotting recorder), plus a
    daily TimeReturn so the returns can be compared day by day."""
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days,
                        compression=1, riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days)


class Exposure(bt.Analyzer):
    """Gross position value / NAV at each close, after that bar's fills."""

    def create_analysis(self):
        self.rets = OrderedDict()

    def next(self):
        s = self.strategy
        gross = sum(abs(s.getposition(d).size) * d.close[0] for d in s.datas)
        self.rets[s.datas[0].datetime.date(0)] = gross / s.broker.getvalue()


def series_from(analysis: dict) -> pd.Series:
    s = pd.Series(list(analysis.values()), index=pd.to_datetime(list(analysis.keys())))
    s.index = s.index.normalize()
    return s


def daily_returns(strat) -> pd.Series:
    return series_from(strat.analyzers.timereturn.get_analysis())


class FullyInvestedDualMA(bt.Strategy):
    """The kit's long-only 5/20 SMA crossover rule (one order in flight at a time), re-implemented
    with a 99 % of NAV target instead of a fixed 10-share stake."""
    params = dict(short_period=5, long_period=20, target=0.99)

    def __init__(self):
        fast = bt.ind.SMA(self.data, period=self.p.short_period)
        slow = bt.ind.SMA(self.data, period=self.p.long_period)
        self.cross = bt.ind.CrossOver(fast, slow)
        self.pending = None
        self.refused = []               # orders refused at the fill: (fill date, open / signal close - 1)

    def notify_order(self, order):
        if order.status in (order.Submitted, order.Accepted):
            return
        if order.status in (order.Margin, order.Rejected, order.Canceled):
            # notified on the bar the fill was attempted: open[0] is that bar's open
            self.refused.append((str(self.data.datetime.date(0)), self.data.open[0] / order.created.price - 1))
        self.pending = None

    def next(self):
        if self.pending is not None:
            return
        if not self.position:
            if self.cross[0] > 0:
                self.pending = self.order_target_percent(target=self.p.target)
        elif self.cross[0] < 0:
            self.pending = self.close()


class WeightReplay(bt.Strategy):
    """Rebalance every feed to a daily target weight of NAV at each close.

    ``targets.loc[d]`` is the weight to hold after the close of d. Orders are sized here in whole
    shares from the NAV and close of d; with cheat-on-close the broker fills them at that close."""
    params = dict(targets=None)

    def __init__(self):
        names = [d._name for d in self.datas]
        tg = self.p.targets
        self.rows = {ts.date(): row for ts, row in zip(tg.index, tg[names].to_numpy())}
        self.orders = 0
        self.failed = 0

    def notify_order(self, order):
        if order.status in (order.Margin, order.Rejected, order.Canceled):
            self.failed += 1

    def next(self):
        row = self.rows.get(self.datas[0].datetime.date(0))
        if row is None:
            return
        nav = self.broker.getvalue()
        for d, w in zip(self.datas, row):
            delta = round(w * nav / d.close[0]) - self.getposition(d).size
            if delta > 0:
                self.buy(data=d, size=delta)
            elif delta < 0:
                self.sell(data=d, size=-delta)
            self.orders += delta != 0


def replay(held: pd.DataFrame, ohlc: dict, cost_bps: dict, leverage: float = MARGIN_LEVERAGE,
           slip_perc: float = 0.0) -> tuple:
    """Execute daily held weights in backtrader with our next_close timing.

    ``held.loc[t]`` is the weight held over return day t (close t-1 -> close t). At the close of t-1
    the strategy orders ``held.loc[t]`` and cheat-on-close fills it at that close, so the position
    earns exactly return day t. Commission is ``cost_bps`` per side on traded notional.

    Leverage: ``broker.set_checksubmit(False)`` alone does not allow it in backtrader 1.9.78, because
    BackBroker._execute still nullifies any opening fill that would take cash below zero (the order
    ends as Margin). We therefore model a margin account: each commission scheme gets
    ``leverage=MARGIN_LEVERAGE``, so a long fill debits notional / leverage of cash; backtrader's NAV
    (cash + position value net of the unlevered part) stays exact, and orders are sized by our own
    code (not order_target_percent, whose getsize() would multiply the size by the leverage).
    """
    cerebro = bt.Cerebro(stdstats=False)
    for t in held.columns:
        cerebro.adddata(make_feed(ohlc, t), name=t)
    cerebro.broker.setcash(REPLAY_CASH)
    cerebro.broker.set_coc(True)
    cerebro.broker.set_checksubmit(False)
    for t in held.columns:
        cerebro.broker.setcommission(commission=cost_bps[t] / 1e4, leverage=leverage, name=t)
    if slip_perc:
        cerebro.broker.set_slippage_perc(slip_perc)          # the kit's setter: opens slipped too, capped at high/low
    cerebro.addstrategy(WeightReplay, targets=held.shift(-1).fillna(0.0))
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days,
                        compression=1, riskfreerate=0.0, annualize=True)
    strat = cerebro.run()[0]
    return daily_returns(strat), strat


def zero_rf(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(0.0, index=index)


# --------------------------------------------------------------------------- section 1

def load_kit_strategy(kit: Path):
    """Import the kit's DualMovingAverageStrategy unchanged.

    dual_ma.py imports webull_bt.logging_utils and webull_bt.timeutils; the package __init__ also pulls
    in the Webull SDK (feed, broker), which is not installed. Registering a bare package object for
    ``webull_bt`` lets Python load those two submodules from the kit without running the __init__."""
    if "webull_bt" not in sys.modules:
        pkg = types.ModuleType("webull_bt")
        pkg.__path__ = [str(kit / "webull_bt")]
        sys.modules["webull_bt"] = pkg
    for p in (kit, kit / "examples" / "strategies"):       # as the kit's main.py arranges sys.path
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    return importlib.import_module("dual_ma").STRATEGY_CLASS


def kit_metrics(cerebro: bt.Cerebro, strat, start_value: float) -> dict:
    """The summary the kit's main.py prints (same fields and formulas)."""
    final = cerebro.broker.getvalue()
    dd = strat.analyzers.drawdown.get_analysis()
    tr = strat.analyzers.trades.get_analysis()
    total = tr.get("total", {}).get("total", 0)
    won = tr.get("won", {}).get("total", 0)
    lost = tr.get("lost", {}).get("total", 0)
    return {"starting_cash": start_value, "final_value": final, "net_pnl": final - start_value,
            "net_pnl_pct": (final / start_value - 1) * 100,
            "max_drawdown_pct": dd.get("max", {}).get("drawdown", 0.0),
            "max_drawdown_money": dd.get("max", {}).get("moneydown", 0.0),
            "sharpe_ratio_annualized": strat.analyzers.sharpe.get_analysis().get("sharperatio"),
            "total_trades": total, "won": won, "lost": lost,
            "win_rate_pct": won / total * 100 if total else 0.0,
            "trades_net_pnl": tr.get("pnl", {}).get("net", {}).get("total", 0.0)}


def section1(kit: Path) -> dict:
    strategy = load_kit_strategy(kit)
    ohlc = E.load_ohlc(["SPY"], "IS")
    cerebro = bt.Cerebro()
    cerebro.adddata(make_feed(ohlc, "SPY"), name="SPY")
    cerebro.addstrategy(strategy)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcash(100000.0)
    add_kit_analyzers(cerebro)
    cerebro.addanalyzer(Exposure, _name="exposure")
    strat = cerebro.run(runonce=False)[0]
    r = daily_returns(strat)
    expo = series_from(strat.analyzers.exposure.get_analysis())
    held = expo[expo > 0]
    close = ohlc["close"]["SPY"]
    return {
        "setup": "kit DualMovingAverageStrategy (5/20 SMA), SPY 2005-01-03..2024-10-02, cash 100000, "
                 "FixedSize stake 10, default broker (no commission, fills at next open), runonce=False",
        "kit_metrics": kit_metrics(cerebro, strat, 100000.0),
        "capital_usage": {
            "share_of_days_in_market": float((expo > 0).mean()),
            "mean_position_over_nav_when_invested": float(held.mean()),
            "min_position_over_nav_when_invested": float(held.min()),
            "max_position_over_nav_when_invested": float(held.max()),
            "mean_position_over_nav_all_days": float(expo.mean()),
            "spy_adjusted_close_first_last": [float(close.iloc[0]), float(close.iloc[-1])],
        },
        "daily_return_stats": {**perf(r), "starter_sharpe_recomputed": starter_sharpe(r)},
    }


# --------------------------------------------------------------------------- section 2

def crossover_state(close: pd.Series, short: int = 5, long: int = 20) -> pd.Series:
    """1 while long, 0 while flat, with backtrader's CrossOver semantics: a cross up at t needs the
    last non-zero fast-slow difference before t to be negative and fast > slow at t (and vice versa);
    enter on a cross up while flat, exit on a cross down while long. Decided at the close."""
    d = close.rolling(short).mean() - close.rolling(long).mean()
    nzd = d.where(d != 0).ffill()                          # backtrader's NonZeroDifference
    up = ((nzd.shift(1) < 0) & (d > 0)).to_numpy()
    down = ((nzd.shift(1) > 0) & (d < 0)).to_numpy()
    state, s = np.zeros(len(close)), 0
    for i in range(len(close)):
        if s == 0 and up[i]:
            s = 1
        elif s == 1 and down[i]:
            s = 0
        state[i] = s
    return pd.Series(state, index=close.index)


def bt_dual_ma(ohlc: dict, commission_bps: float, cash: float) -> tuple:
    cerebro = bt.Cerebro()
    cerebro.adddata(make_feed(ohlc, "SPY"), name="SPY")
    cerebro.addstrategy(FullyInvestedDualMA)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission_bps / 1e4)
    add_kit_analyzers(cerebro)
    cerebro.addanalyzer(Exposure, _name="exposure")
    strat = cerebro.run()[0]
    return daily_returns(strat), strat


def section2() -> dict:
    ohlc = E.load_ohlc(["SPY"], "IS")
    close = ohlc["close"]["SPY"]
    state = crossover_state(close)
    w_dec = (0.99 * state).to_frame("SPY")
    rf0 = zero_rf(close.index)
    out = {"setup": "5/20 SMA golden/death cross on SPY 2005-01-03..2024-10-02; backtrader: "
                    "order_target_percent(0.99) on a golden cross while flat, close() on a death cross, "
                    "default fill at the next open, cash 100000; engine: decision weights 0.99/0 from the "
                    "same crossover, exec=next_open, rf = 0; Sharpe = mean/std(ddof=1) x sqrt(252), rf = 0",
           "spy_buy_and_hold_rf0": perf(close.pct_change().fillna(0.0))}
    for bps in (0.0, 5.0):
        r_bt, strat = bt_dual_ma(ohlc, bps, 100000.0)
        r_en = E.simulate(w_dec, ohlc, rf0, exec="next_open", cost_bps={"SPY": bps})[0]
        expo = series_from(strat.analyzers.exposure.get_analysis())
        tr = strat.analyzers.trades.get_analysis()
        # the engine fed backtrader's realised long/flat path (long at the close of d+1 <=> decision 0.99
        # at d): what is left is share drift between fills (backtrader holds a fixed share count, the
        # engine rebalances to 0.99 daily), whole shares and the cash buffer
        w_bt = (0.99 * (expo.reindex(close.index).shift(-1) > 0)).to_frame("SPY")
        r_en_bt = E.simulate(w_bt, ohlc, rf0, exec="next_open", cost_bps={"SPY": bps})[0]
        res = {
            "backtrader": {**perf(r_bt), "kit_sharpe_analyzer": strat.analyzers.sharpe.get_analysis()["sharperatio"],
                           "trades": int(tr.get("total", {}).get("total", 0)),
                           "entries_refused_for_cash": len(strat.refused),
                           "refused_fill_date_and_gap": [[d, g] for d, g in strat.refused],
                           "mean_position_over_nav_when_invested": float(expo[expo > 0].mean())},
            "engine": {**perf(r_en), "trades": int(((state == 1) & (state.shift(1) == 0)).sum())},
            "agreement_bt_minus_engine": agreement(r_bt, r_en),
            "diagnostic_engine_on_bt_positions": {**perf(r_en_bt), "agreement_bt_minus_this": agreement(r_bt, r_en_bt)},
        }
        if bps == 0.0:
            # same backtrader run with 1e9 cash: isolates whole-share rounding from the other effects
            r_big, strat_big = bt_dual_ma(ohlc, bps, REPLAY_CASH)
            res["diagnostic_bt_cash_1e9"] = {**perf(r_big), "entries_refused_for_cash": len(strat_big.refused),
                                             "agreement_vs_engine": agreement(r_big, r_en)}
        out[f"commission_{bps:g}bp"] = res
    return out


# --------------------------------------------------------------------------- section 3

def section3() -> dict:
    w_dec = ifc_treasury.decision_weights("FWD")
    ohlc = E.load_ohlc(["IEF"], "FWD")
    rf0 = zero_rf(w_dec.index)
    r_en, _, _, held, _ = E.simulate(w_dec, ohlc, rf0, exec="next_close")
    cost = {"IEF": C.cost_bps("IEF")}
    r_bt, strat = replay(held, ohlc, cost)
    r_chk, strat_chk = replay(held, ohlc, cost, leverage=1.0)
    hw = held["IEF"]
    return {
        "setup": "sleeve C decision weights (ifc_treasury.decision_weights('FWD')); engine exec=next_close, "
                 "rf = 0, 3 bp per side; backtrader: order at the close of d+1 for the decision of d, "
                 "cheat-on-close fill, 3 bp commission per side, cash 1e9, margin account (leverage 10)",
        "window": [str(r_en.index[0].date()), str(r_en.index[-1].date())],
        "position_days": int((hw > 0).sum()), "max_weight": float(hw.max()),
        "days_weight_over_1": int((hw > 1).sum()),
        "backtrader": {**perf(r_bt), "kit_sharpe_analyzer": strat.analyzers.sharpe.get_analysis()["sharperatio"],
                       "orders": int(strat.orders), "orders_refused": int(strat.failed)},
        "engine": perf(r_en),
        "agreement_bt_minus_engine": agreement(r_bt, r_en),
        "diagnostic_checksubmit_false_only": {
            "note": "same replay with set_checksubmit(False) but no margin scheme (leverage 1): fills that "
                    "would take cash below zero are refused at execution",
            "orders_refused": int(strat_chk.failed), **perf(r_chk),
            "agreement_vs_engine": agreement(r_chk, r_en)},
    }


# --------------------------------------------------------------------------- section 4 and 5

def submitted_strategy() -> dict:
    """Rebuild F1 (forward.py) and everything needed to replay and decompose it."""
    fr = F.stream_frames("F1", "FWD")
    labels = list(fr["ex"].columns)
    out = EN.combine(fr["ex"], fr["gr"], fr["rf"], labels, F.ENSEMBLE["scheme"], F.ENSEMBLE["vol_target"],
                     F.ENSEMBLE["brake"])
    idx = out["net"].index
    cal = fr["ex"].index
    scale = out["lam"][labels].mul(out["k"], axis=0)                 # applied scale = lam x k
    tickers = sorted(set().union(*(fr["held"][s].columns for s in labels)))
    ohlc = E.load_ohlc(tickers, "FWD")
    rf0 = zero_rf(cal)
    W = pd.DataFrame(0.0, index=cal, columns=tickers)                # ETF weights held over day t
    gross_risky, trade_cost, borrow = {}, {}, {}
    for s in labels:
        h = fr["held"][s]
        W = W.add(h.reindex(idx).mul(scale[s], axis=0).reindex(cal).fillna(0.0), fill_value=0.0)
        # stream re-simulated with rf = 0: gross = the risky leg only; costs are rf-independent
        cols = list(h.columns)
        _, g, _, hh, c = E.simulate(fr["w_dec"][s], E.load_ohlc(cols, "FWD"), rf0, exec=fr["exec"][s])
        b = (hh.clip(upper=0).abs() * C.SHORT_BORROW_BPS_PER_YEAR / 1e4 / TD).sum(axis=1)
        gross_risky[s], borrow[s], trade_cost[s] = g, b, c - b
    # overlay re-scaling costs exactly as ensemble.combine charges them
    S = scale.to_numpy()
    G = fr["gr"][labels].reindex(idx).to_numpy()
    prev_S = np.vstack([np.zeros(len(labels)), S[:-1]])
    prev_G = np.vstack([np.zeros(len(labels)), G[:-1]])
    cb = np.array([EN.STREAM_COST_BPS[s] for s in labels]) / 1e4
    overlay = pd.Series((np.abs(S - prev_S) * np.minimum(G, prev_G)) @ cb, index=idx)
    return {"fr": fr, "out": out, "labels": labels, "idx": idx, "scale": scale, "W": W, "ohlc": ohlc,
            "gross_risky": gross_risky, "trade_cost": trade_cost, "borrow": borrow, "overlay": overlay}


def section4_5(sub: dict) -> tuple[dict, dict]:
    fr, out, labels, idx, scale, W, ohlc = (sub[k] for k in ("fr", "out", "labels", "idx", "scale", "W", "ohlc"))
    rf = fr["rf"].reindex(idx)
    official_file = pd.read_csv(ROOT / "results" / "oos" / "fte_selected_full_returns.csv",
                                index_col=0, parse_dates=True)["net"]
    official = out["net"]
    repro = agreement(official, official_file)

    # our engine on the same held weights, rf = 0, per-ETF costs on the netted ETF trades, no borrow fee
    tickers = list(W.columns)
    cost = {t: C.cost_bps(t) for t in tickers}
    rf0 = zero_rf(W.index)
    net_s, gross_s, _, held_s, cost_s = E.simulate(W.shift(-2), ohlc, rf0, exec="next_close", cost_bps=cost)
    borrow_s = (held_s.clip(upper=0).abs() * C.SHORT_BORROW_BPS_PER_YEAR / 1e4 / TD).sum(axis=1)
    simple = (net_s + borrow_s).reindex(idx)
    simple_cost = (cost_s - borrow_s).reindex(idx)
    assert np.allclose(held_s.reindex(idx).to_numpy(), W.reindex(idx).to_numpy())

    r_bt, strat = replay(W, ohlc, cost)
    r_bt = r_bt.reindex(idx)

    # exact split of official - simple (all daily, on the ensemble's live days):
    #   official = rf (1 - sum W) + sum_s scale_s gross_s - sum_s scale_s (costs_s + borrow_s) - overlay
    #   simple   = sum_j W_j r_cc_j - netted ETF trading costs
    r_cc = ohlc["close"][tickers].ffill(limit=5).pct_change(fill_method=None).fillna(0.0).reindex(idx)
    sumW = W.reindex(idx).sum(axis=1)
    comp = pd.DataFrame(index=idx)
    comp["cash_interest"] = rf * (1 - sumW)
    comp["trend_next_open_fill"] = sum(
        scale[s] * (sub["gross_risky"][s].reindex(idx)
                    - (fr["held"][s].reindex(idx) * r_cc[list(fr["held"][s].columns)]).sum(axis=1))
        for s in labels)
    comp["stream_costs_vs_netted"] = simple_cost - sum(scale[s] * sub["trade_cost"][s].reindex(idx) for s in labels)
    comp["overlay_costs"] = -sub["overlay"]
    comp["borrow_fee"] = -sum(scale[s] * sub["borrow"][s].reindex(idx) for s in labels)
    resid = official - simple - comp.sum(axis=1)

    s4 = {"setup": "F1 = A + C + TSMOM, minvar_lw, 8 % vol target, no brake (src/forward.py). Held ETF "
                   "weights W_t = sum_s lam_s,t k_t held_s,t; backtrader orders W_t at the close of t-1 "
                   "(cheat-on-close), per-ETF commission 3 bp tier 1 / 5 bp tier 2 per side, cash 1e9, "
                   "margin account (leverage 10); engine 'simple' = sum_j W_j r_j - netted per-ETF costs "
                   "with rf = 0 and no borrow fee",
          "official_file_reproduced_by_forward_F1": repro,
          "replay_orders": int(strat.orders), "replay_orders_refused": int(strat.failed),
          "mean_net_etf_weight": float(sumW.mean()), "mean_gross_etf_weight": float(W.reindex(idx).abs().sum(axis=1).mean()),
          "periods": {}}
    for name, w in (("IS_2006-05-08_2024-10-02", IS_WINDOW), ("OOS_2024-10-03_2026-10-02", OOS_WINDOW)):
        b, sm, of = window(r_bt, w), window(simple, w), window(official, w)
        ex = of - window(rf, w)
        s4["periods"][name] = {
            "backtrader_replay": perf(b),
            "engine_simple": perf(sm),
            "official": {**perf(of), "sharpe_excess_official": sharpe(ex)},
            "bt_minus_engine_simple": agreement(b, sm),
            "bt_minus_official": agreement(b, of),
            "engine_simple_minus_official": agreement(sm, of),
            "official_minus_simple_ann": {**{k: float(window(comp[k], w).mean() * TD) for k in comp.columns},
                                          "total": float((of - sm).mean() * TD),
                                          "residual_max_abs": float(window(resid, w).abs().max())},
        }

    s5 = {"definition_starter": "mean / population std of daily TOTAL returns (rf = 0), x sqrt(252): "
                                "backtrader SharpeRatio(timeframe=Days, riskfreerate=0, annualize=True)",
          "definition_ours": "mean / sample std of daily returns in excess of the 3-month T-bill, x sqrt(252)",
          "check_kit_analyzer_vs_formula_on_replay": {
              "analyzer": strat.analyzers.sharpe.get_analysis()["sharperatio"],
              "formula_on_timereturn": starter_sharpe(daily_returns(strat))},
          "periods": {}}
    for name, w in (("IS_2006-05-08_2024-10-02", IS_WINDOW), ("OOS_2024-10-03_2026-10-02", OOS_WINDOW)):
        of, rfw = window(official, w), window(rf, w)
        s5["periods"][name] = {
            "official_sharpe_ours": sharpe(of - rfw),
            "official_sharpe_starter": starter_sharpe(of),
            "official_total_returns_sample_std": sharpe(of),
            "mean_tbill_ann": float(rfw.mean() * TD),
            "ann_vol": float(of.std(ddof=1) * math.sqrt(TD)),
            "backtrader_replay_sharpe_starter": starter_sharpe(window(r_bt, w)),
        }
    return s4, s5


def section6(sub: dict) -> dict:
    """F1 replayed in backtrader at the hacker guide's example costs (5 bp commission + 5 bp slippage),
    the slippage charged inside the commission (see the module docstring, section 6)."""
    W, ohlc, idx, fr = sub["W"], sub["ohlc"], sub["idx"], sub["fr"]
    rf = fr["rf"].reindex(idx)
    r_bt, strat = replay(W, ohlc, {t: 10.0 for t in W.columns})
    r_bt = r_bt.reindex(idx)
    Wi = W.reindex(idx)
    # our definition on the same book: pay the T-bill on net long notional (borrowed cash) and the borrow fee
    borrow = (Wi.clip(upper=0).abs() * C.SHORT_BORROW_BPS_PER_YEAR / 1e4 / TD).sum(axis=1)
    excess = r_bt - Wi.sum(axis=1) * rf - borrow
    out = {"setup": "section 4 replay, 5 bp commission + 5 bp slippage per side charged as setcommission(0.0010) "
                    "(backtrader does not slip cheat-on-close fills correctly); "
                    "5 bp + 5 bp per side on the netted ETF trades",
           "replay_orders": int(strat.orders), "replay_orders_refused": int(strat.failed), "periods": {}}
    for name, w in (("IS_2006-05-08_2024-10-02", IS_WINDOW), ("OOS_2024-10-03_2026-10-02", OOS_WINDOW)):
        b = window(r_bt, w)
        out["periods"][name] = {**perf(b), "sharpe_starter": starter_sharpe(b),
                                "sharpe_ours_excess_financed": sharpe(window(excess, w))}
    return out


# --------------------------------------------------------------------------- main

def _clean(x):
    """JSON-safe, rounded to 8 significant figures."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if not math.isfinite(x) else float(f"{x:.8g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kit", default=os.environ.get("GQH_STARTER_KIT"),
                    help="root of the unzipped gqh-webull-backtrader-starter (or env GQH_STARTER_KIT); "
                         "section 1 is skipped without it")
    args = ap.parse_args()
    if os.environ.get("GQH_OOS_UNLOCK"):
        raise SystemExit("unset GQH_OOS_UNLOCK: this script only needs the IS and FWD periods")
    t0 = time.time()
    res = {"meta": {
        "script": "validation/backtrader_crosscheck.py",
        "backtrader_version": bt.__version__, "pandas_version": pd.__version__,
        "data_dir": C.DATA_DIR.name, "fwd_data_end": E.data_end("FWD"),
        "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                   capture_output=True, text=True).stdout.strip(),
        "kit": Path(args.kit).name if args.kit else None}}
    kit = Path(args.kit) if args.kit else None
    if kit and (kit / "examples" / "strategies" / "dual_ma.py").exists():
        res["1_as_shipped"] = section1(kit)
    else:
        res["1_as_shipped"] = {"skipped": "starter kit not found; pass --kit"}
    res["2_matched_dual_ma"] = section2()
    res["3_sleeve_c_replay"] = section3()
    sub = submitted_strategy()
    res["4_submitted_replay"], res["5_sharpe_definitions"] = section4_5(sub)
    res["6_guide_example_costs"] = section6(sub)
    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(res), indent=2))
    print(f"wrote {OUT.relative_to(ROOT)} in {res['meta']['runtime_s']} s")


if __name__ == "__main__":
    main()
