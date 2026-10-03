"""Run a strategy inside the organisers' starter kit (gqh-webull-backtrader-starter) on our cached data.

Mirrors the kit's examples/backtest/main.py and imports its code at runtime from GQH_STARTER_KIT (nothing in the
kit is modified or copied; the Webull SDK, which needs credentials, is stubbed when it is not installed): the
kit's strategy loader and parameter parser, the FixedSize(stake=10) sizer, the analyzers DrawDown "drawdown",
TradeAnalyzer "trades", SharpeRatio "sharpe" (timeframe of data0, riskfreerate=0, annualize=True) and
RecorderAnalyzer "recorder", cerebro.run(runonce=False), _compute_metrics, _print_results and the report
renderers. The only substitutions:

1. data: a PandasData feed per symbol from our cache (ETFs from etf_daily, CME futures total-return indices F_*
   from futures_daily, both through engine.load_ohlc) instead of WebullData;
2. costs (--costs): none (the kit's default), guide (setcommission 0.0005 + set_slippage_perc 0.0005 per side,
   docs/USAGE_EN.md) or project (the engine's one-way bps per instrument as commission). With costs on, what
   backtrader does not see is debited from cash: futures rolls, short-ETF borrow and the guide slippage it
   misapplies to Close, cheat-on-close and open fills (ExtraCosts; --no-extra-costs turns it off);
3. --margin: a margin account (every commission scheme with leverage 10) for books above 1x gross;
4. cash: 1e9 for our weight-based strategies, because the kit's 100,000 in whole shares distorts target weights
   (one $556 SPY share is 0.56 % of NAV); the kit's 100,000 for its own examples; --cash overrides.

It adds a daily TimeReturn, the net weight held each day, an order log and the traded notional, and writes
per-window metrics
(in-sample to 2024-10-02, out-of-sample 2024-10-03..2026-10-02, full run) under both Sharpe definitions: the
kit's (daily total returns, rf = 0, population std) and ours (daily excess over the T-bill, sample std), with
excess = backtrader return - net weight x T-bill because backtrader pays no interest on cash.

Only the period "FWD" is read (all cached data through 2026-10-02): the out-of-sample lock is never released and
nothing is appended to results/trials.csv.

Run from the repo root:
    GQH_DATA_DIR=<cache> GQH_STARTER_KIT=<kit root> python validation/starter_kit/run_in_starter.py \\
        --strategy dual_ma --symbols SPY --start 2005-01-03 --end 2024-10-02
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import time
import types
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C        # noqa: E402
from src import engine as E        # noqa: E402
from src import forward2 as F2     # noqa: E402

OUR_STRATEGIES = HERE / "strategies"
TD = C.TRADING_DAYS
KIT_CASH = 100000.0           # examples/backtest/main.py
WEIGHT_CASH = 1e9             # whole-share rounding negligible at any weight
MARGIN_LEVERAGE = 10.0        # as validation/backtrader_crosscheck.py
GUIDE_COMMISSION = 0.0005     # docs/USAGE_EN.md: 5 bp commission + 5 bp slippage per side
GUIDE_SLIPPAGE = 0.0005
SESSION_CLOSE_ET = pd.Timedelta(hours=16)
REFUSED = ("Margin", "Rejected")


# --------------------------------------------------------------------------- the kit

_SDK = {"webull.core.client": ("ApiClient",), "webull.data.data_client": ("DataClient",),
        "webull.data.common.category": ("Category",), "webull.data.common.timespan": ("Timespan",),
        "webull.trade.trade_client": ("TradeClient",)}


class _NoSDK:
    """Stands in for a Webull SDK class: bars come from our cache, so it is never instantiated."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("the Webull SDK is not installed; run_in_starter.py feeds cached data instead")


def _stub_module(name: str) -> types.ModuleType:
    if name not in sys.modules:
        mod = types.ModuleType(name)
        mod.__path__ = []
        sys.modules[name] = mod
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(_stub_module(parent), child, mod)
    return sys.modules[name]


def _stub_missing_dependencies() -> None:
    """Placeholders for the SDK names the kit imports at module level (and python-dotenv if absent)."""
    if "webull" not in sys.modules and importlib.util.find_spec("webull") is None:
        for name, classes in _SDK.items():
            mod = _stub_module(name)
            for cls in classes:
                setattr(mod, cls, type(cls, (_NoSDK,), {}))
    if "dotenv" not in sys.modules and importlib.util.find_spec("dotenv") is None:
        _stub_module("dotenv").load_dotenv = lambda *a, **k: False


def load_kit(root: Path) -> types.ModuleType:
    """Import the kit's examples/backtest/main.py by path; it puts the kit root and examples/strategies on
    sys.path itself and imports webull_bt (feed, broker, visualize, visualize_lwc)."""
    main_py = root / "examples" / "backtest" / "main.py"
    if not main_py.exists():
        raise SystemExit(f"starter kit not found at {str(root)!r}: pass --kit or set GQH_STARTER_KIT")
    _stub_missing_dependencies()
    sys.dont_write_bytecode = True               # leave the kit folder exactly as unzipped (no __pycache__)
    spec = importlib.util.spec_from_file_location("kit_backtest_main", main_py)
    kit = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kit
    spec.loader.exec_module(kit)
    return kit


def module_value(module, name: str, params: dict, default=None):
    """A strategy module's optional attribute: a constant, or a callable of the parsed --params."""
    value = getattr(module, name, default)
    return value(params) if callable(value) and not isinstance(value, type) else value


# --------------------------------------------------------------------------- data

class CachedData(bt.feeds.PandasData):
    """Our cached daily OHLCV, plus the ``trading_session`` line of the kit's WebullData (daily bars carry 0) and
    a ``valid`` line: 1 on a session with a real print, 0 on a padded one (before the series starts, or a missing
    print carried forward)."""
    lines = ("trading_session", "valid")
    params = (("trading_session", -1), ("valid", -1), ("openinterest", None))


def session_close_utc(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Session dates -> 16:00 New York as naive UTC (backtrader's convention), so the kit's to_market_tz shows each
    bar at its own close; a bare date would display as 20:00 of the previous day."""
    return (idx + SESSION_CLOSE_ET).tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)


def make_feed(ohlc: dict, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> CachedData:
    """One feed on the engine's NYSE calendar (as backtrader_crosscheck.make_feed): days before a series' first
    print carry its first close and missing prints the last close, i.e. zero return, with valid = 0."""
    raw = ohlc["close"][ticker]
    close = raw.ffill().bfill()
    df = pd.DataFrame({c: ohlc[c][ticker].fillna(close) for c in ("open", "high", "low")})
    df["close"] = close
    df["volume"] = ohlc["volume"][ticker].fillna(0.0)
    df["trading_session"] = 0.0
    df["valid"] = raw.notna().astype(float)
    df = df.loc[start:end]
    df.index = session_close_utc(df.index)
    return CachedData(dataname=df)


# --------------------------------------------------------------------------- costs

def project_cost_bps(ticker: str) -> float:
    """The engine's one-way cost: forward2.S1_COST_BPS for the CME futures it lists, else config.cost_bps
    (3 / 5 / 10 bp ETF tiers, 1.5 bp other futures)."""
    return F2.S1_COST_BPS.get(ticker, C.cost_bps(ticker))


def configure_costs(broker, symbols: list[str], costs: str, margin: bool,
                    overrides: dict[str, float]) -> dict[str, float]:
    """Commission (and the guide's slippage) on the broker; returns each symbol's all-in one-way cost in bp,
    the rate ExtraCosts charges per futures roll leg."""
    lev = MARGIN_LEVERAGE if margin else 1.0
    if costs == "guide":
        broker.setcommission(commission=GUIDE_COMMISSION, leverage=lev)
        broker.set_slippage_perc(GUIDE_SLIPPAGE)
        return {t: (GUIDE_COMMISSION + GUIDE_SLIPPAGE) * 1e4 for t in symbols}
    bps = {t: (overrides.get(t, project_cost_bps(t)) if costs == "project" else 0.0) for t in symbols}
    if costs == "project" or margin:
        for t in symbols:
            broker.setcommission(commission=bps[t] / 1e4, leverage=lev, name=t)
    return bps


# --------------------------------------------------------------------------- analyzers

class NetWeight(bt.Analyzer):
    """Net and gross position value / NAV held over each return day t (close t-1 -> close t): positions after the
    fills of bar t less the bt.Order.Close fills at the close of t (those are held from t+1), valued at the close
    of t-1 over the NAV at the close of t-1."""

    def start(self):
        self.rets = OrderedDict()
        self._close_fills = defaultdict(float)
        self._prev = None

    def notify_order(self, order):
        if order.status == order.Completed and order.exectype == bt.Order.Close:
            self._close_fills[order.data] += order.executed.size

    def next(self):
        s = self.strategy
        net = gross = 0.0
        if self._prev is not None:
            nav, px = self._prev
            for d, p in zip(s.datas, px):
                v = (s.getposition(d).size - self._close_fills.get(d, 0.0)) * p
                net, gross = net + v / nav, gross + abs(v) / nav
        self.rets[s.datas[0].datetime.date(0)] = (net, gross)
        self._close_fills.clear()
        self._prev = (s.broker.getvalue(), [d.close[0] for d in s.datas])

    def get_analysis(self):
        return self.rets


class OrderLog(bt.Analyzer):
    """Each order's creation date and last status (orders still queued when the data ends included), and the
    notional traded per fill session."""

    def start(self):
        self.rets = {}
        self.traded = defaultdict(float)

    def notify_order(self, order):
        self.rets[order.ref] = (bt.num2date(order.created.dt).date(), order.getstatusname())
        if order.status == order.Completed:
            self.traded[bt.num2date(order.executed.dt).date()] += abs(order.executed.size * order.executed.price)

    def stop(self):
        broker = self.strategy.broker
        for o in itertools.chain(broker.submitted, broker.pending):
            if o.ref not in self.rets or o.alive():
                self.rets[o.ref] = (bt.num2date(o.created.dt).date(), o.getstatusname())

    def get_analysis(self):
        return self.rets


class ExtraCosts(bt.Analyzer):
    """Costs the engine charges that backtrader does not see, debited from cash with broker.add_cash:

    * a futures roll is one extra round trip of the position held into the roll session (one-way cost per leg);
    * short ETF positions pay config.SHORT_BORROW_BPS_PER_YEAR;
    * under the guide's costs, what set_slippage_perc gets wrong: bt.Order.Close fills are never slipped, so their
      5 bp is debited; cheat-on-close fills slip the close of t-1 but cap it at the high/low of bar t (a price
      that did not exist at that close), so that charge is replaced by 5 bp of the close of t-1; market orders
      filled at the open are slipped but capped at that bar's high/low, so the shortfall to 5 bp of the open is
      debited.

    After strategy.next() of bar t-1 the position held over return day t is known: current positions plus pending
    orders that fill before the close of t (market orders, cheat-on-close or not). Its roll and borrow cost is
    debited then and lands in the NAV of bar t, the day it belongs to; the roll flag of session t is read at t-1
    (roll dates follow the published contract calendar). A slippage debit is applied when the fill is notified,
    before the bar's NAV reaches the analyzers (broker._get_value() books the cash at once), so it lands on the
    fill's own bar."""
    params = (("roll_next", None), ("side_bps", None), ("borrow_daily", None), ("slip", 0.0))

    def start(self):
        self.rets = OrderedDict()            # date -> debits landing in that bar's NAV / NAV
        self.totals = {"roll": 0.0, "borrow": 0.0, "close_fill_slippage": 0.0, "coc_slippage_correction": 0.0,
                       "open_fill_slippage_topup": 0.0}
        self._queued = 0.0                   # debited at the previous bar, lands at this one
        self._now = 0.0                      # slippage booked at this bar

    def notify_order(self, order):
        if not self.p.slip or order.status != order.Completed:
            return
        size, price = order.executed.size, order.executed.price
        broker = self.strategy.broker
        if order.exectype == bt.Order.Close:
            x = abs(size) * price * self.p.slip
            self.totals["close_fill_slippage"] += x
        elif order.exectype == bt.Order.Market and broker.p.coc and order.info.get("coc", True):
            ref = order.created.pclose
            x = abs(size) * ref * self.p.slip - (price - ref) * size     # intended minus what the fill charged
            self.totals["coc_slippage_correction"] += x
        elif order.exectype == bt.Order.Market:                          # filled at this bar's open
            ref = order.data.open[0]
            x = abs(size) * ref * self.p.slip - (price - ref) * size     # > 0 only where the high/low capped it
            if x <= 1e-9 * abs(size) * ref:
                return
            self.totals["open_fill_slippage_topup"] += x
        else:
            return
        broker.add_cash(-x)
        broker._get_value()
        self._now += x

    def next(self):
        s, broker = self.strategy, self.strategy.broker
        pending = defaultdict(float)
        for o in itertools.chain(broker.submitted, broker.pending):
            if o.alive() and o.exectype != bt.Order.Close:
                pending[o.data] += o.size - o.executed.size
        day = s.datas[0].datetime.date(0)
        flags = self.p.roll_next.get(day)
        roll = borrow = 0.0
        for i, d in enumerate(s.datas):
            v = (s.getposition(d).size + pending.get(d, 0.0)) * d.close[0]
            if flags is not None and flags[i]:
                roll += 2.0 * abs(v) * self.p.side_bps[i] / 1e4
            if v < 0:
                borrow -= v * self.p.borrow_daily[i]
        self.rets[day] = (self._queued + self._now) / broker.getvalue()
        if roll + borrow:
            broker.add_cash(-(roll + borrow))
        self.totals["roll"] += roll
        self.totals["borrow"] += borrow
        self._queued, self._now = roll + borrow, 0.0

    def get_analysis(self):
        return self.rets


# --------------------------------------------------------------------------- statistics

def ann_return(r: pd.Series) -> float:
    """Compound annual growth over the sample, 252 return days a year (engine.perf_stats)."""
    return float((1 + r).prod() ** (TD / len(r)) - 1)


def kit_sharpe(r: pd.Series) -> float:
    """The kit's SharpeRatio(timeframe=Days, riskfreerate=0, annualize=True): mean / population std of daily
    total returns x sqrt(252)."""
    sd = r.std(ddof=0)
    return float(r.mean() / sd * math.sqrt(TD)) if sd > 0 else float("nan")


def our_sharpe(ex: pd.Series) -> float:
    """Ours: mean / sample std of daily excess returns over the T-bill x sqrt(252)."""
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * math.sqrt(TD)) if sd > 0 else float("nan")


def agreement(a: pd.Series, b: pd.Series) -> dict:
    """How closely two daily return series agree (a minus b)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(j) < 2:
        return {"n_days": int(len(j))}
    d = j["a"] - j["b"]
    return {"corr": float(j["a"].corr(j["b"])), "mean_abs_diff_bp": float(d.abs().mean() * 1e4),
            "max_abs_diff_bp": float(d.abs().max() * 1e4), "tracking_diff_ann": float(d.mean() * TD),
            "n_days": int(len(d))}


def window_stats(daily: pd.DataFrame, orders: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> dict | None:
    """Metrics of the run restricted to [lo, hi]; orders are counted by creation date."""
    w = daily.loc[lo:hi]
    if len(w) < 2:
        return None
    r = w["ret"]
    made = orders[(orders["created"] >= w.index[0]) & (orders["created"] <= w.index[-1])]
    out = {"start": str(w.index[0].date()), "end": str(w.index[-1].date()), "n_days": int(len(w)),
           "kit_sharpe_rf0": kit_sharpe(r), "our_sharpe_excess": our_sharpe(w["excess"]),
           "sharpe_total_sample_std": our_sharpe(r), "ann_return": ann_return(r),
           "ann_vol": float(r.std(ddof=1) * math.sqrt(TD)), "max_drawdown": E.max_drawdown(r),
           "mean_tbill_ann": float(w["rf"].mean() * TD), "mean_net_weight": float(w["net_weight"].mean()),
           "mean_gross_weight": float(w["gross_weight"].mean()), "max_gross_weight": float(w["gross_weight"].max()),
           "turnover_per_year": float(w["turnover"].mean() * TD),
           "orders": int(len(made)), "refused_orders": int(made["status"].isin(REFUSED).sum()),
           "canceled_orders": int((made["status"] == "Canceled").sum())}
    if "extra_cost" in w:
        out["extra_costs_ann"] = float(w["extra_cost"].mean() * TD)
    return out


def windows(daily: pd.DataFrame, is_start: pd.Timestamp) -> dict[str, tuple]:
    return {"in_sample": (is_start, pd.Timestamp(C.IS_END)),
            "out_of_sample": (pd.Timestamp(C.OOS_START), pd.Timestamp(C.OOS_END)),
            "full": (daily.index[0], daily.index[-1])}


def reference_agreement(daily: pd.DataFrame, path: str, column: str | None, rf_mode: str,
                        wins: dict[str, tuple]) -> dict:
    """Agreement with a reference daily return series from our engine. rf_mode "tbill": the reference's cash earns
    the T-bill, so excess returns are compared (ours: backtrader return - net weight x T-bill); "zero": the
    reference was simulated with rf = 0 and is compared with the raw backtrader returns."""
    ref = pd.read_csv(path, index_col=0, parse_dates=True)
    col = column or ref.columns[0]
    ref = ref[col].astype(float)
    ref.index = ref.index.normalize()
    rf = E.load_rf("FWD").reindex(ref.index).ffill().fillna(0.0)
    if rf_mode == "tbill":
        ours, theirs, theirs_total = daily["excess"], ref - rf, ref
    else:
        ours, theirs, theirs_total = daily["ret"], ref, ref
    out = {"path": Path(path).name, "column": col, "rf_mode": rf_mode, "windows": {}}
    for name, (lo, hi) in wins.items():
        t = theirs.loc[lo:hi].reindex(daily.loc[lo:hi].index).dropna()
        if len(t) < 2:
            out["windows"][name] = None
            continue
        res = {"agreement_bt_minus_reference": agreement(ours.loc[lo:hi], t),
               "reference_kit_sharpe_rf0": kit_sharpe(theirs_total.loc[t.index]),
               "reference_ann_return": ann_return(theirs_total.loc[t.index]),
               "reference_max_drawdown": E.max_drawdown(theirs_total.loc[t.index])}
        if rf_mode == "tbill":
            res["reference_our_sharpe_excess"] = our_sharpe(t)
        out["windows"][name] = res
    return out


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
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# --------------------------------------------------------------------------- run

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategy", default="dual_ma",
                    help="module name in validation/starter_kit/strategies/ or the kit's examples/strategies/")
    ap.add_argument("--symbols", default="", help="comma list (default: the module's SYMBOLS)")
    ap.add_argument("--start", default=None, help="first session fed (default: first print of any symbol)")
    ap.add_argument("--end", default=None, help="last session fed (default: last cached session)")
    ap.add_argument("--costs", choices=("none", "guide", "project"), default="none")
    ap.add_argument("--cost-bps", default="", help="project-cost overrides, e.g. ES16=1,ZN16=1")
    ap.add_argument("--no-extra-costs", action="store_true",
                    help="literal kit setters only: no debit for futures rolls, short-ETF borrow or the "
                         "guide-slippage corrections")
    ap.add_argument("--margin", action="store_true", help="margin account: leverage-10 commission schemes")
    ap.add_argument("--cash", type=float, default=None)
    ap.add_argument("--params", default="", help="strategy params, k=v,... (the kit's WEBULL_STRATEGY_PARAMS)")
    ap.add_argument("--is-start", default=None, help="first day of the in-sample window (default: module IS_START "
                                                     "or the first session)")
    ap.add_argument("--report", default=None, help="HTML report path (the kit's renderer); off when omitted")
    ap.add_argument("--report-engine", choices=("plotly", "lwc"), default=None,
                    help="default: plotly when installed, else lwc")
    ap.add_argument("--json", default=None, help="metrics JSON path")
    ap.add_argument("--series", default=None, help="daily CSV: return, net/gross weight, T-bill, excess")
    ap.add_argument("--reference", default=None, help="CSV of our engine's daily returns to compare with")
    ap.add_argument("--reference-column", default=None)
    ap.add_argument("--reference-rf", choices=("tbill", "zero"), default="tbill")
    ap.add_argument("--kit", default=os.environ.get("GQH_STARTER_KIT"), help="kit root (or env GQH_STARTER_KIT)")
    return ap.parse_args(argv)


def _parse_overrides(raw: str) -> dict[str, float]:
    out = {}
    for pair in filter(None, (p.strip() for p in raw.split(","))):
        key, _, value = pair.partition("=")
        out[key.strip().upper()] = float(value)
    return out


@dataclass
class Setup:
    """Everything run_backtest needs besides the kit."""
    strategy: str
    strategy_cls: type
    module: types.ModuleType
    params: dict
    ours: bool
    symbols: list[str]
    ohlc: dict
    start: pd.Timestamp
    end: pd.Timestamp
    is_start: pd.Timestamp
    cash: float
    costs: str
    cost_overrides: dict
    margin: bool
    exec: str | None
    extra: bool


def prepare(kit: types.ModuleType, args: argparse.Namespace) -> Setup:
    """Strategy and params through the kit's own loader and parser, then symbols, dates, cash and broker options."""
    os.environ["WEBULL_STRATEGY_PARAMS"] = args.params
    strategy_cls = kit._load_strategy_class(args.strategy)
    module = sys.modules[args.strategy]
    params = kit._parse_strategy_params()
    ours = Path(module.__file__).resolve().parent == OUR_STRATEGIES.resolve()
    symbols = [] if args.symbols else list(module_value(module, "SYMBOLS", params) or [])
    if not symbols:
        os.environ["WEBULL_SYMBOLS"] = args.symbols
        symbols = kit._parse_symbols()           # with neither, the kit falls back to AAPL (not cached)
    try:
        ohlc = E.load_ohlc(symbols, "FWD")
    except KeyError as exc:
        raise SystemExit(f"{exc}; pass --symbols with cached ETFs or F_* futures") from exc
    cal = ohlc["close"].index
    first_print = ohlc["close"].apply(pd.Series.first_valid_index).min()
    start = pd.Timestamp(args.start) if args.start else max(first_print, cal[0])
    end = min(pd.Timestamp(args.end), cal[-1]) if args.end else cal[-1]
    return Setup(strategy=args.strategy, strategy_cls=strategy_cls, module=module, params=params, ours=ours,
                 symbols=symbols, ohlc=ohlc, start=start, end=end,
                 is_start=pd.Timestamp(args.is_start or module_value(module, "IS_START", params) or start),
                 cash=args.cash if args.cash is not None else module_value(
                     module, "CASH", params, WEIGHT_CASH if ours else KIT_CASH),
                 costs=args.costs, cost_overrides=_parse_overrides(args.cost_bps),
                 margin=bool(args.margin or module_value(module, "NEEDS_MARGIN", params, False)),
                 exec=module_value(module, "EXEC", params),
                 extra=args.costs != "none" and not args.no_extra_costs)


def run_backtest(kit: types.ModuleType, su: Setup, recorder: type) -> tuple:
    """The kit's run_backtest() with cached feeds, the cost and margin options and the added analyzers."""
    cerebro = bt.Cerebro()
    for symbol in su.symbols:
        cerebro.adddata(make_feed(su.ohlc, symbol, su.start, su.end), name=symbol)
    kit.logger.info("[Backtest] strategy=%s symbols=%s params=%s", su.strategy, su.symbols, su.params)
    cerebro.addstrategy(su.strategy_cls, **su.params)
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)
    cerebro.broker.setcash(su.cash)
    side_bps = configure_costs(cerebro.broker, su.symbols, su.costs, su.margin, su.cost_overrides)
    if su.margin:
        cerebro.broker.set_checksubmit(False)    # as the cross-check: the margin scheme decides at the fill
    if su.exec == "coc":
        cerebro.broker.set_coc(True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    data0 = cerebro.datas[0]
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=data0._timeframe,
                        compression=data0._compression, riskfreerate=0.0, annualize=True)
    cerebro.addanalyzer(recorder, _name="recorder")
    # additions: daily returns, weights held, orders, costs backtrader does not see
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn", timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(NetWeight, _name="netweight")
    cerebro.addanalyzer(OrderLog, _name="orderlog")
    if su.extra:
        futures = [t for t in su.symbols if C.is_future(t)]
        roll = pd.DataFrame(0.0, index=su.ohlc["close"].index, columns=su.symbols)
        roll[futures] = su.ohlc["roll"][futures].fillna(0.0)
        roll = roll.shift(-1).fillna(0.0)                    # the flag of the next session
        cerebro.addanalyzer(
            ExtraCosts, _name="extra",
            roll_next={ts.date(): row for ts, row in zip(roll.index, roll.to_numpy())},
            side_bps=[side_bps[t] for t in su.symbols],
            borrow_daily=[0.0 if C.is_future(t) else C.SHORT_BORROW_BPS_PER_YEAR / 1e4 / TD for t in su.symbols],
            slip=GUIDE_SLIPPAGE if su.costs == "guide" else 0.0)

    starting_value = cerebro.broker.getvalue()
    kit.logger.info("[Backtest] starting cash: %.2f", starting_value)
    results = cerebro.run(runonce=False)
    strat = results[0]
    metrics = kit._compute_metrics(cerebro, strat, starting_value)
    kit._print_results(strat, metrics)
    return cerebro, strat, metrics, side_bps


def daily_frame(strat, extra: bool) -> pd.DataFrame:
    """Return, net / gross weight held, T-bill, excess (our definition), turnover (notional filled at bar t over
    the NAV at the close of t-1; futures rolls not included) and extra costs per return day."""
    tr = strat.analyzers.timereturn.get_analysis()
    daily = pd.DataFrame({"ret": list(tr.values())}, index=pd.to_datetime(list(tr.keys())).normalize())
    nw = strat.analyzers.netweight.get_analysis()
    daily = daily.join(pd.DataFrame(list(nw.values()), index=pd.to_datetime(list(nw.keys())),
                                    columns=["net_weight", "gross_weight"]))
    daily["rf"] = E.load_rf("FWD").reindex(daily.index).ffill().fillna(0.0)
    daily["excess"] = daily["ret"] - daily["net_weight"] * daily["rf"]
    traded = pd.Series(strat.analyzers.orderlog.traded, dtype=float)
    traded.index = pd.to_datetime(traded.index)
    nav = (1 + daily["ret"]).cumprod() * strat.broker.startingcash
    daily["turnover"] = traded.reindex(daily.index).fillna(0.0) / nav.shift(1).fillna(strat.broker.startingcash)
    if extra:
        xc = pd.Series(strat.analyzers.extra.get_analysis())
        daily["extra_cost"] = xc.set_axis(pd.to_datetime(xc.index)).reindex(daily.index).fillna(0.0)
    return daily


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if os.environ.get("GQH_OOS_UNLOCK"):
        raise SystemExit("unset GQH_OOS_UNLOCK: this harness only reads the FWD period")
    if not args.kit:
        raise SystemExit("pass --kit or set GQH_STARTER_KIT to the unzipped gqh-webull-backtrader-starter")
    t0 = time.time()
    kit = load_kit(Path(args.kit))
    sys.path.insert(0, str(OUR_STRATEGIES))           # ours shadow same-named kit examples
    from webull_bt.logging_utils import get_logger, setup_logging
    from webull_bt.visualize import RecorderAnalyzer
    setup_logging()
    log = get_logger("harness")
    su = prepare(kit, args)
    if su.margin and not su.ours:
        log.warning("[Harness] margin account with a kit strategy: order_target_percent sizes x%g", MARGIN_LEVERAGE)

    cerebro, strat, metrics, side_bps = run_backtest(kit, su, RecorderAnalyzer)
    if args.report:
        engine = args.report_engine or ("plotly" if importlib.util.find_spec("plotly") else "lwc")
        os.environ.update(WEBULL_VISUALIZE="true", WEBULL_VISUALIZE_ENGINE=engine,
                          WEBULL_VISUALIZE_OUTPUT=str(Path(args.report).resolve()))
    else:
        os.environ["WEBULL_VISUALIZE"] = "false"   # the kit would write into its own folder
    kit._maybe_render_report(strat, metrics, su.strategy, su.symbols)

    daily = daily_frame(strat, su.extra)
    ol = strat.analyzers.orderlog.get_analysis()
    orders = pd.DataFrame([(pd.Timestamp(d), s) for d, s in ol.values()], columns=["created", "status"])
    wins = windows(daily, su.is_start)
    broker = cerebro.broker
    out = {
        "meta": {
            "harness": "validation/starter_kit/run_in_starter.py", "strategy": su.strategy,
            "strategy_class": su.strategy_cls.__name__,
            "module": (Path(su.module.__file__).resolve().relative_to(ROOT).as_posix() if su.ours
                       else f"kit:examples/strategies/{Path(su.module.__file__).name}"),
            "params": su.params, "symbols": su.symbols, "start": str(su.start.date()), "end": str(su.end.date()),
            "is_start": str(su.is_start.date()), "costs": su.costs, "cost_bps_per_side": side_bps,
            "guide_slippage": GUIDE_SLIPPAGE if su.costs == "guide" else 0.0, "extra_costs": su.extra,
            "margin": su.margin, "leverage": MARGIN_LEVERAGE if su.margin else 1.0, "cash": su.cash,
            "exec": su.exec, "cheat_on_close": bool(broker.p.coc), "checksubmit": bool(broker.p.checksubmit),
            "backtrader_version": bt.__version__, "kit": Path(args.kit).name, "data_dir": C.DATA_DIR.name,
            "fwd_data_end": E.data_end("FWD"),
            "git_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip(),
            "sharpe_definitions": {
                "kit_sharpe_rf0": "mean / population std of daily backtrader returns x sqrt(252) (the kit's "
                                  "SharpeRatio analyzer); backtrader pays no interest on cash",
                "our_sharpe_excess": "mean / sample std of (backtrader return - net weight held x T-bill) x "
                                     "sqrt(252): our definition, cash at the T-bill, borrowed cash paying it"},
        },
        "kit_metrics": metrics,
        "windows": {k: window_stats(daily, orders, lo, hi) for k, (lo, hi) in wins.items()},
        "orders": {"total": int(len(orders)), "completed": int((orders["status"] == "Completed").sum()),
                   "refused": int(orders["status"].isin(REFUSED).sum()),
                   "by_status": orders["status"].value_counts().to_dict()},
    }
    if su.extra:
        out["extra_costs_money"] = strat.analyzers.extra.totals
    if args.reference:
        out["reference"] = reference_agreement(daily, args.reference, args.reference_column, args.reference_rf, wins)
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)

    for name, ws in out["windows"].items():
        if ws:
            log.info("[Harness] %-13s %s..%s  kit Sharpe %.4f  our Sharpe %.4f  ann %.2f%%  max DD %.2f%%  "
                     "orders %d (refused %d)", name, ws["start"], ws["end"], ws["kit_sharpe_rf0"],
                     ws["our_sharpe_excess"], 100 * ws["ann_return"], 100 * ws["max_drawdown"], ws["orders"],
                     ws["refused_orders"])
    if args.series:
        Path(args.series).parent.mkdir(parents=True, exist_ok=True)
        daily.rename_axis("date").to_csv(args.series)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(_clean(out), indent=2, default=str))
        log.info("[Harness] metrics written: %s", args.json)
    return out


if __name__ == "__main__":
    main()
