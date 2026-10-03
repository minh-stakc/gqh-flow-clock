"""Shared base for strategies run through run_in_starter.py: trading to target weights of NAV.

Orders are sized here in whole shares from the NAV and close of the bar that submits them, never with
``order_target_percent``: on the margin account (commission schemes with leverage 10) backtrader's getsize()
multiplies the size by the leverage.

Execution conventions (the engine's, src/engine.py):

* ``next_open``: a market order submitted in next() of bar d fills at the open of d+1.
* ``next_close``: a ``bt.Order.Close`` submitted in next() of bar d fills at the close of d+1.
* ``coc``: a market order submitted in next() of bar d fills at the close of d (cheat-on-close; the broker
  needs ``set_coc(True)``). Exact weights at the fill: for replaying weights decided one session earlier.

Sizing at the close of d for a fill at the open or close of d+1 lets each weight drift by the instrument's move
relative to the book between sizing and fill; the engine assumes the exact target at the fill.
"""
from __future__ import annotations

import backtrader as bt

from src import calendar_utils as CU
from src import engine as E

EXEC_TYPES = {"next_open": bt.Order.Market, "next_close": bt.Order.Close, "coc": bt.Order.Market}


def planned_month_ends() -> set:
    """Last session of each complete month on the published NYSE schedule (the FWD sessions plus
    calendar_utils.UNSCHEDULED_CLOSURES), i.e. what was known at each close. When none of them is an unscheduled
    closure they equal the realised month ends (forward2._month_ends); otherwise a strategy could not have known at
    the close that its month had ended, so this raises."""
    real = E.trading_calendar("FWD")
    mo = CU.month_offsets(CU.planned_calendar(real))
    ends = mo.index[mo["off_own"] == 0]
    if not ends.isin(real).all():
        raise ValueError(f"month end on an unscheduled closure: {list(ends[~ends.isin(real)].date)}")
    return {ts.date() for ts in ends}


def to_market_tz(dt):
    """The kit's display conversion (webull_bt.timeutils), imported once the harness has loaded the kit."""
    from webull_bt.timeutils import to_market_tz as convert
    return convert(dt)


class WeightStrategy(bt.Strategy):
    """Trades feeds to target weights of NAV and records closed trades in the kit's format."""

    def __init__(self):
        self.set_tradehistory(True)
        self.closed_trades = []         # the kit's convention: trade log and report markers
        self.orders_refused = 0

    def notify_order(self, order):
        if order.status in (order.Margin, order.Rejected):
            self.orders_refused += 1

    def notify_trade(self, trade):
        """Same record as the kit's example strategies (dual_ma.py, portfolio.py)."""
        if not trade.isclosed:
            return
        self.closed_trades.append({
            "symbol": trade.getdataname(),
            "direction": "LONG" if trade.long else "SHORT",
            "size": trade.history[0].event.size if trade.history else trade.size,
            "entry_price": trade.price,
            "exit_price": trade.history[-1].event.price if trade.history else trade.price,
            "open_dt": to_market_tz(trade.open_datetime()),
            "close_dt": to_market_tz(trade.close_datetime()),
            "pnl": trade.pnl,
            "pnlcomm": trade.pnlcomm,
            "commission": trade.commission,
            "bars_held": trade.barlen,
        })

    @staticmethod
    def has_print(data, ago: int = 0) -> bool:
        """False before the series starts or on a carried-forward missing print (the feed's ``valid`` line)."""
        return bool(data.valid[ago])

    def order_shares(self, data, delta: float, exec: str):
        """Buy (delta > 0) or sell whole shares with the given execution convention."""
        exectype = EXEC_TYPES[exec]
        if delta > 0:
            return self.buy(data=data, size=delta, exectype=exectype)
        return self.sell(data=data, size=-delta, exectype=exectype)

    def rebalance(self, targets: dict[str, float], exec: str, names: list[str] | None = None,
                  nav: float | None = None) -> int:
        """Order each feed in ``names`` (default: every feed) to ``targets[name]`` x NAV, missing names to zero,
        in whole shares at this bar's close; sells go first. Returns the number of orders submitted."""
        nav = self.broker.getvalue() if nav is None else nav
        deltas = []
        for name in names or [d._name for d in self.datas]:
            d = self.getdatabyname(name)
            delta = round(targets.get(name, 0.0) * nav / d.close[0]) - self.getposition(d).size
            if delta:
                deltas.append((d, delta))
        for d, delta in sorted(deltas, key=lambda x: x[1] > 0):
            self.order_shares(d, delta, exec)
        return len(deltas)
