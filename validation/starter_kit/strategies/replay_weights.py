"""Replay a CSV of daily held weights in the starter kit (not a native strategy: a plumbing check).

The CSV has a date index (the feed calendar) and one column per symbol: ``W.loc[t]`` is the weight of NAV held over
return day t, the engine's ``held`` output (src/engine.simulate). Every order is sized from the close of the bar
that submits it.

* ``fill=coc`` (default; next_close weights, held from the close of t-1): order W_t at the close of t-1 with
  cheat-on-close, so the book holds W_t exactly (validation/backtrader_crosscheck.py section 4).
* ``fill=close`` (next_close weights): submit a ``bt.Order.Close`` for W_t at the close of t-2, the decision
  session; it fills at the close of t-1. The native next_close timing: weights drift one session before the fill.
* ``fill=open`` (next_open weights, held from the open of t): submit a market order for W_t at the close of t-1;
  it fills at the open of t. The native next_open timing: weights drift overnight before the fill.

    --strategy replay_weights --params path=<csv>[,fill=close|open]
"""
from __future__ import annotations

import pandas as pd

from validation.starter_kit.base import WeightStrategy

EXEC = "coc"
NEEDS_MARGIN = True     # weights from our engine assume a margin account (cash may go negative)
FILLS = {"coc": (1, "coc"), "close": (2, "next_close"), "open": (1, "next_open")}   # fill -> (lead, exec)


def SYMBOLS(params: dict) -> list[str]:
    """Every column of the weights file."""
    return list(pd.read_csv(params["path"], index_col=0, nrows=0).columns)


class WeightReplay(WeightStrategy):
    params = dict(path="", fill="coc")

    def __init__(self):
        super().__init__()
        if self.p.fill not in FILLS:
            raise ValueError(f"fill must be one of {sorted(FILLS)}, got {self.p.fill!r}")
        lead, self.exec = FILLS[self.p.fill]
        w = pd.read_csv(self.p.path, index_col=0, parse_dates=True).sort_index().fillna(0.0)
        names = [d._name for d in self.datas]
        self.targets = {ts.date(): dict(zip(names, row)) for ts, row in
                        zip(w.index, w.shift(-lead).fillna(0.0)[names].to_numpy())}

    def start(self):
        self.broker.set_coc(self.p.fill == "coc")

    def next(self):
        row = self.targets.get(self.datas[0].datetime.date(0))
        if row is not None:
            self.rebalance(row, self.exec)


STRATEGY_CLASS = WeightReplay
