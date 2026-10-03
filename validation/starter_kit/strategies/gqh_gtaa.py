"""Forward test 2's S3, Faber (2007) GTAA, computed natively in backtrader (and the BH5 yardstick).

Signal for signal as src/forward2.s3_decisions. At the close of the last session of each month (calendar_utils
month_offsets on the NYSE calendar, as forward2._month_ends) each ETF gets ``weight`` (20 %) of NAV when its
month-end close is above the mean of its last ``months`` (10) month-end closes, this one included, and nothing
otherwise. The mean needs all ten closes (rolling(10, min_periods=10)): a series sits out until it has ten
month-end prints, and for the ten month ends starting at a missing one. ``buy_and_hold=true`` gives BH5: every ETF
at ``weight`` once its mean exists. Between month ends the decision is carried, and every session the book trades
back to it, as the engine does (drift costs included).

Execution is the engine's next_open: in next() of bar d one plain market order per ETF, in whole shares sized at the
NAV and close of d, fills at the open of d+1. The engine trades the exact target at that open, so here each weight
drifts overnight before the fill. Uninvested NAV sits in cash, which backtrader does not pay interest on; the
harness's "our Sharpe" credits it with the T-bill. check_gtaa.py compares the decisions and returns with the engine.

The only side data is the exchange's session schedule, published in advance, to know at the close of d that d ends
its month (base.planned_month_ends: the planned calendar, unscheduled closures included; it raises if a month end
fell on one, and none did, so these are the engine's month ends).

Gross is at most 1, so the kit's default cash broker works. There, ``reserve`` keeps a slice of NAV in cash: the
broker refuses any buy that would take cash below zero, and an order sized at the close of d still has to pay
commission, share rounding and the overnight gap at the open of d+1. ``reserve=auto`` is 0 on a margin account
(the engine's fully invested book) and RESERVE_CASH_ACCOUNT on a cash account: of 0, 0.25 %, 0.5 % and 1 %, the
smallest with no refused order for S3 and BH5 at project and guide costs with 1e9 and 100,000 of cash (0 gave 638
to 4,989 refusals). It was chosen by counting refused orders over the whole sample, out-of-sample years included:
an operational cash buffer picked on refusals, not on returns, but picked after seeing those data.

    --strategy gqh_gtaa [--params buy_and_hold=true,reserve=0.01,decisions=<csv of month-end weights>]
"""
from __future__ import annotations

import math
from collections import deque

import pandas as pd

from src import forward2 as F2
from validation.starter_kit.base import WeightStrategy, planned_month_ends

SYMBOLS = list(F2.S3_TICKERS)
EXEC = "next_open"
IS_START = "2005-11-01"         # first session held: feeds start 2005-01-03, ten month ends of warm-up
RESERVE_CASH_ACCOUNT = 0.01     # share of NAV left in cash on a cash account (reserve=auto)


class FaberGTAA(WeightStrategy):
    """S3 (BH5 with buy_and_hold) on the feeds SPY, EFA, IEF, VNQ, DBC; ``decisions`` names a CSV for the month-end
    weights."""
    params = dict(months=10, weight=0.20, buy_and_hold=False, reserve="auto", decisions="")

    def __init__(self):
        super().__init__()
        self.month_ends = planned_month_ends()
        self.closes = {d._name: deque(maxlen=self.p.months) for d in self.datas}
        self.targets: dict[str, float] = {}
        self.decision_log: list[tuple] = []

    def start(self):
        margin = any(self.broker.getcommissioninfo(d).get_leverage() > 1 for d in self.datas)
        reserve = self.p.reserve
        if reserve == "auto":
            reserve = 0.0 if margin else RESERVE_CASH_ACCOUNT
        self.invested = 1.0 - float(reserve)

    def decide(self) -> dict[str, float]:
        """Month-end weights from month-end closes; a padded session counts as a missing print."""
        row = {}
        for d in self.datas:
            hist = self.closes[d._name]
            hist.append(d.close[0] if self.has_print(d) else math.nan)
            full = len(hist) == self.p.months and all(math.isfinite(x) for x in hist)
            on = full and (self.p.buy_and_hold or hist[-1] > math.fsum(hist) / self.p.months)
            row[d._name] = self.p.weight if on else 0.0
        return row

    def next(self):
        today = self.datas[0].datetime.date(0)
        if today in self.month_ends:
            self.targets = self.decide()
            self.decision_log.append((today, self.targets))
        self.rebalance({k: w * self.invested for k, w in self.targets.items()}, EXEC)

    def stop(self):
        if self.p.decisions:
            dates, rows = zip(*self.decision_log) if self.decision_log else ((), ())
            pd.DataFrame(list(rows), index=pd.DatetimeIndex(dates, name="date")).to_csv(self.p.decisions)


STRATEGY_CLASS = FaberGTAA
