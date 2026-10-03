"""Institutional Flow Clock (IFC) composite = sleeves A + B + C (HYPOTHESES.md section 1).

Sleeve multipliers lambda_s = (0.10 / sqrt(3)) / sigma_s, where sigma_s is the annualised standard
deviation of sleeve s's own daily net excess returns over all prior days since the sleeve first
traded (expanding, at least 252 days, lagged one day: the multiplier applied to decision weights at
the close of d uses sleeve returns up to d-1). Sleeve decision weights are summed per ETF; caps
|w| <= 3 per ETF and gross <= 4. All sleeves use next_close execution, so they combine at the
weight level and the engine nets their trades.
"""
from __future__ import annotations

import importlib
import math

import numpy as np
import pandas as pd

from src import analysis as AN
from src import engine as E

FAMILY = "IFC"
EXEC = "next_close"
TICKERS = ["SPY", "IEF"]
COST_BPS = None
SLEEVES = {"A": "ifc_rebalance", "B": "ifc_dash", "C": "ifc_treasury"}
BASE_PARAMS: dict = {"weighting": "invvol", "sleeve_overrides": {}, "extra_delay": 0, "sleeves": ["A", "B", "C"],
                     "equity": "SPY", "bond": "IEF"}
# IFC-F replication (HYPOTHESES.md section 4): same rules on CME futures sampled at 16:00 ET
FUTURES_PARAMS: dict = {"equity": "ES16", "bond": "ZN16"}
FUTURES_COST_BPS = {"ES16": 1.0, "ZN16": 1.0}
VARIANTS: dict[str, dict] = {
    # pre-registered neighbourhood (HYPOTHESES.md section 3, gate G3)
    "shift_early": {"sleeve_overrides": {"A": {"obs": -7, "window": [-5, -1]}, "B": {"shift": -1}, "C": {"window": (-3, -1)}}},
    "shift_late": {"sleeve_overrides": {"A": {"obs": -5, "window": [-3, 1]}, "B": {"shift": 1}, "C": {"window": (-1, 1)}}},
    "A_win4": {"sleeve_overrides": {"A": {"obs": -5, "window": [-3, 0]}}},
    "A_win6": {"sleeve_overrides": {"A": {"obs": -7, "window": [-5, 0]}}},
    "C_win2": {"sleeve_overrides": {"C": {"window": (-1, 0)}}},
    "C_win4": {"sleeve_overrides": {"C": {"window": (-3, 0)}}},
    "equal_mult": {"weighting": "equal"},
    "C_q1": {"sleeve_overrides": {"C": {"issuance_sizing": False}}},
}
NEIGHBOURHOOD = list(VARIANTS)
ROBUSTNESS = {"delay_1d": {"extra_delay": 1}}


def _instrument_overrides(equity: str, bond: str) -> dict:
    return {"A": {"equity": equity, "bond": bond}, "B": {"ticker": equity}, "C": {"ticker": bond}}


def _sleeve_decisions(period: str, overrides: dict, sleeves: list[str], equity: str, bond: str) -> dict[str, pd.DataFrame]:
    out = {}
    inst = _instrument_overrides(equity, bond)
    for s in sleeves:
        mod = importlib.import_module(f"src.strategies.{SLEEVES[s]}")
        p = {**mod.BASE_PARAMS, **inst[s], **overrides.get(s, {})}
        p.pop("cost_mult", None)
        out[s] = mod.decision_weights(period=period, **p).reindex(columns=[equity, bond]).fillna(0.0)
    return out


def sleeve_returns(period: str = "IS", overrides: dict | None = None, sleeves: list[str] | None = None,
                   equity: str = "SPY", bond: str = "IEF") -> tuple[pd.DataFrame, dict]:
    sleeves = sleeves or BASE_PARAMS["sleeves"]
    ws = _sleeve_decisions(period, overrides or {}, sleeves, equity, bond)
    ohlc = E.load_ohlc([equity, bond], period)
    rf = E.load_rf(period)
    cost = FUTURES_COST_BPS if equity.endswith("16") else None
    ex, live = {}, {}
    for s, w in ws.items():
        net, _, _, held, _ = E.simulate(w, ohlc, rf, exec=EXEC, cost_bps=cost)
        ex[s] = net - rf.reindex(net.index).ffill().fillna(0.0)
        live[s] = (held.abs().sum(axis=1) > 0).cumsum() > 0
    return pd.DataFrame(ex), {"weights": ws, "live": pd.DataFrame(live)}


def decision_weights(period: str = "IS", **params) -> pd.DataFrame:
    p = {**BASE_PARAMS, **params}
    ex, info = sleeve_returns(period, p["sleeve_overrides"], p["sleeves"], p["equity"], p["bond"])
    ws, live = info["weights"], info["live"]
    idx = ex.index
    total = pd.DataFrame(0.0, index=idx, columns=[p["equity"], p["bond"]])
    k = len(ws)
    for s, w in ws.items():
        if p["weighting"] == "equal":
            lam = pd.Series(1.0, index=idx)
        else:
            x = ex[s].where(live[s])
            sd = x.expanding(min_periods=252).std() * math.sqrt(252)
            lam = ((0.10 / math.sqrt(k)) / sd).shift(1)         # returns up to d-1
        total += w.reindex(idx).fillna(0.0).mul(lam.fillna(0.0), axis=0)
    total = total.clip(-3, 3)
    gross = total.abs().sum(axis=1)
    total = total.mul((4.0 / gross).clip(upper=1.0).fillna(1.0), axis=0)
    if p["extra_delay"]:
        total = total.shift(int(p["extra_delay"])).fillna(0.0)
    return total


def run(period: str = "IS", log: bool = True, futures: bool = False) -> dict:
    inst = FUTURES_PARAMS if futures else {"equity": "SPY", "bond": "IEF"}
    tag = "IFCF" if futures else "IFC"
    fam = "IFC_F" if futures else FAMILY
    cost = FUTURES_COST_BPS if futures else None
    ohlc = E.load_ohlc([inst["equity"], inst["bond"]], period)
    rf = E.load_rf(period)
    out: dict = {"period": period, "variants": {}, "instruments": inst}
    res = {}
    for name, ov in [("base", {})] + list(VARIANTS.items()) + list(ROBUSTNESS.items()):
        w = decision_weights(period, **{**inst, **ov})
        r = E.run_backtest(f"{tag}_composite_{name}", fam, w, ohlc, rf, period=period,
                           params={**BASE_PARAMS, **inst, **ov}, exec=EXEC, log=log, cost_bps=cost)
        res[name] = r
        out["variants"][name] = r.stats
    r2 = E.run_backtest(f"{tag}_composite_base", fam, decision_weights(period, **inst), ohlc, rf, period=period,
                        params={**BASE_PARAMS, **inst}, exec=EXEC, cost_mult=2.0, log=log, cost_bps=cost)
    base = res["base"]
    out["base"] = base.stats
    out["base_2x_costs"] = r2.stats
    out["neighbourhood_median_sharpe"] = float(np.median([res[n].stats["sharpe"] for n in NEIGHBOURHOOD]))
    out["subperiods"] = AN.subperiod_table(base).to_dict("records")
    out["concentration"] = AN.pnl_concentration(base)
    out["factor"] = AN.factor_table(base)
    out["bootstrap_sharpe_90"] = AN.bootstrap_sharpe_ci(base.excess)
    out["yearly"] = {int(k): float(v) for k, v in AN.yearly_returns(base.returns).items()}
    ex, _ = sleeve_returns(period, equity=inst["equity"], bond=inst["bond"])
    out["sleeve_corr"] = ex.loc[base.returns.index].corr().round(3).to_dict()
    out["capacity"] = AN.capacity_curve(base, ohlc).to_dict("records")
    out["_results"] = res
    return out
