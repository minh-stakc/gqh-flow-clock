"""Forward test frozen on Sat 2026-10-03 (FORWARD_TEST.md). Do not edit after the freeze tag.

F1  the submitted strategy, unchanged: ensemble of A (rebalancing pressure) + C (Treasury month end)
    + TSMOM on ETFs, Ledoit-Wolf minimum-variance weights, 8 % volatility target, no brake.
F2  the same construction on CME futures (Databento): A and C on ES/ZN sampled at 16:00 ET with
    1 bp one-way costs, TSMOM on 20 futures (PCT-F rules, measure "none"); same ensemble rules.
F3  information-discreteness-conditioned TSMOM on the same 20 futures (PCT-F rules, measure "ID").

Only return days from config.FWD_START count. The functions below take a period ("FWD" for the
forward evaluation; any other period only reproduces history with the frozen code).
"""
from __future__ import annotations

import pandas as pd

from . import config as C
from . import engine as E
from . import ensemble as EN

STRATEGIES = ["F1", "F2", "F3"]
BENCHMARKS = ["TSMOM_F"]          # plain TSMOM on the 20 futures: the yardstick for F3, not a strategy
PRIMARY = "F1"

FUT_TICKERS = ["F_ES", "F_NQ", "F_RTY", "F_YM", "F_ZT", "F_ZF", "F_ZN", "F_ZB", "F_CL", "F_NG", "F_GC", "F_SI",
               "F_HG", "F_6E", "F_6J", "F_6B", "F_6A", "F_6C", "F_ZC", "F_ZS"]
FUT_COST_BPS = {t: (5.0 if t[2:] in {"NG", "RTY", "ZC", "ZS", "6A", "6C"} else 1.5) for t in FUT_TICKERS}
F2_STREAM_COST_BPS = {"A_F": 1.0, "C_F": 1.0, "TSMOM_F": 2.0}
ENSEMBLE = {"scheme": "minvar_lw", "vol_target": 0.08, "brake": False}


def _stream_specs(name: str) -> dict:
    """label -> (module, params, exec, cost map) for the ensemble strategies."""
    from .strategies import ifc_rebalance, ifc_treasury, pct

    def base(mod, **kw):
        p = {k: v for k, v in mod.BASE_PARAMS.items() if k != "cost_mult"}
        p.update(kw)
        return p

    if name == "F1":
        return {"A": (ifc_rebalance, base(ifc_rebalance), "next_close", None),
                "C": (ifc_treasury, base(ifc_treasury), "next_close", None),
                "TSMOM": (pct, base(pct, measure="none"), "next_open", None)}
    if name == "F2":
        return {"A_F": (ifc_rebalance, base(ifc_rebalance, equity="ES16", bond="ZN16"), "next_close",
                        {"ES16": 1.0, "ZN16": 1.0}),
                "C_F": (ifc_treasury, base(ifc_treasury, ticker="ZN16"), "next_close", {"ZN16": 1.0}),
                "TSMOM_F": (pct, {**base(pct, measure="none"), "tickers": FUT_TICKERS}, "next_close", FUT_COST_BPS)}
    raise ValueError(name)


def _decisions(mod, params: dict, period: str) -> pd.DataFrame:
    return mod.decision_weights(period=period, **params)


def stream_frames(name: str, period: str) -> dict:
    """Per-stream decision weights, held weights, net excess returns and gross exposure."""
    rf = E.load_rf(period)
    out = {"w_dec": {}, "held": {}, "ex": {}, "gr": {}, "exec": {}}
    for lab, (mod, params, ex_conv, cost) in _stream_specs(name).items():
        w = _decisions(mod, params, period)
        ohlc = E.load_ohlc(list(w.columns), period)
        net, _, _, held, _ = E.simulate(w, ohlc, rf, exec=ex_conv, cost_bps=cost)
        out["w_dec"][lab], out["held"][lab], out["exec"][lab] = w, held, ex_conv
        out["ex"][lab] = net - rf.reindex(net.index).ffill().fillna(0.0)
        out["gr"][lab] = held.abs().sum(axis=1)
    out["ex"] = pd.DataFrame(out["ex"])
    out["gr"] = pd.DataFrame(out["gr"]).reindex(out["ex"].index).fillna(0.0)
    out["rf"] = rf.reindex(out["ex"].index).ffill().fillna(0.0)
    return out


def returns(name: str, period: str) -> pd.Series:
    """Daily net returns of a forward-test strategy over the period's whole data range."""
    if name in ("F1", "F2"):
        fr = stream_frames(name, period)
        labels = list(fr["ex"].columns)
        out = EN.combine(fr["ex"], fr["gr"], fr["rf"], labels, ENSEMBLE["scheme"], ENSEMBLE["vol_target"],
                         ENSEMBLE["brake"], cost_bps=F2_STREAM_COST_BPS if name == "F2" else None)
        return out["net"]
    if name in ("F3", "TSMOM_F"):
        from .strategies import pct

        p = {k: v for k, v in pct.BASE_PARAMS.items() if k != "cost_mult"}
        p.update({"measure": "ID" if name == "F3" else "none", "tickers": FUT_TICKERS})
        w = pct.decision_weights(period=period, **p)
        ohlc = E.load_ohlc(FUT_TICKERS, period)
        rf = E.load_rf(period)
        net, *_ = E.simulate(w, ohlc, rf, exec="next_close", cost_bps=FUT_COST_BPS)
        return net
    raise ValueError(name)


def positions_for_next_session(name: str, period: str, pad_days: list[pd.Timestamp]) -> pd.Series:
    """Instrument weights (fraction of NAV) to hold over the return day pad_days[-1], computed only from
    data up to the last date of the period. pad_days are the next NYSE sessions after that date."""
    if name == "F3":
        from .strategies import pct

        p = {k: v for k, v in pct.BASE_PARAMS.items() if k != "cost_mult"}
        p.update({"measure": "ID", "tickers": FUT_TICKERS})
        w = pct.decision_weights(period=period, **p)
        # next_close: held over pad_days[-1] (two sessions after the last date) = decision at the last date
        return w.iloc[-1][w.iloc[-1] != 0]
    fr = stream_frames(name, period)
    labels = list(fr["ex"].columns)
    pad = pd.DatetimeIndex(pad_days)
    held_next = {}
    for lab in labels:
        w_dec = fr["w_dec"][lab]
        if fr["exec"][lab] == "next_close":      # held over t = decision at t-2
            seq = [w_dec.iloc[-2], w_dec.iloc[-1]]
        else:                                    # next_open: held over t = decision at t-1 (unchanged mid-month)
            seq = [w_dec.iloc[-1], w_dec.iloc[-1]]
        held_next[lab] = pd.DataFrame(seq, index=pad)
    ex = pd.concat([fr["ex"], pd.DataFrame(0.0, index=pad, columns=labels)])
    gr = pd.concat([fr["gr"], pd.DataFrame({lab: held_next[lab].abs().sum(axis=1) for lab in labels})])
    rf = pd.concat([fr["rf"], pd.Series(0.0, index=pad)])
    out = EN.combine(ex, gr, rf, labels, ENSEMBLE["scheme"], ENSEMBLE["vol_target"], ENSEMBLE["brake"],
                     cost_bps=F2_STREAM_COST_BPS if name == "F2" else None)
    t = pad[-1]
    scale = out["lam"].loc[t, labels] * out["k"].loc[t]          # uses data up to the last real date only
    total = None
    for lab in labels:
        pos = held_next[lab].loc[t] * scale[lab]
        total = pos if total is None else total.add(pos, fill_value=0.0)
    return total[total.abs() > 1e-12]
