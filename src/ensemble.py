"""Flow-and-Trend ensemble (FTE), HYPOTHESES.md Amendment 1.

Each stream is a pre-registered strategy simulated on its own by the engine (its own instruments,
execution convention and costs). The ensemble scales the streams' daily excess returns with
weights that use only past stream returns, then applies a volatility-target overlay and an
optional drawdown brake:

    r_t = rf_t + k_t * b_t * sum_s lambda_{s,t} * ex_{s,t} - overlay costs_t

* lambda_{s,t}, k_t and b_t use information up to the close of t-2: the earliest decision time of
  any stream's position for return day t (next_close streams decide at t-2), so the scaling is
  known before the underlying trades are placed.
* Changing the scale of a stream is a real trade: overlay costs charge |delta scale| x the
  stream's gross exposure x a per-stream cost (3 bp SPY/IEF sleeves, 5 bp FX, 4 bp trend).
* Netting between streams is ignored (each stream pays its own costs), which overstates costs.
"""
from __future__ import annotations

import importlib
import itertools
import math

import numpy as np
import pandas as pd

from . import config as C
from . import engine as E

STREAMS = {   # stream label -> (module, variant name or None for base)
    "A": ("ifc_rebalance", None),
    "B": ("ifc_dash", None),
    "C": ("ifc_treasury", None),
    "D": ("ifc_fx", None),
    "E": ("ifc_auction", None),
    "PCT": ("pct", None),
    "TSMOM": ("pct", "tsmom_benchmark"),
}
STREAM_COST_BPS = {"A": 3.0, "B": 3.0, "C": 3.0, "D": 5.0, "E": 3.0, "PCT": 4.0, "TSMOM": 4.0}
FLOW = ["A", "B", "C", "D", "E"]
TREND_LEGS = [None, "PCT", "TSMOM"]
WEIGHTINGS = ["eqrisk", "invvol_shrunk", "minvar_lw"]
VOL_TARGETS = [0.06, 0.08, 0.10]
BRAKES = [False, True]
MIN_HISTORY = 252
LEV_CAP = 4.0


def _variant_params(mod, variant: str | None) -> dict:
    params = dict(getattr(mod, "BASE_PARAMS", {}) or {})
    if variant is None:
        return params
    variants = getattr(mod, "VARIANTS", {}) or {}
    key = variant if variant in variants else next((k for k in variants if variant.split("_")[0] in k), None)
    if key is None:
        raise KeyError(f"variant {variant} not in {mod.__name__}.VARIANTS: {list(variants)}")
    params.update(variants[key])
    return params


def stream_frame(period: str = "IS", cost_mult: float = 1.0, labels: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Daily net excess returns and gross exposures of each stream, on the NYSE calendar."""
    labels = labels or list(STREAMS)
    rf = E.load_rf(period)
    exs, gross = {}, {}
    for lab in labels:
        modname, variant = STREAMS[lab]
        mod = importlib.import_module(f"src.strategies.{modname}")
        params = _variant_params(mod, variant)
        params.pop("cost_mult", None)
        try:
            w = mod.decision_weights(period=period, **params)
        except TypeError:
            w = mod.decision_weights(period, **params)
        ohlc = E.load_ohlc(list(w.columns), period)
        net, _, _, held, _ = E.simulate(w, ohlc, rf, exec=mod.EXEC, cost_mult=cost_mult,
                                       cost_bps=getattr(mod, "COST_BPS", None))
        exs[lab] = net - rf.reindex(net.index).ffill().fillna(0.0)
        gross[lab] = held.abs().sum(axis=1)
    ex = pd.DataFrame(exs)
    gr = pd.DataFrame(gross).reindex(ex.index).fillna(0.0)
    return ex, gr, rf.reindex(ex.index).ffill().fillna(0.0)


def _live_mask(gr: pd.DataFrame) -> pd.DataFrame:
    started = (gr > 0).cumsum() > 0
    return started


def _weights(ex: pd.DataFrame, live: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Combination weights at t from stream returns up to t-2 (expanding, >= MIN_HISTORY live days)."""
    x = ex.where(live)
    n = x.notna().cumsum()
    sd = x.expanding(min_periods=MIN_HISTORY).std()
    sd = sd.where(n >= MIN_HISTORY)
    if scheme == "eqrisk":
        w = 1.0 / sd
    elif scheme == "invvol_shrunk":
        iv = 1.0 / sd
        iv = iv.div(iv.sum(axis=1), axis=0)
        k = iv.notna().sum(axis=1).replace(0, np.nan)
        eq = iv.notna().astype(float).div(k, axis=0)
        w = 0.5 * iv + 0.5 * eq.where(iv.notna())
        w = w / sd.mean(axis=1).values[:, None]          # common scale; overlay rescales anyway
    elif scheme == "minvar_lw":
        from sklearn.covariance import LedoitWolf

        w = pd.DataFrame(np.nan, index=ex.index, columns=ex.columns)
        recompute = ex.index[::21]
        last = None
        for d in ex.index:
            if d in recompute:
                ok = [c for c in ex.columns if n.loc[d, c] >= MIN_HISTORY]
                if len(ok) >= 1:
                    hist = x.loc[:d, ok].dropna(how="any")
                    if len(hist) >= MIN_HISTORY:
                        cov = LedoitWolf().fit(hist.to_numpy()).covariance_
                        raw = np.linalg.solve(cov, np.ones(len(ok)))
                        raw = np.clip(raw, 0, None)
                        if raw.sum() > 0:
                            last = pd.Series(raw / raw.sum(), index=ok) / np.sqrt(np.diag(cov)).mean()
            if last is not None:
                w.loc[d, last.index] = last.values
    else:
        raise ValueError(scheme)
    return w.shift(2)                                    # known before the stream trades for day t


_LAM_CACHE: dict = {}


def stream_weights(ex: pd.DataFrame, gr: pd.DataFrame, labels: list[str], scheme: str) -> pd.DataFrame:
    key = (tuple(labels), scheme, id(ex))
    if key not in _LAM_CACHE:
        _LAM_CACHE[key] = _weights(ex[labels], _live_mask(gr[labels]), scheme).fillna(0.0)
    return _LAM_CACHE[key]


def combine(ex: pd.DataFrame, gr: pd.DataFrame, rf: pd.Series, labels: list[str], scheme: str,
            vol_target: float, brake: bool, cost_mult: float = 1.0) -> dict:
    lam = stream_weights(ex, gr, labels, scheme)
    exl, grl = ex[labels].fillna(0.0), gr[labels]
    c = (lam * exl).sum(axis=1)
    var = (c ** 2).ewm(span=63, min_periods=63).mean()
    sig = np.sqrt(var * C.TRADING_DAYS).shift(2)
    k = (vol_target / sig).clip(upper=LEV_CAP)
    gross_scaled = (lam * grl).sum(axis=1)
    lev_scale = (LEV_CAP / (k * gross_scaled)).clip(upper=1.0).fillna(1.0)
    k = (k * lev_scale).fillna(0.0)
    start = (lam.abs().sum(axis=1) > 0) & sig.notna()
    if not start.any():
        raise ValueError("no live period")
    idx = ex.index[ex.index >= start.idxmax()]
    L = lam.reindex(idx).to_numpy()
    X = exl.reindex(idx).to_numpy()
    G = grl.reindex(idx).to_numpy()
    K = k.reindex(idx).to_numpy()
    RF = rf.reindex(idx).to_numpy()
    cb = np.array([STREAM_COST_BPS[s] for s in labels]) * cost_mult / 1e4
    n = len(idx)
    r = np.zeros(n)
    to = np.zeros(n)
    eq = np.ones(n)
    peak = np.ones(n)
    prev = np.zeros(len(labels))
    for i in range(n):
        b = 1.0
        if brake and i >= 2 and eq[i - 2] / peak[i - 2] - 1 < -0.10:   # drawdown known at t-2
            b = 0.5
        scale = K[i] * b * L[i]
        trade = np.abs(scale - prev) * G[i]
        r[i] = scale @ X[i] - trade @ cb
        to[i] = trade.sum()
        prev = scale
        eq[i] = (eq[i - 1] if i else 1.0) * (1 + r[i] + RF[i])
        peak[i] = max(peak[i - 1] if i else 1.0, eq[i])
    excess = pd.Series(r, index=idx)
    net = excess + rf.reindex(idx)
    return {"net": net, "excess": excess, "turnover_overlay": pd.Series(to, index=idx),
            "k": k.reindex(idx), "lam": lam.reindex(idx)}


def configs() -> list[dict]:
    out = []
    for size in range(2, len(FLOW) + 1):
        for sub in itertools.combinations(FLOW, size):
            for trend in TREND_LEGS:
                labels = list(sub) + ([trend] if trend else [])
                for scheme in WEIGHTINGS:
                    for vt in VOL_TARGETS:
                        for br in BRAKES:
                            out.append({"labels": labels, "scheme": scheme, "vol_target": vt, "brake": br})
    return out


def config_name(cfg: dict) -> str:
    return f"{'+'.join(cfg['labels'])}|{cfg['scheme']}|vt{int(cfg['vol_target'] * 100)}|{'brake' if cfg['brake'] else 'nobrake'}"


def to_result(name: str, period: str, out: dict, params: dict) -> E.BacktestResult:
    net, ex = out["net"], out["excess"]
    res = E.BacktestResult(name, period, net, net, ex, out["turnover_overlay"], pd.DataFrame(index=net.index),
                           pd.Series(0.0, index=net.index), params)
    res.stats = E.perf_stats(net, ex, out["turnover_overlay"])
    return res
