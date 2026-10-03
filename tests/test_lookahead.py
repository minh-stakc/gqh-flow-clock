"""Lookahead perturbation test (in-sample data only; no OOS unlock needed).

For each strategy module and several random cut dates, every price after the cut is replaced by a
random walk (same calendar). If a module used any information from after the cut, its decision
weights on or before the cut would change. The test asserts they are bit-identical.

    python -m pytest tests/test_lookahead.py -q      (or: python tests/test_lookahead.py)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import engine as E  # noqa: E402

MODULES = ["ifc_rebalance", "ifc_dash", "ifc_treasury", "pct"]
CUTS = ["2009-03-09", "2015-06-15", "2020-03-16", "2023-10-31"]


def _perturbed_loader(original, cut: pd.Timestamp, seed: int, volume: bool = False):
    def load(tickers=None, period="IS"):
        out = original(tickers, period)
        rng = np.random.default_rng(seed)
        after = out["close"].index > cut
        for col in ("open", "high", "low", "close"):
            px = out[col].copy()
            noise = np.exp(np.cumsum(rng.normal(0, 0.02, size=(after.sum(), px.shape[1])), axis=0))
            base = px.loc[~after].ffill().iloc[-1].to_numpy() if (~after).any() else np.ones(px.shape[1])
            px.loc[after] = base * noise
            out[col] = px
        if volume and "volume" in out:                 # forward2 eligibility reads volume: zero half of it later
            v = out["volume"].copy()
            v.loc[after] = v.loc[after] * (rng.random((after.sum(), v.shape[1])) < 0.5)
            out["volume"] = v
        return out
    return load


def _clear_caches(mod) -> None:
    """Modules may memoise their price panel (functools.lru_cache); clear it so the patched loader is used."""
    for obj in vars(mod).values():
        if callable(getattr(obj, "cache_clear", None)):
            obj.cache_clear()


def check_module(name: str) -> None:
    mod = importlib.import_module(f"src.strategies.{name}")
    params = {k: v for k, v in (getattr(mod, "BASE_PARAMS", {}) or {}).items() if k != "cost_mult"}
    original = E.load_ohlc
    _clear_caches(mod)
    base = mod.decision_weights(period="IS", **params)
    for i, c in enumerate(CUTS):
        cut = pd.Timestamp(c)
        E.load_ohlc = _perturbed_loader(original, cut, seed=100 + i)
        _clear_caches(mod)
        try:
            pert = mod.decision_weights(period="IS", **params)
        finally:
            E.load_ohlc = original
            _clear_caches(mod)
        a = base.loc[:cut].fillna(0.0)
        b = pert.reindex(a.index).fillna(0.0)
        diff = float((a - b).abs().to_numpy().max())
        assert diff == 0.0, f"{name}: decisions up to {c} changed by {diff} after perturbing later prices"
        # power check: the perturbation must reach the module, i.e. later decisions do change
        later = float((base.loc[cut:].iloc[5:].fillna(0.0) - pert.loc[cut:].iloc[5:].fillna(0.0)).abs().to_numpy().max())
        assert later > 0, f"{name}: perturbation had no effect after {c} (test has no power)"
        print(f"  {name:14s} cut {c}: OK (max diff before cut {diff}, after cut {later:.3f})")


def test_no_lookahead():
    for m in MODULES:
        check_module(m)


FWD2 = ["S1", "S2", "S3", "LONG_ONLY_F", "ES_10VOL", "BH5"]
FWD2_CUTS = ["2015-06-15", "2020-03-16", "2023-10-31"]      # futures start 2010-06 plus 300 sessions of history


def check_forward2(name: str) -> None:
    from src import forward2 as F2

    fn = F2.SPECS[name][0]
    original = E.load_ohlc
    base = fn("IS")
    for i, c in enumerate(FWD2_CUTS):
        cut = pd.Timestamp(c)
        E.load_ohlc = _perturbed_loader(original, cut, seed=200 + i, volume=True)
        try:
            pert = fn("IS")
        finally:
            E.load_ohlc = original
        a = base.loc[:cut].fillna(0.0)
        b = pert.reindex(a.index).fillna(0.0)
        diff = float((a - b).abs().to_numpy().max())
        assert diff == 0.0, f"forward2 {name}: decisions up to {c} changed by {diff} after perturbing later data"
        later = float((base.loc[cut:].iloc[5:].fillna(0.0) - pert.loc[cut:].iloc[5:].fillna(0.0)).abs().to_numpy().max())
        if name != "BH5":                              # BH5's weights are constant after warm-up: nothing to move
            assert later > 0, f"forward2 {name}: perturbation had no effect after {c} (test has no power)"
        print(f"  forward2 {name:11s} cut {c}: OK (max diff before cut {diff}, after cut {later:.3f})")


def test_forward2_no_lookahead():
    for m in FWD2:
        check_forward2(m)


if __name__ == "__main__":
    test_no_lookahead()
    test_forward2_no_lookahead()
    print("no lookahead detected")
