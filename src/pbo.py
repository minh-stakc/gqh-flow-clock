"""Probability of Backtest Overfitting by combinatorially symmetric cross-validation (CSCV).

Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest Overfitting",
Journal of Computational Finance. Given a T x N matrix of daily returns for N strategy
configurations, cut T into S contiguous blocks; for every way of choosing S/2 blocks as the
training set, pick the configuration with the best training Sharpe and record its relative rank
on the complementary (test) blocks. PBO is the fraction of splits where that rank falls below the
median, i.e. where the in-sample winner is an out-of-sample loser.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd


def cscv_pbo(returns: pd.DataFrame, n_blocks: int = 16, max_splits: int | None = None) -> dict:
    X = returns.dropna(how="any").to_numpy(dtype=float)
    T, N = X.shape
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    # per-block sufficient statistics: sum, sum of squares, count
    s1 = np.stack([X[a:b].sum(0) for a, b in zip(edges[:-1], edges[1:])])          # S x N
    s2 = np.stack([(X[a:b] ** 2).sum(0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges).astype(float)
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if max_splits is not None and len(combos) > max_splits:
        step = len(combos) / max_splits
        combos = [combos[int(i * step)] for i in range(max_splits)]
    all_idx = np.arange(n_blocks)
    logits, is_best_sr, oos_of_best = [], [], []
    for tr in combos:
        tr = np.array(tr)
        te = np.setdiff1d(all_idx, tr)

        def sharpe(idx):
            n = cnt[idx].sum()
            m = s1[idx].sum(0) / n
            v = s2[idx].sum(0) / n - m ** 2
            return m / np.sqrt(np.maximum(v, 1e-18))

        sr_tr, sr_te = sharpe(tr), sharpe(te)
        best = int(np.argmax(sr_tr))
        rank = (sr_te < sr_te[best]).sum() + 0.5 * ((sr_te == sr_te[best]).sum() - 1) + 1  # 1..N
        omega = rank / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
        is_best_sr.append(sr_tr[best] * math.sqrt(252))
        oos_of_best.append(sr_te[best] * math.sqrt(252))
    logits = np.array(logits)
    oos = np.array(oos_of_best)
    return {
        "pbo": float((logits <= 0).mean()),
        "n_configs": N,
        "n_splits": len(combos),
        "median_logit": float(np.median(logits)),
        "is_best_sharpe_median": float(np.median(is_best_sr)),
        "oos_of_is_best_sharpe_median": float(np.median(oos)),
        "prob_oos_loss_of_is_best": float((oos < 0).mean()),
        "degradation_slope": float(np.polyfit(is_best_sr, oos, 1)[0]) if len(oos) > 2 else float("nan"),
    }


def effective_trials(returns: pd.DataFrame) -> float:
    """Eigenvalue participation ratio of the configurations' correlation matrix: a rough count of
    independent strategies among highly correlated variants (used alongside the raw trial count)."""
    c = np.corrcoef(returns.dropna(how="any").to_numpy(dtype=float), rowvar=False)
    ev = np.clip(np.linalg.eigvalsh(c), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())
