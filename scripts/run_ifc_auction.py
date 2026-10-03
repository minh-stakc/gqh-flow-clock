"""Run exploratory IFC sleeve E (Treasury auction cycle) in-sample. Reported only.

Usage (from the repo root):
    GQH_DATA_DIR=C:/Users/hoang/.cache/gqh python scripts/run_ifc_auction.py

Writes results/is/ifc_auction.json, results/is/ifc_auction_returns.csv and two figures in
results/figures/. Every backtest goes through engine.run_backtest(log=True).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import analysis as A  # noqa: E402
from src import config as C  # noqa: E402
from src.strategies import ifc_auction as M  # noqa: E402


def _default(o):
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def plot_event_study(evt: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offs = M.EVENT_OFFSETS
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=150)
    groups = (("pre-2014", evt[evt["A"] < M.SPLIT_DATE], "#1f5fa8", -0.15),
              ("2014 on", evt[evt["A"] >= M.SPLIT_DATE], "#c0392b", 0.15))
    for lab, g, col, dx in groups:
        mean = np.array([g[k].mean() for k in offs]) * 1e4
        se = np.array([g[k].std(ddof=1) / np.sqrt(len(g)) for k in offs]) * 1e4
        x = np.array(offs, dtype=float)
        axes[0].errorbar(x + dx, mean, yerr=se, fmt="o-", ms=3, lw=1, capsize=2, color=col,
                         label=f"{lab} (n={len(g)})")
        axes[1].plot(x, np.cumsum(mean), "o-", ms=3, lw=1.2, color=col, label=lab)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.6)
        ax.axvspan(-4.5, 0.5, color="grey", alpha=0.10)
        ax.axvline(0.5, color="grey", ls=":", lw=0.8)
        ax.set_xticks(offs)
        ax.set_xticklabels([f"{k:+d}" if k else "A" for k in offs], fontsize=6)
        ax.set_xlabel("return day relative to auction day A (trading days)", fontsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean IEF excess return (bp), +/-1 se", fontsize=7)
    axes[1].set_ylabel("cumulative mean from A-5 (bp)", fontsize=7)
    axes[0].set_title("Nominal Note/Bond auctions: IEF by event day (shaded = short window)", fontsize=8)
    axes[1].set_title("Cumulative event-time path", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    summary, extra = M.run("IS", log=True, return_results=True)
    out_dir = C.RESULTS_DIR / "is"
    out_dir.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ifc_auction.json", "w", encoding="utf-8") as fh:
        json.dump(_clean(json.loads(json.dumps(summary, default=_default))), fh, indent=2)

    res = extra["results"]
    rets = pd.concat({name: r.returns for name, r in res.items()}, axis=1, sort=True)
    rets.index.name = "date"
    rets.to_csv(out_dir / "ifc_auction_returns.csv", float_format="%.8g")

    A.plot_equity(res, C.FIG_DIR / "ifc_auction_equity.png",
                  title="IFC exploratory sleeve E (Treasury auction cycle), IEF, in-sample, net", log=False)
    plot_event_study(extra["events"], C.FIG_DIR / "ifc_auction_event_study.png")

    b, b2 = summary["base"], summary["base_cost_2x"]
    print(f"base   SR {b['sharpe']:.3f}  ann {b['ann_return']:.4f}  vol {b['ann_vol']:.4f}  MDD {b['max_drawdown']:.3f}")
    print(f"2x     SR {b2['sharpe']:.3f}  ann {b2['ann_return']:.4f}")
    for v, s in summary["variants"].items():
        print(f"{v:14s} SR {s['sharpe']:.3f}")


if __name__ == "__main__":
    main()
