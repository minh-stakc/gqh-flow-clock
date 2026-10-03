"""Run IFC sleeve B (settlement-cycle dash for cash) in-sample and write its results.

Usage (from the repo root):
    GQH_DATA_DIR=C:/Users/hoang/.cache/gqh python scripts/run_ifc_dash.py

Writes results/is/ifc_dash.json, results/is/ifc_dash_returns.csv and two figures in
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
from src.strategies import ifc_dash as M  # noqa: E402


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
    """Replace NaN/inf with None so the JSON is strict."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def plot_profile(prof: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offs = M.PROFILE_OFFSETS
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=150)
    colors = {"T+3": "#1f5fa8", "T+2": "#c0392b"}
    for reg, dx in (("T+3", -0.15), ("T+2", 0.15)):
        ps = M._profile_stats(prof, reg, offs)
        x = np.array(offs, dtype=float)
        axes[0].errorbar(x + dx, ps["mean"] * 1e4, yerr=ps["se"] * 1e4, fmt="o-", ms=3, lw=1,
                         capsize=2, color=colors[reg], label=f"{reg} regime (n={int(ps['n'].iloc[0])})")
        axes[1].plot(x, ps["cum_mean"] * 1e4, "o-", ms=3, lw=1.2, color=colors[reg], label=reg)
    for ax in axes:
        ax.axhline(0, color="black", lw=0.6)
        # pressure/liquidity boundary: between T-k-1 and T-k
        ax.axvline(-3.5, color=colors["T+3"], ls="--", lw=0.8)
        ax.axvline(-2.5, color=colors["T+2"], ls="--", lw=0.8)
        ax.axvline(0.5, color="grey", ls=":", lw=0.8)
        ax.set_xticks(offs)
        ax.set_xticklabels([f"{o:+d}" if o else "T" for o in offs], fontsize=6)
        ax.grid(alpha=0.3)
        ax.set_xlabel("return day relative to month-end T (trading days)", fontsize=7)
    axes[0].set_ylabel("mean SPY excess return (bp), +/-1 se", fontsize=7)
    axes[1].set_ylabel("cumulative mean from T-10 (bp)", fontsize=7)
    axes[0].set_title("B2: event-time profile by settlement regime", fontsize=8)
    axes[1].set_title("dashed: predicted pressure->liquidity boundary (T-k-1 | T-k)", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    summary, extra = M.run("IS", log=True, return_results=True)
    out_dir = C.RESULTS_DIR / "is"
    out_dir.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ifc_dash.json", "w", encoding="utf-8") as fh:
        json.dump(_clean(json.loads(json.dumps(summary, default=_default))), fh, indent=2)

    res = extra["results"]
    rets = pd.concat({name: r.returns for name, r in res.items()}, axis=1, sort=True)
    rets.index.name = "date"
    rets.to_csv(out_dir / "ifc_dash_returns.csv", float_format="%.8g")
    extra["spreads"].drop(columns=["pa", "pb", "la", "lb"]).to_csv(
        out_dir / "ifc_dash_monthly_spreads.csv", index=False, float_format="%.8g")

    A.plot_equity({k: res[k] for k in ("base", "shift_earlier", "shift_later", "fixed_T3", "cost_2x")},
                  C.FIG_DIR / "ifc_dash_equity.png",
                  title="IFC sleeve B (dash for cash), SPY, in-sample, net of costs", log=False)
    plot_profile(extra["profile"], C.FIG_DIR / "ifc_dash_event_profile.png")

    b, b2 = summary["base"], summary["base_cost_2x"]
    print(f"base   SR {b['sharpe']:.3f}  ann {b['ann_return']:.4f}  vol {b['ann_vol']:.4f}  MDD {b['max_drawdown']:.3f}")
    print(f"2x     SR {b2['sharpe']:.3f}  ann {b2['ann_return']:.4f}")
    for v, s in summary["variants"].items():
        print(f"{v:14s} SR {s['sharpe']:.3f}")


if __name__ == "__main__":
    main()
