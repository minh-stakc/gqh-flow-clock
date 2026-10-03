"""In-sample run of IFC exploratory sleeve D (FX hedge rebalancing): base + 2x costs + extra lag.

Usage (repo root):  GQH_DATA_DIR=<cache> python scripts/run_ifc_fx.py
Writes results/is/ifc_fx.json, results/is/ifc_fx_returns.csv and two figures in results/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import analysis as A  # noqa: E402
from src import config as C  # noqa: E402
from src.strategies import ifc_fx as M  # noqa: E402


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def plot_signal(tab: pd.DataFrame, tests: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=150, sharey=True)
    for ax, t in zip(axes, M.TICKERS):
        g = tab[(tab["ticker"] == t) & tab["active"] & tab["window_ret"].notna()]
        x, y = g["signal"] * 100, g["window_ret"] * 1e4
        ax.scatter(x, y, s=6, alpha=0.5, color="#3b6ea5", lw=0)
        q = pd.qcut(x, 5, labels=False)
        bx, by = x.groupby(q).mean(), y.groupby(q).mean()
        ax.plot(bx, by, "o-", color="#c0392b", ms=4, lw=1.2, label="quintile means")
        ax.axhline(0, color="grey", lw=0.6)
        ax.axvline(0, color="grey", lw=0.6)
        s = tests[t]
        ax.set_title(f"{t} vs {M.FOREIGN[t]}: hit {s['hit_rate']:.1%}, signed {s['mean_signed_ret_bp']:.1f} bp "
                     f"(t={s['mean_signed_ret_t']:.2f})", fontsize=7.5)
        ax.set_xlabel("signal at T-3: SPY MTD - local index MTD (%)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ETF return over [T-1, T] (bp)", fontsize=7)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Sleeve D prediction: currency ETF moves in the signal's direction into month end (IS)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    summary = M.run("IS")
    rets = summary.pop("_returns")
    results = summary.pop("_results")
    tab = summary.pop("_monthly")
    out_dir = C.RESULTS_DIR / "is"
    out_dir.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    (out_dir / "ifc_fx.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")
    rets.index.name = "date"
    rets.to_csv(out_dir / "ifc_fx_returns.csv", float_format="%.8g")
    A.plot_equity({"base": results["base"], "2x costs": results["cost2x"], "extra day lag": results["extra_lag"]},
                  C.FIG_DIR / "ifc_fx_equity.png", title="IFC exploratory sleeve D (FX hedge rebalancing), in-sample, net",
                  log=False)
    plot_signal(tab, summary["prediction_tests"], C.FIG_DIR / "ifc_fx_signal.png")
    s = summary["base"]["stats"]
    print(f"base IS Sharpe {s['sharpe']:.3f}  2x {summary['base_2x_costs']['stats']['sharpe']:.3f}  "
          f"extra lag {summary['variant_sharpe']['extra_lag']:.3f}")


if __name__ == "__main__":
    main()
