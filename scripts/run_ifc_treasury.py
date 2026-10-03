"""Run IFC sleeve C (Treasury month-end index extension, sized by issuance DV01) in-sample.

Usage (from the repo root):
    GQH_DATA_DIR=C:/Users/hoang/.cache/gqh python scripts/run_ifc_treasury.py

Writes results/is/ifc_treasury.json, results/is/ifc_treasury_returns.csv,
results/is/ifc_treasury_monthly.csv and two figures in results/figures/. Every backtest goes
through engine.run_backtest(log=True).
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
from src.strategies import ifc_treasury as M  # noqa: E402


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


def plot_predictions(prof: pd.DataFrame, wr: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    offs = M.PROFILE_OFFSETS
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), dpi=150)
    ax = axes[0]
    mean = np.array([prof[o].mean() for o in offs]) * 1e4
    se = np.array([prof[o].std(ddof=1) / np.sqrt(prof[o].notna().sum()) for o in offs]) * 1e4
    a, b = M.BASE_PARAMS["window"]
    ax.axvspan(a - 0.5, b + 0.5, color="#1f5fa8", alpha=0.12, label="hold window [T-2, T]")
    ax.errorbar(offs, mean, yerr=se, fmt="o-", ms=3, lw=1, capsize=2, color="#1f5fa8")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(offs)
    ax.set_xticklabels([f"{o:+d}" if o else "T" for o in offs], fontsize=6)
    ax.set_xlabel("return day relative to month-end T (trading days)", fontsize=7)
    ax.set_ylabel("mean IEF excess return (bp), +/-1 se", fontsize=7)
    ax.set_title(f"IEF around month end, IS ({len(prof)} months)", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    pre = wr["T"] < M.SPLIT_DATE
    ax.scatter(wr.loc[pre, "q"], wr.loc[pre, "ret"] * 1e4, s=8, alpha=0.6, color="#7f8c8d", label="T < 2015")
    ax.scatter(wr.loc[~pre, "q"], wr.loc[~pre, "ret"] * 1e4, s=8, alpha=0.6, color="#c0392b", label="T >= 2015")
    x = wr["q"].astype(float).to_numpy()
    y = wr["ret"].astype(float).to_numpy() * 1e4
    k, c = np.polyfit(x, y, 1)
    xs = np.linspace(0.5, 2.0, 10)
    ax.plot(xs, c + k * xs, color="black", lw=1, label=f"OLS slope {k:.1f} bp per unit q")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("q = clip(S_m / median prev. 12 months, 0.5, 2)", fontsize=7)
    ax.set_ylabel("IEF excess return over [T-2, T] (bp)", fontsize=7)
    ax.set_title("Issuance (DV01) vs month-end window return", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    summary, extra = M.run("IS", log=True, return_results=True)
    out_dir = C.RESULTS_DIR / "is"
    out_dir.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "ifc_treasury.json", "w", encoding="utf-8") as fh:
        json.dump(_clean(json.loads(json.dumps(summary, default=_default))), fh, indent=2)

    res = extra["results"]
    rets = pd.concat({name: r.returns for name, r in res.items()}, axis=1, sort=True)
    rets.index.name = "date"
    rets.to_csv(out_dir / "ifc_treasury_returns.csv", float_format="%.8g")
    mt = extra["monthly"].copy()
    mt.index = mt.index.astype(str)
    mt.to_csv(out_dir / "ifc_treasury_monthly.csv", float_format="%.8g")

    A.plot_equity({k: res[k] for k in ("base", "window2", "window4", "shift_early", "shift_late", "q1", "cost_2x")},
                  C.FIG_DIR / "ifc_treasury_equity.png",
                  title="IFC sleeve C (Treasury month-end extension, issuance-sized), IEF, in-sample, net",
                  log=False)
    plot_predictions(extra["profile"], extra["window_returns"], C.FIG_DIR / "ifc_treasury_predictions.png")

    b, b2 = summary["base"], summary["base_cost_2x"]
    print(f"base   SR {b['sharpe']:.3f}  ann {b['ann_return']:.4f}  vol {b['ann_vol']:.4f}  MDD {b['max_drawdown']:.3f}")
    print(f"2x     SR {b2['sharpe']:.3f}  ann {b2['ann_return']:.4f}")
    for v, s in summary["variants"].items():
        print(f"{v:14s} SR {s['sharpe']:.3f}")


if __name__ == "__main__":
    main()
