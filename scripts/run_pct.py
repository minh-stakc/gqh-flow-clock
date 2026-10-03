"""Run the Persistence-Conditioned Trend candidate (base + pre-registered variants), in-sample.

Usage (from the repo root):
    GQH_DATA_DIR=<cache> python scripts/run_pct.py

Writes results/is/pct.json, results/is/pct_returns.csv, results/figures/pct_equity.png and
results/figures/pct_diagnostics.png. Every backtest is logged to results/trials.csv.
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
from src.strategies import pct as P  # noqa: E402

# reference categorical palette (slots 1-3, validated all-pairs) and light surface/ink tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def equity_figure(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = summary["_results"]
    show = [("base", "PCT (base)"), ("tsmom_benchmark", "TSMOM benchmark (m = 1)"), ("id_variant", "ID-conditioned")]
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=150, facecolor=SURFACE)
    _style(ax)
    for (key, label), col in zip(show, SERIES):
        eq = (1 + res[key].returns).cumprod()
        sr = res[key].stats["sharpe"]
        ax.plot(eq.index, eq.values, color=col, lw=1.4, label=f"{label}  SR {sr:.2f}")
        ax.annotate(f"{eq.iloc[-1]:.2f}", (eq.index[-1], eq.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=INK2, va="center")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ax.yaxis.set_major_locator(FixedLocator([0.8, 1, 1.5, 2, 3, 4, 5]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("growth of $1 (log, net of costs)", fontsize=7, color=INK2)
    ax.set_title("PCT vs plain TSMOM, in-sample 2006-2024, 10% ex-ante vol, next-open execution",
                 fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def diagnostics_figure(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pr = summary["predictions"]
    fwd = summary["_fwd_panel"]
    pan = summary["_panel"]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=150, facecolor=SURFACE)
    ax = axes[0]
    _style(ax)
    width = 0.38
    for k, (key, label, col) in enumerate([("P2_R", "R (bridge excursion)", SERIES[0]),
                                           ("P2_ID", "ID (info. discreteness)", SERIES[1])]):
        zc = pr[key]["zcol"]
        d = fwd[["date", zc, "y"]].dropna().copy()
        d["terc"] = pd.qcut(d[zc].rank(method="first"), 3, labels=[1, 2, 3]).astype(int)
        g = d.groupby("terc")["y"]
        m, se = g.mean(), g.std() / np.sqrt(g.count())
        x = np.arange(3) + (k - 0.5) * width
        ax.bar(x, m.values, width=width - 0.04, color=col, label=f"{label}: top-bottom t = "
               f"{pr[key]['t_cluster_month']:.2f}")
        ax.errorbar(x, m.values, yerr=1.645 * se.values, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(3), ["bottom z", "middle z", "top z"], fontsize=7)
    ax.set_ylabel("next-month s * r_excess / vol_ann", fontsize=7, color=INK2)
    ax.set_title("P2: next-month TSMOM return by z tercile (90% CI, iid)", fontsize=8, color=INK, loc="left")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax.legend(fontsize=6.5, frameon=False, labelcolor=INK, loc="upper left")

    ax = axes[1]
    _style(ax)
    ax.scatter(pan["vol"], pan["z"], s=8, color=SERIES[0], alpha=0.35, edgecolors="none")
    ax.axhline(0, color=INK2, lw=0.6)
    p3 = pr["P3_vol_neutrality"]
    ax.set_xlabel("63-day realised vol (annualised)", fontsize=7, color=INK2)
    ax.set_ylabel("persistence z", fontsize=7, color=INK2)
    ax.set_title(f"P3: z vs vol, Spearman {p3['spearman_z_vol']:.2f} "
                 f"(area alone: {p3['contrast_spearman_Abar_vol']:.2f})", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    summary = P.run("IS")
    out_dir = C.RESULTS_DIR / "is"
    out_dir.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    rets = summary["_returns"]
    rets.index.name = "date"
    rets.to_csv(out_dir / "pct_returns.csv", float_format="%.8g")
    equity_figure(summary, C.FIG_DIR / "pct_equity.png")
    diagnostics_figure(summary, C.FIG_DIR / "pct_diagnostics.png")
    public = {k: v for k, v in summary.items() if not k.startswith("_")}
    (out_dir / "pct.json").write_text(json.dumps(public, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: public[k] for k in ("base_stats", "variant_sharpe", "gates")}, indent=1))
    print(json.dumps(public["predictions"], indent=1))


if __name__ == "__main__":
    main()
