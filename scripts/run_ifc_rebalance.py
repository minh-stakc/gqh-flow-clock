"""Run IFC sleeve A (stock/bond month-end rebalancing pressure) in-sample and save results.

    GQH_DATA_DIR=C:/Users/hoang/.cache/gqh python scripts/run_ifc_rebalance.py

Writes results/is/ifc_rebalance.json, results/is/ifc_rebalance_returns.csv and two figures in
results/figures/. Every backtest is logged to results/trials.csv by the engine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src.strategies import ifc_rebalance as M  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
INK, INK2 = "#0b0b0b", "#52514e"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c8c7c0")
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(alpha=0.25, lw=0.6)


def figures(summary: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = summary["_results"]
    paths = []

    # 1. cumulative net excess return (arithmetic sum, %): base vs 2x costs; base vs variants
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), dpi=150, sharey=True)
    ax = axes[0]
    for key, col, lab in (("base", BLUE, "base"), ("base_2x", ORANGE, "base, 2x costs")):
        cum = res[key].excess.cumsum() * 100
        ax.plot(cum.index, cum.values, color=col, lw=1.4, label=lab)
    ax.set_title("Sleeve A base, in-sample", fontsize=8.5, color=INK, loc="left")
    ax.set_ylabel("cumulative excess return, % (sum)", fontsize=7.5, color=INK2)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax)
    ax = axes[1]
    for v in M.NEIGHBOURHOOD + ["delay1"]:
        cum = res[v].excess.cumsum() * 100
        ax.plot(cum.index, cum.values, color=GREY, lw=0.8, alpha=0.9)
        ax.annotate(v, (cum.index[-1], cum.values[-1]), fontsize=6, color=INK2, xytext=(2, 0),
                    textcoords="offset points", va="center")
    for key, col, lab in (("base", BLUE, "base"), ("threshold", AQUA, "threshold bands")):
        cum = res[key].excess.cumsum() * 100
        ax.plot(cum.index, cum.values, color=col, lw=1.4, label=lab)
    ax.set_title("Neighbourhood (grey) and robustness variants", fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax)
    fig.tight_layout()
    p = out_dir / "ifc_rebalance_equity.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # 2. prediction diagnostic: window SPY-minus-IEF return vs month-to-date drift at T-6
    mt = summary["_month_table"]
    start, end = C.HISTORY_START, C.IS_END
    mt = mt[(mt["start"] >= pd.Timestamp(start)) & (mt["end"] <= pd.Timestamp(end))]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), dpi=150)
    ax = axes[0]
    pos, neg = mt["corr252"] > 0, mt["corr252"] <= 0
    na = mt["corr252"].isna()
    ax.scatter(mt.loc[neg, "D"] * 100, mt.loc[neg, "spread"] * 100, s=12, color=BLUE, alpha=0.75,
               edgecolor="white", lw=0.4, label="trailing SPY/IEF corr <= 0")
    ax.scatter(mt.loc[pos, "D"] * 100, mt.loc[pos, "spread"] * 100, s=12, color=ORANGE, alpha=0.85,
               edgecolor="white", lw=0.4, label="corr > 0")
    ax.scatter(mt.loc[na, "D"] * 100, mt.loc[na, "spread"] * 100, s=12, color=GREY, alpha=0.8,
               edgecolor="white", lw=0.4, label="corr n/a (first year)")
    t = summary["predictions"]["base"]["all_months_spread_on_D"]
    xs = np.linspace(mt["D"].min(), mt["D"].max(), 50)
    ax.plot(xs * 100, (t["intercept"] + t["slope"] * xs) * 100, color=INK, lw=1.2)
    ax.axhline(0, color="#c8c7c0", lw=0.6)
    ax.axvline(0, color="#c8c7c0", lw=0.6)
    ax.set_xlabel("60/40 drift D at T-6, equity-weight pp", fontsize=7.5, color=INK2)
    ax.set_ylabel("SPY minus IEF return over [T-4, T], %", fontsize=7.5, color=INK2)
    ax.set_title(f"slope {t['slope']:.3f} (HC1 t {t['slope_t_hc1']:.2f}), n = {t['n']}",
                 fontsize=8.5, color=INK, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    _style(ax)
    ax = axes[1]
    q = summary["predictions"]["base"]["spread_by_D_quintile"]
    ks = sorted(int(k) for k in q)
    vals = [q[k] * 100 for k in ks]
    ax.bar([k + 1 for k in ks], vals, width=0.6, color=BLUE, edgecolor="white", lw=2)
    for k, v in zip(ks, vals):
        ax.annotate(f"{v:.2f}", (k + 1, v), fontsize=6.5, color=INK2, ha="center",
                    va="bottom" if v >= 0 else "top", xytext=(0, 2 if v >= 0 else -2), textcoords="offset points")
    ax.axhline(0, color="#c8c7c0", lw=0.6)
    ax.set_xlabel("quintile of D (1 = most toward bonds, 5 = most toward stocks)", fontsize=7.5, color=INK2)
    ax.set_ylabel("mean window SPY minus IEF, %", fontsize=7.5, color=INK2)
    ax.set_title("Prediction: falls from left to right", fontsize=8.5, color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    p = out_dir / "ifc_rebalance_prediction.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def main(period: str = "IS") -> None:
    if period != "IS":
        raise SystemExit("this script runs the in-sample period only")
    summary = M.run(period, log=True)
    out = C.RESULTS_DIR / "is"
    out.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary["_returns"].to_csv(out / "ifc_rebalance_returns.csv", float_format="%.8g")
    figs = figures(summary, C.FIG_DIR)
    js = M.to_jsonable(summary)
    js["figures"] = [str(p.relative_to(C.ROOT)).replace("\\", "/") for p in figs]
    (out / "ifc_rebalance.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    s = summary["stats"]
    print(pd.DataFrame(s).T[["sharpe", "ann_return", "ann_vol", "max_drawdown", "turnover_per_year"]].round(4))


if __name__ == "__main__":
    main()
