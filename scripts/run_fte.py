"""Amendment 1: flow-and-trend ensemble (FTE) search, overfitting tests and selection, in-sample.

Runs every pre-registered FTE configuration (HYPOTHESES.md section 6), logs each as a trial,
computes the Probability of Backtest Overfitting (CSCV, 16 blocks), Deflated Sharpe Ratios with the
full in-sample trial count, gates G1-G3, and applies the amendment's selection rule.

    python scripts/run_fte.py            -> results/is/fte.json, results/is/fte_configs.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import analysis as AN  # noqa: E402
from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402
from src import ensemble as EN  # noqa: E402
from src import pbo as P  # noqa: E402

OUT = C.RESULTS_DIR / "is"


def _key(cfg: dict) -> tuple:
    return (frozenset(cfg["labels"]), cfg["scheme"], cfg["vol_target"], cfg["brake"])


def neighbours(cfg: dict, all_cfgs: list[dict]) -> list[dict]:
    """Configurations that differ in exactly one dimension (one stream added/removed or the trend
    leg swapped; or the weighting scheme, the volatility target, or the brake)."""
    out = []
    a = set(cfg["labels"])
    for o in all_cfgs:
        if o is cfg:
            continue
        b = set(o["labels"])
        same_rest = (o["scheme"], o["vol_target"], o["brake"]) == (cfg["scheme"], cfg["vol_target"], cfg["brake"])
        if a == b:
            diffs = sum([o["scheme"] != cfg["scheme"], o["vol_target"] != cfg["vol_target"], o["brake"] != cfg["brake"]])
            if diffs == 1:
                out.append(o)
        elif same_rest:
            sd = a ^ b
            trend_swap = len(sd) == 2 and sd <= {"PCT", "TSMOM"}
            if len(sd) == 1 or trend_swap:
                out.append(o)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ex, gr, rf = EN.stream_frame("IS")
    ex.to_csv(OUT / "fte_streams.csv")
    cfgs = EN.configs()
    print(f"{len(cfgs)} configurations")
    results, rets, outs = [], {}, {}
    for i, cfg in enumerate(cfgs):
        name = EN.config_name(cfg)
        out = EN.combine(ex, gr, rf, cfg["labels"], cfg["scheme"], cfg["vol_target"], cfg["brake"])
        res = EN.to_result(f"FTE_{name}", "IS", out, cfg)
        results.append(res)
        rets[name] = out["excess"]
        outs[name] = res
        if i % 200 == 0:
            print(f"  {i}/{len(cfgs)}")
    E.log_trials_bulk(results, "FTE", 1.0, note="amendment-1 search")

    R = pd.DataFrame(rets)
    start = R.apply(lambda s: s.first_valid_index()).max()
    Rc = R.loc[start:].fillna(0.0)
    pbo = P.cscv_pbo(Rc, n_blocks=16)
    n_eff = P.effective_trials(Rc)
    n_trials, var_sr = E.trial_count(family=None, period="IS", cost_mult=1.0)
    print(f"PBO {pbo['pbo']:.3f}  trials {n_trials}  var(SR) {var_sr:.4f}  effective configs {n_eff:.1f}")

    rows = []
    for cfg in cfgs:
        name = EN.config_name(cfg)
        res = outs[name]
        dsr = E.deflated_sharpe(res.excess, n_trials, var_sr)
        conc = AN.pnl_concentration(res)
        rows.append({"name": name, "labels": "+".join(cfg["labels"]), "scheme": cfg["scheme"],
                     "vol_target": cfg["vol_target"], "brake": cfg["brake"], **{k: res.stats[k] for k in (
                         "start", "ann_return", "ann_vol", "sharpe", "max_drawdown", "calmar", "worst_month")},
                     "dsr": dsr["dsr"], "sr0_ann": dsr["sr0_ann"], "best_year_share": conc["best_year_share"]})
    tab = pd.DataFrame(rows).set_index("name")
    # G3: neighbourhood median Sharpe >= half own Sharpe
    sr = tab["sharpe"]
    tab["nbr_median_sharpe"] = [float(np.median([sr[EN.config_name(o)] for o in neighbours(c, cfgs)])) for c in cfgs]
    tab["G2"] = tab["best_year_share"] <= 0.40
    tab["G3"] = tab["nbr_median_sharpe"] >= 0.5 * tab["sharpe"]
    # G1 at 2x costs for the 40 highest-DSR configurations that pass G2 and G3
    cand = tab[tab["G2"] & tab["G3"]].sort_values("dsr", ascending=False).head(40)
    ex2, gr2, rf2 = EN.stream_frame("IS", cost_mult=2.0)
    res2 = []
    tab["sharpe_2x"] = np.nan
    for name in cand.index:
        cfg = next(c for c in cfgs if EN.config_name(c) == name)
        o2 = EN.combine(ex2, gr2, rf2, cfg["labels"], cfg["scheme"], cfg["vol_target"], cfg["brake"], cost_mult=2.0)
        r2 = EN.to_result(f"FTE_{name}", "IS", o2, cfg)
        res2.append(r2)
        tab.loc[name, "sharpe_2x"] = r2.stats["sharpe"]
    E.log_trials_bulk(res2, "FTE", 2.0, note="amendment-1 G1 check")
    tab["G1"] = tab["sharpe_2x"] > 0
    passing = tab[tab["G1"] & tab["G2"] & tab["G3"]].sort_values("dsr", ascending=False)
    selected = passing.index[0] if (len(passing) and pbo["pbo"] < 0.25) else None
    tab.sort_values("dsr", ascending=False).to_csv(OUT / "fte_configs.csv")

    summary = {
        "n_configs": len(cfgs), "pbo": pbo, "effective_configs": n_eff,
        "n_trials_all_is": n_trials, "var_sharpe_all_is": var_sr,
        "config_sharpe_quantiles": tab["sharpe"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict(),
        "selected": selected,
        "selection_rule": "highest DSR passing G1-G3, provided PBO < 0.25 (else fall back to section 0 rule)",
        "top10_by_dsr_passing": passing.head(10).round(4).reset_index().to_dict("records"),
    }
    if selected:
        res = outs[selected]
        summary["selected_stats"] = res.stats
        summary["selected_subperiods"] = AN.subperiod_table(res).to_dict("records")
        summary["selected_factor"] = E.factor_regression(res.excess, AN.factor_panel("IS"))
        summary["selected_bootstrap_sharpe_90"] = AN.bootstrap_sharpe_ci(res.excess)
        summary["selected_yearly"] = {int(k): float(v) for k, v in AN.yearly_returns(res.returns).items()}
        summary["selected_rank_among_configs"] = int((tab["sharpe"] > tab.loc[selected, "sharpe"]).sum()) + 1
        res.returns.to_frame("net").to_csv(OUT / "fte_selected_returns.csv")
    (OUT / "fte.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("pbo", "selected", "config_sharpe_quantiles")}, indent=2, default=str))


if __name__ == "__main__":
    main()
