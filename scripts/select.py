"""Apply the committed selection rule and freeze the choice in results/selection.json.

Amendment 1 (HYPOTHESES.md section 6): the submitted strategy is the ensemble configuration with
the highest in-sample Deflated Sharpe Ratio that passes gates G1-G3, provided PBO < 0.25; otherwise
the section 0 rule (IFC base vs PCT base) decides. This file is committed before the out-of-sample
window is opened; scripts/run_oos.py reads it and nothing else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402
from src import ensemble as EN  # noqa: E402


def main() -> None:
    fte = json.loads((C.RESULTS_DIR / "is" / "fte.json").read_text())
    ifc = json.loads((C.RESULTS_DIR / "is" / "ifc.json").read_text())
    pct = json.loads((C.RESULTS_DIR / "is" / "pct.json").read_text())
    n_trials, var_sr = E.trial_count(family=None, period="IS")
    sel = {"rule": "Amendment 1: highest in-sample DSR among FTE configurations passing G1-G3, if PBO < 0.25; "
                   "else section 0 (IFC vs PCT).",
           "pbo": fte["pbo"]["pbo"], "n_trials_is": n_trials}
    if fte.get("selected") and fte["pbo"]["pbo"] < 0.25:
        name = fte["selected"]
        labels, scheme, vt, br = name.split("|")
        cfg = {"labels": labels.split("+"), "scheme": scheme, "vol_target": int(vt[2:]) / 100,
               "brake": br == "brake"}
        sel.update({"submitted": "FTE", "fte_selected": name, "fte_config": cfg, "is_stats": fte["selected_stats"]})
        ex, gr, rf = EN.stream_frame("IS", labels=cfg["labels"])
        out = EN.combine(ex, gr, rf, cfg["labels"], cfg["scheme"], cfg["vol_target"], cfg["brake"])
        to = EN.stream_turnover("IS", cfg["labels"])
        tt = EN.total_turnover(out, to, cfg["labels"])
        sel["is_total_turnover_per_year"] = float(tt.sum() / (len(tt) / C.TRADING_DAYS))
        sel["is_avg_weights"] = {k: float(v) for k, v in (out["lam"] * out["k"].values[:, None]).mean().items()}
    else:
        sel.update({"submitted": "section0", "fte_selected": None})
    # section 0 comparison, for the record
    sel["section0_reference"] = {
        "IFC_base_sharpe": ifc["base"]["sharpe"], "IFC_2x": ifc["base_2x_costs"]["sharpe"],
        "IFC_neighbourhood_median": ifc["neighbourhood_median_sharpe"],
        "PCT_base_sharpe": pct["base_stats"]["sharpe"],
    }
    (C.RESULTS_DIR / "selection.json").write_text(json.dumps(sel, indent=2, default=str))
    print(json.dumps({k: sel[k] for k in ("submitted", "fte_selected", "pbo", "n_trials_is")}, indent=2))
    print("total turnover/yr", round(sel.get("is_total_turnover_per_year", float("nan")), 1))


if __name__ == "__main__":
    main()
