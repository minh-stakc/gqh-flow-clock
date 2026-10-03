"""Out-of-sample evaluation (2024-10-03 .. 2026-10-02). Requires GQH_OOS_UNLOCK=1.

Written and committed before the out-of-sample window was opened. It evaluates, once, the base
specification of every pre-registered strategy and the configuration chosen by the committed
selection (results/selection.json), at 1x and 2x costs, and writes results/oos/summary.json.
Nothing here feeds back into any parameter.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src import analysis as AN  # noqa: E402
from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402
from src import ensemble as EN  # noqa: E402

OUT = C.RESULTS_DIR / "oos"
STANDALONE = {  # label -> (module, variant or None)
    "A_rebalance": ("ifc_rebalance", None),
    "B_dash": ("ifc_dash", None),
    "C_treasury": ("ifc_treasury", None),
    "D_fx": ("ifc_fx", None),
    "E_auction": ("ifc_auction", None),
    "PCT": ("pct", None),
    "TSMOM_benchmark": ("pct", "tsmom_benchmark"),
    "PCT_ID_variant": ("pct", "id_variant"),
}


def _run_module(label: str, modname: str, variant: str | None, cost_mult: float) -> E.BacktestResult:
    mod = importlib.import_module(f"src.strategies.{modname}")
    params = EN._variant_params(mod, variant)
    params.pop("cost_mult", None)
    w = mod.decision_weights(period="OOS", **params)
    ohlc = E.load_ohlc(list(w.columns), "OOS")
    rf = E.load_rf("OOS")
    return E.run_backtest(f"OOS_{label}", mod.FAMILY, w, ohlc, rf, period="OOS", params=params,
                          exec=mod.EXEC, cost_mult=cost_mult, cost_bps=getattr(mod, "COST_BPS", None))


def main() -> None:
    if not E.oos_unlocked():
        raise SystemExit("GQH_OOS_UNLOCK=1 required")
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((C.RESULTS_DIR / "selection.json").read_text())
    summary: dict = {"selection": sel, "standalone": {}, "standalone_2x": {}}
    curves = {}
    for label, (m, v) in STANDALONE.items():
        r = _run_module(label, m, v, 1.0)
        summary["standalone"][label] = r.stats
        summary["standalone_2x"][label] = _run_module(label, m, v, 2.0).stats
        curves[label] = r.returns

    from src.strategies import ifc

    for fut in (False, True):
        if fut and not (C.DATA_DIR / "futures_1600.parquet").exists():
            continue
        inst = ifc.FUTURES_PARAMS if fut else {"equity": "SPY", "bond": "IEF"}
        cost = ifc.FUTURES_COST_BPS if fut else None
        tag = "IFC_F" if fut else "IFC"
        ohlc = E.load_ohlc([inst["equity"], inst["bond"]], "OOS")
        rf = E.load_rf("OOS")
        w = ifc.decision_weights("OOS", **inst)
        r = E.run_backtest(f"OOS_{tag}_composite", tag.replace("IFC_F", "IFC_F"), w, ohlc, rf, period="OOS",
                           params=inst, exec=ifc.EXEC, cost_bps=cost)
        r2 = E.run_backtest(f"OOS_{tag}_composite", tag, w, ohlc, rf, period="OOS", params=inst, exec=ifc.EXEC,
                            cost_mult=2.0, cost_bps=cost)
        summary[tag] = {"stats": r.stats, "stats_2x": r2.stats, "factor_to_2026_08": AN.factor_table(r, "OOS")}
        curves[tag] = r.returns

    # PCT-F replication on 20 CME futures (Databento)
    if (C.DATA_DIR / "futures_daily.parquet").exists():
        from scripts import run_pct_f as PF

        ohlc = E.load_ohlc(PF.TICKERS, "OOS")
        rf = E.load_rf("OOS")
        summary["PCT_F"] = {}
        for name, ov in [("base", {}), ("tsmom_benchmark", {"measure": "none"}), ("id_variant", {"measure": "ID"})]:
            w = PF.weights("OOS", **ov)
            r = E.run_backtest(f"OOS_PCTF_{name}", PF.FAMILY, w, ohlc, rf, period="OOS",
                               params={**ov, "universe": "futures20"}, exec=PF.EXEC, cost_bps=PF.COST_BPS)
            summary["PCT_F"][name] = r.stats

    # the ensemble configuration chosen in-sample (Amendment 1), evaluated on the OOS window
    if sel.get("fte_selected"):
        cfg = sel["fte_config"]
        for cm in (1.0, 2.0):
            ex, gr, rf = EN.stream_frame("FULL", cost_mult=cm)
            out = EN.combine(ex, gr, rf, cfg["labels"], cfg["scheme"], cfg["vol_target"], cfg["brake"], cost_mult=cm)
            sl = slice(pd.Timestamp(C.OOS_START), pd.Timestamp(C.OOS_END))
            o = {k: (v.loc[sl] if hasattr(v, "loc") else v) for k, v in out.items()}
            res = EN.to_result(f"OOS_FTE_{sel['fte_selected']}", "OOS", o, cfg)
            E.log_trials_bulk([res], "FTE", cm, note="OOS evaluation of the selected configuration")
            key = "FTE_selected" if cm == 1.0 else "FTE_selected_2x"
            summary[key] = res.stats
            if cm == 1.0:
                to = EN.stream_turnover("FULL", cfg["labels"])
                tt = EN.total_turnover(out, to, cfg["labels"]).loc[sl]
                summary["FTE_selected_total_turnover_per_year"] = float(tt.sum() / (len(tt) / C.TRADING_DAYS))
                summary["FTE_selected_factor_to_2026_08"] = E.factor_regression(res.excess, AN.factor_panel("OOS"))
                curves["FTE_selected"] = res.returns
                full = EN.to_result("FULL", "FULL", out, cfg)
                full.returns.to_frame("net").to_csv(OUT / "fte_selected_full_returns.csv")

    pd.DataFrame(curves).to_csv(OUT / "oos_returns.csv")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: (v.get("sharpe") if isinstance(v, dict) and "sharpe" in v else None)
                      for k, v in summary["standalone"].items()}, indent=2))
    for k in ("IFC", "IFC_F"):
        if k in summary:
            print(k, round(summary[k]["stats"]["sharpe"], 3))
    if "FTE_selected" in summary:
        print("FTE_selected", round(summary["FTE_selected"]["sharpe"], 3))


if __name__ == "__main__":
    main()
