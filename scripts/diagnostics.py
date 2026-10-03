"""Post-hoc diagnostics for the quant note (run after the out-of-sample evaluation).

Nothing here changes or selects a strategy; it explains the results:
  1. how unusual the OOS Sharpe is relative to the strategy's own 2-year rolling Sharpe in-sample;
  2. whether each mechanism weakened or reversed OOS (raw event-time effects, no strategy code);
  3. gross (pre-cost) vs net OOS Sharpe;
  4. capacity of the selected ensemble's combined ETF positions (square-root impact, recent liquidity).
Writes results/diagnostics.json and figures used in the note.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import analysis as AN  # noqa: E402
from src import calendar_utils as CU  # noqa: E402
from src import config as C  # noqa: E402
from src import engine as E  # noqa: E402
from src import ensemble as EN  # noqa: E402


def rolling_sharpe(x: pd.Series, n: int = 504) -> pd.Series:
    return x.rolling(n).mean() / x.rolling(n).std() * math.sqrt(C.TRADING_DAYS)


def main() -> None:
    if not E.oos_unlocked():
        raise SystemExit("needs GQH_OOS_UNLOCK=1 (reads out-of-sample prices for the diagnostics)")
    sel = json.loads((C.RESULTS_DIR / "selection.json").read_text())
    oos = json.loads((C.RESULTS_DIR / "oos" / "summary.json").read_text())
    cfg = sel["fte_config"]
    labels = cfg["labels"]
    out = {}

    # 1. rolling 2-year Sharpe of the selected ensemble, in-sample
    ex, gr, rf = EN.stream_frame("FULL", labels=labels)
    comb = EN.combine(ex, gr, rf, labels, cfg["scheme"], cfg["vol_target"], cfg["brake"])
    exc = comb["excess"]
    rs = rolling_sharpe(exc.loc[:C.IS_END]).dropna()
    oos_sr = oos["FTE_selected"]["sharpe"]
    out["rolling2y_is"] = {"min": float(rs.min()), "p05": float(rs.quantile(0.05)), "p10": float(rs.quantile(0.10)),
                           "median": float(rs.median()), "frac_below_oos": float((rs <= oos_sr).mean()),
                           "oos_sharpe": oos_sr}
    # gross (before stream and overlay costs) OOS Sharpe of the selected ensemble
    ex0, gr0, rf0 = EN.stream_frame("FULL", labels=labels, cost_mult=0.0)
    comb0 = EN.combine(ex0, gr0, rf0, labels, cfg["scheme"], cfg["vol_target"], cfg["brake"], cost_mult=0.0)
    g = comb0["excess"].loc[C.OOS_START:]
    out["oos_gross_sharpe"] = float(g.mean() / g.std() * math.sqrt(C.TRADING_DAYS))
    gi = comb0["excess"].loc[:C.IS_END]
    out["is_gross_sharpe"] = float(gi.mean() / gi.std() * math.sqrt(C.TRADING_DAYS))
    # per-stream OOS contribution (scaled stream excess return, annualised)
    scale = comb["lam"][labels] * comb["k"].values[:, None]
    contrib = (scale * ex[labels].reindex(scale.index))
    out["oos_contribution_ann"] = {k: float(v) for k, v in (contrib.loc[C.OOS_START:].mean() * C.TRADING_DAYS).items()}
    out["is_contribution_ann"] = {k: float(v) for k, v in (contrib.loc[:C.IS_END].mean() * C.TRADING_DAYS).items()}

    # 2. raw mechanism tests, no strategy code: event-time excess returns relative to month end
    o = E.load_ohlc(["SPY", "IEF"], "FULL")
    rfl = E.load_rf("FULL")
    c = o["close"]
    r = c.pct_change(fill_method=None)
    exr = r.sub(rfl.reindex(r.index).ffill(), axis=0)
    mo = CU.month_offsets(c.index)
    off = mo["off_own"].where(mo["off_own"] >= -10)
    off = off.fillna(mo["off_prev"].where(mo["off_prev"] <= 5))
    # sign of the 60/40 month-to-date drift observed at T-6, applied to SPY-IEF in that month
    prevT = mo["prevT_i"]
    drift = pd.Series(np.nan, index=c.index)
    ce, cb = c["SPY"].to_numpy(), c["IEF"].to_numpy()
    for i, (pt, oo) in enumerate(zip(prevT, mo["off_own"])):
        if oo == -6 and pt >= 0:
            ge, gb = ce[i] / ce[int(pt)], cb[i] / cb[int(pt)]
            drift.iloc[i] = 0.6 * ge / (0.6 * ge + 0.4 * gb) - 0.6
    # each day is signed by the drift observed at T-6 of the month whose window it follows: days up to
    # T use their own month's drift, the first days of the next month use the month that just ended
    msign = np.sign(drift).groupby(mo["month"].values).max()
    own = pd.Series(mo["month"].map(msign).values, index=c.index)
    prev = pd.Series((mo["month"] - 1).map(msign).values, index=c.index)
    post_turn = mo["off_prev"].le(5) & mo["off_own"].lt(-10)          # T+1..T+5 of the previous month
    sgn = pd.Series(np.where(post_turn, prev, own), index=c.index)
    spread_signed = -(exr["SPY"] - exr["IEF"]) * sgn                   # positive if rebalancing pressure pays
    ev = {}
    for name, sl in (("IS", slice(C.HISTORY_START, C.IS_END)), ("OOS", slice(C.OOS_START, C.OOS_END))):
        d = pd.DataFrame({"off": off, "ief": exr["IEF"], "spy": exr["SPY"], "rb": spread_signed}).loc[sl].dropna(subset=["off"])
        prof = d.groupby("off")[["ief", "spy", "rb"]].mean() * 1e4
        ev[name] = prof
        lastk = d["off"].between(-2, 0)
        lastw = d["off"].between(-4, 0)
        allx = exr["IEF"].loc[sl]
        last_all = mo["off_own"].loc[sl].between(-2, 0)
        out[f"mech_{name}"] = {
            "C_ief_last3_bp_per_day": float(d.loc[lastk, "ief"].mean() * 1e4),
            "C_ief_other_bp_per_day": float(d.loc[~lastk, "ief"].mean() * 1e4),          # other days T-10..T+5
            "C_ief_other_alldays_bp_per_day": float(allx[~last_all].mean() * 1e4),
            "A_signed_spread_lastweek_bp_per_day": float(d.loc[lastw, "rb"].mean() * 1e4),
            "A_signed_spread_T1_bp": float(d.loc[d["off"] == 1, "rb"].mean() * 1e4),
            "n_months": int(lastk.sum() / 3),
            "n_days": int(len(allx)),
            "rf_avg_ann": float(rfl.loc[sl].mean() * C.TRADING_DAYS),
            "ief_daily_vol": float(exr["IEF"].loc[sl].std()),
        }
    pd.concat(ev, axis=1).to_csv(C.RESULTS_DIR / "event_time_profiles.csv")

    # 3. capacity of the combined ETF positions of the selected ensemble
    held_sum = None
    for lab in labels:
        modname, variant = EN.STREAMS[lab]
        import importlib

        mod = importlib.import_module(f"src.strategies.{modname}")
        p = EN._variant_params(mod, variant)
        p.pop("cost_mult", None)
        w = mod.decision_weights(period="FULL", **p)
        oh = E.load_ohlc(list(w.columns), "FULL")
        _, _, _, held, _ = E.simulate(w, oh, rfl, exec=mod.EXEC)
        hs = held.mul(scale[lab].reindex(held.index).fillna(0.0), axis=0)
        held_sum = hs if held_sum is None else held_sum.add(hs, fill_value=0.0)
    held_sum = held_sum.fillna(0.0)
    tick = list(held_sum.columns)
    oh = E.load_ohlc(tick, "FULL")
    # capacity starts from GROSS returns (the zero-cost combination); capacity_curve then charges the
    # fixed costs once plus square-root impact on the strategy's actual trades
    gross_ex = comb0["excess"].reindex(exc.index).fillna(0.0)
    res = E.BacktestResult("FTE", "FULL", comb["net"], comb["net"], exc, comb["turnover_overlay"],
                           held_sum.reindex(exc.index).fillna(0.0), gross_ex - exc)
    nx = exc.loc["2019-10-01":C.IS_END]
    out["net_sharpe_2019_2024"] = float(nx.mean() / nx.std() * math.sqrt(C.TRADING_DAYS))
    cap_recent = AN.capacity_curve(res, oh, start="2019-10-01", end=C.IS_END)   # recent in-sample liquidity
    cap_oos = AN.capacity_curve(res, oh, start=C.OOS_START)
    out["capacity_recent"] = cap_recent.to_dict("records")
    # flow sleeves on CME futures (IFC-F), in-sample, with the same AUM grid
    from src.strategies import ifc as IFC

    if (C.DATA_DIR / "futures_1600.parquet").exists():
        ohf = E.load_ohlc(["ES16", "ZN16"], "IS")
        rff = E.load_rf("IS")
        rf_res = E.run_backtest("IFCF_composite_base", "IFC_F", IFC.decision_weights("IS", **IFC.FUTURES_PARAMS), ohf,
                                rff, period="IS", exec=IFC.EXEC, cost_bps=IFC.FUTURES_COST_BPS, log=False)
        out["capacity_futures_ifc"] = AN.capacity_curve(rf_res, ohf).to_dict("records")
        fo = E.load_ohlc(["ES16", "ZN16", "SPY", "IEF"], "FULL")
        fr = fo["close"].pct_change(fill_method=None).loc[:C.IS_END]
        out["corr_ES_SPY"] = float(fr["ES16"].corr(fr["SPY"]))
        out["corr_ZN_IEF"] = float(fr["ZN16"].corr(fr["IEF"]))
        dv = fo["close"] * fo["volume"]
        for nm, sl in (("2019_2024", slice("2019-10-01", C.IS_END)), ("oos", slice(C.OOS_START, C.OOS_END))):
            out[f"median_dollar_volume_{nm}"] = {k: float(v) for k, v in dv.loc[sl].median().items()}
    else:  # no Databento data: keep the committed futures numbers
        prev = C.RESULTS_DIR / "diagnostics.json"
        old = json.loads(prev.read_text()) if prev.exists() else {}
        for k in ("capacity_futures_ifc", "corr_ES_SPY", "corr_ZN_IEF", "median_dollar_volume_2019_2024",
                  "median_dollar_volume_oos"):
            if k in old:
                out[k] = old[k]
    out["capacity_oos_window"] = cap_oos.to_dict("records")
    out["gross_exposure"] = {"mean": float(held_sum.abs().sum(axis=1).mean()),
                             "p95": float(held_sum.abs().sum(axis=1).quantile(0.95)),
                             "max": float(held_sum.abs().sum(axis=1).max())}
    (C.RESULTS_DIR / "diagnostics.json").write_text(json.dumps(out, indent=2, default=str))
    comb["net"].to_frame("FTE_selected").to_csv(C.RESULTS_DIR / "oos" / "fte_selected_full_net.csv")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("capacity")}, indent=2, default=str))
    print("capacity recent", [(f"{c['aum']:.0e}", round(c["net_sharpe"], 2)) for c in out["capacity_recent"]])


if __name__ == "__main__":
    main()
