"""Build the quant note's numbers (note/numbers.tex), figures (note/fig_*.pdf) and the PDF.

Every number quoted in note/quant_note.tex is a LaTeX macro defined here from results/*.json,
so the note cannot drift from the code. Run after run_all.py and run_all.py --final (which
also runs scripts/diagnostics.py). Needs the data cache (GQH_DATA_DIR) for the T-bill series, no unlock:

    python note/make_note.py           # writes numbers.tex, figures, and compiles quant_note.pdf
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402

NOTE = ROOT / "note"
R = C.RESULTS_DIR
J = lambda p: json.loads((R / p).read_text())  # noqa: E731


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%".replace("-", "\\ensuremath{-}")


def num(x, d=2):
    return f"{x:.{d}f}".replace("-", "\\ensuremath{-}")


def macros() -> dict:
    sel, oos, fte, diag = J("selection.json"), J("oos/summary.json"), J("is/fte.json"), J("diagnostics.json")
    ifc, ifcf, pctj, pctf = J("is/ifc.json"), J("is/ifc_f.json"), J("is/pct.json"), J("is/pct_f.json")
    t = pd.read_csv(R / "trials.csv")
    t_is = t[t["period"] == "IS"]
    m = {}
    s_is, s_oos = sel["is_stats"], oos["FTE_selected"]
    cfg_row = pd.read_csv(R / "is" / "fte_configs.csv", index_col=0).loc[sel["fte_selected"]]
    m.update({
        "ISsr": num(s_is["sharpe"]), "OOSsr": num(s_oos["sharpe"]),
        "ISret": pct(s_is["ann_return"]), "OOSret": pct(s_oos["ann_return"]),
        "ISvol": pct(s_is["ann_vol"]), "OOSvol": pct(s_oos["ann_vol"]),
        "ISdd": pct(s_is["max_drawdown"]), "OOSdd": pct(s_oos["max_drawdown"]),
        "ISworst": pct(s_is["worst_month"]), "OOSworst": pct(s_oos["worst_month"]),
        "ISsrtwo": num(cfg_row["sharpe_2x"]), "OOSsrtwo": num(oos["FTE_selected_2x"]["sharpe"]),
        "ISto": f"{sel['is_total_turnover_per_year']:.0f}", "OOSto": f"{oos['FTE_selected_total_turnover_per_year']:.0f}",
        "ISgross": num(diag["is_gross_sharpe"]), "OOSgross": num(diag["oos_gross_sharpe"]),
        "ISnw": num(s_is["nw_tstat_mean_excess"]), "OOSnw": num(s_oos["nw_tstat_mean_excess"]),
        "ISskew": num(s_is["skew_daily"]),
        "ISalpha": pct(fte["selected_factor"]["alpha_ann"]), "ISalphat": num(fte["selected_factor"]["alpha_t"]),
        "OOSalpha": pct(oos["FTE_selected_factor_to_2026_08"]["alpha_ann"]),
        "OOSalphat": num(oos["FTE_selected_factor_to_2026_08"]["alpha_t"]),
        "ISbondbeta": num(fte["selected_factor"]["betas"]["Bond"]),
        "ISmktbeta": num(fte["selected_factor"]["betas"]["Mkt_RF"]),
        "ISmombeta": num(fte["selected_factor"]["betas"]["Mom"]),
        "bootlo": num(fte["selected_bootstrap_sharpe_90"][0]), "boothi": num(fte["selected_bootstrap_sharpe_90"][1]),
        "PBO": num(fte["pbo"]["pbo"], 3), "nconfigs": f"{fte['n_configs']:,}", "nsplits": f"{fte['pbo']['n_splits']:,}",
        "ntrials": f"{sel['n_trials_is']:,}", "neff": num(fte["effective_trials_all"], 1),
        "DSR": num(cfg_row["dsr"]), "DSReff": num(cfg_row["dsr_eff_trials"], 3),
        "SRzero": num(cfg_row["sr0_ann"]), "SRzeroeff": num(cfg_row["sr0_ann_eff"]),
        "nbrmed": num(cfg_row["nbr_median_sharpe"]), "bestyr": pct(cfg_row["best_year_share"], 0),
        "cfgmedian": num(fte["config_sharpe_quantiles"]["0.5"]),
        "rollmed": num(diag["rolling2y_is"]["median"]), "rollpfive": num(diag["rolling2y_is"]["p05"]),
        "rollmin": num(diag["rolling2y_is"]["min"]), "rollfrac": pct(diag["rolling2y_is"]["frac_below_oos"], 0),
        "CisIEF": num(diag["mech_IS"]["C_ief_last3_bp_per_day"], 1), "CoosIEF": num(diag["mech_OOS"]["C_ief_last3_bp_per_day"], 1),
        "CisOther": num(diag["mech_IS"]["C_ief_other_bp_per_day"], 1), "CoosOther": num(diag["mech_OOS"]["C_ief_other_bp_per_day"], 1),
        "AisSpr": num(diag["mech_IS"]["A_signed_spread_lastweek_bp_per_day"], 1),
        "AoosSpr": num(diag["mech_OOS"]["A_signed_spread_lastweek_bp_per_day"], 1),
        "contA": pct(diag["oos_contribution_ann"]["A"]), "contC": pct(diag["oos_contribution_ann"]["C"]),
        "contT": pct(diag["oos_contribution_ann"]["TSMOM"]),
        "grossmean": num(diag["gross_exposure"]["mean"]), "grossp": num(diag["gross_exposure"]["p95"]),
        "IFCsr": num(ifc["base"]["sharpe"]), "IFCtwo": num(ifc["base_2x_costs"]["sharpe"]),
        "IFCdd": pct(ifc["base"]["max_drawdown"]), "IFCnbr": num(ifc["neighbourhood_median_sharpe"]),
        "IFCalpha": pct(ifc["factor"]["alpha_ann"]), "IFCalphat": num(ifc["factor"]["alpha_t"]),
        "IFCdelay": num(ifc["variants"]["delay_1d"]["sharpe"]),
        "IFCshiftE": num(ifc["variants"]["shift_early"]["sharpe"]), "IFCshiftL": num(ifc["variants"]["shift_late"]["sharpe"]),
        "IFCFsr": num(ifcf["base"]["sharpe"]), "IFCFtwo": num(ifcf["base_2x_costs"]["sharpe"]),
        "IFCoos": num(oos["IFC"]["stats"]["sharpe"]),
        "IFCFoos": num(oos["IFC_F"]["stats"]["sharpe"]) if "IFC_F" in oos else "n/a",
        "PCTsr": num(pctj["base_stats"]["sharpe"]), "TSMOMsr": num(pctj["variant_sharpe"]["tsmom_benchmark"]),
        "IDsr": num(pctj["variant_sharpe"]["id_variant"]),
        "PCTFsr": num(pctf["base"]["stats"]["sharpe"]), "TSMOMFsr": num(pctf["tsmom_benchmark"]["stats"]["sharpe"]),
        "IDFsr": num(pctf["id_variant"]["stats"]["sharpe"]),
        "PCTFoos": num(oos["PCT_F"]["base"]["sharpe"]) if "PCT_F" in oos else "n/a",
        "TSMOMFoos": num(oos["PCT_F"]["tsmom_benchmark"]["sharpe"]) if "PCT_F" in oos else "n/a",
        "IDFoos": num(oos["PCT_F"]["id_variant"]["sharpe"]) if "PCT_F" in oos else "n/a",
    })
    for k, lab in [("A", "A_rebalance"), ("B", "B_dash"), ("C", "C_treasury"), ("D", "D_fx"), ("E", "E_auction"),
                   ("P", "PCT"), ("T", "TSMOM_benchmark"), ("I", "PCT_ID_variant")]:
        m[f"oos{k}"] = num(oos["standalone"][lab]["sharpe"])
        m[f"oostwo{k}"] = num(oos["standalone_2x"][lab]["sharpe"])
    streams = pd.read_csv(R / "is" / "fte_streams.csv", index_col=0, parse_dates=True)
    for k, lab in [("A", "A"), ("B", "B"), ("C", "C"), ("D", "D"), ("E", "E"), ("P", "PCT"), ("T", "TSMOM")]:
        x = streams[lab].dropna()
        x = x.loc[x.ne(0).idxmax():]
        m[f"is{k}"] = num(x.mean() / x.std() * 252 ** 0.5)
    m["isI"] = m["IDsr"]
    # capacity: ETF ensemble (recent liquidity) and futures IFC-F
    cap = {int(c["aum"]): c["net_sharpe"] for c in diag["capacity_recent"]}
    capf = {int(c["aum"]): c["net_sharpe"] for c in diag.get("capacity_futures_ifc", [])}
    for a, tag in [(10**3, "Zero"), (10**6, "One"), (10**7, "Ten"), (10**8, "Hund"), (10**9, "Bil")]:
        m[f"capE{tag}"] = num(cap.get(a, float("nan")))
        m[f"capF{tag}"] = num(capf.get(a, float("nan")))
    m["nISfam"] = str(t_is["family"].nunique())
    # sub-periods of the selected configuration
    for i, s in enumerate(fte["selected_subperiods"]):
        m[f"sub{'abcde'[i]}"] = num(s["sharpe"])
    # B2 settlement test and sleeve-level facts from module outputs
    dash = J("is/ifc_dash.json")
    m["Bfixed"] = num(next(v["sharpe"] for k, v in dash["variants"].items() if "fixed" in k.lower()) if isinstance(dash.get("variants"), dict) else float("nan"))
    tre = J("is/ifc_treasury.json")
    m["Cq"] = num(tre["variants"]["q1"]["sharpe"]) if "q1" in tre.get("variants", {}) else "n/a"
    rr = dash["sharpe_by_regime_range"]["T+2"]
    m["BtwoFixed"], m["BtwoBase"] = num(rr["fixed_T3"]), num(rr["base"])
    # factor t-statistics, R^2, out-of-sample betas
    fi, fo = fte["selected_factor"], oos["FTE_selected_factor_to_2026_08"]
    m.update({"ISmktt": num(fi["t"]["Mkt_RF"], 1), "ISmomt": num(fi["t"]["Mom"], 1), "ISbondt": num(fi["t"]["Bond"], 1),
              "ISrsq": num(fi["r2"]), "OOSmktbeta": num(fo["betas"]["Mkt_RF"]), "OOSmktt": num(fo["t"]["Mkt_RF"], 1),
              "OOSbondbeta": num(fo["betas"]["Bond"]), "OOSbondt": num(fo["t"]["Bond"], 1)})
    # search, trial accounting and CSCV haircut
    cfgs = pd.read_csv(R / "is" / "fte_configs.csv", index_col=0)
    pre = t_is[pd.to_datetime(t_is["utc"]) < pd.Timestamp("2026-10-03T11:02:45Z")]    # before Amendment 1
    m.update({"ISsrbrake": num(cfgs.loc[sel["fte_selected"].replace("|nobrake", "|brake"), "sharpe"]),
              "nISruns": f"{len(t_is):,}", "nPreAmendRuns": str(len(pre)),
              "nPreAmendSpecs": str(pre[pre["cost_mult"] == 1.0].drop_duplicates(["family", "name", "params_hash"]).shape[0]),
              "cscvIS": num(fte["pbo"]["is_best_sharpe_median"]), "cscvOOS": num(fte["pbo"]["oos_of_is_best_sharpe_median"]),
              "ISkurt": num(s_is["kurtosis_daily"], 1), "ISbest": pct(s_is["best_month"])})
    ci = pctj["predictions"]["P1_id_minus_tsmom"]["ci90"]
    m["IDcilo"], m["IDcihi"] = num(ci[0]), num(ci[1])
    cb = t[(t["name"] == "IFC_C_base") & (t["cost_mult"] == 1.0)].drop_duplicates("params_hash")
    m["CpreFix"], m["CpostFix"] = num(cb["sharpe"].iloc[0]), num(cb["sharpe"].iloc[-1])
    # in-sample contributions, implied out-of-sample gross Sharpe of A and C (2*SR_1x - SR_2x)
    m.update({"contAis": pct(diag["is_contribution_ann"]["A"]), "contCis": pct(diag["is_contribution_ann"]["C"]),
              "contTis": pct(diag["is_contribution_ann"]["TSMOM"])})
    sa, s2 = oos["standalone"], oos["standalone_2x"]
    m["grossOOSA"] = num(2 * sa["A_rebalance"]["sharpe"] - s2["A_rebalance"]["sharpe"])
    m["grossOOSC"] = num(2 * sa["C_treasury"]["sharpe"] - s2["C_treasury"]["sharpe"])
    # data facts used in the text
    mi, mo_ = diag["mech_IS"], diag["mech_OOS"]
    m.update({"nOOSdays": str(mo_["n_days"]), "nOOSmonths": str(mo_["n_months"]),
              "rfIS": pct(mi["rf_avg_ann"]), "rfOOS": pct(mo_["rf_avg_ann"]),
              "AisTone": num(mi["A_signed_spread_T1_bp"], 0),
              "stressOne": pct(4 * 3 ** 0.5 * mi["ief_daily_vol"]), "stressTwo": pct(8 * 3 ** 0.5 * mi["ief_daily_vol"]),
              "capEnet": num(diag["net_sharpe_2019_2024"])})
    if "corr_ES_SPY" in diag:
        dv = diag["median_dollar_volume_2019_2024"]
        m.update({"corrES": num(diag["corr_ES_SPY"], 3), "corrZN": num(diag["corr_ZN_IEF"], 3),
                  "dvIEF": f"{dv['IEF'] / 1e9:.1f}", "dvZN": f"{dv['ZN16'] / 1e9:.0f}", "dvES": f"{dv['ES16'] / 1e9:.0f}"})
    fwd = C.ROOT / "forward" / "historical_context.json"
    if fwd.exists():
        fc = json.loads(fwd.read_text())["strategies"]
        m["FtwoIS"] = num(fc["F2"]["in_sample"]["sharpe"])
        m["FtwoOOS"] = num(fc["F2"]["out_of_sample"]["sharpe"])
    else:
        m["FtwoIS"] = m["FtwoOOS"] = "n/a"
    return m


def write_macros(m: dict) -> None:
    lines = ["% generated by note/make_note.py from results/ -- do not edit by hand"]
    for k, v in m.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    (NOTE / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figures() -> None:
    full = pd.read_csv(R / "oos" / "fte_selected_full_net.csv", index_col=0, parse_dates=True)["FTE_selected"]
    ifc_is = pd.read_csv(R / "is" / "ifc_returns.csv", index_col=0, parse_dates=True)["base"]
    oosr = pd.read_csv(R / "oos" / "oos_returns.csv", index_col=0, parse_dates=True)
    ifc_full = pd.concat([ifc_is, oosr["IFC"].loc[C.OOS_START:]])
    streams = pd.read_csv(R / "is" / "fte_streams.csv", index_col=0, parse_dates=True)
    rf = pd.read_parquet(C.DATA_DIR / "rf_daily.parquet").set_index("date")["rf"]
    rf.index = pd.to_datetime(rf.index)
    ts_is = streams["TSMOM"] + rf.reindex(streams.index).ffill()
    ts_full = pd.concat([ts_is.dropna(), oosr["TSMOM_benchmark"].loc[C.OOS_START:]])
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.6, 2.05))
    for s, lab, col, lw in [(full, "Submitted: A+C+TSMOM ensemble", "#1f4e79", 1.4),
                            (ifc_full, "IFC flow composite (A+B+C)", "#c55a11", 0.9),
                            (ts_full, "Plain TSMOM benchmark", "#7f7f7f", 0.9)]:
        s = s.loc[s.ne(0).idxmax():]
        ax.plot((1 + s).cumprod(), label=lab, color=col, lw=lw)
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 1.5, 2, 3, 4, 6)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.axvspan(pd.Timestamp(C.OOS_START), pd.Timestamp(C.OOS_END), color="#e2efda", alpha=0.8, lw=0)
    ax.text(pd.Timestamp("2025-01-15"), ax.get_ylim()[0] * 1.15, "out-of-sample\n(run once)", fontsize=7)
    ax.set_ylabel("growth of $1, net (log)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(NOTE / "fig_equity.pdf")
    plt.close(fig)

    ev = pd.read_csv(R / "event_time_profiles.csv", header=[0, 1], index_col=0)
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 1.8))
    for ax, col, title in [(axs[0], "ief", "C: IEF excess return (bp/day)"),
                           (axs[1], "rb", "A: drift-signed bond-minus-stock (bp/day)")]:
        for per, c in [("IS", "#1f4e79"), ("OOS", "#c00000")]:
            s = ev[(per, col)]
            if col == "rb":                      # drift is observed at T-6: show only later days
                s = s[s.index >= -5]             # (post-turn days are signed by the month just ended)
            ax.bar(s.index + (-0.2 if per == "IS" else 0.2), s.values, width=0.4, color=c,
                   label=f"{'in-sample 2005-24' if per == 'IS' else 'out-of-sample 2024-26'}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(range(-10, 6, 2))
        ax.set_xticklabels([f"T{x:+d}" if x <= 0 else f"T+{x}" for x in range(-10, 6, 2)], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.25, axis="y")
    axs[0].axvspan(-2.5, 0.5, color="#fff2cc", alpha=0.7, lw=0, zorder=0)
    axs[1].axvspan(-4.5, 0.5, color="#fff2cc", alpha=0.7, lw=0, zorder=0)
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=7.5, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(NOTE / "fig_event.pdf")
    plt.close(fig)


def compile_pdf() -> None:
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "quant_note.tex"],
                       cwd=NOTE, check=True, capture_output=True)


if __name__ == "__main__":
    m = macros()
    write_macros(m)
    figures()
    if (NOTE / "quant_note.tex").exists() and "--no-pdf" not in sys.argv:
        compile_pdf()
    print(f"{len(m)} macros written")
