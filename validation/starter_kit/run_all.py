"""Regenerate results/validation/starter_kit/: our four strategies run natively inside the organisers' starter kit.

For the submission (F1, strategies/gqh_flow_clock.py), S2 (strategies/gqh_equity_trend.py), S3 and BH5
(strategies/gqh_gtaa.py) at the kit guide's costs (5 bp commission + 5 bp slippage per side) and at our project
costs, over the full cached history to 2026-10-02, this runs run_in_starter.py and writes:

  results/validation/starter_kit/<name>_<costs>.json   the harness's metrics, plus agreement with our engine at the
                                                       same cost rate (and, for F1, with close-fill costs aligned)
  results/validation/starter_kit/<name>_report.html    the kit's HTML report of the guide-cost run, if under 5 MB;
                                                       larger reports stay in --work
  results/validation/starter_kit/SUMMARY.md            the tables and how the runs map onto the kit

Engine references (cash at the T-bill): "official" is the strategy's own daily net return (forward.returns /
forward2.returns, project costs); "same costs" re-simulates the same decisions at 10 bp one way per trade (and per
roll leg and F1 overlay trade) for the guide runs. Daily series, references and logs go to --work.

Reads only the FWD period: the out-of-sample lock is never released, engine.run_backtest is never called and nothing
is written to results/trials.csv or results/oos_log.csv.

Run from the repo root (Git Bash):
    GQH_DATA_DIR=<cache> GQH_STARTER_KIT=<kit root> python validation/starter_kit/run_all.py --work <scratch dir>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C                                  # noqa: E402
from src import engine as E                                  # noqa: E402
from src import forward2 as F2                               # noqa: E402
from validation.starter_kit import check_flow_clock as CF    # noqa: E402
from validation.starter_kit import run_in_starter as H       # noqa: E402

OUT = ROOT / "results" / "validation" / "starter_kit"
GUIDE_BPS = (H.GUIDE_COMMISSION + H.GUIDE_SLIPPAGE) * 1e4
REPORT_LIMIT = 5 * 2 ** 20
COSTS = ("guide", "project")
WINDOWS = ("in_sample", "out_of_sample")
# name -> (title, harness arguments, broker note)
STRATEGIES = {
    "f1_flow_clock": ("Submission: F1 flow clock (A + C + TSMOM, min-variance ensemble, 8 % vol target, 4x cap)",
                      ["--strategy", "gqh_flow_clock"], "margin account (up to 4x gross), 1e9 cash"),
    "s2_equity_trend": ("S2: equities + trend (30 CME futures total-return indices)",
                        ["--strategy", "gqh_equity_trend"], "margin account (up to 3x gross), 1e9 cash"),
    "s3_gtaa": ("S3: Faber GTAA (SPY EFA IEF VNQ DBC, 10-month mean)",
                ["--strategy", "gqh_gtaa"], "the kit's cash broker, 1e9 cash, 1 % kept in cash"),
    "bh5_buy_hold": ("BH5 yardstick: 20 % each of SPY EFA IEF VNQ DBC",
                     ["--strategy", "gqh_gtaa", "--params", "buy_and_hold=true"],
                     "the kit's cash broker, 1e9 cash, 1 % kept in cash"),
}


# --------------------------------------------------------------------------- engine references

def _save(s: pd.Series, path: Path) -> Path:
    s.rename("ret").rename_axis("date").to_csv(path)
    return path


def references(work: Path) -> dict[tuple[str, str], Path]:
    """(name, "official" | "guide") -> CSV of our engine's daily net returns, cash at the T-bill."""
    rf = E.load_rf("FWD")
    f1, *_ = CF.references()
    refs = {("f1_flow_clock", "official"): _save(f1["official"], work / "ref_f1_flow_clock_official.csv"),
            ("f1_flow_clock", "guide"): _save(f1["guide_cost"], work / "ref_f1_flow_clock_guide.csv")}
    ohlc = E.load_ohlc(F2.S1_TICKERS, "FWD")
    guide, *_ = E.simulate(F2.s2_decisions("FWD"), ohlc, rf, exec="next_close",
                           cost_bps={t: GUIDE_BPS for t in F2.S1_TICKERS})
    refs["s2_equity_trend", "official"] = _save(F2.returns("S2", "FWD"), work / "ref_s2_equity_trend_official.csv")
    refs["s2_equity_trend", "guide"] = _save(guide, work / "ref_s2_equity_trend_guide.csv")
    ohlc = E.load_ohlc(F2.S3_TICKERS, "FWD")
    for name, label, bh in (("s3_gtaa", "S3", False), ("bh5_buy_hold", "BH5", True)):
        guide, *_ = E.simulate(F2.s3_decisions("FWD", buy_and_hold=bh), ohlc, rf, exec="next_open",
                               cost_bps={t: GUIDE_BPS for t in F2.S3_TICKERS})
        refs[name, "official"] = _save(F2.returns(label, "FWD"), work / f"ref_{name}_official.csv")
        refs[name, "guide"] = _save(guide, work / f"ref_{name}_guide.csv")
    return refs


# --------------------------------------------------------------------------- runs

def run(name: str, costs: str, work: Path, refs: dict) -> dict:
    tag = f"{name}_{costs}"
    args = list(STRATEGIES[name][1])
    if name == "f1_flow_clock":                 # internals, for the close-fill cost alignment
        args += ["--params", f"dump={(work / f'{tag}_dump.csv').as_posix()}"]
    cmd = [sys.executable, str(HERE / "run_in_starter.py"), *args, "--costs", costs,
           "--json", str(work / f"{tag}.json"), "--series", str(work / f"{tag}.csv"),
           "--reference", str(refs[name, "official"]), "--reference-rf", "tbill"]
    if costs == "guide":
        cmd += ["--report", str(work / f"{name}_report.html")]
    t0 = time.time()
    with open(work / f"{tag}.log", "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"{tag} failed ({code}); see {work / f'{tag}.log'}")
    return {"run": tag, "seconds": round(time.time() - t0, 1)}


def finish(name: str, costs: str, work: Path, refs: dict) -> dict:
    """The harness's JSON plus agreement with the engine at the same cost rate; written under results/."""
    tag = f"{name}_{costs}"
    res = json.loads((work / f"{tag}.json").read_text())
    daily = pd.read_csv(work / f"{tag}.csv", index_col=0, parse_dates=True)
    wins = H.windows(daily, pd.Timestamp(res["meta"]["is_start"]))
    res["reference"]["role"] = "official engine return (project costs)"
    if costs == "guide":
        res["reference_same_costs"] = H.reference_agreement(daily, str(refs[name, "guide"]), None, "tbill", wins)
        res["reference_same_costs"]["role"] = "engine on the same decisions at 10 bp one way"
    else:
        res["reference_same_costs"] = res["reference"]
    if name == "f1_flow_clock":
        _, aligned, _ = CF.book_excess(work, tag)
        ref = pd.read_csv(refs[name, "guide" if costs == "guide" else "official"], index_col=0, parse_dates=True)
        ref_ex = ref["ret"] - E.load_rf("FWD").reindex(ref.index).ffill().fillna(0.0)
        res["aligned_close_fill_costs"] = {
            "note": "close-fill commission (and the guide slippage debited for it) moved one session later, to the "
                    "engine's booking day; nothing else changes",
            "windows": {w: {"agreement_bt_minus_reference": H.agreement(aligned.loc[lo:hi], ref_ex.loc[lo:hi]),
                            "our_sharpe_excess": H.our_sharpe(aligned.loc[lo:hi])}
                        for w, (lo, hi) in wins.items()}}
    report = work / f"{name}_report.html"
    if costs == "guide":
        size = report.stat().st_size if report.exists() else None
        kept = size is not None and size < REPORT_LIMIT
        if kept:
            shutil.copyfile(report, OUT / report.name)
        res["meta"]["report"] = {"bytes": size, "location": f"results/validation/starter_kit/{report.name}" if kept
                                 else "work folder only (over 5 MB)"}
    if "dump" in res["meta"]["params"]:                     # a work-folder path: keep the file name only
        res["meta"]["params"]["dump"] = Path(res["meta"]["params"]["dump"]).name
    res["meta"]["regenerated_by"] = "validation/starter_kit/run_all.py"
    (OUT / f"{tag}.json").write_text(json.dumps(H._clean(res), indent=2))
    return res


# --------------------------------------------------------------------------- SUMMARY.md

def _f(x, nd=2, pct=False, sign=False) -> str:
    if x is None:
        return "n/a"
    v = 100 * x if pct else x
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def strategy_table(name: str, res: dict[str, dict]) -> list[str]:
    lines = ["| costs | window | kit Sharpe | our Sharpe | engine (ours) | ann. return | max DD | turnover / yr "
             "| refused | corr. vs engine | mean abs diff | corr. same costs | diff same costs |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for costs in COSTS:
        r = res[costs]
        for w in WINDOWS:
            x = r["windows"][w]
            off = r["reference"]["windows"][w]
            same = r["reference_same_costs"]["windows"][w]
            a, b = off["agreement_bt_minus_reference"], same["agreement_bt_minus_reference"]
            lines.append(
                f"| {costs} | {'IS' if w == 'in_sample' else 'OOS'} {x['start'][:4]}-{x['end'][:4]} "
                f"| {_f(x['kit_sharpe_rf0'], 3)} | {_f(x['our_sharpe_excess'], 3)} "
                f"| {_f(same['reference_our_sharpe_excess'], 3)} | {_f(x['ann_return'], 1, pct=True)} % "
                f"| {_f(x['max_drawdown'], 1, pct=True)} % | {_f(x['turnover_per_year'], 1)}x | {x['refused_orders']} "
                f"| {a['corr']:.4f} | {a['mean_abs_diff_bp']:.2f} bp | {b['corr']:.4f} | {b['mean_abs_diff_bp']:.2f} bp |")
    return lines


def _min_corr(res: dict[str, dict], key: str = "reference_same_costs", costs=COSTS) -> float:
    return min(res[c][key]["windows"][w]["agreement_bt_minus_reference"]["corr"] for c in costs for w in WINDOWS)


def summary(results: dict[str, dict[str, dict]], timing: list[dict]) -> str:
    f1, s2, s3, bh = (results[n] for n in STRATEGIES)
    win = lambda r, w: r["windows"][w]                                          # noqa: E731
    al = {c: f1[c]["aligned_close_fill_costs"]["windows"] for c in COSTS}
    al_shift = max(abs(al[c][w]["our_sharpe_excess"] - win(f1[c], w)["our_sharpe_excess"])
                   for c in COSTS for w in WINDOWS)
    netting = [100 * al["project"][w]["agreement_bt_minus_reference"]["tracking_diff_ann"] for w in WINDOWS]
    f1_al_min = min(al[c][w]["agreement_bt_minus_reference"]["corr"] for c in COSTS for w in WINDOWS)
    sizes = [results[n]["guide"]["meta"]["report"]["bytes"] or 0 for n in STRATEGIES]
    kept = [n for n in STRATEGIES if results[n]["guide"]["meta"]["report"]["location"].startswith("results")]
    data_end = f1["guide"]["meta"]["fwd_data_end"]
    out = [
        "# Our strategies in the organisers' starter kit",
        "",
        f"Generated by `validation/starter_kit/run_all.py` on {time.strftime('%Y-%m-%d')} with backtrader "
        f"{f1['guide']['meta']['backtrader_version']}, cached data through {data_end}. The four strategies run inside "
        "the kit's own backtest code (`examples/backtest/main.py`, imported at runtime), and each one computes its "
        "signals inside backtrader from the price feeds it receives. None of them reads our engine's weights.",
        "",
        "## How to read the tables",
        "",
        "* **Windows.** In-sample (IS) runs from each strategy's first live day to 2024-10-02. Out-of-sample (OOS) "
        "runs from 2024-10-03 to 2026-10-02.",
        "* **Costs.** *guide* is the kit's suggestion in docs/USAGE_EN.md: 5 bp commission plus 5 bp slippage per "
        "side. The harness charges the slippage that backtrader leaves out: on `bt.Order.Close` fills it applies none, "
        "and on fills at the open it caps the slip at the bar's high or low. Each futures roll is charged as an extra "
        "round trip, 10 bp per leg. *project* uses our engine's per-instrument one-way costs, plus roll legs and short "
        "borrow.",
        "* **Kit Sharpe.** The kit's `SharpeRatio` analyzer: daily backtrader returns, rf = 0, population std, "
        "times sqrt(252). Backtrader pays no interest on cash, so this figure is not comparable with our engine's.",
        "* **Our Sharpe.** Daily excess return over the T-bill, sample std, times sqrt(252). The excess return is "
        "the backtrader return minus the net weight held times the T-bill, which matches our engine's convention. "
        "*Engine (ours)* is our engine's Sharpe on the same days, at the same cost rate.",
        "* **Turnover.** Notional filled per year as a multiple of NAV, one way. Roll legs are not included.",
        "* **Refused** counts orders backtrader rejected or cancelled for margin.",
        "* **Agreement** compares daily excess returns. *vs engine* means the strategy's official engine return at "
        "project costs. *Same costs* means our engine on the same decisions at the run's cost rate: 10 bp one way "
        "for guide runs, and the official return for project runs.",
        "",
    ]
    for name, (title, _, broker) in STRATEGIES.items():
        r = results[name]
        out += [f"## {title}", "", f"Broker: {broker}. Strategy module: `{r['guide']['meta']['module']}`.", ""]
        out += strategy_table(name, r)
        out.append("")
        km = r["guide"]["kit_metrics"]
        out.append(f"The kit's own summary of the guide-cost run (full history): final value "
                   f"{km['final_value']:,.0f} from {r['guide']['meta']['cash']:,.0f}, Sharpe analyzer "
                   f"{_f(km['sharpe_ratio'], 4)}, {r['guide']['orders']['total']:,} orders, "
                   f"{r['guide']['orders']['refused']} refused. Report: {r['guide']['meta']['report']['location']}.")
        out.append("")
        if name == "f1_flow_clock":
            out += [
                "At guide costs the raw daily correlation with the same-cost engine return is low, "
                f"{win(r['guide']['reference_same_costs'], 'in_sample')['agreement_bt_minus_reference']['corr']:.3f} "
                "IS and "
                f"{win(r['guide']['reference_same_costs'], 'out_of_sample')['agreement_bt_minus_reference']['corr']:.3f}"
                " OOS. This comes from when costs are booked, not from what is traded. Backtrader books a close "
                "fill's commission on the fill session. The engine books it on the next return day. The A and C "
                "window trades are about 3x NAV, so roughly 30 bp lands one day early. With those costs moved "
                "one session later, the correlation is "
                f"{al['guide']['in_sample']['agreement_bt_minus_reference']['corr']:.4f} / "
                f"{al['guide']['out_of_sample']['agreement_bt_minus_reference']['corr']:.4f} "
                f"({al['guide']['in_sample']['agreement_bt_minus_reference']['mean_abs_diff_bp']:.2f} / "
                f"{al['guide']['out_of_sample']['agreement_bt_minus_reference']['mean_abs_diff_bp']:.2f} bp), and "
                f"our Sharpe changes by at most {al_shift:.3f}. At project costs, the aligned figures are "
                f"{al['project']['in_sample']['agreement_bt_minus_reference']['corr']:.4f} / "
                f"{al['project']['out_of_sample']['agreement_bt_minus_reference']['corr']:.4f}. The native book "
                "has a higher project-cost Sharpe than the official F1 because one netted book pays less than the "
                f"engine's per-stream accounting does: once aligned, it earns {netting[0]:.2f} % a year more "
                f"in-sample and {netting[1]:.2f} % out-of-sample.",
                ""]
        if name == "s2_equity_trend":
            out += [
                "S2 is sized at the close of d and fills at the close of d+1 (`bt.Order.Close`), so its weights "
                "drift for one session before the fill. The trade cost is also booked one day earlier than in the "
                "engine. At guide costs, most of the extra cost comes from rolling a book of about "
                f"{win(r['guide'], 'in_sample')['mean_gross_weight']:.1f}x gross, charged at 10 bp a leg. Whether "
                "the guide's stock and ETF figure should apply to futures rolls is a judgement call.",
                ""]
    out += [
        "## What the numbers say",
        "",
        f"* **Submission (F1).** At the kit guide's 5 + 5 bp, our Sharpe is "
        f"{_f(win(f1['guide'], 'in_sample')['our_sharpe_excess'])} in-sample and "
        f"{_f(win(f1['guide'], 'out_of_sample')['our_sharpe_excess'])} out-of-sample (kit Sharpe "
        f"{_f(win(f1['guide'], 'in_sample')['kit_sharpe_rf0'])} / "
        f"{_f(win(f1['guide'], 'out_of_sample')['kit_sharpe_rf0'])}). At project costs it is "
        f"{_f(win(f1['project'], 'in_sample')['our_sharpe_excess'])} / "
        f"{_f(win(f1['project'], 'out_of_sample')['our_sharpe_excess'])}. The book turns over about "
        f"{win(f1['guide'], 'in_sample')['turnover_per_year']:.0f}x NAV a year, so the cost assumption decides the "
        "result. If close fills are left unslipped, as the kit's literal setters would do, guide costs look much "
        "better than they are (`--no-extra-costs`; see the README).",
        f"* **S2.** {_f(win(s2['project'], 'in_sample')['our_sharpe_excess'])} / "
        f"{_f(win(s2['project'], 'out_of_sample')['our_sharpe_excess'])} at project costs, "
        f"{_f(win(s2['guide'], 'in_sample')['our_sharpe_excess'])} / "
        f"{_f(win(s2['guide'], 'out_of_sample')['our_sharpe_excess'])} at guide costs.",
        f"* **S3.** {_f(win(s3['project'], 'in_sample')['our_sharpe_excess'])} / "
        f"{_f(win(s3['project'], 'out_of_sample')['our_sharpe_excess'])} at project costs, "
        f"{_f(win(s3['guide'], 'in_sample')['our_sharpe_excess'])} / "
        f"{_f(win(s3['guide'], 'out_of_sample')['our_sharpe_excess'])} at guide costs. It trades little, so the "
        f"cost setting barely matters. BH5: {_f(win(bh['guide'], 'in_sample')['our_sharpe_excess'])} / "
        f"{_f(win(bh['guide'], 'out_of_sample')['our_sharpe_excess'])} at guide costs.",
        "* **Agreement with our engine** (lowest daily correlation over both cost settings and both windows, "
        f"same-cost engine return): S3 {_min_corr(s3):.5f}, BH5 {_min_corr(bh):.5f}, S2 {_min_corr(s2):.4f}, F1 "
        f"{_min_corr(f1, costs=('project',)):.4f} at project costs and {_min_corr(f1, costs=('guide',)):.4f} at "
        f"guide costs, or {f1_al_min:.4f} with the cost booking day aligned. The remaining differences come from "
        "execution (sizing at the close before a fill at the next open or close) and from the day costs are booked. "
        "The signals agree exactly: in the per-strategy checks, every decision equals the engine's "
        "(check_flow_clock.py, check_gtaa.py and the S2 module docstring).",
        "",
        "## How this maps to the kit",
        "",
        "Identical to `examples/backtest/main.py`, and called from the kit at runtime rather than copied:",
        "",
        "* the strategy loader (module-level `STRATEGY_CLASS`), the `WEBULL_STRATEGY_PARAMS` parser and the symbol "
        "parser;",
        "* `bt.Cerebro()` with its default observers and the `FixedSize(stake=10)` sizer. Our strategies size their own "
        "orders, so the sizer is never used;",
        "* the analyzers `DrawDown`, `TradeAnalyzer`, `SharpeRatio` (data0's timeframe, rf = 0, annualised) and the "
        "kit's `RecorderAnalyzer`;",
        "* `cerebro.run(runonce=False)`, `_compute_metrics`, `_print_results` and the kit's HTML report renderer.",
        "",
        "Substituted, and why:",
        "",
        "* **Data.** `WebullData` needs the Webull SDK and account credentials, which we do not have. It also returns "
        "at most 1,200 bars, under five years of daily data. The strategies need 16 to 22 years of history for their "
        "warm-ups and the in-sample window. The harness instead feeds `PandasData` from our cache: ETFs, and CME futures as "
        "fully collateralised total-return indices, priced like an ETF with no multiplier. The Webull SDK modules "
        "are replaced by placeholders so that the kit's code can be imported.",
        "* **Costs.** The kit ships with none. The guide's 5 + 5 bp is set with the kit's own setters. The harness "
        "debits what backtrader misses: slippage on close fills and capped open fills, futures rolls and short "
        "borrow. Project costs are our engine's per-instrument rates.",
        "* **Leverage.** F1 reaches 4x gross and S2 3x. Backtrader's cash broker cancels any fill that would take "
        "cash below zero, so these run on a margin account: each commission scheme has leverage 10, and the "
        "strategies size orders themselves. S3 and BH5 stay on the kit's cash broker with 1 % of NAV kept as cash.",
        "* **Cash.** The starting cash is 1e9 instead of the kit's 100,000, because whole shares at 100,000 distort "
        "the target weights. Backtrader pays no interest on idle cash and charges none on borrowed cash. 'Our "
        "Sharpe' adds that back, as the net weight held times the T-bill, so that it matches the engine's "
        "convention. The kit Sharpe is left as the kit computes it.",
        "* **Execution.** The engine's `next_open` is a plain market order placed at the close of d, filling at the "
        "open of d+1. `next_close` is a `bt.Order.Close` placed at the close of d, filling at the close of d+1. Both "
        "are sized at the close of d, so weights drift before the fill. This is the main execution difference from "
        "the engine.",
        "",
        "## Caveats",
        "",
        "* Booking-day and sizing differences are measured and explained in check_flow_clock.py, check_gtaa.py and "
        "the strategy docstrings.",
        "* The F1 book's realised gross goes slightly above the 4x cap on a few sessions, because orders are sized "
        f"a session before they fill: at most {win(f1['project'], 'full')['max_gross_weight']:.3f}x.",
        "* The harness books extra costs through backtrader's private `broker._get_value()` and reads the broker's "
        "order queues. It has been tested with backtrader 1.9.78.123 only.",
        "* `engine.simulate`'s `next_open` formula applies the overnight move to the target weights, not to the "
        "weights drifted since the open fill. This overstates the S3 and BH5 engine Sharpe by 0.001 to 0.005 "
        "(check_gtaa.py). The engine was not changed.",
        "* The 1 % cash reserve for S3 and BH5 on the cash broker was chosen by counting refused orders over the "
        "whole sample, out-of-sample years included.",
        "* The kit's `TradeAnalyzer` counts only trades that went flat. For books rebalanced daily, its win rate "
        "and net P&L are not meaningful.",
        "",
        "## Files",
        "",
        "* `<name>_guide.json`, `<name>_project.json`: the harness's metrics for each run. They hold the kit's "
        "metrics, per-window statistics under both Sharpe definitions, order counts, the extra costs, and agreement "
        "with the engine (`reference`, `reference_same_costs`, and for F1 `aligned_close_fill_costs`).",
        f"* The kit's HTML reports of the guide-cost runs are {min(sizes) / 2 ** 20:.1f} to "
        f"{max(sizes) / 2 ** 20:.1f} MB, because they plot every fill. Reports under 5 MB are copied here "
        f"({', '.join(kept) or 'none'}); the others stay in the work folder.",
        "* Runs: " + ", ".join(f"{t['run']} {t['seconds']:.0f} s" for t in timing) + ".",
        "",
        "Regenerate with `python validation/starter_kit/run_all.py --work <scratch dir>` from the repo root, with "
        "`GQH_DATA_DIR` and `GQH_STARTER_KIT` set (validation/starter_kit/README.md).",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work", required=True, help="folder for daily series, references, logs and large reports")
    ap.add_argument("--jobs", type=int, default=4, help="harness runs in parallel")
    args = ap.parse_args()
    if os.environ.get("GQH_OOS_UNLOCK"):
        raise SystemExit("unset GQH_OOS_UNLOCK: only the FWD period is read")
    if not os.environ.get("GQH_STARTER_KIT"):
        raise SystemExit("set GQH_STARTER_KIT to the unzipped gqh-webull-backtrader-starter")
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    refs = references(work)
    todo = [(n, c) for n in STRATEGIES for c in COSTS]
    with ThreadPoolExecutor(args.jobs) as pool:
        timing = list(pool.map(lambda nc: run(*nc, work, refs), todo))
    results = {n: {} for n in STRATEGIES}
    for n, c in todo:
        results[n][c] = finish(n, c, work, refs)
    (OUT / "SUMMARY.md").write_text(summary(results, timing), encoding="utf-8")
    for n in STRATEGIES:
        for c in COSTS:
            for w in WINDOWS:
                x = results[n][c]["windows"][w]
                a = results[n][c]["reference_same_costs"]["windows"][w]["agreement_bt_minus_reference"]
                print(f"{n:16s} {c:8s} {w:14s} kit {x['kit_sharpe_rf0']:+.4f} ours {x['our_sharpe_excess']:+.4f} "
                      f"ann {100 * x['ann_return']:+6.2f}% dd {100 * x['max_drawdown']:6.2f}% "
                      f"turn {x['turnover_per_year']:6.1f} refused {x['refused_orders']} | same-cost engine corr "
                      f"{a['corr']:.5f} {a['mean_abs_diff_bp']:.2f} bp")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
