"""Check the native F1 (strategies/gqh_flow_clock.py) against our engine, layer by layer, inside the starter kit.

1. Signals and ensemble: the strategy's daily internals (its ``dump``) against forward.stream_frames("F1", "FWD")
   and ensemble.combine (each stream's holdings and decisions, paper excess returns, gross, lam, k) and its paper
   ensemble return against forward.returns("F1", "FWD").
2. Book: run_in_starter.py runs of the strategy (fill native / coc, TSMOM re-scale at the close / open; costs none,
   project, guide), their daily returns in excess of the T-bill against engine references at matched cost rates:
     official    forward.returns("F1", "FWD"): each stream at its own costs, overlay re-scaling costs, short borrow;
     zero_cost   rf + sum_s lam_s k ex_s before any cost (the same scales), for --costs none;
     guide_cost  the same at 10 bp per stream trade and per overlay trade, plus short borrow, for --costs guide.
   Backtrader books a close fill's commission (and the guide slippage debited for it) on the fill session t-1; the
   engine books the cost of a position taken at the close of t-1 on return day t. The ``aligned`` figures move those
   costs one session later; nothing else changes.
3. Attribution: agreement between runs that differ in one execution choice only.

Writes <out>/summary.json with each run's JSON, daily CSV, internals dump and log. Reads only the FWD period, never
calls engine.run_backtest and writes nothing under results/.

Run from the repo root (Git Bash):
    GQH_DATA_DIR=<cache> GQH_STARTER_KIT=<kit root> python validation/starter_kit/check_flow_clock.py --out <dir> \\
        [--replays <folder with the harness's replay_f1_*.csv series>]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C                                  # noqa: E402
from src import engine as E                                  # noqa: E402
from src import ensemble as EN                               # noqa: E402
from src import forward as F                                 # noqa: E402
from validation.starter_kit import run_in_starter as H       # noqa: E402

GUIDE_BPS = (H.GUIDE_COMMISSION + H.GUIDE_SLIPPAGE) * 1e4
WINDOWS = {"in_sample": (pd.Timestamp("2006-05-08"), pd.Timestamp(C.IS_END)),
           "out_of_sample": (pd.Timestamp(C.OOS_START), pd.Timestamp(C.OOS_END))}
# name -> (strategy params, harness cost flags, matched reference)
RUNS = {
    "native_none": ("fill=native", ["--costs", "none"], "zero_cost"),
    "native_project": ("fill=native", ["--costs", "project"], "official"),
    "native_guide": ("fill=native", ["--costs", "guide"], "guide_cost"),
    "native_guide_literal": ("fill=native", ["--costs", "guide", "--no-extra-costs"], "guide_cost"),
    "open_none": ("fill=native,tsmom_scale=open", ["--costs", "none"], "zero_cost"),
    "open_project": ("fill=native,tsmom_scale=open", ["--costs", "project"], "official"),
    "coc_none": ("fill=coc", ["--costs", "none"], "zero_cost"),
    "coc_project": ("fill=coc", ["--costs", "project"], "official"),
    "coc_guide": ("fill=coc", ["--costs", "guide"], "guide_cost"),
}
PAIRS = {   # (a, b, what differs)
    "close_order_sizing": ("native_none", "coc_none", "A/C and TSMOM re-scale by bt.Order.Close sized a session "
                                                      "early vs cheat-on-close at exact weights"),
    "tsmom_open_sizing": ("coc_none", "@zero_cost", "TSMOM market orders sized at the previous close vs the engine's "
                                                    "exact weights at the open"),
    "tsmom_rescale_at_open": ("open_none", "native_none", "TSMOM re-scaled at the open vs at the close"),
    "cost_netting_and_timing": ("coc_project", "@official", "one netted book vs per-stream and overlay costs"),
}
REPLAYS = {"native_project": ["replay_f1_project", "replay_f1_close_project"],
           "native_guide": ["replay_f1_guide", "replay_f1_close_guide"]}


# --------------------------------------------------------------------------- engine side

def references() -> tuple[dict[str, pd.Series], dict, dict, dict]:
    """Official F1 and the two matched-cost references, plus the engine's stream frames and ensemble output."""
    fr = F.stream_frames("F1", "FWD")
    labels = list(fr["ex"].columns)
    out = EN.combine(fr["ex"], fr["gr"], fr["rf"], labels, F.ENSEMBLE["scheme"], F.ENSEMBLE["vol_target"],
                     F.ENSEMBLE["brake"])
    official = F.returns("F1", "FWD")
    idx = official.index
    rf = fr["rf"].reindex(idx)
    scale = out["lam"][labels].mul(out["k"], axis=0)
    G = fr["gr"][labels].reindex(idx)
    overlay = (scale - scale.shift(1).fillna(0.0)).abs() * np.minimum(G, G.shift(1).fillna(0.0))
    refs = {"zero_cost": rf.copy(), "guide_cost": rf.copy(), "official_rebuilt": rf.copy()}
    for s in labels:
        w = fr["w_dec"][s]
        ohlc = E.load_ohlc(list(w.columns), "FWD")
        _, gross, *_ = E.simulate(w, ohlc, fr["rf"], exec=fr["exec"][s])
        guide, *_ = E.simulate(w, ohlc, fr["rf"], exec=fr["exec"][s], cost_bps={t: GUIDE_BPS for t in w.columns})
        refs["zero_cost"] += scale[s] * (gross.reindex(idx) - rf)
        refs["guide_cost"] += scale[s] * (guide.reindex(idx) - rf)
        refs["official_rebuilt"] += scale[s] * fr["ex"][s].reindex(idx)
    refs["guide_cost"] -= overlay.sum(axis=1) * GUIDE_BPS / 1e4
    refs["official_rebuilt"] -= (overlay * pd.Series(EN.STREAM_COST_BPS)[labels] / 1e4).sum(axis=1)
    refs["official"] = official
    checks = {"official_minus_combine_max_abs": float((official - out["net"]).abs().max()),
              "official_rebuilt_max_abs": float((refs.pop("official_rebuilt") - official).abs().max())}
    return refs, fr, out, checks


def _diff(name: str, a: pd.Series, b: pd.Series, tol: float = 1e-12) -> dict:
    a, b = a.align(b, join="inner")
    d = (a - b).abs()
    bad = d[d > tol]
    return {"quantity": name, "max_abs_diff": float(d.max()), "days_over_1e-12": int(len(bad)),
            "days": [str(t.date()) for t in bad.index[:10]]}


def internals(dump: pd.DataFrame, fr: dict, out: dict) -> list[dict]:
    """The strategy's signals and ensemble against the engine's, day by day."""
    h, wd = fr["held"], fr["w_dec"]
    rows = [_diff("held A SPY", dump["heldA_SPY"], h["A"]["SPY"]), _diff("held A IEF", dump["heldA_IEF"], h["A"]["IEF"]),
            _diff("held C IEF", dump["heldC_IEF"], h["C"]["IEF"])]
    rows += [_diff(f"decision TSMOM {t}", dump[f"wT_{t}"], wd["TSMOM"][t]) for t in h["TSMOM"].columns]
    rows += [_diff("close-order target A SPY vs engine decision", dump["ordA_SPY"], wd["A"]["SPY"]),
             _diff("close-order target A IEF vs engine decision", dump["ordA_IEF"], wd["A"]["IEF"]),
             _diff("close-order target C IEF vs engine decision", dump["ordC_IEF"], wd["C"]["IEF"])]
    for s in fr["ex"].columns:
        rows += [_diff(f"paper excess {s}", dump[f"ex_{s}"], fr["ex"][s]), _diff(f"gross {s}", dump[f"gr_{s}"], fr["gr"][s]),
                 _diff(f"lam {s} (relative)", dump[f"lam_{s}"] / out["lam"][s].abs().max(),
                       out["lam"][s] / out["lam"][s].abs().max())]
    rows.append(_diff("k", dump["k"], out["k"]))
    kp = dump["kproxy"].dropna()
    held = (dump["heldA_SPY"].abs() + dump["heldA_IEF"].abs() + dump["heldC_IEF"].abs()).reindex(kp.index) > 0
    rows.append(_diff("k sizing the A/C close orders, on days A or C hold", kp[held], out["k"]))
    rows.append(_diff("k sizing the A/C close orders, all days", kp, out["k"]))
    rows.append(_diff("paper ensemble vs forward.returns F1", dump["paper_net"].dropna(), out["net"]))
    return rows


# --------------------------------------------------------------------------- runs

def run(name: str, out: Path, refs_dir: Path, python: str) -> dict:
    params, cost_flags, ref = RUNS[name]
    cmd = [python, str(HERE / "run_in_starter.py"), "--strategy", "gqh_flow_clock",
           "--params", f"{params},dump={(out / f'{name}_dump.csv').as_posix()}", *cost_flags,
           "--json", str(out / f"{name}.json"), "--series", str(out / f"{name}.csv"),
           "--reference", str(refs_dir / f"ref_{ref}.csv"), "--reference-rf", "tbill"]
    t0 = time.time()
    with open(out / f"{name}.log", "w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"{name} failed ({code}); see {out / f'{name}.log'}")
    return {"name": name, "seconds": round(time.time() - t0, 1)}


def book_excess(out: Path, name: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """A run's daily excess return, the same with close-fill costs moved one session later, and its JSON."""
    daily = pd.read_csv(out / f"{name}.csv", index_col=0, parse_dates=True)
    dump = pd.read_csv(out / f"{name}_dump.csv", index_col=0, parse_dates=True)
    meta = json.loads((out / f"{name}.json").read_text())["meta"]
    late = dump["comm_close_fills"].copy()
    if meta["costs"] == "guide" and meta["extra_costs"]:
        late += dump["notional_close_fills"] * meta["guide_slippage"]       # ExtraCosts' close-fill debit
    nav_prev = dump["nav"].shift(1)
    shift = ((late - late.shift(1).fillna(0.0)) / nav_prev).reindex(daily.index).fillna(0.0)
    return daily["excess"], daily["excess"] + shift, daily


def window_agreement(a: pd.Series, b: pd.Series) -> dict:
    return {w: H.agreement(a.loc[lo:hi], b.loc[lo:hi]) for w, (lo, hi) in WINDOWS.items()}


def sharpes(ex: pd.Series, total: pd.Series) -> dict:
    return {w: {"our_sharpe_excess": H.our_sharpe(ex.loc[lo:hi]), "kit_sharpe_rf0": H.kit_sharpe(total.loc[lo:hi])}
            for w, (lo, hi) in WINDOWS.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--replays", default=None, help="folder with replay_f1_*.csv daily series from run_in_starter.py")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="comma list of runs (default: all)")
    args = ap.parse_args()
    if os.environ.get("GQH_OOS_UNLOCK"):
        raise SystemExit("unset GQH_OOS_UNLOCK: only the FWD period is read")
    if not os.environ.get("GQH_STARTER_KIT"):
        raise SystemExit("set GQH_STARTER_KIT to the unzipped gqh-webull-backtrader-starter")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    refs, fr, eng, ref_checks = references()
    rf = fr["rf"]
    for k, s in refs.items():
        s.rename("ret").rename_axis("date").to_csv(out / f"ref_{k}.csv")
    names = [n for n in RUNS if not args.only or n in args.only.split(",")]
    with ThreadPoolExecutor(args.jobs) as pool:
        timing = list(pool.map(lambda n: run(n, out, out, sys.executable), names))

    ref_ex = {k: s - rf.reindex(s.index) for k, s in refs.items()}
    summary = {"meta": {"script": "validation/starter_kit/check_flow_clock.py", "fwd_data_end": E.data_end("FWD"),
                        "windows": {w: [str(lo.date()), str(hi.date())] for w, (lo, hi) in WINDOWS.items()},
                        "reference_checks": ref_checks, "runs": timing},
               "references": {k: {"sharpe": sharpes(ref_ex[k], refs[k]),
                                  **{w: {"ann_return": H.ann_return(refs[k].loc[lo:hi]),
                                         "max_drawdown": E.max_drawdown(refs[k].loc[lo:hi])}
                                     for w, (lo, hi) in WINDOWS.items()}} for k in refs},
               "runs": {}, "attribution": {}}
    if "native_project" in names:
        dump = pd.read_csv(out / "native_project_dump.csv", index_col=0, parse_dates=True)
        summary["internals_vs_engine"] = internals(dump, fr, eng)
        summary["meta"]["desync_bars"] = int(dump["desync"].iloc[0])

    books = {}
    for name in names:
        res = json.loads((out / f"{name}.json").read_text())
        ex, ex_al, daily = book_excess(out, name)
        books[name] = ex
        ref = RUNS[name][2]
        row = {"params": RUNS[name][0], "costs": res["meta"]["costs"], "extra_costs": res["meta"]["extra_costs"],
               "orders": res["orders"], "windows": {}}
        for w in WINDOWS:
            ws = res["windows"][w]
            row["windows"][w] = {k: ws[k] for k in ("kit_sharpe_rf0", "our_sharpe_excess", "ann_return", "ann_vol",
                                                    "max_drawdown", "mean_gross_weight", "orders", "refused_orders")}
        for w, (lo, hi) in WINDOWS.items():
            row["windows"][w]["our_sharpe_excess_aligned"] = H.our_sharpe(ex_al.loc[lo:hi])
        row["vs_matched_reference"] = {"reference": ref, "raw": window_agreement(ex, ref_ex[ref]),
                                       "aligned": window_agreement(ex_al, ref_ex[ref])}
        row["vs_official"] = {"raw": window_agreement(ex, ref_ex["official"]),
                              "aligned": window_agreement(ex_al, ref_ex["official"])}
        if args.replays:
            row["vs_weight_replays"] = {}
            for rp in REPLAYS.get(name, []):
                f = Path(args.replays) / f"{rp}.csv"
                if f.exists():
                    r = pd.read_csv(f, index_col=0, parse_dates=True)["excess"]
                    row["vs_weight_replays"][rp] = window_agreement(ex, r)
        d = (ex_al - ref_ex[ref]).dropna() * 1e4
        row["largest_aligned_daily_diffs_bp"] = {str(t.date()): float(d[t]) for t in d.abs().nlargest(5).index}
        summary["runs"][name] = row
    for key, (a, b, what) in PAIRS.items():
        if a not in books or (not b.startswith("@") and b not in books):
            continue
        sb = ref_ex[b[1:]] if b.startswith("@") else books[b]
        summary["attribution"][key] = {"a": a, "b": b, "differs_in": what, "agreement": window_agreement(books[a], sb)}
    summary["meta"]["runtime_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(H._clean(summary), indent=2))
    print_table(summary)
    print(f"wrote {out / 'summary.json'} in {summary['meta']['runtime_s']} s")


def print_table(summary: dict) -> None:
    """Per run and window: both Sharpe ratios and agreement with the matched reference (raw / costs aligned)."""
    for name, ref in summary["references"].items():
        print(f"{'ref ' + name:22s}", "  ".join(f"{w[:3]} ours {x['our_sharpe_excess']:+.4f} kit {x['kit_sharpe_rf0']:+.4f}"
                                            for w, x in ref["sharpe"].items()))
    for name, row in summary["runs"].items():
        for w in WINDOWS:
            x, m = row["windows"][w], row["vs_matched_reference"]
            raw, al = m["raw"][w], m["aligned"][w]
            print(f"{name:22s} {w[:3]} kit {x['kit_sharpe_rf0']:+.4f} ours {x['our_sharpe_excess']:+.4f} "
                  f"ann {100 * x['ann_return']:+6.2f}% dd {100 * x['max_drawdown']:6.2f}% | vs {m['reference']}: "
                  f"corr {raw['corr']:.5f} {raw['mean_abs_diff_bp']:.2f} bp, aligned {al['corr']:.5f} "
                  f"{al['mean_abs_diff_bp']:.2f} bp")


if __name__ == "__main__":
    main()
