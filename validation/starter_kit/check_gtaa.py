"""Check the native S3 / BH5 port (strategies/gqh_gtaa.py) against forward test 2's engine.

Runs run_in_starter.py once per setting and writes into --out:

  engine_<name>_<costs>.csv          our engine's daily net returns on forward2.s3_decisions, cash at the T-bill:
                                     project = forward2.returns(<name>, "FWD"); guide = 10 bp one way (the guide's
                                     5 bp commission + 5 bp slippage); none = no costs
  engine_<name>_<costs>_drift.csv    the same decisions with each weight drifting from the open fill to the next
                                     close (drift_exact); engine.simulate applies the overnight move to the targets
  engine_<name>_held.csv             the engine's held weights, for the replay_weights runs
  <run>.json, <run>.csv, <run>.log   the harness's metrics, daily series and log
  <run>_decisions.csv                the native strategy's month-end decisions
  check_gtaa.json                    per run and window: both Sharpe definitions, annual return, maximum drawdown,
                                     refused orders, agreement of daily excess returns with both engine series
                                     (correlation, mean absolute difference in bp), and whether every month-end
                                     decision equals forward2.s3_decisions

Settings, for S3 and BH5: project and guide costs on three brokers (a margin account at 1e9, the kit's cash broker
at 1e9 and at the kit's 100,000); the kit as shipped (cash broker, 100,000, no costs); and the engine's own held
weights replayed with replay_weights fill=open (margin, project costs), which separates the signal from the timing.

Run from the repo root:
    GQH_DATA_DIR=<cache> GQH_STARTER_KIT=<kit root> python validation/starter_kit/check_gtaa.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C                              # noqa: E402
from src import engine as E                              # noqa: E402
from src import forward2 as F2                           # noqa: E402
from validation.starter_kit import run_in_starter as R   # noqa: E402

NAMES = {"S3": False, "BH5": True}                       # name -> buy_and_hold
GUIDE_BPS = (R.GUIDE_COMMISSION + R.GUIDE_SLIPPAGE) * 1e4
BROKERS = {"margin_1e9": ["--margin", "--cash", "1e9"], "cash_1e9": ["--cash", "1e9"],
           "cash_100k": ["--cash", "100000"]}
IS_START = "2005-11-01"
WINDOWS = ("in_sample", "out_of_sample")
KEEP = ("kit_sharpe_rf0", "our_sharpe_excess", "ann_return", "max_drawdown", "mean_net_weight", "orders",
        "refused_orders")


def drift_exact(w_dec: pd.DataFrame, ohlc: dict, rf: pd.Series, cost_bps: dict[str, float]) -> pd.Series:
    """next_open net returns (engine.simulate's prices, costs and cash at the T-bill) with each weight drifting from
    the open fill to the next close. engine.simulate applies the overnight move to the targets themselves, as if the
    book were also rebalanced at every close for free."""
    t = list(w_dec.columns)
    close, open_ = ohlc["close"][t].ffill(limit=5), ohlc["open"][t].ffill(limit=5)
    r_co = (open_ / close.shift(1) - 1).fillna(0.0).to_numpy()
    r_oc = (close / open_ - 1).fillna(0.0).to_numpy()
    w_new = w_dec.reindex(close.index).ffill().fillna(0.0).shift(1).fillna(0.0).to_numpy()
    cb = np.array([cost_bps[x] for x in t]) / 1e4
    cash = (1.0 - w_new.sum(axis=1)) * rf.reindex(close.index).ffill().fillna(0.0).to_numpy()
    u, out = np.zeros(len(t)), np.zeros(len(close))      # u: weights at the previous close
    for i in range(len(close)):
        on = u @ r_co[i]
        cost = np.abs(w_new[i] - u * (1 + r_co[i]) / (1 + on)) @ cb
        day = 1 - cost + w_new[i] @ r_oc[i]
        out[i] = (1 + on) * day - 1 + cash[i]
        u = w_new[i] * (1 + r_oc[i]) / day
    return pd.Series(out, index=close.index)


def export_references(out: Path) -> dict[tuple[str, str, str], pd.Series]:
    """Daily net returns per (name, costs, "engine" | "drift_exact"); the engine's held weights for the replays."""
    ohlc = E.load_ohlc(F2.S3_TICKERS, "FWD")
    rf = E.load_rf("FWD")
    refs = {}
    for name, bh in NAMES.items():
        w = F2.s3_decisions("FWD", buy_and_hold=bh)
        for costs, bps in (("project", None), ("guide", GUIDE_BPS), ("none", 0.0)):
            cmap = {t: C.cost_bps(t) if bps is None else bps for t in F2.S3_TICKERS}
            refs[name, costs, "engine"] = (F2.returns(name, "FWD") if costs == "project" else
                                           E.simulate(w, ohlc, rf, exec="next_open", cost_bps=cmap)[0])
            refs[name, costs, "drift_exact"] = drift_exact(w, ohlc, rf, cmap)
        E.simulate(w, ohlc, rf, exec="next_open")[3].rename_axis("date").to_csv(out / f"engine_{name}_held.csv")
    for (name, costs, kind), net in refs.items():
        tail = "" if kind == "engine" else "_drift"
        net.rename("ret").rename_axis("date").to_csv(out / f"engine_{name}_{costs}{tail}.csv")
    return refs


def jobs(out: Path) -> list[dict]:
    todo = []
    for name, bh in NAMES.items():
        params = f"buy_and_hold={str(bh).lower()},decisions={(out / '{run}_decisions.csv').as_posix()}"
        for costs in ("project", "guide"):
            for broker, flags in BROKERS.items():
                todo.append(dict(run=f"{name}_{costs}_{broker}", name=name, costs=costs, broker=broker,
                                 args=["--strategy", "gqh_gtaa", "--costs", costs, *flags, "--params", params]))
        todo.append(dict(run=f"{name}_kit_as_shipped", name=name, costs="none", broker="cash_100k",
                         args=["--strategy", "gqh_gtaa", "--cash", "100000", "--params", params]))
        todo.append(dict(run=f"{name}_replay_engine_weights", name=name, costs="project", broker="margin_1e9",
                         replay=True, args=["--strategy", "replay_weights", "--costs", "project", "--params",
                                            f"path={(out / f'engine_{name}_held.csv').as_posix()},fill=open"]))
    for j in todo:
        j["args"] = [a.replace("{run}", j["run"]) for a in j["args"]]
    return todo


def run(job: dict, out: Path) -> int:
    cmd = [sys.executable, str(HERE / "run_in_starter.py"), *job["args"], "--is-start", IS_START,
           "--json", str(out / f"{job['run']}.json"), "--series", str(out / f"{job['run']}.csv"),
           "--reference", str(out / f"engine_{job['name']}_project.csv"), "--reference-rf", "tbill"]
    with open(out / f"{job['run']}.log", "w", encoding="utf-8") as log:
        return subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def decision_check(path: Path, buy_and_hold: bool) -> dict:
    """The strategy's month-end decisions against forward2.s3_decisions on every month end."""
    got = pd.read_csv(path, index_col=0, parse_dates=True)
    w = F2.s3_decisions("FWD", buy_and_hold=buy_and_hold)
    want = w.loc[F2._month_ends(w.index), F2.S3_TICKERS]
    diff = (got.reindex(want.index)[F2.S3_TICKERS] - want).abs()
    return {"month_ends": int(len(want)), "logged": int(len(got)), "same_dates": bool(got.index.equals(want.index)),
            "months_differing": int((diff.max(axis=1).fillna(1.0) > 1e-12).sum()),
            "max_abs_diff": float(diff.max().max())}


def compare(daily: pd.DataFrame, ref: pd.Series, lo, hi) -> dict:
    """Agreement of the run's daily excess returns with an engine series, and the engine's own ratios there."""
    d = daily.loc[lo:hi]
    total = ref.reindex(d.index)
    ex = total - d["rf"]
    return {**R.agreement(d["excess"], ex), "engine_our_sharpe_excess": R.our_sharpe(ex),
            "engine_kit_style_sharpe_rf0": R.kit_sharpe(total), "engine_ann_return": R.ann_return(total),
            "engine_max_drawdown": E.max_drawdown(total)}


def analyse(job: dict, out: Path, refs: dict) -> dict:
    res = json.loads((out / f"{job['run']}.json").read_text())
    daily = pd.read_csv(out / f"{job['run']}.csv", index_col=0, parse_dates=True)
    wins = R.windows(daily, pd.Timestamp(IS_START))
    name, costs = job["name"], job["costs"]
    row = {k: job.get(k) for k in ("run", "name", "costs", "broker")}
    row.update(native=not job.get("replay"), cash=res["meta"]["cash"], margin=res["meta"]["margin"],
               orders=res["orders"]["total"], refused=res["orders"]["refused"],
               kit_analyzer_sharpe_full=res["kit_metrics"]["sharpe_ratio"], windows={})
    for w in WINDOWS:
        lo, hi = wins[w]
        x = {k: res["windows"][w][k] for k in KEEP}
        x["vs_engine_same_costs"] = compare(daily, refs[name, costs, "engine"], lo, hi)
        x["vs_drift_exact_same_costs"] = compare(daily, refs[name, costs, "drift_exact"], lo, hi)
        if costs != "project":
            x["vs_forward2_returns"] = compare(daily, refs[name, "project", "engine"], lo, hi)
        row["windows"][w] = x
    if not job.get("replay"):
        row["decisions"] = decision_check(out / f"{job['run']}_decisions.csv", NAMES[name])
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--jobs", type=int, default=4, help="harness runs in parallel")
    args = ap.parse_args()
    if os.environ.get("GQH_OOS_UNLOCK"):
        raise SystemExit("unset GQH_OOS_UNLOCK: only the FWD period is needed")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    refs = export_references(out)
    todo = jobs(out)
    with ThreadPoolExecutor(args.jobs) as pool:
        codes = list(pool.map(lambda j: run(j, out), todo))
    failed = [j["run"] for j, c in zip(todo, codes) if c]
    if failed:
        raise SystemExit(f"harness failed for {failed}; see the .log files in {out}")
    rows = [analyse(j, out, refs) for j in todo]
    for name in NAMES:                       # the native signal against the replay of the engine's weights
        a = pd.read_csv(out / f"{name}_project_margin_1e9.csv", index_col=0)["ret"]
        b = pd.read_csv(out / f"{name}_replay_engine_weights.csv", index_col=0)["ret"]
        next(r for r in rows if r["run"] == f"{name}_replay_engine_weights")[
            "max_abs_diff_vs_native_bp"] = float((a - b).abs().max() * 1e4)
    (out / "check_gtaa.json").write_text(json.dumps(R._clean(rows), indent=2))

    print(f"{'run':34s} {'window':13s} {'kit':>7s} {'ours':>7s} {'engine':>7s} {'corr':>9s} {'mad bp':>7s} "
          f"{'exact':>7s} {'mad bp':>7s} {'refused':>7s} decisions")
    for r in rows:
        dec = r.get("decisions")
        tag = "-" if dec is None else ("identical" if not dec["months_differing"] and dec["same_dates"]
                                       else f"{dec['months_differing']} differ")
        for w in WINDOWS:
            x = r["windows"][w]
            v, e = x["vs_engine_same_costs"], x["vs_drift_exact_same_costs"]
            print(f"{r['run']:34s} {w:13s} {x['kit_sharpe_rf0']:7.4f} {x['our_sharpe_excess']:7.4f} "
                  f"{v['engine_our_sharpe_excess']:7.4f} {v['corr']:9.6f} {v['mean_abs_diff_bp']:7.3f} "
                  f"{e['engine_our_sharpe_excess']:7.4f} {e['mean_abs_diff_bp']:7.3f} {x['refused_orders']:7d} {tag}")
    print(f"wrote {out / 'check_gtaa.json'}")


if __name__ == "__main__":
    main()
