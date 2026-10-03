"""One command that reproduces every number in the quant note.

    python reproduce.py

Steps: download the free data if the cache is empty -> in-sample research (every strategy, the
ensemble search, the selection rule) -> check that the selection equals the committed one ->
the out-of-sample evaluation of that selection (+ post-hoc diagnostics) -> regenerate the note's
numbers and figures (and the PDF if pdflatex is installed) -> show what changed versus the commit.

The CME futures replication is optional: it runs only if DATABENTO_API_KEY is set (environment or
.env); otherwise the committed futures results are kept. A run appends to results/trials.csv and
results/oos_log.csv, so the committed logs remain the record of the original research.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

PY = sys.executable


def run(args: list[str], env: dict | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True, env=env)


def has_databento_key() -> bool:
    if os.environ.get("DATABENTO_API_KEY"):
        return True
    env = ROOT / ".env"
    return env.exists() and any(l.startswith("DATABENTO_API_KEY=") and "your-key-here" not in l
                                for l in env.read_text().splitlines())


def main() -> None:
    if not (C.DATA_DIR / "etf_daily.parquet").exists():
        run([PY, "data/download.py"])
    if has_databento_key() and not (C.DATA_DIR / "futures_1600.parquet").exists():
        run([PY, "data/download_databento.py"])
    committed = json.loads((C.RESULTS_DIR / "selection.json").read_text())["fte_selected"]
    run([PY, "run_all.py"])
    now = json.loads((C.RESULTS_DIR / "selection.json").read_text())["fte_selected"]
    if now != committed:
        sys.exit(f"in-sample selection {now!r} differs from the committed {committed!r}; stopping before OOS")
    run([PY, "run_all.py", "--final"], env={**os.environ, "GQH_OOS_UNLOCK": "1"})
    run([PY, "note/make_note.py"] + ([] if shutil.which("pdflatex") else ["--no-pdf"]))
    if shutil.which("git"):
        subprocess.run(["git", "-c", "safe.directory=*", "diff", "--stat", "--", "note/numbers.tex",
                        "results/oos/summary.json"], cwd=ROOT)
    print("done: compare note/numbers.tex with the committed version (git diff note/numbers.tex)")


if __name__ == "__main__":
    main()
