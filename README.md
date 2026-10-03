# The Institutional Flow Clock

Gator Quant Hacks 2026, Systematic Trading track. The quant note is [`note/quant_note.pdf`](note/quant_note.pdf).
**Team:** Minh Hoang, University of Florida (solo entry).

## The idea

Pension, target-date and balanced funds rebalance stocks against bonds at month end. Index-benchmarked bond managers must buy duration when Treasury indices extend at month end. Institutions sell to raise cash before payment dates. All of them trade on calendars fixed by mandates, index rules and settlement plumbing, and they pay for immediacy when they do.

We take the position two to four days ahead of these flows and hand it to them at the month-end close. The edge should persist because these counterparties cannot change their rules without a governance decision.

## Results at a glance (net of costs)

| | In-sample 2006-05 to 2024-10 | Out-of-sample 2024-10 to 2026-10 (run once) |
|---|---|---|
| Submitted ensemble (A + C + TSMOM, min-variance, 8% vol) | Sharpe **0.98**, 9.0%/yr, max DD −11.7% | Sharpe **−0.12**, 3.0%/yr (below T-bills), max DD −8.7% |
| Same at 2x costs | 0.62 | −0.61 |
| Same rules on CME futures (Databento), flow sleeves A+B+C | 1.05 (from 2011-09) | 0.01 |

- **Overfitting:** PBO is 0.023 over 1,404 ensemble configurations. The Deflated Sharpe under the pre-registered trial count is 0.40, so the in-sample Sharpe is not statistically significant after the search.
- **Out-of-sample:** both raw effects kept their sign but shrank. The rebalancing spread fell from 11.0 to 2.5 bp/day; the Treasury month-end IEF return fell from 7.2 to 3.0 bp/day. That was too small to cover costs, and gross Sharpe fell from 1.35 to 0.38. Only the trend leg made money.
- **Everything else** (every stream, the replications, risk, capacity and the deviations from the pre-registration) is in the note.

## Forward test (frozen 2026-10-03, before its data exists)

The competition's out-of-sample window has been used, so the only honest test left is the future. [`FORWARD_TEST.md`](FORWARD_TEST.md) freezes three strategies, with git tag `forward-test-2026-10-03`:

| Strategy | In-sample Sharpe | Out-of-sample Sharpe |
|---|---|---|
| F1: the submission, unchanged (primary) | 0.98 | −0.12 |
| F2: its CME-futures implementation | 0.96 | −0.10 |
| F3: information-discreteness trend on 20 futures | 0.43 | 0.06 |

F2's history was computed only after the freeze, in a separate commit (`forward/historical_context.json`).

- **Window and reporting:** only return days from **2026-10-06** count. Quarterly reports are committed whatever they show.
- **Pass thresholds** are written down in advance (tiers at Sharpe 0.7 and 0; 24-month verdict), along with an honest power note: a two-year Sharpe has a standard error near 0.7.
- **Frozen positions:** `forward/positions_for_2026-10-06.csv` holds the target positions for the first forward session, computed from data through 2026-10-02.

To run the forward evaluation after refreshing the data:

```bash
python scripts/run_forward.py evaluate
```

## How it was tested

- **Hypothesis first.** [`HYPOTHESES.md`](HYPOTHESES.md) holds every window, sign, sizing rule, variant grid and the selection rule. It was committed before the first backtest (commit `d20c9d5`). Amendment 1 adds an ensemble search with overfitting tests (commit `06b22c9`). It was committed before the author read any in-sample result, though build agents had already logged 89 sleeve-level runs; see the `git` and `utc` columns of `results/trials.csv`.
- **Out-of-sample lock.** In-sample is 2005-01-03 to 2024-10-02; out-of-sample is 2024-10-03 to 2026-10-02, the brief's 20%-or-2-years rule. The data loaders refuse to return out-of-sample data unless `GQH_OOS_UNLOCK=1`. The selection was frozen in `results/selection.json` (commit `b054990`), then the out-of-sample window was evaluated once (commit `1ce3ff0`).
- **Trial log.** Every backtest is appended to `results/trials.csv`: 3,134 in-sample runs and 1,467 distinct specifications at 1x costs. A clean rerun logs 1,460; the other 7 are sleeve C runs from before a calendar fix changed its parameters.
- **Execution and costs.** Signals use information up to the close of day d and fill no earlier than the next session. Unscheduled market closures count only once announced. Costs are 3–10 bp one-way for ETFs plus borrow on shorts, 1–5 bp for futures plus a round trip per roll, and every result is repeated at 2x costs.
- **AI agents.** Strategy modules were written by AI coding agents from the pre-registered text and attacked by separate AI reviewer agents (lookahead, spec drift, bugs, independent re-derivation). `tests/test_lookahead.py` replays the lookahead check. The hypotheses, pre-registration, selection rule and note are the author's responsibility.
- **Databento replication.** GLBX.MDP3 CME futures, with returns always computed within one contract and one round trip charged per roll. ES and ZN are sampled at 16:00 ET from hourly bars to align with the ETF closes.

### Verify the audit trail

```bash
git log --format='%h %ad %s' --date=iso
```

Show the trial log as it was frozen at the selection commit:

```bash
git show b054990:results/trials.csv
```

It contains no `OOS` rows.

Show the single out-of-sample pass:

```bash
git show 1ce3ff0:results/oos_log.csv
```

All rows are from 11:46 UTC and carry the selection commit's original hash.

Commit messages were edited after the fact. File contents, tree hashes and dates are unchanged, but commit hashes changed. `results/commit_map.csv` maps the original hashes recorded in `trials.csv` and `oos_log.csv` to the current ones.

## Reproduce

Python 3.12+ (numpy 2.5.3 and scipy 1.18.1 require it).

Set up the environment:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, activate it with `.venv\Scripts\activate` instead.

```bash
pip install -r requirements.txt
```

Run everything with one command:

```bash
python reproduce.py
```

It downloads data, runs the in-sample research, checks the selection against the committed one, runs the out-of-sample evaluation, then regenerates the note.

`reproduce.py` runs these steps, which you can also run one by one:

```bash
python data/download.py
```

This fetches the free data: Yahoo Finance, FRED, Ken French and Treasury Fiscal Data.

```bash
python data/download_databento.py
```

This step is optional. It is the futures replication, needs `DATABENTO_API_KEY` in `.env`, and costs about $5 of usage.

```bash
python run_all.py
```

This is the in-sample stage: every strategy, the ensemble search and `selection.json`. It takes about 25 minutes.

```bash
GQH_OOS_UNLOCK=1 python run_all.py --final
```

This is the out-of-sample evaluation plus the post-hoc diagnostics. In PowerShell, run `$env:GQH_OOS_UNLOCK="1"; python run_all.py --final`.

```bash
python note/make_note.py
```

This regenerates `note/numbers.tex` and the figures and compiles the PDF. Add `--no-pdf` if pdflatex is not installed.

```bash
python tests/test_lookahead.py
```

This is the lookahead perturbation test. It uses in-sample data only.

Notes:
- Set `GQH_DATA_DIR` to keep the data cache elsewhere; the default is `data/cache/`, which is git-ignored.
- Copy `.env.example` to `.env` for the Databento key. Raw data and keys are never committed.
- Without a Databento key, the futures rows in the note show as "n/a".
- Yahoo revises adjusted history retroactively, so a later download can move numbers slightly. On 3 Oct 2026 a fresh clone with freshly downloaded data reproduced every number in the note.
- A rerun appends to `results/trials.csv` and `results/oos_log.csv`. The committed versions are the record of the original research. On a rerun, every number in the note is reproduced except the logged-run counter, which grows because the log is appended to.

## Layout

```
HYPOTHESES.md            pre-registration (+ Amendment 1)
reproduce.py, run_all.py one-command reproduction; in-sample / --final out-of-sample pipeline
src/engine.py            backtest engine: next-session fills, costs, OOS lock, trial log, statistics
src/calendar_utils.py    month-end calendar, settlement regimes, planned-vs-realised sessions
src/strategies/          ifc_rebalance (A), ifc_dash (B), ifc_treasury (C), ifc_fx (D), ifc_auction (E),
                         ifc (composite), pct (persistence-conditioned trend)
src/ensemble.py, pbo.py  flow-and-trend ensemble, CSCV probability of backtest overfitting
src/analysis.py          factor regressions, sub-periods, bootstrap, square-root-impact capacity
scripts/                 per-module runners, ensemble search, selection, out-of-sample run, diagnostics
tests/                   lookahead perturbation test
data/                    download scripts only (no data)
results/                 every number in the note (in-sample, out-of-sample, trial log, diagnostics)
note/                    quant note source, generated numbers and figures, PDF
```

## Data sources and license

- Yahoo Finance, via `yfinance`: total-return-adjusted ETF prices.
- FRED: the 3-month T-bill (DTB3) and Treasury yields.
- Kenneth R. French Data Library: daily Fama-French 5 factors and momentum.
- U.S. Treasury Fiscal Data: the auction record with announcement dates.
- Databento GLBX.MDP3: CME futures, daily and hourly bars.

No data is redistributed. The code is MIT-licensed (see `LICENSE`). Published methods are cited in the note: the Deflated Sharpe ratio, PBO/CSCV, Ledoit-Wolf shrinkage and square-root impact. The path-geometry statistic extends the author's earlier repository `minh-stakc/physics-quant-research`.
