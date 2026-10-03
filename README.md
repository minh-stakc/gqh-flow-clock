# Institutional Flow Clock: trading against calendar-scheduled institutional flows

Gator Quant Hacks 2026, Systematic Trading track. Quant note: [`note/quant_note.pdf`](note/quant_note.pdf).

**Idea.** Large pools of price-insensitive money trade on fixed calendars set by mandates, settlement
plumbing and index rules: balanced and target-date funds rebalance stocks against bonds at month
end, institutions raise cash before month-end payment dates, and index-benchmarked bond managers
buy duration when Treasury indices extend at month end. Those flows are predictable from public
information. A liquidity provider who takes the other side earns a premium that persists because
the counterparties cannot change their rules without governance changes.

**How we tested it (the part judges asked about).**

* The hypotheses, every window, sign and sizing rule, the variant grid and the selection rule were
  committed **before the first backtest** ([`HYPOTHESES.md`](HYPOTHESES.md), commit `fb235f7`).
  An amendment adding a diversified ensemble search with overfitting tests was committed before any
  in-sample result was read (commit `12e28bf`).
* In-sample 2005-01-03 to 2024-10-02; out-of-sample 2024-10-03 to 2026-10-02 (the brief's
  20 %-or-2-years rule). The data loaders refuse to return out-of-sample data unless
  `GQH_OOS_UNLOCK=1`; the out-of-sample evaluation was run once, after the selection was committed.
* Every backtest ever run is in [`results/trials.csv`](results/trials.csv) (that count feeds the
  Deflated Sharpe ratio); every out-of-sample evaluation is in `results/oos_log.csv`.
* Signals use information up to the close of day d and fill no earlier than the next session;
  unscheduled market closures are only known once announced; all results are net of costs
  (3-10 bp one-way for ETFs, plus borrow on shorts) and repeated at 2x costs.
* Each strategy module was built by one agent and attacked by an independent reviewer that looked
  for lookahead, spec drift and bugs and re-derived the headline numbers with separate code.
* Independent replication on CME futures (ES, ZN and 20 markets) from **Databento** GLBX.MDP3, with
  explicit roll handling (returns always within one contract) and prices sampled at 16:00 ET.

## Reproduce

Python 3.11+ (tested on 3.12).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/download.py                    # free data: Yahoo Finance, FRED, Ken French, Treasury Fiscal Data
python data/download_databento.py          # optional: futures replication, needs DATABENTO_API_KEY (~$5 of usage)
python run_all.py                          # in-sample: every strategy, ensemble search, selection.json (~25 min)
GQH_OOS_UNLOCK=1 python run_all.py --final # out-of-sample evaluation of the committed selection + diagnostics
python note/make_note.py                   # regenerates note/numbers.tex + figures, compiles the PDF (pdflatex)
```

Set `GQH_DATA_DIR` to keep the data cache elsewhere (default `data/cache/`, git-ignored). Copy
`.env.example` to `.env` for the Databento key. Raw licensed data and keys are never committed.

## Layout

```
HYPOTHESES.md            pre-registration (+ Amendment 1)
src/engine.py            backtest engine: next-session fills, costs, OOS lock, trial log, statistics
src/calendar_utils.py    month-end calendar, settlement regimes, planned-vs-realised sessions
src/strategies/          ifc_rebalance (A), ifc_dash (B), ifc_treasury (C), ifc_fx (D), ifc_auction (E),
                         ifc (composite), pct (persistence-conditioned trend)
src/ensemble.py, pbo.py  flow-and-trend ensemble, CSCV probability of backtest overfitting
src/analysis.py          factor regressions, sub-periods, bootstrap, square-root-impact capacity
scripts/                 one runner per module, the ensemble search, the out-of-sample evaluation
data/                    download scripts only
results/                 every number in the note (in-sample, out-of-sample, trial log, figures)
```

## Data sources

Yahoo Finance (via `yfinance`) daily ETF prices, total-return adjusted; FRED (DTB3 T-bill and
Treasury yields); Kenneth French Data Library (daily Fama-French 5 factors and momentum); U.S.
Treasury Fiscal Data API (auction record with announcement dates); Databento GLBX.MDP3 (CME
futures, daily and hourly bars). See the note's references.

Code is original; published methods are cited where used (Deflated Sharpe ratio, PBO/CSCV,
square-root impact). The path-geometry statistic extends the author's earlier repository
`minh-stakc/physics-quant-research`.
