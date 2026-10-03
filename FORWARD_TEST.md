# Forward test: pre-registration (frozen Sat 2026-10-03)

## Why this exists

The competition's out-of-sample window (2024-10-03 to 2026-10-02) was evaluated once and the submitted strategy failed it (Sharpe −0.12). We have now seen that window, so any new idea scored on it would be fit to it. The only honest test left is **data that does not exist yet**.

This file fixes, before any such data arrives:
- which strategies are tested;
- the exact code;
- the evaluation window;
- the metrics and decision thresholds;
- the reporting schedule.

The git tag `forward-test-2026-10-03` marks the frozen code. Nothing in `src/` or `scripts/run_forward.py` may change during the test.

**What we already knew when we froze it** (in-sample 2005/2010–2024, out-of-sample 2024–26):
- F1's history was known (0.98 in-sample, −0.12 out-of-sample).
- F3's was known (0.43 / 0.06), as was the TSMOM-F benchmark's (0.28 / 0.14).
- F2's history had **not** been computed. It is computed only after this file is committed (`forward/historical_context.json`, a later commit), so it cannot have influenced the choice of F2.

## Strategies (code: `src/forward.py` at the tag)

| | Definition | Why it is here |
|---|---|---|
| **F1 (primary)** | The submitted strategy, unchanged. Ensemble of sleeve A (rebalancing pressure) + sleeve C (Treasury month-end extension) + TSMOM on 15 ETFs, Ledoit-Wolf minimum-variance stream weights, 8% volatility target, no brake, the costs in HYPOTHESES.md. Verified identical to `results/oos/fte_selected_full_returns.csv` (max difference 1e-16). | The cleanest test: it was selected before any out-of-sample data was seen. |
| **F2** | The same construction on CME futures (Databento GLBX.MDP3): A and C on ES/ZN sampled at 16:00 ET with 1 bp one-way costs, plus TSMOM on 20 futures (PCT-F rules, measure `none`, next-close fills, 1.5–5 bp costs). Same ensemble rules; re-scaling costs of 1/1/2 bp. | The out-of-sample diagnosis was that costs (gross 0.38 vs net −0.12) and capacity bind. The note recommends futures as the vehicle. |
| **F3** | Information-discreteness-conditioned TSMOM on the same 20 futures (PCT-F rules, measure `ID`). | Promised in the note as a new hypothesis: in-sample it beat plain TSMOM on both ETF and futures universes. |
| Benchmark | Plain TSMOM on the 20 futures (`TSMOM_F`). Tracked only to judge F3. | |

## Forward window and data

- **Window.** Only return days on or after **2026-10-06** count. That is the first day on which every position was traded after the freeze: positions held over Oct 6 are traded at the close of Mon Oct 5, decided from data through Fri Oct 2.
- **Frozen positions.** `forward/positions_for_2026-10-06.csv`, committed with this file, lists those exact target weights. Because Yahoo revises adjusted history, a later recomputation may differ in the last digits.
- **Data.** Same sources and adjustments as the study:
  - Yahoo Finance via `data/download.py`;
  - Databento via `GQH_DATA_END=<date> python data/download_databento.py` into an empty `GQH_DATA_DIR`;
  - FRED for the T-bill.
- **Delisting.** If an instrument stops trading or a vendor drops it, it leaves its universe from that date (exactly as the warm-up rules treat late starters), and the change is recorded in the report. Nothing else may change.

## Hypotheses and thresholds

**Primary metric.** Annualised Sharpe ratio of daily net returns in excess of the T-bill over the forward window. Reported alongside it: annualised return, volatility, maximum drawdown, cumulative return, a 21-day block-bootstrap 90% interval, and the standard error √(252/n).

**H-F1 (primary): F1's forward Sharpe is positive.**
- **Power.** With 24 months the standard error is about 0.71, so only a realised Sharpe above about 1.2 would reject zero at 5% (one-sided). We therefore pre-commit to reading the estimate by tiers:
  - **≥ 0.7:** consistent with the in-sample result after the cross-validated selection haircut (0.98 → 0.75 in the CSCV).
  - **0 to 0.7:** weak or inconclusive.
  - **< 0:** consistent with the out-of-sample decay. Retire the strategy.
- **Checkpoints.** First business day after each quarter end (2027-01-04, 2027-04-01, 2027-07-01, 2027-10-01, …). The **12-month** checkpoint (2027-10-01) is a status read. The **24-month** checkpoint (first business day after 2028-10-06) is the verdict.

**H-F2: F2's forward Sharpe exceeds F1's** (the cost and capacity argument). Reported as the paired difference with a block-bootstrap interval.

**H-F3: F3's forward Sharpe exceeds the TSMOM_F benchmark's.** Reported as the paired difference with a block-bootstrap interval.

**Multiple testing.** Three hypotheses, so any claim of statistical significance uses α = 0.05/3.

## Conduct

- **No changes to code, parameters, universes or costs.** If a bug is found, the frozen version keeps running and is the one reported. A fixed version may be reported alongside it, labelled as such.
- **Every evaluation is logged and published.** `python scripts/run_forward.py evaluate` appends to `forward/forward_log.csv` and writes `forward/report_<date>.json`. Each quarterly report is committed whatever it shows, including ad-hoc runs.
- **Paper evaluation only.** Nothing here is investment advice.

## How to run

```bash
python data/download.py
```

This refreshes the Yahoo, FRED and Ken French data to the latest session.

```bash
GQH_DATA_END=2027-01-04 python data/download_databento.py
```

This refreshes the CME futures (F2, F3, benchmark). Use a fresh `GQH_DATA_DIR` for it.

```bash
python scripts/run_forward.py evaluate
```

This writes `forward/report_<date>.json` and appends to `forward/forward_log.csv`.
