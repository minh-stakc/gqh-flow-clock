# Forward test 2: pre-registration (frozen Sat 2026-10-03)

## Why this exists

Forward test 1 (`FORWARD_TEST.md`) tracks the submitted strategy and two close relatives. This second test registers the strategies that a literature review (`research/high_sharpe_literature.md`, kept local) found to have the **best live or post-publication evidence that daily, liquid data can reproduce**. The review's headline finding:
- No strategy has a verified out-of-sample net Sharpe of 1 or more over a full cycle in liquid markets.
- Live records cluster at 0.25–0.6 net.
- The backtest-to-live haircut is about 70%.

So the expectations below are deliberately modest. Nothing here was chosen or tuned on any backtest of ours: these strategies' history is computed only **after** this file and the code are committed and tagged (`forward-test-2-2026-10-03`), in a separate commit.

The 2024-10 to 2026-10 window has been seen and cannot validate anything. Only data from 2026-10-06 counts. Forward test 1's files (`src/forward.py`, `scripts/run_forward.py`, `FORWARD_TEST.md`) are untouched; this test lives in new files: `src/forward2.py` and `scripts/run_forward2.py`.

## Strategies (code: `src/forward2.py` at the tag)

| | Rule | Evidence that motivated it | Pre-registered realistic net Sharpe |
|---|---|---|---|
| **S1** | **Broad CME trend.** 30 futures (Databento GLBX.MDP3, front contract, within-contract returns, one round trip per roll).<br>• *Signal:* mean of the signs of the trailing 21/63/252-day excess returns.<br>• *Sizing:* weight = signal × 0.40/σ/N, where σ is EWMA volatility with a 60-day centre of mass and N is the number of eligible contracts.<br>• *Book:* scaled to 10% ex-ante volatility with the trailing 252-day covariance; gross ≤ 3. When the gross cap binds (calm, highly diversified months) the ex-ante volatility is below 10%.<br>• *Trading:* monthly decisions, next-close fills. Costs: 1.5 bp one-way for ES NQ YM ZT ZF ZN ZB UB CL GC SI HG 6E 6J 6B 6S, 5 bp for RTY HO RB NG PL ZC ZS ZW ZL ZM LE HE 6A 6C. | Hurst, Ooi & Pedersen (2013, 2017): trend positive in every decade since 1880. Live trend indices: SG Trend 0.30 net, BTOP50 0.40–0.50 net. | 0.2–0.6 (central 0.4) |
| **S2 (primary)** | **Equity plus trend, equal risk.** 0.5 × (long ES at 10% ex-ante vol) + 0.5 × S1, rescaled to 10% ex-ante vol with the trailing 252-day instrument covariance; gross ≤ 3; monthly; next-close fills; S1's costs. | A 50/50 mix of US equities and live trend indices: 0.72 (1987–2024), against 0.55 for equities and 0.40 for trend alone. | 0.4–0.7 (central 0.6) |
| **S3** | **Faber GTAA.** 20% each in SPY, EFA, IEF, VNQ and DBC when the month-end adjusted close is above the mean of the last 10 month-end closes, otherwise T-bills; monthly; next-open fills; project ETF costs (Yahoo data). | Faber (2007): 0.73 gross 1973–2012; 0.61 after publication (2006–12). | 0.3–0.6 (central 0.45) |

**Benchmarks** (yardsticks, not strategies; no hypotheses of their own):
- **TSMOM_F**: forward test 1's benchmark, computed by forward test 1's frozen code (`src/forward.py`). Yardstick for S1.
- **LONG_ONLY_F**: S1's construction with every signal set to +1. It asks whether S1's trend signal adds anything over simply holding the same risk-weighted futures long.
- **ES_10VOL**: the S2 equity sleeve alone (long ES at 10% ex-ante vol, gross ≤ 3).
- **BH5**: constant 20% weights in the five S3 ETFs. It holds nothing until the same 10-month warm-up as S3 is over, so the two start together.

**Mechanics shared by all strategies:**
- "Monthly" means monthly target decisions. The engine trades back to the targets every session, so drift-rebalancing costs are included.
- *Eligibility* of a futures contract at a decision needs all three of:
  - at least 300 sessions of data (`MIN_HISTORY`);
  - a positive EWMA volatility;
  - at least one traded session (volume > 0) in the last 10 sessions (`ACTIVE_WINDOW`; the longest past vendor gap was 5 sessions).

  The ES sleeve of S2 and ES_10VOL follow the same rule. An ETF without a month-end close is not held.

**Data sources.** Futures come from Databento, because Hugging Face has no maintained continuous CME futures data (checked 2026-10-03). The ETFs and the T-bill come from free Yahoo and FRED data.

## Window, hypotheses and thresholds

- **Window.** Return days on or after **2026-10-06** (`config.FWD_START`, shared with forward test 1). Target positions for that day, computed from data through Fri 2026-10-02, are committed as `forward/positions2_for_2026-10-06.csv`.
- **Primary metric.** Annualised Sharpe of daily net returns over the T-bill. Also reported: annual return, volatility, maximum drawdown, cumulative return, a 21-day block-bootstrap 90% interval (once there are 63 days), and the standard error √(252/n).
- **Hypotheses:**
  - **H2-S2 (primary):** S2's forward Sharpe is positive.
  - **H2-S1:** S1's forward Sharpe is positive.
  - **H2-S3:** S3's forward Sharpe is above BH5's.
- **Paired comparisons.** Each report gives the Sharpe difference and a paired 21-day block-bootstrap 90% interval (the same resampled blocks for both series; 2,000 draws; seed 7) for:
  - S1 − TSMOM_F
  - S1 − LONG_ONLY_F
  - S2 − ES_10VOL
  - S3 − BH5

  Within 24 months these intervals are expected to include zero. They are reported to show the size of the difference, not to claim a detectable one.
- **Tiers at 24 months.** ≥ the central value: consistent with the literature. 0 to central: weak. < 0: below expectations. The pre-registered 90% interval for a 24-month Sharpe at the central value is about −0.8 to +1.6 for S1, −0.7 to +1.9 for S2, and −0.75 to +1.7 for S3.
- **What a strong result means.** A 24-month Sharpe of 1 or more will be read as luck-assisted, not as validation.
- **Multiple testing.** α = 0.05/3 within this test, and 0.05/6 for any claim that spans both forward tests.
- **Implementation check (S1).**
  - *What is measured:* the correlation of S1's weekly (Friday-to-Friday) returns with those of DBMF and AQMIX, two public managed-futures funds (Yahoo prices).
  - *Blocks:* six calendar months counted from 2026-10-06, so the first block is 2026-10-06 to 2027-04-05. A block with fewer than 8 weeks gets no value. The last block may be partial.
  - *Flag:* a correlation below 0.5 in a block flags the build for review; the frozen version still runs.
  - *Download failure:* recorded in the report as an error, never treated as a pass.

## Conduct

- **Frozen.** No changes to code, parameters, universes or costs.
- **Delisting and data gaps.**
  - A contract that stops trading or loses vendor coverage fails the 10-session activity rule. It leaves the book at the first month-end decision after that; until then it earns a zero excess return.
  - Every report lists each instrument's last traded session and the instruments with no data on the report date.
- **Quarterly reports.** One on the first business day after each quarter end (2027-01-04, 2027-04-01, …), with the verdict at 24 months. Each is committed whatever it shows; every evaluation is also appended to `forward/forward2_log.csv`.
- **Paper evaluation only.** Not investment advice.

## How to run

Run everything **after 20:00 ET** on a trading day:
- Yahoo returns the current day's bar, which is partial before the close.
- Databento's daily CME bars are cut at UTC midnight (20:00 ET in summer, 19:00 in winter).

Use **one fresh, empty `GQH_DATA_DIR` for all three steps**. The futures builder reads the ETF calendar and T-bill written by the first step, and the evaluation must read the files built in that same folder.

```bash
export GQH_DATA_DIR=/path/to/new/empty/folder
```

```bash
python data/download.py
```

```bash
GQH_DATA_END=<the day after the last session> python data/download_databento.py --batch
```

```bash
python scripts/run_forward2.py evaluate
```

Notes on the futures step:
- `GQH_DATA_END` is exclusive. To include Monday 2027-01-04, set it to 2027-01-05.
- `--batch` fetches the daily bars through one Databento batch job. Streaming requests timed out repeatedly on 2026-10-03. If the job outlives the script, resume with `--batch-job <id>`.
- Without `--batch` the builder streams one request per root.

`evaluate` refuses to score or log anything if no futures contract has data on the last ETF session (stale or missing futures files); partial gaps produce a warning and appear in the report.

Forward test 1's run example (`GQH_DATA_END=2027-01-04`) follows the same rule: the end date is the day after the last session included. Its frozen runner has no stale-data check, so run it from the same folder, after this test's `evaluate` has accepted the data.
