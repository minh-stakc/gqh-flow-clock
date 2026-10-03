# Pre-registration: hypotheses, exact specifications and test protocol

Written Sat 2026-10-03, before any backtest was run. The git commit that adds this file is the
timestamp; `results/trials.csv` (written by the engine) records every backtest made after it.

Two independent candidate strategies are specified below. The **Institutional Flow Clock (IFC)**
trades against calendar-scheduled, price-insensitive institutional flows. The **Persistence-
Conditioned Trend (PCT)** extends the author's earlier path-geometry research
(github.com/minh-stakc/physics-quant-research) into a scale-free persistence statistic that
conditions time-series momentum. A pre-committed rule picks one of them for submission.

---------------------------------------------------------------------------------------------

## 0. Protocol (applies to everything below)

**Data.** Daily total-return-adjusted OHLCV of US-listed ETFs (Yahoo Finance via yfinance,
`auto_adjust=True`), 3-month T-bill (FRED `DTB3`) as the cash rate, Fama-French 5 factors +
momentum (Ken French library), and the U.S. Treasury auction record (Fiscal Data API
`auctions_query`, with announcement dates). Every source is cited in the note.

**Sample split.** History used: 2005-01-03 to 2026-10-02 (21.75 years). The brief's rule is the most
recent 20 % or 2 years, whichever is shorter, so the 2-year cap binds:

| | dates |
|---|---|
| In-sample (all research, all tuning, the selection) | 2005-01-03 .. 2024-10-02 |
| Out-of-sample (evaluated once) | 2024-10-03 .. 2026-10-02 |

The loaders refuse to return data after 2024-10-02 unless `GQH_OOS_UNLOCK=1` is set. That flag is
set exactly once, by `run_all.py --final`, after the selection below has been committed.

**Calendar.** Trading days are SPY trading days. For month m, `T` is its last trading day, `T-j` is
the j-th trading day before `T`, and `T+j` is the j-th trading day of the following month.
"Return day t" is close(t-1) -> close(t). "Hold over return days [a, b]" means the position is
established at the close of a-1 and closed at the close of b.

**Execution and lags.** A decision may use information up to the close of day d and is filled no
earlier than the next session (`next_close`: at the close of d+1; `next_open`: at the open of d+1).
So a data-dependent signal used for a hold window [a, b] under `next_close` is observed at the close
of a-2. Calendar facts (month ends, the settlement regime, announced auction sizes) are used only
from the date they were public.

**Costs.** One-way cost per unit of turnover: 3 bp for tier-1 ETFs (SPY, QQQ, IWM, EFA, EEM, TLT,
IEF, SHY, LQD, HYG, GLD, ...), 5 bp for tier-2 (sector, commodity, currency ETFs, VNQ, TIP, DBC,
UUP, SLV, ...), 10 bp otherwise; plus 30 bp/yr to borrow ETFs sold short; cash earns the T-bill.
Every reported number is net. Every headline number is repeated at 2x costs.

**Trial accounting.** The engine appends every backtest to `results/trials.csv`. Deflated Sharpe
ratios (Bailey & Lopez de Prado 2014) use the number of distinct in-sample trials across both
candidates and the cross-trial variance of their Sharpe ratios.

**Selection rule (fixed now).** The submitted strategy is the candidate (IFC base or PCT base)
with the higher in-sample Deflated Sharpe Ratio among those that pass all three gates:

* G1: in-sample Sharpe > 0 at 2x costs;
* G2: no single calendar year contributes more than 40 % of in-sample cumulative excess return;
* G3: the median in-sample Sharpe over its pre-registered neighbourhood (section 3) is at least
  half of the base specification's Sharpe.

The base specification is submitted, never the best neighbourhood point. If neither passes, the
one with the higher DSR is submitted and the note reports the failure.

**OOS reporting.** After the selection is committed, the OOS window is run once for both
candidates' base specifications, their sleeves/benchmarks and the robustness variants listed here,
and every result is reported.

---------------------------------------------------------------------------------------------

## 1. Candidate 1: Institutional Flow Clock (IFC)

**Economic hypothesis (one sentence).** Large pools of price-insensitive money (balanced and
target-date funds, pensions paying benefits, index-benchmarked bond managers) trade on fixed
calendars dictated by mandates, settlement plumbing and index rules; their flows are predictable
from public information, so a liquidity provider who takes the other side ahead of them earns a
premium that persists because the counterparties cannot change their rules without governance
changes.

All three core sleeves use only SPY and IEF, decide on `next_close`, and are sized by 63-day
realised volatility observed at the decision date.

### Sleeve A: stock/bond rebalancing pressure

* **Counterparty.** Funds that hold fixed equity/bond weights (60/40-style, target-date funds,
  DB pensions; Harvey, Mazzoleni & Melone 2025, NBER w33554) and rebalance at month end. After
  stocks beat bonds they must sell stocks and buy bonds near the month end, whatever the price.
* **Prediction.** When the month-to-date drift of a 60/40 portfolio is toward stocks, SPY
  underperforms IEF over the last five trading days of the month, and vice versa.
* **Rule.** Shadow portfolio: 60 % SPY / 40 % IEF, reset to 60/40 at the close of every `T`.
  Drift `D_d` = shadow equity weight at the close of d minus 0.60. Observe `D` at the close of
  `T-6`; standardise `z = D / s`, where `s` is the root-mean-square of all earlier months' `D`
  values at `T-6` (expanding, at least 12 months, sleeve inactive before that); `q = clip(z, -2, 2)`.
  Hold over return days [`T-4`, `T`]: `w_SPY = -q * 0.05 / vol_SPY`, `w_IEF = +q * 0.05 / vol_IEF`.
* **Kill condition.** The sign of the SPY-minus-IEF return in the window does not oppose the drift.

### Sleeve B: settlement-cycle "dash for cash"

* **Counterparty.** Institutions that must have cash on month-end payment dates (Etula, Rinne,
  Suominen & Vaittinen 2020, RFS) sell equities in the days before the last trade date that still
  settles by the month end, then reinvest after the turn of the month.
* **Our novel prediction (a natural experiment).** The selling-pressure window ends exactly `k`
  trading days before `T`, where `k` is the U.S. equity settlement lag: `k = 3` for months whose
  `T` is before 2017-09-05, `k = 2` from 2017-09-05 to 2024-05-27, `k = 1` from 2024-05-28 (T+1).
  The windows therefore shift one day later at each settlement change. The out-of-sample window
  lies entirely in the T+1 regime, so the single OOS run tests a structural prediction made in
  advance.
* **Rule.** Pressure window: return days [`T-k-5`, `T-k-1`]; liquidity window: [`T-k`, `T+3`].
  Hold `w_SPY = -0.10 / vol_SPY` over the pressure window, `+0.10 / vol_SPY` over the liquidity
  window, 0 otherwise.
* **Secondary predictions (reported, not traded).** (B2) In the T+2 regime, the settlement-aware
  windows beat the old T+3 windows. (B3) The liquidity-minus-pressure spread is larger when `T` is
  a Friday (the monthly and weekly pay cycles coincide; Etula et al.).
* **Kill condition.** The liquidity-window minus pressure-window spread is not positive.

### Sleeve C: Treasury month-end index extension, sized by issuance

* **Counterparty.** Bond indices add newly issued Treasuries only at month end, so the index's
  duration jumps on the last day of the month; index-benchmarked managers and insurers buy duration
  into the rebalance (Hartley & Schwarz 2019).
* **Our twist.** The size of the duration extension is roughly the duration-weighted (DV01-like)
  amount of coupon securities issued in that month, which the auction record shows before the trade.
* **Rule.** `S_m` = sum over Note and Bond auctions with issue date in month m and announcement
  date on or before `T-4` of offering amount x approximate modified duration of the original term
  (2y 1.9, 3y 2.8, 5y 4.5, 7y 6.2, 10y 8.3, 20y 13.5, 30y 18.0). `q = clip(S_m / median(S over the
  previous 12 months), 0.5, 2.0)`. Hold over return days [`T-2`, `T`]: `w_IEF = q * 0.10 / vol_IEF`,
  capped at 2.0.
* **Kill condition.** The IEF excess return over [`T-2`, `T`] is not positive on average.

### Exploratory sleeves (reported, never part of the composite)

* **D. FX hedge rebalancing** (Melvin & Prins 2015). Signal = SPY month-to-date return minus the
  local-currency month-to-date return of the foreign index (Euro Stoxx 50 for FXE, Nikkei 225 for
  FXY), observed at `T-3`. Hold the currency ETF in the signal's direction over return days
  [`T-1`, `T`] at 0.05 / vol.
* **E. Treasury auction cycle** (Lou, Yan & Zhang 2013; reported decayed after 2014 by Fleming,
  Liu & Nguyen 2026). For each Note/Bond auction on day `A` (announced by the decision date), short
  IEF over return days [`A-4`, `A`] and long over [`A+1`, `A+4`], sized by the auction's DV01
  relative to the trailing 12-month mean auction DV01, 0.10 / vol_IEF per unit, capped at 2.

### IFC composite (the candidate)

* IFC = A + B + C. Sleeve multipliers `lambda_s = (0.10 / sqrt(3)) / sigma_s`, where `sigma_s` is
  the annualised standard deviation of sleeve s's own daily net returns over all prior days
  (expanding, at least 252 days, lagged one day); sleeve weights are summed per ETF.
  Caps: |w| <= 3 per ETF, gross <= 4.
* **Prediction.** Positive in-sample Sharpe net of costs, low correlation to the market and to
  momentum (it is contrarian liquidity provision), and alpha in a regression on FF5 + Mom + a
  Treasury duration factor.

---------------------------------------------------------------------------------------------

## 2. Candidate 2: Persistence-Conditioned Trend (PCT)

**Economic hypothesis.** Time-series momentum exists because information diffuses slowly and
investors under-react, then herd (Moskowitz, Ooi & Pedersen 2012; Hong & Stein 1999). When an
asset's recent price path shows persistent increments, that slow-diffusion state is present and
the trend should continue; when the path chops around its chord, the trend is noise. The
author's earlier path-geometry gate failed (permutation p = 0.38) because its area and arc-length
statistics scale with volatility. Dividing one by the other removes the scale.

* **Statistic.** For a leg of W+1 log prices, `A` = mean absolute deviation from the chord between
  the endpoints, `L` = mean absolute daily log return. `R = A / (L * sqrt(W))`. Under a Gaussian
  random walk `R -> pi/8 = 0.3927` as W grows; it rises with positively autocorrelated increments
  and falls with mean-reverting ones. `R` is invariant to the volatility level.
* **Universe.** SPY, QQQ, IWM, EFA, EEM, TLT, IEF, LQD, HYG, TIP, GLD, SLV, DBC, UUP, VNQ; an ETF
  is eligible once it has 300 trading days of history.
* **Rule.** Monthly: decide at the close of `T`, trade at the next open (`next_open`).
  * Trend sign `s_i` = sign of the asset's past 252-day return minus the T-bill return over the
    same period.
  * Persistence: `Rbar_i` = average of `R` over the 4 non-overlapping 63-day legs covering the past
    252 days. `z_i = (Rbar_i - mu0) / sd0`, where `mu0` and `sd0` are the mean and standard
    deviation of `Rbar` under a Gaussian random walk with the same W and leg count (Monte Carlo,
    fixed seed, computed once in code). `m_i = 1 + clip(z_i, -1, 1)`, so it lies in [0, 2].
  * Raw weight `w_i = s_i * m_i * (0.40 / vol_i) / N`, with `vol_i` the 63-day realised volatility
    and N the number of eligible ETFs; the portfolio is then scaled to 10 % ex-ante volatility using
    the trailing 252-day covariance matrix, gross exposure capped at 3.
* **Benchmark (same code, `m_i = 1`).** Plain volatility-scaled TSMOM.
* **Comparison variant.** Information discreteness (Da, Gurun & Warachka 2014) in place of `R`:
  `ID = sign(PRET) * (%neg - %pos)` over 252 days, `z = -ID * sqrt(252)`, `m = 1 + clip(z, -1, 1)`.
* **Predictions.** (P1) PCT has a higher in-sample Sharpe than plain TSMOM; the block-bootstrap
  interval of the Sharpe difference is reported. (P2) Pooled over assets and months, the next-month
  volatility-adjusted TSMOM return is higher in the top tercile of `z` than in the bottom tercile.
* **Kill condition.** P1 and P2 both fail (the conditioning adds nothing over plain TSMOM).

---------------------------------------------------------------------------------------------

## 3. Pre-registered neighbourhoods (gate G3) and robustness runs

**IFC neighbourhood (8 points, one change at a time).** All windows shifted one day earlier;
one day later; sleeve A window of 4 days; of 6 days; sleeve C window of 2 days; of 4 days; equal
sleeve multipliers; no issuance sizing in C (q = 1).

**PCT neighbourhood (6 points).** W = 42 (6 legs); W = 126 (2 legs); trend lookback 126 days;
clip range +/-0.5; clip range +/-2 (m in [0, 3], gross cap unchanged); vol window 126 days.

**Robustness runs for both (reported, not used for selection).** 2x costs; one extra day of
execution delay; sub-periods 2005-08, 2009-12, 2013-16, 2017-20, 2021-24; P&L concentration;
FF5 + Mom + Treasury + commodity factor regression with Newey-West errors; block-bootstrap Sharpe
interval; square-root-impact capacity curve. For IFC additionally: each sleeve alone, sleeve
correlations, the B2/B3 settlement and Friday tests, sleeve A threshold-rebalancing variant
(average over 0.5/1/1.5/2 % bands), and a stock/bond-correlation regime split.

---------------------------------------------------------------------------------------------

## 4. Independent replication on CME futures (Databento), reported, not used for selection

The same rules are re-run on a different instrument set and data vendor, as a replication:
Databento `GLBX.MDP3` (CME Globex) continuous contracts (volume-ranked front `.v.0` and next
`.v.1`), rolled on the vendor's front-contract change, with returns always computed within one
contract (no roll gaps), held fully collateralised at the T-bill rate, no borrow fee.

* **IFC-F.** ES replaces SPY and ZN replaces IEF (same windows, signals and sizing; DV01 issuance
  data unchanged). Prices are sampled from hourly bars at the bar ending 16:00 ET so they are
  synchronous with the ETF closes. Costs: 1.0 bp one-way for ES and ZN plus one round trip of the
  position at every roll.
* **PCT-F.** The PCT rules on 20 liquid futures (ES NQ RTY YM ZT ZF ZN ZB CL NG GC SI HG 6E 6J 6B
  6A 6C ZC ZS), daily bars, `next_close` execution, 1.5 bp one-way (5 bp for NG, RTY, ZC, ZS, 6A,
  6C), with RTY eligible only from its 2017 CME relisting.
* Sample: in-sample 2010-06-07 .. 2024-10-02; out-of-sample 2024-10-03 .. 2026-10-02 (evaluated once,
  together with the ETF versions). Capacity for the futures versions uses Databento contract volume.

---------------------------------------------------------------------------------------------

## 6. Amendment 1 (Sat 2026-10-03, committed before the author read any in-sample result)

**Why.** The team wants to know whether a materially higher risk-adjusted return (in-sample Sharpe
around 1.2-1.5 with small drawdowns) is reachable without overfitting. The honest route is
diversification across independent pre-registered edges plus risk control, searched in-sample only,
with every configuration logged and the search itself tested for overfitting. The out-of-sample
window stays locked and is still evaluated once.

**Candidate 3: Flow-and-Trend ensemble (FTE).** A risk-balanced combination of return streams that
were each specified above before any result: IFC sleeves A, B, C, exploratory sleeves D and E, and
PCT (and the plain TSMOM benchmark as an alternative trend leg).

**Exploration space (all in-sample, all logged, all counted in the Deflated Sharpe).**

* Stream subsets: every subset of {A, B, C, D, E} of size >= 2, combined with {none, PCT, TSMOM}
  as the trend leg.
* Weighting: equal risk (inverse trailing volatility); inverse volatility with a 0.5 shrink toward
  equal weights; minimum variance with Ledoit-Wolf shrinkage. All estimated on expanding windows
  of past stream returns only (at least 252 days), lagged one day.
* Risk overlay: portfolio volatility target of 6 %, 8 % or 10 % (trailing 63-day EWMA of the
  combined returns, lagged, leverage cap 4); and with or without a drawdown brake that halves
  exposure while the strategy is more than 10 % below its running peak (re-armed at a new peak).

**Overfitting controls and selection (replaces the section 0 rule for the final pick).**

1. Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado & Zhu 2017) by
   combinatorially symmetric cross-validation over all FTE configurations, with the in-sample
   period cut into 16 contiguous blocks (12,870 train/test splits).
2. The final strategy is the FTE configuration with the highest in-sample Deflated Sharpe Ratio
   (trial count = every in-sample trial in `results/trials.csv`), provided PBO < 0.25 and the
   configuration passes gates G1-G3; if PBO >= 0.25 the search is declared overfit and the
   section 0 rule (IFC vs PCT) decides instead.
3. Reported for the chosen configuration: its rank stability across the CSCV splits, the median
   Sharpe of all configurations (to show the choice is not a lucky outlier), and the OOS once.

No in-sample result has been looked at by the author when this amendment was written; build agents
were already running the section 1-2 specifications in parallel and their logs (`trials.csv`) were
not opened before this commit.

---------------------------------------------------------------------------------------------

## 5. Considered and not pursued (no backtests run)

Pre-FOMC drift (disappeared after 2015, Kurov, Wolfe & Gilbert 2021); the overnight drift (about
zero since 2021, NY Fed 2026); Goldman-roll front-running (arbitraged; Mou 2011); VIX-futures carry
(short-volatility tail risk); crypto funding carry (short sample, venue risk); intraday
momentum / leveraged-ETF rebalancing (needs long intraday history we do not have for free).
