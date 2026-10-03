# Running our strategies inside the organisers' starter kit

`run_in_starter.py` runs a backtrader strategy through the organisers' starter kit
(`gqh-webull-backtrader-starter`) on our cached data. It mirrors the kit's `examples/backtest/main.py` and
imports the kit's own code at runtime from `GQH_STARTER_KIT`. Nothing from the kit is copied into this repo or
changed. The Webull SDK needs credentials, so when it is not installed it is replaced by placeholder modules.

## What is the kit's and what is substituted

Taken from the kit unchanged, at runtime:

* the strategy loader `_load_strategy_class` (module-level `STRATEGY_CLASS`), the parameter parser
  `_parse_strategy_params` (the `WEBULL_STRATEGY_PARAMS` format) and the symbol parser `_parse_symbols`;
* `bt.Cerebro()` with its default observers, the `FixedSize(stake=10)` sizer, and the analyzers `DrawDown`
  "drawdown", `TradeAnalyzer` "trades", `SharpeRatio` "sharpe" (data0's timeframe, `riskfreerate=0`,
  `annualize=True`) and `RecorderAnalyzer` "recorder";
* `cerebro.run(runonce=False)`, then `_compute_metrics`, `_print_results` and `_maybe_render_report`. The report
  uses the Plotly renderer when plotly is installed and the Lightweight Charts renderer otherwise.

Substituted, and nothing else:

1. **Data.** One `PandasData` feed per symbol from our cache, through `engine.load_ohlc`: ETFs from
   `etf_daily`, and CME futures as total-return indices `F_*` from `futures_daily`. A fully collateralised index
   level behaves like an ETF price, with no multiplier. Each bar is stamped at 16:00 New York time (as naive UTC),
   so the kit's `to_market_tz` displays the right session. Before a series starts, the feed repeats its first close;
   a missing print repeats the last close. Both give a zero return. Besides OHLCV the feed carries the kit's
   `trading_session` line (0 for daily bars) and a `valid` line (1 means a real print), so a native strategy can
   skip padded days.
2. **Costs** (`--costs`):
   * `none` is the kit's default.
   * `guide` follows docs/USAGE_EN.md: `setcommission(0.0005)` plus `set_slippage_perc(0.0005)`, i.e. 5 bp + 5 bp
     per side.
   * `project` uses our one-way bps per instrument as commission: `forward2.S1_COST_BPS` for the listed
     futures, otherwise `config.cost_bps`. `--cost-bps T=x,...` overrides individual symbols.
3. **`--margin`.** A margin account: every commission scheme gets `leverage=10` and `checksubmit` is off, as in
   `validation/backtrader_crosscheck.py`. It is needed above 1x gross. Without it, backtrader cancels any fill that
   would take cash below zero. Size orders yourself (see `base.py`): with leverage 10, `order_target_percent`
   oversizes by 10x.
4. **Cash.** The default is 1e9 for modules in `strategies/`. The kit's 100,000 in whole shares distorts target
   weights: one $556 SPY share is 0.56 % of NAV. Kit examples keep the kit's 100,000. `--cash` overrides both.

The harness also adds a daily `TimeReturn`, the net and gross weight held each day, an order log with the notional
filled per session, and (with costs on) `ExtraCosts`, described below.

## Usage

From the repo root (Git Bash):

```
export GQH_DATA_DIR=<data cache> GQH_STARTER_KIT=<unzipped kit root>
python validation/starter_kit/run_in_starter.py --strategy dual_ma --symbols SPY \
    --start 2005-01-03 --end 2024-10-02 --json out/dual_ma.json --report out/dual_ma.html
```

| option | meaning |
|---|---|
| `--strategy` | module in `strategies/` (searched first) or in the kit's `examples/strategies/` |
| `--symbols` | comma list; default: the module's `SYMBOLS` |
| `--start`, `--end` | sessions fed; default: the first print of any symbol through the last cached session (2026-10-02) |
| `--is-start` | first day of the in-sample window; default: the module's `IS_START`, else `--start` |
| `--costs none\|guide\|project`, `--cost-bps` | see above |
| `--no-extra-costs` | the kit's setters only (no `ExtraCosts`) |
| `--margin`, `--cash`, `--params k=v,...` | see above; `--params` uses the kit's format and coercion |
| `--json`, `--series`, `--report`, `--report-engine` | metrics JSON, daily CSV, the kit's HTML report |
| `--reference <csv>`, `--reference-column`, `--reference-rf tbill\|zero` | agreement with our engine's daily returns |

Only the `FWD` period is read. The out-of-sample lock is never released, and nothing is written to
`results/trials.csv` or `results/oos_log.csv`. The harness refuses to start when `GQH_OOS_UNLOCK` is set.

## Strategy modules

Strategy modules follow the kit's convention: a module-level `STRATEGY_CLASS`. Optional attributes, each a
constant or a callable of the parsed `--params`, are `SYMBOLS`, `EXEC` (`"coc"` turns on cheat-on-close),
`NEEDS_MARGIN`, `CASH` and `IS_START`.

`base.WeightStrategy` sizes orders in whole shares from the NAV and close of the submitting bar. Sells go before
buys. It records `closed_trades` in the kit's format. Our engine's execution conventions map onto it as follows:

* `next_open`: a plain market order submitted in `next()` of bar d. It fills at the open of d+1.
* `next_close`: a `bt.Order.Close` submitted in `next()` of bar d. It fills at the close of d+1.
* `coc`: a market order with cheat-on-close. It fills at the close of the submitting bar, at exact weights.
  It is used only to replay weights decided a session earlier.

A native strategy computes its signals inside backtrader from the feeds. Side data, such as T-bill rates or
Treasury auction announcements, is loaded in `__init__` and read only up to the current bar's date.

### Native ports of our strategies

Each computes its signals inside backtrader from the feeds it receives, plus side data read only up to the current
bar. Each needs the harness's default `--start` for its warm-ups and raises otherwise. Month ends come from the
published NYSE schedule (`base.planned_month_ends`).

| module | strategy | timing | check |
|---|---|---|---|
| `gqh_flow_clock` | F1, the submission (A + C + TSMOM, min-variance ensemble, 8 % vol target, 4x cap); margin | A/C `bt.Order.Close`, TSMOM market at the open | `check_flow_clock.py` |
| `gqh_equity_trend` | S2 (`book=S1` for S1 alone), 30 CME futures indices; margin | `bt.Order.Close` | module docstring |
| `gqh_gtaa` | S3, Faber GTAA (`buy_and_hold=true` for BH5); the kit's cash broker | market at the open | `check_gtaa.py` |

The signals equal the engine's: the F1 internals to 1e-16, all 196 S2 month-end decisions to 4e-15, and all 261 S3
and BH5 month-end decisions exactly. What remains is execution (sizing at the close of d before a fill at the next
open or close) and the day costs are booked: backtrader books a close fill's commission on the fill session, while
the engine books it on the first return day held. A `fill=coc` diagnostic (F1, S2) holds the engine's exact book.

`run_all.py --work <dir>` regenerates `results/validation/starter_kit/`. That covers the four strategies (F1, S2,
S3, BH5) at guide and project costs, over the full history to 2026-10-02: a metrics JSON per run, agreement with
our engine at the same cost rate, and `SUMMARY.md`. The kit's HTML reports go to `<dir>`, and are copied into
`results/` only when under 5 MB.

### Replays

`strategies/replay_weights.py` is a plumbing check, not a native strategy. It replays a CSV of the engine's held
weights with `fill=coc|close|open`. `export_f1.py --out <dir>` writes the submitted strategy's (F1) ETF weights and
two reference series from the cross-check's `submitted_strategy()`.

## Costs backtrader does not see (`ExtraCosts`, on unless `--no-extra-costs`)

* **Futures rolls.** A total-return index hides the roll trade. Each roll is charged as one extra round trip of
  the position held into the roll session, at the one-way cost (the engine's rule). Roll dates follow the published
  contract calendar, so the flag of session t is read at t-1.
* **Short-ETF borrow.** `config.SHORT_BORROW_BPS_PER_YEAR` on short ETF positions.
* **Guide slippage that backtrader gets wrong.**
  * `bt.Order.Close` fills are never slipped, so their 5 bp is debited.
  * Cheat-on-close fills slip the close of t-1 but cap the result at bar t's high or low, a price that did not
    exist at that close. The cap often turns the charge into a gain, so it is replaced by 5 bp of the close of t-1.
  * Market orders filled at the open are slipped from the open but capped at that bar's high or low, so the
    shortfall to 5 bp of the open is debited. Before this top-up, 9 to 11 % of the open fills of TSMOM and S3
    paid less than 5 bp (a mean of about 4.7 bp).

Roll and borrow for return day t are debited after `next()` of bar t-1 and land in bar t's NAV. Slippage is booked
when the fill is notified, through `broker.add_cash` plus the broker's private `_get_value()`, so it lands on the
fill's own bar.

## Metrics

The JSON holds:

* the kit's `_compute_metrics` output;
* per window, on the daily `TimeReturn`, the kit Sharpe and our Sharpe (definitions below), annual return,
  annual volatility, maximum drawdown, the mean T-bill rate, mean and maximum net/gross weight, turnover (notional
  filled per year over the NAV of the previous close, one way, futures rolls not included), orders and refused
  orders (Margin or Rejected, counted by creation date), and the extra-cost drag;
* order totals;
* agreement with `--reference`: correlation, mean and maximum absolute daily difference in bp, and the reference's
  Sharpe ratios on the same days.

The windows are in-sample (`--is-start` to 2024-10-02), out-of-sample (2024-10-03 to 2026-10-02) and the full run.

The two Sharpe definitions:

* **Kit Sharpe:** mean / population std of daily backtrader returns, times sqrt(252), with rf = 0. This is the
  kit's analyzer formula.
* **Our Sharpe:** mean / sample std of daily excess returns, times sqrt(252), where excess = backtrader return -
  net weight held x T-bill. Backtrader pays no interest on cash, so this equals our engine's convention: cash earns
  the T-bill, borrowed cash pays it. "Net weight held over day t" is the position after bar t's fills, minus the
  Close fills at t, valued at the close of t-1 over the NAV of t-1.

## Verification

Run on 2026-10-03, backtrader 1.9.78.123, data through 2026-10-02.

**Smoke test** (kit `dual_ma`, SPY 2005-01-03 to 2024-10-02, the kit's defaults). It reproduces
`backtrader_crosscheck.json` section 1 exactly: final value 102,880.31, Sharpe 0.5631, 141 trades, maximum
drawdown 0.68 %.

**F1 weight replay** (`fill=coc`, project costs, `--no-extra-costs`, `--is-start 2006-05-08`). It reproduces
section 4 to all eight printed digits:

| window | ann. return | Sharpe (total, sample std) | kit Sharpe | max DD |
|---|---|---|---|---|
| in-sample | 8.4836 % | 1.1091 | 1.1093 | -11.64 % |
| out-of-sample | 1.8239 % | 0.3074 | 0.3078 | -8.98 % |

Over the whole run the kit analyzer gives 1.0069669 (section 5: 1.0069669), from 72,917 orders with none refused.
Against our engine on the same weights (rf = 0), the daily returns correlate at 1.0, and the mean absolute
difference is 0.0005 bp in-sample and 0.0006 bp out-of-sample.

**F1 replay at the guide's costs.** With the literal setters (`--no-extra-costs`) it reproduces the first version of
the cross-check's section 6: kit Sharpe 0.9467 / 0.1888, annual return 7.01 % / 1.06 %. Those figures understate the
cost (section 6 has since been corrected to charge the slippage inside a 10 bp commission), for two reasons:

* The F1 netted book turns over 81x a year in-sample and 95x out-of-sample.
* Under cheat-on-close, backtrader's capped slippage earned money. The literal run beats a 5 bp commission-only
  run, whose kit Sharpe is 0.9004 / 0.0282.

With the slippage booked correctly, the guide's 5 + 5 bp gives the following. A plain 10 bp commission gives the
same (kit Sharpe 0.3575 / -0.6825):

| fill | kit Sharpe IS / OOS | our Sharpe IS / OOS |
|---|---|---|
| `coc` | 0.3575 / -0.6826 | 0.2837 / -1.0477 |
| `close` | 0.3646 / -0.6936 | 0.2902 / -1.0645 |

**Native timing, measured on weight replays.**

* F1 with `bt.Order.Close` (`fill=close`, sized one session before the fill), against exact weights, project
  costs without extras: correlation 0.9966 in-sample and 0.9941 out-of-sample, mean absolute difference 1.92 / 2.29
  bp a day. Kit Sharpe 1.1162 / 0.3129 against 1.1093 / 0.3078.
* S3 with `fill=open` (market orders sized at the previous close), project costs, against our engine's S3 net
  returns in excess of the T-bill: correlation 0.99996 / 0.99994, mean absolute difference 0.16 / 0.21 bp. Our
  Sharpe 0.5054 / 0.6982 against the engine's 0.5106 / 0.7002.

**Futures path** (S2 engine weights, `fill=coc`, project costs with roll debits, `--is-start 2011-09-02`): excess
returns correlate at 1.0 with our engine's, mean absolute difference 0.0001 bp. Our Sharpe 0.5032 / 0.4627 against
the engine's 0.5032 / 0.4628. Roll costs come to 0.55 % a year in-sample.

## Known deviations

* The kit Sharpe of any book with idle cash is lower than our engine's total-return Sharpe, because backtrader
  pays no interest on cash. Use our Sharpe to compare with the engine.
* Whole shares. At 1e9 the rounding is negligible; at 100,000 it is not.
* Sizing at the submitting bar's close: weights drift between sizing and fill (the native timing figures above).
* The order counts include orders still queued when the data ends.
* `ExtraCosts` calls the broker's private `_get_value()`, and it reads roll flags one session ahead. These are
  cost-accounting choices; signals never see them.
