# signals/

Spread construction and entry/exit signal generation.

`hedge_ratio.py` implements both hedge ratio estimators named in the project
plan:

- **Static**: OLS refit every `refit_every` bars on a trailing `min_window`,
  held fixed in between (`static_hedge_ratio_series`).
- **Dynamic**: a Kalman filter treating `[hedge_ratio, intercept]` as a
  random walk, updated every observation (`kalman_hedge_ratio_series`),
  seeded from a static OLS fit on the first `warmup` bars rather than an
  arbitrary initial guess.

`compare_hedge_ratios` scores both by strictly out-of-sample one-step-ahead
prediction error: at each t, only the hedge ratio/intercept known as of t-1
is used to predict y_t from x_t. This is a narrower question than "which one
makes more money" (that needs the full event-driven simulator in
`backtest/`, not built yet) — it only asks which update mechanism tracks the
true relationship more closely.

`run_hedge_ratio_comparison.py` is the CLI entry point:

```bash
python -m signals.run_hedge_ratio_comparison --ticker-y BAC --ticker-x PNC \
    --start 2018-01-01 --end 2025-01-01
```

**Live result on BAC/PNC** (the one pair that survived FDR correction in
`pairs/`): the Kalman filter's one-step-ahead RMSE was 65-84% lower than
static's, tested across refit cadences from 10 to 252 days — the finding
holds regardless of how favorably (or not) the static baseline is tuned, not
an artifact of picking an unfair static window. On this pair, at least, the
added complexity of the Kalman filter earned its keep. Whether that holds up
once transaction costs and realistic entry/exit rules are added (rather than
raw tracking error) is a question for `backtest/`, once it exists.

`spread.py` builds on top of the hedge ratios: `pair_spread` computes
`y - hedge_ratio * x - intercept` (the same quantity the Engle-Granger ADF
test was run on), accepting either a scalar or a per-bar `pd.Series` hedge
ratio/intercept so the spread stays point-in-time correct when fed a static
or Kalman series rather than one full-sample estimate. `basket_spread` is
the basket analogue, a weighted sum of legs using e.g. a Johansen
cointegrating-vector's weights. `rolling_zscore` z-scores the spread against
its own trailing mean/std (window length an explicit, disclosed parameter).
`generate_signals` is a stateful entry/exit/stop-loss machine (defaults
2.0/0.5/3.5, the plan's own worked example): flat positions enter on
`|z| >= entry_threshold`, held positions exit on `|z| <= exit_threshold`
(reversion) or `|z| >= stop_loss_threshold` (stop-loss); positions never
flip directly between long and short without passing through flat.
`summarize_trades` turns a position series into a trade-by-trade table
(side, dates, holding period, exit reason) — deliberately descriptive, not a
P&L calculation, since entries/exits here are signal transitions with no
transaction costs or fills modeled; real performance numbers need the
event-driven simulator in `backtest/`, not yet built.

`run_spread_signals.py` is the CLI entry point:

```bash
python -m signals.run_spread_signals --ticker-y BAC --ticker-x PNC \
    --start 2018-01-01 --end 2025-01-01
```

**Live result on BAC/PNC**: with the Kalman hedge ratio, 77 trades over the
full window, averaging 2.5 bars held, and *zero* stop-losses hit — every
trade reverted cleanly. With the static hedge ratio on the same pair and
thresholds, only 65 trades but averaging 10.6 bars held and 5 stop-losses.
This is a coherent continuation of the Step 3 finding: a hedge ratio that
tracks the true relationship more closely produces a spread that reverts
faster and more reliably, while a stale periodically-refit hedge ratio lets
the spread wander further before reverting (or blowing through the
stop-loss instead).
