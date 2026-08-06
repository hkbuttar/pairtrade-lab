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

Spread construction and z-score entry/exit/stop-loss rules on top of these
hedge ratios, and Johansen cointegrating-vector weights for baskets (already
computed in `pairs/johansen.py`, not yet wired into a spread here), are not
yet implemented.
