# backtest/

Event-driven, walk-forward pairs-trading simulator.

`engine.py` steps through trading days chronologically (`_run_backtest_on_panel`),
matching alpha-signal-lab's no-lookahead discipline: orders decided using
information through day t are filled at day t+1's open (`fills.py`, a flat
bps-per-trade cost, disclosed as simpler than alpha-signal-lab's ADV-scaled
market-impact model since pairs positions here are small and the point is to
test the pairs logic, not execution microstructure). `portfolio.py` is the
same generic cash/position ledger shape used across this portfolio's other
backtesters.

**Walk-forward**: at each refit date, `pairs.cointegration.test_sector_pairs`
re-runs using only data strictly before that date, and the resulting
FDR-significant pairs (up to `max_pairs`) are traded purely out-of-sample
until the next refit. A pair that continues to qualify keeps its position
and its hedge ratio/signal state (recomputed from its original tenure start,
not reset to flat at every refit); a pair that drops out is flattened.
Structural-break monitoring (`monitor/`) runs throughout and forces an
immediate flat ("halted" exit reason) whenever a pair's rolling
cointegration test or CUSUM detector fires, exactly the reactive pipeline
described in `monitor/README.md`.

**Position sizing**: each active pair gets a fixed notional, split across
its two legs in the *exact share ratio* implied by its hedge ratio (not
dollar-neutral: dollar-neutral would trade a different, non-stationary
linear combination than the one actually tested for cointegration). Share
counts are locked in at entry and held fixed until exit, not continuously
re-targeted to notional every day.

**Risk layer** (`risk/`): every new pair entry's notional is clipped by
`risk.limits.clip_new_pair_notional` (a per-pair cap, then whatever gross
exposure budget remains once already-open positions are counted, both as a
fraction of *current* equity) before it becomes an order. A portfolio-level
`risk.kill_switch.KillSwitch` checks equity every bar; once triggered
(sticky, no auto-resume, manual `reset()` only — the same design shared
across alpha-signal-lab, bookmaker, and execedge, see `risk/README.md`),
every open position is flattened and no further selection or signal
processing happens for the rest of the run. Stop-loss on spread divergence
is `signals/spread.py`'s own `stop_loss_threshold`, already wired in.

**Not yet included**: baskets (`pairs/johansen.py`) aren't wired into this
engine — multi-leg fills add real complexity, and the basket search has
found very few significant baskets to trade anyway.

`bootstrap.py` adds statistical rigor on top of the point estimates: a
circular block bootstrap (Politis & Romano, 1992) resamples whole
contiguous blocks of consecutive days (not individual days, which would
destroy the autocorrelation a held pairs position or a flat stretch both
create) to build confidence intervals for CAGR, Sharpe, Sortino, max
drawdown, and win rate. `block_length` is a disclosed, unfitted parameter
(default 20 trading days). Validated against a synthetic AR(1) series with
known autocorrelation: the block bootstrap recovers it closely while a
naive i.i.d. bootstrap on the same series destroys it almost entirely (see
`tests/backtest/test_bootstrap.py`).

`run_backtest.py` is the CLI entry point; add `--bootstrap` for the CIs:

```bash
python -m backtest.run_backtest --start 2018-01-01 --end 2025-01-01 --bootstrap
```

**Live result over 2018-2025**: CAGR -0.94%, Sharpe -0.69, max drawdown
6.6%, 17 trades total across 5 different pairs surfaced at different refit
windows (ES/WEC, AVGO/ORCL, ADBE/CRM, GOOGL/NVDA, AMD/MSFT), 3 of which were
force-closed by structural-break monitoring rather than a normal
reversion/stop-loss exit. Re-run with `--cost-bps 0`: CAGR -0.79%, Sharpe
-0.59, so transaction costs account for only a small part of the
underperformance (about -0.15pp of CAGR); the strategy loses money on this
universe and window even before costs, not because of them.

**Block bootstrap 95% CIs** (2000 resamples, 20-day blocks): Sharpe
**[-1.14, -0.17]**, CAGR **[-1.96%, -0.16%]**. Neither interval crosses
zero. With only 17 trades, a natural assumption is that the negative point
estimate is just small-sample noise; the bootstrap says otherwise, at
least at this confidence level, block length, and window — the negative
result is reasonably robust to resampling, not an artifact of one unlucky
trade sequence.

This is a plain "no, it doesn't (yet) work" result, reported as exactly
that rather than reframed. A few honest caveats on how to read it, not
excuses for the result:
- Only 17 trades over 7 years is a very small sample. The bootstrap CIs
  above account for this better than a bare point estimate would, but a
  block bootstrap still can't manufacture information that was never in
  the original 7-year sample; a materially longer backtest window would
  still be worth more than a wider resampling budget on this one.
- `win_rate` (1.3%) is computed over *every* calendar day in the backtest,
  and the strategy is flat (0% return that day) on the large majority of
  them since so few pairs are ever tradable at once; it is not a
  per-trade win rate and reads far worse than the trade log itself, which
  shows 14 of 17 trades exiting on clean reversion.
- `max_pairs=3` but rarely more than one pair actually qualified at once,
  so most of the portfolio's capital sat in cash most of the time; this
  under-utilization is itself a consequence of how few pairs survive FDR
  correction and rolling monitoring in this universe (see `pairs/` and
  `monitor/`), not a bug in the engine.
- The kill-switch never triggered and the notional limits never bound at
  this scale (see `risk/README.md`): a 6.6% max drawdown is well under the
  15% default kill-switch threshold. That's the risk layer behaving
  correctly as a backstop, not evidence it did anything here.
