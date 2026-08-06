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

**Not yet included**: baskets (`pairs/johansen.py`) aren't wired into this
engine — multi-leg fills add real complexity, and the basket search has
found very few significant baskets to trade anyway. The risk layer
(`risk/`, kill-switch and position limits) doesn't exist yet either, so
nothing here stops a losing streak beyond the strategy's own stop-loss and
monitoring logic. Block bootstrap confidence intervals on the results below
are `backtest/` future work, not yet built.

`run_backtest.py` is the CLI entry point:

```bash
python -m backtest.run_backtest --start 2018-01-01 --end 2025-01-01
```

**Live result over 2018-2025**: CAGR -0.94%, Sharpe -0.69, max drawdown
6.6%, 17 trades total across 5 different pairs surfaced at different refit
windows (ES/WEC, AVGO/ORCL, ADBE/CRM, GOOGL/NVDA, AMD/MSFT), 3 of which were
force-closed by structural-break monitoring rather than a normal
reversion/stop-loss exit. Re-run with `--cost-bps 0`: CAGR -0.79%, Sharpe
-0.59, so transaction costs account for only a small part of the
underperformance (about -0.15pp of CAGR); the strategy loses money on this
universe and window even before costs, not because of them.

This is a plain "no, it doesn't (yet) work" result, reported as exactly
that rather than reframed. A few honest caveats on how to read it, not
excuses for the result:
- Only 17 trades over 7 years is a very small sample; none of the headline
  metrics (Sharpe, Sortino, win rate) should be trusted as a precise
  estimate without the block bootstrap confidence intervals this project
  plan calls for, not yet built.
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
