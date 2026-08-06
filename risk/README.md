# risk/

Portfolio-level risk controls, wired into `backtest/engine.py`.

`kill_switch.py` is a fractional-drawdown kill-switch, deliberately close
to line-for-line identical to alpha-signal-lab's (bookmaker and execedge
each have their own variant too). All four independent trading systems in
this portfolio converged on the same shape: a stateful, streaming monitor
of drawdown from the running equity peak, sticky once triggered (does not
auto-resume just because equity ticks back up), reinstated only by an
explicit, deliberate `reset()` call. That convergence is treated as a
portfolio-wide design principle, not a coincidence worth re-deriving from
scratch each time: a kill-switch that can silently forgive itself isn't
one. `backtest/engine.py` checks it every bar; once triggered, every open
position is flattened and no further selection or signal processing
happens for the rest of the run.

`limits.py` is the position-limit backstop: `clip_new_pair_notional` caps
any single new pair entry's notional (as a fraction of *current* equity,
so the cap tightens automatically as equity drops) and then caps whatever
gross exposure budget remains once already-open positions are counted.
Existing positions are never resized to make room for a new one, only new
entries are constrained. Applied downstream of `backtest/engine.py`'s own
sizing scheme (a fixed notional per pair), the same "clip after sizing"
layering as alpha-signal-lab's `risk/limits.py`.

Stop-loss on spread divergence is not duplicated here: it already lives in
`signals/spread.py`'s `generate_signals` (`stop_loss_threshold`), wired
into `backtest/engine.py` via `_compute_pair_series`. It's part of this
risk layer conceptually even though the code sits in `signals/`.

At the defaults used in the live backtest result (`backtest/README.md`),
neither the kill-switch nor the notional limits ever actually bind: max
drawdown over 2018-2025 was 6.6%, well under the 15% kill-switch threshold,
and at most one pair was ever open at a time, well under the 50%-per-pair /
100%-gross limits. This is expected and correct for a backstop: it should
be invisible until the strategy actually needs it, not a constraint that
shapes ordinary behavior.
