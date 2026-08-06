# monitor/

Rolling structural-break monitoring, combining two complementary signals in
`structural_break.py`:

- **Rolling cointegration re-test** (`rolling_cointegration_pvalue`):
  re-runs the same Engle-Granger test from `pairs/cointegration.py` on a
  trailing window at every new bar (default: every bar, not just at fixed
  refit points), so relationship decay shows up as it develops.
- **CUSUM structural-break detection** (`cusum_detect`): Page's (1954)
  two-sided CUSUM control chart applied to the spread's rolling z-score
  (`signals/spread.py`). A cointegrated pair's z-score should oscillate
  around 0; CUSUM accumulates evidence of a *sustained* directional drift,
  a faster, more principled signal than waiting for one large z-score
  excursion that could just be noise. This is Page's control-chart CUSUM,
  not the same-named Brown-Durbin-Evans regression-stability test from the
  econometrics literature; disclosed to avoid ambiguity between the two.

`monitor_pair_status` combines both into a halt/reinstate state machine: a
pair goes HALTED the instant either signal fires, and only goes back to
ACTIVE after `requalify_bars` consecutive bars with the rolling p-value back
under threshold. This is a disclosed simplification of "route it back
through the Step 2/2b selection pipeline": true re-qualification would rerun
the full FDR-corrected batch test across the current universe, not just
recheck this one pair's raw p-value, which would reintroduce the
multiple-comparisons problem the batch pipeline exists to control for. That
fuller loop needs the batch pipeline in `pairs/`, not a per-pair monitor.

`run_monitor.py` is the CLI entry point:

```bash
python -m monitor.run_monitor --ticker-y BAC --ticker-x PNC \
    --start 2018-01-01 --end 2025-01-01
```

**Live result on BAC/PNC**, and it's a genuinely important finding: despite
passing the full-sample Engle-Granger test decisively (raw p=0.00004, the
only pair to survive FDR correction across the whole universe), the pair
spends **86% of the 2018-2025 window HALTED** under continuous 90-day
rolling re-testing, with individual halt stretches as long as 632 calendar
days. This isn't a monitor false-positive: checked across rolling windows
from 90 to 252 days (a full year), the halted fraction only drops from 86%
to 70%, so it doesn't go away as the test gets more powerful with more data
per window. The honest takeaway is that full-sample cointegration
significance is not the same claim as "the relationship holds throughout
the sample": a static, once-and-done selection test can create false
confidence that the monitoring in this module exists specifically to catch.
Whether trading only during the ACTIVE windows would have been profitable
net of the resulting low utilization and the whipsaw of repeated halt/resume
cycles is a question for `backtest/`, not yet built.
