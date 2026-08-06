# backtest/

Event-driven simulator (point-in-time correct, no lookahead, next-bar fills,
explicit transaction cost assumptions) with walk-forward validation: periodic
re-selection and re-estimation using only data available before each refit
point, tested purely out-of-sample thereafter. Block bootstrap over the
walk-forward results for confidence intervals on every major comparison in
the README.

Not yet implemented.
