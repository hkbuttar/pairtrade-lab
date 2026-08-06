# monitor/

Rolling structural-break monitoring: continuous re-testing of cointegration
on a trailing window (not just at fixed refit points) plus CUSUM detection on
spread residuals. When a break is flagged, the affected pair/basket is halted
and routed back through `pairs/` selection; it only re-trades if it
re-qualifies.

Not yet implemented.
