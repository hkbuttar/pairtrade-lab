"""CLI entry point: run rolling cointegration re-testing + CUSUM structural-
break detection on one pair and report the resulting active/halted timeline.

Usage:
    python -m monitor.run_monitor --ticker-y BAC --ticker-x PNC \\
        --start 2018-01-01 --end 2025-01-01

Reports every halt event (date, trigger: cointegration breakdown vs. CUSUM,
or both), how long each halt lasted, and the pair's final status. This is
monitoring/diagnostic output, not a backtest: no positions or P&L are
computed here (that's signals/ + backtest/, once the latter exists), only
whether the pair would have been flagged as tradable at each point in time.
"""

from __future__ import annotations

import argparse

from data.prices import load_prices, to_price_panel
from monitor.structural_break import (
    cusum_detect,
    monitor_pair_status,
    rolling_cointegration_pvalue,
)
from signals.hedge_ratio import kalman_hedge_ratio_series, static_hedge_ratio_series
from signals.spread import pair_spread, rolling_zscore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker-y", required=True)
    parser.add_argument("--ticker-x", required=True)
    parser.add_argument("--start", required=True, help="ISO start date, inclusive")
    parser.add_argument("--end", required=True, help="ISO end date, exclusive")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    parser.add_argument("--hedge-ratio-method", default="kalman", choices=["kalman", "static"])
    parser.add_argument("--zscore-window", type=int, default=20)
    parser.add_argument("--rolling-window", type=int, default=90)
    parser.add_argument("--rolling-step", type=int, default=1)
    parser.add_argument("--pvalue-threshold", type=float, default=0.05)
    parser.add_argument("--cusum-k", type=float, default=0.5)
    parser.add_argument("--cusum-h", type=float, default=5.0)
    parser.add_argument("--requalify-bars", type=int, default=20)
    args = parser.parse_args()

    tickers = [args.ticker_y, args.ticker_x]
    long_prices = load_prices(tickers, args.start, args.end, source=args.source)
    panel = to_price_panel(long_prices)
    y, x = panel[args.ticker_y], panel[args.ticker_x]

    if args.hedge_ratio_method == "kalman":
        estimate = kalman_hedge_ratio_series(y, x)
    else:
        estimate = static_hedge_ratio_series(y, x)
    spread = pair_spread(y, x, estimate["hedge_ratio"], estimate["intercept"])
    z_score = rolling_zscore(spread, window=args.zscore_window)

    rolling_pvalue = rolling_cointegration_pvalue(
        y, x, window=args.rolling_window, step=args.rolling_step
    )
    cusum = cusum_detect(z_score, k=args.cusum_k, h=args.cusum_h)
    status = monitor_pair_status(
        rolling_pvalue,
        cusum["break_flagged"],
        pvalue_threshold=args.pvalue_threshold,
        requalify_bars=args.requalify_bars,
    )

    print(f"{args.ticker_y} ~ {args.ticker_x}, {len(status)} monitored bars\n")

    prev = "ACTIVE"
    halt_start = None
    n_halts = 0
    for date, current in status.items():
        if prev == "ACTIVE" and current == "HALTED":
            halt_start = date
            trigger_cointegration = rolling_pvalue.loc[date] > args.pvalue_threshold
            trigger_cusum = bool(cusum["break_flagged"].loc[date])
            trigger = ", ".join(
                t
                for t, fired in [
                    ("cointegration breakdown", trigger_cointegration),
                    ("CUSUM", trigger_cusum),
                ]
                if fired
            )
            n_halts += 1
            print(f"HALT   {date.date()}  (trigger: {trigger})")
        elif prev == "HALTED" and current == "ACTIVE":
            duration = (date - halt_start).days
            print(f"RESUME {date.date()}  (halted {duration} calendar days)")
        prev = current

    print(f"\n{n_halts} halt events over the window. Final status: {status.iloc[-1]}")


if __name__ == "__main__":
    main()
