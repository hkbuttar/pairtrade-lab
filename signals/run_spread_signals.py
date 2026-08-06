"""CLI entry point: build a pair's spread, z-score it, and generate
entry/exit/stop-loss signals.

Usage:
    python -m signals.run_spread_signals --ticker-y BAC --ticker-x PNC \\
        --start 2018-01-01 --end 2025-01-01

Reports descriptive trade statistics only (count, side breakdown, holding
period, exit reason): entries/exits here are signal transitions, not fills,
and no transaction costs or slippage are modeled, so this is not a backtest.
Real P&L needs the event-driven simulator in backtest/, not yet built.
"""

from __future__ import annotations

import argparse

from data.prices import load_prices, to_price_panel
from signals.hedge_ratio import kalman_hedge_ratio_series, static_hedge_ratio_series
from signals.spread import generate_signals, pair_spread, rolling_zscore, summarize_trades


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker-y", required=True)
    parser.add_argument("--ticker-x", required=True)
    parser.add_argument("--start", required=True, help="ISO start date, inclusive")
    parser.add_argument("--end", required=True, help="ISO end date, exclusive")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    parser.add_argument("--hedge-ratio-method", default="kalman", choices=["kalman", "static"])
    parser.add_argument("--zscore-window", type=int, default=20)
    parser.add_argument("--entry-threshold", type=float, default=2.0)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--stop-loss-threshold", type=float, default=3.5)
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
    positions = generate_signals(
        z_score,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_loss_threshold=args.stop_loss_threshold,
    )
    trades = summarize_trades(positions, z_score, exit_threshold=args.exit_threshold)

    print(
        f"{args.ticker_y} ~ {args.ticker_x}, {args.hedge_ratio_method} hedge ratio, "
        f"z-score window={args.zscore_window}\n"
        f"entry={args.entry_threshold}, exit={args.exit_threshold}, "
        f"stop_loss={args.stop_loss_threshold}\n"
    )

    if trades.empty:
        print("No trades triggered over this window.")
        return

    print(trades.to_string(index=False))

    n_trades = len(trades)
    n_long = int((trades["side"] == "long").sum())
    n_short = n_trades - n_long
    reason_counts = trades["exit_reason"].value_counts()
    avg_bars_held = trades["bars_held"].mean()

    print(f"\n{n_trades} trades ({n_long} long, {n_short} short)")
    print(f"Avg bars held: {avg_bars_held:.1f}")
    print("Exit reasons: " + ", ".join(f"{k}={v}" for k, v in reason_counts.items()))


if __name__ == "__main__":
    main()
