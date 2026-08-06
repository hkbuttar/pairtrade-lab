"""CLI entry point: compare static (periodic OLS refit) vs. dynamic (Kalman
filter) hedge ratio estimation on one pair.

Usage:
    python -m signals.run_hedge_ratio_comparison --ticker-y BAC --ticker-x PNC \\
        --start 2018-01-01 --end 2025-01-01

Reports out-of-sample one-step-ahead prediction RMSE for both methods (see
signals/hedge_ratio.py module docstring for why this metric, not a
backtested P&L) plus how much each hedge ratio actually moved over the
window, and states plainly whether the Kalman filter's added complexity
earned its keep on this pair, rather than assuming it did.
"""

from __future__ import annotations

import argparse

from data.prices import load_prices, to_price_panel
from signals.hedge_ratio import compare_hedge_ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker-y", required=True)
    parser.add_argument("--ticker-x", required=True)
    parser.add_argument("--start", required=True, help="ISO start date, inclusive")
    parser.add_argument("--end", required=True, help="ISO end date, exclusive")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    parser.add_argument("--refit-every", type=int, default=60)
    parser.add_argument("--min-window", type=int, default=60)
    parser.add_argument("--delta", type=float, default=1e-4)
    args = parser.parse_args()

    tickers = [args.ticker_y, args.ticker_x]
    long_prices = load_prices(tickers, args.start, args.end, source=args.source)
    panel = to_price_panel(long_prices)
    y, x = panel[args.ticker_y], panel[args.ticker_x]

    comparison = compare_hedge_ratios(
        y, x, refit_every=args.refit_every, min_window=args.min_window, delta=args.delta
    )

    static_range = comparison.static["hedge_ratio"].agg(["min", "max"])
    kalman_range = comparison.kalman["hedge_ratio"].agg(["min", "max"])

    print(f"{args.ticker_y} ~ {args.ticker_x}, {comparison.n_obs} observations\n")
    print(f"Static  (refit every {args.refit_every}d): range [{static_range['min']:.3f}, "
          f"{static_range['max']:.3f}], one-step-ahead RMSE = {comparison.static_rmse:.4f}")
    print(f"Kalman  (delta={args.delta:g}):           range [{kalman_range['min']:.3f}, "
          f"{kalman_range['max']:.3f}], one-step-ahead RMSE = {comparison.kalman_rmse:.4f}")

    better = "Kalman" if comparison.kalman_rmse < comparison.static_rmse else "Static"
    improvement = abs(comparison.kalman_rmse - comparison.static_rmse) / comparison.static_rmse
    print(
        f"\n{better} hedge ratio had lower one-step-ahead prediction error "
        f"({improvement:.1%} {'lower' if better == 'Kalman' else 'higher than Kalman'})."
    )


if __name__ == "__main__":
    main()
