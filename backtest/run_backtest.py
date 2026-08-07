"""CLI entry point: run the full walk-forward pairs backtest.

Usage:
    python -m backtest.run_backtest --start 2018-01-01 --end 2025-01-01
    python -m backtest.run_backtest --start 2018-01-01 --end 2025-01-01 --bootstrap

Walk-forward: pair selection and hedge ratio estimation are re-run at each
refit date using only data available strictly before that date; the
resulting pairs are traded purely out-of-sample until the next refit.
Structural-break monitoring (monitor/) runs throughout and forces a flat
position whenever a pair is HALTED. See backtest/engine.py for the full
mechanics and backtest/README.md for a live result and its honest caveats.

--bootstrap adds block-bootstrap confidence intervals (backtest/bootstrap.py)
on top of the point-estimate metrics, since with as few trades as this
strategy tends to produce, a bare point estimate (e.g. "Sharpe -0.69") is
easy to overread as more precise than it is.
"""

from __future__ import annotations

import argparse

from backtest.bootstrap import bootstrap_backtest_metrics
from backtest.engine import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="ISO start date, inclusive")
    parser.add_argument("--end", required=True, help="ISO end date, exclusive")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    parser.add_argument("--refit-every-days", type=int, default=365)
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--hedge-ratio-method", default="kalman", choices=["kalman", "static"])
    parser.add_argument("--zscore-window", type=int, default=20)
    parser.add_argument("--entry-threshold", type=float, default=2.0)
    parser.add_argument("--exit-threshold", type=float, default=0.5)
    parser.add_argument("--stop-loss-threshold", type=float, default=3.5)
    parser.add_argument("--rolling-cointegration-window", type=int, default=90)
    parser.add_argument("--rolling-cointegration-step", type=int, default=5)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--starting-cash", type=float, default=1_000_000.0)
    parser.add_argument("--max-notional-per-pair-fraction", type=float, default=0.5)
    parser.add_argument("--max-gross-exposure-fraction", type=float, default=1.0)
    parser.add_argument("--kill-switch-drawdown", type=float, default=0.15)
    parser.add_argument(
        "--bootstrap", action="store_true", help="Add block-bootstrap CIs on the headline metrics"
    )
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    result = run_backtest(
        start=args.start,
        end=args.end,
        price_source=args.source,
        refit_every_days=args.refit_every_days,
        max_pairs=args.max_pairs,
        hedge_ratio_method=args.hedge_ratio_method,
        zscore_window=args.zscore_window,
        entry_threshold=args.entry_threshold,
        exit_threshold=args.exit_threshold,
        stop_loss_threshold=args.stop_loss_threshold,
        rolling_cointegration_window=args.rolling_cointegration_window,
        rolling_cointegration_step=args.rolling_cointegration_step,
        fdr_alpha=args.fdr_alpha,
        cost_bps=args.cost_bps,
        starting_cash=args.starting_cash,
        max_notional_per_pair_fraction=args.max_notional_per_pair_fraction,
        max_gross_exposure_fraction=args.max_gross_exposure_fraction,
        kill_switch_drawdown=args.kill_switch_drawdown,
    )

    print(f"Backtest {args.start} -> {args.end}, refit every {args.refit_every_days}d\n")

    trades = result["trades"]
    if trades.empty:
        print("No trades triggered over this window.")
    else:
        print(trades.to_string(index=False))
        print()

    for key, value in result["metrics"].items():
        print(f"  {key}: {value:.4f}" if value == value else f"  {key}: nan")
    print(f"  kill_switch_triggered: {result['kill_switch_triggered']}")

    if args.bootstrap:
        returns = result["equity_curve"].pct_change()
        try:
            bootstrapped = bootstrap_backtest_metrics(
                returns,
                block_length=args.block_length,
                n_resamples=args.n_resamples,
                confidence_level=args.confidence_level,
            )
        except ValueError as e:
            print(f"\nBootstrap skipped: {e}")
            return

        pct = int(args.confidence_level * 100)
        print(f"\nBlock bootstrap {pct}% CIs (block_length={args.block_length}, "
              f"n_resamples={args.n_resamples}):")
        for name, r in bootstrapped.items():
            print(f"  {name}: {r.point_estimate:.4f}  [{r.ci_low:.4f}, {r.ci_high:.4f}]")


if __name__ == "__main__":
    main()
