"""CLI entry point: Step 9's honest comparisons, with block bootstrap CIs.

Usage:
    python -m backtest.run_comparison --start 2018-01-01 --end 2025-01-01

Runs a baseline walk-forward backtest (Kalman hedge ratio, proactive
structural-break monitoring, hand-picked sectors) and three variants, each
changing exactly one dimension from the baseline:

    - static hedge ratio instead of Kalman
    - reactive-only (stop-loss only, no rolling re-test/CUSUM) instead of
      proactive structural-break monitoring
    - ML-discovered clusters (pairs.clustering.cluster_tickers, computed
      once from data up to --start, so no lookahead) instead of hand-picked
      sectors

Each run's Sharpe/CAGR/max-drawdown/win-rate gets a block bootstrap CI
(backtest/bootstrap.py). Overlapping CIs between a variant and the baseline
are read as "no meaningful difference"; this is a simpler read than a
formal paired-difference test (each CI is computed independently, not as a
joint hypothesis test on the difference), a disclosed simplification, not
a rigorous significance test.

Pairs vs. baskets is deliberately not run here: baskets aren't wired into
the event-driven backtest (see backtest/README.md), so fabricating a
backtest number for that comparison would misrepresent what's actually been
tested. The best available evidence is pairs/johansen.py's own selection-
stage finding (a handful of significant baskets out of thousands tested,
comparably rare to pairs), printed below instead of a backtest result that
doesn't exist.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from backtest.bootstrap import bootstrap_backtest_metrics
from backtest.engine import DEFAULT_HISTORY_BUFFER_DAYS, run_backtest
from config.universe import UNIVERSE
from data.prices import load_prices, to_price_panel
from pairs.clustering import cluster_tickers


def _run_and_bootstrap(
    label: str, block_length: int, n_resamples: int, seed: int | None, **kwargs
) -> dict:
    print(f"Running: {label}...")
    result = run_backtest(**kwargs)
    returns = result["equity_curve"].pct_change()
    try:
        ci = bootstrap_backtest_metrics(
            returns, block_length=block_length, n_resamples=n_resamples, seed=seed
        )
    except ValueError:
        ci = None
    return {"label": label, "result": result, "ci": ci}


def _print_comparison(baseline: dict, variant: dict) -> None:
    print(f"\n--- {variant['label']} vs. baseline ---")
    for name in ("cagr", "sharpe_ratio", "max_drawdown", "win_rate"):
        base_metric = baseline["result"]["metrics"][name]
        variant_metric = variant["result"]["metrics"][name]
        print(f"  {name}: baseline={base_metric:.4f}  variant={variant_metric:.4f}")
        if baseline["ci"] and variant["ci"]:
            b, v = baseline["ci"][name], variant["ci"][name]
            print(f"    baseline 95% CI [{b.ci_low:.4f}, {b.ci_high:.4f}]")
            print(f"    variant  95% CI [{v.ci_low:.4f}, {v.ci_high:.4f}]")
            overlap = max(b.ci_low, v.ci_low) <= min(b.ci_high, v.ci_high)
            verdict = "no meaningful difference" if overlap else "CIs distinguishable"
            print(f"    CIs overlap: {overlap} ({verdict})")
    print(f"  n_trades: baseline={baseline['result']['metrics']['n_trades']} "
          f"variant={variant['result']['metrics']['n_trades']}")


def _ml_clusters_before(start: str, source: str, n_clusters: int) -> dict[str, list[str]]:
    """ML-discovered clusters using only data strictly before `start`."""
    history_start = (pd.Timestamp(start) - pd.Timedelta(days=DEFAULT_HISTORY_BUFFER_DAYS)).strftime(
        "%Y-%m-%d"
    )
    long_prices = load_prices(UNIVERSE, history_start, start, source=source)
    panel = to_price_panel(long_prices)
    return cluster_tickers(panel, n_clusters=n_clusters)


def _serialize_run(run: dict) -> dict:
    metrics = run["result"]["metrics"]
    ci = run["ci"]
    return {
        "label": run["label"],
        "metrics": metrics,
        "ci": (
            {
                name: {"ci_low": r.ci_low, "ci_high": r.ci_high, "point_estimate": r.point_estimate}
                for name, r in ci.items()
            }
            if ci
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument(
        "--seed", type=int, default=None, help="Bootstrap seed, for reproducibility"
    )
    parser.add_argument(
        "--output-json", default=None, help="Path to save all results + CIs as JSON"
    )
    args = parser.parse_args()

    common = dict(start=args.start, end=args.end, price_source=args.source)

    baseline = _run_and_bootstrap(
        "Baseline (Kalman, proactive monitoring, hand-picked sectors)",
        args.block_length,
        args.n_resamples,
        args.seed,
        **common,
    )
    static_variant = _run_and_bootstrap(
        "Static hedge ratio",
        args.block_length,
        args.n_resamples,
        args.seed,
        **common,
        hedge_ratio_method="static",
    )
    reactive_variant = _run_and_bootstrap(
        "Reactive-only (no structural-break monitoring)",
        args.block_length,
        args.n_resamples,
        args.seed,
        **common,
        use_monitor=False,
    )

    ml_clusters = _ml_clusters_before(args.start, args.source, args.n_clusters)
    ml_variant = _run_and_bootstrap(
        "ML-discovered clusters instead of hand-picked sectors",
        args.block_length,
        args.n_resamples,
        args.seed,
        **common,
        sector_tickers=ml_clusters,
    )

    if args.output_json:
        payload = {
            "start": args.start,
            "end": args.end,
            "block_length": args.block_length,
            "n_resamples": args.n_resamples,
            "seed": args.seed,
            "baseline": _serialize_run(baseline),
            "static_hedge_ratio": _serialize_run(static_variant),
            "reactive_only": _serialize_run(reactive_variant),
            "ml_clusters": _serialize_run(ml_variant),
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"Saved comparison results to {args.output_json}\n")

    print(f"\n=== Baseline ===\nn_trades={baseline['result']['metrics']['n_trades']}")
    for name in ("cagr", "sharpe_ratio", "max_drawdown", "win_rate"):
        point = baseline["result"]["metrics"][name]
        if baseline["ci"]:
            ci = baseline["ci"][name]
            print(f"  {name}: {point:.4f}  95% CI [{ci.ci_low:.4f}, {ci.ci_high:.4f}]")
        else:
            print(f"  {name}: {point:.4f}")

    _print_comparison(baseline, static_variant)
    _print_comparison(baseline, reactive_variant)
    _print_comparison(baseline, ml_variant)

    print(
        "\n--- Pairs vs. baskets ---\n"
        "Not run: baskets aren't wired into the event-driven backtest (see\n"
        "backtest/README.md). Best available evidence is pairs/johansen.py's\n"
        "own selection-stage result: a live run over the full universe found\n"
        "0/1252 3-leg and 4/2930 4-leg baskets significant after FDR correction,\n"
        "comparably rare to the 1/369 pairs that survived the same discipline.\n"
        "A backtest comparison would face at least as severe a small-sample\n"
        "problem as the pairs backtest above, likely worse."
    )


if __name__ == "__main__":
    main()
