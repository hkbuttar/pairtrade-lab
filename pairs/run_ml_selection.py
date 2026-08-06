"""CLI entry point: ML-assisted candidate discovery, and an honest comparison
against the hand-picked sector groupings in config/universe.py.

Usage:
    python -m pairs.run_ml_selection --start 2018-01-01 --end 2025-01-01

Clusters tickers by residual (market-factor-removed) return correlation
(pairs.clustering.cluster_tickers), runs the same Engle-Granger/Johansen/FDR
pipeline used for hand-picked sectors on those ML-discovered clusters, and
reports whether the ML-discovered candidates found anything the hand-picked
sector groupings didn't, or vice versa. Reported plainly either way: this
script does not pick a "winner" narrative, since the honest answer here may
well be "no meaningful difference" or "the ML clusters mostly just
reproduced the sectors."
"""

from __future__ import annotations

import argparse

import pandas as pd

from config.universe import SECTOR_TICKERS, UNIVERSE
from data.prices import load_prices, to_price_panel
from pairs.clustering import cluster_tickers, market_factor_variance_ratio
from pairs.cointegration import test_sector_pairs
from pairs.johansen import test_sector_baskets


def _pair_key(row: pd.Series) -> frozenset:
    return frozenset({row["ticker_y"], row["ticker_x"]})


def _basket_key(row: pd.Series) -> frozenset:
    return frozenset(row["tickers"])


def run(
    start: str,
    end: str,
    source: str = "yfinance",
    n_clusters: int = 12,
    n_market_factors: int = 1,
    basket_sizes: tuple[int, ...] = (3, 4),
    fdr_alpha: float = 0.05,
) -> dict:
    long_prices = load_prices(UNIVERSE, start, end, source=source)
    panel = to_price_panel(long_prices)

    variance_ratios = market_factor_variance_ratio(panel, n_components=5)
    clusters = cluster_tickers(panel, n_clusters=n_clusters, n_market_factors=n_market_factors)

    ml_pairs = test_sector_pairs(panel, clusters, fdr_alpha=fdr_alpha)
    handpicked_pairs = test_sector_pairs(panel, SECTOR_TICKERS, fdr_alpha=fdr_alpha)
    ml_baskets = test_sector_baskets(
        panel, clusters, basket_sizes=basket_sizes, fdr_alpha=fdr_alpha
    )
    handpicked_baskets = test_sector_baskets(
        panel, SECTOR_TICKERS, basket_sizes=basket_sizes, fdr_alpha=fdr_alpha
    )

    return {
        "variance_ratios": variance_ratios,
        "clusters": clusters,
        "ml_pairs": ml_pairs,
        "handpicked_pairs": handpicked_pairs,
        "ml_baskets": ml_baskets,
        "handpicked_baskets": handpicked_baskets,
    }


def _print_comparison(
    label: str, ml_table: pd.DataFrame, handpicked_table: pd.DataFrame, key_fn
) -> None:
    ml_n, ml_sig = len(ml_table), int(ml_table["cointegrated"].sum()) if len(ml_table) else 0
    hp_n, hp_sig = (
        len(handpicked_table),
        int(handpicked_table["cointegrated"].sum()) if len(handpicked_table) else 0,
    )

    ml_winners = {key_fn(row) for _, row in ml_table[ml_table["cointegrated"]].iterrows()}
    hp_sig_table = handpicked_table[handpicked_table["cointegrated"]]
    hp_winners = {key_fn(row) for _, row in hp_sig_table.iterrows()}
    overlap = ml_winners & hp_winners

    print(f"\n--- {label} ---")
    print(f"ML-discovered clusters:  {ml_sig} / {ml_n} significant after FDR correction")
    print(f"Hand-picked sectors:     {hp_sig} / {hp_n} significant after FDR correction")
    print(f"Found by both:           {len(overlap)}")
    print(f"ML-only:                 {len(ml_winners - hp_winners)}")
    print(f"Hand-picked-only:        {len(hp_winners - ml_winners)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="ISO start date, inclusive")
    parser.add_argument("--end", required=True, help="ISO end date, exclusive")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    parser.add_argument("--n-clusters", type=int, default=12)
    parser.add_argument("--n-market-factors", type=int, default=1)
    parser.add_argument("--basket-sizes", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        args.start,
        args.end,
        source=args.source,
        n_clusters=args.n_clusters,
        n_market_factors=args.n_market_factors,
        basket_sizes=tuple(args.basket_sizes),
        fdr_alpha=args.fdr_alpha,
    )

    ratios = result["variance_ratios"]
    print("Variance explained by top 5 principal components of standardized returns:")
    print("  " + ", ".join(f"PC{i + 1}={r:.1%}" for i, r in enumerate(ratios)))
    print(f"  (top {args.n_market_factors} regressed out before clustering)")

    print(f"\n{len(result['clusters'])} clusters found (from {len(UNIVERSE)} universe tickers):")
    for name, tickers in sorted(result["clusters"].items(), key=lambda kv: -len(kv[1])):
        print(f"  {name} ({len(tickers)}): {', '.join(tickers)}")

    _print_comparison("Pairs", result["ml_pairs"], result["handpicked_pairs"], _pair_key)
    _print_comparison("Baskets", result["ml_baskets"], result["handpicked_baskets"], _basket_key)


if __name__ == "__main__":
    main()
