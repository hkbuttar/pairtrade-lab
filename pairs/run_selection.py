"""CLI entry point: run Engle-Granger pair selection across the full universe.

Usage:
    python -m pairs.run_selection --start 2018-01-01 --end 2025-01-01

Prints every tested pair (winners and rejected pairs alike), ranked by
FDR-corrected p-value, plus a summary line of how many pairs were tested
versus how many survived correction, so the search space stays transparent
rather than only showing survivors.
"""

from __future__ import annotations

import argparse

import pandas as pd

from config.universe import SECTOR_TICKERS, UNIVERSE
from data.prices import load_prices, to_price_panel
from pairs.cointegration import test_sector_pairs


def run(start: str, end: str, source: str = "yfinance", fdr_alpha: float = 0.05) -> pd.DataFrame:
    long_prices = load_prices(UNIVERSE, start, end, source=source)
    panel = to_price_panel(long_prices)
    return test_sector_pairs(panel, SECTOR_TICKERS, fdr_alpha=fdr_alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="ISO start date, inclusive")
    parser.add_argument("--end", required=True, help="ISO end date, exclusive")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    args = parser.parse_args()

    table = run(args.start, args.end, source=args.source, fdr_alpha=args.fdr_alpha)

    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(table.to_string(index=False))

    n_tested = len(table)
    n_significant = int(table["cointegrated"].sum()) if n_tested else 0
    print(
        f"\n{n_significant} / {n_tested} pairs cointegrated after FDR correction "
        f"(alpha={args.fdr_alpha})"
    )


if __name__ == "__main__":
    main()
