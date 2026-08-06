"""CLI entry point: run Engle-Granger pair selection across the full universe.

Usage:
    python -m pairs.run_selection --start 2018-01-01 --end 2025-01-01

By default prints every tested pair (winners and rejected pairs alike),
ranked by FDR-corrected p-value, so the search space stays transparent
rather than only showing survivors. Pass --only-significant to print just
the pairs that passed correction (the full table, both winners and rejects,
is still available via --output-csv regardless).
"""

from __future__ import annotations

import argparse

import pandas as pd

from config.universe import SECTOR_TICKERS, UNIVERSE
from data.prices import load_prices, to_price_panel
from pairs.cli_output import add_output_args, report
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
    add_output_args(parser)
    args = parser.parse_args()

    table = run(args.start, args.end, source=args.source, fdr_alpha=args.fdr_alpha)
    report(table, "pairs", args, args.fdr_alpha)


if __name__ == "__main__":
    main()
