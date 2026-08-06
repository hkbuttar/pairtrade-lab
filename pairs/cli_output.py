"""Shared CLI plumbing for pairs.run_selection and pairs.run_basket_selection:
printing a results table to the terminal and optionally saving the full,
unfiltered table to CSV. Split out since both scripts need the same
"don't dump thousands of rows to stdout by default" behavior.
"""

from __future__ import annotations

import argparse

import pandas as pd


def add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--only-significant",
        action="store_true",
        help="Only print rows that passed FDR correction (the full table is still "
        "used for the summary count, and for --output-csv if given).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Path to save the full table (winners and rejects) as CSV. Independent "
        "of --only-significant, which only affects what's printed.",
    )


def report(table: pd.DataFrame, label: str, args: argparse.Namespace, fdr_alpha: float) -> None:
    """Print (optionally filtered), optionally save to CSV, and summarize."""
    to_print = table[table["cointegrated"]] if args.only_significant else table

    with pd.option_context("display.max_rows", None, "display.width", 160):
        if to_print.empty and args.only_significant:
            print(f"(no {label} passed FDR correction; rerun without --only-significant)")
        elif to_print.empty:
            print(f"(no {label} tested)")
        else:
            print(to_print.to_string(index=False))

    if args.output_csv:
        table.to_csv(args.output_csv, index=False)
        print(f"\nFull table ({len(table)} {label}) saved to {args.output_csv}")

    n_tested = len(table)
    n_significant = int(table["cointegrated"].sum()) if n_tested else 0
    print(
        f"\n{n_significant} / {n_tested} {label} cointegrated after FDR correction "
        f"(alpha={fdr_alpha})"
    )
