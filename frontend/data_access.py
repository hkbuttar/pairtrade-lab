"""Compute (not query) the data each dashboard view needs, straight from
this project's own library modules (pairs/, signals/, monitor/) rather than
a database. There is no live paper-trading scheduler in this project (see
README Future Work), so "live" here means "recomputed against the latest
cached price data on interaction," not a continuous feed; the dashboard
says so rather than implying more than it delivers.

Every function is wrapped in @pn.cache so repeated widget interactions
(switching the selected pair, revisiting a tab) don't re-run an expensive
selection or rolling-monitor computation that was already done this
session. Cache keys are the plain argument values, so calls with the same
(start, end, source, ...) share a cache entry.
"""

from __future__ import annotations

import pandas as pd
import panel as pn

from config.universe import SECTOR_TICKERS, UNIVERSE
from data.prices import load_prices, to_price_panel
from monitor.structural_break import cusum_detect, monitor_pair_status, rolling_cointegration_pvalue
from pairs.cointegration import test_sector_pairs
from signals.hedge_ratio import kalman_hedge_ratio_series
from signals.spread import generate_signals, pair_spread, rolling_zscore, summarize_trades

DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-01-01"


@pn.cache
def get_price_panel(start: str = DEFAULT_START, end: str = DEFAULT_END, source: str = "yfinance"):
    long_prices = load_prices(UNIVERSE, start, end, source=source)
    return to_price_panel(long_prices)


@pn.cache
def get_significant_pairs(
    start: str = DEFAULT_START, end: str = DEFAULT_END, source: str = "yfinance"
) -> pd.DataFrame:
    """FDR-significant pairs over the full window, ranked by corrected p-value."""
    panel = get_price_panel(start, end, source)
    table = test_sector_pairs(panel, SECTOR_TICKERS)
    return table[table["cointegrated"]].sort_values("p_value_fdr").reset_index(drop=True)


@pn.cache
def get_pair_monitor_data(
    ticker_y: str,
    ticker_x: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    source: str = "yfinance",
) -> dict:
    """Everything the spread/z-score, live-status, and alert-feed views need
    for one pair: hedge ratio, spread, z-score, rolling monitor status, the
    resulting position signal, and a trade summary.
    """
    panel = get_price_panel(start, end, source)
    y, x = panel[ticker_y], panel[ticker_x]

    estimate = kalman_hedge_ratio_series(y, x)
    spread = pair_spread(y, x, estimate["hedge_ratio"], estimate["intercept"])
    zscore = rolling_zscore(spread)
    rolling_pvalue = rolling_cointegration_pvalue(y, x, step=5)
    cusum = cusum_detect(zscore)
    status = monitor_pair_status(rolling_pvalue, cusum["break_flagged"])
    positions = generate_signals(zscore, monitor_status=status)
    trades = summarize_trades(positions, zscore)

    return {
        "y": y,
        "x": x,
        "spread": spread,
        "zscore": zscore,
        "rolling_pvalue": rolling_pvalue,
        "status": status,
        "positions": positions,
        "trades": trades,
    }


def halt_events(status: pd.Series) -> pd.DataFrame:
    """ACTIVE->HALTED transition dates and, where the pair has since
    reinstated, how many calendar days the halt lasted.
    """
    events = []
    halt_start = None
    prev = "ACTIVE"
    for date, current in status.items():
        if prev == "ACTIVE" and current == "HALTED":
            halt_start = date
        elif prev == "HALTED" and current == "ACTIVE":
            events.append(
                {
                    "halted_at": halt_start,
                    "resumed_at": date,
                    "days_halted": (date - halt_start).days,
                }
            )
            halt_start = None
        prev = current

    if halt_start is not None:
        events.append({"halted_at": halt_start, "resumed_at": None, "days_halted": None})

    return pd.DataFrame(events, columns=["halted_at", "resumed_at", "days_halted"])


def days_since_last_break(status: pd.Series) -> int | None:
    """Calendar days since the most recent HALT began, or None if the pair
    has never been halted over the loaded window.
    """
    events = halt_events(status)
    if events.empty:
        return None
    last_halt_date = events["halted_at"].iloc[-1]
    return (status.index[-1] - last_halt_date).days
