"""Single source of truth for the tradable equity universe and sector map.

Universe is ~70 liquid US large/mid-caps across six sectors. ``Technology``,
``Healthcare``, ``Financials``, and ``Energy`` are the same constituent lists as
alpha-signal-lab's ``config/universe.py`` (deliberate continuity, not a
coincidence: it lets pairs discovered here be cross-referenced against that
project's factor scores). ``Utilities`` and ``Airlines`` are added on top,
since same-sector cointegration among regulated utilities and among network
airlines is one of the more textbook, economically-motivated setups for pairs
trading and neither sector is represented in alpha-signal-lab's universe.

This is the current constituent list only; no delisted, acquired, or merged
names are included, so survivorship bias is present (see README Limitations).
"""

from __future__ import annotations

SECTOR_TICKERS: dict[str, list[str]] = {
    "Technology": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AVGO",
        "ORCL",
        "CRM",
        "ADBE",
        "CSCO",
        "AMD",
        "INTC",
        "TXN",
    ],
    "Healthcare": [
        "JNJ",
        "UNH",
        "LLY",
        "PFE",
        "MRK",
        "ABBV",
        "TMO",
        "ABT",
        "DHR",
        "BMY",
        "AMGN",
        "GILD",
    ],
    "Financials": [
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "C",
        "SCHW",
        "AXP",
        "BLK",
        "SPGI",
        "PNC",
        "USB",
        "TFC",
    ],
    "Energy": [
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "MPC",
        "PSX",
        "VLO",
        "OXY",
        "WMB",
        "KMI",
        "DVN",
    ],
    "Utilities": [
        "NEE",
        "DUK",
        "SO",
        "D",
        "AEP",
        "EXC",
        "XEL",
        "ED",
        "WEC",
        "ES",
        "PEG",
        "EIX",
    ],
    "Airlines": [
        "DAL",
        "UAL",
        "AAL",
        "LUV",
        "ALK",
        "JBLU",
    ],
}

TICKER_SECTOR: dict[str, str] = {
    ticker: sector for sector, tickers in SECTOR_TICKERS.items() for ticker in tickers
}

UNIVERSE: list[str] = sorted(TICKER_SECTOR)
