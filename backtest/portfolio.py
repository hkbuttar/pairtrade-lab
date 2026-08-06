"""Cash and position ledger for the event-driven backtest.

Tracks cash, share positions (across any number of tickers, from any number
of simultaneously active pairs), a daily mark-to-market equity curve, and a
daily position snapshot history. Has no knowledge of pairs, signals, or
sizing; it only records the effect of fills that engine.py hands it. Mirrors
alpha-signal-lab's backtest/portfolio.py, the same generic ledger shape
reused across this portfolio's backtesters.
"""

from __future__ import annotations

import pandas as pd

POSITION_EPSILON = 1e-9


class Portfolio:
    def __init__(self, starting_cash: float = 1_000_000.0) -> None:
        self.cash = starting_cash
        self.positions: dict[str, float] = {}
        self._equity_records: list[dict] = []
        self._position_records: list[dict] = []

    def apply_fill(
        self, ticker: str, shares_delta: float, fill_price: float, cost: float
    ) -> None:
        """Apply a single fill: update cash and the position in ``ticker``."""
        self.cash -= shares_delta * fill_price
        self.cash -= cost
        new_shares = self.positions.get(ticker, 0.0) + shares_delta
        if abs(new_shares) < POSITION_EPSILON:
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = new_shares

    def mark_to_market(self, date: pd.Timestamp, close_prices: dict[str, float]) -> float:
        """Record equity and position snapshot for ``date`` using ``close_prices``.

        Returns:
            Total equity (cash + mark-to-market position value).
        """
        position_value = sum(
            shares * close_prices.get(ticker, 0.0) for ticker, shares in self.positions.items()
        )
        equity = self.cash + position_value
        self._equity_records.append({"date": date, "equity": equity})
        for ticker, shares in self.positions.items():
            self._position_records.append({"date": date, "ticker": ticker, "shares": shares})
        return equity

    def equity_series(self) -> pd.Series:
        df = pd.DataFrame(self._equity_records)
        if df.empty:
            return pd.Series(dtype=float)
        return df.set_index("date")["equity"].sort_index()

    def position_history(self) -> pd.DataFrame:
        return pd.DataFrame(self._position_records, columns=["date", "ticker", "shares"])

    def flatten_orders(self) -> dict[str, float]:
        """Share deltas needed to close every open position."""
        return {ticker: -shares for ticker, shares in self.positions.items()}
