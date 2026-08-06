"""Fill simulation: next-day-open execution with a flat transaction cost in bps.

alpha-signal-lab's fill model scales slippage by participation rate against
average daily volume, appropriate for a cross-sectional book trading against
liquidity constraints. Pairs trading here doesn't need that: positions are
small, concentrated in one or two pairs, and the point of this backtest is
to test the pairs-trading logic (selection, hedge ratios, signals,
monitoring), not execution microstructure. A flat cost in bps per trade,
applied to both legs on every fill, is the disclosed simplification here,
per the project plan's own wording ("explicit transaction cost modeling, bps
per trade, disclosed assumption"). No price impact is modeled: the fill
price is exactly next-bar's open, and the cost is charged as a separate fee
rather than baked into the price, which also makes the fee amount visible
directly in the ledger for reporting.

Orders decided using information through day t are never filled at day t's
close; they're filled at day t+1's open, matching the same no-lookahead
discipline as alpha-signal-lab's backtest/fills.py.
"""

from __future__ import annotations

DEFAULT_COST_BPS = 5.0


def compute_fill(
    shares_delta: float, next_open: float, cost_bps: float = DEFAULT_COST_BPS
) -> tuple[float, float]:
    """Simulate a fill for an order, given the next bar's open price.

    Args:
        shares_delta: Signed order size (positive = buy, negative = sell).
        next_open: Next trading day's open price.
        cost_bps: Flat transaction cost in bps of trade notional.

    Returns:
        (fill_price, cost). fill_price is exactly next_open (no price
        impact modeled); cost is a non-negative dollar fee.
    """
    if shares_delta == 0:
        return next_open, 0.0
    cost = abs(shares_delta * next_open) * cost_bps / 10_000
    return next_open, cost
