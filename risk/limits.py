"""Hard portfolio-level exposure limits for pairs trading.

A backstop applied at position-sizing time, downstream from whatever
scheme decided a pair's "raw" desired notional (backtest.engine's fixed
notional-per-pair scheme, currently). Same "clip after sizing" discipline as
alpha-signal-lab's risk/limits.py: per-pair first (bounds any single
position), then whatever gross exposure budget remains once existing
positions are counted (bounds total exposure across all simultaneously
active pairs) - each step operating on the previous one's output.

Existing open positions are never resized to make room for a new one: the
limit constrains how big a *new* entry is allowed to be, not a retroactive
shrink of positions already held, which would just be an extra unplanned
trade (and its own transaction cost) triggered by an unrelated signal.
"""

from __future__ import annotations

MAX_NOTIONAL_PER_PAIR_FRACTION = 0.5
MAX_GROSS_EXPOSURE_FRACTION = 1.0


def clip_new_pair_notional(
    raw_notional: float,
    existing_notional: dict,
    equity: float,
    max_notional_per_pair_fraction: float = MAX_NOTIONAL_PER_PAIR_FRACTION,
    max_gross_exposure_fraction: float = MAX_GROSS_EXPOSURE_FRACTION,
) -> float:
    """Clip a new pair entry's notional to the per-pair cap, then to
    whatever gross exposure budget remains after already-open positions.

    Args:
        raw_notional: The signed notional a sizing scheme would otherwise
            allocate to this new position (positive or negative; sign is
            preserved, only magnitude is clipped).
        existing_notional: {pair: signed notional} for every currently open
            position; only the magnitudes matter here, to compute how much
            of the gross budget is already committed.
        equity: Current portfolio equity, the base the fractional limits
            are computed against (not a fixed starting-cash figure, so the
            limit tightens automatically as equity drops).
        max_notional_per_pair_fraction: Cap on any single pair's notional,
            as a fraction of current equity.
        max_gross_exposure_fraction: Cap on total gross notional (sum of
            |notional| across all open pairs plus this new one), as a
            fraction of current equity.

    Returns:
        The clipped notional (same sign as raw_notional, magnitude reduced
        as needed; 0.0 if equity <= 0, raw_notional is 0, or no gross
        budget remains).
    """
    if equity <= 0 or raw_notional == 0:
        return 0.0

    sign = 1.0 if raw_notional > 0 else -1.0
    magnitude = min(abs(raw_notional), equity * max_notional_per_pair_fraction)

    existing_gross = sum(abs(v) for v in existing_notional.values())
    remaining_budget = max(0.0, equity * max_gross_exposure_fraction - existing_gross)
    magnitude = min(magnitude, remaining_budget)

    return sign * magnitude
