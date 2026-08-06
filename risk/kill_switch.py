"""Drawdown kill-switch.

Consistent with alpha-signal-lab's design
(~/alpha-signal-lab/risk/kill_switch.py, also mirrored in bookmaker and
execedge): a stateful, streaming monitor of drawdown from the running
equity peak. Once triggered it stays triggered, it does not auto-resume
just because equity ticks back up, and only ``reset()``, called
deliberately, re-arms it. Treated the same way across all four projects in
this portfolio: the single most important safety control in the system,
and one that should never silently forgive itself. Worth naming
explicitly: this is a portfolio-wide design principle, not a coincidence
that four independent trading systems in this portfolio all converged on
the same sticky-trigger, manual-reset-only kill-switch shape.

This module has no knowledge of order execution or position sizing; it
only tracks equity and reports a boolean. backtest/engine.py owns the
responsibility of actually flattening positions when ``check`` returns True.

Uses fractional drawdown (drawdown / peak), the same choice as
alpha-signal-lab and meaningful here for the same reason: this portfolio
starts from a real capital base (``starting_cash``), so a fraction of that
base is a stable, well-defined quantity throughout (unlike bookmaker's
market-making portfolio, which starts flat at exactly $0 and uses an
absolute-dollar drawdown instead for that reason).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_MAX_DRAWDOWN = 0.15


class KillSwitch:
    """Stateful, streaming drawdown monitor for use in a day-by-day loop."""

    def __init__(self, max_drawdown: float = DEFAULT_MAX_DRAWDOWN) -> None:
        self.max_drawdown = max_drawdown
        self._peak_equity: float | None = None
        self.triggered = False

    def check(self, equity: float) -> bool:
        """Update with the latest equity value and report kill-switch state.

        Args:
            equity: Current total portfolio equity (positions + cash).

        Returns:
            True if drawdown from the running peak is at or beyond
            ``max_drawdown``. Once triggered, stays triggered until ``reset``
            is called explicitly (a kill-switch should not silently
            re-arm itself just because equity ticked back up).
        """
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = 0.0 if not self._peak_equity else 1 - equity / self._peak_equity
        if drawdown >= self.max_drawdown:
            self.triggered = True
        return self.triggered

    def reset(self) -> None:
        """Manually re-arm the kill-switch after review."""
        self._peak_equity = None
        self.triggered = False


def running_drawdown(equity_curve: pd.Series) -> pd.Series:
    """Vectorized drawdown-from-peak series, for reporting/backtesting analysis."""
    peak = equity_curve.cummax()
    return 1 - equity_curve / peak
