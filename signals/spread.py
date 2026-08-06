"""Spread construction and z-score entry/exit/stop-loss signal generation.

Spread = price_A - hedge_ratio * price_B - intercept for pairs (the same
quantity pairs.cointegration.engle_granger_test's ADF test is run on, so a
pair's spread here is directly the residual whose stationarity was already
verified), or the weighted sum of basket legs using the Johansen
cointegrating vector for baskets. Both hedge_ratio and intercept can be a
scalar (single full-window estimate) or a per-bar pd.Series (from
signals.hedge_ratio's static or Kalman estimators); passing a Series is what
makes the spread and its z-score point-in-time correct rather than using a
single full-sample hedge ratio to compute history that a real strategy could
never have known at the time.

Entry/exit/stop-loss thresholds are the plan's own worked example (2.0 /
0.5 / 3.5), used here as the defaults but always explicit, disclosed
parameters rather than hidden constants.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_ENTRY_THRESHOLD = 2.0
DEFAULT_EXIT_THRESHOLD = 0.5
DEFAULT_STOP_LOSS_THRESHOLD = 3.5


def pair_spread(
    y: pd.Series, x: pd.Series, hedge_ratio: float | pd.Series, intercept: float | pd.Series = 0.0
) -> pd.Series:
    """Spread = y - hedge_ratio * x - intercept, aligned on (y, x)'s common index."""
    aligned_y, aligned_x = y.align(x, join="inner")
    return aligned_y - hedge_ratio * aligned_x - intercept


def basket_spread(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Spread = sum(weight_i * price_i) for a basket, using e.g. the
    normalized cointegrating-vector weights from pairs.johansen.johansen_test.
    """
    legs = list(weights)
    return (prices[legs] * pd.Series(weights)).sum(axis=1)


def rolling_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """Z-score the spread against its own trailing rolling mean/std.

    Args:
        spread: Spread series (see pair_spread / basket_spread).
        window: Rolling window length in bars, an explicit, disclosed
            parameter (not derived from the data). The first ``window - 1``
            bars have no full trailing window and are NaN.
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    return (spread - rolling_mean) / rolling_std


def generate_signals(
    z_score: pd.Series,
    entry_threshold: float = DEFAULT_ENTRY_THRESHOLD,
    exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
    stop_loss_threshold: float = DEFAULT_STOP_LOSS_THRESHOLD,
    monitor_status: pd.Series | None = None,
) -> pd.Series:
    """Stateful entry/exit/stop-loss position series from a z-score path.

    Rules, applied bar by bar (position only changes on the transitions
    listed; while holding a position, new entry signals in the same
    direction or the opposite direction are both ignored until an exit or
    stop-loss fires, i.e. positions never flip directly from long to short):

    - Flat and z >= entry_threshold: enter short the spread (position -1).
    - Flat and z <= -entry_threshold: enter long the spread (position +1).
    - Holding and |z| <= exit_threshold: exit to flat (reversion).
    - Holding and |z| >= stop_loss_threshold: exit to flat (stop-loss; the
      divergence kept going instead of reverting).
    - Otherwise: hold the current position (including staying flat).

    Args:
        z_score: Output of rolling_zscore. Leading NaNs (no full window yet)
            are treated as flat/no-signal.
        entry_threshold: |z| to open a position.
        exit_threshold: |z| to close a position on reversion.
        stop_loss_threshold: |z| to force-close a position on continued
            divergence. Must be >= entry_threshold for the state machine
            above to make sense.
        monitor_status: Optional per-bar "ACTIVE"/"HALTED" series (see
            monitor.structural_break.monitor_pair_status). A HALTED bar
            forces the position flat immediately, the same as a stop-loss,
            and requires a fresh flat -> threshold-crossing entry signal
            once ACTIVE resumes, rather than resuming whatever the
            z-score-only state machine would otherwise have been doing.
            Masking the output of an unmonitored run after the fact would
            get this wrong: the state machine itself needs to know about
            the halt so a stale "still notionally in a trade" state
            doesn't reappear the instant monitoring clears.

    Returns:
        Position series aligned to z_score's index, values in {-1, 0, 1}.
    """
    positions = pd.Series(0, index=z_score.index, dtype=int)
    position = 0
    for i, z in enumerate(z_score.to_numpy()):
        active = monitor_status is None or monitor_status.iloc[i] == "ACTIVE"
        if not active or pd.isna(z):
            position = 0
        elif position == 0:
            if z >= entry_threshold:
                position = -1
            elif z <= -entry_threshold:
                position = 1
        else:
            abs_z = abs(z)
            if abs_z <= exit_threshold or abs_z >= stop_loss_threshold:
                position = 0
        positions.iloc[i] = position

    return positions


def summarize_trades(
    positions: pd.Series,
    z_score: pd.Series,
    exit_threshold: float = DEFAULT_EXIT_THRESHOLD,
) -> pd.DataFrame:
    """Descriptive summary of each trade implied by a position series: side,
    entry/exit dates, holding period, and why it closed.

    Deliberately descriptive, not a P&L calculation: entries/exits here are
    signal transitions, not fills, and no transaction costs or slippage are
    modeled. Real performance numbers need the event-driven simulator in
    backtest/, not yet built.

    Args:
        positions: Output of generate_signals.
        z_score: The same z-score series generate_signals was run on, used
            to classify why each trade closed.
        exit_threshold: Must match what generate_signals was called with.
            Any exit with |z| above this is classified "stop_loss" (the
            only other way generate_signals closes a position), so
            stop_loss_threshold itself isn't needed here.

    Returns:
        One row per closed or still-open trade, columns [side, entry_date,
        exit_date, bars_held, exit_reason], exit_reason one of "reversion",
        "stop_loss", or "open" (still held at the end of the series).
    """
    trades = []
    entry_idx: int | None = None
    entry_side = 0

    values = positions.to_numpy()
    for i in range(len(values)):
        prev = values[i - 1] if i > 0 else 0
        curr = values[i]
        if prev == 0 and curr != 0:
            entry_idx, entry_side = i, curr
        elif prev != 0 and curr == 0:
            abs_z = abs(z_score.iloc[i])
            reason = "reversion" if abs_z <= exit_threshold else "stop_loss"
            trades.append(
                {
                    "side": "long" if entry_side == 1 else "short",
                    "entry_date": positions.index[entry_idx],
                    "exit_date": positions.index[i],
                    "bars_held": i - entry_idx,
                    "exit_reason": reason,
                }
            )
            entry_idx = None

    if entry_idx is not None:
        trades.append(
            {
                "side": "long" if entry_side == 1 else "short",
                "entry_date": positions.index[entry_idx],
                "exit_date": positions.index[-1],
                "bars_held": len(values) - 1 - entry_idx,
                "exit_reason": "open",
            }
        )

    return pd.DataFrame(
        trades, columns=["side", "entry_date", "exit_date", "bars_held", "exit_reason"]
    )
