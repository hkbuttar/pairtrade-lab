"""Backtest performance metrics: CAGR, Sharpe, Sortino, drawdown, win rate,
plus pairs-specific trade stats (count, average holding period, stop-loss rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk.kill_switch import running_drawdown

TRADING_DAYS_PER_YEAR = 252
ZERO_VOL_EPSILON = 1e-9


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate over the equity curve's full span."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    years = len(equity) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio of daily returns."""
    excess = returns.dropna() - risk_free / TRADING_DAYS_PER_YEAR
    if excess.empty or excess.std() < ZERO_VOL_EPSILON:
        return float("nan")
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio (downside-deviation-only Sharpe) of daily returns."""
    excess = returns.dropna() - risk_free / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    if downside.empty or downside.std() < ZERO_VOL_EPSILON:
        return float("nan")
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough drawdown over the equity curve."""
    if equity.empty:
        return float("nan")
    return float(running_drawdown(equity).max())


def win_rate(returns: pd.Series) -> float:
    """Fraction of trading days with a positive return."""
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    return float((clean > 0).mean())


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    """Descriptive stats over a combined trades table (concatenated
    signals.spread.summarize_trades output across every pair/refit cycle).
    """
    if trades.empty:
        return {
            "n_trades": 0,
            "avg_bars_held": float("nan"),
            "stop_loss_rate": float("nan"),
        }
    closed = trades[trades["exit_reason"] != "open"]
    stop_loss_rate = (
        float((closed["exit_reason"] == "stop_loss").mean()) if not closed.empty else float("nan")
    )
    return {
        "n_trades": int(len(trades)),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "stop_loss_rate": stop_loss_rate,
    }


def compute_metrics(equity: pd.Series, trades: pd.DataFrame) -> dict[str, float]:
    """Full metrics table for a backtest run."""
    returns = equity.pct_change()
    return {
        "cagr": cagr(equity),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(returns),
        **trade_stats(trades),
    }
