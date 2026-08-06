import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    cagr,
    compute_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    trade_stats,
    win_rate,
)


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_cagr_known_growth_rate():
    # Doubles over exactly 252 trading days (1 year) -> CAGR should be ~100%.
    equity = pd.Series([100.0, 200.0], index=_dates(2))
    # Force the "years" calc to treat this as ~1 year by using 252 rows.
    equity = pd.Series(np.linspace(100.0, 200.0, 252), index=_dates(252))

    result = cagr(equity)

    assert result == pytest.approx(1.0, abs=0.05)


def test_cagr_nan_on_single_point():
    assert np.isnan(cagr(pd.Series([100.0], index=_dates(1))))


def test_sharpe_ratio_nan_on_zero_volatility():
    returns = pd.Series([0.0] * 10, index=_dates(10))

    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_positive_for_consistently_positive_returns():
    returns = pd.Series([0.001] * 100, index=_dates(100))
    # Add tiny noise so std isn't exactly zero.
    returns = returns + np.random.default_rng(0).normal(0, 1e-6, 100)

    assert sharpe_ratio(returns) > 0


def test_sortino_nan_when_no_downside():
    returns = pd.Series([0.01, 0.02, 0.01], index=_dates(3))

    assert np.isnan(sortino_ratio(returns))


def test_max_drawdown_known_path():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0], index=_dates(4))

    result = max_drawdown(equity)

    assert result == pytest.approx((120.0 - 90.0) / 120.0)


def test_max_drawdown_empty_series_is_nan():
    assert np.isnan(max_drawdown(pd.Series(dtype=float)))


def test_win_rate_known_fraction():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0], index=_dates(5))

    assert win_rate(returns) == pytest.approx(2 / 5)


def test_trade_stats_empty_table():
    stats = trade_stats(pd.DataFrame(columns=["side", "exit_reason", "bars_held"]))

    assert stats["n_trades"] == 0
    assert np.isnan(stats["avg_bars_held"])


def test_trade_stats_counts_and_stop_loss_rate():
    trades = pd.DataFrame(
        {
            "side": ["long", "short", "long"],
            "exit_reason": ["reversion", "stop_loss", "open"],
            "bars_held": [2, 4, 6],
        }
    )

    stats = trade_stats(trades)

    assert stats["n_trades"] == 3
    assert stats["avg_bars_held"] == pytest.approx(4.0)
    # Only 2 closed trades (reversion, stop_loss); 1 of those is stop_loss.
    assert stats["stop_loss_rate"] == pytest.approx(0.5)


def test_compute_metrics_returns_all_expected_keys():
    equity = pd.Series(np.linspace(100.0, 105.0, 50), index=_dates(50))
    trades = pd.DataFrame(columns=["side", "exit_reason", "bars_held"])

    metrics = compute_metrics(equity, trades)

    assert set(metrics) == {
        "cagr",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "n_trades",
        "avg_bars_held",
        "stop_loss_rate",
    }
