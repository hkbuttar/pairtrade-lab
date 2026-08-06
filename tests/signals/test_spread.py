import numpy as np
import pandas as pd

from signals.spread import (
    basket_spread,
    generate_signals,
    pair_spread,
    rolling_zscore,
    summarize_trades,
)


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_pair_spread_with_scalar_hedge_ratio_and_intercept():
    dates = _dates(3)
    y = pd.Series([10.0, 11.0, 12.0], index=dates)
    x = pd.Series([5.0, 5.0, 5.0], index=dates)

    spread = pair_spread(y, x, hedge_ratio=2.0, intercept=1.0)

    assert list(spread) == [-1.0, 0.0, 1.0]


def test_pair_spread_with_time_varying_hedge_ratio():
    dates = _dates(3)
    y = pd.Series([10.0, 11.0, 12.0], index=dates)
    x = pd.Series([5.0, 5.0, 5.0], index=dates)
    hedge_ratio = pd.Series([1.0, 2.0, 3.0], index=dates)

    spread = pair_spread(y, x, hedge_ratio=hedge_ratio, intercept=0.0)

    assert list(spread) == [5.0, 1.0, -3.0]


def test_basket_spread_weighted_sum():
    dates = _dates(3)
    prices = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [7.0, 8.0, 9.0]}, index=dates
    )
    weights = {"A": 1.0, "B": -0.5, "C": 0.0}

    spread = basket_spread(prices, weights)

    assert list(spread) == [-1.0, -0.5, 0.0]


def test_rolling_zscore_matches_manual_computation():
    rng = np.random.default_rng(0)
    spread = pd.Series(rng.normal(size=50), index=_dates(50))

    result = rolling_zscore(spread, window=10)
    expected = (spread - spread.rolling(10).mean()) / spread.rolling(10).std()

    pd.testing.assert_series_equal(result, expected)
    assert result.iloc[:9].isna().all()


def test_generate_signals_full_lifecycle():
    z = pd.Series(
        [np.nan, np.nan, 2.1, 1.0, 0.4, -2.2, -3.6, -1.0, 2.6, 2.0, 0.3],
        index=_dates(11),
    )

    positions = generate_signals(
        z, entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=3.5
    )

    assert list(positions) == [0, 0, -1, -1, 0, 1, 0, 0, -1, -1, 0]


def test_generate_signals_does_not_flip_directly_between_long_and_short():
    # Holding short (-1); z swings to a level that would be a long-entry
    # signal if flat, but since it's between exit and stop-loss, the
    # existing short position should just be held, not flipped to long.
    z = pd.Series([2.5, -2.5, -2.5], index=_dates(3))

    positions = generate_signals(
        z, entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=3.5
    )

    assert list(positions) == [-1, -1, -1]


def test_generate_signals_forces_flat_during_halt_and_requires_fresh_entry():
    dates = _dates(6)
    z = pd.Series([2.5, 1.0, 1.0, 1.0, 1.0, 2.5], index=dates)
    monitor_status = pd.Series(
        ["ACTIVE", "HALTED", "HALTED", "HALTED", "ACTIVE", "ACTIVE"], index=dates
    )

    positions = generate_signals(z, monitor_status=monitor_status)

    # Without monitoring, holding -1 from bar 0 would persist straight
    # through to bar 5 (z never crosses the exit/stop-loss thresholds).
    # With monitoring, the halt forces flat, and z=1.0 at reactivation
    # (bar 4) is below entry_threshold, so it stays flat until the fresh
    # entry signal at bar 5.
    assert list(positions) == [-1, 0, 0, 0, 0, -1]


def test_generate_signals_without_monitor_status_matches_unmonitored_behavior():
    dates = _dates(3)
    z = pd.Series([2.5, 1.0, 1.0], index=dates)

    assert list(generate_signals(z, monitor_status=None)) == [-1, -1, -1]


def test_generate_signals_treats_leading_nan_as_flat():
    z = pd.Series([np.nan, np.nan, np.nan], index=_dates(3))

    positions = generate_signals(z)

    assert list(positions) == [0, 0, 0]


def test_summarize_trades_classifies_reversion_and_stop_loss_exits():
    dates = _dates(11)
    z = pd.Series(
        [np.nan, np.nan, 2.1, 1.0, 0.4, -2.2, -3.6, -1.0, 2.6, 2.0, 0.3], index=dates
    )
    positions = generate_signals(
        z, entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=3.5
    )

    trades = summarize_trades(positions, z, exit_threshold=0.5)

    assert len(trades) == 3
    first, second, third = trades.to_dict("records")

    assert first == {
        "side": "short",
        "entry_date": dates[2],
        "exit_date": dates[4],
        "bars_held": 2,
        "exit_reason": "reversion",
    }
    assert second == {
        "side": "long",
        "entry_date": dates[5],
        "exit_date": dates[6],
        "bars_held": 1,
        "exit_reason": "stop_loss",
    }
    assert third == {
        "side": "short",
        "entry_date": dates[8],
        "exit_date": dates[10],
        "bars_held": 2,
        "exit_reason": "reversion",
    }


def test_summarize_trades_reports_still_open_position():
    dates = _dates(3)
    z = pd.Series([np.nan, np.nan, 2.1], index=dates)
    positions = generate_signals(z, entry_threshold=2.0)

    trades = summarize_trades(positions, z)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["side"] == "short"
    assert trade["entry_date"] == dates[2]
    assert trade["exit_date"] == dates[2]
    assert trade["exit_reason"] == "open"


def test_summarize_trades_empty_when_never_in_a_position():
    dates = _dates(3)
    z = pd.Series([0.1, -0.2, 0.3], index=dates)
    positions = generate_signals(z)

    trades = summarize_trades(positions, z)

    assert trades.empty
