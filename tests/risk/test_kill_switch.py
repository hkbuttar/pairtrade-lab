import pandas as pd
import pytest

from risk.kill_switch import KillSwitch, running_drawdown


def test_kill_switch_does_not_trigger_below_threshold():
    ks = KillSwitch(max_drawdown=0.15)

    ks.check(100.0)
    triggered = ks.check(90.0)  # 10% drawdown

    assert triggered is False
    assert ks.triggered is False


def test_kill_switch_triggers_at_threshold():
    ks = KillSwitch(max_drawdown=0.15)

    ks.check(100.0)
    triggered = ks.check(84.0)  # 16% drawdown

    assert triggered is True
    assert ks.triggered is True


def test_kill_switch_is_sticky_and_does_not_auto_resume():
    ks = KillSwitch(max_drawdown=0.15)

    ks.check(100.0)
    ks.check(80.0)  # triggers
    still_triggered = ks.check(110.0)  # equity recovers well past the old peak

    assert still_triggered is True


def test_kill_switch_reset_rearms_from_a_fresh_peak():
    ks = KillSwitch(max_drawdown=0.15)
    ks.check(100.0)
    ks.check(80.0)  # triggers

    ks.reset()

    assert ks.triggered is False
    assert ks.check(79.0) is False  # fresh peak established at 79, no drawdown yet


def test_kill_switch_tracks_running_peak_not_first_value():
    ks = KillSwitch(max_drawdown=0.15)

    ks.check(100.0)
    ks.check(120.0)  # new peak
    triggered = ks.check(105.0)  # 12.5% off the 120 peak, not the original 100

    assert triggered is False


def test_running_drawdown_matches_manual_calculation():
    equity = pd.Series([100.0, 120.0, 90.0, 110.0])

    result = running_drawdown(equity)

    assert result.iloc[0] == 0.0
    assert result.iloc[1] == 0.0
    assert result.iloc[2] == pytest.approx((120.0 - 90.0) / 120.0)
    assert result.iloc[3] == pytest.approx((120.0 - 110.0) / 120.0)
