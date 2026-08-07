import pandas as pd

from frontend.data_access import days_since_last_break, halt_events


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_halt_events_empty_when_never_halted():
    status = pd.Series(["ACTIVE"] * 10, index=_dates(10))

    events = halt_events(status)

    assert events.empty


def test_halt_events_captures_resumed_halt():
    dates = _dates(10)
    values = ["ACTIVE", "ACTIVE"] + ["HALTED"] * 3 + ["ACTIVE"] * 5
    status = pd.Series(values, index=dates)

    events = halt_events(status)

    assert len(events) == 1
    assert events.iloc[0]["halted_at"] == dates[2]
    assert events.iloc[0]["resumed_at"] == dates[5]
    assert events.iloc[0]["days_halted"] == (dates[5] - dates[2]).days


def test_halt_events_captures_still_open_halt():
    dates = _dates(5)
    status = pd.Series(["ACTIVE", "ACTIVE", "HALTED", "HALTED", "HALTED"], index=dates)

    events = halt_events(status)

    assert len(events) == 1
    assert events.iloc[0]["halted_at"] == dates[2]
    assert events.iloc[0]["resumed_at"] is None


def test_halt_events_captures_multiple_halts():
    dates = _dates(9)
    status = pd.Series(
        ["ACTIVE", "HALTED", "ACTIVE", "ACTIVE", "HALTED", "HALTED", "ACTIVE", "HALTED", "HALTED"],
        index=dates,
    )

    events = halt_events(status)

    assert len(events) == 3
    assert list(events["halted_at"]) == [dates[1], dates[4], dates[7]]
    assert events.iloc[0]["resumed_at"] == dates[2]
    assert events.iloc[1]["resumed_at"] == dates[6]
    assert pd.isna(events.iloc[2]["resumed_at"])


def test_days_since_last_break_none_when_never_halted():
    status = pd.Series(["ACTIVE"] * 10, index=_dates(10))

    assert days_since_last_break(status) is None


def test_days_since_last_break_measures_from_most_recent_halt_start():
    dates = _dates(10)
    values = ["ACTIVE", "HALTED", "ACTIVE", "ACTIVE", "ACTIVE", "HALTED"] + ["ACTIVE"] * 4
    status = pd.Series(values, index=dates)

    result = days_since_last_break(status)

    # Most recent halt started at dates[5]; window ends at dates[9].
    assert result == (dates[9] - dates[5]).days
