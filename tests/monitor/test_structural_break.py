import numpy as np
import pandas as pd
import pytest

from monitor.structural_break import (
    cusum_detect,
    monitor_pair_status,
    rolling_cointegration_pvalue,
)
from pairs.cointegration import MIN_OBSERVATIONS


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def test_rolling_cointegration_pvalue_detects_a_relationship_breaking_down():
    rng = np.random.default_rng(0)
    n = 250
    dates = _dates(n)
    x = np.cumsum(rng.normal(0, 1, n)) + 100

    # First half: y is genuinely cointegrated with x. Second half: y instead
    # follows its own independent random walk, so the relationship breaks.
    y = np.empty(n)
    y[:150] = 2.0 * x[:150] + 5 + rng.normal(0, 0.5, 150)
    y[150:] = y[149] + np.cumsum(rng.normal(0, 1, n - 150))

    y = pd.Series(y, index=dates)
    x = pd.Series(x, index=dates)

    pvalues = rolling_cointegration_pvalue(y, x, window=90, step=1)

    # Well within the still-cointegrated region, the rolling p-value should
    # be low; well after the break, it should have risen above 0.05.
    assert pvalues.iloc[140] < 0.05
    assert pvalues.iloc[-1] > 0.05


def test_rolling_cointegration_pvalue_rejects_too_small_window():
    dates = _dates(100)
    y = pd.Series(np.arange(100, dtype=float), index=dates)
    x = pd.Series(np.arange(100, dtype=float), index=dates)

    with pytest.raises(ValueError):
        rolling_cointegration_pvalue(y, x, window=MIN_OBSERVATIONS - 1)


def test_cusum_does_not_flag_stable_in_control_noise():
    rng = np.random.default_rng(1)
    z = pd.Series(rng.normal(0, 1, 200), index=_dates(200))

    result = cusum_detect(z, k=0.5, h=5.0)

    assert not result["break_flagged"].any()


def test_cusum_flags_a_sustained_mean_shift_promptly():
    rng = np.random.default_rng(2)
    n = 300
    shift_start = 150
    z = np.concatenate(
        [rng.normal(0, 1, shift_start), rng.normal(2.5, 1, n - shift_start)]
    )
    z = pd.Series(z, index=_dates(n))

    result = cusum_detect(z, k=0.5, h=5.0)

    flagged_after_shift = result["break_flagged"].iloc[shift_start:]
    assert flagged_after_shift.any()
    first_flag_offset = flagged_after_shift.to_numpy().argmax()
    assert first_flag_offset < 30  # flagged reasonably promptly, not eventually


def test_cusum_resets_accumulator_after_flagging():
    rng = np.random.default_rng(3)
    n = 300
    z = pd.Series(
        np.concatenate([rng.normal(0, 1, 150), rng.normal(3.0, 1, 150)]), index=_dates(n)
    )

    result = cusum_detect(z, k=0.5, h=5.0)

    flag_indices = np.flatnonzero(result["break_flagged"].to_numpy())
    assert len(flag_indices) > 0
    first_flag = flag_indices[0]
    # The accumulator that triggered the flag must have reset to 0 that same bar.
    reset_pos = result["cusum_pos"].iloc[first_flag] == 0.0
    reset_neg = result["cusum_neg"].iloc[first_flag] == 0.0
    assert reset_pos or reset_neg


def test_cusum_treats_leading_nan_as_zero_not_a_crash():
    z = pd.Series([np.nan, np.nan, 0.1, -0.2, 0.3], index=_dates(5))

    result = cusum_detect(z)

    assert not result["break_flagged"].any()
    assert len(result) == 5


def test_monitor_pair_status_halts_and_requalifies():
    pvalues = pd.Series([0.01] * 10 + [0.2] * 3 + [0.01] * 10, index=_dates(23))
    cusum_flagged = pd.Series([False] * 23, index=_dates(23))

    status = monitor_pair_status(pvalues, cusum_flagged, pvalue_threshold=0.05, requalify_bars=5)

    assert list(status.iloc[0:10]) == ["ACTIVE"] * 10
    assert list(status.iloc[10:17]) == ["HALTED"] * 7
    assert list(status.iloc[17:23]) == ["ACTIVE"] * 6


def test_monitor_pair_status_halts_on_cusum_alone():
    pvalues = pd.Series([0.01] * 23, index=_dates(23))
    cusum_flagged = pd.Series([False] * 5 + [True] + [False] * 17, index=_dates(23))

    status = monitor_pair_status(pvalues, cusum_flagged, pvalue_threshold=0.05, requalify_bars=5)

    assert list(status.iloc[0:5]) == ["ACTIVE"] * 5
    assert status.iloc[5] == "HALTED"
    # Requalifies after 5 consecutive low-p-value bars post-halt (bars 6-10).
    assert list(status.iloc[6:10]) == ["HALTED"] * 4
    assert status.iloc[10] == "ACTIVE"
