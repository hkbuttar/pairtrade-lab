import pytest

from risk.limits import clip_new_pair_notional


def test_clip_below_limits_is_unchanged():
    result = clip_new_pair_notional(
        raw_notional=10_000,
        existing_notional={},
        equity=1_000_000,
        max_notional_per_pair_fraction=0.5,
        max_gross_exposure_fraction=1.0,
    )

    assert result == 10_000


def test_clip_to_per_pair_cap():
    result = clip_new_pair_notional(
        raw_notional=800_000,
        existing_notional={},
        equity=1_000_000,
        max_notional_per_pair_fraction=0.5,
        max_gross_exposure_fraction=1.0,
    )

    assert result == 500_000


def test_clip_to_remaining_gross_budget():
    result = clip_new_pair_notional(
        raw_notional=400_000,
        existing_notional={"AAA/BBB": 700_000},
        equity=1_000_000,
        max_notional_per_pair_fraction=0.5,
        max_gross_exposure_fraction=1.0,
    )

    # Per-pair cap allows 500k, but only 300k of gross budget remains
    # (1.0 * 1_000_000 - 700_000).
    assert result == 300_000


def test_clip_returns_zero_when_no_gross_budget_remains():
    result = clip_new_pair_notional(
        raw_notional=100_000,
        existing_notional={"AAA/BBB": 1_000_000},
        equity=1_000_000,
        max_gross_exposure_fraction=1.0,
    )

    assert result == 0.0


def test_clip_preserves_sign():
    result = clip_new_pair_notional(
        raw_notional=-800_000,
        existing_notional={},
        equity=1_000_000,
        max_notional_per_pair_fraction=0.5,
    )

    assert result == -500_000


def test_clip_returns_zero_for_nonpositive_equity():
    assert clip_new_pair_notional(100_000, {}, equity=0.0) == 0.0
    assert clip_new_pair_notional(100_000, {}, equity=-500.0) == 0.0


def test_clip_returns_zero_for_zero_raw_notional():
    assert clip_new_pair_notional(0.0, {}, equity=1_000_000) == 0.0


def test_clip_existing_notional_only_uses_magnitude():
    # A short (-600k) and a long (-600k... use distinct pairs) should both
    # count toward gross budget by absolute value, not net out.
    result = clip_new_pair_notional(
        raw_notional=100_000,
        existing_notional={"A": -600_000, "B": 350_000},
        equity=1_000_000,
        max_gross_exposure_fraction=1.0,
    )

    # existing gross = 950_000, remaining budget = 50_000
    assert result == pytest.approx(50_000)
