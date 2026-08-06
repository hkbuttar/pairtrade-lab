import numpy as np
import pandas as pd
import pytest

from signals.hedge_ratio import (
    compare_hedge_ratios,
    kalman_hedge_ratio_series,
    static_hedge_ratio_series,
)

T = 300


def _constant_beta_pair(seed: int, beta: float = 2.0, intercept: float = 5.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    x = pd.Series(np.cumsum(rng.normal(0, 1, T)) + 100, index=dates)
    y = beta * x + intercept + rng.normal(0, 0.5, T)
    return y, x


def _drifting_beta_pair(seed: int, beta_start: float = 1.0, beta_end: float = 3.0):
    """Hedge ratio ramps linearly from beta_start to beta_end over the window,
    a case dynamic hedge ratio estimation should track better than a
    piecewise-constant one refit only every ``refit_every`` bars.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    x = pd.Series(np.cumsum(rng.normal(0, 1, T)) + 100, index=dates)
    true_beta = np.linspace(beta_start, beta_end, T)
    y = pd.Series(true_beta * x.to_numpy() + 5 + rng.normal(0, 0.3, T), index=dates)
    return y, x


def test_kalman_recovers_known_constant_hedge_ratio():
    # Sanity check against the static special case: with a truly constant
    # relationship, the Kalman estimate should settle near the same hedge
    # ratio a static OLS fit would find.
    y, x = _constant_beta_pair(seed=0, beta=2.0, intercept=5.0)

    result = kalman_hedge_ratio_series(y, x)

    # Ignore the initial transient; after burn-in the filter should be
    # stable and close to the true, constant beta.
    settled = result["hedge_ratio"].iloc[100:]
    assert settled.mean() == pytest.approx(2.0, abs=0.1)
    assert settled.std() < 0.05  # shouldn't wander once settled on a constant truth


def test_static_hedge_ratio_is_piecewise_constant():
    y, x = _constant_beta_pair(seed=1)

    result = static_hedge_ratio_series(y, x, refit_every=60, min_window=60)

    valid = result.dropna()
    assert valid.index[0] == y.index[60]
    # Within a 60-bar block between refits, the hedge ratio must be exactly
    # constant (that's the entire point of "static").
    block = valid["hedge_ratio"].iloc[0:60]
    assert (block == block.iloc[0]).all()


def test_kalman_tracks_drifting_hedge_ratio_better_than_static():
    y, x = _drifting_beta_pair(seed=2)

    comparison = compare_hedge_ratios(y, x, refit_every=60, min_window=60)

    assert comparison.kalman_rmse < comparison.static_rmse
    assert comparison.n_obs == T


def test_compare_hedge_ratios_output_shape():
    y, x = _constant_beta_pair(seed=3)

    comparison = compare_hedge_ratios(y, x)

    assert list(comparison.static.columns) == ["hedge_ratio", "intercept"]
    assert list(comparison.kalman.columns) == ["hedge_ratio", "intercept"]
    assert comparison.static_rmse > 0
    assert comparison.kalman_rmse > 0


def test_kalman_raises_on_too_few_observations():
    dates = pd.bdate_range("2020-01-01", periods=10)
    y = pd.Series(np.arange(10, dtype=float), index=dates)
    x = pd.Series(np.arange(10, dtype=float), index=dates)

    with pytest.raises(ValueError):
        kalman_hedge_ratio_series(y, x, warmup=30)
