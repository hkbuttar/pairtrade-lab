import numpy as np
import pandas as pd
import pytest

from backtest.bootstrap import (
    BootstrapResult,
    block_bootstrap_resample,
    bootstrap_backtest_metrics,
)


def _ar1_series(seed: int, n: int = 2000, phi: float = 0.7) -> np.ndarray:
    """A synthetic series with a known, strong lag-1 autocorrelation (~phi),
    the standard way to test that a block bootstrap actually preserves
    autocorrelation structure rather than just resampling i.i.d.
    """
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for t in range(1, n):
        values[t] = phi * values[t - 1] + rng.normal(0, 1)
    return values


def _lag1_autocorr(values: np.ndarray) -> float:
    return float(pd.Series(values).autocorr(lag=1))


def test_block_bootstrap_preserves_known_autocorrelation():
    series = _ar1_series(seed=0, phi=0.7)
    true_autocorr = _lag1_autocorr(series)

    resampled = block_bootstrap_resample(series, block_length=20, n_resamples=200, seed=1)
    block_autocorrs = [_lag1_autocorr(row) for row in resampled]

    assert np.mean(block_autocorrs) == pytest.approx(true_autocorr, abs=0.15)


def test_block_bootstrap_preserves_autocorrelation_better_than_iid():
    series = _ar1_series(seed=2, phi=0.7)
    true_autocorr = _lag1_autocorr(series)

    block_resampled = block_bootstrap_resample(series, block_length=20, n_resamples=200, seed=3)
    block_autocorrs = [_lag1_autocorr(row) for row in block_resampled]

    iid_rng = np.random.default_rng(4)
    n = len(series)
    iid_autocorrs = [
        _lag1_autocorr(series[iid_rng.integers(0, n, size=n)]) for _ in range(200)
    ]

    block_error = abs(np.mean(block_autocorrs) - true_autocorr)
    iid_error = abs(np.mean(iid_autocorrs) - true_autocorr)
    assert block_error < iid_error
    # The naive i.i.d. bootstrap should destroy the autocorrelation almost
    # entirely (mean near 0), which is exactly the failure mode block
    # bootstrap exists to avoid.
    assert np.mean(iid_autocorrs) == pytest.approx(0.0, abs=0.1)


def test_block_bootstrap_resample_shape():
    series = np.arange(100, dtype=float)

    resampled = block_bootstrap_resample(series, block_length=10, n_resamples=50, seed=0)

    assert resampled.shape == (50, 100)


def test_block_bootstrap_resample_only_uses_values_from_original_series():
    series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    resampled = block_bootstrap_resample(series, block_length=2, n_resamples=20, seed=0)

    assert set(np.unique(resampled)) <= set(series)


def test_bootstrap_backtest_metrics_ci_contains_point_estimate_direction():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 500))

    results = bootstrap_backtest_metrics(returns, block_length=20, n_resamples=300, seed=1)

    assert set(results) == {"cagr", "sharpe_ratio", "sortino_ratio", "max_drawdown", "win_rate"}
    for result in results.values():
        assert isinstance(result, BootstrapResult)
        assert result.ci_low <= result.ci_high
        assert result.ci_width >= 0


def test_bootstrap_backtest_metrics_narrower_interval_with_more_data():
    # More independent observations of the same underlying process should
    # narrow the confidence interval, a basic statistical fact any CI
    # procedure should respect.
    rng = np.random.default_rng(0)
    short = pd.Series(rng.normal(0.0005, 0.01, 200))
    long_ = pd.Series(np.concatenate([short.to_numpy(), rng.normal(0.0005, 0.01, 1800)]))

    short_result = bootstrap_backtest_metrics(short, block_length=20, n_resamples=300, seed=2)
    long_result = bootstrap_backtest_metrics(long_, block_length=20, n_resamples=300, seed=2)

    assert long_result["sharpe_ratio"].ci_width < short_result["sharpe_ratio"].ci_width


def test_bootstrap_backtest_metrics_raises_on_too_few_observations():
    returns = pd.Series([0.01, -0.01, 0.02])

    with pytest.raises(ValueError):
        bootstrap_backtest_metrics(returns, block_length=20)


def test_bootstrap_backtest_metrics_reproducible_with_seed():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0, 0.01, 300))

    result_a = bootstrap_backtest_metrics(returns, block_length=15, n_resamples=200, seed=42)
    result_b = bootstrap_backtest_metrics(returns, block_length=15, n_resamples=200, seed=42)

    assert result_a["sharpe_ratio"].ci_low == result_b["sharpe_ratio"].ci_low
    assert result_a["sharpe_ratio"].ci_high == result_b["sharpe_ratio"].ci_high
