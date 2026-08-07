"""Block bootstrap confidence intervals for backtest performance metrics.

A pairs-trading return series is autocorrelated: a trade held for several
days produces a short run of correlated daily returns, not independent
draws, and the flat (zero-return) stretches between trades are themselves
runs, not noise. Resampling individual days with replacement (a naive i.i.d.
bootstrap) destroys that structure and would understate how uncertain a
metric like Sharpe or CAGR really is. The circular block bootstrap (Politis
& Romano, 1992) instead resamples whole contiguous blocks of ``block_length``
consecutive days (wrapping around circularly, so every original day is
equally likely to start a block, including ones near the end of the
series), which preserves within-block autocorrelation while still varying
which stretches of history make it into any given resample.

``block_length`` is a disclosed, unfitted parameter, not derived from the
data: too short and it degenerates toward the naive i.i.d. bootstrap it's
meant to improve on; too long and too few effectively-independent blocks
exist to vary across resamples, understating uncertainty a different way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import metrics as backtest_metrics

DEFAULT_BLOCK_LENGTH = 20
DEFAULT_N_RESAMPLES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    n_resamples: int
    confidence_level: float

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


def _circular_block_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """One resampled index path of length n, built from blocks of
    block_length consecutive (circularly wrapped) original indices.
    """
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])
    return indices[:n]


def block_bootstrap_resample(
    values: np.ndarray, block_length: int, n_resamples: int, seed: int | None = None
) -> np.ndarray:
    """n_resamples resampled paths of the same length as ``values``, via
    circular block bootstrap.

    Returns:
        Array of shape (n_resamples, len(values)).
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    resampled = np.empty((n_resamples, n))
    for i in range(n_resamples):
        indices = _circular_block_bootstrap_indices(n, block_length, rng)
        resampled[i] = values[indices]
    return resampled


def _equity_from_returns(returns: pd.Series, starting_equity: float = 1.0) -> pd.Series:
    """Normalized equity curve implied by a return series, with a leading
    point at ``starting_equity`` before the first return (so the resulting
    series has the same shape a real N+1-point equity curve would).
    """
    growth = (1 + returns).cumprod()
    return pd.concat([pd.Series([starting_equity]), starting_equity * growth], ignore_index=True)


def _compute_all_metrics(returns: pd.Series) -> dict[str, float]:
    equity = _equity_from_returns(returns)
    return {
        "cagr": backtest_metrics.cagr(equity),
        "sharpe_ratio": backtest_metrics.sharpe_ratio(returns),
        "sortino_ratio": backtest_metrics.sortino_ratio(returns),
        "max_drawdown": backtest_metrics.max_drawdown(equity),
        "win_rate": backtest_metrics.win_rate(returns),
    }


def bootstrap_backtest_metrics(
    returns: pd.Series,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int | None = None,
) -> dict[str, BootstrapResult]:
    """Block bootstrap confidence intervals for CAGR, Sharpe, Sortino, max
    drawdown, and win rate, all from one shared set of resampled return paths.

    Args:
        returns: Daily return series (e.g. backtest equity curve's
            pct_change()), NaN-dropped internally.
        block_length: Consecutive-day block length for the circular block
            bootstrap. See module docstring for the tradeoff in choosing it.
        n_resamples: Number of bootstrap resamples.
        confidence_level: E.g. 0.95 for a 95% percentile interval.
        seed: Optional seed for reproducibility.

    Returns:
        {metric_name: BootstrapResult}, one entry per metric in
        _compute_all_metrics.

    Raises:
        ValueError: if fewer than block_length non-NaN returns are available.
    """
    clean = returns.dropna()
    if len(clean) < block_length:
        raise ValueError(
            f"need at least block_length={block_length} return observations, got {len(clean)}"
        )

    point_estimates = _compute_all_metrics(clean)

    arr = clean.to_numpy()
    resampled_paths = block_bootstrap_resample(arr, block_length, n_resamples, seed=seed)

    distributions: dict[str, list[float]] = {name: [] for name in point_estimates}
    for i in range(n_resamples):
        resampled_metrics = _compute_all_metrics(pd.Series(resampled_paths[i]))
        for name, value in resampled_metrics.items():
            distributions[name].append(value)

    alpha = 1 - confidence_level
    results = {}
    for name, point in point_estimates.items():
        dist = np.array(distributions[name])
        dist = dist[~np.isnan(dist)]
        if len(dist) == 0:
            ci_low, ci_high = float("nan"), float("nan")
        else:
            ci_low, ci_high = (float(q) for q in np.quantile(dist, [alpha / 2, 1 - alpha / 2]))
        results[name] = BootstrapResult(
            point_estimate=point,
            ci_low=ci_low,
            ci_high=ci_high,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
        )

    return results
