"""Static (OLS) and dynamic (Kalman filter) hedge ratio estimation for pairs.

The static approach (periodic OLS refit, held constant in between) is the
classic starting point and is known to decay as the true relationship
drifts: the hedge ratio is frozen at whatever it was at the last refit,
right up until the next one. The dynamic approach treats the hedge ratio as
a latent, slowly-evolving state and updates it every observation via a
Kalman filter (the local-linear-trend formulation from Chan, "Algorithmic
Trading", ch. 3: y_t = beta_t * x_t + alpha_t + noise, with [beta_t, alpha_t]
following a random walk).

``compare_hedge_ratios`` puts both on equal footing with a strictly
out-of-sample one-step-ahead prediction error: at each t, the hedge
ratio/intercept estimated using only information through t-1 is used to
predict y_t from x_t, so neither method gets to peek at the observation
it's being scored on. This is deliberately a narrower question than "which
one makes more backtested money" (that needs the full event-driven
simulator in backtest/, not built yet); it only asks which hedge ratio
update mechanism tracks the true relationship more closely, out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter

MIN_WARMUP = 20


def static_hedge_ratio_series(
    y: pd.Series, x: pd.Series, refit_every: int = 60, min_window: int = 60
) -> pd.DataFrame:
    """Piecewise-constant hedge ratio: OLS-refit every ``refit_every`` bars on
    a trailing ``min_window`` of history, held fixed until the next refit.

    The first ``min_window`` observations have no prior data to fit on and
    are dropped (NaN hedge ratio would just be silently carried as the first
    valid value otherwise, which is not what "no estimate yet" should mean).

    Args:
        y: Dependent price series, indexed by date.
        x: Independent price series, indexed by date.
        refit_every: Bars between OLS re-estimation.
        min_window: Trailing window used for each refit.

    Returns:
        DataFrame indexed like the aligned (y, x), columns [hedge_ratio,
        intercept], NaN for the initial min_window warmup bars.
    """
    aligned = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    hedge_ratio = pd.Series(np.nan, index=aligned.index)
    intercept = pd.Series(np.nan, index=aligned.index)

    current_beta, current_alpha = np.nan, np.nan
    for i in range(len(aligned)):
        if i >= min_window and (i - min_window) % refit_every == 0:
            window = aligned.iloc[max(0, i - min_window) : i]
            x_with_const = sm.add_constant(window["x"])
            fit = sm.OLS(window["y"], x_with_const).fit()
            current_beta = fit.params["x"]
            current_alpha = fit.params["const"]
        if i >= min_window:
            hedge_ratio.iloc[i] = current_beta
            intercept.iloc[i] = current_alpha

    return pd.DataFrame({"hedge_ratio": hedge_ratio, "intercept": intercept})


def kalman_hedge_ratio_series(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_covariance: float = 1.0,
    warmup: int = 30,
) -> pd.DataFrame:
    """Kalman-filtered hedge ratio, treating [beta_t, alpha_t] as a random walk.

    The filter's initial state is seeded from a static OLS fit on the first
    ``warmup`` observations (per the project plan: "initialized from the
    static OLS estimate and evolving from there"), not an arbitrary [0, 0].

    Args:
        y: Dependent price series, indexed by date.
        x: Independent price series, indexed by date.
        delta: Controls the state transition covariance (process noise) as
            delta / (1 - delta); larger delta lets the hedge ratio move
            faster between bars. Disclosed as a modeling assumption, not
            fit from data.
        obs_covariance: Observation noise variance.
        warmup: Number of leading observations used for the seeding OLS fit.

    Returns:
        DataFrame indexed like the aligned (y, x), columns [hedge_ratio,
        intercept], one row per observation including the warmup window
        (the filter runs over the full series; only its *initial state* uses
        the warmup fit).

    Raises:
        ValueError: if fewer than ``max(MIN_WARMUP, warmup)`` observations
            are available.
    """
    aligned = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    warmup = max(warmup, MIN_WARMUP)
    if len(aligned) < warmup:
        raise ValueError(f"need at least {warmup} observations, got {len(aligned)}")

    seed = sm.add_constant(aligned["x"].iloc[:warmup])
    seed_fit = sm.OLS(aligned["y"].iloc[:warmup], seed).fit()
    initial_state_mean = np.array([seed_fit.params["x"], seed_fit.params["const"]])

    observation_matrices = np.stack([aligned["x"].to_numpy(), np.ones(len(aligned))], axis=1)[
        :, None, :
    ]
    transition_covariance = delta / (1 - delta) * np.eye(2)

    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=np.eye(2),
        transition_matrices=np.eye(2),
        observation_matrices=observation_matrices,
        observation_covariance=obs_covariance,
        transition_covariance=transition_covariance,
    )
    state_means, _ = kf.filter(aligned["y"].to_numpy())

    return pd.DataFrame(
        {"hedge_ratio": state_means[:, 0], "intercept": state_means[:, 1]}, index=aligned.index
    )


@dataclass(frozen=True)
class HedgeRatioComparison:
    static: pd.DataFrame
    kalman: pd.DataFrame
    static_rmse: float
    kalman_rmse: float
    n_obs: int


def _one_step_ahead_rmse(y: pd.Series, x: pd.Series, estimates: pd.DataFrame) -> float:
    """RMSE of predicting y_t from x_t using estimates.shift(1) (strictly prior
    information only), skipping bars with no estimate yet.
    """
    lagged = estimates.shift(1)
    predicted = lagged["hedge_ratio"] * x + lagged["intercept"]
    error = (y - predicted).dropna()
    return float(np.sqrt((error**2).mean()))


def compare_hedge_ratios(
    y: pd.Series,
    x: pd.Series,
    refit_every: int = 60,
    min_window: int = 60,
    delta: float = 1e-4,
    obs_covariance: float = 1.0,
) -> HedgeRatioComparison:
    """Compare static and Kalman hedge ratios on the same pair via out-of-sample
    one-step-ahead prediction error (see module docstring for why this metric,
    not a backtested P&L, is used at this stage).
    """
    static = static_hedge_ratio_series(y, x, refit_every=refit_every, min_window=min_window)
    kalman = kalman_hedge_ratio_series(y, x, delta=delta, obs_covariance=obs_covariance)

    aligned_y, aligned_x = y.align(x, join="inner")
    static_rmse = _one_step_ahead_rmse(aligned_y, aligned_x, static)
    kalman_rmse = _one_step_ahead_rmse(aligned_y, aligned_x, kalman)

    return HedgeRatioComparison(
        static=static,
        kalman=kalman,
        static_rmse=static_rmse,
        kalman_rmse=kalman_rmse,
        n_obs=len(aligned_y),
    )
