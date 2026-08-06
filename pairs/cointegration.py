"""Engle-Granger two-step cointegration testing for candidate pairs.

Two assets can be highly correlated in *returns* while never being
cointegrated in *price levels*, and vice versa: a pair can wander apart for
years with near-zero return correlation yet still share a common stochastic
trend that occasionally pulls them back together. Correlation is about how
two series move together day to day; cointegration is about whether a linear
combination of their price *levels* is stationary, i.e. mean-reverting rather
than free to drift apart forever. Pairs trading needs the latter: a
correlated-but-not-cointegrated spread has no reason to revert, so
``test_sector_pairs`` below reports return correlation alongside the
cointegration test on every pair, winners and losers alike, so the two can be
compared directly rather than conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller

MIN_OBSERVATIONS = 60

RESULT_COLUMNS = [
    "sector",
    "ticker_y",
    "ticker_x",
    "hedge_ratio",
    "intercept",
    "adf_stat",
    "p_value",
    "p_value_fdr",
    "return_corr",
    "n_obs",
    "cointegrated",
]


@dataclass(frozen=True)
class EngleGrangerResult:
    hedge_ratio: float
    intercept: float
    adf_stat: float
    p_value: float
    n_obs: int


def engle_granger_test(y: pd.Series, x: pd.Series) -> EngleGrangerResult:
    """Engle-Granger two-step test for cointegration between ``y`` and ``x``.

    Step 1: OLS regress y on x (with intercept) to estimate the hedge ratio.
    Step 2: Augmented Dickey-Fuller test on the regression residuals; a low
    p-value is evidence the residual spread ``y - hedge_ratio * x - intercept``
    is stationary, i.e. mean-reverting rather than a random walk.

    The test is not symmetric: regressing y on x can give a different p-value
    than regressing x on y. Callers testing many pairs should fix one
    canonical (y, x) order per pair (``test_sector_pairs`` uses alphabetical
    order) rather than trying both directions and keeping the lower p-value,
    which would be a form of data snooping that invalidates the FDR
    correction applied downstream.

    Args:
        y: Dependent price series, indexed by date.
        x: Independent price series, indexed by date.

    Returns:
        EngleGrangerResult with the estimated hedge ratio/intercept and the
        ADF test statistic/p-value on the residual spread.

    Raises:
        ValueError: if fewer than MIN_OBSERVATIONS overlapping, non-NaN
            observations are available.
    """
    aligned = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    if len(aligned) < MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {MIN_OBSERVATIONS} overlapping observations, got {len(aligned)}"
        )

    x_with_const = sm.add_constant(aligned["x"])
    ols_result = sm.OLS(aligned["y"], x_with_const).fit()
    residuals = ols_result.resid

    adf_stat, p_value, *_ = adfuller(residuals, autolag="AIC")

    return EngleGrangerResult(
        hedge_ratio=float(ols_result.params["x"]),
        intercept=float(ols_result.params["const"]),
        adf_stat=float(adf_stat),
        p_value=float(p_value),
        n_obs=len(aligned),
    )


def test_sector_pairs(
    price_panel: pd.DataFrame,
    sector_tickers: dict[str, list[str]],
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Engle-Granger on every within-sector ticker pair, with Benjamini-Hochberg
    FDR correction applied across all pairs tested.

    Testing dozens of pairs at a raw p < 0.05 threshold guarantees false
    positives by chance alone; the FDR correction controls the expected
    fraction of false discoveries among the pairs flagged ``cointegrated``,
    at the cost of raising the effective bar for any single pair.

    Args:
        price_panel: Wide date-indexed price frame (see ``data.prices.to_price_panel``).
        sector_tickers: Sector name -> ticker list (see ``config.universe.SECTOR_TICKERS``).
            Only tickers present as columns in ``price_panel`` are tested.
        fdr_alpha: FDR significance level passed to ``multipletests``.

    Returns:
        One row per tested pair (both winners and rejected pairs), sorted by
        FDR-corrected p-value ascending, with columns ``RESULT_COLUMNS``.
        Pairs with fewer than MIN_OBSERVATIONS overlapping observations are
        silently skipped (not enough data to test, not evidence against
        cointegration).
    """
    rows = []
    for sector, tickers in sector_tickers.items():
        available = sorted(t for t in tickers if t in price_panel.columns)
        for ticker_y, ticker_x in combinations(available, 2):
            aligned = price_panel[[ticker_y, ticker_x]].dropna()
            if len(aligned) < MIN_OBSERVATIONS:
                continue

            result = engle_granger_test(aligned[ticker_y], aligned[ticker_x])
            return_corr = aligned.pct_change().dropna().corr().loc[ticker_y, ticker_x]

            rows.append(
                {
                    "sector": sector,
                    "ticker_y": ticker_y,
                    "ticker_x": ticker_x,
                    "hedge_ratio": result.hedge_ratio,
                    "intercept": result.intercept,
                    "adf_stat": result.adf_stat,
                    "p_value": result.p_value,
                    "return_corr": float(return_corr),
                    "n_obs": result.n_obs,
                }
            )

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    table = pd.DataFrame(rows)
    reject, p_fdr, _, _ = multipletests(table["p_value"], alpha=fdr_alpha, method="fdr_bh")
    table["p_value_fdr"] = p_fdr
    table["cointegrated"] = reject
    return table[RESULT_COLUMNS].sort_values("p_value_fdr").reset_index(drop=True)
