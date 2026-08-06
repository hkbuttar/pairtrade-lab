"""Johansen basket selection: extends pair cointegration testing to 3-5 leg
baskets, where the Johansen procedure (statsmodels.tsa.vector_ar.vecm.coint_johansen)
tests cointegration rank across N price series simultaneously. Its
eigenvectors directly give the cointegrating vector(s), which double as the
basket's hedge ratios: unlike pairs, there is no separate OLS step.

statsmodels' coint_johansen does not return an exact p-value for the trace
statistic (unlike the ADF test pairs/cointegration.py uses for pairs), only
critical values at the 90%, 95%, and 99% levels. To apply the same
Benjamini-Hochberg FDR discipline used for pairs, this module approximates a
p-value by log-linear interpolation/extrapolation across those three
critical values. This is a standard practical approximation (the tail of the
trace statistic's null distribution is close to log-linear over this range),
disclosed here as exactly that: an approximation, not an exact p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from pairs.cointegration import MIN_OBSERVATIONS

RESULT_COLUMNS = [
    "sector",
    "basket_size",
    "tickers",
    "weights",
    "trace_stat",
    "p_value",
    "p_value_fdr",
    "n_obs",
    "cointegrated",
]


@dataclass(frozen=True)
class JohansenResult:
    weights: dict[str, float]
    trace_stat: float
    p_value: float
    n_obs: int


def _approximate_trace_p_value(statistic: float, critical_values: np.ndarray) -> float:
    """Approximate the p-value of a Johansen trace-test statistic.

    Fits log(alpha) = a + b * critical_value by least squares on the three
    published (alpha, critical_value) pairs at the 90%/95%/99% levels, then
    evaluates that fit at ``statistic``. See module docstring for why this
    approximation is necessary and its limits.
    """
    alphas = np.array([0.10, 0.05, 0.01])
    design = np.column_stack([np.ones(3), critical_values])
    (intercept, slope), *_ = np.linalg.lstsq(design, np.log(alphas), rcond=None)
    p_value = np.exp(intercept + slope * statistic)
    return float(np.clip(p_value, 1e-6, 1.0))


def johansen_test(prices: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
    """Johansen test for cointegration rank >= 1 among a basket of price series.

    Only tests the H0: rank == 0 hypothesis (no cointegration among the
    basket at all), the basket analogue of the single p-value Engle-Granger
    produces for a pair. A basket can in principle have more than one
    cointegrating relationship; this only recovers the strongest one (the
    eigenvector with the largest eigenvalue), which is what defines the
    traded spread.

    Args:
        prices: Wide date-indexed price frame, one column per leg (3-5
            columns expected, though any width >= 2 works).
        det_order: Deterministic term in the VECM. 0 = constant, no trend
            (the standard choice for price levels with a nonzero mean spread).
        k_ar_diff: Lag order of the VECM in first differences. Fixed rather
            than selected per basket via an information criterion, disclosed
            here as a modeling assumption (see README).

    Returns:
        JohansenResult with basket weights (the leading eigenvector,
        normalized so its largest-magnitude leg has weight 1, since the
        vector's own first element can be near zero and blow up a
        divide-by-first-element normalization), the trace statistic, and the
        approximate p-value for the rank == 0 null.

    Raises:
        ValueError: if fewer than MIN_OBSERVATIONS non-NaN aligned rows exist.
    """
    aligned = prices.dropna()
    if len(aligned) < MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {MIN_OBSERVATIONS} overlapping observations, got {len(aligned)}"
        )

    result = coint_johansen(aligned.to_numpy(), det_order, k_ar_diff)

    trace_stat = float(result.lr1[0])
    p_value = _approximate_trace_p_value(trace_stat, result.cvt[0])

    eigenvector = result.evec[:, 0]
    anchor = int(np.argmax(np.abs(eigenvector)))
    normalized = eigenvector / eigenvector[anchor]
    weights = dict(zip(aligned.columns, (float(w) for w in normalized), strict=True))

    return JohansenResult(
        weights=weights, trace_stat=trace_stat, p_value=p_value, n_obs=len(aligned)
    )


def test_sector_baskets(
    price_panel: pd.DataFrame,
    sector_tickers: dict[str, list[str]],
    basket_sizes: tuple[int, ...] = (3, 4, 5),
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Run the Johansen test on every within-sector basket of the given sizes,
    with Benjamini-Hochberg FDR correction applied across all baskets tested.

    Same discipline as pairs.cointegration.test_sector_pairs: every basket
    tested is reported (winners and rejects alike), corrected p-values are
    computed across the full batch (not per sector or per basket size), and
    baskets with too little overlapping data are silently skipped rather than
    counted as a rejection.

    Args:
        price_panel: Wide date-indexed price frame.
        sector_tickers: Sector name -> ticker list.
        basket_sizes: Leg counts to search, e.g. (3, 4, 5). Larger baskets and
            more sectors multiply the number of combinations tested quickly
            (e.g. 13 tickers has 715 4-leg combinations), so callers with a
            large universe may want to restrict this for runtime.
        fdr_alpha: FDR significance level.

    Returns:
        One row per tested basket, sorted by FDR-corrected p-value ascending,
        columns ``RESULT_COLUMNS``.
    """
    rows = []
    for sector, tickers in sector_tickers.items():
        available = sorted(t for t in tickers if t in price_panel.columns)
        for size in basket_sizes:
            for combo in combinations(available, size):
                aligned = price_panel[list(combo)].dropna()
                if len(aligned) < MIN_OBSERVATIONS:
                    continue
                try:
                    result = johansen_test(aligned)
                except np.linalg.LinAlgError:
                    continue

                rows.append(
                    {
                        "sector": sector,
                        "basket_size": size,
                        "tickers": combo,
                        "weights": result.weights,
                        "trace_stat": result.trace_stat,
                        "p_value": result.p_value,
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
