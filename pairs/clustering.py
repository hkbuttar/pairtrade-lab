"""ML-assisted candidate discovery: hierarchical clustering on price series to
surface pair/basket candidates beyond the hand-picked sector groupings in
config/universe.py.

Two ingredients, both named in the project plan as alternatives, are
combined here rather than picked between:

1. PCA-based factor extraction: clustering raw price-level correlation
   collapses almost the entire universe into one dominant cluster, since most
   equities share the same broad-market trend (empirically, ~40% of daily
   return variance in this universe loads onto a single first principal
   component). That one giant cluster is useless as a candidate generator: it
   is too large to basket-search and it is just "the market," not a
   meaningful economic grouping. Regressing out the top principal
   component(s) from each ticker's standardized returns before clustering
   removes that common factor and leaves the idiosyncratic co-movement that
   actually distinguishes tickers from each other.
2. Hierarchical clustering: agglomerative clustering with average linkage on
   the Mantegna (1999) correlation distance, d_ij = sqrt(2 * (1 - corr_ij)),
   between residual return series, cut into a fixed number of clusters.

The output is a {cluster_name: [tickers]} mapping in exactly the shape
config.universe.SECTOR_TICKERS is, so pairs.cointegration.test_sector_pairs
and pairs.johansen.test_sector_baskets can consume ML-discovered clusters
with no changes: neither function actually assumes its groups are sectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


def _standardized_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    prices = price_panel.dropna()
    returns = prices.pct_change().dropna()
    return (returns - returns.mean()) / returns.std()


def market_factor_variance_ratio(price_panel: pd.DataFrame, n_components: int = 5) -> np.ndarray:
    """Fraction of standardized-return variance explained by the top principal
    components, for reporting how dominant the common market factor is.
    """
    standardized = _standardized_returns(price_panel)
    pca = PCA(n_components=n_components)
    pca.fit(standardized.to_numpy())
    return pca.explained_variance_ratio_


def _residualize_market_factor(standardized: pd.DataFrame, n_market_factors: int) -> pd.DataFrame:
    pca = PCA(n_components=n_market_factors)
    factors = pca.fit_transform(standardized.to_numpy())
    loadings, *_ = np.linalg.lstsq(factors, standardized.to_numpy(), rcond=None)
    residual = standardized.to_numpy() - factors @ loadings
    return pd.DataFrame(residual, index=standardized.index, columns=standardized.columns)


def cluster_tickers(
    price_panel: pd.DataFrame,
    n_clusters: int = 12,
    n_market_factors: int = 1,
    linkage_method: str = "average",
) -> dict[str, list[str]]:
    """Cluster tickers by residual (market-factor-removed) return correlation.

    Args:
        price_panel: Wide date-indexed price frame across the full universe
            (not restricted to one sector); rows with any missing ticker are
            dropped so every ticker is compared over the same window.
        n_clusters: Number of clusters to cut the dendrogram into.
        n_market_factors: How many leading principal components to regress
            out before clustering. 1 (just the market factor) is the
            default; disclosed as a modeling choice, not derived from data.
        linkage_method: Passed to scipy.cluster.hierarchy.linkage.

    Returns:
        {cluster_name: sorted ticker list}, restricted to clusters with 2 or
        more members (a singleton cluster cannot form a pair or basket).
    """
    standardized = _standardized_returns(price_panel)
    residual = _residualize_market_factor(standardized, n_market_factors)

    correlation = residual.corr()
    distance = np.sqrt(2 * (1 - correlation.clip(-1, 1))).to_numpy().copy()
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)

    tree = linkage(condensed, method=linkage_method)
    labels = fcluster(tree, t=n_clusters, criterion="maxclust")

    clusters: dict[str, list[str]] = {}
    for ticker, label in zip(residual.columns, labels, strict=True):
        clusters.setdefault(f"cluster_{label}", []).append(ticker)

    return {name: sorted(tickers) for name, tickers in clusters.items() if len(tickers) >= 2}
