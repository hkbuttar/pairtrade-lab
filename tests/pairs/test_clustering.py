import numpy as np
import pandas as pd

from pairs.clustering import cluster_tickers, market_factor_variance_ratio

T = 300


def _synthetic_grouped_prices(seed: int, n_groups: int = 3, group_size: int = 4) -> pd.DataFrame:
    """n_groups * group_size tickers, each return = market factor + its group's
    factor + idiosyncratic noise. Group factors dominate idiosyncratic noise,
    so after the (shared) market factor is regressed out, tickers should
    cluster cleanly by group.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    market = rng.normal(0, 1, T)

    columns = {}
    for g in range(n_groups):
        group_factor = rng.normal(0, 1, T)
        for i in range(group_size):
            idio = rng.normal(0, 0.3, T)
            returns = 0.5 * market + 1.0 * group_factor + idio
            price = 100 * np.cumprod(1 + returns / 100)
            columns[f"g{g}_{i}"] = pd.Series(price, index=dates)
    return pd.DataFrame(columns)


def test_market_factor_variance_ratio_shape_and_bounds():
    prices = _synthetic_grouped_prices(seed=0)

    ratios = market_factor_variance_ratio(prices, n_components=5)

    assert len(ratios) == 5
    assert (ratios >= 0).all()
    assert ratios.sum() <= 1.0 + 1e-9
    # Explained variance is sorted descending by construction of PCA.
    assert list(ratios) == sorted(ratios, reverse=True)


def test_cluster_tickers_recovers_synthetic_groups():
    prices = _synthetic_grouped_prices(seed=1, n_groups=3, group_size=4)

    clusters = cluster_tickers(prices, n_clusters=3, n_market_factors=1)

    # Every returned cluster should be "pure": all members share the same
    # true group prefix (g0/g1/g2), since group factors dominate the noise.
    for tickers in clusters.values():
        true_groups = {ticker.split("_")[0] for ticker in tickers}
        assert len(true_groups) == 1

    # All 12 tickers should show up somewhere (no singleton groups expected
    # here, since each true group has 4 members).
    clustered_tickers = {t for tickers in clusters.values() for t in tickers}
    assert clustered_tickers == set(prices.columns)


def test_cluster_tickers_drops_singleton_clusters():
    prices = _synthetic_grouped_prices(seed=2, n_groups=3, group_size=4)

    # Asking for as many clusters as tickers forces most into singletons.
    clusters = cluster_tickers(prices, n_clusters=prices.shape[1], n_market_factors=1)

    for tickers in clusters.values():
        assert len(tickers) >= 2
    clustered_tickers = {t for tickers in clusters.values() for t in tickers}
    assert len(clustered_tickers) < prices.shape[1]


def test_cluster_tickers_output_matches_sector_tickers_shape():
    prices = _synthetic_grouped_prices(seed=3, n_groups=2, group_size=3)

    clusters = cluster_tickers(prices, n_clusters=2, n_market_factors=1)

    assert isinstance(clusters, dict)
    for name, tickers in clusters.items():
        assert isinstance(name, str)
        assert isinstance(tickers, list)
        assert tickers == sorted(tickers)


def test_cluster_tickers_drops_rows_with_missing_data():
    prices = _synthetic_grouped_prices(seed=4, n_groups=2, group_size=3)
    prices_with_gap = prices.copy()
    prices_with_gap.iloc[5, 0] = np.nan

    clusters = cluster_tickers(prices_with_gap, n_clusters=2, n_market_factors=1)

    clustered_tickers = {t for tickers in clusters.values() for t in tickers}
    assert clustered_tickers <= set(prices.columns)
