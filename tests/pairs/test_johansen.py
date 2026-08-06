import numpy as np
import pandas as pd
import pytest

from pairs import johansen
from pairs.cointegration import MIN_OBSERVATIONS
from pairs.johansen import johansen_test

N = 300


def _rank_one_basket(seed: int):
    """Three series driven by two independent random-walk trends, so exactly
    one stationary combination exists: a + b - c is stationary by construction.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N)
    w1 = np.cumsum(rng.normal(0, 1, N))
    w2 = np.cumsum(rng.normal(0, 1, N))
    a = pd.Series(w1 + rng.normal(0, 0.3, N) + 100, index=dates)
    b = pd.Series(w2 + rng.normal(0, 0.3, N) + 100, index=dates)
    c = pd.Series(w1 + w2 + rng.normal(0, 0.3, N) + 100, index=dates)
    return pd.DataFrame({"a": a, "b": b, "c": c})


def _independent_walks_basket(seed: int, n_legs: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N)
    data = {
        chr(ord("a") + i): pd.Series(np.cumsum(rng.normal(0, 1, N)) + 100, index=dates)
        for i in range(n_legs)
    }
    return pd.DataFrame(data)


def test_johansen_detects_known_rank_one_basket():
    prices = _rank_one_basket(seed=0)

    result = johansen_test(prices)

    assert result.p_value < 0.01
    assert result.n_obs == N
    # Recovered weights should be proportional to the true [1, 1, -1] relation
    # (a + b - c is the stationary combination by construction).
    assert result.weights["a"] == pytest.approx(1.0, abs=0.15)
    assert result.weights["b"] == pytest.approx(1.0, abs=0.15)
    assert result.weights["c"] == pytest.approx(-1.0, abs=0.15)


def test_johansen_does_not_reject_independent_walks():
    prices = _independent_walks_basket(seed=1)

    result = johansen_test(prices)

    assert result.p_value > 0.05


def test_johansen_raises_on_too_few_observations():
    dates = pd.bdate_range("2020-01-01", periods=MIN_OBSERVATIONS - 1)
    prices = pd.DataFrame(
        {
            "a": np.arange(len(dates), dtype=float),
            "b": np.arange(len(dates), dtype=float) * 2,
            "c": np.arange(len(dates), dtype=float) * 3,
        },
        index=dates,
    )

    with pytest.raises(ValueError):
        johansen_test(prices)


def test_approximate_trace_p_value_is_monotonically_decreasing():
    critical_values = np.array([27.07, 29.80, 35.46])

    low_stat_p = johansen._approximate_trace_p_value(20.0, critical_values)
    mid_stat_p = johansen._approximate_trace_p_value(29.80, critical_values)
    high_stat_p = johansen._approximate_trace_p_value(60.0, critical_values)

    assert low_stat_p > mid_stat_p > high_stat_p
    assert mid_stat_p == pytest.approx(0.05, abs=0.01)
    assert 0.0 <= high_stat_p <= 1.0


def test_sector_baskets_flags_cointegrated_and_rejected_baskets_alike():
    cointegrated = _rank_one_basket(seed=2)
    independent = _independent_walks_basket(seed=3, n_legs=3).rename(
        columns={"a": "d", "b": "e", "c": "f"}
    )
    panel = pd.concat([cointegrated, independent], axis=1)
    sector_tickers = {"TestSector": ["a", "b", "c", "d", "e", "f"]}

    table = johansen.test_sector_baskets(panel, sector_tickers, basket_sizes=(3,), fdr_alpha=0.05)

    assert (table["basket_size"] == 3).all()
    assert not table["cointegrated"].all()  # rejected baskets are reported too

    winner = table[table["tickers"] == ("a", "b", "c")].iloc[0]
    assert winner["cointegrated"]

    loser = table[table["tickers"] == ("d", "e", "f")].iloc[0]
    assert not loser["cointegrated"]

    assert (table["p_value_fdr"] >= table["p_value"] - 1e-12).all()
    assert table["p_value_fdr"].is_monotonic_increasing


def test_sector_baskets_only_tests_tickers_present_in_panel():
    cointegrated = _rank_one_basket(seed=4)
    sector_tickers = {"S": ["a", "b", "c", "NOT_IN_PANEL"]}

    table = johansen.test_sector_baskets(cointegrated, sector_tickers, basket_sizes=(3,))

    assert len(table) == 1
    assert table.iloc[0]["tickers"] == ("a", "b", "c")


def test_sector_baskets_skips_baskets_with_too_little_overlap():
    prices = _rank_one_basket(seed=5)
    prices_short = prices.copy()
    prices_short.loc[prices_short.index[: -(MIN_OBSERVATIONS - 5)], "a"] = np.nan

    table = johansen.test_sector_baskets(prices_short, {"S": ["a", "b", "c"]}, basket_sizes=(3,))

    assert table.empty
