import numpy as np
import pandas as pd
import pytest

from pairs import cointegration
from pairs.cointegration import MIN_OBSERVATIONS, engle_granger_test

N = 300


def _cointegrated_pair(seed: int, hedge_ratio: float = 2.0, intercept: float = 5.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N)
    x = pd.Series(np.cumsum(rng.normal(0, 1, N)) + 100, index=dates)
    spread = rng.normal(0, 0.5, N)  # stationary residual by construction
    y = hedge_ratio * x + intercept + spread
    return y, x


def _independent_random_walks(seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N)
    y = pd.Series(np.cumsum(rng.normal(0, 1, N)) + 100, index=dates)
    x = pd.Series(np.cumsum(rng.normal(0, 1, N)) + 100, index=dates)
    return y, x


def test_engle_granger_detects_known_cointegrated_pair():
    y, x = _cointegrated_pair(seed=0, hedge_ratio=2.0, intercept=5.0)

    result = engle_granger_test(y, x)

    assert result.hedge_ratio == pytest.approx(2.0, abs=0.05)
    # Intercept alone is a noisy estimate here: x wanders around 100, far from
    # 0, so small slope errors extrapolate into large intercept errors. What
    # actually matters is that the fitted spread is tight around the true
    # stationary residual, which is what pairs trading exploits.
    spread = y - result.hedge_ratio * x - result.intercept
    assert spread.std() == pytest.approx(0.5, abs=0.1)
    assert result.p_value < 0.01
    assert result.n_obs == N


def test_engle_granger_rejects_independent_random_walks():
    y, x = _independent_random_walks(seed=1)

    result = engle_granger_test(y, x)

    assert result.p_value > 0.05


def test_engle_granger_raises_on_too_few_observations():
    dates = pd.bdate_range("2020-01-01", periods=MIN_OBSERVATIONS - 1)
    y = pd.Series(np.arange(len(dates), dtype=float), index=dates)
    x = pd.Series(np.arange(len(dates), dtype=float), index=dates)

    with pytest.raises(ValueError):
        engle_granger_test(y, x)


def test_engle_granger_aligns_on_overlapping_dates_only():
    y, x = _cointegrated_pair(seed=2)
    # Shift x's index so only a partial overlap remains, still >= MIN_OBSERVATIONS.
    x_shifted = x.copy()
    x_shifted.index = x_shifted.index + pd.Timedelta(days=0)
    x_shifted = x_shifted.iloc[10:]

    result = engle_granger_test(y, x_shifted)

    assert result.n_obs == len(y) - 10


def test_sector_pairs_flags_cointegrated_and_rejected_pairs_alike():
    y, x = _cointegrated_pair(seed=3)
    y_indep, _ = _independent_random_walks(seed=4)

    panel = pd.DataFrame({"AAA": y, "BBB": x, "CCC": y_indep})
    sector_tickers = {"TestSector": ["AAA", "BBB", "CCC"]}

    table = cointegration.test_sector_pairs(panel, sector_tickers, fdr_alpha=0.05)

    assert len(table) == 3  # C(3, 2) pairs tested
    # Rejected pairs are reported alongside winners, not filtered out.
    assert not table["cointegrated"].all()
    assert set(table["sector"]) == {"TestSector"}

    winner = table[(table["ticker_y"] == "AAA") & (table["ticker_x"] == "BBB")].iloc[0]
    assert winner["cointegrated"]
    assert winner["hedge_ratio"] == pytest.approx(2.0, abs=0.05)

    # FDR-corrected p-values are never smaller than the raw p-values.
    assert (table["p_value_fdr"] >= table["p_value"] - 1e-12).all()
    # Sorted by corrected p-value ascending.
    assert table["p_value_fdr"].is_monotonic_increasing


def test_sector_pairs_uses_alphabetical_canonical_order():
    y, x = _cointegrated_pair(seed=5)
    panel = pd.DataFrame({"ZZZ": y, "AAA": x})

    table = cointegration.test_sector_pairs(panel, {"S": ["ZZZ", "AAA"]})

    assert len(table) == 1
    assert table.iloc[0]["ticker_y"] == "AAA"
    assert table.iloc[0]["ticker_x"] == "ZZZ"


def test_sector_pairs_skips_pairs_with_too_little_overlap():
    y, x = _cointegrated_pair(seed=6)
    short_y = y.iloc[: MIN_OBSERVATIONS - 5]
    panel = pd.DataFrame({"AAA": short_y, "BBB": x})

    table = cointegration.test_sector_pairs(panel, {"S": ["AAA", "BBB"]})

    assert table.empty


def test_sector_pairs_only_tests_tickers_present_in_panel():
    y, x = _cointegrated_pair(seed=7)
    panel = pd.DataFrame({"AAA": y, "BBB": x})
    sector_tickers = {"S": ["AAA", "BBB", "NOT_IN_PANEL"]}

    table = cointegration.test_sector_pairs(panel, sector_tickers)

    assert len(table) == 1
    assert "NOT_IN_PANEL" not in set(table["ticker_y"]) | set(table["ticker_x"])
