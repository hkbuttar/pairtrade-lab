from config.universe import SECTOR_TICKERS, TICKER_SECTOR, UNIVERSE


def test_universe_is_sorted_and_deduplicated():
    assert UNIVERSE == sorted(set(UNIVERSE))


def test_every_ticker_maps_to_exactly_one_sector():
    all_tickers = [t for tickers in SECTOR_TICKERS.values() for t in tickers]
    assert len(all_tickers) == len(set(all_tickers)), "a ticker appears in more than one sector"
    assert set(TICKER_SECTOR) == set(all_tickers)


def test_every_sector_has_enough_names_for_pair_search():
    # Cointegration search needs at least a handful of names per sector to be
    # a meaningful combinatorial search, not just an edge case.
    for sector, tickers in SECTOR_TICKERS.items():
        assert len(tickers) >= 5, f"{sector} has too few tickers for a pair search"
