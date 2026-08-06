import pandas as pd

from data import prices


class _FakeBarsResult:
    def __init__(self, df):
        self.df = df


class _FakeStockHistoricalDataClient:
    def __init__(self, *args, **kwargs):
        pass

    def get_stock_bars(self, request):
        symbol = request.symbol_or_symbols
        dates = pd.bdate_range("2022-01-03", "2022-01-07", tz="UTC")
        idx = pd.MultiIndex.from_product([[symbol], dates], names=["symbol", "timestamp"])
        df = pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100},
            index=idx,
        )
        return _FakeBarsResult(df)


def _fake_downloader(calls):
    def _download(ticker, start, end):
        calls["count"] += 1
        dates = pd.bdate_range(start, end)
        return pd.DataFrame(
            {
                "date": dates,
                "ticker": ticker,
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
                "volume": 100,
            }
        )[prices.PRICE_COLUMNS]

    return _download


def test_load_prices_returns_expected_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(prices, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(prices, "_download_yfinance", _fake_downloader({"count": 0}))

    result = prices.load_prices(["AAA"], "2022-01-03", "2022-01-10", source="yfinance")

    assert list(result.columns) == prices.PRICE_COLUMNS
    assert (result["date"] >= pd.Timestamp("2022-01-03")).all()
    assert (result["date"] < pd.Timestamp("2022-01-10")).all()


def test_second_call_within_cached_range_skips_redownload(monkeypatch, tmp_path):
    monkeypatch.setattr(prices, "CACHE_DIR", tmp_path)
    calls = {"count": 0}
    monkeypatch.setattr(prices, "_download_yfinance", _fake_downloader(calls))

    prices.load_prices(["AAA"], "2022-01-03", "2022-01-14", source="yfinance")
    assert calls["count"] == 1

    prices.load_prices(["AAA"], "2022-01-04", "2022-01-08", source="yfinance")
    assert calls["count"] == 1  # fully covered by the cache written above


def test_disabling_cache_always_redownloads(monkeypatch, tmp_path):
    monkeypatch.setattr(prices, "CACHE_DIR", tmp_path)
    calls = {"count": 0}
    monkeypatch.setattr(prices, "_download_yfinance", _fake_downloader(calls))

    prices.load_prices(["AAA"], "2022-01-03", "2022-01-10", source="yfinance", use_cache=False)
    prices.load_prices(["AAA"], "2022-01-03", "2022-01-10", source="yfinance", use_cache=False)
    assert calls["count"] == 2


def test_download_alpaca_returns_expected_columns(monkeypatch):
    import alpaca.data.historical as alpaca_historical

    monkeypatch.setattr(
        alpaca_historical, "StockHistoricalDataClient", _FakeStockHistoricalDataClient
    )
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")

    result = prices._download_alpaca("AAA", "2022-01-03", "2022-01-10")

    assert list(result.columns) == prices.PRICE_COLUMNS
    assert not result.columns.duplicated().any()
    assert (result["ticker"] == "AAA").all()
    assert result["date"].dt.tz is None


def test_load_prices_alpaca_source_end_to_end(monkeypatch, tmp_path):
    import alpaca.data.historical as alpaca_historical

    monkeypatch.setattr(prices, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        alpaca_historical, "StockHistoricalDataClient", _FakeStockHistoricalDataClient
    )
    monkeypatch.setenv("ALPACA_API_KEY", "test")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test")

    result = prices.load_prices(["AAA"], "2022-01-03", "2022-01-10", source="alpaca")

    assert list(result.columns) == prices.PRICE_COLUMNS
    assert (result["date"] >= pd.Timestamp("2022-01-03")).all()
    assert (result["date"] < pd.Timestamp("2022-01-10")).all()


def test_to_price_panel_pivots_long_to_wide(monkeypatch, tmp_path):
    monkeypatch.setattr(prices, "CACHE_DIR", tmp_path)
    calls = {"count": 0}
    monkeypatch.setattr(prices, "_download_yfinance", _fake_downloader(calls))

    long_df = prices.load_prices(["AAA", "BBB"], "2022-01-03", "2022-01-06", source="yfinance")
    panel = prices.to_price_panel(long_df)

    assert set(panel.columns) == {"AAA", "BBB"}
    assert panel.index.name == "date"
    assert panel.columns.name is None
    assert (panel["AAA"] == 1.0).all()
