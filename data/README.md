# data/

Price loading and caching. `prices.py` loads daily OHLCV bars from `yfinance`
(default, no key required) or Alpaca (`ALPACA_API_KEY`/`ALPACA_SECRET_KEY`,
reusing alpha-signal-lab's paper account), caching to parquet under
`data/cache/prices/` keyed by ticker.

`to_price_panel` pivots the long-format loader output into a wide
date-indexed panel (one column per ticker), the shape everything in `pairs/`,
`signals/`, and `monitor/` operates on.
