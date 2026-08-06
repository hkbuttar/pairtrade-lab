# PairTrade Lab — Statistical Arbitrage & Pairs Trading
Statistical arbitrage backtester for pairs and basket trading. Cointegration testing, Kalman-filter dynamic hedge ratios, and mean-reversion spread trading on real equities/crypto data, with walk-forward validation and honest reporting of what actually holds out-of-sample. CPU-only.

## Motivation
Every other project in this portfolio trades directionally: cross-sectional factors (alpha-signal-lab), market-making (bookmaker), execution (execedge), all bet on absolute price movement or liquidity provision. Statistical arbitrage is fundamentally different: it bets on the *relationship* between assets reverting to historical norms, market-neutral by construction. This project builds that from scratch, with the same discipline as the rest of the portfolio: real data, walk-forward validation, disclosed modeling assumptions, and honest reporting when something doesn't work.

## Data

**Asset class: US equities**, via `yfinance` for research/backtests (default, no key required) with an Alpaca Market Data path available (`source="alpaca"` in `data/prices.py`), reusing the same paper-trading credentials as alpha-signal-lab.

Equities were chosen over crypto because same-sector cointegration (utilities, banks, airlines) has a cleaner economic rationale to write about than crypto pairs, and it lets the universe build directly on alpha-signal-lab's `config/universe.py` rather than starting from scratch. Crypto's 24/7 data means no market-hours gaps to handle, but its correlation structure is driven far more by shared beta to a handful of majors than by the kind of sector-level economic linkage that gives a cointegration finding a real story.

**Universe** (`config/universe.py`): ~70 large/mid-cap US tickers across six sectors.
- `Technology`, `Healthcare`, `Financials`, `Energy` are the same constituent lists as alpha-signal-lab's universe, deliberately, so pairs discovered here can be cross-referenced against that project's factor scores.
- `Utilities` and `Airlines` are added on top: both are classic same-sector cointegration setups (regulated-monopoly utilities, network-effect airlines) that alpha-signal-lab's universe doesn't cover.

**Adjustments**: `yfinance` bars are loaded with `auto_adjust=True` (split- and dividend-adjusted). This matters more here than in a pure factor-momentum context: an unadjusted split shows up as a spurious jump in the price *level*, which can break the residual stationarity of a genuinely cointegrated pair even though nothing about the underlying relationship changed.

**Known limitations, disclosed up front**:
- Current constituent lists only, no delisted/acquired/renamed names included, so survivorship bias is present. A pair that looks cointegrated over history that includes a name still trading today is not evidence that pairs involving names that got delisted along the way would have behaved the same.
- No intraday data yet; daily bars only. Entry/exit signals and structural-break detection operate at daily resolution, which understates how fast a real break could be detected and reacted to intraday.
- Caching is local parquet under `data/cache/prices/` (gitignored), not versioned, so results are only exactly reproducible on a machine that has re-pulled the same date range from the same source.

## Methodology
See `pairs/`, `signals/`, `monitor/`, `backtest/`, and `risk/` (in progress) for cointegration/basket selection with FDR correction, static vs. dynamic hedge ratios, spread/signal construction, structural-break monitoring, and the block bootstrap procedure, as each is built out.

## Results
Not yet available; walk-forward backtest and bootstrap comparisons come after the selection, signal, and monitoring pipeline exist.

## Limitations
See Data above for data-layer limitations. Methodology-level limitations (cointegration can still break unpredictably even with monitoring, thin spreads are transaction-cost-sensitive, basket complexity-vs-benefit tradeoff, ML-discovered pairs may lack economic rationale even if statistically valid) will be documented here as each piece lands.

## Future Work
Capacity/transaction-cost-scaling analysis reusing execedge's impact modeling, live paper-trading extension consistent with alpha-signal-lab, regime-switching models.
