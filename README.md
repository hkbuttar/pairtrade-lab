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

**Pair selection: correlation vs. cointegration.** Two assets can be highly correlated in *returns* while never being cointegrated in *price levels*, and vice versa: a pair can have near-zero return correlation yet still share a common stochastic trend that pulls their prices back together. Correlation is about co-movement day to day; cointegration is about whether a linear combination of the price *levels* is stationary, i.e. mean-reverting. Pairs trading needs the latter, since a correlated-but-not-cointegrated spread has no statistical reason to revert. `pairs/cointegration.py` reports both on every tested pair so the two are never conflated.

**Engle-Granger two-step test** (`pairs/cointegration.py`): for each within-sector candidate pair, OLS-regress one price series on the other to get a hedge ratio, then run an Augmented Dickey-Fuller test on the regression residuals. The test isn't symmetric (regressing A on B can give a different p-value than B on A), so each pair is tested in one fixed, canonical direction (alphabetical) rather than both directions with the better p-value kept, which would be a form of data snooping.

**Multiple-comparisons correction**: testing dozens of pairs at a raw p < 0.05 threshold guarantees false positives by chance alone, so Benjamini-Hochberg FDR correction is applied across *every* pair tested in a run (`pairs.run_selection`), not per-sector. The output table shows every tested pair, both the ones that pass correction and the ones that don't, so the search space stays visible rather than only showing survivors.

**Basket selection: Johansen procedure** (`pairs/johansen.py`): extends the pairwise idea to 3-5 leg baskets. Johansen's test natively handles N series at once and its leading eigenvector *is* the cointegrating vector, so basket hedge weights fall out of the test itself rather than needing a follow-up regression. It also only returns critical values, not an exact p-value, so an approximate p-value (log-linear interpolation across the 90/95/99% critical values) is used to keep applying the same FDR correction as pairs; this is disclosed as an approximation, not treated as exact. Same discipline as pairs: every basket tested is reported, and FDR correction runs across the whole batch. 5-leg baskets are opt-in (not run by default) since the combinatorics grow fast; a live run over the full universe found 0/1252 3-leg baskets and 4/2930 4-leg baskets cointegrated after correction, results consistent with how few pairs (1/369) survived the same discipline.

**ML-assisted candidate discovery** (`pairs/clustering.py`): hierarchical clustering as an alternative to hand-picking sectors. Clustering directly on price-level correlation collapses almost the entire universe into one dominant cluster (everything shares the same broad-market trend; empirically the first principal component of standardized returns explains ~40% of the variance here), which is useless as a candidate generator. So the top principal component is regressed out of each ticker's standardized returns first (PCA-based factor extraction), and clustering runs on the *residual* correlation instead (Mantegna distance, average linkage). A live run's honest result: after removing the market factor, the clusters found were, with one exception, near-exact reproductions of the hand-picked sectors. The one significant pair was identical between both approaches. The one basket flagged as significant by the ML clusters but not the hand-picked sectors (AEP/WEC/XEL, all Utilities) had the *identical* raw p-value in both runs; it crossed the FDR threshold under the ML clustering (which tested 805 baskets total) but not under the hand-picked sectors (which tested 1252), purely a multiple-comparisons artifact of the two runs having different total test counts, not a different underlying finding. Reported as exactly that, not oversold as "ML found something sector search couldn't": on this universe and window, the hand-picked sector groupings hold up about as well as the ML-discovered ones.

**Hedge ratio estimation: static vs. Kalman** (`signals/hedge_ratio.py`): the static approach OLS-refits periodically and holds the hedge ratio fixed in between; the dynamic approach tracks `[hedge_ratio, intercept]` as a Kalman-filtered random walk, updated every observation, seeded from a static OLS fit rather than an arbitrary initial guess. The two are compared by out-of-sample one-step-ahead prediction error (using only the hedge ratio known as of t-1 to predict y_t from x_t), a narrower question than "which makes more backtested money" since that needs the event-driven simulator in `backtest/`, not yet built. Live result on BAC/PNC (the one pair that survived FDR correction): Kalman's prediction error was 65-84% lower than static's, tested across refit cadences from 10 to 252 days, so the finding isn't an artifact of picking an unfavorable static window. On this pair, the added complexity of the Kalman filter earned its keep, reported plainly since the honest answer here could just as easily have been "no meaningful difference."

Spread/signal construction, structural-break monitoring, and the block bootstrap procedure are tracked in `signals/`, `monitor/`, `backtest/`, and `risk/` and not yet implemented.

## Results
Not yet available; walk-forward backtest and bootstrap comparisons come after the selection, signal, and monitoring pipeline exist.

## Limitations
See Data above for data-layer limitations. Methodology-level limitations (cointegration can still break unpredictably even with monitoring, thin spreads are transaction-cost-sensitive, basket complexity-vs-benefit tradeoff, ML-discovered pairs may lack economic rationale even if statistically valid) will be documented here as each piece lands.

## Future Work
Capacity/transaction-cost-scaling analysis reusing execedge's impact modeling, live paper-trading extension consistent with alpha-signal-lab, regime-switching models.
