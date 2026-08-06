# pairs/

Cointegration testing and pair/basket selection.

`cointegration.py` implements the Engle-Granger two-step test for pairs:
OLS-regress one price series on another (canonical alphabetical order, fixed
per pair rather than tried both ways, to avoid data-snooping the more
favorable direction), then run an Augmented Dickey-Fuller test on the
residuals. `test_sector_pairs` runs this across every within-sector
combination in a universe, applies Benjamini-Hochberg FDR correction across
*all* pairs tested (not per-sector), and returns every tested pair, winners
and rejected pairs alike, ranked by corrected p-value. It also reports each
pair's return correlation alongside its cointegration p-value, since the two
can disagree.

`run_selection.py` is the CLI entry point:

```bash
python -m pairs.run_selection --start 2018-01-01 --end 2025-01-01
```

`johansen.py` extends the same idea to 3-5 leg baskets via the Johansen
procedure (`statsmodels.tsa.vector_ar.vecm.coint_johansen`), which tests
cointegration rank across N series at once; its leading eigenvector doubles
directly as the basket's hedge weights, no separate OLS step needed the way
pairs need one. Since `coint_johansen` only returns critical values (90/95/99%)
rather than an exact p-value, `johansen_test` approximates one by log-linear
interpolation across those three points, disclosed in the module docstring as
an approximation rather than an exact result. `test_sector_baskets` applies
the same Benjamini-Hochberg FDR discipline as `test_sector_pairs`, across all
baskets tested in one batch, and reports rejects alongside winners.

`run_basket_selection.py` is the CLI entry point (defaults to 3- and 4-leg
baskets; 5-leg is opt-in via `--basket-sizes` since it multiplies the search
space considerably):

```bash
python -m pairs.run_basket_selection --start 2018-01-01 --end 2025-01-01
```

Both CLI scripts print the full tested table by default (winners and
rejects). `--only-significant` restricts what's printed to rows that passed
FDR correction, useful once the universe gets large enough that the full
table floods the terminal; `--output-csv <path>` saves the full table (both
winners and rejects, independent of `--only-significant`) regardless. Shared
in `cli_output.py`.

`clustering.py` implements ML-assisted candidate discovery: hierarchical
clustering on price series, as an alternative to the hand-picked sector
groupings in `config/universe.py`. Clustering directly on price-level
correlation collapses almost the whole universe into one dominant cluster
(everything shares the same broad-market trend; empirically the first
principal component of standardized returns explains about 40% of the
variance in this universe), which is useless as a candidate generator. So
`cluster_tickers` first regresses the top principal component(s) out of each
ticker's standardized returns (PCA-based factor extraction), then clusters
the *residual* correlation (Mantegna 1999 distance, average linkage). The
result is a `{cluster_name: tickers}` mapping in the exact same shape as
`config.universe.SECTOR_TICKERS`, so `test_sector_pairs` and
`test_sector_baskets` consume it with no changes.

`run_ml_selection.py` runs the full pipeline on both the ML-discovered
clusters and the hand-picked sectors and reports the comparison plainly:

```bash
python -m pairs.run_ml_selection --start 2018-01-01 --end 2025-01-01
```

What a live run over the full universe actually found: after removing the
market factor, the clusters produced were, with one exception, near-exact
reproductions of the hand-picked sectors (Utilities, Energy, Technology,
Financials, Airlines each came back as their own clean cluster). The one
surviving pair (BAC/PNC) was found by both approaches identically. The one
basket the ML run flagged as significant that the hand-picked run didn't
(AEP/WEC/XEL, all Utilities) turned out to have the *identical* raw p-value
in both runs (0.000059); it crossed the FDR threshold in the ML run
(p_fdr=0.048) and not the hand-picked run (p_fdr=0.074) purely because the
two runs tested a different total number of baskets (805 vs. 1252), so the
correction's stringency differed, not because the candidate itself was
different. Reported as exactly that rather than oversold as "ML discovered
a hidden pair": on this universe and window, ML clustering mostly rediscovers
the sector structure that was already economically obvious, and the one
apparent discrepancy is a multiple-comparisons artifact, not a new signal.
