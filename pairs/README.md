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

ML-assisted candidate discovery (hierarchical clustering / PCA) on top of the
hand-picked sector groupings in `config/universe.py` is not yet implemented.
