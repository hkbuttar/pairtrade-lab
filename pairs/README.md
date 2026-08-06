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

Johansen basket selection (3-5 leg baskets) and ML-assisted candidate
discovery (hierarchical clustering / PCA) on top of the hand-picked sector
groupings in `config/universe.py` are not yet implemented.
