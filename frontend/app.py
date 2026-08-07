"""PairTrade Lab dashboard: Panel's param-reactive model, the third distinct
Python dashboard paradigm across this portfolio (alpha-signal-lab's
Streamlit script-rerun model, bookmaker/execedge's raw Bokeh server with
periodic DB-polling callbacks, and this project's param.Parameterized
widgets driving recomputation on change). Panel wraps Bokeh plots directly
(frontend/plots.py) and runs on Bokeh's server under the hood, so widget
interactions push updates rather than the browser polling.

There is no live paper-trading scheduler in this project (see README Future
Work), so "live" here means "recomputed against the latest cached price
data when a widget changes," not a continuous feed; every view says so
rather than implying more than it delivers.

Run with:
    panel serve frontend/app.py --show
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# panel/bokeh serve exec this file with only frontend/ on sys.path, not the
# repo root -- add the root so `frontend.*` imports below resolve, the same
# fix bookmaker's frontend/main.py uses for the same reason.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import panel as pn
import param

from frontend.data_access import (
    DEFAULT_END,
    DEFAULT_START,
    days_since_last_break,
    get_pair_monitor_data,
    get_significant_pairs,
    halt_events,
)
from frontend.plots import comparison_bar_plot, zscore_plot

pn.extension("tabulator", sizing_mode="stretch_width")

COMPARISON_RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "backtest"
    / "results"
    / "comparison_2018-01-01_2025-01-01.json"
)

COMPARISON_METRICS = [
    ("cagr", "CAGR"),
    ("sharpe_ratio", "Sharpe ratio"),
    ("max_drawdown", "Max drawdown"),
    ("win_rate", "Win rate"),
]


class Dashboard(param.Parameterized):
    pair = param.Selector(default=None, objects=[], label="Pair")

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.significant = get_significant_pairs(DEFAULT_START, DEFAULT_END)
        pair_options = {
            f"{row.ticker_y}/{row.ticker_x}": (row.ticker_y, row.ticker_x)
            for row in self.significant.itertuples()
        }
        self.param.pair.objects = pair_options
        if pair_options:
            self.pair = next(iter(pair_options.values()))

    def monitoring_table(self) -> pn.viewable.Viewable:
        if self.significant.empty:
            return pn.pane.Markdown("No FDR-significant pairs over this window.")

        rows = []
        for row in self.significant.itertuples():
            data = get_pair_monitor_data(row.ticker_y, row.ticker_x, DEFAULT_START, DEFAULT_END)
            status = data["status"].iloc[-1]
            zscore = data["zscore"].dropna().iloc[-1]
            since = days_since_last_break(data["status"])
            rows.append(
                {
                    "pair": f"{row.ticker_y}/{row.ticker_x}",
                    "status": status,
                    "current z-score": round(float(zscore), 2),
                    "days since last break": since if since is not None else "never halted",
                    "rolling p-value": round(float(data["rolling_pvalue"].iloc[-1]), 4),
                }
            )
        return pn.widgets.Tabulator(pd.DataFrame(rows), disabled=True, show_index=False)

    @param.depends("pair")
    def spread_plot_pane(self) -> pn.viewable.Viewable:
        if self.pair is None:
            return pn.pane.Markdown("No FDR-significant pairs over this window.")
        ticker_y, ticker_x = self.pair
        data = get_pair_monitor_data(ticker_y, ticker_x, DEFAULT_START, DEFAULT_END)
        return pn.pane.Bokeh(zscore_plot(data, ticker_y, ticker_x))

    @param.depends("pair")
    def alert_feed(self) -> pn.viewable.Viewable:
        if self.pair is None:
            return pn.pane.Markdown("No FDR-significant pairs over this window.")
        ticker_y, ticker_x = self.pair
        data = get_pair_monitor_data(ticker_y, ticker_x, DEFAULT_START, DEFAULT_END)
        events = halt_events(data["status"])
        if events.empty:
            return pn.pane.Markdown(
                f"No structural breaks detected for {ticker_y}/{ticker_x} over this window."
            )
        return pn.widgets.Tabulator(events, disabled=True, show_index=False)


def comparison_view() -> pn.viewable.Viewable:
    if not COMPARISON_RESULTS_PATH.exists():
        return pn.pane.Markdown(
            "No committed comparison results found. Run "
            "`python -m backtest.run_comparison --start ... --end ... --output-json ...` first."
        )
    with open(COMPARISON_RESULTS_PATH) as f:
        comparison = json.load(f)

    plots = [
        pn.pane.Bokeh(comparison_bar_plot(comparison, key, label))
        for key, label in COMPARISON_METRICS
    ]
    note = pn.pane.Markdown(
        f"Precomputed snapshot: {comparison['start']} to {comparison['end']}, "
        f"{comparison['n_resamples']} bootstrap resamples, {comparison['block_length']}-day blocks "
        f"(seed={comparison['seed']}). Not recomputed live here (a full four-way comparison run "
        "takes roughly two minutes); see `backtest/run_comparison.py` and "
        "`backtest/README.md` for the full write-up behind these numbers, including which "
        "differences are and aren't statistically distinguishable."
    )
    return pn.Column(note, pn.GridBox(*plots, ncols=2))


def build_app() -> pn.template.FastListTemplate:
    dashboard = Dashboard()
    pair_select = pn.Param(
        dashboard.param.pair, widgets={"pair": pn.widgets.Select}, show_name=False
    )

    monitoring_tab = pn.Column(
        pn.pane.Markdown(
            "## Live monitoring\n"
            "Cointegration status for every pair that currently survives FDR-corrected "
            "selection over the loaded window, from the same rolling re-test + CUSUM "
            "monitor as `monitor/structural_break.py`. Recomputed against the latest "
            "cached price data on load, not a continuous feed (this project has no live "
            "paper-trading scheduler yet, see the README's Future Work)."
        ),
        dashboard.monitoring_table(),
    )

    spread_tab = pn.Column(
        pn.pane.Markdown(
            "## Spread & z-score\n"
            "Rolling z-score with entry (red)/exit (green)/stop-loss (orange) reference "
            "lines, trade-entry markers, and shaded regions where structural-break "
            "monitoring halted the pair."
        ),
        pair_select,
        dashboard.spread_plot_pane,
    )

    comparison_tab = pn.Column(
        pn.pane.Markdown("## Comparison dashboard"),
        comparison_view(),
    )

    alerts_tab = pn.Column(
        pn.pane.Markdown(
            "## Alert feed\nEvery structural break detected for the selected pair: "
            "when it halted, when (if ever) it requalified, and how long the halt lasted."
        ),
        pair_select,
        dashboard.alert_feed,
    )

    tabs = pn.Tabs(
        ("Live Monitoring", monitoring_tab),
        ("Spread & Z-Score", spread_tab),
        ("Comparisons", comparison_tab),
        ("Alert Feed", alerts_tab),
    )

    return pn.template.FastListTemplate(
        title="PairTrade Lab",
        theme_toggle=True,
        main=[tabs],
    )


build_app().servable()
