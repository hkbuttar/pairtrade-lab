"""Bokeh figure builders for the Panel dashboard. Panel wraps these directly
via pn.pane.Bokeh, reusing this portfolio's existing Bokeh plotting fluency
(bookmaker, execedge) for the actual charts, while Panel's reactivity and
layout are what's new to this project.

Color follows the dataviz skill's rules: categorical hues assigned by
identity in a fixed order (never cycled, never repainted when the selected
pair changes), status colors (frontend.colors.MONITOR_STATUS_COLORS)
reserved for ACTIVE/HALTED and never reused for a series.
"""

from __future__ import annotations

import pandas as pd
from bokeh.models import BoxAnnotation, ColumnDataSource, HoverTool, Span, Whisker
from bokeh.plotting import figure

from frontend import colors
from frontend.data_access import halt_events

TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


def _style_figure(fig: figure) -> None:
    fig.background_fill_color = colors.CHART_SURFACE
    fig.border_fill_color = colors.PAGE_PLANE
    fig.grid.grid_line_color = colors.GRIDLINE
    fig.axis.axis_line_color = colors.BASELINE
    fig.axis.major_label_text_color = colors.TEXT_SECONDARY
    fig.axis.axis_label_text_color = colors.TEXT_SECONDARY
    fig.title.text_color = colors.TEXT_PRIMARY
    fig.axis.axis_label_text_font = colors.FONT
    fig.axis.major_label_text_font = colors.FONT


def zscore_plot(
    monitor_data: dict,
    ticker_y: str,
    ticker_x: str,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss_threshold: float = 3.5,
) -> figure:
    """Rolling z-score with entry/exit/stop-loss reference lines, trade entry
    markers, and shaded HALTED regions from structural-break monitoring.
    """
    zscore = monitor_data["zscore"].dropna()
    trades = monitor_data["trades"]

    fig = figure(
        title=f"{ticker_y} ~ {ticker_x}: spread z-score",
        x_axis_type="datetime",
        height=380,
        sizing_mode="stretch_width",
        tools=TOOLS,
        toolbar_location="above",
    )
    _style_figure(fig)

    for halt in halt_events(monitor_data["status"]).itertuples():
        end = halt.resumed_at if pd.notna(halt.resumed_at) else zscore.index[-1]
        fig.add_layout(
            BoxAnnotation(
                left=halt.halted_at,
                right=end,
                fill_color=colors.MONITOR_STATUS_COLORS["HALTED"],
                fill_alpha=0.08,
                line_color=None,
            )
        )

    source = ColumnDataSource({"date": zscore.index, "zscore": zscore.to_numpy()})
    line = fig.line(
        "date",
        "zscore",
        source=source,
        line_width=2,
        color=colors.SPREAD_LINE_COLOR,
        legend_label="z-score",
    )
    fig.add_tools(
        HoverTool(
            renderers=[line],
            tooltips=[("date", "@date{%F}"), ("z-score", "@zscore{0.00}")],
            formatters={"@date": "datetime"},
        )
    )

    threshold_lines = [
        (entry_threshold, colors.ENTRY_COLOR),
        (-entry_threshold, colors.ENTRY_COLOR),
        (exit_threshold, colors.EXIT_COLOR),
        (-exit_threshold, colors.EXIT_COLOR),
        (stop_loss_threshold, colors.STOP_LOSS_COLOR),
        (-stop_loss_threshold, colors.STOP_LOSS_COLOR),
    ]
    for value, color in threshold_lines:
        fig.add_layout(
            Span(
                location=value,
                dimension="width",
                line_color=color,
                line_dash="dashed",
                line_width=1,
            )
        )

    if not trades.empty:
        for side, marker_color, marker in [
            ("long", colors.LONG_MARKER_COLOR, "triangle"),
            ("short", colors.SHORT_MARKER_COLOR, "inverted_triangle"),
        ]:
            side_trades = trades[trades["side"] == side]
            if side_trades.empty:
                continue
            entry_z = zscore.reindex(side_trades["entry_date"])
            entry_source = ColumnDataSource(
                {"date": side_trades["entry_date"], "zscore": entry_z.to_numpy()}
            )
            fig.scatter(
                "date",
                "zscore",
                source=entry_source,
                size=10,
                marker=marker,
                color=marker_color,
                legend_label=f"{side} entry",
            )

    fig.legend.location = "top_left"
    fig.legend.click_policy = "hide"
    fig.legend.background_fill_color = colors.CHART_SURFACE
    fig.legend.label_text_color = colors.TEXT_SECONDARY
    fig.yaxis.axis_label = "z-score"
    return fig


def comparison_bar_plot(comparison: dict, metric: str, metric_label: str) -> figure:
    """Point estimate + 95% CI whiskers for one metric, across the baseline
    and its three Step 9 variants. Bar colors follow the categorical slot
    order, one fixed slot per run identity across every metric's chart.
    """
    runs = ["baseline", "static_hedge_ratio", "reactive_only", "ml_clusters"]
    labels = {
        "baseline": "Baseline",
        "static_hedge_ratio": "Static hedge ratio",
        "reactive_only": "Reactive-only",
        "ml_clusters": "ML clusters",
    }
    run_colors = colors.CATEGORICAL[:4]

    names = [labels[r] for r in runs]
    points = [comparison[r]["metrics"][metric] for r in runs]
    ci_low = [
        comparison[r]["ci"][metric]["ci_low"] if comparison[r]["ci"] else p
        for r, p in zip(runs, points, strict=True)
    ]
    ci_high = [
        comparison[r]["ci"][metric]["ci_high"] if comparison[r]["ci"] else p
        for r, p in zip(runs, points, strict=True)
    ]

    source = ColumnDataSource(
        {"name": names, "point": points, "ci_low": ci_low, "ci_high": ci_high, "color": run_colors}
    )

    fig = figure(
        x_range=names,
        title=metric_label,
        height=320,
        sizing_mode="stretch_width",
        tools=TOOLS,
        toolbar_location="above",
    )
    _style_figure(fig)

    bars = fig.vbar(x="name", top="point", width=0.6, source=source, color="color", line_color=None)
    fig.add_tools(
        HoverTool(renderers=[bars], tooltips=[("run", "@name"), (metric_label, "@point{0.0000}")])
    )

    whisker = Whisker(
        base="name",
        upper="ci_high",
        lower="ci_low",
        source=source,
        line_color=colors.TEXT_SECONDARY,
    )
    whisker.upper_head.size = 8
    whisker.lower_head.size = 8
    fig.add_layout(whisker)

    fig.xgrid.grid_line_color = None
    fig.y_range.start = min(0, min(ci_low)) * 1.1 if min(ci_low) < 0 else 0
    return fig
