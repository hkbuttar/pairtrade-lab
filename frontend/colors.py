"""Validated palette constants (dataviz skill's reference instance), the
same values bookmaker's frontend/colors.py uses. Color assignment follows
the skill's rules: categorical hues assigned by series identity in fixed
order (never cycled, never repainted when a filter changes which series
are visible), status colors reserved and never reused for a series.
"""

from __future__ import annotations

# Categorical, fixed order -- validated (adjacent-pair CVD/normal-vision
# floors clear in both light and dark). Never reorder per-chart; a series
# always gets the same slot everywhere it appears.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Fixed threshold-line -> slot assignment for the spread/z-score view, so
# the same line always means the same thing across every pair selected.
ENTRY_COLOR = CATEGORICAL[7]  # red: divergence, the trigger to enter
EXIT_COLOR = CATEGORICAL[5]  # green: reversion, the trigger to exit
STOP_LOSS_COLOR = CATEGORICAL[1]  # orange: continued divergence past tolerance
SPREAD_LINE_COLOR = CATEGORICAL[0]  # blue: the z-score series itself

LONG_MARKER_COLOR = CATEGORICAL[0]  # blue
SHORT_MARKER_COLOR = CATEGORICAL[6]  # violet

# Status palette (fixed, never themed, distinct from the categorical slots
# so a status color never impersonates a series).
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

MONITOR_STATUS_COLORS = {"ACTIVE": STATUS_GOOD, "HALTED": STATUS_CRITICAL}

# Chart chrome.
CHART_SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
