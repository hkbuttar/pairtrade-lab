"""Thin FastAPI layer over the same data/results modules the Panel
dashboard uses (frontend/data_access.py), for programmatic access beyond
the Panel app itself. Every endpoint here just JSON-shapes data already
computed elsewhere; if the underlying logic ever needs to change, change
it there (pairs/, signals/, monitor/, backtest/), not here, so the
dashboard and this API never drift apart on what "current z-score" or
"days since last break" actually means.

Optional per the project plan, and deliberately NOT part of the Render
deployment: Step 13's single-service plan runs only `panel serve`, since
Panel already serves its own frontend and there's no separate JS client
that needs a JSON API. This exists for local/programmatic use and as a
second interface that reuses the dashboard's own logic rather than
duplicating it.

Run locally: uvicorn backend.main:app --reload
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from frontend.data_access import (
    DEFAULT_END,
    DEFAULT_START,
    days_since_last_break,
    get_pair_monitor_data,
    get_significant_pairs,
    halt_events,
)

COMPARISON_RESULTS_PATH = (
    Path(__file__).resolve().parent.parent
    / "backtest"
    / "results"
    / "comparison_2018-01-01_2025-01-01.json"
)

app = FastAPI(title="PairTrade Lab API")

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    # Defaults wide open for local/exploratory use; tighten via
    # ALLOWED_ORIGINS (comma-separated) if this is ever deployed standalone.
    allow_origins=["*"] if _allowed_origins == "*" else _allowed_origins.split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/pairs/significant")
def significant_pairs(start: str = DEFAULT_START, end: str = DEFAULT_END) -> dict:
    """Every pair currently surviving FDR-corrected selection over [start, end)."""
    table = get_significant_pairs(start, end)
    return {"start": start, "end": end, "pairs": table.to_dict(orient="records")}


@app.get("/api/pairs/{ticker_y}/{ticker_x}/status")
def pair_status(
    ticker_y: str, ticker_x: str, start: str = DEFAULT_START, end: str = DEFAULT_END
) -> dict:
    """Current monitor status, z-score, and rolling p-value for one pair."""
    try:
        data = get_pair_monitor_data(ticker_y, ticker_x, start, end)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "pair": f"{ticker_y}/{ticker_x}",
        "status": data["status"].iloc[-1],
        "current_zscore": float(data["zscore"].dropna().iloc[-1]),
        "days_since_last_break": days_since_last_break(data["status"]),
        "rolling_pvalue": float(data["rolling_pvalue"].iloc[-1]),
    }


@app.get("/api/pairs/{ticker_y}/{ticker_x}/alerts")
def pair_alerts(
    ticker_y: str, ticker_x: str, start: str = DEFAULT_START, end: str = DEFAULT_END
) -> dict:
    """Every structural break detected for one pair: halt/resume dates and duration."""
    try:
        data = get_pair_monitor_data(ticker_y, ticker_x, start, end)
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    events = halt_events(data["status"])
    return {
        "pair": f"{ticker_y}/{ticker_x}",
        "halt_events": json.loads(events.to_json(orient="records", date_format="iso")),
    }


@app.get("/api/comparison")
def comparison() -> dict:
    """The precomputed Step 9 comparison snapshot (backtest/run_comparison.py
    --output-json). Not recomputed live; see backend module docstring and
    backtest/README.md for why.
    """
    if not COMPARISON_RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="No committed comparison results found")
    with open(COMPARISON_RESULTS_PATH) as f:
        return json.load(f)
