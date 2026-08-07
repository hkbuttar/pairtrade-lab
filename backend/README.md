# backend/

Optional thin FastAPI layer over the same modules the Panel dashboard uses
(`frontend/data_access.py`), for programmatic access beyond the Panel app.
Every endpoint just JSON-shapes data already computed elsewhere
(`pairs/`, `signals/`, `monitor/`, `backtest/`); if the underlying logic
changes, it changes there, not here, so the dashboard and this API can't
drift apart on what a status or a comparison metric means.

**Not part of the Render deployment.** `render.yaml` runs a single service
(`panel serve frontend/app.py`, per the project plan), since Panel already
serves its own frontend and there's no separate JS client that needs a
JSON API. This exists for local/programmatic use only.

Endpoints:
- `GET /health`
- `GET /api/pairs/significant?start=...&end=...` — every pair currently
  surviving FDR-corrected selection.
- `GET /api/pairs/{ticker_y}/{ticker_x}/status?start=...&end=...` —
  current monitor status, z-score, days since last break, rolling p-value.
- `GET /api/pairs/{ticker_y}/{ticker_x}/alerts?start=...&end=...` — every
  structural break detected for that pair.
- `GET /api/comparison` — the precomputed Step 9 comparison snapshot
  (`backtest/results/comparison_2018-01-01_2025-01-01.json`), not
  recomputed live (same reason as the dashboard's Comparisons tab: a full
  run takes ~2 minutes).

Run locally:

```bash
uvicorn backend.main:app --reload
```
