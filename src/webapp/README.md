# Local Web App

## Stack

- Backend: FastAPI (Python) with Uvicorn
- Frontend: React + TypeScript (Vite)
- Shared: JSON schemas in `src/webapp/shared/`

## Prereqs

- Python 3.11+
- Node.js 20+

## Folder layout

- `backend/` API server (calls `src/scripts/` as the source of truth)
- `frontend/` UI (dashboard, pipeline runner, results)
- `shared/` Cross-cutting schemas and constants

## Backend setup

```bash
cd src/webapp/backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn main:app --reload --port 8000
```

## Job concurrency

By default, the web app allows multiple jobs to run at the same time. To cap
concurrency, set:

- Backend: `MAX_ACTIVE_JOBS` (set to `1` to restore single-job behavior)
- Frontend: `VITE_MAX_ACTIVE_JOBS` (set in `frontend/.env` to mirror the limit)

## Frontend setup

```bash
cd src/webapp/frontend
npm install
```

On macOS, if native Vite dependencies are quarantined after install, `run-webapp.sh`
automatically clears the quarantine flag for the local `esbuild` and `rollup`
binaries before starting the dev server.

Run the dev server:

```bash
npm run dev
```

## Analysis database

Create a PostgreSQL database for the research layer and expose it via:

```bash
POLYMARKET_ANALYSIS_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/polymarket_analysis
```

Set it in `.env`. `config/polymarket_analysis.env.sample` is a template only and is not auto-loaded by the backend.

## Polymarket analysis refresh

The `Data Analysis for Polymarket` page reads from the Postgres research store.
Refresh it by running:

```bash
python src/scripts/09-polymarket-analysis-refresh-v1.0.py
```

Useful flags:

- `--run-id <run-id>` to limit the import to selected weekly-history runs
- `--skip-trade-backfill` to avoid orderbook subgraph pulls
- `--force-full-rebuild` to clear and rebuild raw + mart + research tables

Notes:

- If `POLYMARKET_ANALYSIS_DATABASE_URL` is missing or points to an unreachable database, the analysis endpoints fail fast instead of silently falling back to a sample URL.
- The refresh script now requires core weekly-history artifacts such as `manifest.json`, `weekly_markets.csv`, and `price_history.csv` for each selected run.
- The dashboard is still usable without trade backfill, but volume-focused outputs are flagged as non-authoritative when coverage is incomplete.

The webapp also exposes this through `POST /analysis/refresh`.

## Local route

Once backend and frontend are running, open:

```text
http://localhost:5173/data-analysis
```

## Current focus

1. Weekly stock ladder market research and monitoring
2. Postgres-backed empirical thresholds and drift monitoring
3. Local notes and diagnostics for ongoing Polymarket analysis
