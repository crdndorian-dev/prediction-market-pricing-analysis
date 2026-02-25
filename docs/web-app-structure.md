# Web app structure plan for a local pipeline UI/UX

Goal: propose a clear structure for building a local web application that keeps the current pipeline as the source of truth while offering a more intuitive interface to run steps, visualize outputs, and iterate faster.

---

## Overview (project root)

```
prediction-market-pricing-analysis/
├─ docs/
│  ├─ WEB_APP_STRUCTURE.md
│  └─ ui-ux/
│     ├─ user-flows.md
│     ├─ wireframes/
│     └─ design-notes.md
├─ src/
│  ├─ data/                 # Historical data / snapshots
│  ├─ notebooks/            # Exploration / prototypes
│  ├─ scripts/              # Existing pipeline scripts (source of truth)
│  └─ webapp/
│     ├─ backend/
│     │  ├─ app/
│     │  │  ├─ api/          # REST/JSON routes to drive the pipeline
│     │  │  ├─ core/         # Orchestration (pipeline execution, jobs)
│     │  │  ├─ models/       # Data schemas (pydantic/DTO)
│     │  │  ├─ services/     # Access to existing scripts + I/O
│     │  │  ├─ settings.py   # Config (paths, env, timeouts)
│     │  │  └─ __init__.py
│     │  ├─ tests/
│     │  └─ main.py          # Server entrypoint (FastAPI/Flask)
│     ├─ frontend/
│     │  ├─ public/
│     │  └─ src/
│     │     ├─ components/   # Reusable UI (charts, cards, tables)
│     │     ├─ pages/        # Screens (dashboard, pipeline, datasets)
│     │     ├─ hooks/        # Data fetching / state
│     │     ├─ api/          # Backend API client
│     │     ├─ styles/       # Design system (tokens, themes)
│     │     └─ App.tsx
│     ├─ shared/
│     │  ├─ schemas/         # Shared front/back schemas (JSON)
│     │  └─ constants/
│     └─ README.md           # Local app setup
└─ README.md
```

---

## Organization principles

### 1) **Do not duplicate the pipeline**
- Existing scripts remain the source of truth (`src/scripts/`).
- The backend exposes endpoints that *call* those scripts and normalize outputs.

### 2) **A clear API layer**
- `api/`: endpoints to run the pipeline, fetch status, and download outputs.
- `core/`: orchestration (local queue, long-running jobs, logs).

### 3) **Standardized outputs**
- `models/` and `shared/schemas/` describe expected outputs (e.g., `CalibrationResult`, `MarketSnapshot`).
- This keeps frontend data predictable and improves UX iteration.

### 4) **A frontend focused on iteration**
- `pages/` reflects the main flows: *Dashboard*, *Polymarket Snapshots*, *Datasets*, *Calibrate Models*.
- `components/` keeps reusable UI building blocks.

---

## Page-by-page feature plan (pipeline-first UX)

These pages should make the pipeline runnable end-to-end with minimal friction, clear status, and reproducible runs. Use progressive disclosure to keep defaults simple and advanced knobs tucked away.

### 1) Dashboard
Purpose: surface essential insights and link users into the right next action.
- Snapshot of latest pipeline status (last run time, success/fail, duration).
- Quick actions: "Fetch Polymarket Snapshot", "Build Dataset", "Calibrate Model".
- Key metrics: counts of snapshots, datasets, calibrated models; most recent versions.
- Alerts and warnings: missing inputs, stale data, failed jobs.
- Recent activity log with links to the originating page/run.
- Backlinks to the other three pages with short "what you can do here" captions.

### 2) Polymarket Snapshots
Purpose: fetch Polymarket data and run pHAT compute logic.
- Data source controls: market selection, time window, pagination, rate limits.
- Run controls: start, stop, retry, dry-run toggle.
- pHAT parameters: smoothing, filters, calibration flags (collapsed by default).
- Live run status with structured logs and error output.
- Output preview: raw snapshot table + computed pHAT summary (small charts).
- Persist run artifacts with run metadata (timestamp, params, git hash).
- Export/download snapshot outputs (CSV/JSON) and link to dataset builder.

### 3) Datasets
Purpose: build and manage historic options datasets with customizable parameters.
- Dataset builder form: date range, markets/strikes, resolution, filters.
- Input selection: choose snapshot runs or raw sources.
- Run controls with progress, logs, and resumable jobs.
- Dataset registry: list with versioning, size, date range, and status.
- Validation checks (missing intervals, outliers) with warnings.
- One-click handoff to "Calibrate Models" with selected dataset.

### 4) Calibrate Models
Purpose: run calibration, store models, and access prior results.
- Dataset picker with quick context (date range, size, last updated).
- Calibration config: model family, priors, optimization settings.
- Advanced settings (constraints, convergence criteria) collapsed by default.
- Run controls and live logs; highlight convergence or failure reasons.
- Model registry: list of calibrated models with metadata and metrics.
- Model details page/section: parameter table, diagnostics, pRN vs pHAT plots.
- Actions: compare models, export model artifacts, re-run with tweaks.

---

## Suggested UX flows (to document)

Create in `docs/ui-ux/user-flows.md`:
1. **Run a full pipeline (snapshot -> dataset -> calibration)**
2. **Re-run only pHAT with different parameters**
3. **Inspect and validate a historical dataset**
4. **Compare two model calibrations**

---

## Recommended next steps

1. Validate this structure plan (you + me).
2. Pick a minimal stack (e.g., FastAPI + React).
3. Create `src/webapp/README.md` with local startup commands.
4. Add a first API endpoint (e.g., `POST /run-calibration`).
