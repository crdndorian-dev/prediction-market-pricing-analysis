# Web app structure plan for a local pipeline UI/UX

Goal: propose a clear structure for building a local web application that keeps the current pipeline as the source of truth while offering a more intuitive interface to run steps, visualize outputs, and iterate faster.

---

## Overview (project root)

```
polyedgetool/
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
- `pages/` reflects the main flows: *Dashboard*, *Pipeline Runner*, *Results*, *Datasets*.
- `components/` keeps reusable UI building blocks.

---

## Suggested UX flows (to document)

Create in `docs/ui-ux/user-flows.md`:
1. **Run a pipeline**
2. **Compare pRN vs pHAT vs pPM**
3. **Inspect a historical dataset**
4. **Export or compare two runs**

---

## Recommended next steps

1. Validate this structure plan (you + me).
2. Pick a minimal stack (e.g., FastAPI + React).
3. Create `src/webapp/README.md` with local startup commands.
4. Add a first API endpoint (e.g., `POST /run-calibration`).
