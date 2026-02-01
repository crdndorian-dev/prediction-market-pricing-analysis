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
pip install fastapi uvicorn
```

Run the server:

```bash
uvicorn main:app --reload --port 8000
```

## Frontend setup

```bash
cd src/webapp/frontend
npm create vite@latest . -- --template react-ts
npm install
```

Run the dev server:

```bash
npm run dev
```

## Next steps

1. Add a `/health` route in `backend/app/api/` and wire it in `backend/main.py`.
2. Create a simple API client in `frontend/src/api/` and call `/health`.
3. Define the first schema in `shared/schemas/` for a pipeline run result.
