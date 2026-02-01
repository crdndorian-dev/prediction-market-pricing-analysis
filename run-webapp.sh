#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/src/webapp/backend"
FRONTEND_DIR="$ROOT_DIR/src/webapp/frontend"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python not found. Install Python 3.11+ and retry." >&2
    exit 1
  fi
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm not found. Install Node.js 20+ and retry." >&2
  exit 1
fi

pids=()

start_backend() {
  (
    cd "$BACKEND_DIR"
    if [[ ! -d ".venv" ]]; then
      echo "Backend venv not found. Creating .venv..."
      "$PYTHON_BIN" -m venv .venv
    fi
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
    if ! python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
      echo "Installing backend deps (fastapi, uvicorn)..."
      pip install -U pip
      pip install fastapi uvicorn
    fi
    echo "Starting backend on http://localhost:8000"
    uvicorn main:app --reload --port 8000
  ) &
  pids+=($!)
}

start_frontend() {
  (
    cd "$FRONTEND_DIR"
    if [[ ! -d "node_modules" ]]; then
      echo "Installing frontend deps..."
      npm install
    fi
    echo "Starting frontend dev server..."
    npm run dev
  ) &
  pids+=($!)
}

cleanup() {
  echo "Shutting down..."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid"
    fi
  done
}

trap cleanup INT TERM EXIT

start_backend
start_frontend

wait
