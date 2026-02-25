#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$ROOT_DIR/src/webapp/backend"
FRONTEND_DIR="$ROOT_DIR/src/webapp/frontend"

# Prefer a Homebrew Node that satisfies Vite's minimum version.
for node_prefix in \
  "/opt/homebrew/opt/node@22" \
  "/opt/homebrew/opt/node@20" \
  "/usr/local/opt/node@22" \
  "/usr/local/opt/node@20"; do
  if [[ -x "$node_prefix/bin/node" ]]; then
    export PATH="$node_prefix/bin:$PATH"
    break
  fi
done

# Load .env if present; otherwise fall back to the sample.
ENV_FILE="$ROOT_DIR/.env"
if [[ ! -f "$ENV_FILE" && -f "$ROOT_DIR/config/polymarket_subgraph.env.sample" ]]; then
  ENV_FILE="$ROOT_DIR/config/polymarket_subgraph.env.sample"
fi
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

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

is_port_free() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

find_free_port() {
  "$PYTHON_BIN" - <<'PY'
import socket
import sys

start = 8000
end = 8050

for port in range(start, end + 1):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError:
        continue
    finally:
        sock.close()
    print(port)
    sys.exit(0)

sys.exit(1)
PY
}

BACKEND_PORT="${BACKEND_PORT:-8000}"
if ! is_port_free "$BACKEND_PORT"; then
  echo "Port $BACKEND_PORT is in use. Selecting a free port..."
  BACKEND_PORT="$(find_free_port)"
  if [[ -z "${BACKEND_PORT:-}" ]]; then
    echo "No free port found in the 8000-8050 range." >&2
    exit 1
  fi
fi

start_backend() {
  (
    cd "$BACKEND_DIR"
    if [[ ! -d ".venv" ]]; then
      echo "Backend venv not found. Creating .venv..."
      "$PYTHON_BIN" -m venv .venv
    fi
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
    if ! python -c "import fastapi, uvicorn, numpy, pandas, requests, yfinance, scipy" >/dev/null 2>&1; then
      echo "Installing backend deps (fastapi, uvicorn, numpy, pandas, requests, yfinance, scipy)..."
      pip install -U pip
      pip install fastapi uvicorn numpy pandas requests yfinance scipy
    fi
    echo "Starting backend on http://localhost:$BACKEND_PORT"
    uvicorn main:app --reload --port "$BACKEND_PORT"
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
    VITE_API_BASE_URL="http://localhost:$BACKEND_PORT" npm run dev
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
