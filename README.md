# Prediction Market Pricing Analysis

A local webapp and research pipeline for comparing Polymarket-style binary prices to option-implied benchmarks.
Run the full data ingestion, calibration, and analysis workflow on your machine with reproducible outputs.

Author: Dorian Cardon  
Year: 2025-2026  
Contact: crdn.dorian@gmail.com  
GitHub: https://github.com/crdndorian-dev

# Abstract

This project develops a quantitative research framework designed to analyze the pricing structure and potential inefficiencies of Polymarket-style binary prediction markets. The objective is not to claim superior forecasting accuracy, but rather to explore whether prices observed in decentralized prediction markets are consistent with economically grounded benchmarks derived from traditional financial markets.

Polymarket contracts resemble digital options: they deliver a fixed payoff conditional on the realization of a binary event at a specified expiration date. Their prices can therefore be interpreted as implied probabilities. However, unlike exchange-listed derivatives, Polymarket prices are formed in environments characterized by heterogeneous beliefs, varying liquidity, retail participation, and limited arbitrage enforcement. As a result, quoted probabilities may reflect sentiment, positioning effects, or microstructure noise in addition to information about expected outcomes.

In contrast, listed equity options markets provide a rich cross-section of arbitrage-constrained prices. Under standard no-arbitrage assumptions, these prices encode risk-neutral probabilities that can be extracted using option pricing theory. While risk-neutral probabilities do not correspond directly to real-world probabilities due to risk premia and demand imbalances, they provide a coherent and economically disciplined reference point.

The central contribution of this project is to construct a systematic and time-safe pipeline that links these two environments. The framework first extracts risk-neutral probabilities from listed options data for events comparable to those traded on Polymarket. It then investigates, through statistical calibration methods, whether a stable relationship can be learned between risk-neutral measures and realized event frequencies. This mapping is subsequently used to generate fair-value probability estimates for Polymarket contracts, allowing for a structured comparison between decentralized market prices and derivative-implied benchmarks.

The project places particular emphasis on dataset design, temporal consistency, and robustness across different market regimes. Rather than assuming persistent inefficiency, it evaluates empirically whether deviations between Polymarket prices and calibrated benchmarks are economically meaningful, statistically significant, or largely attributable to noise. Backtesting procedures are implemented to assess whether observed discrepancies translate into consistent expected value under realistic assumptions.

# Technical stack

- Backend: FastAPI + Uvicorn (Python 3.11+).
- Frontend: React + TypeScript + Vite (Node 20+).
- Shared: JSON schemas in `src/webapp/shared/`.
- Data tooling: numpy, pandas, scipy, yfinance, requests.
- Storage: local files under `src/data/` and `data/`.

# Installation and setup

- Prerequisites: Python 3.11+, Node.js 20+ (Node 22 works too), and Java only if you plan to run the Theta Terminal data source.
- Environment (recommended for Polymarket pages): `cp config/polymarket_subgraph.env.sample .env`, set `GRAPH_API_KEY` in `.env`, and optionally override `POLYMARKET_SUBGRAPH_ID`, `ORDERBOOK_SUBGRAPH_ID`, `PNL_SUBGRAPH_ID`, or `POLYMARKET_SUBGRAPH_URL`.
- Quickstart (recommended): run the script and open the Vite URL it prints (usually `http://localhost:5173`). The script starts FastAPI on `http://localhost:8000` or the first free port in `8000-8050` and wires the frontend to that API.
  ```bash
  ./run-webapp.sh
  ```
- Manual setup (backend):
  ```bash
  cd src/webapp/backend
  python -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install fastapi uvicorn numpy pandas requests yfinance scipy
  uvicorn main:app --reload --port 8000
  ```
- Manual setup (frontend):
  ```bash
  cd src/webapp/frontend
  npm install
  VITE_API_BASE_URL="http://localhost:8000" npm run dev
  ```
- Notes: `run-webapp.sh` loads `.env` if present and otherwise falls back to `config/polymarket_subgraph.env.sample`. If you do not set `GRAPH_API_KEY`, Polymarket subgraph jobs will fail but the UI still loads. On Windows, use WSL or follow the manual setup steps. For more webapp details, see `src/webapp/README.md`.

# How to use

- Dashboard (`/`): check data freshness, run queue, and quick actions.
- Option Chain History Builder (`/option-chain`): build the pRN option-chain dataset and write outputs to `src/data/raw/option-chain`.
- Polymarket History Builder (`/polymarket-pipeline`): fetch markets, build `dim_market`, and run history jobs (requires `GRAPH_API_KEY`).
- Snapshot capture (`/polymarket`): generate the latest Polymarket snapshot files under `src/data/raw/polymarket`.
- Calibrate (`/calibrate-models`): train calibration models and write outputs to `src/data/models`.
- Markets (`/markets`): refresh and inspect market metadata and series summaries.
- Backtests (`/backtests`): experimental per-strike price explorer with pRN overlays.
- Documentation (`/docs`): in-app guides for every page, input, and output.
- Edge scoring (`/edge`, not linked in the nav yet): score a snapshot with a saved calibration and write outputs to `src/data/analysis/phat-edge`.

# Configuration

- `BACKEND_PORT`: backend port for `run-webapp.sh` (default 8000, auto-fallback to 8000-8050).
- `VITE_API_BASE_URL`: frontend API base.
- `MAX_ACTIVE_JOBS` and `VITE_MAX_ACTIVE_JOBS`: job concurrency caps.
- `GRAPH_API_KEY`: required for Polymarket subgraph pulls.
- `POLYMARKET_SUBGRAPH_ID`, `ORDERBOOK_SUBGRAPH_ID`, `PNL_SUBGRAPH_ID`, `POLYMARKET_SUBGRAPH_URL`: subgraph routing overrides.
- `THETA_TERMINAL_CMD`, `THETA_TERMINAL_JAR`, `THETA_TERMINAL_WORKDIR`, `THETA_TERMINAL_CREDS`, `THETA_TERMINAL_LOG`, `THETA_TERMINAL_STARTUP_WAIT`: optional Theta Terminal launcher settings.

# Data outputs

- `src/data/raw/option-chain`: option-chain history runs (pRN datasets).
- `src/data/raw/polymarket`: Polymarket snapshots, history, and subgraph runs.
- `src/data/models`: calibration models and `dim_market` outputs.
- `src/data/analysis/phat-edge`: edge scoring outputs.

# Future work

Future work includes expanding backtests with trade simulation, PnL metrics, and report exports, adding multi-ticker and multi-strike comparison views, introducing persistent storage for multi-user deployments, and adding production deployment assets (Docker, CI/CD) with authentication for shared environments.
