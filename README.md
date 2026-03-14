# Prediction Market Pricing Analysis

A local webapp and research pipeline for comparing Polymarket-style binary prices to option-implied benchmarks.
Run the full data ingestion, calibration, and analysis workflow on your machine with reproducible outputs.

## Project team

**Lead author**  
Name : Dorian Cardon  
Role : Project lead  
Year : 2025 - current  
Contact : crdn.dorian@gmail.com  
GitHub : https://github.com/crdndorian-dev  

**Collaborator**  
Name : Paul Mieussens  
Joined : March 2026  
GitHub : https://github/paulmatthewmieussens  

# Abstract

This project develops a quantitative research framework to analyze the pricing of certain finance-specific Polymarket binary prediction markets. It does not claim superior forecasting ability. Instead, it asks whether decentralized market prices align with economically grounded benchmarks derived from traditional financial markets.

Polymarket contracts resemble digital options, with fixed payoffs contingent on binary events at a given expiry. Their prices can therefore be interpreted as implied probabilities. Unlike exchange-listed derivatives, however, these markets typically operate with thinner liquidity and are more exposed to sentiment-driven flows. As a result, quoted probabilities may reflect microstructure noise and trading frictions as much as genuine information about outcomes.

Listed equity options, by contrast, are priced in markets governed by stronger arbitrage constraints. Under standard no-arbitrage assumptions, they embed risk-neutral probabilities that can be extracted using option pricing methods. Although such probabilities differ from real-world probabilities because of risk premia, inventory effects, and demand imbalances, they still provide a coherent and economically meaningful reference point.

The core contribution of the project is the construction of a systematic, time-safe pipeline linking these two market environments. The framework extracts risk-neutral probabilities from listed options for events comparable to Polymarket contracts, then uses logistic regression and machine learning techniques to estimate a stable mapping between risk-neutral measures and realized outcomes. This mapping is used to generate fair-value estimates that can be directly compared with decentralized market prices.

A central emphasis of the project is placed on dataset design, temporal consistency, and robustness of implementation. Its objective is to determine whether observed deviations between Polymarket prices and derivative-implied benchmarks are economically and statistically meaningful, or whether they are better understood as noise.

The current stage of development focuses on strengthening the calibration layer of the pipeline. Ongoing work is dedicated to enriching the feature set used to map risk-neutral probabilities into estimates that better match realized frequencies and refining the construction of the training dataset itself while improving the machine-learning logic governing model selection and validation. These efforts are intended to produce a more stable and better-calibrated relationship between derivative-implied information and eventual outcomes.  

Future work will shift from model construction toward market evaluation. In particular, the next objective is to assess whether statistically significant mispricings can be identified consistently in Polymarket contracts. To support this, a dedicated backtesting interface is planned, allowing the user to compare observed Polymarket prices with model-implied fair values through time and to evaluate whether any apparent pricing discrepancies would have translated into persistent and economically meaningful edge. The long-run ambition of the project is not merely to estimate probabilities, but to test whether prediction-market prices in this setting depart from financially grounded benchmarks in a repeatable way.

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

# Configuration

- `BACKEND_PORT`: backend port for `run-webapp.sh` (default 8000, auto-fallback to 8000-8050).
- `VITE_API_BASE_URL`: frontend API base.
- `MAX_ACTIVE_JOBS` and `VITE_MAX_ACTIVE_JOBS`: job concurrency caps.
- `GRAPH_API_KEY`: required for Polymarket subgraph pulls.
- `POLYMARKET_SUBGRAPH_ID`, `ORDERBOOK_SUBGRAPH_ID`, `PNL_SUBGRAPH_ID`, `POLYMARKET_SUBGRAPH_URL`: subgraph routing overrides.
- `THETA_TERMINAL_CMD`, `THETA_TERMINAL_JAR`, `THETA_TERMINAL_WORKDIR`, `THETA_TERMINAL_CREDS`, `THETA_TERMINAL_LOG`, `THETA_TERMINAL_STARTUP_WAIT`: optional Theta Terminal launcher settings.

# Script layout

- Maintained CLIs live under `src/scripts/entrypoints/`, with implementations grouped by stage under `src/scripts/`.  

# Future work

Future work includes : 
- expanding backtests page (right now, no real backtest can be performed).
- 2-staged hyperparameter search to improve calibration metrics (micro-adjustments of regularization coefficient : after evaluating initial log-spaced C grid and determining best candidate, test points around this value to assess existence of better value of C in surroundings)
