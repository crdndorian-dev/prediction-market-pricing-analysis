# Polymarket Pipeline Quick Run

## Prereqs
- Python 3.11+ and Node.js 20+ installed.
- Optional: copy `config/polymarket_subgraph.env.sample` to `.env` and fill in credentials.

## Start The Webapp
1. From the repo root, run:
   ```bash
   ./run-webapp.sh
   ```
2. Open the frontend URL printed by the script (Vite default is usually `http://localhost:5173`).

## Run The Pipeline In Order
Run each stage from the left nav in this order. Each page has a Run/Jobs section plus optional previews.

1. Subgraph Ingest\n+   - What it does: pulls Polymarket trades and market data from the subgraph into `src/data/raw/polymarket/...`.\n+   - How to run (webapp): open `Subgraph`, set query/run options (or leave defaults), then Run/Jobs.\n+   - How to run (CLI): `python src/scripts/6-polymarket-subgraph-ingest-trades-v1.0.py` with your flags.\n+\n+2. Market Map\n+   - What it does: builds `dim_market.parquet` mapping tickers to Polymarket market IDs.\n+   - How to run (webapp): open `Market Map`, set overrides/tickers if needed, then Run/Jobs.\n+   - How to run (CLI): `python src/scripts/5-polymarket-market-map-v1.0.py`.\n+\n+3. Build Bars\n+   - What it does: aggregates raw trades into OHLC bars at multiple frequencies in `src/data/analysis/polymarket/bars/`.\n+   - How to run (webapp): open `Bars`, set frequencies/date range, then Run/Jobs.\n+   - How to run (CLI): `python src/scripts/7-polymarket-build-bars-v1.0.py`.\n+\n+4. Build Features\n+   - What it does: creates `decision_features.parquet` and `feature_manifest.json` (anti-leak enforced).\n+   - How to run (webapp): open `Features`, set decision frequency and windows, then Run/Jobs.\n+   - How to run (CLI): `python src/scripts/8-polymarket-build-features-v1.0.py`.\n+\n+5. Train Model\n+   - What it does: trains the mixed Polymarket + pRN model and writes outputs under `src/data/models/mixed/`.\n+   - How to run (webapp): open `Train`, choose model type and settings, then Run/Jobs.\n+   - How to run (CLI): `python src/scripts/9-train-mixed-model-v1.0.py`.\n+\n+6. Backtests\n+   - What it does: runs historical backtests against the trained model(s).\n+   - How to run (webapp): open `Backtests`, select model/run and settings, then Run/Jobs.\n+   - How to run (CLI): `python src/scripts/10-backtest-polymarket-v1.0.py`.\n+\n+7. Signals\n+   - What it does: generates current trading signals and writes outputs under `src/data/analysis/polymarket/`.\n+   - How to run (webapp): open `Signals`, configure inputs, then Run/Jobs.\n+   - How to run (CLI): `python src/scripts/11-generate-signals-v1.0.py`.\n*** End Patch

## CLI Alternative (Optional)
If you prefer CLI, run the scripts in order from `src/scripts/` with your desired flags:

1. `6-polymarket-subgraph-ingest-trades-v1.0.py`
2. `5-polymarket-market-map-v1.0.py`
3. `7-polymarket-build-bars-v1.0.py`
4. `8-polymarket-build-features-v1.0.py`
5. `9-train-mixed-model-v1.0.py`
6. `10-backtest-polymarket-v1.0.py`
7. `11-generate-signals-v1.0.py`
