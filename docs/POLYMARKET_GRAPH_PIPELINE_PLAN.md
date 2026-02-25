# Polymarket Subgraph Backtesting + Trading Plan (Codex-Ready)

**Goal**
Build a production-minded pipeline that uses The Graph (Polymarket subgraph) to backtest, train a mixed pRN + Polymarket model, simulate trades with realistic constraints, and surface real edges without data leakage. This plan is designed to fit the existing polyedgetool pipeline and webapp.

**Scope**
- Ingest Polymarket on-chain events from The Graph.
- Normalize to fact tables and generate time-safe features.
- Join with your pRN dataset to train mixed models.
- Run a realistic backtest that uses bid/ask or conservative slippage.
- Add webapp support for ingestion, datasets, backtests, and signals.

**Non-goals**
- This document does not include live order placement implementation details. It provides a safe phased path and integration points.

---

## 0. Truth Layers and Anti-Leak Contract

Define these two truths and never mix them:

1. Settlement truth: on-chain events (fills, positions, liquidity, resolution).
2. Decision truth: best bid/ask or mid at decision time (CLOB). If historical orderbook is unavailable, use trade-derived mid with a conservative slippage model.

**Anti-leak rules**
- All features for a decision time `t` must be computed using data strictly before `t`.
- Labels use resolution outcome at expiry or settlement time only.
- Backtests must include latency and spread/slippage.

---

## 1. Add Graph Access and a Deterministic Subgraph Client

**Files to add**
- `src/scripts/polymarket/subgraph_client.py`
- `src/scripts/polymarket/graphql_queries.py`
- `config/polymarket_subgraph.env.sample`

**Env vars**
- `GRAPH_API_KEY`
- `POLYMARKET_SUBGRAPH_ID`
- `POLYMARKET_SUBGRAPH_URL` (optional override)

**Client requirements**
- Retries with exponential backoff.
- Pagination with `first` + `skip` or cursor if supported.
- Optional block or time filters where schema supports them.
- Deterministic pulls: write raw JSON responses to disk and log query variables.
- Strict rate limiting to avoid gateway throttling.

**Raw data storage**
- `src/data/raw/polymarket/subgraph/runs/<run-id>/` with:
- `manifest.json` (query, variables, timestamps, schema version, subgraph ID)
- `raw/` folder containing one file per page or per entity.

**Deliverable**
- `subgraph_client.py` used by all subgraph ingestion scripts.

---

## 2. Market Identity Mapping Layer (Critical)

**Purpose**
Create a stable join key between Polymarket markets and your internal instruments (ticker, threshold, expiry). This prevents mismatched backtests.

**Files to add**
- `src/scripts/5-polymarket-market-map-v1.0.py`
- `config/polymarket_market_overrides.csv`
- `src/data/models/polymarket/dim_market.parquet`

**dim_market columns**
- `market_id` or `condition_id`
- `question`
- `ticker`
- `threshold`
- `expiry_date_utc`
- `resolution_time_utc`
- `outcome_yes_token_id`
- `outcome_no_token_id`
- `slug`
- `source` (auto vs manual)
- `mapping_confidence` (0-1)

**Mapping rules**
- Use regex parsing from `question` and `slug`.
- Provide manual overrides in `config/polymarket_market_overrides.csv`.
- Fail fast if thresholds or tickers cannot be parsed and no manual override exists.

**Deliverable**
- `dim_market.parquet` or CSV with a stable join key and explicit overrides.

---

## 3. Ingest Minimum Viable Historical Data

### 3A. Trades / Fills

**Files to add**
- `src/scripts/6-polymarket-subgraph-ingest-trades-v1.0.py`

**fact_trade columns**
- `trade_id`
- `block_number`
- `timestamp_utc`
- `market_id`
- `outcome` (YES or NO)
- `price`
- `size`
- `side` (buy or sell if derivable)
- `tx_hash`

**Storage**
- `src/data/raw/polymarket/fact_trade/` partitioned by date.

### 3B. Candles from Trades

**Files to add**
- `src/scripts/7-polymarket-build-bars-v1.0.py`

**bars tables**
- `bars_1m`, `bars_5m`, `bars_1h`
- Columns: `timestamp_utc`, `market_id`, `open`, `high`, `low`, `close`, `volume`, `trade_count`

**Storage**
- `src/data/analysis/polymarket/bars/<freq>/` partitioned by market and day.

### 3C. Optional Orderbook Snapshots

If historical CLOB snapshots are available:
- Add `fact_book_top` table with `timestamp_utc`, `best_bid`, `best_ask`, `spread`, `mid`, `depth_1`, `depth_5`.
- If not available, compute decision price from trades and apply conservative slippage.

---

## 4. Time-Safe Feature Builder (Mixed pRN + Polymarket)

**Files to add**
- `src/scripts/8-polymarket-build-features-v1.0.py`
- `src/data/models/polymarket/feature_manifest.json`

**Decision-row schema**
- `timestamp_utc` (decision time)
- `market_id`
- `ticker`
- `threshold`
- `expiry_date_utc`
- Polymarket features at `t`
- pRN features at `t`
- `label` (resolved yes/no)

**Time-safe Polymarket features**
- `pm_mid`, `pm_last`
- `pm_bid`, `pm_ask`, `pm_spread`
- `pm_liquidity_proxy` (volume or depth proxy)
- `pm_momentum_5m`, `pm_momentum_1h`, `pm_momentum_1d`
- `pm_volatility` (realized volatility of pm price)
- `pm_time_to_resolution`

**Leak checks**
- Any feature row must only use data strictly before `timestamp_utc`.
- Include a boolean column `leak_check_passed` and fail the run if false.

---

## 5. Mixed Model Training

**Files to add**
- `src/scripts/9-train-mixed-model-v1.0.py`
- `src/data/models/mixed/`

**Two model options**
- Residual model: predict residual between pm implied probability and realized outcome.
- Ensemble model: learn a blending weight between pRN and pm conditional on liquidity and time-to-expiry.

**Model outputs**
- `model.joblib`
- `feature_manifest.json`
- `metadata.json` with train window and embargo settings.

**Split strategy**
- Strict chronological splits.
- Walk-forward evaluation.
- Embargo windows near creation and resolution times.

---

## 6. Realistic Backtesting Engine

**Files to add**
- `src/scripts/10-backtest-polymarket-v1.0.py`
- `src/data/analysis/backtests/`

**Required mechanics**
- Use bid/ask at decision time or apply conservative slippage.
- Apply fees and spread costs.
- Position sizing capped by liquidity proxy.
- Latency buffer: execute at `t + delta`.
- No-trade if spread or liquidity is too weak.

**Outputs**
- `equity_curve.csv`
- `trade_log.csv`
- `metrics.json` with max drawdown, hit rate, Sharpe-like stats, capacity estimate.

---

## 7. Signal Generator

**Files to add**
- `src/scripts/11-generate-signals-v1.0.py`
- `src/data/analysis/signals/`

**Trade rules**
- Buy YES if `p_model - p_ask > edge_threshold`.
- Sell YES if `p_bid - p_model > edge_threshold`.
- Edge threshold covers spread, fees, and model error.

**Output schema**
- `timestamp_utc`, `market_id`, `direction`, `size`, `expected_value`, `confidence`, `reason_codes`.

---

## 8. Live Trading Phases (Guardrails First)

1. Paper trading: log virtual fills using bid/ask rules.
2. Human-confirmed execution: bot proposes, human approves.
3. Automated execution with guardrails: daily loss limit, max exposure per ticker, max open positions, stale data kill switch.

---

## 9. Webapp Integration

### 9A. Backend API

**New services**
- `src/webapp/backend/app/services/polymarket_subgraph.py`
- `src/webapp/backend/app/services/backtests.py`
- `src/webapp/backend/app/services/signals.py`

**New API routes**
- `src/webapp/backend/app/api/polymarket_subgraph.py`
- `src/webapp/backend/app/api/backtests.py`
- `src/webapp/backend/app/api/signals.py`

**Suggested endpoints**
- `POST /polymarket-subgraph/ingest`
- `GET /polymarket-subgraph/runs`
- `GET /polymarket-subgraph/preview`
- `POST /backtests/run`
- `GET /backtests/runs`
- `GET /backtests/preview`
- `POST /signals/run`
- `GET /signals/latest`

**Backend settings**
- Add Graph API settings in `src/webapp/backend/app/settings.py`.
- Load API key from environment without hardcoding.

### 9B. Shared Schemas

**Files to add**
- `src/webapp/shared/schemas/polymarket_subgraph.json`
- `src/webapp/shared/schemas/backtests.json`
- `src/webapp/shared/schemas/signals.json`

### 9C. Frontend Pages

**New pages**
- `SubgraphIngestPage` for run + preview.
- `BacktestsPage` for parameterized backtests.
- `SignalsPage` for edge review and export.

**API clients**
- `src/webapp/frontend/src/api/polymarketSubgraph.ts`
- `src/webapp/frontend/src/api/backtests.ts`
- `src/webapp/frontend/src/api/signals.ts`

**Dashboard additions**
- Show latest ingest run, latest backtest, and latest signals batch.

---

## 10. Concrete GraphQL Query Templates

Add these to `src/scripts/polymarket/graphql_queries.py`.

**Markets**
```graphql
query Markets($first: Int!, $skip: Int!) {
  markets(first: $first, skip: $skip, orderBy: createdAt, orderDirection: asc) {
    id
    conditionId
    question
    slug
    outcomeTokenIds
    resolved
    resolvedTime
    createdAt
  }
}
```

**Trades**
```graphql
query Trades($first: Int!, $skip: Int!, $since: Int!) {
  trades(first: $first, skip: $skip, orderBy: timestamp, orderDirection: asc, where: { timestamp_gt: $since }) {
    id
    marketId
    outcomeTokenId
    price
    size
    side
    timestamp
    blockNumber
    transactionHash
  }
}
```

**Liquidity events (if available)**
```graphql
query LiquidityEvents($first: Int!, $skip: Int!, $since: Int!) {
  liquidityEvents(first: $first, skip: $skip, orderBy: timestamp, orderDirection: asc, where: { timestamp_gt: $since }) {
    id
    marketId
    provider
    deltaShares
    timestamp
    blockNumber
  }
}
```

Use `block` or `timestamp` filters if supported by the Polymarket subgraph schema.

---

## 11. Anti-Leak Validation Checklist

- Decision rows only use data strictly before `timestamp_utc`.
- Features explicitly record their data window.
- Labels are only attached after resolution.
- Walk-forward train/test splits with embargo windows.
- Backtest execution uses spread and latency.

---

## 12. Milestone Checklist

**Week 1**
1. Subgraph client + raw ingestion.
2. dim_market mapping + manual override file.
3. fact_trade ingestion + bar builder.

**Week 2**
1. Decision dataset builder.
2. First mixed model + walk-forward eval.

**Week 3**
1. Backtest engine with slippage + latency.
2. Paper trading loop + logging.

**Week 4**
1. Human-confirmed trading.
2. Risk controls + kill switch.

---

## 13. Integration Notes for Existing Pipeline

- Keep `src/scripts/3-polymarket-fetch-data-v1.0.py` as the live snapshot pipeline.
- The subgraph pipeline should be a separate historical backbone feeding backtests and feature joins.
- Use the same ticker list defaults where reasonable.
- Store outputs under `src/data/raw/polymarket/` and `src/data/analysis/` to align with existing paths.

---

## 14. Expected Outputs

- Deterministic subgraph raw data with run manifests.
- Normalized fact tables and bar tables.
- Time-safe feature dataset suitable for mixed model training.
- Backtest output with PnL curves and metrics.
- Signal outputs with reason codes.
- Webapp pages to run, inspect, and monitor everything.

