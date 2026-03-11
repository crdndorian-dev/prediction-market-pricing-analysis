# `src/scripts`

This directory is organized so scripts do not live directly at the root.

Public wrappers and helper layers live here:

- `entrypoints/`
- `compatibility/`
- `support/`

The real implementations live under stage-first directories:

- `data_collection/`
- `dataset_building/`
- `feature_engineering/`
- `model_training/`
- `orchestration/`

When adding or moving script implementations, keep the public entrypoint path
stable unless the external contract is intentionally changing.

Package marker files named `__init__.py` are not listed below. They only make
the folders importable.

## Entry Points

- `entrypoints/option-chain-build-historic-dataset.py`: Public CLI for building the option-chain training dataset. This is the file the backend should call when it wants to generate or refresh option-chain data.
- `entrypoints/calibrate-logit-model.py`: Public CLI for training the calibration model. It is the stable wrapper around the current calibration implementation.
- `entrypoints/auto-calibrate-logit-model.py`: Public CLI for the automatic model search and selection workflow. It runs multiple calibration trials and keeps the best result.
- `entrypoints/polymarket-weekly-history.py`: Public CLI for downloading and assembling weekly Polymarket market history. It backfills the raw market and price-history inputs used later in the pipeline.
- `entrypoints/polymarket-build-features.py`: Public CLI for turning Polymarket raw data plus pRN inputs into model-ready features. This is the feature-building step for Polymarket datasets.
- `entrypoints/polymarket-market-map.py`: Public CLI for building the Polymarket market identity table. It creates the joinable mapping between markets, tickers, thresholds, and expiries.
- `entrypoints/polymarket-fetch-snapshot.py`: Public CLI for pulling a current Polymarket snapshot and enriching it with pRN-style fields. It is mainly used for point-in-time snapshot generation.
- `entrypoints/polymarket-markets-refresh.py`: Public CLI for the end-to-end Polymarket refresh flow. It stitches together snapshot pulls, history, and feature building into one orchestration step.

## Compatibility

- `compatibility/01-option-chain-build-historic-dataset-v1.0.py`: Legacy filename kept so older references still resolve. It simply forwards to the newer option-chain dataset builder implementation.
- `compatibility/option_chain_weighting_v3.py`: Legacy import path for the option-chain weighting helpers. It re-exports the current weighting functions so older backend code and notebooks do not break.

## Support

- `support/script_paths.py`: Shared path resolver for the scripts tree. It finds the repo root and scripts root without relying on fragile relative-directory assumptions.
- `support/legacy_facade.py`: Shared loader for wrapper scripts and compatibility shims. It imports an implementation module by path and re-exports its public globals.

## Data Collection

- `data_collection/polymarket/fetch_snapshot_v1.py`: Pulls a Polymarket snapshot and computes the enriched snapshot dataset. It combines live Polymarket data with option-derived context.
- `data_collection/polymarket/weekly_history_v1.py`: Downloads historical weekly Polymarket market data and builds bar/history outputs. This is the main raw-history ingestion script for weekly contracts.

## Dataset Building

- `dataset_building/option_chain/build_historic_dataset_v1_0.py`: Builds the historical option-chain training dataset used for calibration. It fetches option data, computes pRN-related fields, and writes the train/prn/snapshot outputs.
- `dataset_building/polymarket/market_map_v1.py`: Builds the Polymarket `dim_market` reference table. It standardizes market metadata so downstream joins are consistent.

## Feature Engineering

- `feature_engineering/option_chain/weighting_v3.py`: Computes the v3 training weights for option-chain rows. It creates group, ticker, and trade-focus weights used during model training.
- `feature_engineering/polymarket/build_features_v1.py`: Builds the decision-feature dataset for Polymarket modeling. It joins raw Polymarket bars with pRN data and enforces anti-leak rules.

## Model Training

- `model_training/calibration/calibrate_logit_model_v2.py`: Runs the calibration trainer itself. It defines the CLI contract and passes execution into the shared calibration core under `src/calibration/`.

## Orchestration

- `orchestration/calibration/auto_calibrate_logit_model_v2.py`: Coordinates automatic hyperparameter and feature-set search for calibration models. It repeatedly calls the calibrator and compares candidate results.
- `orchestration/polymarket/markets_refresh_v1.py`: Coordinates the broader Polymarket refresh pipeline. It loads snapshots, weekly history, pRN data, and feature-building steps into one update flow.

## Shared Polymarket Helpers

- `polymarket/subgraph_client.py`: Small client for querying the Polymarket-related Graph subgraphs. It handles request setup and query execution.
- `polymarket/graphql_queries.py`: Stores named GraphQL query templates used by the Polymarket scripts. This keeps query text out of the pipeline logic files.
- `polymarket/prn_loader.py`: Loads and normalizes the latest pRN datasets for Polymarket-related joins. It also contains helpers for finding compatible option-chain datasets.
- `polymarket/snapshot_enrichment.py`: Shared enrichment helpers used when snapshot rows need forward prices, logits, or related derived fields. It centralizes logic reused by snapshot and refresh scripts.
- `polymarket/weekly_history_io.py`: Shared I/O and bar-building helpers for weekly history processing. It handles cleaning, schema-safe CSV writes, and converting price histories into bars.
