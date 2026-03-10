# Current Branch Patch Notes

Scope: `fix/stale-pm-history-price` relative to `main`.

This note replaces the earlier narrow Markets/Backtests bug log with a branch-wide summary of the user-facing features and fixes added here.

## Summary

- Polymarket history and Markets now use exact run-local pRN refreshes and stop fabricating historical bid/ask spreads.
- Backtests now uses Theta as the only pRN overlay source and adds strike-quality diagnostics plus filtering controls.
- Calibrate gained richer artifact inspection, better AUTO-run summaries, and support for the new realized-volatility feature family.
- New maintenance scripts were added to backfill RV features, refresh run-local pRN datasets, and enrich weekly markets with Gamma volume.

## 1. Polymarket History, Markets, and Data Integrity

- Weekly-history runs now execute an exact run-local pRN refresh after the raw history fetch and before optional decision-feature generation.
- The selected Option Chain training dataset is used to seed that exact pRN refresh, so downstream Markets and feature-generation outputs stay aligned to the same source dataset.
- Historical Polymarket bid/ask values are no longer synthesized from a hardcoded spread around a single CLOB price field.
- `markets_prn_hourly.csv` now stores `polymarket_mid`; bid/ask is only populated when a refresh captures real live quote data.
- Run rename now updates the run label, the run directory name, prefixed CSV filenames, and the active-run pointer when needed.

Why this matters:

- Historical charts no longer imply fake liquidity via artificial bid/ask spreads.
- Markets refresh, weekly-history outputs, and decision features now share one exact pRN source instead of drifting across datasets or approximations.
- Run-directory management is safer because renames keep the filesystem state and UI state consistent.

## 2. Markets and Backtests UX

- Markets charts now render a dedicated PM mid line and only show bid/ask when real quote data exists.
- Backtests dropped the old markets-proxy pRN line and now uses Theta as the sole pRN overlay source.
- Backtests renders PM mid by default, with optional bid/ask when live quote data is available.
- Long gaps in the series now break the rendered chart path instead of drawing a misleading continuous line.
- Holiday weeks no longer require a perfect `{1,2,3,4}` DTE set; the overlay logic relaxes the requirement when exchange holidays are detected.

New Backtests strike diagnostics:

- Gamma volume is shown on strike cards when available.
- Low-volume, suspect-midprice, stale-price, sparse-data, and missing-overlay warnings are surfaced directly in the UI.
- Users can hide suspect strikes and apply a minimum-volume threshold before selecting a strike.

Why this matters:

- Charts now distinguish real quote coverage from historical mid-like data.
- The Backtests page is less likely to show misleading strikes or silently drop valid holiday weeks.
- Volume and quality warnings make it easier to reject empty-book or stale-price markets before interpreting the chart.

## 3. Calibrate Workflow

- Option-chain datasets now expose the expanded realized-volatility family when available: `rv5`, `rv10`, and RV ratio features in addition to the prior `rv20` context.
- The Models tab now auto-opens the default metrics artifact instead of dropping users into a blank artifact state.
- AUTO runs now surface richer selection diagnostics, including selection-rule context, fold-gate summaries, and no-viable-model reasons when relevant.
- Metrics display was expanded with clearer split coverage, richer metrics cards, and equation notes.
- Saved Calibrate form state now recovers cleanly when a previously selected dataset path is stale or no longer present.

Why this matters:

- Feature selection can use shorter realized-volatility horizons and volatility-regime ratios without manual file inspection.
- Reviewing AUTO runs is faster because the selection logic and artifact layout are visible immediately.
- Stale local UI state is less likely to leave the page in a broken configuration.

## 4. Dataset and Maintenance Scripts

- Added `src/scripts/01-option-chain-backfill-rv-features-v1.0.py` to backfill `rv5` and `rv10` into existing main option-chain artifacts and refresh snapshot volatility columns.
- Added `src/scripts/08-polymarket-run-prn-refresh-v1.0.py` to rebuild exact run-local pRN coverage for an existing weekly-history run and optionally rebuild markets artifacts from that dataset.
- Added `src/scripts/backfill_gamma_volume.py` to enrich `weekly_markets.csv` with Gamma API volume fields.

Operational note:

- Existing historical outputs may look different after refreshes because false historical bid/ask lines were intentionally removed in favor of PM mid plus real live quotes only.

## 5. Files Most Relevant To This Branch

- `src/webapp/frontend/src/pages/DocumentationPage.tsx`
- `src/webapp/frontend/src/pages/MarketsPage.tsx`
- `src/webapp/frontend/src/pages/BacktestsPage.tsx`
- `src/webapp/frontend/src/pages/PolymarketPipelinePage.tsx`
- `src/webapp/frontend/src/pages/CalibrateModelsPage.tsx`
- `src/webapp/backend/app/services/polymarket_history.py`
- `src/webapp/backend/app/services/polymarket_run_prn.py`
- `src/webapp/backend/app/services/markets.py`
- `src/webapp/backend/app/services/bars.py`
- `src/webapp/backend/app/services/calibrate_models.py`
