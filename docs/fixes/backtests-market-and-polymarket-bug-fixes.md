# Backtests, Markets & Polymarket Pipeline — Bug Fixes & Changes

## Overview

This document details every bug fixed and every meaningful change introduced across the `BacktestsPage`, `MarketsPage`, Polymarket history builder pipeline, and the on-demand pRN computation layer during this branch.

---

## 1. Polymarket Bid / Ask / Mid Price — Root-Cause & Full Fix

### Main Bug

The single most impactful problem in this branch.

The Polymarket history builder (`07-polymarket-markets-refresh-v1.0.py`) **never fetched real bid or ask prices**. The CLOB `/prices-history` endpoint returns only a single, ambiguous `price` field. The script interpreted this as `polymarket_buy` and then **synthetically derived** `polymarket_bid` and `polymarket_ask` by applying a fixed hardcoded spread (e.g. 200 bps) around `polymarket_buy`:

```python
# OLD — Fabricated spread, completely artificial
polymarket_bid = polymarket_buy * (1 - SPREAD_BPS / 10_000)
polymarket_ask = polymarket_buy * (1 + SPREAD_BPS / 10_000)
```

These values:
- Were **not real market quotes** from any exchange or liquidity provider.
- Created a **false impression of a real bid–ask spread** stored permanently in `markets_prn_hourly.csv`.
- Propagated to the frontend `MarketsPage` and `BacktestsPage`, where they were rendered as real market prices on charts.
- Made all historical bid/ask data meaningless and unusable for model training or backtesting.

### What Changed

#### `07-polymarket-markets-refresh-v1.0.py`
- **Removed** the `bidask_spread_bps` config parameter and all synthetic bid–ask derivation logic.
- The CLOB price (from `/prices-history`) is now correctly labelled as **`polymarket_mid`** — it is an opaque price closest in nature to a mid price, not a bid or an ask.
- `polymarket_bid` and `polymarket_ask` are set to `NaN` for all historical rows.
- For **active (live) markets**, real bid/ask is now fetched from the CLOB `POST /prices` endpoint at the end of each refresh run, and stored **only at the latest timestamp** of each market series. These are real quotes.
- A new `fetch_live_bidask()` helper was added to call `SNAPSHOT.get_prices_bulk()` and compute real mid from `(bid + ask) / 2`.
- `polymarket_mid` is added to the `PRN_COLUMNS` list and written into `markets_prn_hourly.csv`.

#### Existing Data Cleanup
- Both active Polymarket datasets (`main-polymarket` and `main-polymarket-2`) had their `markets_prn_hourly.csv` files retroactively cleaned:
  - `polymarket_bid` and `polymarket_ask` columns zeroed to `NaN` for all historical rows (artificial spread removed).
  - `polymarket_mid` column added, set to `polymarket_buy` (the CLOB price) for all rows.
- Total rows cleaned: ~30,680 (`main-polymarket`) and ~242,254 (`main-polymarket-2`).

---

## 2. Backend — Markets Service

### `app/services/markets.py`

- **Before:** `_build_series_point()` synthesised `polymarket_bid` and `polymarket_ask` by computing them from `polymarket_buy` with a hardcoded spread, discarding any real values that may have been stored.
- **After:** The function now reads `polymarket_mid`, `polymarket_bid`, `polymarket_ask` directly from the CSV row, with no synthetic derivation. If `polymarket_mid` is absent (legacy rows), it falls back to `polymarket_buy`.
- `polymarket_mid` was added to the `_col_flags()` column set and to the `MarketsSeriesPoint` model.

### `app/models/markets.py`

- Added `polymarket_mid: Optional[float] = None` to `MarketsSeriesPoint`.

---

## 3. Frontend — MarketsPage

### `MarketsPage.tsx`

- Added `polymarketMid` to the `ChartPoint` type, populated from `p.polymarket_mid ?? p.polymarket_buy`.
- Chart now plots a dedicated **mid price line** (purple) distinct from bid/ask.
- Bid and ask lines are now rendered **conditionally** — only when real bid/ask values are present in the data (i.e. the latest rows after a live refresh). Historical rows show only the mid line.
- Tooltip and legend updated accordingly.

### `MarketsPage.css`

- Added `.chart-line-mid`, `.mdc-swatch-mid`, `.tt-mid` styles (purple, same weight as other lines).
- `.chart-line-bid` and `.chart-line-ask` are now thinner and dashed to distinguish them visually from the mid price.

---

## 4. Frontend — BacktestsPage: Bid/Ask Display

### `BacktestsPage.tsx`

- **Removed** the `SYNTH_BIDASK_SPREAD` constant (was 200 bps, used to re-derive artificial bid/ask on the frontend — now unnecessary).
- `MarketChartPoint` type extended with `polymarketMid`.
- `buildChartPointsFromMarkets()` now reads `polymarket_mid`, `polymarket_bid`, `polymarket_ask` directly from the API response, with no frontend-side spread calculation.
- `buildChartPointsFromBars()` (fallback path when market series is unavailable) sets `polymarketMid` from the bar price and leaves bid/ask as `null`.
- `BacktestChart` renders mid unconditionally; bid/ask are rendered only when non-null (i.e. real values from a live refresh).

### `BacktestsPage.css`

- Same visual changes as MarketsPage: purple mid line, thinner dashed bid/ask lines.

---

## 5. pRN Computation — Theta as the Sole Source

### Problem

The `BacktestsPage` previously pulled pRN from **three competing sources** with incompatible parameters:

| Source | Parameters | Priority |
|--------|-----------|----------|
| Stored overlay (`training-*.csv`) | Strict: 10 min strikes, 0.06 moneyness band, real dividend yield | Primary |
| Theta on-demand (`prn_on_demand.py`) | Relaxed: 5 min strikes, 0.10 band, `q=0` | Gap-fill |
| Market proxy (`markets_prn_hourly.csv`) | Carried-forward from option chain, hourly interpolated | Fallback |

This created **consistency failures**: a calibrated model trained on the strict-parameter dataset could not be reliably applied to backtest pRN values computed with looser parameters. The market proxy pRN introduced a continuous line that implied more data than was actually computed, and it blurred the distinction between trading days and non-trading days.

### What Changed

#### `prn_on_demand.py` — Parameter Alignment

All parameters now exactly match the `Config` defaults in `01-option-chain-build-historic-dataset-v1.0.py`:

| Parameter | Old (relaxed) | New (aligned) |
|-----------|--------------|---------------|
| `MIN_STRIKES_FOR_CURVE` | 5 | 10 |
| `MAX_ABS_LOGM` | 0.10 | 0.06 |
| `MAX_ABS_LOGM_CAP` | 0.15 | 0.10 |
| `PREFER_BIDASK_MIN` | 5 | 10 |

#### `prn_on_demand.py` — Dividend Yield

- Added `_fetch_dividend_yield(ticker, asof)`: fetches trailing annual dividend yield from `yfinance` over a 365-day lookback window ending on `asof`.
- Results are cached on disk (`src/data/cache/dividend_yield/`) with a 7-day TTL, plus an in-process memory cache.
- `q` (dividend yield) is passed into `_build_call_curve()`, where the intrinsic floor now uses `np.exp(-q * T)` instead of the previous `np.exp(0.0)` (i.e. `q=0` hardcode).
- Falls back to `0.0` gracefully if `yfinance` is unavailable.

#### `prn_on_demand.py` — Market Calendar (Holiday Handling)

- Added `_get_exchange_calendar()`: loads the NYSE (`XNYS`) calendar from `exchange_calendars` once at startup (thread-safe singleton).
- `_trading_dates_in_range()` now uses real NYSE sessions via `cal.sessions_in_range()` instead of the naive `weekday() < 5` check.
- When `exchange_calendars` is unavailable, falls back to weekday-only filtering.
- This prevents Theta from being queried on market holidays (which returned empty chains, wasting disk cache entries), and ensures DTE values are computed correctly around holidays.
- The response `metadata` now includes `holidays_in_range` — a list of ISO date strings for weekdays in the range that were skipped due to being holidays.

#### `BacktestsPage.tsx` — Theta as the Only Source

- **Removed** `getPrnOverlay` import and call entirely. The stored-overlay endpoint is no longer called.
- **Removed** `mergePrnOverlays()` function (was responsible for merging stored + Theta; no longer needed).
- `handleRun` now follows a simpler two-phase flow:
  1. Parallel: `getBarsByStrike` + `getMarketsSeriesByTicker`
  2. Sequential: `getPrnOverlayTheta` (sole pRN source), result set directly via `setPrnData(thetaPrn)`
- No merge step; no priority logic; no stored-overlay dependency.

#### `BacktestsPage.tsx` — Market Proxy pRN Removed

- `buildChartPointsFromMarkets()` now always sets `prn: null`, even when `p.pRN` exists in the markets series. The pRN line from `markets_prn_hourly.csv` is never rendered.
- The "pRN source: markets proxy" chip label was removed.
- All pRN source labels now show "pRN source: Theta".

#### `BacktestsPage.tsx` — Graceful Holiday Handling in `hasRequiredPrnDtes`

- Previously: a strike was only displayed if it had pRN for **all 4 DTEs** {1, 2, 3, 4}. A single market holiday in the week dropped the entire strike from the chart.
- Now: when `holidays_in_range` in the Theta metadata indicates at least one holiday exists in the range, the filter relaxes to require **at least 2 of 4 DTEs** (constant `MIN_DTES_WITH_HOLIDAYS = 2`). This preserves valid pRN dots during holiday weeks without sacrificing data quality.

---

## 6. Files Modified

| File | Change Type |
|------|-------------|
| `src/scripts/07-polymarket-markets-refresh-v1.0.py` | Removed synthetic bid/ask; added `polymarket_mid`; added live bid/ask fetch via CLOB |
| `src/webapp/backend/app/models/markets.py` | Added `polymarket_mid` field to `MarketsSeriesPoint` |
| `src/webapp/backend/app/services/markets.py` | Reads real mid/bid/ask from CSV; no more synthetic spread |
| `src/webapp/backend/app/services/prn_on_demand.py` | Aligned parameters; added dividend yield; added exchange calendar; added holiday metadata |
| `src/webapp/frontend/src/api/markets.ts` | Added `polymarket_mid` to `MarketsSeriesPoint` TypeScript interface |
| `src/webapp/frontend/src/pages/MarketsPage.tsx` | Mid price line; conditional bid/ask rendering |
| `src/webapp/frontend/src/pages/MarketsPage.css` | Purple mid line style; dashed bid/ask lines |
| `src/webapp/frontend/src/pages/BacktestsPage.tsx` | Theta-only pRN; removed synthetic spread; removed stored overlay; holiday-aware DTE filter |
| `src/webapp/frontend/src/pages/BacktestsPage.css` | Purple mid line style; dashed bid/ask lines |
| `src/data/raw/.../main-polymarket/markets_prn_hourly.csv` | Retroactive cleanup: artificial bid/ask nulled, `polymarket_mid` added |
| `src/data/raw/.../main-polymarket-2/markets_prn_hourly.csv` | Retroactive cleanup: artificial bid/ask nulled, `polymarket_mid` added |

---

## 7. Impact on Model Training & Backtesting Reliability

- **Before:** pRN values displayed on the Backtests page could come from three different sources with three different parameter sets. A model trained on the strict dataset could not be applied to on-demand pRN without unknown error.
- **After:** Theta on-demand pRN now uses the same parameters as the training dataset. Any calibrated model's inputs (pRN at specific strikes and DTEs) are computed identically in training and at inference time during backtesting.
- Bid/ask prices on the chart now reflect **real market quotes** for active markets, or explicitly show `null` (no line rendered) for historical rows where only a single opaque CLOB price was available.
