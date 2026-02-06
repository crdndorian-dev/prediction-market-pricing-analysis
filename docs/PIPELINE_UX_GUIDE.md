# Polyedgetool Pipeline UX Guide

**Version:** 1.0.0
**Last Updated:** 2026-02-02

---

## Overview

This guide explains how to use the polyedgetool pipeline **safely and correctly**. The pipeline has been hardened to prevent schema mismatches, silent failures, and garbage predictions.

**Key Principle:** The pipeline is now **UX-safe by construction**. You should be able to run it end-to-end with default commands, and any failures will be **loud, early, and actionable**.

---

## Quick Start (Happy Path)

### 1. Generate Polymarket Snapshot

```bash
python src/scripts/3-polymarket-fetch-data-v1.0.py \
  --out-dir src/data/raw/polymarket \
  --tickers NVDA,AAPL,GOOGL
```

**What it does:**
- Fetches live Polymarket contracts for specified tickers
- Computes yfinance pRN estimates
- Enriches with ALL guaranteed baseline features
- Exports: `pPM-dataset-snapshot-<date>.csv`

**Guaranteed:** Snapshot will contain all columns from `config/pipeline_schema_contract.json` baseline schema (even if NaN).

### 2. Train Calibration Model (Snapshot-Only Mode)

**IMPORTANT:** For production inference on snapshots, use **snapshot-compatible features only**:

```bash
python src/scripts/2-calibrate-logit-model-v1.5.py \
  --csv src/data/raw/option-chain/pRN__history__mon_thu__PM10__v1.6.0.csv \
  --out-dir src/data/models/calibration-snapshot-only \
  --features x_logit_prn,log_m,abs_log_m,T_days,sqrt_T_years,log_T_days,log_m_over_volT,log_rel_spread,x_prn_x_tdays,x_prn_x_logm \
  --mode pooled \
  --ticker-intercepts non_foundation \
  --foundation-tickers SPY,QQQ,IWM
```

**What it does:**
- Validates requested features against schema contract
- Warns if features depend on NaN-prone columns (rv20, dividend_yield, forward_price)
- Trains model on historical data
- Emits feature manifest for downstream consumption

**Output:**
- `model.joblib` - Trained model bundle
- `feature_manifest.json` - Required columns for inference
- `metadata.json` - Training metadata
- `metrics.csv` - Performance metrics

### 3. Apply Model to Snapshot

```bash
python src/scripts/4-compute-edge-v1.1.py \
  --model-path src/data/models/calibration-snapshot-only/model.joblib \
  --snapshot-csv src/data/raw/polymarket/runs/<run-id>/pPM-dataset-snapshot-<date>.csv \
  --out-csv src/data/analysis/phat-edge/phat-snapshot-<date>.csv \
  --require-columns-strict true
```

**What it does:**
- Loads model and feature manifest
- Validates snapshot schema against baseline contract
- Checks for high NaN fractions in required features
- Computes pHAT probabilities
- Computes edges (pHAT - buy_price)
- FAILS LOUDLY if critical features are missing or all-NaN

**Output:**
- CSV with pHAT predictions and edges
- Console warnings for any schema issues

---

## Schema Contract

The pipeline enforces a **guaranteed schema contract** defined in `config/pipeline_schema_contract.json`.

### Three Tiers of Columns

#### 1. Guaranteed Baseline Columns
**Always present** in every snapshot (even if NaN):
- `ticker`, `K`, `S`, `T_days`, `pRN`, `pRN_raw`
- `event_endDate`, `snapshot_time_utc`, `r`
- `dividend_yield`, `forward_price`, `rv20` ⚠️ **NaN in snapshot-only mode**
- `rel_spread_median`, `n_chain_*`, `asof_fallback_days`, etc.

#### 2. Guaranteed Derived Features
**Always computable** from baseline (never NaN unless baseline is corrupt):
- `x_logit_prn` - logit(pRN)
- `log_m`, `abs_log_m` - Log-moneyness
- `T_years`, `sqrt_T_years`, `log_T_days` - Time transformations

#### 3. Conditional Derived Features
**Computable IF dependencies are non-NaN**:
- `rv20_sqrtT`, `log_m_over_volT` → Need `rv20` (**NaN in snapshots**)
- `log_m_fwd`, `abs_log_m_fwd` → Need `forward_price` (**NaN in snapshots**)
- `x_prn_x_rv20` → Need `rv20` (**NaN in snapshots**)

### Known Limitations

**Columns that are NaN in snapshot-only mode:**
- `rv20` - Realized volatility (requires historical stock data)
- `dividend_yield` - Dividend yield (requires historical dividend data)
- `forward_price` - Forward price (computed from rv20 + dividend_yield)

**Impact:** Any features depending on these columns will be NaN in snapshot-only inference.

**Solution:** Train **two model variants**:
1. **Snapshot-only model** - Excludes rv20/dividend features (use for production)
2. **Full-feature model** - Includes all features (requires historical data join)

---

## Feature Selection Best Practices

### ✅ Safe Snapshot-Only Features

These features are **always available** in snapshots:

```python
SNAPSHOT_SAFE_FEATURES = [
    "x_logit_prn",           # Primary signal (always present)
    "log_m",                 # Log-moneyness from K/S
    "abs_log_m",             # Absolute log-moneyness
    "T_days",                # Time to expiration (days)
    "sqrt_T_years",          # Square root of time (years)
    "log_T_days",            # Log of time
    "x_prn_x_tdays",         # Interaction: logit(pRN) * T_days
    "x_prn_x_logm",          # Interaction: logit(pRN) * log_m
    "log_rel_spread",        # Log of relative spread
    "fallback_any",          # Fallback indicator
    "prn_raw_gap",           # pRN - pRN_raw gap
]
```

### ⚠️ Historical-Data-Only Features

These features **require historical data join** and will be NaN in snapshots:

```python
HISTORICAL_ONLY_FEATURES = [
    "rv20",                  # 20-day realized volatility
    "rv20_sqrtT",            # rv20 * sqrt(T_years)
    "log_m_over_volT",       # log_m / (rv20 * sqrt_T_years)
    "abs_log_m_over_volT",   # abs_log_m / (rv20 * sqrt_T_years)
    "x_prn_x_rv20",          # x_logit_prn * rv20
    "log_m_fwd",             # Log-moneyness using forward price
    "abs_log_m_fwd",         # Absolute forward log-moneyness
    "log_m_fwd_over_volT",   # Forward moneyness / vol-time
    "abs_log_m_fwd_over_volT",
    "dividend_yield",        # Dividend yield
    "forward_price",         # Forward price
]
```

### 🎯 Recommended Default Features

For **production snapshot inference**, use:

```bash
--features x_logit_prn,log_m,abs_log_m,T_days,sqrt_T_years,log_T_days,x_prn_x_tdays,x_prn_x_logm,log_rel_spread,fallback_any,prn_raw_gap
```

For **research with historical data**, add:

```bash
--features <snapshot-safe-features>,rv20,rv20_sqrtT,log_m_fwd,abs_log_m_fwd,log_m_over_volT,x_prn_x_rv20,dividend_yield
```

---

## Validation & Error Handling

### Calibration Validation

When training, the calibrator **validates features** against the schema contract:

```
=== VALIDATING FEATURE AVAILABILITY ===
[INFO] Model includes 3 features that depend on NaN-prone columns.
       These features will likely be NaN in snapshot-only inference:
       - rv20
       - rv20_sqrtT
       - x_prn_x_rv20

[RECOMMENDATION] Consider training a snapshot-only model variant without these features
                 for production inference on live Polymarket snapshots.
```

**Action:** If you see this warning and plan to use the model for snapshot inference, **retrain without NaN-prone features**.

### Snapshot Validation

When generating snapshots, the script **guarantees baseline schema**:

```python
# All columns from guaranteed_baseline_columns MUST exist
# Conditional features are computed (may be NaN if dependencies are NaN)
# No silent failures - schema is explicit and documented
```

### Inference Validation

When applying model to snapshot, inference script **validates schema**:

```
=== VALIDATING SNAPSHOT SCHEMA ===
[WARNING] Snapshot schema issues detected:
  - Column 'rv20' is >90% NaN (100.0%)
  - Column 'forward_price' is >90% NaN (100.0%)

[CRITICAL WARNING] Features with >90% NaN values:
  - rv20 (100.0% NaN)
  - rv20_sqrtT (100.0% NaN)

Predictions may be unreliable due to missing feature data.
Consider using a model trained without these features,
or join historical data to populate missing columns.
```

**Action:** If you see critical warnings, either:
1. **Use a snapshot-only model** (recommended), OR
2. **Join historical data** before inference (advanced)

---

## Troubleshooting

### Error: "Missing required columns in snapshot"

**Cause:** Snapshot CSV doesn't have baseline schema columns.

**Fix:** Regenerate snapshot with latest version of Script 3 (includes schema guarantees).

### Error: "Unknown features requested that cannot be computed from baseline schema"

**Cause:** Calibration requested features not in schema contract.

**Fix:** Check `config/pipeline_schema_contract.json` for available features, or add custom feature to `ensure_engineered_features()`.

### Warning: "Features depending on NaN-prone columns"

**Cause:** Model trained with features that require historical data (rv20, dividend_yield, forward_price).

**Fix:** Train two model variants:
- **Snapshot-only** (exclude NaN-prone features) → Use for production
- **Full-feature** (include all features) → Use for research with historical data

### Warning: "Column 'rv20' is >90% NaN"

**Cause:** Snapshot doesn't include historical stock data (expected behavior).

**Fix:** This is **expected and OK** if using a snapshot-only model. If using a full-feature model, you need to join historical data.

---

## Advanced: Historical Data Join

If you need to use full-feature models with snapshot data, join historical stock data:

```python
import pandas as pd

# Load snapshot
snapshot = pd.read_csv("snapshot.csv")

# Load historical data (from dataset generation script)
historical = pd.read_csv("pRN__history__mon_thu__PM10__v1.6.0.csv")

# Join on (ticker, date, strike)
enriched = snapshot.merge(
    historical[["ticker", "week_friday", "K", "rv20", "dividend_yield", "forward_price"]],
    left_on=["ticker", "event_endDate", "K"],
    right_on=["ticker", "week_friday", "K"],
    how="left"
)

# Now rv20, dividend_yield, forward_price will be populated
enriched.to_csv("snapshot_with_historical.csv", index=False)
```

**Note:** This requires that historical dataset covers the same ticker/date/strike combinations.

---

## Schema Evolution

If you need to **add new features**:

1. **Update** `config/pipeline_schema_contract.json`:
   - Add to `guaranteed_derived_features` if always computable
   - Add to `conditional_derived_features` if depends on optional columns

2. **Update** `calibrate_common.py`:
   - Add computation logic to `ensure_engineered_features()`

3. **Update** `src/scripts/3-polymarket-fetch-data-v1.0.py`:
   - Add to `SNAPSHOT_STANDARD_FEATURE_COLUMNS`
   - Add computation in `enrich_snapshot_features()`

4. **Regenerate** snapshots and retrain models

---

## Validation Checklist

Before deploying a model to production:

- [ ] Model trained on **snapshot-compatible features only** (no rv20/dividend/forward_price)
- [ ] Inference script tested on **real snapshot** (not historical dataset)
- [ ] Validation warnings reviewed (no critical NaN fractions)
- [ ] Edge predictions are **reasonable** (not all NaN or all zero)
- [ ] Model metrics on test set are **acceptable** (ECE < 0.05, Logloss < target)
- [ ] Feature manifest saved and version-controlled
- [ ] Model artifact and metadata saved to `src/data/models/`

---

## Contact & Support

For schema issues or feature requests:
1. Check `config/pipeline_schema_contract.json` for guaranteed features
2. Review this guide's troubleshooting section
3. Inspect validation warnings during calibration/inference
4. File an issue with reproducible example

---

**Remember:** The pipeline is UX-safe by construction. Trust the validation warnings, and follow the recommended feature sets for your use case (snapshot-only vs. full-feature).
