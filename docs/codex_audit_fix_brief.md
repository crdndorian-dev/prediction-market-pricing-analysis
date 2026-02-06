# Codex Implementation Brief — Time-safe pHAT Logit Calibrator v1.5 (Scripts + Webapp Coherence)

**Goal:** Apply the audit fixes to both scripts:
- `2-calibrate-logit-model-v1.5.py`
- `calibrate_common.py`

…and update the **front-end/back-end webapp** so it remains coherent with the updated training/inference pipeline.

This document is written as *implementation instructions* for Codex. Favor **minimal-diff edits**, strong **time-safety**, and **robust calibration** over complexity.

---

## 0) Non-negotiables (do not break)

1. **No leakage / look-ahead.** Never use any feature that requires *future* information relative to the as-of timestamp.
2. **Train/Val/Test discipline must stay time-safe.** Anything like Platt/Isotonic calibration must be fit **only** on a **past** calibration slice (not VAL/TEST).
3. **Keep interpretability.** Default model should remain regularized logistic regression on a stable feature set.
4. **Minimal diffs.** Prefer adding guardrails and feature gating rather than rewriting architecture.

---

## 1) Critical fixes to implement in code

### 1.1 Ban “pipeline diagnostic” columns as numeric predictors (CRITICAL)

These columns encode *pipeline behavior* (fallbacks, adaptive banding, number of quotes used, etc.) and can act as **leakage proxies** or brittle predictors. They must **not** be used as numeric features.

**Examples (by naming patterns):**
- Any column matching: `.*fallback.*`, `.*_used.*`, `.*inside_frac.*`, `.*drop_.*`, `.*band_.*`, `.*n_calls_.*`, `.*calls_k_.*`, `.*deltaK.*`
- Specific known risky examples mentioned in audits:
  - `asof_fallback_days`
  - `split_events_in_preload_range`
  - `n_calls_used`, `calls_k_min`, `calls_k_max`, `band_inside_frac`, `drop_intrinsic_frac` (or similarly named)

#### Required behavior
- **Do not include these in numeric feature set** even if they exist in the CSV.
- Replace with **binary indicators** only when useful:
  - e.g. `had_fallback = (asof_fallback_days != 0)` (or `>0` depending on your convention)
  - `had_intrinsic_drop = (drop_intrinsic_frac > 0)`
  - `had_band_clip = (band_inside_frac < 1)` (example, adapt to semantics)

#### Where to implement
- In `calibrate_common.py` (preferred): inside `ensure_engineered_features()` and/or feature selection logic used by `make_pipeline()`.
- Also add a **defensive layer** in `2-calibrate-logit-model-v1.5.py` so even if the feature list requests them, they are removed.

#### Minimal-diff strategy
- Add a `FORBIDDEN_NUMERIC_FEATURE_PATTERNS` list of regex patterns.
- Create a helper: `filter_forbidden_features(feature_names: list[str]) -> list[str]`.
- Apply it to both:
  - user-specified `--features`
  - any defaults / auto-added engineered features

---

### 1.2 Make forward-looking fallback illegal (CRITICAL)

If dataset creation ever did “forward fallback” for as-of close (or any price), that is **look-ahead leakage**.

#### Required behavior
- Calibrator must **never** trust forward-fallback-derived values.
- If a column indicates forward fallback, either:
  1) **Drop those rows** from TRAIN_FIT / CALIB / VAL / TEST, or
  2) Require fallback to be **backward-only**, and rename semantics accordingly.

#### Implementation guidance
- If `asof_fallback_days` exists:
  - Decide convention:
    - `>0` = looked forward; `<0` = looked backward; `0` = exact
  - Enforce: **no positive fallback** in any split used for training/evaluation.
  - If positives exist, log a clear warning and either:
    - drop affected rows, or
    - hard-fail with actionable error.

---

### 1.3 Auto-drop near-constant features after regime filters (CRITICAL)

When training on a regime subset (e.g., `--tdays-allowed 4` and/or `--asof-dow-allowed 0`), features like `T_days`, `sqrt_T_years`, and `asof_dow` become constants. Keeping them can cause numeric instability and coefficient noise.

#### Required behavior
After applying TRAIN_FIT mask (and any regime filters):
- drop any numeric feature with variance below a small epsilon in TRAIN_FIT:
  - e.g. `var < 1e-12` (or based on std)
- propagate the same feature mask to all splits.

#### Where to implement
- In `calibrate_common.py` pipeline builder (best), after imputation but before scaling; or just after assembling `X_train_fit`.
- Ensure feature name lists are updated consistently.

---

### 1.4 Ticker effects must be support-thresholded (IMPORTANT → CRITICAL if interactions enabled)

Ticker intercepts and especially ticker×`x_logit_prn` interactions can overfit, particularly with regime filtering.

#### Required behavior
- Introduce `--ticker-min-support` (default reasonable, e.g. 300 rows) to scope tickers:
  - tickers with count < threshold in TRAIN_FIT become `"OTHER"`
- Introduce a stricter threshold for interactions:
  - e.g. `--ticker-min-support-interactions` (default 1000 rows)
- If interactions enabled but threshold unmet for most tickers, disable interactions and warn.

#### Where to implement
- In `calibrate_common.py` where ticker column is prepared before `OneHotEncoder`.

---

### 1.5 Select hyperparameters by rolling **delta vs baseline** (IMPORTANT)

Current selection chooses C by minimizing absolute rolling logloss. This can pick a “best among bad models” that still underperforms baseline.

#### Required behavior
During rolling selection:
- compute baseline logloss in each window (baseline = pRN or other defined baseline)
- compute model logloss in each window
- select C that minimizes average **delta**:
  - `avg_delta = avg(model_ll - baseline_ll)`
- safety:
  - if best `avg_delta > 0`, fall back to baseline-only deployment (or disable risky add-ons like interactions)

---

### 1.6 ECE: add equal-mass binning (IMPORTANT)

Equal-width ECE can be unstable if predictions cluster.

#### Required behavior
- Keep existing ECE (equal-width).
- Add **ECE-Q** (equal-mass / quantile binning), default 10 bins.
- Report both on TRAIN (optional), VAL_POOL, TEST.

---

### 1.7 Calibration must respect regimes (IMPORTANT)

Platt calibration is time-safe but can become a global compromise across regimes.

#### Required behavior (minimal diff)
- If regime filters are active (`--tdays-allowed` and/or `--asof-dow-allowed`), calibrate on that same regime only (already implied by filtering).
- If pooled model across regimes:
  - report ECE by regime (group metrics by `T_days` and `asof_dow`)
  - optionally allow `--platt-by-regime` (separate calibrators), but keep default simple.

---

## 2) Concrete code changes (what to edit where)

### 2.1 `calibrate_common.py` — add feature gating utilities

Add helpers (minimal additions):

1) **Forbidden feature filter**
- Inputs: list of candidate feature names
- Outputs: filtered list + list of removed features for logging

2) **Near-constant dropper**
- Inputs: `X_train_fit` (numpy or sparse), feature_names
- Outputs: mask / filtered arrays

3) **Ticker scoping**
- Inputs: df, train_fit_mask, `ticker_min_support`
- Outputs: new column `ticker_scoped`

4) **ECE-Q computation**
- Add `ece_equal_mass(y, p, n_bins=10, sample_weight=None)`

Ensure these are used by the pipeline builder so both scripts benefit.

---

### 2.2 `2-calibrate-logit-model-v1.5.py` — enforce guardrails

Add guardrails before calling pipeline creation:

- Apply `filter_forbidden_features()` to `args.features` (and defaults).
- Enforce fallback rules:
  - if `asof_fallback_days` exists and any > 0 in relevant slices:
    - drop rows or error (choose one, but be consistent)
- Add model selection based on **delta vs baseline**:
  - record baseline LL per window
  - compute delta
  - pick best C by delta
  - fallback if best delta > 0

Add reporting:
- Print removed forbidden features
- Print near-constant features removed
- Print ticker scoping counts (how many mapped to OTHER)
- Report ECE and ECE-Q on all evaluation slices.

---

## 3) Webapp coherence updates (Front-end + Back-end)

Assume the webapp allows:
- choosing training regime (e.g., which day / `T_days`)
- choosing features
- enabling ticker effects / interactions
- reading model metrics and outputs

After code changes above, update UI/Backend to avoid contradictory config.

### 3.1 Configuration schema (single source of truth)

Create or update a shared config schema used by:
- backend API endpoints
- frontend forms
- training script CLI wrapper

Schema fields to support:

**Regime controls**
- `tdays_allowed: number[] | null`
- `asof_dow_allowed: number[] | null` (0=Mon..6=Sun)

**Feature controls**
- `features: string[]`
- `forbidden_numeric_patterns: string[]` (optional surfaced; usually hidden)
- `auto_drop_near_constant: boolean` (default true)

**Ticker controls**
- `use_ticker_intercepts: boolean`
- `ticker_min_support: number`
- `ticker_x_interactions: boolean`
- `ticker_min_support_interactions: number`

**Selection**
- `C_grid: number[]`
- `selection_objective: "delta_vs_baseline"` (default)
- `fallback_to_baseline_if_worse: boolean` (default true)

**Calibration**
- `use_platt: boolean`
- `platt_by_regime: boolean` (optional; default false)
- `ece_bins: number`
- `eceq_bins: number`

#### Backend: validate coherence rules
Enforce:
- If `tdays_allowed` is a single value, automatically remove `T_days`-derived features from the chosen list (or mark as redundant).
- If `asof_dow_allowed` single value, similarly drop `asof_dow` as a feature.
- If user tries to include forbidden numeric features, reject and display which ones were removed.
- If `ticker_x_interactions=true` but support thresholds unmet, disable and warn.

### 3.2 Front-end UX changes

**Feature picker UI**
- Mark forbidden features as:
  - “Not allowed (leakage/diagnostic)”
- Mark redundant features as:
  - “Constant under current regime; auto-removed”

**Regime UI**
- Provide a clear toggle:
  - “Train pooled across all snapshots”
  - “Train a regime-specific model” (choose `T_days` and/or weekday)
- Explain: regime-specific models often calibrate better.

**Model training results**
- Show:
  - Baseline metrics (LL/Brier/ECE/ECE-Q)
  - Model metrics
  - Delta metrics (Model - Baseline)
- Show if fallback-to-baseline was triggered.

### 3.3 Backend endpoints and artifacts

Backend should store training outputs with:
- `model_config.json` (the schema above)
- `metrics.json` including:
  - baseline + model + delta
  - ECE + ECE-Q
  - per-regime metrics if computed
  - list of removed features (forbidden + near-constant)
- model bundle / pickle / joblib

Frontend reads those artifacts and renders:
- “what changed” and “what was removed”
- avoids confusion for users.

---

## 4) Acceptance tests (Codex must add/update)

Add small tests or assertions (even if not a full test suite):

1) **No forbidden numeric features** present in final pipeline feature_names
2) **No forward fallback rows** in training/eval (or rows are dropped deterministically)
3) **Near-constant dropper** removes constant features under single-regime filters
4) **Ticker scoping** maps low-support tickers to OTHER
5) **Selection objective** uses delta vs baseline, and fallback triggers when worse
6) **ECE-Q** computed and included in outputs

---

## 5) Implementation sequencing (do in this order)

1) Update `calibrate_common.py`:
   - feature filter + near-constant dropper + ticker scoping + ECE-Q
2) Update `2-calibrate-logit-model-v1.5.py`:
   - enforce guardrails + delta-vs-baseline selection + reporting
3) Update backend:
   - config schema + validation + training invocation updated for new args/behavior
4) Update frontend:
   - controls + warnings + metrics display
5) Run end-to-end:
   - train a model
   - confirm metrics.json has baseline/model/delta and ECE-Q
   - confirm UI displays “removed forbidden features” etc.

---

## 6) Notes for Codex on “minimal diff” style

- Prefer **small helper functions** and **early validation** rather than refactors.
- Keep existing CLI args; add new ones with sane defaults.
- When a user selects a disallowed feature, **auto-remove** and **warn**, rather than hard failing, unless it implies leakage.

---

## 7) Definition of baseline (keep consistent)

Baseline should remain:
- `pRN` (or `qRN`) depending on pipeline choice
- computed time-safely (as-of)

When comparing:
- Always report baseline metrics on the same slice and weights as the model.

---

## 8) Deliverables Codex must produce

1) Updated scripts:
   - `2-calibrate-logit-model-v1.5.py`
   - `calibrate_common.py`
2) Updated webapp:
   - backend: config validation + updated training invocation + artifact storage
   - frontend: controls + warnings + metrics display
3) Output artifacts from a sample run:
   - `model_config.json`
   - `metrics.json`
   - model bundle
4) A short changelog describing what was fixed and why.

---

**End of brief.**
