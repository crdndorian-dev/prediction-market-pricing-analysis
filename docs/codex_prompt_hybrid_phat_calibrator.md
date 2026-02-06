# Codex Prompt — Hybrid Time‑Safe pHAT Calibrator Improvements (Minimal Diff)

You are Codex acting as a senior quant/ML engineer. Modify **only** the following files with **minimal diff** while keeping all existing time‑safety/leakage guardrails:

- `2-calibrate-logit-model-v1.5.py`
- `calibrate_common.py`
- Webapp backend + frontend (where config is defined/validated and training is triggered, and where metrics are displayed)

## Objective

Improve out‑of‑sample probabilistic accuracy (LogLoss, Brier, ECE/ECE‑Q) while preserving:
- strict time‑series split discipline
- forward‑fallback row removal (no look‑ahead)
- forbidden diagnostic numeric feature bans
- near‑constant feature auto‑drop
- delta‑vs‑baseline hyperparam selection + fallback-to-baseline if worse
- ticker scoping to `OTHER` by minimum support + cautious interactions
- ECE‑Q reporting

Add two high‑ROI improvements:
1) **`x_m` interaction feature** (and optional `x_abs_m`) using a single consistent moneyness definition
2) **Group reweighting by chain snapshot** to prevent “strike spam” dominating training

Also add **optional** data filters (OFF by default) that are transparent and logged:
- max absolute moneyness filter
- pRN extremes filter

Update the webapp UI/validation so it is coherent with new arguments and behavior.

---

## 1) Feature Engineering: add `x_m` (+ optional `x_abs_m`)

### 1.1 Choose a single moneyness definition
In `calibrate_common.py`, ensure there is one consistent moneyness column used for interactions:

- Prefer `log_m_fwd` if present; else fallback to `log_m` if present.
- Do **not** mix both simultaneously.

Implement helper:
- `resolve_moneyness_column(df) -> str | None` returning `"log_m_fwd"` or `"log_m"` or None.

### 1.2 Engineer interaction features
In `ensure_engineered_features()` add:
- `x_m = x_logit_prn * m`
- optionally `x_abs_m = x_logit_prn * abs(m)` if requested

Rules:
- Only create if both `x_logit_prn` and moneyness exist and are finite.
- Do not create NaN explosions; coerce to numeric, keep NaNs where inputs missing.

Expose these as selectable features.

### 1.3 Default feature set update
In `2-calibrate-logit-model-v1.5.py`, update the default `--features` list (or default features chosen when `--features` absent) to include:
- `x_logit_prn`
- the chosen moneyness column (whichever exists)
- `x_m`
Optionally include `x_abs_m` only if explicitly enabled.

Add CLI flags:
- `--enable-x-abs-m` (default false)

Keep forbidden feature filter in place (do not allow diagnostic numerics).

---

## 2) Data Treatment: implement chain-snapshot group reweighting

### 2.1 Define a chain snapshot group id
Add a helper in `calibrate_common.py`:

- `build_chain_group_id(df) -> pd.Series[str]`

Use the most stable available columns to represent a “chain snapshot”:
Prefer:
- `ticker`
- `asof_date` (or `asof_ts` if available)
- `expiry_date`
If `asof_date` is not available, fall back to `week_friday` (or equivalent) + `T_days`.

Implementation must be robust to missing columns; if group id cannot be built, disable and warn.

### 2.2 Apply group reweighting to sample weights
Add function:
- `apply_group_reweight(weights: np.ndarray, group_id: pd.Series, mask: np.ndarray) -> np.ndarray`

Behavior:
- On TRAIN_FIT only: scale weights so that each group contributes equal total weight.
  - For each group g in TRAIN_FIT, multiply each row weight by `1 / sum_w_in_group`.
  - Then renormalize TRAIN_FIT weights to keep mean weight ~ 1 (optional), but preserve relative weights outside TRAIN_FIT unchanged.
- For VAL/TEST metrics: keep raw weights (as current design), but allow reporting with both raw and reweighted if already supported; otherwise keep metrics as-is.

### 2.3 New CLI args
In `2-calibrate-logit-model-v1.5.py` add:
- `--group-reweight` with choices: `none` (default), `chain`

When `chain`:
- build group id
- apply reweighting to TRAIN_FIT weights only before model fitting
- log number of groups and median group size

---

## 3) Optional filters (OFF by default, transparent)

Add CLI args in `2-calibrate-logit-model-v1.5.py`:
- `--max-abs-logm` (float, default None)
- `--drop-prn-extremes` (bool, default false)
- `--prn-eps` (float, default 1e-4)

Behavior:
- Apply filters **before** split masks are created (or consistently across all splits).
- Log rows dropped and remaining rows per split.
- `--max-abs-logm` uses whichever moneyness column was resolved; filter rows with `abs(m) <= max_abs_logm`.
- `--drop-prn-extremes` drops rows with `pRN < prn_eps` or `pRN > 1 - prn_eps`.

Do not enable by default.

---

## 4) Keep and extend existing safety/selection behavior

Do not remove:
- forward fallback row removal (asof_fallback_days > 0)
- forbidden diagnostic numeric feature bans + binary flags
- near-constant feature auto-drop
- ticker scoping with `OTHER`
- rolling selection by delta vs baseline + fallback-to-baseline
- ECE + ECE-Q reporting

Extend reporting:
- Print which moneyness column was used.
- Print whether `x_m` / `x_abs_m` were included.
- Print group reweighting status + number of groups.

---

## 5) Webapp updates (backend + frontend)

### 5.1 Config schema and validation
Update webapp config to include:
- `enable_x_abs_m` (bool)
- `group_reweight: "none" | "chain"`
- `max_abs_logm: number | null`
- `drop_prn_extremes: boolean`
- `prn_eps: number`
- `moneyness_mode: "auto"` (default; backend resolves `"log_m_fwd"` vs `"log_m"`)

Validation rules:
- If forbidden features are selected, auto-remove and show warning.
- If `max_abs_logm` set but moneyness column missing, reject with clear message.
- If group id cannot be built, auto-fallback to `none` and warn.

### 5.2 Training invocation
Ensure backend passes new CLI args to training script.

### 5.3 Metrics display
Frontend should display:
- baseline vs model metrics + delta (LogLoss/Brier/ECE/ECE-Q)
- “Removed features” (forbidden + near-constant)
- “Data filters applied” + rows dropped
- “Group reweighting” status
- “Moneyness column used” + whether x_m / x_abs_m included

---

## 6) Acceptance checks (must pass)

Add lightweight assertions/log checks:
1. If `--group-reweight chain` enabled, TRAIN_FIT group totals are equalized (within numeric tolerance).
2. `x_m` is present and finite when inputs exist.
3. Filters drop rows as expected and report counts.
4. No forbidden numeric features appear in final feature list.
5. Existing time-safety checks (no forward fallback rows) remain enforced.
6. Webapp can train with default settings and show new fields in UI.

---

## 7) Minimal-diff constraints

- Do not rename existing public functions/CLI flags/outputs unless required for correctness.
- Keep file output formats stable; only add new fields.
- Keep defaults conservative: group reweighting OFF, filters OFF, x_abs_m OFF, x_m ON only if moneyness exists.

Implement now.
