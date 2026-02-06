# CODEX TASK: Minimal-diff refactor to make pRN non-primitive + keep pipeline working

## Context / Goal
We have a Python script:
- `/mnt/data/1-option-chain-build-historic-dataset-v1.0.py`
that currently outputs a single model-ready dataset where **pRN is treated as a primitive** (computed inline, used for banding/filters/weights, etc.).

We want a **minimal-diff** refactor to:
1) Split the output into **three conceptual layers** while keeping backward compatibility:
   - **Snapshot (truth layer):** only what is known at `asof_ts` (no pRN, no training bands, no weights).
   - **Derived RN view:** pRN computed from snapshot + chain, with full provenance + quality metrics + versioning.
   - **Training view (model-ready):** joins snapshot + RN view + adds model-specific fields (band flags, weights, any target columns if applicable).
2) Ensure the **rest of the pipeline does not break**.
   - If downstream scripts expect the old single CSV, keep generating it by default (same schema as much as possible), but also generate the new outputs.
   - If unavoidable, apply **minimal changes** to the calibrate step to read the new training view.

Also:
- Update the **webapp UX/UI** minimally so user can run this script and select which dataset to output/use (old vs new views), without breaking existing flows.

## Hard Constraints
- Minimal token usage: implement **minimal diffs**; do not rewrite the whole script.
- Keep existing defaults and CLI behavior working.
- Do NOT introduce hindsight/leakage.
- No paid/proprietary data.
- Code must remain production-safe (good logging, deterministic outputs, no hidden time leakage).
- Any new schema fields must be clearly named and documented inline.

---

## Part A — Script changes (minimal diff)

### A1) Introduce 3 outputs (without breaking old output)
Modify `/mnt/data/1-option-chain-build-historic-dataset-v1.0.py` so it can write:

1) `snapshot.csv`  (new)
2) `prn_view.csv`  (new)
3) `train_view.csv` (new)
4) `legacy_model_ready.csv` (existing output, keep)

Add CLI args (with safe defaults):
- `--out-dir` (default existing behavior)
- `--write-snapshot` (default true)
- `--write-prn-view` (default true)
- `--write-train-view` (default true)
- `--write-legacy` (default true; keeps old pipeline working)
- `--prn-version` (default "v1"; string)
- `--prn-config-hash` (optional; if not provided, compute a stable hash from relevant config knobs)
- `--train-view-name` (default "train_view.csv") for webapp selection

All outputs must be **row-aligned** by a stable key.

### A2) Add a stable primary key for joins
Add a deterministic key column to every output:
- `row_id` = stable hash of: (`asof_ts`, `ticker`, `expiry_date_used`, `K`, `option_type` if present)
Prefer a short hex string (e.g., first 16 chars of sha1).

If script currently doesn’t store `asof_ts`, add it (UTC ISO string).

### A3) Define the Snapshot schema (no pRN)
Snapshot must include (minimum):
- `row_id`
- `asof_ts` (UTC)
- `ticker`
- `expiry_date_requested` (if exists)
- `expiry_date_used`
- `expiry_used_reason` (enum-ish string; e.g. "requested", "fallback_prev", "fallback_next")
- `event_end_ts` (if present) or `event_end_date` + `event_end_tz`
- `K`
- `S0_close_used`
- `S0_close_source` (e.g. "yfinance_raw", "yfinance_adj", "theta", etc.)
- `S0_close_lag_days` (0 if same-day close; >0 if fallback)
- `expiry_close_used` (if computed)
- `expiry_close_source`
- `expiry_close_lag_days`
- `T_days`
- any raw chain metadata already fetched that is time-safe and needed to recompute pRN later (e.g. strike band used, counts, quote source)
Do NOT include:
- `pRN`, `pRN_monotone`, training weights, “in-band”, etc.

### A4) Define the pRN view schema (derived + provenance)
pRN view must include:
- `row_id`
- `pRN_raw` (if computed)
- `pRN` (final used pRN)
- `rn_method` (string)
- `rn_quote_source` (mid/bid/ask/last)
- `rn_monotone_adjusted` (bool)
- `curve_id` (stable id: ticker + expiry_used + asof_date + rn_method + prn_version)
- `prn_version` (from CLI)
- `prn_config_hash` (computed if not provided)
- Quality fields (minimum):
  - `n_calls_used` (or strikes used)
  - `calls_k_min`, `calls_k_max`, `deltaK`
  - `spread_proxy` or `avg_rel_spread` if available (cheap to compute from bid/ask)
  - `curve_smoothness` or `fit_error` if already computed; otherwise omit (don’t invent complex metrics)
This file must be computable solely from snapshot + cached chain data that is time-safe.

### A5) Define Train view schema (model-ready)
Train view = snapshot join prn_view + existing model-ready fields:
- keep existing feature columns (e.g., log_moneyness, x_logit_prn, RV, etc.)
- keep existing banding flags/weights IF the pipeline depends on them
- add metadata columns:
  - `prn_version`, `prn_config_hash`
  - `dataset_view` = "train_view"
  - `build_version` (script version string)
- ensure no column name collisions; prefer prefixing new provenance fields with `rn_` or `prn_`.

### A6) Backward compatibility output
If `--write-legacy=true`, output the same CSV the script used to output (same filename unless `--out-dir` changes).
Implement legacy output as:
- legacy = train_view with minimal renames to match old schema
- do NOT change existing downstream expectations unless unavoidable

If unavoidable, then:
- implement minimal changes to calibrator (Part B).

### A7) No-leakage invariants
Ensure:
- all timestamps used for feature calc are <= asof close timestamp
- any fallback close selection is logged via `*_lag_days` and `*_source`
- do NOT use future data to “improve” pRN

---

## Part B — Minimal calibrator changes (only if needed)
If downstream model calibration currently reads the legacy CSV and breaks, do the smallest change:
- accept a new CLI arg: `--dataset-view {legacy,train_view}`
- if `train_view`: read `train_view.csv` and map expected feature columns:
  - if old code expects `pRN`, it still exists in train_view
  - if old expects `x_logit_prn`, ensure it is present or compute it safely from pRN within calibrator (using stable clipping)
- keep old default behavior untouched.

Do not refactor calibrator heavily.

---

## Part C — Webapp UX/UI minimal update
Goal: user can run dataset build from the webapp and choose which view to use, without breaking existing flows.

### C1) Add a selector in UI
Add UI controls (minimal):
- checkbox: "Generate snapshot.csv" (default ON)
- checkbox: "Generate prn_view.csv" (default ON)
- checkbox: "Generate train_view.csv" (default ON)
- checkbox: "Generate legacy CSV (compat)" (default ON)
- dropdown: "Dataset to use for training" = {legacy_model_ready.csv, train_view.csv}
- text input (optional): "pRN version" default "v1"

### C2) Update backend invocation
Wherever the webapp triggers the script, pass flags based on UI selection.
Ensure paths match existing pipeline directories.

### C3) Ensure downstream training uses chosen file
If webapp has a training step:
- when user selects "train_view.csv", pass that path to calibrator
- else keep legacy path.

Do NOT redesign the whole webapp; minimal wiring.

---

## Part D — Acceptance checks (must pass)
Add simple sanity checks (asserts/logging) without expensive compute:
1) `row_id` uniqueness within each output.
2) Snapshot contains no `pRN` columns.
3) `train_view` row count matches snapshot row count (unless rows intentionally filtered; if filtered, log counts with reasons).
4) Legacy output is identical schema (or nearly) to prior version; if any column name changes, add explicit compatibility mapping.
5) Script runs end-to-end with existing pipeline commands.

---

## Output required from you (Code changes)
- Provide a minimal diff implementation.
- Do not delete existing logic; wrap/redirect it into view builders.
- Keep filenames stable by default; new outputs go alongside old output.
- Add concise inline comments where new concepts are introduced.

DONE means: existing pipeline still works, but we now also have snapshot/prn_view/train_view + UI can select them.