# Claude Code Prompt — Upgrade Auto-Tuner for pHAT Calibrator (Smart Search, Minimal Tokens)

You are Claude Code acting as a senior quant/ML engineer. Make **minimal-diff** edits to the auto-tuner script:

- `2-auto-calibrate-logit-model.py`

Goal: improve the tuner so it can reliably find the best configuration for a **fixed dataset + fixed time regime** (e.g., `T_days=4`, `asof_dow=Mon`) while staying time-safe and not exploding compute.

## Non-negotiables
- Do not introduce leakage. Calibrators (Platt/Isotonic) must fit only on past CALIB slice, never VAL/TEST.
- Keep existing CLI compatibility as much as possible.
- Do not rename existing outputs/paths. Only add fields.
- Keep runtime bounded: target **<= 200–400 trials** for a full search unless user asks otherwise.

---

## 1) Fix feature search granularity (avoid redundant combos)

Current issue: feature groups add **multiple collinear features at once** (e.g., `rv20` + `rv20_sqrtT`), causing redundancy and instability.

### Implement `FEATURE_CHOICES` instead of `FEATURE_GROUPS`
Replace group inclusion logic with **choice sets** per “family”. Example:

```python
FEATURE_CHOICES = {
  "volatility": [[], ["rv20"], ["rv20_sqrtT"]],
  "moneyness":  [[], ["log_m_fwd"], ["abs_log_m_fwd"], ["log_m_fwd","abs_log_m_fwd"]],
  "quality":    [[], ["log_rel_spread"]],
  "trend":      [[], ["dividend_yield"]],
  "rv":         [[], ["rv20"], ["rv20_sqrtT"]],  # if you want separate naming, otherwise keep in volatility
}
```

Rules:
- The tuner must be able to choose **one** volatility variant (or none), not always both.
- For single-regime runs (e.g., `--tdays-allowed 4`), do not include time-derived variants that become constants (let the calibrator auto-drop near-constant, but also avoid selecting them).

### Update feature builder
Implement helper:

- `build_features_from_choices(config) -> list[str]`

Always include `x_logit_prn`. Then append selected lists from each choice family.

---

## 2) Add interaction search: `x_m` (high ROI)

Add ability to enable engineered interaction features:
- `x_m` (x_logit_prn * moneyness)
- optional `x_abs_m`

In tuner, add boolean flags:
- `enable_x_m` (default True when a moneyness feature is chosen, else False)
- `enable_x_abs_m` (default False)

When enabled, pass corresponding `--features x_m` and/or `x_abs_m` to the calibrator. Do not compute inside tuner; just request feature names.

---

## 3) Add calibration mode search beyond Platt

Current: `["none", "platt"]`.

Add:
- `"isotonic"` (only if calibrator supports; otherwise gate behind `--allow-isotonic` and skip if unsupported)

Implementation:
- Extend `CALIBRATION_OPTIONS` to include `"isotonic"`.
- Ensure tuner passes `--calibrate isotonic` to calibrator.
- Add safety: if calibrator returns an error for isotonic (unsupported or insufficient calib rows), treat as failed trial and continue.

---

## 4) Replace greedy coordinate descent with beam search (small K)

Current coordinate updates miss synergies between features/interactions/C/calibration.

Implement **beam search** with small width:
- `--beam-width` default 5
- `--max-trials` default 250 (or keep existing budgets)

Algorithm:
1. Start with `beam = [base_config]`
2. For each “family expansion step” (C, features, interactions, calibration, group_reweight, ticker options, decay):
   - For each config in beam:
     - generate candidate variants for that family (small set)
   - Evaluate all candidates (deduplicate)
   - Keep top K by objective (see section 5)
3. Stop early if no improvement across a full pass or if max trials reached.

This keeps compute bounded but captures interactions between choices.

---

## 5) Improve objective scoring (more robust than single metric)

Primary objective must stay:
- **avg delta logloss vs baseline** (lower is better)

Add tie-breakers:
- delta Brier (lower)
- delta ECE-Q (lower)
- complexity penalty (very small): `+ 0.0005 * n_features + 0.001 * (ticker_interactions_enabled)`

Implement `score_trial(metrics) -> float` using:
- `score = delta_logloss`
- if within epsilon of best, use tie-breakers.

Do not overcomplicate; keep deterministic.

---

## 6) Add group reweighting as a searchable family

Add `GROUP_REWEIGHT_OPTIONS = ["none", "chain"]`

If calibrator supports `--group-reweight`, include this in tuner search space.
If calibrator errors, mark trial failed.

---

## 7) Caching + deduplication (big speed win, minimal code)

### Deduplicate configs
Implement a stable hash key:
- sorted feature list
- key flags (C, calibrate mode, ticker options, group reweight, decay params, regime filters)

Skip evaluating duplicate keys.

### Cache results
Keep `results_cache[key] = metrics` in-memory for the run.
Optionally persist to JSONL in `--out-dir` so reruns can reuse if same key found.

---

## 8) Logging / output additions (keep small)

For each trial, log:
- key config summary (C, calibrate mode, feature list, interactions flags, group reweight, ticker flags)
- objective score + delta metrics
- trial status (ok/failed)

At the end:
- print best config
- write `best_config.json` and `leaderboard.csv` (top N = 25)

Do not change existing output names unless none exist; otherwise add these files.

---

## 9) Keep CLI coherent with fixed regime usage

Add (if not present) passthrough args that define the regime, and ensure they are constant during tuning:
- `--tdays-allowed ...`
- `--asof-dow-allowed ...`
- `--train-tickers ...` / `--foundation-tickers ...` if used

Tuner should not vary the regime unless explicitly asked.

---

## Acceptance checks
- The tuner can choose `rv20` alone without `rv20_sqrtT` and vice versa.
- It can include `x_m` when moneyness present.
- Beam search evaluates combinations (C + feature + calibration) rather than one-at-a-time only.
- Total evaluations respect `--max-trials`.
- No leakage logic added.
- Best config is saved and reproducible.

Implement now with minimal changes.
