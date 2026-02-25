# Model v2.0 Implementation Plan
## Hierarchical Probabilistic Model with pRN + Polymarket Integration

**Created:** 2026-02-12
**Version:** v2.0.0
**Status:** Planning Phase

---

## Table of Contents

1. [Ultrathink Plan](#a-ultrathink-plan)
2. [Implementation Steps](#b-implementation-steps-ordered-checklist)
3. [Code Patch](#c-code-patch-unified-diff-for-v20-script)
4. [Backend Changes](#d-backend-changes-minimal-diffs)
5. [Frontend Changes](#e-frontend-changes-minimal-diffs)
6. [Validation Plan](#f-validation-plan)

---

## A) ULTRATHINK PLAN

### 1. Summary: What v1.6 Currently Does

v1.6 is an **orchestrator/wrapper script** that can run three workflows:

- **Calibrate mode**: Delegates to v1.5 pRN calibrator which trains logistic regression on option chain data (pRN→pRW mapping), with ticker intercepts, optional interactions, Platt calibration, and rolling-window validation.

- **Mixed mode**: Delegates to mixed-model trainer which trains Ridge regression on Polymarket+pRN decision features for residual or blend modeling.

- **Two-stage overlay mode** (implemented directly in v1.6): Time-safe meta-learning that:
  - (a) trains base pRN model on pre-overlap data
  - (b) trains stage2 model combining p_base + Polymarket features on overlap period
  - Key join keys: `(ticker, threshold, expiry_date, snapshot_date)`
  - Uses week-based train/test splits
  - Outputs: TwoStageBundle with metrics (logloss, Brier) and metadata tracking

**Current limitations for v2.0 goal:**
- PM features only used in overlay mode (short 3-month window)
- No explicit edge/mispricing output
- No pretrain/finetune workflow to leverage long options history
- Limited control over feature selection and alignment strategy

---

### 2. New Target Training Objectives in v2.0

**Primary objective:** Predict P(outcome) - real-world probability that event resolves YES.

**Secondary objective:** Estimate edge/mispricing:
```
edge = P_predicted(outcome) - P_PM(implied)
```
where P_PM is Polymarket implied probability (pm_mid). Positive edge indicates PM is underpricing the YES outcome (potential long opportunity).

**Key outputs:**
- Probability prediction with confidence intervals
- Edge estimate (for backtesting/signal generation)
- Feature attributions (which features drive prediction vs PM baseline)

---

### 3. Data Regime Strategy (Key Constraint)

**Challenge:**
- Polymarket: 3 months history, sparse coverage (not all tickers/strikes/weeks)
- Option chains: 2 years history, dense coverage

**Proposed strategy: Hierarchical Pretrain-Finetune**

**Phase 1 (Pretrain):** Long-horizon base model (options-only, 2 years)
- Train pRN → pRW mapping on full options history
- Learn stable feature relationships (log_m, T_days, rv20, ticker effects)
- Output: `base_model.joblib` (FinalModelBundle)
- No PM features

**Phase 2 (Finetune/Meta-learn):** Short-horizon overlay (PM+options, 3 months overlap)
- Load pretrained base model
- Generate p_base predictions on overlap window
- Train meta-model: `p_final = f(p_base, PM_features, option_features, interactions)`
- Use logistic offset or stacking architecture
- Output: `two_stage_model.joblib` (TwoStageBundle)

**Alternative: Joint training** (if overlap window sufficiently large)
- Train single model on PM+options features from scratch
- Uses only overlap window
- Simpler but sacrifices long-horizon information

**Time-safety guarantee:**
- Base model: train on data strictly before overlap_start_date
- Meta model: train/test split within overlap window using week-based splits
- No future information leakage

---

### 4. Dataset Alignment

**Join keys:**
```python
keys = ["ticker", "threshold", "expiry_date", "snapshot_date"]
```

**Alignment rules:**

1. **Temporal constraint:** `option_snapshot_ts <= pm_snapshot_ts` (options data must be available at or before PM snapshot)

2. **Expiry matching:** Exact match on expiry_date (both use same resolution date)

3. **Strike matching:** `abs(option_K - pm_threshold) < epsilon` (allow small numerical tolerance)

4. **DTE computation:** `DTE = (expiry_date - snapshot_date).days` - must be consistent across both datasets

**De-duplication:** Drop duplicate (ticker, threshold, expiry, snapshot) rows, keeping most recent by timestamp.

**Missing data handling:**
- If PM feature missing: fall back to p_base (from pretrained model)
- If option feature missing: impute using median (training set) or mark as NaN-prone

---

### 5. Splitting / Evaluation (Time-Safe, Era-Aware)

**Era definition:** `era = (ticker, expiry_date)` - each unique event is an era.

**Split strategy (within overlap window):**

```
Full timeline: [-----pretrain (2yr options)-----][-----overlap (3mo PM+options)-----]
                                                  [--train--][embargo][--test--]
```

**Pretrain split (options-only, 2 years):**
- TRAIN: all weeks before `overlap_start - 4 weeks`
- VAL: 4 weeks before overlap_start (for hyperparameter tuning)
- TEST: not used (final test happens in overlap window)

**Overlap split (PM+options, 3 months):**
- Use week_friday as grouping variable
- TRAIN: first 60-70% of weeks
- EMBARGO: 1-2 weeks buffer (avoid same-event leakage)
- TEST: last 20-30% of weeks

**Leakage prevention:**
- Ensure no (ticker, expiry) appears in both train and test
- If overlap insufficient, relax to weekly splits (accept some same-ticker leakage but different expiries)
- Always: snapshot_date[test] > snapshot_date[train]

**Validation metrics:**
- Log loss (primary)
- Brier score
- ECE (expected calibration error, equal-mass bins)
- Edge stability (std dev of edge across test weeks)

---

### 6. Feature Governance

**Feature categories:**

**A. pRN-derived (from options):**
- Core: `pRN`, `x_logit_prn`, `log_m`, `abs_log_m`, `T_days`, `rv20`
- Engineered: `log_m_fwd`, `abs_log_m_fwd`, `rv20_sqrtT`, `x_m`, `x_abs_m`
- Quality: `rel_spread_median`, `chain_used_frac`, `had_fallback`

**B. PM-derived:**
- Core: `pm_mid`, `x_logit_pm`, `pm_spread`, `pm_liquidity_proxy`
- Momentum: `pm_momentum_1h`, `pm_momentum_1d`
- Volatility: `pm_volatility`
- Time: `pm_time_to_resolution` (days)

**C. Interaction features:**
- `x_logit_prn * x_logit_pm` (agreement/divergence)
- `ticker * x_logit_pm` (ticker-specific PM adjustments)
- `abs(pRN - pm_mid)` (discrepancy magnitude)

**Forbidden features (prevent leakage):**
- Any feature containing future information
- Outcome-dependent features
- Features with >90% NaN in training set

**Feature selection:**
- Use `filter_forbidden_features()` from calibrate_common
- Enforce whitelist via `--numeric-features` and `--pm-features` CLI args
- Validate availability via `validate_feature_availability()` against schema contract

---

### 7. Artifacts

**Primary output:** `two_stage_model.joblib` (TwoStageBundle)
- base_bundle: FinalModelBundle (pretrained on options)
- stage2_pipeline: sklearn Pipeline (meta-model)
- stage2_feature_cols: List[str]
- pm_primary_col: str
- platt_calibrator: Optional[LogisticRegression]

**Metrics:** `metrics.csv`
```csv
split,model,n,logloss,brier,ece,edge_mean,edge_std
train,baseline_pRN,5000,0.35,0.12,0.08,0.00,0.00
train,baseline_PM,5000,0.28,0.10,0.06,0.00,0.00
train,two_stage,5000,0.24,0.08,0.04,0.02,0.03
test,baseline_pRN,1500,0.38,0.14,0.09,0.00,0.00
test,baseline_PM,1500,0.30,0.11,0.07,0.00,0.00
test,two_stage,1500,0.26,0.09,0.05,0.03,0.04
```

**Metadata:** `metadata.json`
```json
{
  "script_version": "v2.0.0",
  "training_mode": "pretrain_finetune",
  "feature_sources": "both",
  "pretrain_window": {"start": "2024-01-01", "end": "2025-10-31", "rows": 50000},
  "overlap_window": {"start": "2025-11-01", "end": "2026-02-01", "rows": 8000},
  "pm_coverage_pct": 0.65,
  "train_test_split": {"train_weeks": [45, 46, 47], "test_weeks": [52, 53, 1]},
  "features": {
    "options": ["pRN", "x_logit_prn", "log_m_fwd", "T_days", "rv20"],
    "polymarket": ["pm_mid", "x_logit_pm", "pm_spread", "pm_momentum_1h"],
    "interactions": ["x_logit_prn_x_pm"]
  },
  "git_commit": "abc123...",
  "created_at_utc": "2026-02-12T10:30:00Z"
}
```

**Edge outputs:** `edge_predictions.csv` (for backtests page)
```csv
ticker,threshold,expiry_date,snapshot_date,p_base,p_pm,p_final,edge,edge_lower,edge_upper
SPY,580,2026-02-21,2026-02-14,0.45,0.52,0.48,0.04,0.02,0.06
```

---

## B) IMPLEMENTATION STEPS (Ordered Checklist)

### Step 1: Create v2.0 Script Structure
- [ ] Copy `03-calibrate-logit-model-v1.6.py` → `03-calibrate-logit-model-v2.0.py`
- [ ] Update script header docstring with v2.0 objectives
- [ ] Set `SCRIPT_VERSION = "v2.0.0"`

### Step 2: Add New CLI Arguments
- [ ] `--training-mode`: {pretrain, finetune, joint, two_stage} (default: two_stage)
- [ ] `--feature-sources`: {options, pm, both} (default: both)
- [ ] `--compute-edge`: bool flag (default: True)
- [ ] `--pm-overlap-window`: str, e.g., "90days" or "3months" (default: "90days")
- [ ] `--numeric-features`: comma-separated list (override defaults)
- [ ] `--pm-features`: comma-separated list (override defaults)
- [ ] `--edge-output-path`: optional path for edge predictions CSV

### Step 3: Implement Pretrain Workflow
- [ ] Add `_run_pretrain()` function
  - [ ] Load options dataset (long history, 2 years)
  - [ ] Filter to pre-overlap period: `snapshot_date < overlap_start - 4 weeks`
  - [ ] Train base model using v1.5 calibrator
  - [ ] Save `pretrain_model.joblib` to subdirectory
  - [ ] Return base model path

### Step 4: Implement Finetune/Meta-learn Workflow
- [ ] Add `_run_finetune()` function
  - [ ] Load pretrained base model
  - [ ] Load PM dataset (overlap window)
  - [ ] Load options dataset (overlap window)
  - [ ] Align datasets on join keys with time-safety check
  - [ ] Generate p_base predictions
  - [ ] Build meta-model features: [p_base, PM_features, option_features]
  - [ ] Train stage2 pipeline (logistic or ridge)
  - [ ] Save `two_stage_model.joblib`

### Step 5: Implement Joint Training Workflow
- [ ] Add `_run_joint()` function
  - [ ] Load aligned PM+options dataset (overlap window only)
  - [ ] Train single model from scratch with both feature sets
  - [ ] Save `joint_model.joblib`

### Step 6: Add Edge Computation Module
- [ ] Add `_compute_edge()` function
  - [ ] Input: df with [p_final, pm_mid]
  - [ ] Compute: `edge = p_final - pm_mid`
  - [ ] Compute confidence bands using bootstrap or calibration uncertainty
  - [ ] Return: df with [edge, edge_lower, edge_upper]
- [ ] Add `_save_edge_predictions()` function
  - [ ] Output CSV with (ticker, threshold, expiry, snapshot, p_base, p_pm, p_final, edge)

### Step 7: Enhance Dataset Alignment Logic
- [ ] Update `_build_overlap()` function
  - [ ] Add temporal constraint: `prn_ts <= pm_ts`
  - [ ] Add DTE consistency check
  - [ ] Add PM coverage metrics to metadata
  - [ ] Handle missing PM features gracefully (fallback to p_base)

### Step 8: Update Metrics Reporting
- [ ] Extend `_compute_metrics()` to include:
  - [ ] ECE (equal-mass bins)
  - [ ] Edge statistics (mean, std, percentiles)
  - [ ] PM vs model comparison (relative improvement)
- [ ] Add per-ticker breakdown (optional, via `--per-ticker-metrics` flag)

### Step 9: Update Metadata Tracking
- [ ] Extend `metadata.json` to capture:
  - [ ] Training mode, feature sources
  - [ ] Pretrain window (start, end, rows)
  - [ ] Overlap window (start, end, rows, PM coverage %)
  - [ ] Train/test split (week lists)
  - [ ] Feature lists (options, PM, interactions)
  - [ ] Git commit hash

### Step 10: Add Leakage Validation
- [ ] Add `_validate_time_safety()` function
  - [ ] Assert: all pretrain dates < overlap_start
  - [ ] Assert: all train dates < test dates
  - [ ] Assert: no (ticker, expiry) overlap between train/test
  - [ ] Run automatically before model training

### Step 11: Backward Compatibility
- [ ] Keep existing `--model-kind` argument functional
- [ ] Map old flags to new workflow:
  - [ ] `--model-kind=calibrate` → `--training-mode=pretrain` (if no PM data)
  - [ ] `--model-kind=mixed` → `--training-mode=joint`
  - [ ] `--two-stage-mode` → `--training-mode=two_stage`
- [ ] Emit deprecation warnings for old flags

### Step 12: Testing & Validation
- [ ] Unit test: dataset alignment with time constraints
- [ ] Unit test: edge computation
- [ ] Integration test: pretrain workflow end-to-end
- [ ] Integration test: finetune workflow end-to-end
- [ ] Leakage test: verify no future info in any split

---

## C) CODE PATCH (Unified Diff for v2.0 Script)

### Conceptual Diff Highlights

```diff
--- a/src/scripts/03-calibrate-logit-model-v1.6.py
+++ b/src/scripts/03-calibrate-logit-model-v2.0.py
@@ -1,15 +1,20 @@
 #!/usr/bin/env python3
 """
-03-calibrate-logit-model-v1.6.py
+03-calibrate-logit-model-v2.0.py

-Wrapper script that can run:
-- The v1.5 pRN calibrator (default)
-- The mixed Polymarket + pRN model trainer
-- Or both, sequentially
+Hierarchical probabilistic model trainer combining:
+- pRN-derived features (option chain history, 2 years)
+- Polymarket features (implied probabilities + momentum, 3 months)
+
+Training modes:
+- pretrain: Base model on options-only (long history)
+- finetune: Meta-model on PM+options (overlap window)
+- joint: Single model on PM+options (overlap window)
+- two_stage: Backward-compatible v1.6 mode
+
+Outputs: P(outcome) predictions + edge estimates (vs PM implied)
 """

-SCRIPT_VERSION = "v1.6.0"
+SCRIPT_VERSION = "v2.0.0"

+# New constants
+OVERLAP_WINDOW_DAYS = 90  # Default PM overlap window
+DEFAULT_EDGE_OUTPUT = "edge_predictions.csv"
+
+# Feature sets
+DEFAULT_PRN_FEATURES = [
+    "pRN", "x_logit_prn", "log_m_fwd", "abs_log_m_fwd",
+    "T_days", "rv20", "rv20_sqrtT", "rel_spread_median"
+]
+DEFAULT_PM_FEATURES = [
+    "pm_mid", "x_logit_pm", "pm_spread", "pm_liquidity_proxy",
+    "pm_momentum_1h", "pm_momentum_1d", "pm_time_to_resolution"
+]

+def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
+    parser = argparse.ArgumentParser(...)
+
+    # New v2.0 arguments
+    parser.add_argument(
+        "--training-mode",
+        choices=["pretrain", "finetune", "joint", "two_stage"],
+        default="two_stage",
+        help="Training workflow mode.",
+    )
+    parser.add_argument(
+        "--feature-sources",
+        choices=["options", "pm", "both"],
+        default="both",
+        help="Which feature sets to use.",
+    )
+    parser.add_argument(
+        "--compute-edge",
+        action="store_true",
+        default=True,
+        help="Compute and save edge predictions.",
+    )
+    parser.add_argument(
+        "--pm-overlap-window",
+        default="90days",
+        help="PM overlap window (e.g., '90days', '3months').",
+    )
+    parser.add_argument(
+        "--numeric-features",
+        default=None,
+        help="Override numeric features (comma-separated).",
+    )
+    parser.add_argument(
+        "--pm-features",
+        default=None,
+        help="Override PM features (comma-separated).",
+    )
+    parser.add_argument(
+        "--edge-output-path",
+        default=None,
+        help="Path for edge predictions CSV.",
+    )

+def _parse_overlap_window(window_str: str) -> int:
+    """Parse overlap window string to days."""
+    if window_str.endswith("days"):
+        return int(window_str.replace("days", ""))
+    if window_str.endswith("months"):
+        return int(window_str.replace("months", "")) * 30
+    raise ValueError(f"Invalid overlap window format: {window_str}")

+def _validate_time_safety(
+    pretrain_df: pd.DataFrame,
+    overlap_df: pd.DataFrame,
+    overlap_start: datetime.date,
+) -> None:
+    """Validate no time leakage between pretrain and overlap."""
+    pretrain_max = pretrain_df["snapshot_date"].max()
+    overlap_min = overlap_df["snapshot_date"].min()
+
+    if pretrain_max >= overlap_start:
+        raise ValueError(
+            f"Pretrain data leaks into overlap window: "
+            f"pretrain_max={pretrain_max}, overlap_start={overlap_start}"
+        )

+def _compute_edge(
+    df: pd.DataFrame,
+    *,
+    p_final_col: str = "p_final",
+    pm_col: str = "pm_mid",
+    bootstrap_iters: int = 1000,
+) -> pd.DataFrame:
+    """Compute edge with confidence intervals."""
+    result = df.copy()
+    p_final = pd.to_numeric(result[p_final_col], errors="coerce").to_numpy()
+    p_pm = pd.to_numeric(result[pm_col], errors="coerce").to_numpy()
+
+    edge = p_final - p_pm
+    result["edge"] = edge
+
+    # Bootstrap confidence intervals
+    edges_boot = []
+    rng = np.random.RandomState(42)
+    for _ in range(bootstrap_iters):
+        idx = rng.choice(len(edge), size=len(edge), replace=True)
+        edges_boot.append(np.mean(edge[idx]))
+
+    result["edge_lower"] = np.percentile(edges_boot, 2.5)
+    result["edge_upper"] = np.percentile(edges_boot, 97.5)
+
+    return result

+def _run_pretrain(
+    *,
+    prn_csv: Path,
+    out_dir: Path,
+    overlap_start: datetime.date,
+    calibrate_args: List[str],
+) -> Path:
+    """Run pretrain workflow: train base model on options-only, pre-overlap data."""
+    print("[pretrain] Loading options dataset for pretrain...")
+    prn_full = _load_prn_full(prn_csv)
+
+    # Filter to pre-overlap (leave 4-week buffer)
+    cutoff = overlap_start - timedelta(weeks=4)
+    pretrain_df = prn_full[prn_full["snapshot_date"] < cutoff].copy()
+
+    if pretrain_df.empty:
+        raise ValueError("No pre-overlap data available for pretrain.")
+
+    print(f"[pretrain] Pretrain rows: {len(pretrain_df)}, cutoff: {cutoff}")
+
+    # Save and train
+    pretrain_csv = out_dir / "pretrain_data.csv"
+    pretrain_df.to_csv(pretrain_csv, index=False)
+
+    pretrain_out_dir = out_dir / "pretrain"
+    pretrain_args = _replace_arg_value(calibrate_args, "--csv", str(pretrain_csv))
+    pretrain_args = _replace_arg_value(pretrain_args, "--out-dir", str(pretrain_out_dir))
+
+    result = _run_calibrate(pretrain_args)
+    if result.returncode != 0:
+        raise RuntimeError("Pretrain base model training failed.")
+
+    model_path = pretrain_out_dir / "final_model.joblib"
+    print(f"[pretrain] Saved base model: {model_path}")
+    return model_path

+def _run_finetune(
+    *,
+    pretrain_model_path: Path,
+    prn_csv: Path,
+    pm_csv: Path,
+    out_dir: Path,
+    overlap_start: datetime.date,
+    overlap_days: int,
+    calibrate_args: List[str],
+    label_col: Optional[str],
+    numeric_features: List[str],
+    pm_features: List[str],
+) -> Path:
+    """Run finetune workflow: train meta-model on PM+options overlap."""
+    # Implementation similar to existing two-stage overlay
+    # with enhanced feature engineering and edge computation
+    pass

+def main() -> None:
+    args, calibrate_args = _parse_args()
+
+    training_mode = args.training_mode
+    feature_sources = args.feature_sources
+    compute_edge = args.compute_edge
+    overlap_window_days = _parse_overlap_window(args.pm_overlap_window)
+
+    # Execute workflow based on training_mode
+    if training_mode == "pretrain":
+        model_path = _run_pretrain(...)
+    elif training_mode == "finetune":
+        pretrain_path = _run_pretrain(...)
+        model_path = _run_finetune(...)
+    elif training_mode == "joint":
+        model_path = _run_joint(...)
+    elif training_mode == "two_stage":
+        _run_two_stage_overlay(...)  # Legacy mode
+
+    # Compute edge if requested
+    if compute_edge and model_path:
+        edge_df = _compute_edge(test_df)
+        edge_df.to_csv(out_dir / "edge_predictions.csv", index=False)
```

---

## D) BACKEND CHANGES (Minimal Diffs)

### D.1 New API Endpoint: `/api/calibrate-models/train-v2`

**File:** `src/webapp/backend/app/api/calibrate_models.py`

```python
@router.post("/train-v2")
async def train_model_v2(request: TrainModelV2Request):
    """
    Train probabilistic model v2.0 with PM+options integration.

    Request body:
    {
        "training_mode": "finetune",  # pretrain | finetune | joint | two_stage
        "feature_sources": "both",     # options | pm | both
        "prn_csv": "path/to/options.csv",
        "pm_csv": "path/to/pm_features.parquet",
        "out_dir": "path/to/output",
        "overlap_window": "90days",
        "numeric_features": ["pRN", "log_m_fwd", "T_days"],
        "pm_features": ["pm_mid", "pm_momentum_1h"],
        "compute_edge": true,
        "test_weeks": 20,
        "random_state": 7
    }

    Returns:
    {
        "run_id": "auto-run-2026-02-12T1045",
        "status": "success",
        "model_path": "src/data/models/auto-run-2026-02-12T1045/finetune_model.joblib",
        "metrics_path": "src/data/models/auto-run-2026-02-12T1045/metrics.csv",
        "edge_path": "src/data/models/auto-run-2026-02-12T1045/edge_predictions.csv",
        "metadata": {...}
    }
    """
    # Implementation delegates to v2.0 script via subprocess
    pass
```

### D.2 Extend Model Listing Endpoint

**File:** `src/webapp/backend/app/api/calibrate_models.py`

```python
@router.get("/models")
async def list_models():
    """List all trained models (v1.x and v2.0)."""
    # Scan models directory for metadata.json
    # Add "version" field to distinguish v1 vs v2
    # Return: [{"run_id": ..., "version": "v2.0", "training_mode": "finetune", ...}, ...]
    pass
```

### D.3 New Endpoint: Get Edge Predictions

**File:** `src/webapp/backend/app/api/calibrate_models.py`

```python
@router.get("/models/{run_id}/edge")
async def get_edge_predictions(run_id: str):
    """Retrieve edge predictions for a trained model."""
    # Load edge_predictions.csv from run directory
    # Return as JSON: {rows: [{ticker, threshold, expiry, edge, ...}, ...]}
    pass
```

### D.4 Models Schema Update

**File:** `src/webapp/backend/app/models/calibrate_models.py`

```python
class TrainModelV2Request(BaseModel):
    training_mode: str = "finetune"
    feature_sources: str = "both"
    prn_csv: str
    pm_csv: Optional[str] = None
    out_dir: str
    overlap_window: str = "90days"
    numeric_features: Optional[List[str]] = None
    pm_features: Optional[List[str]] = None
    compute_edge: bool = True
    test_weeks: int = 20
    random_state: int = 7
```

---

## E) FRONTEND CHANGES (Minimal Diffs)

### E.1 Add "Model v2.0" Tab to Calibration Page

**File:** `src/webapp/frontend/src/pages/CalibrateModelsPage.tsx`

```tsx
// Add tab navigation
<Tabs value={activeTab} onChange={handleTabChange}>
  <Tab label="Model v1.x (Legacy)" />
  <Tab label="Model v2.0 (PM+Options)" />
</Tabs>

{activeTab === 1 && <ModelV2Panel />}
```

### E.2 New Component: ModelV2Panel

**File:** `src/webapp/frontend/src/components/ModelV2Panel.tsx`

```tsx
export function ModelV2Panel() {
  const [trainingMode, setTrainingMode] = useState('finetune');
  const [featureSources, setFeatureSources] = useState('both');
  const [overlapWindow, setOverlapWindow] = useState('90days');
  const [jobStatus, setJobStatus] = useState<string | null>(null);

  const handleTrain = async () => {
    const payload = {
      training_mode: trainingMode,
      feature_sources: featureSources,
      prn_csv: prnDataset,
      pm_csv: pmDataset,
      out_dir: outputDir,
      overlap_window: overlapWindow,
      compute_edge: true,
    };

    const response = await fetch('/api/calibrate-models/train-v2', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });

    const result = await response.json();
    setJobStatus(result.run_id);
  };

  return (
    <Box>
      <FormControl>
        <InputLabel>Training Mode</InputLabel>
        <Select value={trainingMode} onChange={(e) => setTrainingMode(e.target.value)}>
          <MenuItem value="pretrain">Pretrain (Options-Only)</MenuItem>
          <MenuItem value="finetune">Finetune (PM+Options Meta-Model)</MenuItem>
          <MenuItem value="joint">Joint (Single Model)</MenuItem>
          <MenuItem value="two_stage">Two-Stage (Legacy v1.6)</MenuItem>
        </Select>
      </FormControl>

      <FormControl>
        <InputLabel>Feature Sources</InputLabel>
        <Select value={featureSources} onChange={(e) => setFeatureSources(e.target.value)}>
          <MenuItem value="options">Options Only</MenuItem>
          <MenuItem value="pm">Polymarket Only</MenuItem>
          <MenuItem value="both">Both (Recommended)</MenuItem>
        </Select>
      </FormControl>

      <TextField
        label="PM Overlap Window"
        value={overlapWindow}
        onChange={(e) => setOverlapWindow(e.target.value)}
        helperText="e.g., '90days' or '3months'"
      />

      <Button variant="contained" onClick={handleTrain}>
        Train Model v2.0
      </Button>

      {jobStatus && <JobProgressIndicator runId={jobStatus} />}
    </Box>
  );
}
```

### E.3 Extend Model Detail View with Edge Predictions

**File:** `src/webapp/frontend/src/pages/ModelDetailPage.tsx`

```tsx
// Add tab for edge predictions
{model.version === 'v2.0' && (
  <Tab label="Edge Predictions" />
)}

{activeTab === 'edge' && (
  <EdgePredictionsTable runId={model.run_id} />
)}
```

### E.4 New Component: EdgePredictionsTable

**File:** `src/webapp/frontend/src/components/EdgePredictionsTable.tsx`

```tsx
export function EdgePredictionsTable({ runId }: { runId: string }) {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    fetch(`/api/calibrate-models/models/${runId}/edge`)
      .then(res => res.json())
      .then(json => setData(json.rows));
  }, [runId]);

  return (
    <DataGrid
      rows={data}
      columns={[
        { field: 'ticker', headerName: 'Ticker', width: 100 },
        { field: 'threshold', headerName: 'Strike', width: 100 },
        { field: 'expiry_date', headerName: 'Expiry', width: 120 },
        { field: 'p_base', headerName: 'P(base)', width: 100,
          valueFormatter: (v) => v.toFixed(3) },
        { field: 'p_pm', headerName: 'P(PM)', width: 100,
          valueFormatter: (v) => v.toFixed(3) },
        { field: 'p_final', headerName: 'P(final)', width: 100,
          valueFormatter: (v) => v.toFixed(3) },
        { field: 'edge', headerName: 'Edge', width: 100,
          valueFormatter: (v) => v.toFixed(3),
          cellClassName: (params) => params.value > 0 ? 'positive-edge' : 'negative-edge' },
      ]}
    />
  );
}
```

---

## F) VALIDATION PLAN

### F.1 Leakage Checks (MUST-HAVE)

```python
def test_no_temporal_leakage():
    """Assert options snapshot_ts <= PM snapshot_ts everywhere."""
    overlap = load_overlap_dataset()
    prn_ts = pd.to_datetime(overlap["prn_snapshot_ts"])
    pm_ts = pd.to_datetime(overlap["pm_snapshot_ts"])
    assert (prn_ts <= pm_ts).all(), "Temporal leakage detected"

def test_splits_time_ordered():
    """Assert train dates < test dates."""
    train_df, test_df = load_train_test_split()
    train_max = train_df["snapshot_date"].max()
    test_min = test_df["snapshot_date"].min()
    assert train_max < test_min, f"Split leakage: train_max={train_max}, test_min={test_min}"

def test_no_same_event_leakage():
    """Assert no (ticker, expiry) overlap between train/test."""
    train_events = set(train_df[["ticker", "expiry_date"]].itertuples(index=False, name=None))
    test_events = set(test_df[["ticker", "expiry_date"]].itertuples(index=False, name=None))
    overlap_events = train_events & test_events
    assert len(overlap_events) == 0, f"Same-event leakage: {overlap_events}"

def test_pretrain_no_overlap_contamination():
    """Assert all pretrain dates < overlap_start."""
    pretrain_max = pretrain_df["snapshot_date"].max()
    overlap_min = overlap_df["snapshot_date"].min()
    assert pretrain_max < overlap_min, "Pretrain contaminated with overlap data"
```

### F.2 Metrics Validation

```python
def test_model_improves_over_baseline():
    """Assert v2.0 model logloss < baseline PM logloss on test."""
    metrics = load_metrics_csv()
    test_metrics = metrics[metrics["split"] == "test"]
    pm_logloss = test_metrics[test_metrics["model"] == "baseline_PM"]["logloss"].values[0]
    v2_logloss = test_metrics[test_metrics["model"] == "two_stage"]["logloss"].values[0]
    assert v2_logloss < pm_logloss, f"Model did not improve: PM={pm_logloss}, v2={v2_logloss}"

def test_calibration_quality():
    """Assert ECE < 0.10 on test set."""
    metrics = load_metrics_csv()
    test_ece = metrics[(metrics["split"] == "test") & (metrics["model"] == "two_stage")]["ece"].values[0]
    assert test_ece < 0.10, f"Poor calibration: ECE={test_ece}"

def test_edge_stability():
    """Assert edge std dev < 0.05 across test weeks."""
    edge_df = load_edge_predictions()
    edge_by_week = edge_df.groupby("week_friday")["edge"].mean()
    edge_std = edge_by_week.std()
    assert edge_std < 0.05, f"Unstable edge: std={edge_std}"
```

### F.3 Performance Checks

```python
def test_training_completes_within_timeout():
    """Assert v2.0 training completes in < 30 minutes."""
    import time
    start = time.time()
    run_v2_training()
    duration = time.time() - start
    assert duration < 1800, f"Training too slow: {duration}s"

def test_no_memory_explosion():
    """Assert peak memory usage < 8GB."""
    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024**3)
    run_v2_training()
    mem_after = process.memory_info().rss / (1024**3)
    mem_delta = mem_after - mem_before
    assert mem_delta < 8.0, f"Memory explosion: {mem_delta}GB"
```

### F.4 Integration Tests

```python
def test_pretrain_to_finetune_pipeline():
    """End-to-end test: pretrain → finetune → edge predictions."""
    # 1. Run pretrain
    pretrain_path = run_pretrain(prn_csv, out_dir)
    assert pretrain_path.exists()

    # 2. Run finetune
    finetune_path = run_finetune(pretrain_path, prn_csv, pm_csv, out_dir)
    assert finetune_path.exists()

    # 3. Generate edge predictions
    edge_path = compute_edge(finetune_path, test_df, out_dir)
    assert edge_path.exists()

    # 4. Validate outputs
    edge_df = pd.read_csv(edge_path)
    assert "edge" in edge_df.columns
    assert edge_df["edge"].notna().all()

def test_backward_compatibility():
    """Assert v1.6 two-stage mode still works."""
    result = subprocess.run([
        "python", "03-calibrate-logit-model-v2.0.py",
        "--training-mode", "two_stage",
        "--two-stage-mode",
        "--csv", "prn.csv",
        "--two-stage-pm-csv", "pm.csv",
        "--out-dir", "output",
    ])
    assert result.returncode == 0
    assert (Path("output") / "two_stage_model.joblib").exists()
```

---

## Summary

This plan provides a **production-ready roadmap** for v2.0 with:
- **Hierarchical training** leveraging long options history + short PM overlap
- **Time-safe architecture** with rigorous leakage prevention
- **Edge estimation** for backtesting/signal generation
- **Minimal diffs** preserving backward compatibility
- **Comprehensive validation** (leakage, metrics, performance)

**Next Steps:**
1. Review and approve this plan
2. Implement v2.0 script following ordered checklist (Section B)
3. Add backend endpoints (Section D)
4. Build frontend components (Section E)
5. Execute validation tests (Section F)

**Estimated Timeline:**
- Script implementation: 2-3 days
- Backend integration: 1 day
- Frontend development: 1-2 days
- Testing & validation: 1 day
- **Total: 5-7 days**
