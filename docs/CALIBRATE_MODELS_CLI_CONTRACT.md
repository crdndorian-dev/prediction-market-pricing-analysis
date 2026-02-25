# Calibrate Models CLI Contract System

## Overview

This document describes the CLI contract system that ensures the Calibrate Models page stays in sync with the v2.0 calibration script's argparse configuration.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  src/scripts/03-calibrate-logit-model-v2.0.py           │
│  (SOURCE OF TRUTH - argparse defines valid choices)     │
└────────────────────┬─────────────────────────────────────┘
                     │
                     ↓ (must stay in sync)
┌──────────────────────────────────────────────────────────┐
│  src/webapp/shared/cli_contract_v2.py                    │
│  - VALID_MODEL_KINDS = ["calibrate", "mixed", "both"]   │
│  - VALID_TRAINING_MODES = [...]                          │
│  - validate_payload(payload) -> List[error]             │
└────────────────────┬─────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ↓                         ↓
┌──────────────────┐    ┌─────────────────────┐
│  Backend Service │    │  Frontend UI        │
│  - Validates req │    │  - Dropdown options │
│  - Builds CLI    │    │  - Form validation  │
└──────────────────┘    └─────────────────────┘
```

## Key Files

### 1. CLI Contract (Source)
**Location:** `src/webapp/shared/cli_contract_v2.py`

Defines:
- Valid enum values for all choice-based arguments
- Validation functions for payloads
- Inter-argument constraints (e.g., two_stage requires pm_csv)
- Deprecated arguments list

### 2. Backend Service
**Location:** `src/webapp/backend/app/services/calibrate_models.py`

Uses contract to:
- Validate requests before subprocess execution
- Build CLI command with only valid v2.0 arguments
- Reject invalid combinations early

### 3. Backend API
**Location:** `src/webapp/backend/app/api/calibrate_models.py`

Provides:
- `POST /api/calibrate-models/run` — Execute calibration
- `GET /api/calibrate-models/cli-schema` — Get allowed values (NEW)

### 4. Frontend UI
**Location:** `src/webapp/frontend/src/pages/CalibrateModelsPage.tsx`

Uses contract (implicitly via backend):
- Form dropdowns match backend validation
- Clearer labels (no confusing "pooled" references)

### 5. Unit Tests
**Location:** `tests/backend/services/test_calibrate_models_validation.py`

Tests:
- Valid enum values accepted
- Invalid values rejected (e.g., "pooled")
- Inter-arg constraints enforced
- Deprecated values rejected

**Run:** `python3 tests/backend/services/run_tests_simple.py`

### 6. CI Sync Check
**Location:** `ci/check_cli_contract_sync.py`

Verifies:
- Contract constants match v2.0 script argparse
- Fails CI if out of sync

**Run:** `python3 ci/check_cli_contract_sync.py`

## Usage Guide

### For Developers: Making Changes

#### Scenario A: Updating v2.0 Script Arguments

If you modify `src/scripts/03-calibrate-logit-model-v2.0.py` argparse:

1. **Update the script:**
   ```python
   # Example: Adding a new choice to --model-kind
   parser.add_argument(
       "--model-kind",
       choices=["calibrate", "mixed", "both", "ensemble"],  # Added 'ensemble'
       ...
   )
   ```

2. **Update the contract:**
   ```python
   # src/webapp/shared/cli_contract_v2.py
   VALID_MODEL_KINDS = ["calibrate", "mixed", "both", "ensemble"]
   ```

3. **Run CI check:**
   ```bash
   python3 ci/check_cli_contract_sync.py
   ```
   Expected: `✅ CLI contract is IN SYNC`

4. **Update frontend (if needed):**
   - If users should see the new option, add it to UI dropdown
   - If it's internal-only, no frontend change needed

5. **Update tests:**
   ```python
   def test_ensemble_model_kind():
       errors = validate_model_kind("ensemble")
       assert len(errors) == 0
   ```

6. **Run tests:**
   ```bash
   python3 tests/backend/services/run_tests_simple.py
   ```

#### Scenario B: Adding New Constraints

If you need to enforce a new constraint:

1. **Add constraint logic to contract:**
   ```python
   # src/webapp/shared/cli_contract_v2.py
   def validate_inter_arg_constraints(payload):
       errors = []

       # NEW: ensemble requires ensemble_config
       if hasattr(payload, 'model_kind') and payload.model_kind == 'ensemble':
           if not hasattr(payload, 'ensemble_config') or not payload.ensemble_config:
               errors.append("model_kind='ensemble' requires ensemble_config path")

       return errors
   ```

2. **Add test:**
   ```python
   def test_ensemble_requires_config():
       payload = MockPayload(model_kind="ensemble", ensemble_config=None)
       errors = validate_payload(payload)
       assert "requires ensemble_config" in errors[0]
   ```

3. **Backend automatically enforces it** (via `validate_payload()` call)

### For Frontend Developers: Using CLI Schema

The backend exposes `/api/calibrate-models/cli-schema` to get valid values:

```typescript
// Fetch schema from backend
const response = await fetch('/api/calibrate-models/cli-schema');
const schema = await response.json();

// Use for dropdown options
<select>
  {schema.model_kind.choices.map(choice => (
    <option value={choice}>{choice}</option>
  ))}
</select>
```

### For QA: Testing

#### Manual Testing

1. **Start webapp:**
   ```bash
   ./run-webapp.sh
   ```

2. **Navigate to:** `http://localhost:5173/calibrate-models`

3. **Try to submit with invalid values:**
   - Backend should reject with clear error message
   - No more "invalid choice: 'pooled'" errors from script

4. **Check command after run:**
   - Should NOT contain `--mode` argument
   - Should only contain v2.0-compatible arguments

#### Automated Testing

```bash
# Run validation tests
python3 tests/backend/services/run_tests_simple.py

# Run CI contract sync check
python3 ci/check_cli_contract_sync.py
```

## Troubleshooting

### Error: "Invalid model_kind: 'pooled'"

**Cause:** Frontend or backend is sending deprecated "pooled" value

**Fix:**
1. Check frontend form state doesn't set `model_kind="pooled"`
2. Backend validation should catch this before subprocess
3. If error comes from script, validation was bypassed

### Error: "CLI contract out of sync"

**Cause:** v2.0 script argparse changed but contract wasn't updated

**Fix:**
1. Run: `python3 ci/check_cli_contract_sync.py` (shows mismatches)
2. Update `src/webapp/shared/cli_contract_v2.py` to match script
3. Re-run CI check to verify

### Error: "two_stage_mode requires two_stage_pm_csv"

**Cause:** User enabled two-stage mode without providing Polymarket dataset

**Fix:**
1. This is correct behavior (enforcing constraint)
2. Frontend should show Polymarket dataset picker when `twoStageMode=true`
3. User must select a PM dataset to proceed

## Maintenance Checklist

### When v2.0 Script Changes:

- [ ] Update `src/scripts/03-calibrate-logit-model-v2.0.py`
- [ ] Update `src/webapp/shared/cli_contract_v2.py` to match
- [ ] Run `python3 ci/check_cli_contract_sync.py` (must pass)
- [ ] Update frontend UI if new options should be exposed
- [ ] Add tests for new constraints
- [ ] Run `python3 tests/backend/services/run_tests_simple.py`
- [ ] Update docs if behavior changed

### Before Merging PR:

- [ ] CI contract check passes
- [ ] Unit tests pass
- [ ] Manual smoke test (run calibration from UI)
- [ ] No `--mode` in generated CLI command

## Reference: v2.0 Valid Arguments

### Enum Arguments (Choice-Based)

| Argument | Valid Choices | Default |
|----------|---------------|---------|
| `--model-kind` | `calibrate`, `mixed`, `both` | `calibrate` |
| `--training-mode` | `pretrain`, `finetune`, `joint`, `two_stage` | `two_stage` |
| `--feature-sources` | `options`, `pm`, `both` | `both` |
| `--mixed-model` | `residual`, `blend` | `residual` |

### Flag Arguments (Boolean)

- `--compute-edge`
- `--two-stage-mode`
- `--mixed-walk-forward`

### Deprecated Arguments (Don't Use)

- `--mode` — Replaced by `--training-mode` or `--two-stage-mode` flag

## FAQ

**Q: Why did "pooled" stop working?**
A: It was never valid for v2.0. The backend was incorrectly coercing it to "calibrate" and passing a deprecated `--mode` arg.

**Q: Can I still use mode="baseline" or mode="two_stage"?**
A: Yes, but these are for backward compatibility. v2.0 doesn't use `--mode` directly; the backend maps them internally.

**Q: How do I add a new argument to v2.0?**
A: Follow "Scenario A" in the Usage Guide above.

**Q: What if I need to support v1.5 and v2.0 simultaneously?**
A: Backend would need to detect which script version and use appropriate contract. Not currently supported.

**Q: Can the frontend get schema dynamically?**
A: Yes! Call `GET /api/calibrate-models/cli-schema` to get valid values.

---

**Last updated:** 2026-02-13
**Contract version:** v2.0
**Status:** ✅ Production Ready
