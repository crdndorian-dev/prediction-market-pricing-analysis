#!/usr/bin/env python3
"""
Simple test runner (no pytest required).
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "webapp"))

from shared.cli_contract_v2 import (
    validate_model_kind,
    validate_training_mode,
    validate_mixed_model,
    validate_payload,
    VALID_MODEL_KINDS,
)


class MockPayload:
    """Mock request payload for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_valid_model_kinds():
    """Test that valid model_kind values pass validation."""
    print("Test: Valid model_kind values should pass")
    for kind in VALID_MODEL_KINDS:
        errors = validate_model_kind(kind)
        assert len(errors) == 0, f"FAIL: {kind} should be valid, got errors: {errors}"
    print("  ✅ PASS: All valid model_kind values accepted")


def test_invalid_model_kind():
    """Test that invalid model_kind 'pooled' is rejected."""
    print("Test: Invalid model_kind 'pooled' should be rejected")
    errors = validate_model_kind("pooled")
    assert len(errors) == 1, f"FAIL: Expected 1 error, got {len(errors)}"
    assert "Invalid model_kind" in errors[0], f"FAIL: Wrong error message: {errors[0]}"
    assert "pooled" in errors[0], f"FAIL: Error should mention 'pooled': {errors[0]}"
    print("  ✅ PASS: 'pooled' correctly rejected")


def test_two_stage_requires_pm_csv():
    """Test that two_stage_mode requires two_stage_pm_csv."""
    print("Test: two_stage_mode=True requires two_stage_pm_csv")
    payload = MockPayload(
        csv="src/data/raw/option-chain/dataset.csv",
        model_kind="calibrate",
        two_stage_mode=True,
        two_stage_pm_csv=None,
    )
    errors = validate_payload(payload)
    assert len(errors) == 1, f"FAIL: Expected 1 error, got {len(errors)}: {errors}"
    assert "requires two_stage_pm_csv" in errors[0], f"FAIL: Wrong error: {errors[0]}"
    print("  ✅ PASS: Constraint enforced")


def test_valid_payload():
    """Test that a valid payload passes validation."""
    print("Test: Valid payload should pass")
    payload = MockPayload(
        csv="src/data/raw/option-chain/dataset.csv",
        model_kind="calibrate",
        two_stage_mode=False,
    )
    errors = validate_payload(payload)
    assert len(errors) == 0, f"FAIL: Expected no errors, got: {errors}"
    print("  ✅ PASS: Valid payload accepted")


def test_deprecated_mode_rejected():
    """Test that deprecated mode='pooled' is rejected."""
    print("Test: Deprecated mode='pooled' should be rejected")
    payload = MockPayload(
        csv="src/data/raw/option-chain/dataset.csv",
        model_kind="calibrate",
        mode="pooled",
    )
    errors = validate_payload(payload)
    assert len(errors) >= 1, f"FAIL: Expected error for mode='pooled', got none"
    print("  ✅ PASS: Deprecated mode value rejected")


def main():
    print("=" * 70)
    print("Running Calibrate Models Validation Tests")
    print("=" * 70)
    print()

    tests = [
        test_valid_model_kinds,
        test_invalid_model_kind,
        test_two_stage_requires_pm_csv,
        test_valid_payload,
        test_deprecated_mode_rejected,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
