"""
Unit tests for calibrate_models validation against v2.0 CLI contract.

These tests ensure that the backend properly validates requests before
executing the v2.0 calibration script, preventing invalid CLI invocations.
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "webapp"))

import pytest
from shared.cli_contract_v2 import (
    validate_model_kind,
    validate_training_mode,
    validate_mixed_model,
    validate_payload,
    ValidationError,
    VALID_MODEL_KINDS,
)


class MockPayload:
    """Mock request payload for testing."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestModelKindValidation:
    """Test model_kind enum validation."""

    def test_valid_model_kinds_accepted(self):
        """All valid model_kind values should pass validation."""
        for kind in VALID_MODEL_KINDS:
            errors = validate_model_kind(kind)
            assert len(errors) == 0, f"{kind} should be valid"

    def test_invalid_model_kind_rejected(self):
        """Invalid model_kind values should be rejected."""
        errors = validate_model_kind("pooled")
        assert len(errors) == 1
        assert "Invalid model_kind" in errors[0]
        assert "pooled" in errors[0]

    def test_none_model_kind_accepted(self):
        """None model_kind should be accepted (uses script default)."""
        errors = validate_model_kind(None)
        assert len(errors) == 0


class TestPayloadValidation:
    """Test full payload validation with inter-arg constraints."""

    def test_valid_calibrate_only_payload(self):
        """Valid calibrate-only payload should pass."""
        payload = MockPayload(
            csv="src/data/raw/option-chain/dataset.csv",
            model_kind="calibrate",
            two_stage_mode=False,
        )
        errors = validate_payload(payload)
        assert len(errors) == 0

    def test_two_stage_mode_requires_pm_csv(self):
        """two_stage_mode=True without two_stage_pm_csv should fail."""
        payload = MockPayload(
            csv="src/data/raw/option-chain/dataset.csv",
            model_kind="calibrate",
            two_stage_mode=True,
            two_stage_pm_csv=None,
        )
        errors = validate_payload(payload)
        assert len(errors) == 1
        assert "requires two_stage_pm_csv" in errors[0]

    def test_two_stage_mode_with_pm_csv_passes(self):
        """two_stage_mode=True with two_stage_pm_csv should pass."""
        payload = MockPayload(
            csv="src/data/raw/option-chain/dataset.csv",
            model_kind="calibrate",
            two_stage_mode=True,
            two_stage_pm_csv="src/data/models/polymarket/decision_features.parquet",
        )
        errors = validate_payload(payload)
        assert len(errors) == 0

    def test_invalid_model_kind_rejected_in_payload(self):
        """Payload with invalid model_kind should fail."""
        payload = MockPayload(
            csv="src/data/raw/option-chain/dataset.csv",
            model_kind="pooled",  # Invalid for v2.0
        )
        errors = validate_payload(payload)
        assert len(errors) == 1
        assert "Invalid model_kind" in errors[0]
        assert "pooled" in errors[0]

    def test_deprecated_mode_value_rejected(self):
        """Deprecated mode values should be rejected."""
        payload = MockPayload(
            csv="src/data/raw/option-chain/dataset.csv",
            model_kind="calibrate",
            mode="pooled",  # Deprecated mode value
        )
        errors = validate_payload(payload)
        assert len(errors) == 1
        assert "mode=pooled" in errors[0] or "not valid" in errors[0]

    def test_mode_baseline_or_two_stage_allowed(self):
        """mode='baseline' or 'two_stage' should be allowed for compatibility."""
        for mode_val in ["baseline", "two_stage"]:
            payload = MockPayload(
                csv="src/data/raw/option-chain/dataset.csv",
                model_kind="calibrate",
                mode=mode_val,
            )
            errors = validate_payload(payload)
            # Should pass - these are the only allowed mode values
            assert len(errors) == 0, f"mode={mode_val} should be allowed"

    def test_mixed_model_with_both(self):
        """model_kind='mixed' or 'both' should pass."""
        for kind in ["mixed", "both"]:
            payload = MockPayload(
                csv="src/data/raw/option-chain/dataset.csv",
                model_kind=kind,
                mixed_features="src/data/models/polymarket/decision_features.parquet",
            )
            errors = validate_payload(payload)
            assert len(errors) == 0, f"model_kind={kind} should be valid"


class TestMixedModelValidation:
    """Test mixed_model type validation."""

    def test_valid_mixed_models(self):
        """Valid mixed_model types should pass."""
        for model_type in ["residual", "blend"]:
            errors = validate_mixed_model(model_type)
            assert len(errors) == 0, f"{model_type} should be valid"

    def test_invalid_mixed_model_rejected(self):
        """Invalid mixed_model types should be rejected."""
        errors = validate_mixed_model("invalid_type")
        assert len(errors) == 1
        assert "Invalid mixed_model" in errors[0]


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
