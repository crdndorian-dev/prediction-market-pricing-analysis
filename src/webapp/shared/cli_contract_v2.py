"""
CLI Contract for 03-calibrate-logit-model-v2.0.py

This file defines the valid argument values and constraints for the v2.0 calibration script.
It serves as the single source of truth for both frontend and backend validation.

IMPORTANT: Keep this file in sync with src/scripts/03-calibrate-logit-model-v2.0.py argparse.
Run `python ci/check_cli_contract_sync.py` to verify sync.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Valid enum values (must match v2.0 script argparse choices)
VALID_MODEL_KINDS = ["calibrate", "mixed", "both"]
VALID_TRAINING_MODES = ["pretrain", "finetune", "joint", "two_stage"]
VALID_FEATURE_SOURCES = ["options", "pm", "both"]
VALID_MIXED_MODELS = ["residual", "blend"]

# Deprecated arguments (v1.5/v1.6 that v2.0 doesn't accept)
DEPRECATED_ARGS = {
    "--mode": "Use --training-mode or --two-stage-mode flag instead",
}

# Argument constraints and dependencies
CONSTRAINTS = {
    "two_stage_mode_requires_pm_csv": "If two_stage_mode is enabled, two_stage_pm_csv must be provided",
    "mixed_model_requires_features": "If model_kind is 'mixed' or 'both', mixed_features path should be provided",
}


class ValidationError(ValueError):
    """Raised when payload validation fails."""
    pass


def validate_model_kind(value: Optional[str]) -> List[str]:
    """Validate model_kind argument."""
    errors = []
    if value is not None and value not in VALID_MODEL_KINDS:
        errors.append(
            f"Invalid model_kind: '{value}'. Must be one of {VALID_MODEL_KINDS}"
        )
    return errors


def validate_training_mode(value: Optional[str]) -> List[str]:
    """Validate training_mode argument."""
    errors = []
    if value is not None and value not in VALID_TRAINING_MODES:
        errors.append(
            f"Invalid training_mode: '{value}'. Must be one of {VALID_TRAINING_MODES}"
        )
    return errors


def validate_mixed_model(value: Optional[str]) -> List[str]:
    """Validate mixed_model argument."""
    errors = []
    if value is not None and value not in VALID_MIXED_MODELS:
        errors.append(
            f"Invalid mixed_model: '{value}'. Must be one of {VALID_MIXED_MODELS}"
        )
    return errors


def validate_inter_arg_constraints(payload: Any) -> List[str]:
    """Validate inter-argument constraints and dependencies."""
    errors = []

    # Check two-stage mode constraint
    if hasattr(payload, 'two_stage_mode') and payload.two_stage_mode:
        if not hasattr(payload, 'two_stage_pm_csv') or not payload.two_stage_pm_csv:
            errors.append(
                "two_stage_mode requires two_stage_pm_csv (Polymarket dataset path)"
            )

    # Check mixed model requirements
    if hasattr(payload, 'model_kind') and payload.model_kind in ['mixed', 'both']:
        if hasattr(payload, 'mixed_features') and not payload.mixed_features:
            # This is just a warning - script has defaults
            pass

    # Reject deprecated 'mode' argument
    if hasattr(payload, 'mode') and payload.mode is not None:
        # Only allow mode if it's being used for two_stage compatibility
        if payload.mode not in ["baseline", "two_stage"]:
            errors.append(
                f"'mode={payload.mode}' is not valid for v2.0. "
                f"Use two_stage_mode flag or training_mode argument instead."
            )

    return errors


def validate_payload(payload: Any) -> List[str]:
    """
    Validate a calibration request payload against v2.0 CLI contract.

    Args:
        payload: Request payload (CalibrateModelRunRequest or similar)

    Returns:
        List of error messages (empty if valid)

    Raises:
        ValidationError: If validation fails with collected errors
    """
    errors = []

    # Validate enum fields
    if hasattr(payload, 'model_kind'):
        errors.extend(validate_model_kind(payload.model_kind))

    if hasattr(payload, 'training_mode'):
        errors.extend(validate_training_mode(payload.training_mode))

    if hasattr(payload, 'mixed_model'):
        errors.extend(validate_mixed_model(payload.mixed_model))

    # Validate inter-argument constraints
    errors.extend(validate_inter_arg_constraints(payload))

    return errors


def get_cli_schema() -> Dict[str, Any]:
    """
    Get the CLI schema for frontend consumption.

    Returns:
        Dictionary with allowed values for each argument type
    """
    return {
        "model_kind": {
            "type": "enum",
            "choices": VALID_MODEL_KINDS,
            "default": "calibrate",
            "description": "Which model(s) to run (calibrate, mixed, or both)",
        },
        "training_mode": {
            "type": "enum",
            "choices": VALID_TRAINING_MODES,
            "default": "two_stage",
            "description": "Training workflow mode (v2.0 feature)",
        },
        "feature_sources": {
            "type": "enum",
            "choices": VALID_FEATURE_SOURCES,
            "default": "both",
            "description": "Which feature sets to use",
        },
        "mixed_model": {
            "type": "enum",
            "choices": VALID_MIXED_MODELS,
            "default": "residual",
            "description": "Mixed model type (residual or blend)",
        },
        "deprecated": DEPRECATED_ARGS,
        "constraints": CONSTRAINTS,
    }
