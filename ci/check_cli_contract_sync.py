#!/usr/bin/env python3
"""
CI check to ensure cli_contract_v2.py stays in sync with v2.0 script argparse.

This script parses the actual v2.0 calibration script's argparse configuration
and compares it against the constants defined in cli_contract_v2.py.

If they don't match, the script fails with a descriptive error message.

Usage:
    python3 ci/check_cli_contract_sync.py

Exit codes:
    0 - CLI contract is in sync
    1 - CLI contract is out of sync or error occurred
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src" / "webapp"))

# Import constants from shared contract
from shared.cli_contract_v2 import (
    VALID_MODEL_KINDS,
    VALID_TRAINING_MODES,
    VALID_MIXED_MODELS,
    VALID_FEATURE_SOURCES,
)


def parse_v2_script_argparse():
    """
    Parse the v2.0 script to extract argparse choices.

    Returns:
        dict: Mapping of argument names to their choices
    """
    script_path = REPO_ROOT / "src" / "scripts" / "03-calibrate-logit-model-v2.0.py"
    if not script_path.exists():
        raise FileNotFoundError(f"v2.0 script not found at {script_path}")

    # Read script and parse argparse choices using regex
    # (Avoid importing the script directly to prevent side effects)
    import re

    content = script_path.read_text()

    # Extract choices for each argument
    choices_map = {}

    # Pattern to match: parser.add_argument("--arg-name", choices=[...], ...)
    pattern = r'parser\.add_argument\(\s*["\']--([^"\']+)["\'].*?choices\s*=\s*\[([^\]]+)\]'

    for match in re.finditer(pattern, content, re.DOTALL):
        arg_name = match.group(1)
        choices_str = match.group(2)

        # Extract individual choice values
        choices = []
        for choice in re.findall(r'["\']([^"\']+)["\']', choices_str):
            choices.append(choice)

        choices_map[arg_name] = choices

    return choices_map


def check_contract_sync():
    """
    Check if cli_contract_v2.py is in sync with v2.0 script.

    Returns:
        tuple: (is_synced: bool, errors: list[str])
    """
    errors = []

    try:
        script_choices = parse_v2_script_argparse()
    except Exception as e:
        errors.append(f"Failed to parse v2.0 script: {e}")
        return False, errors

    # Check model-kind
    if "model-kind" in script_choices:
        actual = set(script_choices["model-kind"])
        expected = set(VALID_MODEL_KINDS)
        if actual != expected:
            errors.append(
                f"VALID_MODEL_KINDS mismatch:\n"
                f"  Script defines: {sorted(actual)}\n"
                f"  Contract defines: {sorted(expected)}"
            )

    # Check training-mode
    if "training-mode" in script_choices:
        actual = set(script_choices["training-mode"])
        expected = set(VALID_TRAINING_MODES)
        if actual != expected:
            errors.append(
                f"VALID_TRAINING_MODES mismatch:\n"
                f"  Script defines: {sorted(actual)}\n"
                f"  Contract defines: {sorted(expected)}"
            )

    # Check mixed-model
    if "mixed-model" in script_choices:
        actual = set(script_choices["mixed-model"])
        expected = set(VALID_MIXED_MODELS)
        if actual != expected:
            errors.append(
                f"VALID_MIXED_MODELS mismatch:\n"
                f"  Script defines: {sorted(actual)}\n"
                f"  Contract defines: {sorted(expected)}"
            )

    # Check feature-sources
    if "feature-sources" in script_choices:
        actual = set(script_choices["feature-sources"])
        expected = set(VALID_FEATURE_SOURCES)
        if actual != expected:
            errors.append(
                f"VALID_FEATURE_SOURCES mismatch:\n"
                f"  Script defines: {sorted(actual)}\n"
                f"  Contract defines: {sorted(expected)}"
            )

    is_synced = len(errors) == 0
    return is_synced, errors


def main():
    """Main entry point for CI check."""
    print("=" * 70)
    print("CLI Contract Sync Check")
    print("=" * 70)
    print()
    print("Checking if src/webapp/shared/cli_contract_v2.py matches")
    print("src/scripts/03-calibrate-logit-model-v2.0.py argparse...")
    print()

    is_synced, errors = check_contract_sync()

    if is_synced:
        print("✅ CLI contract is IN SYNC with v2.0 script")
        print()
        print("All argument choices match:")
        print(f"  - model-kind: {VALID_MODEL_KINDS}")
        print(f"  - training-mode: {VALID_TRAINING_MODES}")
        print(f"  - mixed-model: {VALID_MIXED_MODELS}")
        print(f"  - feature-sources: {VALID_FEATURE_SOURCES}")
        print()
        return 0
    else:
        print("❌ CLI CONTRACT OUT OF SYNC!")
        print()
        print("The following mismatches were found:")
        print()
        for error in errors:
            print(error)
            print()
        print("=" * 70)
        print("ACTION REQUIRED:")
        print("=" * 70)
        print()
        print("1. Review changes in src/scripts/03-calibrate-logit-model-v2.0.py")
        print("2. Update src/webapp/shared/cli_contract_v2.py to match")
        print("3. Re-run this check: python3 ci/check_cli_contract_sync.py")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
