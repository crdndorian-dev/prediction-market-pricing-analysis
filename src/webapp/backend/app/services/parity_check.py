from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List


def _type_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _numeric_compatible(expected: str, actual: str) -> bool:
    if expected not in {"int", "float"}:
        return False
    return actual in {"int", "float"}


def _compare_schema(expected: Any, actual: Any, path: str) -> List[str]:
    warnings: List[str] = []

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            warnings.append(f"Type mismatch at {path}: expected dict, got {_type_label(actual)}")
            return warnings
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        for key in missing:
            warnings.append(f"Missing key at {path}.{key}")
        for key in extra:
            warnings.append(f"Unexpected key at {path}.{key}")
        for key in sorted(expected_keys & actual_keys):
            warnings.extend(_compare_schema(expected[key], actual[key], f"{path}.{key}"))
        return warnings

    if isinstance(expected, list):
        if not isinstance(actual, list):
            warnings.append(f"Type mismatch at {path}: expected list, got {_type_label(actual)}")
            return warnings
        if not expected:
            return warnings
        elem_schema = expected[0]
        if isinstance(elem_schema, str) and elem_schema in {"unknown", "null", "list"}:
            return warnings
        if actual:
            warnings.extend(_compare_schema(elem_schema, actual[0], f"{path}[0]"))
        return warnings

    if isinstance(expected, str):
        if expected in {"unknown", "null"}:
            return warnings
        actual_type = _type_label(actual)
        if expected == actual_type:
            return warnings
        if _numeric_compatible(expected, actual_type):
            return warnings
        warnings.append(f"Type mismatch at {path}: expected {expected}, got {actual_type}")
        return warnings

    # Fallback: compare type names
    expected_type = _type_label(expected)
    actual_type = _type_label(actual)
    if expected_type != actual_type:
        warnings.append(f"Type mismatch at {path}: expected {expected_type}, got {actual_type}")
    return warnings


def check_artifact_parity(trial_dir: Path, template_path: Path) -> List[str]:
    """
    Compare trial_dir/metadata.json against a schema template.

    Returns a list of warnings (missing keys, extra keys, type mismatches).
    """
    meta_path = trial_dir / "metadata.json"
    if not meta_path.exists():
        return [f"Missing metadata.json at {meta_path}"]
    if not template_path.exists():
        return [f"Missing schema template at {template_path}"]

    try:
        actual = json.loads(meta_path.read_text())
    except Exception as exc:
        return [f"Failed to read metadata.json: {exc}"]

    try:
        expected = json.loads(template_path.read_text())
    except Exception as exc:
        return [f"Failed to read schema template: {exc}"]

    return _compare_schema(expected, actual, path="metadata")
