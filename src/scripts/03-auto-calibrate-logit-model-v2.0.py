#!/usr/bin/env python3
"""
03-auto-calibrate-logit-model-v2.0.py

Auto-calibration orchestrator that evaluates a curated grid of feature sets and
hyperparameters under fixed user-provided splits. Selection defaults to
validation logloss, with optional outer backtest folds to improve generalization.
Test metrics are computed once for the final model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import statistics
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATOR_SCRIPT = REPO_ROOT / "src" / "scripts" / "03-calibrate-logit-model-v2.0.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration.calibrate_v2_core import (
    CalibrationCache,
    _collect_unsupported_compat_args,
    build_args_from_config,
    build_calibration_cache,
    run_calibration_from_cache,
)

BASE_FEATURE = "x_logit_prn"
RISKY_FEATURES = {"prn_raw_gap", "had_fallback", "had_intrinsic_drop", "had_band_clip"}

DEFAULT_FEATURE_SETS = [
    [BASE_FEATURE],
    [BASE_FEATURE, "rv20"],
    [BASE_FEATURE, "abs_log_m_fwd"],
    [BASE_FEATURE, "rv20", "abs_log_m_fwd"],
    [BASE_FEATURE, "rv20", "abs_log_m_fwd", "log_rel_spread"],
]
DEFAULT_C_VALUES = [0.003, 0.01, 0.03, 0.1, 0.3]
DEFAULT_CAL_METHODS = ["none", "platt"]
DEFAULT_UPWEIGHTS = [1.0, 1.25, 1.5]
DEFAULT_FOUNDATION_WEIGHTS = [1.0, 1.25, 1.5]
DEFAULT_TICKER_INTERCEPTS = ["off", "on"]
DEFAULT_OUTER_TEST_WEEKS = 8
DEFAULT_OUTER_GAP_WEEKS = 1
DEFAULT_OUTER_MIN_IMPROVE_FRACTION = 0.75
DEFAULT_OUTER_MAX_WORST_DELTA = 0.005
DEFAULT_OUTER_SELECTION_METRIC = "median_delta_logloss"
DEFAULT_ACCEPT_DELTA_THRESHOLD = 0.0
DEFAULT_MIN_IMPROVE_FRACTION = 0.5
DEFAULT_MAX_WORST_DELTA = 0.01


def _build_env() -> Dict[str, str]:
    env = os.environ.copy()
    root = str(REPO_ROOT)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    base = f"{root}{os.pathsep}{src}"
    env["PYTHONPATH"] = f"{base}{os.pathsep}{existing}" if existing else base
    return env


def _limit_blas_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _load_config(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("config-json must be a JSON object.")
    return payload


def _read_dataset_columns(path: Path) -> List[str]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore

            return list(pq.ParquetFile(path).schema.names)
        except Exception:
            df = pd.read_parquet(path, columns=[])
            return list(df.columns)
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def _load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _compute_week_friday(dates: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dates, errors="coerce")
    if dt.isna().all():
        return pd.Series([pd.NaT] * len(dates))
    weekday = dt.dt.weekday
    offset = (4 - weekday) % 7
    return (dt + pd.to_timedelta(offset, unit="D")).dt.normalize()


def _resolve_outer_week_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if "week_friday" in df.columns:
        parsed = pd.to_datetime(df["week_friday"], errors="coerce")
        if parsed.notna().any():
            return parsed.dt.normalize()
    for cand in ["asof_date", "snapshot_time_utc", "asof_datetime_utc", "snapshot_date"]:
        if cand in df.columns:
            return _compute_week_friday(df[cand])
    return None


def _has_bootstrap_day_column(available_cols: set[str]) -> bool:
    for col in [
        "snapshot_date",
        "asof_date",
        "snapshot_time_utc",
        "asof_datetime_utc",
        "asof_datetime",
        "snapshot_datetime_utc",
        "week_friday",
    ]:
        if col in available_cols:
            return True
    return False


def _validate_bootstrap_precheck(base_args: argparse.Namespace, available_cols: set[str]) -> Optional[str]:
    if not bool(getattr(base_args, "bootstrap_ci", False)):
        return None
    if bool(getattr(base_args, "allow_iid_bootstrap", False)):
        return None

    requested_group = str(getattr(base_args, "bootstrap_group", "auto") or "auto").strip().lower()
    if requested_group in {"auto", "iid"}:
        return None

    if requested_group in {"contract_id", "group_id"}:
        if requested_group not in available_cols:
            return (
                f"Requested bootstrap_group={requested_group!r} is unavailable "
                f"(column {requested_group!r} missing)."
            )
        return None

    has_day = _has_bootstrap_day_column(available_cols)
    ticker_col = str(getattr(base_args, "ticker_col", "ticker") or "ticker")
    has_ticker = ticker_col in available_cols

    if requested_group == "day":
        if has_day:
            return None
        return (
            "Requested bootstrap_group='day' is unavailable "
            "(no usable date-like column found)."
        )

    if requested_group == "ticker_day":
        missing: List[str] = []
        if not has_ticker:
            missing.append(f"ticker column {ticker_col!r}")
        if not has_day:
            missing.append("date-like column")
        if not missing:
            return None
        return (
            "Requested bootstrap_group='ticker_day' is unavailable "
            f"(missing {', '.join(missing)})."
        )

    return None


def _coerce_positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _coerce_positive_bin_count(value: Any, *, default: int, arg_name: str) -> int:
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{arg_name} must be a positive integer (got {value!r}).") from exc
    if parsed <= 0:
        raise ValueError(f"{arg_name} must be > 0 (got {parsed}).")
    return parsed


def _coerce_positive_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except Exception:
        return default
    return parsed if parsed > 0 else default


def _coerce_nonnegative_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except Exception:
        return default
    if not np.isfinite(parsed) or parsed < 0:
        return default
    return parsed


def _coerce_fraction(value: Any, *, default: float) -> float:
    parsed = _coerce_nonnegative_float(value, default=default)
    if parsed > 1.0:
        return default
    return parsed


def _format_week_range(weeks: List[pd.Timestamp]) -> Optional[List[str]]:
    if not weeks:
        return None
    start = min(weeks)
    end = max(weeks)
    return [start.date().isoformat(), end.date().isoformat()]


def _prepare_outer_folds(
    *,
    dataset_path: Path,
    out_dir: Path,
    base_test_weeks: int,
    outer_folds: int,
    outer_test_weeks: int,
    outer_gap_weeks: int,
    df: Optional[pd.DataFrame] = None,
    materialize_csvs: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary: Dict[str, Any] = {
        "enabled": False,
        "outer_folds_requested": int(outer_folds),
        "outer_folds_used": 0,
        "outer_test_weeks": int(outer_test_weeks),
        "outer_gap_weeks": int(outer_gap_weeks),
        "base_test_weeks": int(base_test_weeks),
        "warnings": [],
        "folds": [],
    }
    if outer_folds <= 0 or outer_test_weeks <= 0:
        return [], summary

    if df is None:
        df = _load_dataset(dataset_path)
    df = df.copy()
    week_series = _resolve_outer_week_series(df)
    if week_series is None or week_series.isna().all():
        summary["warnings"].append("No week_friday/asof_date column found; outer folds disabled.")
        return [], summary

    df["_outer_week"] = pd.to_datetime(week_series, errors="coerce").dt.normalize()
    df = df[df["_outer_week"].notna()].copy()
    all_weeks = sorted(df["_outer_week"].dropna().unique())
    summary["weeks_total"] = int(len(all_weeks))
    if not all_weeks:
        summary["warnings"].append("No valid week values after parsing; outer folds disabled.")
        return [], summary

    if base_test_weeks > 0 and len(all_weeks) > base_test_weeks:
        train_weeks = all_weeks[: -base_test_weeks]
    else:
        train_weeks = all_weeks
        if base_test_weeks > 0:
            summary["warnings"].append("Not enough weeks to exclude base test window; using all weeks.")

    summary["weeks_train_pool"] = int(len(train_weeks))
    step = outer_test_weeks + outer_gap_weeks
    if len(train_weeks) < outer_test_weeks + 1:
        summary["warnings"].append("Insufficient weeks for requested outer folds; outer folds disabled.")
        return [], summary

    fold_specs: List[Dict[str, Any]] = []
    fold_records: List[Dict[str, Any]] = []
    for fold_idx in range(int(outer_folds)):
        test_end = len(train_weeks) - fold_idx * step
        test_start = test_end - outer_test_weeks
        gap_start = test_start - outer_gap_weeks
        train_end = gap_start
        if test_start < 0 or train_end <= 0:
            break
        test_weeks = train_weeks[test_start:test_end]
        gap_weeks = train_weeks[gap_start:test_start] if outer_gap_weeks > 0 else []
        train_weeks_fold = train_weeks[:train_end]
        if not test_weeks or not train_weeks_fold:
            break

        allowed_weeks = set(train_weeks_fold) | set(test_weeks)
        fold_df = df[df["_outer_week"].isin(allowed_weeks)].drop(columns=["_outer_week"])
        allowed_idx = fold_df.index.tolist()
        fold_csv: Optional[Path] = None
        if materialize_csvs:
            fold_dir = out_dir / "outer_folds"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_csv = fold_dir / f"fold_{fold_idx + 1:02d}.csv"
            fold_df.to_csv(fold_csv, index=False)

        fold_spec: Dict[str, Any] = {"fold": int(fold_idx + 1), "allowed_idx": allowed_idx}
        if fold_csv is not None:
            fold_spec["csv_path"] = fold_csv
        fold_specs.append(fold_spec)
        fold_records.append(
            {
                "fold": int(fold_idx + 1),
                "csv_path": str(fold_csv) if fold_csv is not None else None,
                "train_weeks_range": _format_week_range(train_weeks_fold),
                "test_weeks_range": _format_week_range(test_weeks),
                "gap_weeks_range": _format_week_range(gap_weeks),
                "n_train_weeks": int(len(train_weeks_fold)),
                "n_test_weeks": int(len(test_weeks)),
                "n_gap_weeks": int(len(gap_weeks)),
                "n_rows": int(len(fold_df)),
            }
        )

    summary["folds"] = fold_records
    summary["outer_folds_used"] = int(len(fold_specs))
    summary["enabled"] = len(fold_specs) > 0

    _write_json(out_dir / "outer_folds.json", {"folds": fold_records, **summary})
    return fold_specs, summary


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _sanitize_feature_sets(
    feature_sets: List[List[str]],
    *,
    available: set[str],
    allow_risky: bool,
) -> List[List[str]]:
    cleaned: List[List[str]] = []
    for raw in feature_sets:
        if not raw:
            continue
        features = _dedupe_preserve_order([f for f in raw if f in available or f == BASE_FEATURE])
        if BASE_FEATURE not in features:
            features = [BASE_FEATURE] + features
        if not allow_risky and any(f in RISKY_FEATURES for f in features):
            continue
        if len(features) == 0:
            continue
        cleaned.append(features)
    # de-dupe sets
    seen = set()
    unique: List[List[str]] = []
    for feat_list in cleaned:
        key = tuple(feat_list)
        if key in seen:
            continue
        seen.add(key)
        unique.append(feat_list)
    return unique


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _write_run_manifest(
    run_dir: Path,
    *,
    auto_status: str,
    selected_trial_id: Optional[int],
    selected_model_relpath: Optional[str],
    auto_search_relpath: str,
) -> None:
    payload = {
        "schema_version": 1,
        "run_type": "auto",
        "auto_status": str(auto_status),
        "selected_trial_id": int(selected_trial_id) if selected_trial_id is not None else None,
        "selected_model_relpath": selected_model_relpath,
        "auto_search_relpath": auto_search_relpath,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(run_dir / "run_manifest.json", payload)


def _parity_enabled() -> bool:
    token = str(os.environ.get("AUTO_CALIBRATE_PARITY", "")).strip().lower()
    return token in {"1", "true", "yes", "y", "on"}


def _compare_json_with_tol(
    left: Any,
    right: Any,
    *,
    tol: float,
    ignore_keys: Optional[set[str]] = None,
    path: str = "",
) -> List[str]:
    diffs: List[str] = []
    ignore_keys = ignore_keys or set()
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left.keys()) | set(right.keys())
        for key in sorted(keys):
            if key in ignore_keys:
                continue
            new_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                diffs.append(f"Missing key {new_path} on left")
                continue
            if key not in right:
                diffs.append(f"Missing key {new_path} on right")
                continue
            diffs.extend(
                _compare_json_with_tol(
                    left[key], right[key], tol=tol, ignore_keys=ignore_keys, path=new_path
                )
            )
        return diffs
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            diffs.append(f"Length mismatch at {path}: {len(left)} != {len(right)}")
            return diffs
        for idx, (a, b) in enumerate(zip(left, right)):
            diffs.extend(
                _compare_json_with_tol(
                    a, b, tol=tol, ignore_keys=ignore_keys, path=f"{path}[{idx}]"
                )
            )
        return diffs
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        if (left is None) != (right is None):
            diffs.append(f"Value mismatch at {path}: {left!r} != {right!r}")
            return diffs
        if left is None or right is None:
            return diffs
        if not np.isfinite(left) and not np.isfinite(right):
            return diffs
        if abs(float(left) - float(right)) > tol:
            diffs.append(f"Value mismatch at {path}: {left!r} != {right!r}")
        return diffs
    if left != right:
        diffs.append(f"Value mismatch at {path}: {left!r} != {right!r}")
    return diffs


def _compare_metrics_csv(left_path: Path, right_path: Path, *, tol: float) -> List[str]:
    diffs: List[str] = []
    if not left_path.exists() or not right_path.exists():
        diffs.append("Missing metrics.csv for parity comparison.")
        return diffs
    left_df = pd.read_csv(left_path)
    right_df = pd.read_csv(right_path)
    if left_df.empty or right_df.empty:
        diffs.append("Empty metrics.csv in parity comparison.")
        return diffs
    left_df = left_df.sort_values(by=[c for c in ["split", "model"] if c in left_df.columns]).reset_index(drop=True)
    right_df = right_df.sort_values(by=[c for c in ["split", "model"] if c in right_df.columns]).reset_index(drop=True)
    if list(left_df.columns) != list(right_df.columns):
        diffs.append("Column mismatch in metrics.csv.")
        return diffs
    if len(left_df) != len(right_df):
        diffs.append("Row count mismatch in metrics.csv.")
        return diffs
    for col in left_df.columns:
        left_vals = left_df[col].to_numpy()
        right_vals = right_df[col].to_numpy()
        if np.issubdtype(left_df[col].dtype, np.number):
            if not np.allclose(left_vals, right_vals, atol=tol, rtol=0, equal_nan=True):
                diffs.append(f"Numeric mismatch in metrics.csv column {col}")
        else:
            if not (left_vals == right_vals).all():
                diffs.append(f"Value mismatch in metrics.csv column {col}")
    return diffs


def _run_parity_check(
    *,
    trial: Dict[str, Any],
    base_config: Dict[str, Any],
    dataset_path: Path,
    out_dir: Path,
    base_cache: CalibrationCache,
    base_unsupported_controls: Dict[str, Any],
    calibrator_path: Path,
    env: Dict[str, str],
) -> None:
    parity_root = out_dir / "parity_check" / uuid.uuid4().hex
    in_dir = parity_root / "inprocess"
    sub_dir = parity_root / "subprocess"
    in_dir.mkdir(parents=True, exist_ok=True)
    sub_dir.mkdir(parents=True, exist_ok=True)

    in_result = _run_trial_inprocess(
        1,
        trial=trial,
        base_config=base_config,
        dataset_path=dataset_path,
        trials_dir=in_dir,
        base_cache=base_cache,
        base_unsupported_controls=base_unsupported_controls,
        outer_fold_caches=None,
    )
    sub_result = _run_trial(
        1,
        trial=trial,
        base_config=base_config,
        dataset_path=dataset_path,
        calibrator_script=calibrator_path,
        trials_dir=sub_dir,
        env=env,
        outer_folds=None,
        outer_test_weeks=None,
        outer_selection_metric=None,
    )

    in_trial_dir = in_dir / "trial_001"
    sub_trial_dir = sub_dir / "trial_001"

    diffs: List[str] = []
    diffs.extend(
        _compare_metrics_csv(
            in_trial_dir / "metrics.csv", sub_trial_dir / "metrics.csv", tol=1e-8
        )
    )
    if (in_trial_dir / "trial_result.json").exists() and (sub_trial_dir / "trial_result.json").exists():
        left = json.loads((in_trial_dir / "trial_result.json").read_text())
        right = json.loads((sub_trial_dir / "trial_result.json").read_text())
        diffs.extend(
            _compare_json_with_tol(
                left,
                right,
                tol=1e-8,
                ignore_keys={"runtime_seconds", "stdout_tail", "stderr_tail"},
            )
        )
    else:
        diffs.append("Missing trial_result.json for parity comparison.")

    best_config = _build_trial_config(
        base_config,
        features=trial["features"],
        c_value=trial["C"],
        calibration=trial["calibration"],
        foundation_weight=trial["foundation_weight"],
        trading_universe_upweight=trial["trading_universe_upweight"],
        ticker_intercepts=trial["ticker_intercepts"],
        ticker_interactions=trial["ticker_interactions"],
        out_dir=parity_root / "best",
        skip_test_metrics=False,
        trial_mode=False,
    )
    _write_json(in_dir / "best_config.json", best_config)
    _write_json(sub_dir / "best_config.json", best_config)
    diffs.extend(
        _compare_json_with_tol(
            json.loads((in_dir / "best_config.json").read_text()),
            json.loads((sub_dir / "best_config.json").read_text()),
            tol=0.0,
        )
    )

    if diffs:
        raise SystemExit("Parity check failed: " + "; ".join(diffs))
_LAST_PROGRESS_HASH: Optional[str] = None
_LAST_PROGRESS_WRITE: float = 0.0


def write_progress(
    out_dir: Path,
    *,
    stage: str,
    trials_total: int,
    trials_done: int,
    trials_failed: int,
    best_score: Optional[float],
    last_error: Optional[str],
    phase: Optional[str] = None,
    candidate_index: Optional[int] = None,
    candidate_total: Optional[int] = None,
    fold_index: Optional[int] = None,
    fold_total: Optional[int] = None,
    message: Optional[str] = None,
    last_log_lines: Optional[List[str]] = None,
    top_candidates: Optional[List[Dict[str, Any]]] = None,
    force: bool = False,
    min_interval_seconds: float = 1.0,
) -> None:
    global _LAST_PROGRESS_HASH, _LAST_PROGRESS_WRITE
    payload = {
        "stage": stage,
        "phase": phase,
        "trials_total": trials_total,
        "trials_done": trials_done,
        "trials_failed": trials_failed,
        "best_score_so_far": best_score,
        "last_error": last_error,
        "candidate_index": candidate_index,
        "candidate_total": candidate_total,
        "fold_index": fold_index,
        "fold_total": fold_total,
        "message": message,
        "last_log_lines": last_log_lines,
        "top_candidates": top_candidates,
    }
    payload_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    now = time.monotonic()
    if not force and payload_hash == _LAST_PROGRESS_HASH and (now - _LAST_PROGRESS_WRITE) < min_interval_seconds:
        return
    _LAST_PROGRESS_HASH = payload_hash
    _LAST_PROGRESS_WRITE = now
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f".progress.{os.getpid()}.{int(time.time() * 1000)}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, out_dir / "progress.json")
    (out_dir / "auto_search_progress.json").write_text(json.dumps(payload, indent=2))


def _parse_metrics(metrics_path: Path) -> Optional[Dict[str, float]]:
    return _parse_metrics_for_split(metrics_path, split="val")


def _parse_metrics_for_split(metrics_path: Path, *, split: str) -> Optional[Dict[str, float]]:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None
    model_row = df[(df["split"] == split) & (df["model"] == "logit")]
    base_row = df[(df["split"] == split) & (df["model"] == "baseline_pRN")]
    if model_row.empty or base_row.empty:
        return None
    try:
        model_logloss = float(model_row["logloss"].iloc[0])
        baseline_logloss = float(base_row["logloss"].iloc[0])
    except Exception:
        return None
    return {
        "model_logloss": model_logloss,
        "baseline_logloss": baseline_logloss,
        "delta_logloss": model_logloss - baseline_logloss,
    }


def _parse_fold_deltas(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "delta_logloss" not in df.columns:
        return None
    values = pd.to_numeric(df["delta_logloss"], errors="coerce").dropna()
    if values.empty:
        return None
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    se = float(std / math.sqrt(len(values))) if len(values) > 0 else None
    improved = int((values < 0).sum())
    worst = float(values.max())
    return {
        "n_folds": int(len(values)),
        "mean_delta_logloss": mean,
        "std_delta_logloss": std,
        "se_delta_logloss": se,
        "folds_improved": improved,
        "worst_delta_logloss": worst,
    }


def _tail_lines(text: str, n: int = 12) -> List[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-n:] if lines else []


def _build_trial_config(
    base_config: Dict[str, Any],
    *,
    features: List[str],
    c_value: float,
    calibration: str,
    foundation_weight: float,
    trading_universe_upweight: float,
    ticker_intercepts: str,
    ticker_interactions: bool,
    out_dir: Path,
    test_weeks_override: Optional[int] = None,
    skip_test_metrics: bool = True,
    trial_mode: bool = True,
) -> Dict[str, Any]:
    config = dict(base_config)
    config["out_dir"] = str(out_dir)
    config["features"] = ",".join(features)
    config["c_grid"] = str(c_value)
    config["calibrate"] = calibration
    config["selection_objective"] = "logloss"
    config["skip_test_metrics"] = bool(skip_test_metrics)

    regularization = dict(config.get("regularization") or {})
    regularization["c_grid"] = str(c_value)
    regularization["calibration_method"] = calibration
    regularization["selection_objective"] = "logloss"
    config["regularization"] = regularization

    model_structure = dict(config.get("model_structure") or {})
    model_structure["foundation_weight"] = foundation_weight
    if ticker_intercepts == "on":
        model_structure["ticker_intercepts"] = model_structure.get("ticker_intercepts") or "non_foundation"
    else:
        model_structure["ticker_intercepts"] = "none"
    model_structure["ticker_x_interactions"] = bool(ticker_interactions)
    config["model_structure"] = model_structure

    weighting = dict(config.get("weighting") or {})
    weighting["trading_universe_upweight"] = trading_universe_upweight
    config["weighting"] = weighting

    if test_weeks_override is not None:
        config["test_weeks"] = int(test_weeks_override)
        split_cfg = dict(config.get("split") or {})
        split_cfg["test_window_weeks"] = int(test_weeks_override)
        config["split"] = split_cfg

    if trial_mode:
        bootstrap = dict(config.get("bootstrap") or {})
        bootstrap["bootstrap_ci"] = False
        config["bootstrap"] = bootstrap

        diagnostics = dict(config.get("diagnostics") or {})
        diagnostics["split_timeline"] = False
        diagnostics["per_fold_delta_chart"] = True
        diagnostics["per_group_delta_distribution"] = False
        config["diagnostics"] = diagnostics

    return config


def _normalize_calibration_args(args: argparse.Namespace) -> None:
    if args.weight_col_strategy:
        strategy = str(args.weight_col_strategy).strip().lower()
        if strategy in {"uniform", "weight_final", "sample_weight_final"}:
            args.weight_col = strategy
        elif strategy == "auto" and not args.weight_col:
            args.weight_col = "weight_final"
    if args.val_windows is not None and int(args.val_windows) > 0:
        args.validation_folds = int(args.val_windows)
    if args.val_window_weeks is not None and int(args.val_window_weeks) > 0:
        args.validation_window_weeks = int(args.val_window_weeks)
    if not args.selection_objective:
        args.selection_objective = "logloss"
    if args.group_equalization and args.no_group_equalization:
        raise ValueError("--group-equalization and --no-group-equalization are mutually exclusive.")
    if int(args.ci_level) not in {90, 95, 99}:
        raise ValueError("--ci-level must be one of 90, 95, 99.")


def _run_calibrator(
    *,
    trial_config: Dict[str, Any],
    dataset_path: Path,
    calibrator_script: Path,
    out_dir: Path,
    env: Dict[str, str],
    skip_test_metrics: bool,
    fast_trial: bool,
) -> Tuple[int, float, List[str], List[str]]:
    trial_config_path = out_dir / "trial_config.json"
    _write_json(trial_config_path, trial_config)

    cmd = [
        sys.executable,
        str(calibrator_script),
        "--csv",
        str(dataset_path),
        "--out-dir",
        str(out_dir),
        "--model-kind",
        "calibrate",
        "--config-json",
        str(trial_config_path),
    ]
    if skip_test_metrics:
        cmd.append("--skip-test-metrics")
    if fast_trial:
        cmd.append("--fast-trial")

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=env,
    )
    runtime = round(time.monotonic() - start, 3)

    stdout_tail = _tail_lines(result.stdout or "")
    stderr_tail = _tail_lines(result.stderr or "")
    return result.returncode, runtime, stdout_tail, stderr_tail


def _run_trial(
    idx: int,
    *,
    trial: Dict[str, Any],
    base_config: Dict[str, Any],
    dataset_path: Path,
    calibrator_script: Path,
    trials_dir: Path,
    env: Dict[str, str],
    outer_folds: Optional[List[Dict[str, Any]]] = None,
    outer_test_weeks: Optional[int] = None,
    outer_selection_metric: Optional[str] = None,
) -> Dict[str, Any]:
    trial_dir = trials_dir / f"trial_{idx:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_config = _build_trial_config(
        base_config,
        features=trial["features"],
        c_value=trial["C"],
        calibration=trial["calibration"],
        foundation_weight=trial["foundation_weight"],
        trading_universe_upweight=trial["trading_universe_upweight"],
        ticker_intercepts=trial["ticker_intercepts"],
        ticker_interactions=trial["ticker_interactions"],
        out_dir=trial_dir,
        skip_test_metrics=True,
    )
    _write_json(trial_dir / "trial_config.json", trial_config)

    return_code, runtime, stdout_tail, stderr_tail = _run_calibrator(
        trial_config=trial_config,
        dataset_path=dataset_path,
        calibrator_script=calibrator_script,
        out_dir=trial_dir,
        env=env,
        skip_test_metrics=True,
        fast_trial=True,
    )
    last_log_lines = (stdout_tail + stderr_tail)[-12:]

    metrics = _parse_metrics_for_split(trial_dir / "metrics.csv", split="val")
    fold_stats = _parse_fold_deltas(trial_dir / "fold_deltas.csv")

    status = "success"
    error_message: Optional[str] = None
    if return_code != 0 or metrics is None:
        status = "failed"
        error_message = f"Trial {idx:03d} failed."

    score = None
    if status == "success" and metrics:
        score = metrics["delta_logloss"]
    elif status == "success" and fold_stats:
        score = fold_stats["mean_delta_logloss"]

    outer_summary: Optional[Dict[str, Any]] = None
    outer_score: Optional[float] = None
    if outer_folds:
        outer_rows: List[Dict[str, Any]] = []
        outer_deltas: List[float] = []
        for fold in outer_folds:
            fold_idx = int(fold["fold"])
            fold_csv = Path(fold["csv_path"])
            fold_dir = trial_dir / f"outer_fold_{fold_idx:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_config = _build_trial_config(
                base_config,
                features=trial["features"],
                c_value=trial["C"],
                calibration=trial["calibration"],
                foundation_weight=trial["foundation_weight"],
                trading_universe_upweight=trial["trading_universe_upweight"],
                ticker_intercepts=trial["ticker_intercepts"],
                ticker_interactions=trial["ticker_interactions"],
                out_dir=fold_dir,
                test_weeks_override=outer_test_weeks,
                skip_test_metrics=False,
            )
            _write_json(fold_dir / "trial_config.json", fold_config)
            fold_return, fold_runtime, _, _ = _run_calibrator(
                trial_config=fold_config,
                dataset_path=fold_csv,
                calibrator_script=calibrator_script,
                out_dir=fold_dir,
                env=env,
                skip_test_metrics=False,
                fast_trial=True,
            )
            fold_metrics = _parse_metrics_for_split(fold_dir / "metrics.csv", split="test")
            delta = fold_metrics["delta_logloss"] if fold_metrics else None
            if delta is not None:
                outer_deltas.append(delta)
            outer_rows.append(
                {
                    "fold": fold_idx,
                    "status": "success" if fold_return == 0 and delta is not None else "failed",
                    "runtime_seconds": fold_runtime,
                    "model_logloss": fold_metrics["model_logloss"] if fold_metrics else None,
                    "baseline_logloss": fold_metrics["baseline_logloss"] if fold_metrics else None,
                    "delta_logloss": delta,
                }
            )

        if outer_rows:
            pd.DataFrame(outer_rows).to_csv(trial_dir / "outer_fold_results.csv", index=False)

        if outer_deltas:
            outer_median = float(statistics.median(outer_deltas))
            outer_mean = float(statistics.mean(outer_deltas))
            outer_worst = float(max(outer_deltas))
            outer_std = float(statistics.stdev(outer_deltas)) if len(outer_deltas) > 1 else 0.0
            outer_se = float(outer_std / math.sqrt(len(outer_deltas))) if outer_deltas else None
            improved = int(sum(1 for value in outer_deltas if value < 0))
            outer_summary = {
                "outer_median_delta_logloss": outer_median,
                "outer_mean_delta_logloss": outer_mean,
                "outer_worst_delta_logloss": outer_worst,
                "outer_std_delta_logloss": outer_std,
                "outer_se_delta_logloss": outer_se,
                "outer_n_folds": int(len(outer_deltas)),
                "outer_improved_folds": improved,
            }
            metric = str(outer_selection_metric or DEFAULT_OUTER_SELECTION_METRIC).strip().lower()
            if metric not in {"median_delta_logloss", "worst_delta_logloss", "mean_delta_logloss"}:
                metric = DEFAULT_OUTER_SELECTION_METRIC
            if metric == "worst_delta_logloss":
                outer_score = outer_worst
            elif metric == "mean_delta_logloss":
                outer_score = outer_mean
            else:
                outer_score = outer_median

    if outer_score is not None:
        score = outer_score

    trial_result = {
        "trial_id": idx,
        "status": status,
        "runtime_seconds": runtime,
        "features": trial["features"],
        "C": trial["C"],
        "calibration": trial["calibration"],
        "trading_universe_upweight": trial["trading_universe_upweight"],
        "foundation_weight": trial["foundation_weight"],
        "ticker_intercepts": trial["ticker_intercepts"],
        "ticker_interactions": trial["ticker_interactions"],
        "model_logloss": metrics["model_logloss"] if metrics else None,
        "baseline_logloss": metrics["baseline_logloss"] if metrics else None,
        "delta_logloss": metrics["delta_logloss"] if metrics else None,
        "mean_delta_logloss": fold_stats["mean_delta_logloss"] if fold_stats else None,
        "std_delta_logloss": fold_stats["std_delta_logloss"] if fold_stats else None,
        "se_delta_logloss": fold_stats["se_delta_logloss"] if fold_stats else None,
        "n_folds": fold_stats["n_folds"] if fold_stats else None,
        "folds_improved": fold_stats["folds_improved"] if fold_stats else None,
        "worst_delta_logloss": fold_stats["worst_delta_logloss"] if fold_stats else None,
        "score": score,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }
    if outer_summary:
        trial_result.update(outer_summary)
    _write_json(trial_dir / "trial_result.json", trial_result)

    leaderboard_row = None
    if status == "success":
        leaderboard_row = {
            "trial_id": idx,
            "score": score,
            "delta_logloss": trial_result["delta_logloss"],
            "mean_delta_logloss": trial_result["mean_delta_logloss"],
            "std_delta_logloss": trial_result["std_delta_logloss"],
            "se_delta_logloss": trial_result["se_delta_logloss"],
            "n_folds": trial_result["n_folds"],
            "folds_improved": trial_result["folds_improved"],
            "worst_delta_logloss": trial_result["worst_delta_logloss"],
            "features": ",".join(trial["features"]),
            "C": trial["C"],
            "calibration": trial["calibration"],
            "ticker_intercepts": trial["ticker_intercepts"],
            "ticker_interactions": trial["ticker_interactions"],
            "trading_universe_upweight": trial["trading_universe_upweight"],
            "foundation_weight": trial["foundation_weight"],
        }
        if outer_summary:
            leaderboard_row.update(outer_summary)

    return {
        "idx": idx,
        "status": status,
        "score": score,
        "trial_result": trial_result,
        "leaderboard_row": leaderboard_row,
        "last_log_lines": last_log_lines,
        "error_message": error_message,
    }


def _run_trial_inprocess(
    idx: int,
    *,
    trial: Dict[str, Any],
    base_config: Dict[str, Any],
    dataset_path: Path,
    trials_dir: Path,
    base_cache: CalibrationCache,
    base_unsupported_controls: Dict[str, Any],
    outer_fold_caches: Optional[Dict[int, CalibrationCache]] = None,
    outer_test_weeks: Optional[int] = None,
    outer_selection_metric: Optional[str] = None,
) -> Dict[str, Any]:
    trial_dir = trials_dir / f"trial_{idx:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_config = _build_trial_config(
        base_config,
        features=trial["features"],
        c_value=trial["C"],
        calibration=trial["calibration"],
        foundation_weight=trial["foundation_weight"],
        trading_universe_upweight=trial["trading_universe_upweight"],
        ticker_intercepts=trial["ticker_intercepts"],
        ticker_interactions=trial["ticker_interactions"],
        out_dir=trial_dir,
        skip_test_metrics=True,
    )

    trial_args, trial_unknown = build_args_from_config(trial_config, calibrate_args=[])
    trial_args.csv = str(dataset_path)
    try:
        _normalize_calibration_args(trial_args)
    except ValueError as exc:
        return {
            "idx": idx,
            "status": "failed",
            "score": None,
            "trial_result": {},
            "leaderboard_row": None,
            "last_log_lines": [],
            "error_message": f"Trial {idx:03d} failed: {exc}",
        }
    try:
        n_bins = _coerce_positive_bin_count(trial_args.n_bins, default=10, arg_name="--n-bins")
        eceq_bins = _coerce_positive_bin_count(trial_args.eceq_bins, default=n_bins, arg_name="--eceq-bins")
    except ValueError as exc:
        return {
            "idx": idx,
            "status": "failed",
            "score": None,
            "trial_result": {},
            "leaderboard_row": None,
            "last_log_lines": [],
            "error_message": f"Trial {idx:03d} failed: {exc}",
        }

    start = time.monotonic()
    run_result = run_calibration_from_cache(
        base_cache,
        trial_args,
        trial_dir,
        config_payload=trial_config,
        unknown=trial_unknown,
        unsupported_controls=base_unsupported_controls,
        n_bins=n_bins,
        eceq_bins=eceq_bins,
        fast_trial=True,
        skip_test_metrics=True,
    )
    runtime = round(time.monotonic() - start, 3)

    metrics = _parse_metrics_for_split(trial_dir / "metrics.csv", split="val")
    fold_stats = _parse_fold_deltas(trial_dir / "fold_deltas.csv")

    status = "success"
    error_message: Optional[str] = None
    if run_result.exit_code != 0 or metrics is None:
        status = "failed"
        error_message = f"Trial {idx:03d} failed."

    score = None
    if status == "success" and metrics:
        score = metrics["delta_logloss"]
    elif status == "success" and fold_stats:
        score = fold_stats["mean_delta_logloss"]

    outer_summary: Optional[Dict[str, Any]] = None
    outer_score: Optional[float] = None
    if outer_fold_caches:
        outer_rows: List[Dict[str, Any]] = []
        outer_deltas: List[float] = []
        for fold_id, fold_cache in outer_fold_caches.items():
            fold_dir = trial_dir / f"outer_fold_{int(fold_id):02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_config = _build_trial_config(
                base_config,
                features=trial["features"],
                c_value=trial["C"],
                calibration=trial["calibration"],
                foundation_weight=trial["foundation_weight"],
                trading_universe_upweight=trial["trading_universe_upweight"],
                ticker_intercepts=trial["ticker_intercepts"],
                ticker_interactions=trial["ticker_interactions"],
                out_dir=fold_dir,
                test_weeks_override=outer_test_weeks,
                skip_test_metrics=False,
            )
            fold_args, fold_unknown = build_args_from_config(fold_config, calibrate_args=[])
            fold_args.csv = str(dataset_path)
            try:
                _normalize_calibration_args(fold_args)
                fold_n_bins = _coerce_positive_bin_count(fold_args.n_bins, default=10, arg_name="--n-bins")
                fold_eceq_bins = _coerce_positive_bin_count(
                    fold_args.eceq_bins, default=fold_n_bins, arg_name="--eceq-bins"
                )
            except ValueError:
                fold_metrics = None
                fold_runtime = 0.0
                delta = None
            else:
                fold_start = time.monotonic()
                fold_result = run_calibration_from_cache(
                    fold_cache,
                    fold_args,
                    fold_dir,
                    config_payload=fold_config,
                    unknown=fold_unknown,
                    unsupported_controls=base_unsupported_controls,
                    n_bins=fold_n_bins,
                    eceq_bins=fold_eceq_bins,
                    fast_trial=True,
                    skip_test_metrics=False,
                )
                fold_runtime = round(time.monotonic() - fold_start, 3)
                fold_metrics = _parse_metrics_for_split(fold_dir / "metrics.csv", split="test")
                if fold_result.exit_code != 0:
                    fold_metrics = None
                delta = fold_metrics["delta_logloss"] if fold_metrics else None

            if delta is not None:
                outer_deltas.append(delta)
            outer_rows.append(
                {
                    "fold": int(fold_id),
                    "status": "success" if fold_metrics is not None else "failed",
                    "runtime_seconds": fold_runtime,
                    "model_logloss": fold_metrics["model_logloss"] if fold_metrics else None,
                    "baseline_logloss": fold_metrics["baseline_logloss"] if fold_metrics else None,
                    "delta_logloss": delta,
                }
            )

        if outer_rows:
            pd.DataFrame(outer_rows).to_csv(trial_dir / "outer_fold_results.csv", index=False)

        if outer_deltas:
            outer_median = float(statistics.median(outer_deltas))
            outer_mean = float(statistics.mean(outer_deltas))
            outer_worst = float(max(outer_deltas))
            outer_std = float(statistics.stdev(outer_deltas)) if len(outer_deltas) > 1 else 0.0
            outer_se = float(outer_std / math.sqrt(len(outer_deltas))) if outer_deltas else None
            improved = int(sum(1 for value in outer_deltas if value < 0))
            outer_summary = {
                "outer_median_delta_logloss": outer_median,
                "outer_mean_delta_logloss": outer_mean,
                "outer_worst_delta_logloss": outer_worst,
                "outer_std_delta_logloss": outer_std,
                "outer_se_delta_logloss": outer_se,
                "outer_n_folds": int(len(outer_deltas)),
                "outer_improved_folds": improved,
            }
            metric = str(outer_selection_metric or DEFAULT_OUTER_SELECTION_METRIC).strip().lower()
            if metric not in {"median_delta_logloss", "worst_delta_logloss", "mean_delta_logloss"}:
                metric = DEFAULT_OUTER_SELECTION_METRIC
            if metric == "worst_delta_logloss":
                outer_score = outer_worst
            elif metric == "mean_delta_logloss":
                outer_score = outer_mean
            else:
                outer_score = outer_median

    if outer_score is not None:
        score = outer_score

    trial_result = {
        "trial_id": idx,
        "status": status,
        "runtime_seconds": runtime,
        "features": trial["features"],
        "C": trial["C"],
        "calibration": trial["calibration"],
        "trading_universe_upweight": trial["trading_universe_upweight"],
        "foundation_weight": trial["foundation_weight"],
        "ticker_intercepts": trial["ticker_intercepts"],
        "ticker_interactions": trial["ticker_interactions"],
        "model_logloss": metrics["model_logloss"] if metrics else None,
        "baseline_logloss": metrics["baseline_logloss"] if metrics else None,
        "delta_logloss": metrics["delta_logloss"] if metrics else None,
        "mean_delta_logloss": fold_stats["mean_delta_logloss"] if fold_stats else None,
        "std_delta_logloss": fold_stats["std_delta_logloss"] if fold_stats else None,
        "se_delta_logloss": fold_stats["se_delta_logloss"] if fold_stats else None,
        "n_folds": fold_stats["n_folds"] if fold_stats else None,
        "folds_improved": fold_stats["folds_improved"] if fold_stats else None,
        "worst_delta_logloss": fold_stats["worst_delta_logloss"] if fold_stats else None,
        "score": score,
        "stdout_tail": [],
        "stderr_tail": [],
    }
    if outer_summary:
        trial_result.update(outer_summary)
    _write_json(trial_dir / "trial_result.json", trial_result)

    leaderboard_row = None
    if status == "success":
        leaderboard_row = {
            "trial_id": idx,
            "score": score,
            "delta_logloss": trial_result["delta_logloss"],
            "mean_delta_logloss": trial_result["mean_delta_logloss"],
            "std_delta_logloss": trial_result["std_delta_logloss"],
            "se_delta_logloss": trial_result["se_delta_logloss"],
            "n_folds": trial_result["n_folds"],
            "folds_improved": trial_result["folds_improved"],
            "worst_delta_logloss": trial_result["worst_delta_logloss"],
            "features": ",".join(trial["features"]),
            "C": trial["C"],
            "calibration": trial["calibration"],
            "ticker_intercepts": trial["ticker_intercepts"],
            "ticker_interactions": trial["ticker_interactions"],
            "trading_universe_upweight": trial["trading_universe_upweight"],
            "foundation_weight": trial["foundation_weight"],
        }
        if outer_summary:
            leaderboard_row.update(outer_summary)

    return {
        "idx": idx,
        "status": status,
        "score": score,
        "trial_result": trial_result,
        "leaderboard_row": leaderboard_row,
        "last_log_lines": [],
        "error_message": error_message,
    }


def _trial_simplicity_key(trial: Dict[str, Any]) -> Tuple:
    features = trial.get("features", [])
    if isinstance(features, str):
        features = [f for f in features.split(",") if f]
    c_val = float(trial.get("C") or trial.get("c") or 0.0)
    intercept_on = 1 if trial.get("ticker_intercepts") == "on" else 0
    interactions_on = 1 if trial.get("ticker_interactions") else 0
    upweight = float(trial.get("trading_universe_upweight") or 1.0)
    foundation_weight = float(trial.get("foundation_weight") or 1.0)
    return (
        len(features),
        intercept_on,
        interactions_on,
        c_val,
        abs(upweight - 1.0),
        abs(foundation_weight - 1.0),
    )


def _write_leaderboard(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.sort_values(by=["score", "mean_delta_logloss"], inplace=True, na_position="last")
    df.to_csv(path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-calibrate logit model by enumerating a curated grid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-json", required=True, help="Path to config JSON payload.")
    parser.add_argument("--out-dir", required=True, help="Output directory for auto-search.")
    parser.add_argument("--calibrator-script", default=str(DEFAULT_CALIBRATOR_SCRIPT))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument(
        "--execution-mode",
        default="inprocess",
        choices=["inprocess", "subprocess"],
        help="Run calibration in-process (fast) or via subprocess (legacy).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(Path(args.config_json))

    base_config = config.get("base_config")
    if not isinstance(base_config, dict):
        raise SystemExit("base_config is required in config-json.")

    search_cfg = config.get("search") or {}
    if not isinstance(search_cfg, dict):
        search_cfg = {}

    execution_mode = str(args.execution_mode or "inprocess").strip().lower()
    use_inprocess = execution_mode == "inprocess"

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_model_dir = out_dir / "selected_model"
    auto_search_dir = out_dir / "auto_search"
    auto_search_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(base_config.get("csv") or "")
    if not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    available_cols = set(_read_dataset_columns(dataset_path))
    allow_risky = bool(search_cfg.get("allow_risky_features", False))
    feature_sets = _sanitize_feature_sets(
        search_cfg.get("feature_sets") or DEFAULT_FEATURE_SETS,
        available=available_cols,
        allow_risky=allow_risky,
    )
    if allow_risky:
        risky_bundle = [BASE_FEATURE, "rv20", "abs_log_m_fwd", "log_rel_spread"]
        risky_bundle += [feat for feat in sorted(RISKY_FEATURES) if feat in available_cols]
        risky_bundle = _dedupe_preserve_order(risky_bundle)
        if risky_bundle and risky_bundle not in feature_sets:
            feature_sets.append(risky_bundle)
    if not feature_sets:
        raise SystemExit("No valid feature sets remain after availability/risk filtering.")

    c_values = search_cfg.get("c_values") or DEFAULT_C_VALUES
    calibration_methods = search_cfg.get("calibration_methods") or DEFAULT_CAL_METHODS
    trading_universe_upweights = search_cfg.get("trading_universe_upweight") or DEFAULT_UPWEIGHTS
    foundation_weights = search_cfg.get("foundation_weight") or DEFAULT_FOUNDATION_WEIGHTS
    ticker_intercepts = search_cfg.get("ticker_intercepts") or DEFAULT_TICKER_INTERCEPTS
    advanced_interactions = bool(search_cfg.get("advanced_interactions", False))
    max_trials = search_cfg.get("max_trials") if search_cfg.get("max_trials") is not None else args.max_trials
    selection_rule = search_cfg.get("selection_rule", "one_se")
    epsilon = float(search_cfg.get("epsilon", 0.002))
    accept_delta_threshold = float(
        search_cfg.get("accept_delta_threshold", DEFAULT_ACCEPT_DELTA_THRESHOLD)
    )
    min_improve_fraction = _coerce_fraction(
        search_cfg.get("min_improve_fraction"),
        default=DEFAULT_MIN_IMPROVE_FRACTION,
    )
    max_worst_delta = _coerce_nonnegative_float(
        search_cfg.get("max_worst_delta"),
        default=DEFAULT_MAX_WORST_DELTA,
    )
    outer_folds = _coerce_positive_int(search_cfg.get("outer_folds"), default=0)
    outer_test_weeks = _coerce_positive_int(
        search_cfg.get("outer_test_weeks"),
        default=DEFAULT_OUTER_TEST_WEEKS,
    )
    outer_gap_weeks = _coerce_positive_int(
        search_cfg.get("outer_gap_weeks"),
        default=DEFAULT_OUTER_GAP_WEEKS,
    )
    outer_selection_metric = str(
        search_cfg.get("outer_selection_metric") or DEFAULT_OUTER_SELECTION_METRIC
    ).strip().lower()
    if outer_selection_metric not in {"median_delta_logloss", "worst_delta_logloss", "mean_delta_logloss"}:
        outer_selection_metric = DEFAULT_OUTER_SELECTION_METRIC
    outer_min_improve_fraction = _coerce_fraction(
        search_cfg.get("outer_min_improve_fraction"),
        default=DEFAULT_OUTER_MIN_IMPROVE_FRACTION,
    )
    outer_max_worst_delta = _coerce_nonnegative_float(
        search_cfg.get("outer_max_worst_delta"),
        default=DEFAULT_OUTER_MAX_WORST_DELTA,
    )

    trial_grid: List[Dict[str, Any]] = []
    for feats in feature_sets:
        for c_val in c_values:
            for cal in calibration_methods:
                for upweight in trading_universe_upweights:
                    for f_weight in foundation_weights:
                        for intercept_flag in ticker_intercepts:
                            if advanced_interactions:
                                interactions_opts = [False, True]
                            else:
                                interactions_opts = [False]
                            for interactions in interactions_opts:
                                trial_grid.append(
                                    {
                                        "features": feats,
                                        "C": c_val,
                                        "calibration": cal,
                                        "trading_universe_upweight": upweight,
                                        "foundation_weight": f_weight,
                                        "ticker_intercepts": intercept_flag,
                                        "ticker_interactions": interactions,
                                    }
                                )

    if max_trials is not None:
        trial_grid = trial_grid[: int(max_trials)]

    base_args, _ = build_args_from_config(base_config, calibrate_args=[])
    base_args.csv = str(dataset_path)
    try:
        _normalize_calibration_args(base_args)
    except ValueError as exc:
        raise SystemExit(f"[base-model] ERROR: {exc}")

    bootstrap_precheck_error = _validate_bootstrap_precheck(base_args, available_cols)
    if bootstrap_precheck_error:
        reason = (
            f"{bootstrap_precheck_error} Set allow_iid_bootstrap=true to permit iid fallback "
            "or choose a valid bootstrap_group."
        )
        write_progress(
            auto_search_dir,
            stage="failed_precheck",
            trials_total=len(trial_grid),
            trials_done=0,
            trials_failed=0,
            best_score=None,
            last_error=reason,
            phase="failed_precheck",
            message=reason,
            force=True,
        )
        _write_json(
            auto_search_dir / "auto_search_summary.json",
            {
                "run_name": config.get("run_name"),
                "status": "failed_precheck",
                "reason": reason,
                "bootstrap": {
                    "bootstrap_ci": bool(getattr(base_args, "bootstrap_ci", False)),
                    "bootstrap_group": str(getattr(base_args, "bootstrap_group", "auto")),
                    "allow_iid_bootstrap": bool(getattr(base_args, "allow_iid_bootstrap", False)),
                },
            },
        )
        _write_run_manifest(
            out_dir,
            auto_status="failed_precheck",
            selected_trial_id=None,
            selected_model_relpath=None,
            auto_search_relpath="auto_search",
        )
        raise SystemExit(reason)

    feature_union = _dedupe_preserve_order([feat for feats in feature_sets for feat in feats])
    base_cache: Optional[CalibrationCache] = None
    base_unsupported_controls: Dict[str, Any] = {}
    base_trainer_warnings: List[str] = []
    df_full: Optional[pd.DataFrame] = None
    if use_inprocess:
        _limit_blas_threads()
        df_full = _load_dataset(dataset_path)
        base_unsupported_controls = _collect_unsupported_compat_args(base_args)
        if base_unsupported_controls:
            unsupported_msg = (
                "[base-model] Unsupported manual controls were provided and ignored: "
                + ", ".join(f"{k}={v}" for k, v in base_unsupported_controls.items())
            )
            if base_args.strict_args:
                raise SystemExit(f"[base-model] ERROR: {unsupported_msg}")
            print(f"[WARN] {unsupported_msg}")
            base_trainer_warnings.append(unsupported_msg)
        base_cache = build_calibration_cache(
            df_full,
            base_args,
            trainer_warnings=base_trainer_warnings,
            numeric_features=feature_union,
            fast_trial=True,
        )
        if base_cache is None:
            raise SystemExit("Failed to prepare in-process calibration cache.")
        if _parity_enabled() and trial_grid:
            _run_parity_check(
                trial=trial_grid[0],
                base_config=base_config,
                dataset_path=dataset_path,
                out_dir=auto_search_dir,
                base_cache=base_cache,
                base_unsupported_controls=base_unsupported_controls,
                calibrator_path=Path(args.calibrator_script).resolve(),
                env=_build_env(),
            )

    base_test_weeks = _coerce_positive_int(
        (base_config.get("split") or {}).get("test_window_weeks") or base_config.get("test_weeks"),
        default=0,
    )
    outer_cv_summary: Optional[Dict[str, Any]] = None
    outer_fold_specs: List[Dict[str, Any]] = []
    if outer_folds > 0:
        write_progress(
            auto_search_dir,
            stage="preparing_outer_folds",
            trials_total=len(trial_grid),
            trials_done=0,
            trials_failed=0,
            best_score=None,
            last_error=None,
            message="Preparing outer backtest folds",
        )
        outer_fold_specs, outer_cv_summary = _prepare_outer_folds(
            dataset_path=dataset_path,
            out_dir=auto_search_dir,
            base_test_weeks=base_test_weeks,
            outer_folds=outer_folds,
            outer_test_weeks=outer_test_weeks,
            outer_gap_weeks=outer_gap_weeks,
            df=df_full if use_inprocess else None,
            materialize_csvs=not use_inprocess,
        )
        if outer_cv_summary is not None:
            outer_cv_summary["outer_selection_metric"] = outer_selection_metric
            outer_cv_summary["outer_min_improve_fraction"] = float(outer_min_improve_fraction)
            outer_cv_summary["outer_max_worst_delta"] = float(outer_max_worst_delta)
        if not outer_fold_specs:
            outer_folds = 0

    outer_fold_caches: Dict[int, CalibrationCache] = {}
    if use_inprocess and outer_folds > 0 and outer_fold_specs and base_cache is not None:
        outer_args = argparse.Namespace(**vars(base_args))
        outer_args.test_weeks = int(outer_test_weeks)
        try:
            _normalize_calibration_args(outer_args)
        except ValueError as exc:
            raise SystemExit(f"[base-model] ERROR: {exc}")
        for spec in outer_fold_specs:
            allowed_idx = spec.get("allowed_idx") or []
            if not allowed_idx:
                continue
            fold_df = df_full.loc[allowed_idx].copy() if df_full is not None else pd.DataFrame()
            fold_cache = build_calibration_cache(
                fold_df,
                outer_args,
                trainer_warnings=list(base_trainer_warnings),
                numeric_features=feature_union,
                fast_trial=True,
            )
            if fold_cache is None:
                outer_fold_caches = {}
                outer_folds = 0
                outer_fold_specs = []
                if outer_cv_summary is not None:
                    outer_cv_summary.setdefault("warnings", []).append(
                        "Failed to prepare outer fold caches; outer CV disabled."
                    )
                    outer_cv_summary["enabled"] = False
                    outer_cv_summary["outer_folds_used"] = 0
                break
            outer_fold_caches[int(spec["fold"])] = fold_cache
        if not outer_fold_caches:
            outer_folds = 0
            if outer_cv_summary is not None:
                outer_cv_summary.setdefault("warnings", []).append(
                    "No usable outer fold caches were built; outer CV disabled."
                )
                outer_cv_summary["enabled"] = False
                outer_cv_summary["outer_folds_used"] = 0

    write_progress(
        auto_search_dir,
        stage="enumerating",
        trials_total=len(trial_grid),
        trials_done=0,
        trials_failed=0,
        best_score=None,
        last_error=None,
        message="Enumerating candidates",
    )

    trials_dir = auto_search_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_rows: List[Dict[str, Any]] = []
    best_score: Optional[float] = None
    trials_done = 0
    trials_failed = 0
    last_error: Optional[str] = None

    calibrator_path = Path(args.calibrator_script).resolve()
    env = _build_env()
    if args.parallel and args.parallel > 0:
        parallel = int(args.parallel)
    else:
        cpu = os.cpu_count() or 2
        parallel = max(1, min(8, cpu - 1))
    parallel = min(parallel, len(trial_grid)) if trial_grid else 1

    def _consume_result(result: Dict[str, Any]) -> None:
        nonlocal trials_done, trials_failed, best_score, last_error
        status = result["status"]
        score = result["score"]
        if status != "success":
            trials_failed += 1
            last_error = result.get("error_message") or last_error
        if score is not None:
            if best_score is None or score < best_score:
                best_score = score
        if result.get("leaderboard_row"):
            leaderboard_rows.append(result["leaderboard_row"])
        trials_done += 1

        top_candidates = None
        if leaderboard_rows:
            sorted_rows = sorted(
                leaderboard_rows,
                key=lambda r: (r.get("score") is None, r.get("score")),
            )[:5]
            top_candidates = [
                {
                    "rank": i + 1,
                    "score": row.get("score"),
                    "features": row.get("features"),
                    "C": row.get("C"),
                    "calibration": row.get("calibration"),
                    "trading_universe_upweight": row.get("trading_universe_upweight"),
                    "foundation_weight": row.get("foundation_weight"),
                    "ticker_intercepts": row.get("ticker_intercepts"),
                    "ticker_interactions": row.get("ticker_interactions"),
                }
                for i, row in enumerate(sorted_rows)
            ]

        write_progress(
            auto_search_dir,
            stage="training_trials",
            trials_total=len(trial_grid),
            trials_done=trials_done,
            trials_failed=trials_failed,
            best_score=best_score,
            last_error=last_error,
            phase="training_trials",
            candidate_index=trials_done,
            candidate_total=len(trial_grid),
            message=f"Completed {trials_done}/{len(trial_grid)} candidates",
            last_log_lines=result.get("last_log_lines"),
            top_candidates=top_candidates,
        )

    if parallel <= 1 or len(trial_grid) <= 1:
        for idx, trial in enumerate(trial_grid, start=1):
            if use_inprocess:
                outcome = _run_trial_inprocess(
                    idx,
                    trial=trial,
                    base_config=base_config,
                    dataset_path=dataset_path,
                    trials_dir=trials_dir,
                    base_cache=base_cache,
                    base_unsupported_controls=base_unsupported_controls,
                    outer_fold_caches=outer_fold_caches or None,
                    outer_test_weeks=outer_test_weeks if outer_folds > 0 else None,
                    outer_selection_metric=outer_selection_metric if outer_folds > 0 else None,
                )
            else:
                outcome = _run_trial(
                    idx,
                    trial=trial,
                    base_config=base_config,
                    dataset_path=dataset_path,
                    calibrator_script=calibrator_path,
                    trials_dir=trials_dir,
                    env=env,
                    outer_folds=outer_fold_specs or None,
                    outer_test_weeks=outer_test_weeks if outer_folds > 0 else None,
                    outer_selection_metric=outer_selection_metric if outer_folds > 0 else None,
                )
            _consume_result(outcome)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = []
            for idx, trial in enumerate(trial_grid, start=1):
                if use_inprocess:
                    futures.append(
                        executor.submit(
                            _run_trial_inprocess,
                            idx,
                            trial=trial,
                            base_config=base_config,
                            dataset_path=dataset_path,
                            trials_dir=trials_dir,
                            base_cache=base_cache,
                            base_unsupported_controls=base_unsupported_controls,
                            outer_fold_caches=outer_fold_caches or None,
                            outer_test_weeks=outer_test_weeks if outer_folds > 0 else None,
                            outer_selection_metric=outer_selection_metric if outer_folds > 0 else None,
                        )
                    )
                else:
                    futures.append(
                        executor.submit(
                            _run_trial,
                            idx,
                            trial=trial,
                            base_config=base_config,
                            dataset_path=dataset_path,
                            calibrator_script=calibrator_path,
                            trials_dir=trials_dir,
                            env=env,
                            outer_folds=outer_fold_specs or None,
                            outer_test_weeks=outer_test_weeks if outer_folds > 0 else None,
                            outer_selection_metric=outer_selection_metric if outer_folds > 0 else None,
                        )
                    )
            for future in as_completed(futures):
                outcome = future.result()
                _consume_result(outcome)

    _write_leaderboard(auto_search_dir / "auto_search_leaderboard.csv", leaderboard_rows)

    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        if not np.isfinite(parsed):
            return None
        return parsed

    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        return bool(value)

    def _baseline_snapshot(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        return {
            "trial_id": row.get("trial_id"),
            "score": row.get("score"),
            "model_logloss": row.get("model_logloss"),
            "baseline_logloss": row.get("baseline_logloss"),
            "delta_logloss": row.get("delta_logloss"),
            "mean_delta_logloss": row.get("mean_delta_logloss"),
            "outer_median_delta_logloss": row.get("outer_median_delta_logloss"),
            "outer_worst_delta_logloss": row.get("outer_worst_delta_logloss"),
            "features": row.get("features"),
            "C": row.get("C"),
            "calibration": row.get("calibration"),
        }

    def _materialize_best_candidate(
        candidate_row: Dict[str, Any],
        *,
        progress_stage: str,
        progress_message: str,
    ) -> Dict[str, Any]:
        write_progress(
            auto_search_dir,
            stage=progress_stage,
            trials_total=len(trial_grid),
            trials_done=trials_done,
            trials_failed=trials_failed,
            best_score=best_score,
            last_error=last_error,
            phase=progress_stage,
            message=progress_message,
        )

        features_value = candidate_row.get("features")
        if isinstance(features_value, str):
            feature_list = [token for token in features_value.split(",") if token]
        elif isinstance(features_value, list):
            feature_list = [str(token) for token in features_value if str(token)]
        else:
            feature_list = []
        if not feature_list:
            raise SystemExit("Unable to materialize best candidate: missing features.")

        c_value = _safe_float(candidate_row.get("C"))
        foundation_weight = _safe_float(candidate_row.get("foundation_weight"))
        trading_upweight = _safe_float(candidate_row.get("trading_universe_upweight"))
        if c_value is None or foundation_weight is None or trading_upweight is None:
            raise SystemExit("Unable to materialize best candidate: missing numeric trial controls.")

        ticker_intercepts_value = str(candidate_row.get("ticker_intercepts") or "off")
        calibration_value = str(candidate_row.get("calibration") or "none")
        ticker_interactions_value = _as_bool(candidate_row.get("ticker_interactions"))

        if selected_model_dir.exists():
            shutil.rmtree(selected_model_dir, ignore_errors=True)
        selected_model_dir.mkdir(parents=True, exist_ok=True)

        final_config = _build_trial_config(
            base_config,
            features=feature_list,
            c_value=float(c_value),
            calibration=calibration_value,
            foundation_weight=float(foundation_weight),
            trading_universe_upweight=float(trading_upweight),
            ticker_intercepts=ticker_intercepts_value,
            ticker_interactions=ticker_interactions_value,
            out_dir=selected_model_dir,
            skip_test_metrics=False,
            trial_mode=False,
        )
        _write_json(auto_search_dir / "best_config.json", final_config)

        if use_inprocess:
            final_args, final_unknown = build_args_from_config(final_config, calibrate_args=[])
            final_args.csv = str(dataset_path)
            try:
                _normalize_calibration_args(final_args)
                final_n_bins = _coerce_positive_bin_count(
                    final_args.n_bins,
                    default=10,
                    arg_name="--n-bins",
                )
                final_eceq_bins = _coerce_positive_bin_count(
                    final_args.eceq_bins,
                    default=final_n_bins,
                    arg_name="--eceq-bins",
                )
            except ValueError as exc:
                raise SystemExit(f"[base-model] ERROR: {exc}")
            final_result = run_calibration_from_cache(
                base_cache,
                final_args,
                selected_model_dir,
                config_payload=final_config,
                unknown=final_unknown,
                unsupported_controls=base_unsupported_controls,
                n_bins=final_n_bins,
                eceq_bins=final_eceq_bins,
                fast_trial=False,
                skip_test_metrics=False,
            )
            if final_result.exit_code != 0:
                raise SystemExit(f"Final refit failed (exit code {final_result.exit_code}).")
        else:
            final_cmd = [
                sys.executable,
                str(Path(args.calibrator_script).resolve()),
                "--csv",
                str(dataset_path),
                "--out-dir",
                str(selected_model_dir),
                "--model-kind",
                "calibrate",
                "--config-json",
                str(auto_search_dir / "best_config.json"),
            ]
            final_proc = subprocess.run(
                final_cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(REPO_ROOT),
                env=_build_env(),
            )
            if final_proc.returncode != 0:
                raise SystemExit(f"Final refit failed (return code {final_proc.returncode}).")

        required_selected_files = [
            selected_model_dir / "metrics.csv",
            selected_model_dir / "metadata.json",
            selected_model_dir / "config.executed.json",
            selected_model_dir / "final_model.joblib",
            selected_model_dir / "base_pipeline.joblib",
        ]
        missing_selected = [
            str(path.relative_to(out_dir))
            for path in required_selected_files
            if not path.exists()
        ]
        if missing_selected:
            raise SystemExit(
                "Final selected model artifacts missing: " + ", ".join(missing_selected)
            )

        return {
            "trial_id": candidate_row.get("trial_id"),
            "score": candidate_row.get("score"),
            "selected_model_relpath": "selected_model",
            "best_config_path": str((auto_search_dir / "best_config.json").relative_to(out_dir)),
            "status": "materialized",
        }

    def _write_no_viable(
        *,
        reasons: List[str],
        reference_row: Optional[Dict[str, Any]],
        rejected_rows: Optional[List[Dict[str, Any]]] = None,
        materialized_best: Optional[Dict[str, Any]] = None,
    ) -> None:
        summary = {
            "run_name": config.get("run_name"),
            "status": "no_viable_model",
            "selection_rule": selection_rule,
            "epsilon": epsilon,
            "score_definition": None if outer_folds > 0 else "pooled_val_delta_logloss",
            "best_score": reference_row.get("score") if reference_row else None,
            "chosen": None,
            "no_viable_reasons": reasons,
            "acceptance": {
                "accept_delta_threshold": float(accept_delta_threshold),
                "min_improve_fraction": float(min_improve_fraction),
                "max_worst_delta": float(max_worst_delta),
                "outer_min_improve_fraction": float(outer_min_improve_fraction),
                "outer_max_worst_delta": float(outer_max_worst_delta),
            },
            "outer_cv": {
                "enabled": bool(outer_folds > 0),
                "outer_folds": int(outer_folds),
                "outer_test_weeks": int(outer_test_weeks),
                "outer_gap_weeks": int(outer_gap_weeks),
                "outer_selection_metric": outer_selection_metric,
                "outer_min_improve_fraction": float(outer_min_improve_fraction),
                "outer_max_worst_delta": float(outer_max_worst_delta),
            },
            "search_space": {
                "feature_sets": feature_sets,
                "c_values": c_values,
                "calibration_methods": calibration_methods,
                "trading_universe_upweight": trading_universe_upweights,
                "foundation_weight": foundation_weights,
                "ticker_intercepts": ticker_intercepts,
                "advanced_interactions": advanced_interactions,
            },
            "best_candidate_artifacts": materialized_best,
        }
        _write_json(auto_search_dir / "auto_search_summary.json", summary)
        _write_json(
            auto_search_dir / "auto_search_no_viable.json",
            {
                "status": "no_viable_model",
                "reasons": reasons,
                "baseline_snapshot": _baseline_snapshot(reference_row),
                "rejected_candidates": rejected_rows or [],
                "best_candidate_artifacts": materialized_best,
            },
        )
        selected_trial_id = None
        selected_model_relpath = None
        if materialized_best is not None:
            try:
                if materialized_best.get("trial_id") is not None:
                    selected_trial_id = int(materialized_best.get("trial_id"))
            except Exception:
                selected_trial_id = None
            selected_model_relpath = "selected_model"
        _write_run_manifest(
            out_dir,
            auto_status="no_viable_model",
            selected_trial_id=selected_trial_id,
            selected_model_relpath=selected_model_relpath,
            auto_search_relpath="auto_search",
        )
        if outer_cv_summary is not None:
            outer_cv_summary["selection"] = {
                "status": "no_viable_model",
                "selection_rule": selection_rule,
                "epsilon": epsilon,
                "chosen_trial_id": materialized_best.get("trial_id") if materialized_best else None,
                "best_score": reference_row.get("score") if reference_row else None,
                "selection_metric": outer_selection_metric,
                "reasons": reasons,
            }
            _write_json(auto_search_dir / "outer_cv_summary.json", outer_cv_summary)
        write_progress(
            auto_search_dir,
            stage="done_no_viable",
            trials_total=len(trial_grid),
            trials_done=trials_done,
            trials_failed=trials_failed,
            best_score=best_score,
            last_error=last_error,
            phase="done_no_viable",
            message="No viable model selected",
            force=True,
        )

    # Selection
    candidates = [row for row in leaderboard_rows if row.get("score") is not None]
    if not candidates:
        _write_no_viable(
            reasons=["No successful trials produced a usable score."],
            reference_row=None,
        )
        return

    best_candidate = min(candidates, key=lambda r: r["score"])
    if _safe_float(best_candidate.get("score")) is not None and float(best_candidate["score"]) > float(epsilon):
        materialized = _materialize_best_candidate(
            best_candidate,
            progress_stage="refit_best_candidate",
            progress_message="Materializing best leaderboard candidate artifacts",
        )
        _write_no_viable(
            reasons=[
                f"Best candidate score {float(best_candidate['score']):.6g} is worse than baseline by more than epsilon={float(epsilon):.6g}.",
            ],
            reference_row=best_candidate,
            materialized_best=materialized,
        )
        return

    def _central_delta(row: Dict[str, Any]) -> Optional[float]:
        if outer_folds > 0:
            central = _safe_float(row.get("outer_median_delta_logloss"))
            if central is not None:
                return central
        central = _safe_float(row.get("delta_logloss"))
        if central is not None:
            return central
        central = _safe_float(row.get("mean_delta_logloss"))
        if central is not None:
            return central
        return _safe_float(row.get("score"))

    def _acceptance_check(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        central = _central_delta(row)
        if central is None:
            reasons.append("missing_central_delta_metric")
        elif central > float(accept_delta_threshold):
            reasons.append(
                f"central_delta_logloss {central:.6g} > accept_delta_threshold {float(accept_delta_threshold):.6g}"
            )

        if outer_folds > 0 and row.get("outer_n_folds"):
            k = int(row.get("outer_n_folds") or 0)
            improved = int(row.get("outer_improved_folds") or 0)
            worst = _safe_float(row.get("outer_worst_delta_logloss"))
            min_required = int(math.ceil(float(outer_min_improve_fraction) * max(1, k)))
            if improved < min_required:
                reasons.append(f"outer_improved_folds {improved} < required {min_required}")
            if worst is None:
                reasons.append("missing_outer_worst_delta_logloss")
            elif worst > float(outer_max_worst_delta):
                reasons.append(
                    f"outer_worst_delta_logloss {worst:.6g} > cap {float(outer_max_worst_delta):.6g}"
                )
            return len(reasons) == 0, reasons

        k = int(row.get("n_folds") or 0)
        if k > 0:
            improved = int(row.get("folds_improved") or 0)
            worst = _safe_float(row.get("worst_delta_logloss"))
            min_required = int(math.ceil(float(min_improve_fraction) * max(1, k)))
        else:
            k = 1
            proxy_delta = _safe_float(row.get("delta_logloss"))
            if proxy_delta is None:
                proxy_delta = central
            improved = 1 if proxy_delta is not None and proxy_delta <= 0.0 else 0
            worst = proxy_delta
            min_required = int(math.ceil(float(min_improve_fraction) * k))
        if improved < min_required:
            reasons.append(f"improved_folds {improved} < required {min_required}")
        if worst is None:
            reasons.append("missing_worst_delta_logloss")
        elif worst > float(max_worst_delta):
            reasons.append(f"worst_delta_logloss {worst:.6g} > cap {float(max_worst_delta):.6g}")
        return len(reasons) == 0, reasons

    filtered: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    for row in candidates:
        accepted, reasons = _acceptance_check(row)
        if accepted:
            filtered.append(row)
        else:
            rejected_rows.append(
                {
                    "trial_id": row.get("trial_id"),
                    "score": row.get("score"),
                    "reasons": reasons,
                }
            )

    if not filtered:
        materialized = _materialize_best_candidate(
            best_candidate,
            progress_stage="refit_best_candidate",
            progress_message="Materializing best leaderboard candidate artifacts",
        )
        _write_no_viable(
            reasons=["No candidate passed acceptance gates."],
            reference_row=best_candidate,
            rejected_rows=rejected_rows,
            materialized_best=materialized,
        )
        return

    best = min(filtered, key=lambda r: r["score"])
    margin = epsilon
    if selection_rule == "one_se":
        se_best = best.get("outer_se_delta_logloss") if outer_folds > 0 else best.get("se_delta_logloss")
        if se_best is not None:
            margin = float(se_best)

    shortlist = [row for row in filtered if row["score"] <= best["score"] + margin]
    shortlist.sort(key=_trial_simplicity_key)
    chosen = shortlist[0]

    write_progress(
        auto_search_dir,
        stage="selecting_best",
        trials_total=len(trial_grid),
        trials_done=trials_done,
        trials_failed=trials_failed,
        best_score=best_score,
        last_error=last_error,
        phase="selecting_best",
        message="Selecting best configuration",
    )

    selected_artifacts = _materialize_best_candidate(
        chosen,
        progress_stage="refit_final",
        progress_message="Refitting final model (includes bootstrap if enabled)",
    )

    summary = {
        "run_name": config.get("run_name"),
        "status": "selected",
        "selection_rule": selection_rule,
        "epsilon": epsilon,
        "score_definition": None if outer_folds > 0 else "pooled_val_delta_logloss",
        "best_score": best.get("score"),
        "chosen": chosen,
        "acceptance": {
            "accept_delta_threshold": float(accept_delta_threshold),
            "min_improve_fraction": float(min_improve_fraction),
            "max_worst_delta": float(max_worst_delta),
            "outer_min_improve_fraction": float(outer_min_improve_fraction),
            "outer_max_worst_delta": float(outer_max_worst_delta),
            "accepted_candidates": int(len(filtered)),
            "rejected_candidates": int(len(rejected_rows)),
        },
        "outer_cv": {
            "enabled": bool(outer_folds > 0),
            "outer_folds": int(outer_folds),
            "outer_test_weeks": int(outer_test_weeks),
            "outer_gap_weeks": int(outer_gap_weeks),
            "outer_selection_metric": outer_selection_metric,
            "outer_min_improve_fraction": float(outer_min_improve_fraction),
            "outer_max_worst_delta": float(outer_max_worst_delta),
        },
        "search_space": {
            "feature_sets": feature_sets,
            "c_values": c_values,
            "calibration_methods": calibration_methods,
            "trading_universe_upweight": trading_universe_upweights,
            "foundation_weight": foundation_weights,
            "ticker_intercepts": ticker_intercepts,
            "advanced_interactions": advanced_interactions,
        },
        "selected_model_artifacts": selected_artifacts,
    }
    _write_json(auto_search_dir / "auto_search_summary.json", summary)
    _write_run_manifest(
        out_dir,
        auto_status="selected",
        selected_trial_id=int(chosen.get("trial_id")) if chosen.get("trial_id") is not None else None,
        selected_model_relpath="selected_model",
        auto_search_relpath="auto_search",
    )
    if outer_cv_summary is not None:
        outer_cv_summary["selection"] = {
            "status": "selected",
            "selection_rule": selection_rule,
            "epsilon": epsilon,
            "chosen_trial_id": chosen.get("trial_id"),
            "best_score": best.get("score"),
            "selection_metric": outer_selection_metric,
        }
        _write_json(auto_search_dir / "outer_cv_summary.json", outer_cv_summary)

    write_progress(
        auto_search_dir,
        stage="done",
        trials_total=len(trial_grid),
        trials_done=trials_done,
        trials_failed=trials_failed,
        best_score=best_score,
        last_error=last_error,
        phase="done",
        message="Auto search complete",
    )


if __name__ == "__main__":
    main()
