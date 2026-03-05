from __future__ import annotations

import csv
import json
import math
import os
import queue
import logging
import shutil
import subprocess
import sys
import time
import threading
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd

from app.models.calibrate_models import (
    AutoModelRunRequest,
    CalibrateModelRunRequest,
    CalibrateModelRunResponse,
    CalibrationJobStatus,
    DatasetFileSummary,
    DatasetFeaturesResponse,
    DatasetListResponse,
    DatasetTickersResponse,
    EdgePrediction,
    EdgePredictionsResponse,
    FeatureStat,
    ModelFileContentResponse,
    ModelFilesListResponse,
    ModelFileSummary,
    RegimeInfo,
    RegimePreviewRequest,
    RegimePreviewResponse,
    WeightingPreviewRequest,
    WeightingPreviewResponse,
    ModelDetailResponse,
    ModelListResponse,
    ModelRunSummary,
    SplitMetricSummary,
    TrainModelV2Request,
    ProgressPayload,
)
from app.services.process_runtime import (
    ManagedProcessHandle,
    clear_runtime_file,
    is_process_alive,
    managed_handle_from_runtime_payload,
    read_runtime_file,
    spawn_managed_process,
    terminate_managed_process,
)


def _unique_dirs(paths: List[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path.absolute())
    return unique


BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "03-calibrate-logit-model-v2.0.py"
AUTO_SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "03-auto-calibrate-logit-model-v2.0.py"
CALIBRATE_DATASET_DIRS = _unique_dirs(
    [
        BASE_DIR / "src" / "data" / "raw" / "option-chain-v3",
        BASE_DIR / "data" / "raw" / "option-chain-v3",
        BASE_DIR / "src" / "data" / "raw" / "option-chains",
        BASE_DIR / "data" / "raw" / "option-chains",
        BASE_DIR / "src" / "data" / "raw" / "option-chain",
        BASE_DIR / "data" / "raw" / "option-chain",
    ],
)
POLYMARKET_DATASET_DIRS = _unique_dirs(
    [
        BASE_DIR / "src" / "data" / "raw" / "polymarket" / "weekly_history" / "runs",
        BASE_DIR / "data" / "raw" / "polymarket" / "weekly_history" / "runs",
        BASE_DIR / "src" / "data" / "models" / "polymarket",
        BASE_DIR / "data" / "models" / "polymarket",
    ],
)
MODELS_DIR = BASE_DIR / "src" / "data" / "models"
AUTO_RUN_EXCLUSIVE_LOCK = threading.Lock()
LOGGER = logging.getLogger(__name__)
DELETE_TERM_TIMEOUT_S = 5.0
DELETE_KILL_TIMEOUT_S = 5.0
DELETE_QUIET_WINDOW_S = 1.0

# Import CLI contract constants for validation
import sys
if str(BASE_DIR / "src" / "webapp") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src" / "webapp"))
from shared.cli_contract_v2 import (
    validate_payload,
    ValidationError,
)


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.absolute()
    try:
        path.relative_to(BASE_DIR)
    except ValueError as exc:
        raise ValueError("Path must be inside the project root.") from exc
    return path


def _is_valid_dataset(path: Path) -> bool:
    """Only accept training CSVs (new and legacy conventions)."""
    if not path.is_file():
        return False
    if path.suffix.lower() != ".csv":
        return False
    lowered = path.name.lower()
    # New convention: training-*.csv
    if lowered.startswith("training-"):
        return True
    # Legacy: train_view.csv
    if lowered == "train_view.csv":
        return True
    # Legacy: {parent_dir_name}.csv (renamed training file)
    if path.parent.is_dir() and lowered == f"{path.parent.name.lower()}.csv":
        return True
    return False


def _is_valid_polymarket_dataset(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() not in {".csv", ".parquet"}:
        return False
    lowered = path.name.lower()
    return lowered.startswith("decision_features")


def _find_calibrate_dataset_base(path: Path) -> Path:
    for base in CALIBRATE_DATASET_DIRS:
        try:
            path.relative_to(base)
            return base
        except ValueError:
            continue
    raise ValueError(
        "Dataset must be under src/data/raw/option-chain-v3, src/data/raw/option-chains, "
        "or src/data/raw/option-chain (or data/raw equivalents)."
    )


def _find_polymarket_dataset_base(path: Path) -> Path:
    for base in POLYMARKET_DATASET_DIRS:
        try:
            path.relative_to(base)
            return base
        except ValueError:
            continue
    raise ValueError(
        "Polymarket dataset must be under src/data/models/polymarket, "
        "src/data/raw/polymarket, or their data equivalents."
    )


def _select_dataset_dirs() -> List[Path]:
    existing_dirs = [path for path in CALIBRATE_DATASET_DIRS if path.exists()]
    return existing_dirs or [CALIBRATE_DATASET_DIRS[0]]


DATASET_DATE_COL_CANDIDATES = [
    "week_friday",
    "asof_date",
    "snapshot_date",
    "snapshot_time_utc",
    "asof_datetime_utc",
]
DATASET_WEIGHT_COL_CANDIDATES = [
    "weight_final",
    "sample_weight_final",
    "weight_raw",
    "weight_group_w",
]
DATASET_GROUP_KEY_CANDIDATES = [
    "group_id",
    "weight_group_key",
    "contract_id",
    "cluster_snapshot",
    "cluster_strike",
    "cluster_week",
]


def _summarize_dataset(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "dataset_id": path.parent.name or path.stem,
        "rows": None,
        "date_col_used": None,
        "date_start": None,
        "date_end": None,
        "ticker_count": None,
        "ticker_sample": None,
        "available_weight_columns": [],
        "available_grouping_keys": [],
    }
    try:
        frame = pd.read_csv(path)
    except Exception:
        return summary

    if frame.empty:
        summary["rows"] = 0
        return summary

    summary["rows"] = int(len(frame))
    summary["available_weight_columns"] = [
        col for col in DATASET_WEIGHT_COL_CANDIDATES if col in frame.columns
    ]
    summary["available_grouping_keys"] = [
        col for col in DATASET_GROUP_KEY_CANDIDATES if col in frame.columns
    ]

    for date_col in DATASET_DATE_COL_CANDIDATES:
        if date_col not in frame.columns:
            continue
        parsed = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
        if parsed.notna().any():
            summary["date_col_used"] = date_col
            summary["date_start"] = parsed.min().date().isoformat()
            summary["date_end"] = parsed.max().date().isoformat()
            try:
                summary["week_count"] = int(parsed.dt.to_period("W-FRI").nunique())
            except Exception:
                summary["week_count"] = None
            break

    if "ticker" in frame.columns:
        tickers = (
            frame["ticker"]
            .astype("string")
            .str.upper()
            .str.strip()
            .dropna()
            .loc[lambda s: s != ""]
            .unique()
            .tolist()
        )
        tickers = sorted(str(t) for t in tickers)
        summary["ticker_count"] = int(len(tickers))
        summary["ticker_sample"] = tickers[:8]

    return summary


def list_datasets() -> DatasetListResponse:
    dataset_dirs = _select_dataset_dirs()
    datasets: List[DatasetFileSummary] = []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        for item in dataset_dir.rglob("*.csv"):
            if not _is_valid_dataset(item):
                continue
            mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()
            metadata = _summarize_dataset(item)
            datasets.append(
                DatasetFileSummary(
                    name=item.name,
                    path=str(item.relative_to(BASE_DIR)),
                    size_bytes=item.stat().st_size,
                    last_modified=mtime,
                    dataset_id=metadata.get("dataset_id"),
                    rows=metadata.get("rows"),
                    date_col_used=metadata.get("date_col_used"),
                    date_start=metadata.get("date_start"),
                    date_end=metadata.get("date_end"),
                    ticker_count=metadata.get("ticker_count"),
                    ticker_sample=metadata.get("ticker_sample"),
                    available_weight_columns=metadata.get("available_weight_columns"),
                    available_grouping_keys=metadata.get("available_grouping_keys"),
                )
            )
    datasets.sort(key=lambda d: d.last_modified or "", reverse=True)
    return DatasetListResponse(
        base_dir=str(dataset_dirs[0].relative_to(BASE_DIR)),
        datasets=datasets,
    )


def list_polymarket_datasets() -> DatasetListResponse:
    dataset_dirs = [path for path in POLYMARKET_DATASET_DIRS if path.exists()]
    if not dataset_dirs:
        dataset_dirs = [POLYMARKET_DATASET_DIRS[0]]

    datasets: List[DatasetFileSummary] = []
    for dataset_dir in dataset_dirs:
        if not dataset_dir.exists():
            continue
        for item in dataset_dir.rglob("*"):
            if not _is_valid_polymarket_dataset(item):
                continue
            mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()
            datasets.append(
                DatasetFileSummary(
                    name=item.name,
                    path=str(item.relative_to(BASE_DIR)),
                    size_bytes=item.stat().st_size,
                    last_modified=mtime,
                )
            )

    datasets.sort(key=lambda d: d.last_modified or "", reverse=True)
    return DatasetListResponse(
        base_dir=str(dataset_dirs[0].relative_to(BASE_DIR)),
        datasets=datasets,
    )


def get_dataset_tickers(dataset_path: str) -> DatasetTickersResponse:
    """
    Extract unique tickers from a calibration dataset.

    Args:
        dataset_path: Relative path to dataset (e.g., "src/data/raw/option-chain/dataset.csv")

    Returns:
        DatasetTickersResponse with sorted list of unique tickers

    Raises:
        FileNotFoundError: If dataset doesn't exist
        ValueError: If dataset has no 'ticker' column or is empty
    """
    path = _resolve_project_path(dataset_path)
    _ensure_dataset_path(path)

    try:
        # Read only ticker column for efficiency
        if path.suffix.lower() == '.parquet':
            df = pd.read_parquet(path, columns=['ticker'])
        else:
            # Try to read just ticker column; fallback to full read if column spec fails
            try:
                df = pd.read_csv(path, usecols=['ticker'], dtype={'ticker': str})
            except ValueError:
                # Column not found or other error; read full file and check
                df = pd.read_csv(path, dtype={'ticker': str})

        if 'ticker' not in df.columns:
            raise ValueError(f"Dataset {path.name} has no 'ticker' column")

        # Extract unique tickers, drop NaN, convert to sorted list
        tickers = df['ticker'].dropna().astype(str).str.strip().str.upper().unique().tolist()
        tickers.sort()

        if not tickers:
            raise ValueError(f"Dataset {path.name} has no valid ticker values")

        return DatasetTickersResponse(
            dataset=dataset_path,
            tickers=tickers,
            count=len(tickers),
        )

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to read tickers from {path.name}: {str(e)}")


def get_dataset_features(dataset_path: str) -> "DatasetFeaturesResponse":
    """
    Inspect dataset columns and return available features with statistics.

    Args:
        dataset_path: Relative path to dataset

    Returns:
        DatasetFeaturesResponse with columns, stats, and regime info

    Raises:
        FileNotFoundError: If dataset doesn't exist
        ValueError: If dataset cannot be read
    """
    path = _resolve_project_path(dataset_path)
    _ensure_dataset_path(path)

    try:
        # Read dataset
        if path.suffix.lower() == '.parquet':
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"Dataset {path.name} is empty")

        # Get column names
        available_columns = df.columns.tolist()

        # Also include engineered features that can be derived from raw columns
        # (e.g. x_logit_prn from pRN, log_m_fwd from K+forward_price, etc.)
        raw_cols = set(available_columns)
        derived: list = []
        if "pRN" in raw_cols and "x_logit_prn" not in raw_cols:
            derived.append("x_logit_prn")
        has_forward = "forward_price" in raw_cols or (
            "K" in raw_cols and "S_asof_close" in raw_cols
            and "r" in raw_cols and "dividend_yield" in raw_cols
        )
        if has_forward:
            if "log_m_fwd" not in raw_cols:
                derived.append("log_m_fwd")
            if "abs_log_m_fwd" not in raw_cols:
                derived.append("abs_log_m_fwd")
        elif "log_m_fwd" in raw_cols and "abs_log_m_fwd" not in raw_cols:
            derived.append("abs_log_m_fwd")
        if "T_days" in raw_cols and "sqrt_T_years" not in raw_cols:
            derived.append("sqrt_T_years")
        if "rv20" in raw_cols and ("T_days" in raw_cols or "sqrt_T_years" in raw_cols):
            if "rv20_sqrtT" not in raw_cols:
                derived.append("rv20_sqrtT")
        has_log_m_fwd = "log_m_fwd" in raw_cols or has_forward
        if has_log_m_fwd and "rv20" in raw_cols and "T_days" in raw_cols:
            if "log_m_fwd_over_volT" not in raw_cols:
                derived.append("log_m_fwd_over_volT")
        if "rel_spread_median" in raw_cols and "log_rel_spread" not in raw_cols:
            derived.append("log_rel_spread")
        if (
            "asof_fallback_days" in raw_cols
            and "expiry_fallback_days" in raw_cols
            and "had_fallback" not in raw_cols
        ):
            derived.append("had_fallback")
        if (
            (
                "dropped_intrinsic" in raw_cols
                and "n_chain_raw" in raw_cols
            )
            or "drop_intrinsic_frac" in raw_cols
        ):
            if "had_intrinsic_drop" not in raw_cols:
                derived.append("had_intrinsic_drop")
        if (
            (
                "n_band_inside" in raw_cols
                and "n_band_raw" in raw_cols
            )
            or "band_inside_frac" in raw_cols
        ):
            if "had_band_clip" not in raw_cols:
                derived.append("had_band_clip")
        if "pRN" in raw_cols and "pRN_raw" in raw_cols and "prn_raw_gap" not in raw_cols:
            derived.append("prn_raw_gap")
        has_x_logit = "x_logit_prn" in raw_cols or "pRN" in raw_cols
        if has_x_logit and has_log_m_fwd:
            if "x_m" not in raw_cols:
                derived.append("x_m")
            if "x_abs_m" not in raw_cols:
                derived.append("x_abs_m")
        available_columns = available_columns + derived

        # Compute feature statistics
        feature_stats: Dict[str, "FeatureStat"] = {}
        for col in df.columns:
            missing_pct = round(df[col].isna().mean() * 100, 2)
            dtype = str(df[col].dtype)
            nunique = int(df[col].nunique())

            feature_stats[col] = {
                "missing_pct": missing_pct,
                "dtype": dtype,
                "nunique": nunique,
            }

        # Detect regime
        tdays_mode = None
        is_weekly = None
        is_daily = None

        if "T_days" in df.columns:
            tdays_mode, is_weekly, is_daily = _classify_tdays_regime(df["T_days"])

        regime_info = {
            "tdays_mode": tdays_mode,
            "is_weekly": is_weekly,
            "is_daily": is_daily,
        }

        return DatasetFeaturesResponse(
            dataset=dataset_path,
            available_columns=available_columns,
            feature_stats=feature_stats,
            regime_info=regime_info,
        )

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to read features from {path.name}: {str(e)}")


RUN_MANIFEST_FILENAME = "run_manifest.json"
AUTO_SEARCH_DIRNAME = "auto_search"
SELECTED_MODEL_DIRNAME = "selected_model"
AUTO_SEARCH_MARKERS = [
    "auto_search_summary.json",
    "auto_search_leaderboard.csv",
    "auto_search_no_viable.json",
]


def _load_run_manifest(run_dir: Path) -> Optional[Dict[str, Any]]:
    manifest = _load_json(run_dir / RUN_MANIFEST_FILENAME)
    return manifest if isinstance(manifest, dict) else None


def _resolve_auto_search_dir(run_dir: Path) -> Path:
    modern = run_dir / AUTO_SEARCH_DIRNAME
    if modern.exists() and modern.is_dir():
        return modern
    return run_dir


def _resolve_selected_model_dir(run_dir: Path) -> Optional[Path]:
    manifest = _load_run_manifest(run_dir)
    if manifest:
        relpath = manifest.get("selected_model_relpath")
        if isinstance(relpath, str) and relpath.strip():
            candidate = (run_dir / relpath).resolve()
            try:
                candidate.relative_to(run_dir.resolve())
            except Exception:
                candidate = None
            if candidate is not None and candidate.exists() and candidate.is_dir():
                return candidate
    modern = run_dir / SELECTED_MODEL_DIRNAME
    if modern.exists() and modern.is_dir():
        return modern
    return None


def _is_auto_run_dir(run_dir: Path) -> bool:
    manifest = _load_run_manifest(run_dir)
    if manifest and str(manifest.get("run_type") or "").strip().lower() == "auto":
        return True
    if (run_dir / AUTO_SEARCH_DIRNAME).exists():
        return True
    for marker in AUTO_SEARCH_MARKERS:
        if (run_dir / marker).exists():
            return True
    for marker in AUTO_SEARCH_MARKERS:
        if (run_dir / AUTO_SEARCH_DIRNAME / marker).exists():
            return True
    return False


def _resolve_effective_model_dir(run_dir: Path) -> Path:
    if not _is_auto_run_dir(run_dir):
        return run_dir
    selected_dir = _resolve_selected_model_dir(run_dir)
    if selected_dir is not None:
        return selected_dir
    return run_dir


def _resolve_auto_status(run_dir: Path) -> Optional[str]:
    manifest = _load_run_manifest(run_dir)
    if manifest:
        status = manifest.get("auto_status")
        if isinstance(status, str) and status.strip():
            return status.strip()
    search_summary = _load_json(_resolve_auto_search_dir(run_dir) / "auto_search_summary.json")
    if isinstance(search_summary, dict):
        status = search_summary.get("status")
        if isinstance(status, str) and status.strip():
            return status.strip()
        if search_summary.get("chosen"):
            return "selected"
    return None


def _resolve_selected_trial_id(run_dir: Path) -> Optional[int]:
    manifest = _load_run_manifest(run_dir)
    if manifest is not None and manifest.get("selected_trial_id") is not None:
        try:
            return int(manifest.get("selected_trial_id"))
        except Exception:
            return None
    search_summary = _load_json(_resolve_auto_search_dir(run_dir) / "auto_search_summary.json")
    if isinstance(search_summary, dict):
        chosen = search_summary.get("chosen")
        if isinstance(chosen, dict):
            trial_id = chosen.get("trial_id")
            if trial_id is not None:
                try:
                    return int(trial_id)
                except Exception:
                    return None
    return None


def list_models() -> ModelListResponse:
    if not MODELS_DIR.exists():
        return ModelListResponse(base_dir=str(MODELS_DIR.relative_to(BASE_DIR)), models=[])

    models: List[ModelRunSummary] = []
    for item in MODELS_DIR.iterdir():
        if not item.is_dir():
            continue
        run_type = "auto" if _is_auto_run_dir(item) else "manual"
        effective_dir = _resolve_effective_model_dir(item)
        metadata_path = effective_dir / "metadata.json"
        metrics_path = effective_dir / "metrics.csv"
        mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()
        metadata = _load_json(metadata_path)
        split_ranges = metadata.get("splits", {}) if isinstance(metadata, dict) else {}
        fit_weights = metadata.get("fit_weights", {}) if isinstance(metadata, dict) else {}
        filters = metadata.get("filters", {}) if isinstance(metadata, dict) else {}
        dataset_path = metadata.get("csv") if isinstance(metadata, dict) else None
        dataset_id = None
        if isinstance(dataset_path, str):
            dataset_id = Path(dataset_path).parent.name or Path(dataset_path).stem
        tickers = filters.get("train_tickers") if isinstance(filters, dict) else None
        if isinstance(tickers, list) and tickers:
            tickers_summary = ", ".join(str(t) for t in tickers[:8])
            if len(tickers) > 8:
                tickers_summary = f"{tickers_summary}, +{len(tickers) - 8}"
        else:
            tickers_summary = None
        tdays = metadata.get("tdays_allowed") if isinstance(metadata, dict) else None
        dow = metadata.get("asof_dow_allowed") if isinstance(metadata, dict) else None
        if tdays or dow:
            dow_regime = f"T_days={tdays or 'all'} DOW={dow or 'all'}"
        else:
            dow_regime = None
        split_strategy = metadata.get("val_split_mode_used") if isinstance(metadata, dict) else None
        c_value = metadata.get("best_C") if isinstance(metadata, dict) else None
        calibration_method = metadata.get("calibration_used") if isinstance(metadata, dict) else None
        weighting_mode = fit_weights.get("group_reweight") if isinstance(fit_weights, dict) else None
        train_range = split_ranges.get("train_rows_range") if isinstance(split_ranges, dict) else None
        train_date_start = (
            train_range[0] if isinstance(train_range, (list, tuple)) and len(train_range) > 0 else None
        )
        train_date_end = (
            train_range[1] if isinstance(train_range, (list, tuple)) and len(train_range) > 1 else None
        )
        selected_model_dir = _resolve_selected_model_dir(item)
        effective_has_metrics = (effective_dir / "metrics.csv").exists()
        models.append(
            ModelRunSummary(
                id=item.name,
                path=str(item.relative_to(BASE_DIR)),
                last_modified=mtime,
                has_metadata=metadata_path.exists(),
                has_metrics=metrics_path.exists(),
                dataset_id=dataset_id,
                dataset_path=dataset_path if isinstance(dataset_path, str) else None,
                train_date_start=train_date_start,
                train_date_end=train_date_end,
                tickers_summary=tickers_summary,
                dow_regime=dow_regime,
                split_strategy=str(split_strategy) if split_strategy else None,
                c_value=float(c_value) if isinstance(c_value, (int, float)) else None,
                calibration_method=str(calibration_method) if calibration_method else None,
                weighting_mode=str(weighting_mode) if weighting_mode else None,
                is_two_stage=bool(metadata.get("two_stage_mode")) if isinstance(metadata, dict) else False,
                run_type=run_type,
                auto_status=_resolve_auto_status(item) if run_type == "auto" else None,
                selected_trial_id=_resolve_selected_trial_id(item) if run_type == "auto" else None,
                has_selected_model=(
                    bool(selected_model_dir is not None and (selected_model_dir / "metrics.csv").exists())
                    or (run_type == "auto" and effective_has_metrics)
                ) if run_type == "auto" else None,
            )
        )
    models.sort(key=lambda m: m.last_modified or "", reverse=True)
    return ModelListResponse(
        base_dir=str(MODELS_DIR.relative_to(BASE_DIR)),
        models=models,
    )


def _model_summary_from_path(item: Path) -> ModelRunSummary:
    run_type = "auto" if _is_auto_run_dir(item) else "manual"
    effective_dir = _resolve_effective_model_dir(item)
    metadata_path = effective_dir / "metadata.json"
    metrics_path = effective_dir / "metrics.csv"
    mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()
    metadata = _load_json(metadata_path)
    dataset_path = metadata.get("csv") if isinstance(metadata, dict) else None
    dataset_id = (
        Path(dataset_path).parent.name or Path(dataset_path).stem
        if isinstance(dataset_path, str)
        else None
    )
    selected_model_dir = _resolve_selected_model_dir(item)
    effective_has_metrics = (effective_dir / "metrics.csv").exists()
    return ModelRunSummary(
        id=item.name,
        path=str(item.relative_to(BASE_DIR)),
        last_modified=mtime,
        has_metadata=metadata_path.exists(),
        has_metrics=metrics_path.exists(),
        dataset_id=dataset_id,
        dataset_path=dataset_path if isinstance(dataset_path, str) else None,
        is_two_stage=bool(metadata.get("two_stage_mode")) if isinstance(metadata, dict) else False,
        run_type=run_type,
        auto_status=_resolve_auto_status(item) if run_type == "auto" else None,
        selected_trial_id=_resolve_selected_trial_id(item) if run_type == "auto" else None,
        has_selected_model=(
            bool(selected_model_dir is not None and (selected_model_dir / "metrics.csv").exists())
            or (run_type == "auto" and effective_has_metrics)
        ) if run_type == "auto" else None,
    )


def _wait_for_jobs_to_stop(job_ids: List[str], timeout_s: float = 8.0) -> bool:
    if not job_ids:
        return True
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    pending = set(job_ids)
    while time.monotonic() < deadline and pending:
        statuses = {status.job_id: status.status for status in CALIBRATION_JOB_MANAGER.list_jobs()}
        pending = {job_id for job_id in pending if statuses.get(job_id) in {"queued", "running"}}
        if pending:
            time.sleep(0.1)
    return not pending


def _kill_runtime_for_model_dir(
    run_dir: Path,
    *,
    deletion_attempt: int,
) -> Tuple[bool, str]:
    runtime_payload = read_runtime_file(run_dir)
    if not runtime_payload:
        LOGGER.info(
            "runtime_handle_found service=calibrate path=%s found=false deletion_attempt=%s",
            run_dir,
            deletion_attempt,
        )
        return True, "no_runtime_handle"

    handle = managed_handle_from_runtime_payload(run_dir, runtime_payload)
    if handle is None:
        LOGGER.warning(
            "runtime_handle_found service=calibrate path=%s found=true valid=false deletion_attempt=%s",
            run_dir,
            deletion_attempt,
        )
        clear_runtime_file(run_dir)
        return True, "runtime_handle_invalid"

    LOGGER.info(
        "runtime_handle_found service=calibrate path=%s found=true pid=%s pgid=%s deletion_attempt=%s",
        run_dir,
        handle.pid,
        handle.pgid,
        deletion_attempt,
    )
    result = terminate_managed_process(
        handle,
        term_timeout_s=DELETE_TERM_TIMEOUT_S,
        kill_timeout_s=DELETE_KILL_TIMEOUT_S,
    )
    if result.term_sent:
        LOGGER.info(
            "group_term_sent service=calibrate path=%s pid=%s pgid=%s deletion_attempt=%s",
            run_dir,
            result.pid,
            result.pgid,
            deletion_attempt,
        )
    if result.kill_sent:
        LOGGER.info(
            "group_kill_sent service=calibrate path=%s pid=%s pgid=%s deletion_attempt=%s",
            run_dir,
            result.pid,
            result.pgid,
            deletion_attempt,
        )
    if result.ok:
        clear_runtime_file(run_dir)
        return True, result.reason

    still_alive = is_process_alive(result.pid)
    return False, f"{result.reason};alive={still_alive}"


def _delete_model_dir_with_quiescence(
    target: Path,
    *,
    retries: int = 1,
) -> None:
    for attempt in range(1, retries + 2):
        if target.exists():
            try:
                shutil.rmtree(target)
            except FileNotFoundError:
                pass
            except Exception as exc:
                raise RuntimeError(f"Failed to delete model run '{target.name}': {exc}") from exc

        if target.exists():
            continue

        time.sleep(DELETE_QUIET_WINDOW_S)
        if not target.exists():
            LOGGER.info(
                "delete_completed service=calibrate path=%s deletion_attempt=%s",
                target,
                attempt,
            )
            return

        LOGGER.warning(
            "delete_refill_detected service=calibrate path=%s deletion_attempt=%s",
            target,
            attempt,
        )
        kill_ok, kill_reason = _kill_runtime_for_model_dir(target, deletion_attempt=attempt)
        if not kill_ok:
            raise RuntimeError(
                f"process_still_alive for model run '{target.name}' after delete attempt "
                f"{attempt}: {kill_reason}"
            )
    raise RuntimeError(f"directory_refilled for model run '{target.name}' after delete.")


def delete_model(model_id: str) -> ModelRunSummary:
    target = (MODELS_DIR / model_id).resolve()
    try:
        target.relative_to(MODELS_DIR.resolve())
    except Exception:
        raise KeyError(model_id)
    if not target.exists() or not target.is_dir():
        raise KeyError(model_id)
    summary = _model_summary_from_path(target)
    LOGGER.info("delete_requested service=calibrate model_id=%s path=%s", model_id, target)
    cancelled_job_ids = CALIBRATION_JOB_MANAGER.cancel_jobs_for_model(model_id)
    if cancelled_job_ids and not _wait_for_jobs_to_stop(cancelled_job_ids):
        raise RuntimeError("kill_timeout: run cancellation is still in progress.")
    kill_ok, kill_reason = _kill_runtime_for_model_dir(target, deletion_attempt=0)
    if not kill_ok:
        raise RuntimeError(f"process_still_alive: {kill_reason}")
    if not target.exists():
        return summary
    _delete_model_dir_with_quiescence(target, retries=1)
    return summary


def rename_model(model_id: str, new_name: str) -> ModelRunSummary:
    old_path = (MODELS_DIR / model_id).resolve()
    try:
        old_path.relative_to(MODELS_DIR.resolve())
    except Exception:
        raise KeyError(model_id)
    if not old_path.exists() or not old_path.is_dir():
        raise KeyError(model_id)

    sanitized_name = _sanitize_name(new_name)
    if not sanitized_name:
        raise ValueError("New name cannot be empty after sanitization.")

    new_path = (MODELS_DIR / sanitized_name).resolve()
    try:
        new_path.relative_to(MODELS_DIR.resolve())
    except Exception:
        raise ValueError("Invalid new name path.")

    if new_path.exists():
        raise ValueError(f"Model '{sanitized_name}' already exists.")

    old_path.rename(new_path)
    return _model_summary_from_path(new_path)


# Files that are important/viewable in the model directory
VIEWABLE_EXTENSIONS = {".json", ".csv", ".md", ".txt"}
MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB max for viewing
MAX_AUTO_TRIAL_FILES = 100
SELECTED_MODEL_IMPORTANT_FILES = [
    "config.executed.json",
    "metadata.json",
    "metrics.csv",
    "metrics_summary.json",
    "split_timeline.json",
    "fold_deltas.csv",
    "group_delta_distribution.csv",
    "audit_split_composition.csv",
    "audit_overlap.json",
    "audit_weight_distribution.json",
    "feature_manifest.json",
    "reliability_bins.csv",
    "rolling_summary.csv",
    "rolling_windows.csv",
    "metrics_groups.csv",
    "two_stage_metrics.csv",
    "two_stage_metrics_summary.json",
    "two_stage_metadata.json",
]
AUTO_SEARCH_IMPORTANT_FILES = [
    "auto_search_leaderboard.csv",
    "auto_search_summary.json",
    "auto_search_no_viable.json",
    "auto_search_progress.json",
    "progress.json",
    "outer_folds.json",
    "outer_cv_summary.json",
    "best_config.json",
]
AUTO_TRIAL_IMPORTANT_FILES = [
    "trial_result.json",
    "outer_fold_results.csv",
]


def _resolve_model_run_dir(model_id: str) -> Path:
    target = (MODELS_DIR / model_id).resolve()
    try:
        target.relative_to(MODELS_DIR.resolve())
    except Exception:
        raise KeyError(model_id)
    if not target.exists() or not target.is_dir():
        raise KeyError(model_id)
    return target


def _collect_curated_file_entries(run_dir: Path) -> List[Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}

    def _add_entry(path: Path, section: str) -> None:
        if not path.exists() or not path.is_file():
            return
        rel = str(path.relative_to(run_dir))
        entries[rel] = {"path": path, "relative_path": rel, "section": section}

    is_auto = _is_auto_run_dir(run_dir)
    if not is_auto:
        for name in SELECTED_MODEL_IMPORTANT_FILES:
            _add_entry(run_dir / name, "legacy_root")
        return list(entries.values())

    selected_dir = _resolve_selected_model_dir(run_dir)
    if selected_dir is not None:
        for name in SELECTED_MODEL_IMPORTANT_FILES:
            _add_entry(selected_dir / name, "selected_model")
    else:
        # Legacy flat auto runs keep selected-model artifacts in run root.
        for name in SELECTED_MODEL_IMPORTANT_FILES:
            _add_entry(run_dir / name, "selected_model")

    _add_entry(run_dir / RUN_MANIFEST_FILENAME, "auto_search")

    auto_search_dir = _resolve_auto_search_dir(run_dir)
    auto_section = "auto_search"
    for name in AUTO_SEARCH_IMPORTANT_FILES:
        _add_entry(auto_search_dir / name, auto_section)

    return list(entries.values())


def list_model_files(model_id: str) -> ModelFilesListResponse:
    """List files in a model directory that can be viewed."""
    target = _resolve_model_run_dir(model_id)
    section_order = {"selected_model": 0, "auto_search": 1, "legacy_root": 2}
    files: List[ModelFileSummary] = []
    for entry in sorted(
        _collect_curated_file_entries(target),
        key=lambda item: (
            section_order.get(str(item.get("section")), 9),
            str(item.get("relative_path") or ""),
        ),
    ):
        path = entry["path"]
        size = path.stat().st_size
        files.append(
            ModelFileSummary(
                name=path.name,
                size_bytes=size,
                is_viewable=size <= MAX_FILE_SIZE_BYTES,
                relative_path=str(entry.get("relative_path")),
                section=str(entry.get("section") or "legacy_root"),
                kind="file",
            )
        )
    return ModelFilesListResponse(model_id=model_id, files=files)


def _resolve_curated_file_path(run_dir: Path, requested_path: str) -> Tuple[Path, str, str]:
    requested = str(requested_path or "").strip()
    if not requested:
        raise ValueError("File path is required.")
    entries = _collect_curated_file_entries(run_dir)
    by_rel = {str(entry.get("relative_path")): entry for entry in entries}
    if requested in by_rel:
        entry = by_rel[requested]
        return entry["path"], str(entry.get("relative_path")), str(entry.get("section"))
    if "/" not in requested and "\\" not in requested:
        matches = [entry for entry in entries if Path(str(entry.get("relative_path"))).name == requested]
        if len(matches) == 1:
            entry = matches[0]
            return entry["path"], str(entry.get("relative_path")), str(entry.get("section"))
        if len(matches) > 1:
            raise ValueError(f"File name '{requested}' is ambiguous. Use relative path instead.")
    raise KeyError(requested)


def _read_model_file_content(*, run_dir: Path, model_id: str, requested_path: str) -> ModelFileContentResponse:
    file_path, relative_path, _section = _resolve_curated_file_path(run_dir, requested_path)
    size = file_path.stat().st_size
    truncated = size > MAX_FILE_SIZE_BYTES
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        content_type = "json"
    elif suffix == ".csv":
        content_type = "csv"
    elif suffix == ".md":
        content_type = "markdown"
    else:
        content_type = "text"

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read(MAX_FILE_SIZE_BYTES) if truncated else f.read()

    return ModelFileContentResponse(
        model_id=model_id,
        filename=file_path.name,
        relative_path=relative_path,
        content=content,
        content_type=content_type,
        truncated=truncated,
    )


def get_model_file_content(model_id: str, filename: str) -> ModelFileContentResponse:
    """Get file content by legacy filename (or relative path when unique)."""
    run_dir = _resolve_model_run_dir(model_id)
    return _read_model_file_content(run_dir=run_dir, model_id=model_id, requested_path=filename)


def get_model_file_content_by_path(model_id: str, relative_path: str) -> ModelFileContentResponse:
    """Get file content using explicit relative path from the model run root."""
    run_dir = _resolve_model_run_dir(model_id)
    return _read_model_file_content(run_dir=run_dir, model_id=model_id, requested_path=relative_path)


def _add_value(cmd: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    cmd.extend([flag, text])


def _add_flag(cmd: List[str], flag: str, enabled: Optional[bool]) -> None:
    if enabled:
        cmd.append(flag)


def _build_python_env() -> Dict[str, str]:
    env = os.environ.copy()
    base_path = str(BASE_DIR)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = base_path if not existing else os.pathsep.join([base_path, existing])
    return env


def _try_acquire_auto_run_lock() -> bool:
    return AUTO_RUN_EXCLUSIVE_LOCK.acquire(blocking=False)


def _release_auto_run_lock() -> None:
    if not AUTO_RUN_EXCLUSIVE_LOCK.locked():
        return
    try:
        AUTO_RUN_EXCLUSIVE_LOCK.release()
    except RuntimeError:
        pass


def _read_progress(out_dir: Path) -> Optional[ProgressPayload]:
    candidates = [out_dir / "progress.json", out_dir / AUTO_SEARCH_DIRNAME / "progress.json"]
    for progress_path in candidates:
        if not progress_path.exists():
            continue
        try:
            payload = json.loads(progress_path.read_text())
            return ProgressPayload(**payload)
        except Exception:
            continue
    return None


def _iter_calibration_runtime_dirs() -> List[Tuple[Path, Dict[str, Any]]]:
    if not MODELS_DIR.exists():
        return []
    runtime_dirs: List[Tuple[Path, Dict[str, Any]]] = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        payload = read_runtime_file(entry)
        if not payload:
            continue
        runtime_dirs.append((entry, payload))
    return runtime_dirs


def _find_runtime_job(job_id: str) -> Optional[Tuple[Path, Dict[str, Any]]]:
    for run_dir, payload in _iter_calibration_runtime_dirs():
        runtime_job_id = payload.get("job_id")
        if runtime_job_id is None:
            continue
        if str(runtime_job_id) == job_id:
            return run_dir, payload
    return None


def _infer_mode_from_runtime_service(service: Optional[str]) -> str:
    service_name = str(service or "").lower()
    if "auto" in service_name:
        return "auto"
    return "manual"


def _parse_runtime_started_at(payload: Dict[str, Any]) -> Optional[datetime]:
    started_at_raw = payload.get("started_at")
    if started_at_raw is None:
        return None
    try:
        return datetime.fromisoformat(str(started_at_raw))
    except Exception:
        return None


def _build_runtime_backed_job_status(
    job_id: str,
    runtime_payload: Dict[str, Any],
    run_dir: Path,
    *,
    status: str = "running",
    error: Optional[str] = None,
    finished_at: Optional[datetime] = None,
) -> CalibrationJobStatus:
    mode = _infer_mode_from_runtime_service(runtime_payload.get("service"))
    progress = _read_progress(run_dir) if mode == "auto" and status == "running" else None
    return CalibrationJobStatus(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        result=None,
        error=error,
        started_at=_parse_runtime_started_at(runtime_payload),
        finished_at=finished_at,
        progress=progress,
    )


def _runtime_backed_status(job_id: str) -> Optional[CalibrationJobStatus]:
    runtime = _find_runtime_job(job_id)
    if runtime is None:
        return None
    run_dir, payload = runtime
    pid = payload.get("pid")
    if not is_process_alive(pid):
        LOGGER.info(
            "job_lookup_runtime_stale job_id=%s path=%s pid=%s",
            job_id,
            run_dir,
            pid,
        )
        clear_runtime_file(run_dir)
        return None
    LOGGER.info(
        "job_lookup_runtime_hit job_id=%s path=%s pid=%s",
        job_id,
        run_dir,
        pid,
    )
    return _build_runtime_backed_job_status(job_id, payload, run_dir, status="running")


def _ensure_dataset_path(dataset_path: Path) -> None:
    valid_dirs = _select_dataset_dirs()
    if not valid_dirs:
        raise ValueError("No dataset directories found under src/data/raw.")
    try:
        _find_calibrate_dataset_base(dataset_path)
    except ValueError as exc:
        raise ValueError(
            "Dataset must be under src/data/raw/option-chain-v3, src/data/raw/option-chains, "
            "or src/data/raw/option-chain (or data/raw equivalents)."
        ) from exc
    if not _is_valid_dataset(dataset_path):
        raise ValueError("Dataset must be a CSV and cannot be a drops file.")


def _ensure_polymarket_dataset_path(dataset_path: Path) -> None:
    dataset_dirs = [path for path in POLYMARKET_DATASET_DIRS if path.exists()]
    if not dataset_dirs:
        raise ValueError("No Polymarket dataset directories found under src/data.")
    try:
        _find_polymarket_dataset_base(dataset_path)
    except ValueError as exc:
        raise ValueError(
            "Polymarket dataset must be under src/data/models/polymarket or src/data/raw/polymarket "
            "or their data equivalents."
        ) from exc
    if not _is_valid_polymarket_dataset(dataset_path):
        raise ValueError("Polymarket dataset must be a decision_features CSV/Parquet.")


def _default_auto_run_name() -> str:
    return f"auto-run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _ensure_auto_run_name(payload: AutoModelRunRequest) -> None:
    if payload.run_name and str(payload.run_name).strip():
        return
    payload.run_name = _default_auto_run_name()


def _parse_optional_float(val: Optional[str]) -> Optional[float]:
    if val in (None, ""):
        return None
    try:
        result = float(val)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _parse_optional_int(val: Optional[str]) -> Optional[int]:
    if val in (None, ""):
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _build_metrics_summary(metrics_path: Path) -> Dict[str, SplitMetricSummary]:
    splits: Dict[str, Dict[str, Optional[float]]] = {}
    if not metrics_path.exists():
        return {}
    with metrics_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row.get("split")
            if not split:
                continue
            model_tag = row.get("model", "")

            logloss_value = row.get("logloss")
            brier_value = row.get("brier")
            ece_value = row.get("ece")
            ece_q_value = row.get("ece_q")

            try:
                logloss = float(logloss_value) if logloss_value not in (None, "") else None
            except ValueError:
                logloss = None
            try:
                brier = float(brier_value) if brier_value not in (None, "") else None
            except ValueError:
                brier = None
            try:
                ece = float(ece_value) if ece_value not in (None, "") else None
            except ValueError:
                ece = None
            try:
                ece_q = float(ece_q_value) if ece_q_value not in (None, "") else None
            except ValueError:
                ece_q = None

            if logloss is None:
                continue

            data = splits.setdefault(split, {
                "baseline_logloss": None, "model_logloss": None,
                "baseline_brier": None, "model_brier": None,
                "baseline_ece": None, "model_ece": None,
                "baseline_ece_q": None, "model_ece_q": None,
                "delta_logloss_ci_lo": None, "delta_logloss_ci_hi": None,
                "delta_brier_ci_lo": None, "delta_brier_ci_hi": None,
                "delta_ece_ci_lo": None, "delta_ece_ci_hi": None,
                "delta_ece_q_ci_lo": None, "delta_ece_q_ci_hi": None,
                "bootstrap_n_groups": None, "bootstrap_B": None,
            })

            if model_tag.startswith("baseline"):
                data["baseline_logloss"] = logloss
                data["baseline_brier"] = brier
                data["baseline_ece"] = ece
                data["baseline_ece_q"] = ece_q
            elif model_tag.startswith("rolling"):
                continue
            elif data["model_logloss"] is None:
                data["model_logloss"] = logloss
                data["model_brier"] = brier
                data["model_ece"] = ece
                data["model_ece_q"] = ece_q
                # Bootstrap CI columns (present only when --bootstrap-ci was used)
                data["delta_logloss_ci_lo"] = _parse_optional_float(row.get("delta_logloss_ci_lo"))
                data["delta_logloss_ci_hi"] = _parse_optional_float(row.get("delta_logloss_ci_hi"))
                data["delta_brier_ci_lo"] = _parse_optional_float(row.get("delta_brier_ci_lo"))
                data["delta_brier_ci_hi"] = _parse_optional_float(row.get("delta_brier_ci_hi"))
                data["delta_ece_ci_lo"] = _parse_optional_float(row.get("delta_ece_ci_lo"))
                data["delta_ece_ci_hi"] = _parse_optional_float(row.get("delta_ece_ci_hi"))
                data["delta_ece_q_ci_lo"] = _parse_optional_float(row.get("delta_ece_q_ci_lo"))
                data["delta_ece_q_ci_hi"] = _parse_optional_float(row.get("delta_ece_q_ci_hi"))
                data["bootstrap_n_groups"] = _parse_optional_int(row.get("bootstrap_n_groups"))
                data["bootstrap_B"] = _parse_optional_int(row.get("bootstrap_B"))

    summary: Dict[str, SplitMetricSummary] = {}
    for split, data in splits.items():
        baseline_logloss = data.get("baseline_logloss")
        model_logloss = data.get("model_logloss")
        if baseline_logloss is None or model_logloss is None:
            continue

        delta_logloss = model_logloss - baseline_logloss
        status = "good" if delta_logloss < 0 else "unusable"
        verdict = (
            "Model improves the baseline logloss."
            if delta_logloss < 0
            else "Model does not improve the baseline."
        )

        baseline_brier = data.get("baseline_brier")
        model_brier = data.get("model_brier")
        delta_brier = None
        if baseline_brier is not None and model_brier is not None:
            delta_brier = model_brier - baseline_brier

        baseline_ece = data.get("baseline_ece")
        model_ece = data.get("model_ece")
        delta_ece = None
        if baseline_ece is not None and model_ece is not None:
            delta_ece = model_ece - baseline_ece

        baseline_ece_q = data.get("baseline_ece_q")
        model_ece_q = data.get("model_ece_q")
        delta_ece_q = None
        if baseline_ece_q is not None and model_ece_q is not None:
            delta_ece_q = model_ece_q - baseline_ece_q

        summary[split] = SplitMetricSummary(
            split=split,
            baseline_logloss=baseline_logloss,
            model_logloss=model_logloss,
            delta_model_minus_baseline=delta_logloss,
            baseline_brier=baseline_brier,
            model_brier=model_brier,
            delta_brier=delta_brier,
            baseline_ece=baseline_ece,
            model_ece=model_ece,
            delta_ece=delta_ece,
            baseline_ece_q=baseline_ece_q,
            model_ece_q=model_ece_q,
            delta_ece_q=delta_ece_q,
            delta_logloss_ci_lo=data.get("delta_logloss_ci_lo"),
            delta_logloss_ci_hi=data.get("delta_logloss_ci_hi"),
            delta_brier_ci_lo=data.get("delta_brier_ci_lo"),
            delta_brier_ci_hi=data.get("delta_brier_ci_hi"),
            delta_ece_ci_lo=data.get("delta_ece_ci_lo"),
            delta_ece_ci_hi=data.get("delta_ece_ci_hi"),
            delta_ece_q_ci_lo=data.get("delta_ece_q_ci_lo"),
            delta_ece_q_ci_hi=data.get("delta_ece_q_ci_hi"),
            bootstrap_n_groups=data.get("bootstrap_n_groups"),
            bootstrap_B=data.get("bootstrap_B"),
            status=status,
            verdict=verdict,
        )
    return summary


def _build_split_counts(out_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    split_row_counts: Dict[str, int] = {}
    split_group_counts: Dict[str, int] = {}
    path = out_dir / "audit_split_composition.csv"
    if not path.exists():
        return split_row_counts, split_group_counts
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row.get("split")
            if not split:
                continue
            n_rows = _parse_optional_int(row.get("n_rows"))
            if n_rows is not None:
                split_row_counts[split] = n_rows
            n_groups = _parse_optional_int(row.get("n_group_id"))
            if n_groups is None:
                n_groups = _parse_optional_int(row.get("n_contract_id"))
            if n_groups is not None:
                split_group_counts[split] = n_groups
    return split_row_counts, split_group_counts


def _sanitize_name(value: str) -> str:
    cleaned = "".join(ch for ch in value.strip() if ch.isalnum() or ch in ("-", "_", "."))
    return cleaned.strip(".-_")

def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _load_two_stage_metrics(run_dir: Path) -> Optional[List[Dict[str, Any]]]:
    summary = _load_json(run_dir / "two_stage_metrics_summary.json")
    if isinstance(summary, dict):
        rows = summary.get("rows")
        if isinstance(rows, list):
            return rows
    return None

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_ABBREV_TO_INDEX = {name[:3].lower(): idx for idx, name in enumerate(DAY_NAMES)}


def _classify_tdays_regime(
    series: pd.Series,
    *,
    daily_share_threshold: float = 0.85,
    weekly_share_threshold: float = 0.95,
) -> Tuple[Optional[List[int]], Optional[bool], Optional[bool]]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return None, None, None

    rounded = vals.round().astype(int)
    counts = rounded.value_counts()
    total = int(counts.sum())
    if total <= 0:
        return None, None, None

    tdays_mode = counts.index.tolist()[:3]
    daily_share = float(counts.get(1, 0)) / float(total)
    weekly_bucket = counts[counts.index.isin([1, 2, 3, 4])]
    weekly_share = float(weekly_bucket.sum()) / float(total)
    weekly_unique = int(len(set(int(v) for v in weekly_bucket.index.tolist())))

    is_daily = daily_share >= daily_share_threshold
    is_weekly = weekly_share >= weekly_share_threshold and weekly_unique >= 2
    return tdays_mode, bool(is_weekly), bool(is_daily)


def _parse_tdays_allowed(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return None
    if len(tokens) != 1:
        raise ValueError(
            "Only one T_days regime is allowed for training. Choose exactly one of: 1, 2, 3, 4."
        )
    token = tokens[0]
    try:
        tdays = int(token)
    except ValueError as exc:
        raise ValueError(f"Invalid T_days value '{token}'.") from exc
    if tdays not in {1, 2, 3, 4}:
        raise ValueError(f"T_days must be one of 1, 2, 3, 4 (got {tdays}).")
    return tdays


def _parse_asof_dow_allowed(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return None
    allowed: List[str] = []
    for token in tokens:
        normalized = token.strip().lower()
        if normalized.isdigit():
            idx = int(normalized)
            if idx < 0 or idx > 6:
                raise ValueError(f"Day-of-week index must be 0..6 (got {idx}).")
            name = DAY_NAMES[idx]
        else:
            key = normalized[:3]
            if key not in DAY_ABBREV_TO_INDEX:
                raise ValueError(f"Invalid day-of-week token '{token}'. Use Mon..Sun or 0..6.")
            name = DAY_NAMES[DAY_ABBREV_TO_INDEX[key]]
        if name not in allowed:
            allowed.append(name)
    return allowed


def _resolve_asof_date_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["asof_date", "snapshot_time_utc", "snapshot_time", "asof_datetime_utc"]:
        if cand in df.columns:
            return cand
    return None


def _resolve_expiry_date_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["expiry_date", "expiry_close_date_used", "event_endDate"]:
        if cand in df.columns:
            return cand
    return None


def _add_asof_dow_column(df: pd.DataFrame, asof_col: str) -> None:
    dt = pd.to_datetime(df[asof_col], errors="coerce", utc=True)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"{asof_col} has {bad} NaT values; cannot derive asof_dow.")
    df["asof_dow"] = dt.dt.weekday.map(lambda idx: DAY_NAMES[int(idx)])


def _read_features(out_dir: Path) -> Optional[List[str]]:
    metadata_path = out_dir / "metadata.json"
    feature_manifest_path = out_dir / "feature_manifest.json"

    if metadata_path.exists():
        try:
            with metadata_path.open() as f:
                metadata = json.load(f)
                features = metadata.get("features")
                if features and isinstance(features, list):
                    return features
        except Exception:
            pass

    if feature_manifest_path.exists():
        try:
            with feature_manifest_path.open() as f:
                manifest = json.load(f)
                features = manifest.get("features")
                if features and isinstance(features, list):
                    return features
        except Exception:
            pass

    return None


def _escape_latex_text(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace("_", "\\_")
    escaped = escaped.replace("{", "\\{")
    escaped = escaped.replace("}", "\\}")
    return escaped


def _format_feature_latex(feat_name: str) -> str:
    """
    Format feature names with proper mathematical notation for KaTeX rendering.

    Returns LaTeX expression (without wrapping \\text{}) for better math rendering.
    """
    # Map of feature names to their LaTeX representations
    latex_map = {
        "x_logit_prn": "x_{\\text{logit-pRN}}",
        "x_logit_pm": "x_{\\text{logit-PM}}",
        "x_m": "x_{\\text{logit-pRN}} \\times m",
        "x_abs_m": "x_{\\text{logit-pRN}} \\times |m|",
        "log_m": "\\log(m)",
        "log_m_fwd": "\\log(m_{\\text{fwd}})",
        "abs_log_m": "|\\log(m)|",
        "abs_log_m_fwd": "|\\log(m_{\\text{fwd}})|",
        "sqrt_T_years": "\\sqrt{T}",
        "rv20_sqrtT": "\\sigma_{20} \\sqrt{T}",
        "log_m_over_volT": "\\frac{\\log(m)}{\\sigma \\sqrt{T}}",
        "abs_log_m_over_volT": "\\frac{|\\log(m)|}{\\sigma \\sqrt{T}}",
        "log_m_fwd_over_volT": "\\frac{\\log(m_{\\text{fwd}})}{\\sigma \\sqrt{T}}",
        "abs_log_m_fwd_over_volT": "\\frac{|\\log(m_{\\text{fwd}})|}{\\sigma \\sqrt{T}}",
        "log_rel_spread": "\\log(\\text{spread})",
        "log_T_days": "\\log(T_{\\text{days}})",
        "T_days": "T_{\\text{days}}",
        "T_years": "T_{\\text{years}}",
        "rv20": "\\sigma_{20}",
        "dividend_yield": "q",
        "x_prn_x_tdays": "x_{\\text{logit-pRN}} \\times T_{\\text{days}}",
        "x_prn_x_rv20": "x_{\\text{logit-pRN}} \\times \\sigma_{20}",
        "x_prn_x_logm": "x_{\\text{logit-pRN}} \\times \\log(m)",
        "prn_raw_gap": "\\Delta_{\\text{pRN}}",
        "had_fallback": "\\mathbb{1}_{\\text{fallback}}",
        "had_intrinsic_drop": "\\mathbb{1}_{\\text{intrinsic-drop}}",
        "had_band_clip": "\\mathbb{1}_{\\text{band-clip}}",
    }

    # Return mapped LaTeX if available, otherwise escape and wrap in \text{}
    if feat_name in latex_map:
        return latex_map[feat_name]
    else:
        return f"\\text{{{_escape_latex_text(feat_name)}}}"


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return float(num)


def _as_str_list(value: Any) -> Optional[List[str]]:
    if not isinstance(value, list):
        return None
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        out.append(str(item))
    return out


def _as_float_list(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list):
        return None
    out: List[float] = []
    for item in value:
        parsed = _as_float(item)
        if parsed is None:
            return None
        out.append(parsed)
    return out


def _load_model_json(path: Path) -> Optional[Dict[str, Any]]:
    data = _load_json(path)
    return data if isinstance(data, dict) else None


def _strip_transform_prefixes(feature_name: str) -> Tuple[List[str], str]:
    parts = str(feature_name).split("__")
    if len(parts) == 1:
        return [], str(feature_name)
    return parts[:-1], parts[-1]


def _parse_onehot_raw_feature(
    raw_name: str,
    categorical_candidates: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    best_col: Optional[str] = None
    best_cat: Optional[str] = None
    for cand in sorted({c for c in categorical_candidates if c}, key=len, reverse=True):
        if raw_name == cand:
            return cand, None
        prefix = f"{cand}_"
        if raw_name.startswith(prefix):
            cat = raw_name[len(prefix):]
            best_col = cand
            best_cat = cat
            break
    if best_col is not None:
        return best_col, best_cat
    if "_" in raw_name:
        col_guess, cat_guess = raw_name.rsplit("_", 1)
        return col_guess or None, cat_guess or None
    return None, None


def _indicator_latex(col: str, category: Optional[str]) -> str:
    if category is None:
        return f"\\mathbb{{1}}_{{\\text{{{_escape_latex_text(col)}}}}}"
    return (
        "\\mathbb{1}_{\\text{"
        + _escape_latex_text(col)
        + "="
        + _escape_latex_text(category)
        + "}}"
    )


def _format_linear_term_latex(coef: float, feature_latex: str) -> str:
    sign = "+" if coef >= 0 else "-"
    return f"{sign} {abs(coef):.4f} \\cdot {feature_latex}"


def _join_linear_terms(intercept: Optional[float], terms: List[str]) -> Optional[str]:
    pieces: List[str] = []
    if intercept is not None:
        pieces.append(f"{intercept:.4f}")
    pieces.extend(terms)
    if not pieces:
        return None
    return " ".join(pieces)


def _categorical_candidates_for_metadata(
    metadata: Dict[str, Any],
    feature_manifest: Optional[Dict[str, Any]],
) -> List[str]:
    candidates: List[str] = []
    for key in [
        "categorical_features_used",
        "categorical_features",
        "stage2_categorical_features",
        "stage1_categorical_features",
    ]:
        vals = _as_str_list(metadata.get(key))
        if vals:
            candidates.extend(vals)
    for key in ["ticker_feature_col", "interaction_ticker_col", "ticker_col"]:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)
    if feature_manifest:
        vals = _as_str_list(feature_manifest.get("categorical_features"))
        if vals:
            candidates.extend(vals)
        for key in ["ticker_feature_col", "interaction_ticker_col", "ticker_col"]:
            value = feature_manifest.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
    seen: set[str] = set()
    out: List[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _ticker_column_candidates(metadata: Dict[str, Any], categorical_candidates: List[str]) -> List[str]:
    candidates: List[str] = []
    for key in ["ticker_feature_col", "interaction_ticker_col", "ticker_col"]:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            candidates.append(value)
    for cand in categorical_candidates:
        if cand == "ticker" or "ticker_feature" in cand or cand.startswith("_ticker"):
            candidates.append(cand)
    seen: set[str] = set()
    out: List[str] = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def _parse_transformed_feature(
    feature_name: str,
    *,
    categorical_candidates: List[str],
    ticker_col_candidates: List[str],
) -> Dict[str, Any]:
    prefixes, raw_name = _strip_transform_prefixes(feature_name)
    prefix_set = set(prefixes)
    is_ticker_interaction = "ticker_x" in prefix_set

    if raw_name == "p_base":
        return {
            "kind": "base_probability",
            "feature_name": feature_name,
            "raw_name": raw_name,
            "prefixes": prefixes,
            "latex": "\\hat{p}_{\\text{base}}",
        }

    if raw_name.startswith("pm_") or raw_name.startswith("pPM_") or raw_name == "x_logit_pm":
        return {
            "kind": "pm_feature",
            "feature_name": feature_name,
            "raw_name": raw_name,
            "prefixes": prefixes,
            "latex": _format_feature_latex(raw_name),
            "pm_feature_name": raw_name,
        }

    col_name, category = _parse_onehot_raw_feature(raw_name, categorical_candidates)
    if "cat" in prefix_set:
        resolved_col = col_name or raw_name
        is_ticker_col = resolved_col in ticker_col_candidates or resolved_col == "ticker"
        if is_ticker_interaction:
            ticker_label = category or "OTHER"
            return {
                "kind": "ticker_interaction",
                "feature_name": feature_name,
                "raw_name": raw_name,
                "prefixes": prefixes,
                "ticker_label": ticker_label,
                "categorical_col": resolved_col,
                "latex": (
                    _indicator_latex("ticker", ticker_label)
                    + " \\cdot x_{\\text{ticker-int}}"
                ),
            }
        if is_ticker_col:
            ticker_label = category or "OTHER"
            return {
                "kind": "ticker_intercept",
                "feature_name": feature_name,
                "raw_name": raw_name,
                "prefixes": prefixes,
                "ticker_label": ticker_label,
                "categorical_col": resolved_col,
                "latex": _indicator_latex("ticker", ticker_label),
            }
        return {
            "kind": "categorical_onehot",
            "feature_name": feature_name,
            "raw_name": raw_name,
            "prefixes": prefixes,
            "categorical_col": resolved_col,
            "category": category,
            "latex": _indicator_latex(resolved_col, category),
        }

    if is_ticker_interaction:
        return {
            "kind": "ticker_interaction",
            "feature_name": feature_name,
            "raw_name": raw_name,
            "prefixes": prefixes,
            "ticker_label": None,
            "latex": "\\delta_{\\text{ticker-int}}(\\text{ticker})",
        }

    return {
        "kind": "numeric",
        "feature_name": feature_name,
        "raw_name": raw_name,
        "prefixes": prefixes,
        "latex": _format_feature_latex(raw_name),
    }


def _build_linear_equation_spec(
    *,
    coefficients: List[float],
    intercept: Optional[float],
    feature_names: List[str],
    metadata: Dict[str, Any],
    feature_manifest: Optional[Dict[str, Any]],
    p_hat_symbol: str,
    linear_symbol: str,
    include_sigmoid: bool,
    platt_mode: bool,
) -> Optional[Dict[str, Any]]:
    if not coefficients or not feature_names:
        return None

    categorical_candidates = _categorical_candidates_for_metadata(metadata, feature_manifest)
    ticker_col_candidates = _ticker_column_candidates(metadata, categorical_candidates)
    n_pairs = min(len(coefficients), len(feature_names))
    mismatch = len(coefficients) != len(feature_names)

    compact_terms: List[str] = []
    expanded_terms: List[str] = []
    ticker_intercept_rows: List[Dict[str, Any]] = []
    ticker_interaction_rows: List[Dict[str, Any]] = []
    pm_feature_names: List[str] = []
    compact_ticker_intercepts_added = False
    compact_ticker_interactions_added = False

    for idx in range(n_pairs):
        feat = str(feature_names[idx])
        coef = float(coefficients[idx])
        parsed = _parse_transformed_feature(
            feat,
            categorical_candidates=categorical_candidates,
            ticker_col_candidates=ticker_col_candidates,
        )
        kind = parsed.get("kind")
        latex = str(parsed.get("latex") or _format_feature_latex(feat))
        term_latex = _format_linear_term_latex(coef, latex)
        expanded_terms.append(term_latex)

        if kind == "ticker_intercept":
            ticker_intercept_rows.append({
                "feature_name": feat,
                "ticker": parsed.get("ticker_label"),
                "coef": coef,
                "latex": term_latex,
            })
            if not compact_ticker_intercepts_added:
                compact_ticker_intercepts_added = True
                compact_terms.append("+ \\delta_{\\text{ticker}}(\\text{ticker})")
            continue
        if kind == "ticker_interaction":
            ticker_interaction_rows.append({
                "feature_name": feat,
                "ticker": parsed.get("ticker_label"),
                "coef": coef,
                "latex": term_latex,
            })
            if not compact_ticker_interactions_added:
                compact_ticker_interactions_added = True
                compact_terms.append("+ \\delta_{\\text{ticker-int}}(\\text{ticker})")
            continue
        if kind == "pm_feature":
            pm_name = parsed.get("pm_feature_name")
            if isinstance(pm_name, str) and pm_name not in pm_feature_names:
                pm_feature_names.append(pm_name)
        compact_terms.append(term_latex)

    compact_linear = _join_linear_terms(intercept, compact_terms)
    expanded_linear = _join_linear_terms(intercept, expanded_terms)
    if compact_linear is None or expanded_linear is None:
        return None

    if include_sigmoid:
        compact_core = f"{p_hat_symbol} = \\operatorname{{sigmoid}}\\left( {compact_linear} \\right)"
        expanded_core = f"{p_hat_symbol} = \\operatorname{{sigmoid}}\\left( {expanded_linear} \\right)"
    else:
        compact_core = f"{p_hat_symbol} = {compact_linear}"
        expanded_core = f"{p_hat_symbol} = {expanded_linear}"

    notes: List[str] = []
    if platt_mode:
        notes.append("Displayed coefficients are the base logistic layer; final pHAT applies an additional Platt calibration transform.")
    if mismatch:
        notes.append(
            f"Coefficient/feature count mismatch: {len(coefficients)} coefficients vs {len(feature_names)} feature names; truncated to {n_pairs} terms."
        )
    if ticker_intercept_rows or ticker_interaction_rows:
        notes.append("Compact equation uses ticker-dependent placeholder terms; expanded terms list enumerates learned ticker basis coefficients.")
    if any("__" in name for name in feature_names):
        notes.append("Equation is shown in transformed model basis (after preprocessing / one-hot encoding).")
    if any("__" in name for name in feature_names) and any("cat__" in name for name in feature_names):
        notes.append("Categorical one-hot encoding uses drop-first reference levels, which are implicit (not shown).")

    return {
        "compact_latex": "\\displaystyle " + compact_core,
        "expanded_latex": "\\displaystyle " + expanded_core,
        "linear_predictor_compact_latex": "\\displaystyle " + f"{linear_symbol} = {compact_linear}",
        "linear_predictor_expanded_latex": "\\displaystyle " + f"{linear_symbol} = {expanded_linear}",
        "term_count": int(n_pairs),
        "intercept": intercept,
        "uses_transformed_features": bool(any("__" in name for name in feature_names)),
        "feature_name_source": "feature_names_out" if any("__" in name for name in feature_names) else "raw_features",
        "pm_feature_names": pm_feature_names,
        "ticker_intercepts": ticker_intercept_rows,
        "ticker_interactions": ticker_interaction_rows,
        "notes": notes,
        "all_feature_names": feature_names[:n_pairs],
    }


def _build_model_equation_spec(out_dir: Path) -> Optional[Dict[str, Any]]:
    metadata = _load_model_json(out_dir / "metadata.json")
    if not metadata:
        return None
    feature_manifest = _load_model_json(out_dir / "feature_manifest.json")

    coefficients = _as_float_list(metadata.get("coefficients"))
    intercept = _as_float(metadata.get("intercept"))
    feature_names = (
        _as_str_list(metadata.get("feature_names_out"))
        or _as_str_list(metadata.get("feature_names"))
        or _as_str_list(metadata.get("features"))
    )
    if not coefficients or not feature_names:
        return None

    calibration_mode = str(
        metadata.get("calibration_used")
        or metadata.get("calibration")
        or metadata.get("calibration_requested")
        or "none"
    ).strip().lower()
    platt_mode = calibration_mode == "platt"

    spec = _build_linear_equation_spec(
        coefficients=coefficients,
        intercept=intercept,
        feature_names=feature_names,
        metadata=metadata,
        feature_manifest=feature_manifest,
        p_hat_symbol="\\hat{p}",
        linear_symbol="\\eta",
        include_sigmoid=not platt_mode,
        platt_mode=platt_mode,
    )
    if spec is None:
        return None
    if platt_mode:
        # Final pHAT is Platt(logit-layer); exact Platt parameters are not always serialized.
        compact_eta = spec.get("linear_predictor_compact_latex")
        expanded_eta = spec.get("linear_predictor_expanded_latex")
        spec["compact_latex"] = (
            "\\displaystyle \\hat{p} = \\operatorname{Platt}\\!\\left(\\eta\\right),\\; "
            + (compact_eta.replace("\\displaystyle ", "") if isinstance(compact_eta, str) else "")
        )
        if isinstance(expanded_eta, str):
            spec["expanded_latex"] = (
                "\\displaystyle \\hat{p} = \\operatorname{Platt}\\!\\left(\\eta\\right),\\; "
                + expanded_eta.replace("\\displaystyle ", "")
            )
    spec["model_family"] = "single_stage"
    return spec


def _build_model_equation(out_dir: Path) -> Optional[str]:
    spec = _build_model_equation_spec(out_dir)
    if not spec:
        return None
    compact = spec.get("compact_latex")
    return compact if isinstance(compact, str) else None


def _build_two_stage_equation_spec(out_dir: Path) -> Optional[Dict[str, Any]]:
    stage2_meta = _load_model_json(out_dir / "two_stage_metadata.json")
    if not stage2_meta:
        return None

    coefficients = _as_float_list(stage2_meta.get("stage2_coefficients"))
    intercept = _as_float(stage2_meta.get("stage2_intercept"))
    feature_names = _as_str_list(stage2_meta.get("stage2_feature_names")) or _as_str_list(stage2_meta.get("stage2_features"))
    if not coefficients or not feature_names:
        return None

    feature_manifest = _load_model_json(out_dir / "feature_manifest.json")
    # Stage2 metadata stores categorical_features directly.
    spec = _build_linear_equation_spec(
        coefficients=coefficients,
        intercept=intercept,
        feature_names=feature_names,
        metadata=stage2_meta,
        feature_manifest=feature_manifest,
        p_hat_symbol="\\hat{p}_{\\text{stageB}}",
        linear_symbol="\\eta_{\\text{stageB}}",
        include_sigmoid=True,
        platt_mode=False,
    )
    if spec is None:
        return None
    # Stage2 may also have its own Platt calibrator; params are not guaranteed in metadata.
    calibration_mode = str(stage2_meta.get("calibration") or "none").strip().lower()
    if calibration_mode == "platt":
        spec["notes"] = list(spec.get("notes") or []) + [
            "Stage B final probabilities may apply an additional Platt calibration transform when configured."
        ]
    spec["model_family"] = "two_stage_overlay"
    return spec


def _build_two_stage_equation(out_dir: Path) -> Optional[str]:
    spec = _build_two_stage_equation_spec(out_dir)
    if not spec:
        return None
    compact = spec.get("compact_latex")
    return compact if isinstance(compact, str) else None


def _build_combined_p_hat_equation_spec(out_dir: Path) -> Optional[Dict[str, Any]]:
    stage1_spec = _build_model_equation_spec(out_dir)
    stage2_spec = _build_two_stage_equation_spec(out_dir)
    if not stage1_spec or not stage2_spec:
        return None

    stage1_compact = stage1_spec.get("compact_latex")
    stage2_compact = stage2_spec.get("compact_latex")
    if not isinstance(stage1_compact, str) or not isinstance(stage2_compact, str):
        return None

    compact = (
        "\\displaystyle \\hat{p}_{\\text{final}} = "
        "\\mathbb{1}_{\\text{PM available}} \\cdot \\hat{p}_{\\text{stageB}} + "
        "\\mathbb{1}_{\\text{PM missing}} \\cdot \\hat{p}_{\\text{base}}"
    )
    expanded = (
        compact
        + "\\\\ "
        + stage1_compact.replace("\\displaystyle ", "")
        + "\\\\ "
        + stage2_compact.replace("\\displaystyle ", "")
    )

    notes: List[str] = [
        "Final pHAT is piecewise: Stage B (Polymarket overlay) when PM features are available, otherwise Stage A base prediction."
    ]
    pm_terms = stage2_spec.get("pm_feature_names")
    if isinstance(pm_terms, list) and pm_terms:
        notes.append("Stage B PM features present: " + ", ".join(str(x) for x in pm_terms))

    return {
        "compact_latex": compact,
        "expanded_latex": expanded,
        "notes": notes,
        "pm_feature_names": stage2_spec.get("pm_feature_names") or [],
        "model_family": "two_stage_combined",
    }


def _build_combined_p_hat_equation(out_dir: Path) -> Optional[str]:
    spec = _build_combined_p_hat_equation_spec(out_dir)
    if not spec:
        return None
    compact = spec.get("compact_latex")
    return compact if isinstance(compact, str) else None


def preview_regime(payload: RegimePreviewRequest) -> RegimePreviewResponse:
    dataset_path = _resolve_project_path(payload.csv)
    _ensure_dataset_path(dataset_path)

    df = pd.read_csv(dataset_path)
    tdays_allowed = _parse_tdays_allowed(payload.tdays_allowed)
    asof_dow_allowed = _parse_asof_dow_allowed(payload.asof_dow_allowed)

    rows_before = int(len(df))
    by_tdays: Dict[str, int] = {}

    if tdays_allowed is not None or asof_dow_allowed:
        missing_filter_cols = []
        if tdays_allowed is not None and "T_days" not in df.columns:
            missing_filter_cols.append("T_days")
        asof_col = _resolve_asof_date_column(df) if asof_dow_allowed else None
        if asof_dow_allowed and not asof_col:
            missing_filter_cols.append("asof_date (or equivalent)")
        if missing_filter_cols:
            raise ValueError(
                f"Regime filtering requires columns: {missing_filter_cols}. "
                "Ensure the selected regime filters have required columns."
            )

        if tdays_allowed is not None:
            df["T_days"] = pd.to_numeric(df["T_days"], errors="coerce")
        if asof_col:
            df[asof_col] = pd.to_datetime(df[asof_col], errors="coerce", utc=True)
            _add_asof_dow_column(df, asof_col)

        if tdays_allowed is not None:
            df = df[df["T_days"] == float(tdays_allowed)]
        if asof_dow_allowed:
            df = df[df["asof_dow"].isin(asof_dow_allowed)]

    if "T_days" in df.columns:
        series = df["T_days"].value_counts(dropna=True).sort_index()
        by_tdays = {str(int(k) if float(k).is_integer() else k): int(v) for k, v in series.items()}

    rows_after = int(len(df))
    tickers_after = int(df["ticker"].nunique(dropna=True)) if "ticker" in df.columns else 0

    return RegimePreviewResponse(
        rows_before=rows_before,
        rows_after=rows_after,
        tickers_after=tickers_after,
        by_tdays=by_tdays,
    )


def _resolve_weight_column_for_preview(
    df: pd.DataFrame,
    strategy: Optional[str],
) -> Optional[str]:
    requested = (strategy or "auto").strip().lower()
    if requested == "uniform":
        return None
    if requested == "weight_final" and "weight_final" in df.columns:
        return "weight_final"
    if requested == "sample_weight_final" and "sample_weight_final" in df.columns:
        return "sample_weight_final"
    if requested == "auto":
        if "weight_final" in df.columns:
            return "weight_final"
        if "sample_weight_final" in df.columns:
            return "sample_weight_final"
    return None


def _resolve_grouping_key_for_preview(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested and requested in df.columns:
        return requested
    for candidate in ["group_id", "weight_group_key", "cluster_snapshot", "contract_id"]:
        if candidate in df.columns:
            return candidate
    if {"ticker", "asof_date", "expiry_date"}.issubset(df.columns):
        return "__derived_chain_group"
    return None


def _assign_preview_splits(
    df: pd.DataFrame,
    *,
    split_strategy: Optional[str],
    test_window_weeks: Optional[int],
    validation_window_weeks: Optional[int],
) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype="string")
    week_series: Optional[pd.Series] = None
    if "week_friday" in df.columns:
        parsed = pd.to_datetime(df["week_friday"], errors="coerce", utc=True).dt.normalize()
        if parsed.notna().any():
            week_series = parsed
    if week_series is None:
        for col in ["asof_date", "snapshot_date", "snapshot_time_utc", "asof_datetime_utc"]:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
                if parsed.notna().any():
                    week_series = (parsed + pd.to_timedelta((4 - parsed.dt.weekday) % 7, unit="D")).dt.normalize()
                    break
    if week_series is None:
        # Fallback to simple row-based split.
        n = len(df)
        test_cut = max(1, int(n * 0.2))
        labels = np.array(["train"] * n, dtype=object)
        labels[-test_cut:] = "test"
        if split_strategy == "walk_forward" and n >= 10:
            val_cut = max(1, int(n * 0.1))
            labels[-(test_cut + val_cut):-test_cut] = "val"
        return pd.Series(labels, index=df.index, dtype="string")

    weeks = week_series.dropna().sort_values().unique().tolist()
    if len(weeks) < 3:
        return pd.Series(["train"] * len(df), index=df.index, dtype="string")

    test_n = max(1, int(test_window_weeks or 20))
    val_n = max(1, int(validation_window_weeks or 8))
    test_weeks = set(weeks[-test_n:])
    labels = pd.Series(["train"] * len(df), index=df.index, dtype="string")
    labels.loc[week_series.isin(test_weeks)] = "test"
    if (split_strategy or "walk_forward") == "walk_forward":
        train_weeks = [w for w in weeks if w not in test_weeks]
        if train_weeks:
            val_weeks = set(train_weeks[-min(val_n, len(train_weeks)):])
            labels.loc[week_series.isin(val_weeks)] = "val"
    return labels


def preview_weighting(payload: WeightingPreviewRequest) -> WeightingPreviewResponse:
    dataset_path = _resolve_project_path(payload.csv)
    _ensure_dataset_path(dataset_path)
    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty.")

    warnings: List[str] = []
    selected_weight_col = _resolve_weight_column_for_preview(df, payload.weight_col_strategy)
    if (payload.base_weight_source or "dataset_weight") == "uniform":
        selected_weight_col = None
    if selected_weight_col is None:
        weights = np.ones(len(df), dtype=float)
        if (payload.base_weight_source or "dataset_weight") == "dataset_weight":
            warnings.append("Requested dataset weights but no supported weight column was found; using uniform.")
    else:
        weights = pd.to_numeric(df[selected_weight_col], errors="coerce").to_numpy(dtype=float)
        invalid = ~np.isfinite(weights) | (weights <= 0)
        if invalid.any():
            warnings.append(f"Detected {int(invalid.sum())} invalid weights in {selected_weight_col}; replaced with 1.0.")
            weights = weights.copy()
            weights[invalid] = 1.0

    working = df.copy()
    working["__w"] = weights

    group_key = _resolve_grouping_key_for_preview(working, payload.grouping_key)
    if group_key == "__derived_chain_group":
        working[group_key] = (
            working["ticker"].astype("string").fillna("UNKNOWN").str.upper()
            + "|"
            + pd.to_datetime(working["asof_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("NA")
            + "|"
            + pd.to_datetime(working["expiry_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d").fillna("NA")
        )

    if payload.trading_universe_tickers and "ticker" in working.columns:
        tickers = {
            token.strip().upper()
            for token in str(payload.trading_universe_tickers).split(",")
            if token.strip()
        }
        mult = float(payload.trading_universe_upweight or 1.0)
        if tickers and mult > 0 and abs(mult - 1.0) > 1e-12:
            mask = working["ticker"].astype("string").str.upper().isin(tickers).to_numpy()
            if mask.any():
                working.loc[mask, "__w"] = working.loc[mask, "__w"] * mult

    if (payload.ticker_balance_mode or "none") == "sqrt_inv_clipped" and "ticker" in working.columns:
        ticker_series = working["ticker"].astype("string").str.upper().fillna("UNKNOWN")
        counts = ticker_series.value_counts(dropna=False)
        if len(counts) > 0:
            mean_count = float(counts.mean())
            factors = {
                key: float(np.clip(np.sqrt(mean_count / max(1.0, float(count))), 0.5, 2.0))
                for key, count in counts.items()
            }
            working["__w"] = working["__w"] * ticker_series.map(factors).astype(float)

    if bool(payload.group_equalization):
        if group_key and group_key in working.columns:
            group_sum = working.groupby(group_key, dropna=False)["__w"].transform("sum")
            group_invalid = ~np.isfinite(group_sum) | (group_sum <= 0)
            if group_invalid.any():
                warnings.append("Some groups have invalid weight sums; equalization skipped for those rows.")
            safe_den = group_sum.where(~group_invalid, 1.0)
            working["__w"] = working["__w"] / safe_den
        else:
            warnings.append("Group equalization requested but grouping key could not be resolved.")

    mean_weight = float(pd.to_numeric(working["__w"], errors="coerce").mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Weight preview failed: non-finite mean weight after adjustments.")
    working["__w"] = working["__w"] / mean_weight

    min_w = float(pd.to_numeric(working["__w"], errors="coerce").min())
    mean_w = float(pd.to_numeric(working["__w"], errors="coerce").mean())
    max_w = float(pd.to_numeric(working["__w"], errors="coerce").max())

    group_sum_min = None
    group_sum_mean = None
    group_sum_max = None
    if group_key and group_key in working.columns:
        sums = working.groupby(group_key, dropna=False)["__w"].sum()
        if not sums.empty:
            group_sum_min = float(sums.min())
            group_sum_mean = float(sums.mean())
            group_sum_max = float(sums.max())
            if group_sum_mean > 0 and (group_sum_max / group_sum_mean > 4.0):
                warnings.append("Per-group weight sums are highly imbalanced (>4x mean).")

    split_labels = _assign_preview_splits(
        working,
        split_strategy=payload.split_strategy,
        test_window_weeks=payload.test_window_weeks,
        validation_window_weeks=payload.validation_window_weeks,
    )
    split_row_counts: Dict[str, int] = {}
    split_group_counts: Dict[str, int] = {}
    for split in ["train", "val", "test"]:
        mask = split_labels == split
        split_row_counts[split] = int(mask.sum())
        if group_key and group_key in working.columns:
            split_group_counts[split] = int(working.loc[mask, group_key].nunique(dropna=True))
        else:
            split_group_counts[split] = 0

    if group_key and split_group_counts.get("val", 0) and split_group_counts["val"] < 30:
        warnings.append("Validation split has fewer than 30 groups; CI estimates may be unstable.")

    return WeightingPreviewResponse(
        selected_weight_column=selected_weight_col,
        min_weight=min_w,
        mean_weight=mean_w,
        max_weight=max_w,
        group_sum_min=group_sum_min,
        group_sum_mean=group_sum_mean,
        group_sum_max=group_sum_max,
        split_group_counts=split_group_counts,
        split_row_counts=split_row_counts,
        warnings=warnings,
    )


def get_model_detail(model_id: str) -> ModelDetailResponse:
    target = (MODELS_DIR / model_id).resolve()
    try:
        target.relative_to(MODELS_DIR.resolve())
    except Exception as exc:
        raise KeyError(model_id) from exc
    if not target.exists() or not target.is_dir():
        raise KeyError(model_id)

    summary = _model_summary_from_path(target)
    effective_dir = _resolve_effective_model_dir(target)
    metadata = _load_json(effective_dir / "metadata.json")
    feature_manifest = _load_json(effective_dir / "feature_manifest.json")
    run_manifest = _load_run_manifest(target)
    auto_search_summary = _load_json(_resolve_auto_search_dir(target) / "auto_search_summary.json")

    metrics_summary = _build_metrics_summary(effective_dir / "metrics.csv")
    if not metrics_summary:
        metrics_summary = None
    split_row_counts, split_group_counts = _build_split_counts(effective_dir)
    if not split_row_counts:
        split_row_counts = None
    if not split_group_counts:
        split_group_counts = None

    if summary.run_type == "auto":
        files = sorted(
            [str(entry.get("relative_path") or "") for entry in _collect_curated_file_entries(target)]
        )
    else:
        files = sorted([item.name for item in target.iterdir() if item.is_file()])
    two_stage_metrics = _load_two_stage_metrics(effective_dir)

    # Detect if this is a two-stage model from the effective model directory.
    is_two_stage = (effective_dir / "two_stage_metadata.json").exists()

    # Build equations based on model type
    if is_two_stage:
        stage1_equation_spec = _build_model_equation_spec(effective_dir)
        two_stage_equation_spec = _build_two_stage_equation_spec(effective_dir)
        combined_p_hat_equation_spec = _build_combined_p_hat_equation_spec(effective_dir)
        stage1_equation = stage1_equation_spec.get("compact_latex") if stage1_equation_spec else None
        two_stage_equation = two_stage_equation_spec.get("compact_latex") if two_stage_equation_spec else None
        combined_p_hat_equation = (
            combined_p_hat_equation_spec.get("compact_latex")
            if combined_p_hat_equation_spec else None
        )
        model_equation = None  # Don't show single equation for two-stage models
        model_equation_spec = None
    else:
        model_equation_spec = _build_model_equation_spec(effective_dir)
        model_equation = model_equation_spec.get("compact_latex") if model_equation_spec else None
        stage1_equation = None
        stage1_equation_spec = None
        two_stage_equation = None
        two_stage_equation_spec = None
        combined_p_hat_equation = None
        combined_p_hat_equation_spec = None

    features_used = None
    categorical_features_used = None
    if metadata:
        features_used = metadata.get("features_used_final") if isinstance(metadata.get("features_used_final"), list) else None
        categorical_features_used = metadata.get("categorical_features_used") if isinstance(metadata.get("categorical_features_used"), list) else None
    if feature_manifest:
        if not features_used and isinstance(feature_manifest.get("numeric_features"), list):
            features_used = feature_manifest.get("numeric_features")
        if not categorical_features_used and isinstance(feature_manifest.get("categorical_features"), list):
            categorical_features_used = feature_manifest.get("categorical_features")

    metadata_payload: Optional[Dict[str, Any]] = metadata if isinstance(metadata, dict) else None
    if summary.run_type == "auto":
        merged = dict(metadata_payload or {})
        if isinstance(run_manifest, dict):
            merged["run_manifest"] = run_manifest
        if isinstance(auto_search_summary, dict):
            merged["auto_search_summary"] = auto_search_summary
        metadata_payload = merged

    return ModelDetailResponse(
        id=summary.id,
        path=summary.path,
        last_modified=summary.last_modified,
        has_metadata=summary.has_metadata,
        has_metrics=summary.has_metrics,
        files=files,
        features_used=features_used,
        categorical_features_used=categorical_features_used,
        metrics_summary=metrics_summary,
        split_row_counts=split_row_counts,
        split_group_counts=split_group_counts,
        model_equation=model_equation,
        model_equation_spec=model_equation_spec,
        metadata=metadata_payload,
        feature_manifest=feature_manifest if isinstance(feature_manifest, dict) else None,
        two_stage_metrics=two_stage_metrics,
        is_two_stage=is_two_stage,
        stage1_equation=stage1_equation,
        stage1_equation_spec=stage1_equation_spec,
        two_stage_equation=two_stage_equation,
        two_stage_equation_spec=two_stage_equation_spec,
        combined_p_hat_equation=combined_p_hat_equation,
        combined_p_hat_equation_spec=combined_p_hat_equation_spec,
    )


def _find_latest_model_dir(base_dir: Path) -> Optional[Path]:
    """Find the most recently modified model directory that contains required files."""
    if not base_dir.exists():
        return None

    candidates = []
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        # Check if directory has at least one of the expected files
        if (item / "metrics.csv").exists() or (item / "metadata.json").exists():
            candidates.append(item)

    if not candidates:
        return None

    # Sort by modification time, most recent first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _build_config_payload(
    payload: CalibrateModelRunRequest,
    *,
    dataset_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    split_cfg = payload.split
    reg_cfg = payload.regularization
    structure_cfg = payload.model_structure
    weighting_cfg = payload.weighting
    bootstrap_cfg = payload.bootstrap
    diagnostics_cfg = payload.diagnostics

    effective_week_col = payload.week_col
    effective_ticker_col = payload.ticker_col
    effective_weight_col = payload.weight_col
    effective_foundation_tickers = payload.foundation_tickers
    effective_foundation_weight = payload.foundation_weight
    effective_train_tickers = payload.train_tickers
    effective_ticker_intercepts = payload.ticker_intercepts
    effective_ticker_x_interactions = payload.ticker_x_interactions
    effective_ticker_min_support = payload.ticker_min_support
    effective_ticker_min_support_interactions = payload.ticker_min_support_interactions
    effective_calibrate = payload.calibrate
    effective_c_grid = payload.c_grid
    effective_selection_objective = payload.selection_objective
    effective_test_weeks = payload.test_weeks
    effective_random_state = payload.random_state if payload.random_state is not None else payload.random_seed
    effective_group_reweight = payload.group_reweight
    effective_bootstrap_ci = payload.bootstrap_ci
    effective_bootstrap_b = payload.bootstrap_b
    effective_bootstrap_seed = payload.bootstrap_seed
    effective_bootstrap_group = payload.bootstrap_group
    effective_allow_iid_bootstrap = payload.allow_iid_bootstrap

    if structure_cfg:
        if structure_cfg.train_tickers:
            effective_train_tickers = structure_cfg.train_tickers
        if structure_cfg.foundation_tickers:
            effective_foundation_tickers = structure_cfg.foundation_tickers
        if structure_cfg.foundation_weight is not None:
            effective_foundation_weight = structure_cfg.foundation_weight
        if structure_cfg.ticker_intercepts:
            effective_ticker_intercepts = structure_cfg.ticker_intercepts
        if structure_cfg.ticker_x_interactions is not None:
            effective_ticker_x_interactions = structure_cfg.ticker_x_interactions
        if structure_cfg.ticker_min_support is not None:
            effective_ticker_min_support = structure_cfg.ticker_min_support
        if structure_cfg.ticker_min_support_interactions is not None:
            effective_ticker_min_support_interactions = structure_cfg.ticker_min_support_interactions
        if structure_cfg.trading_universe_tickers and effective_train_tickers is None:
            effective_train_tickers = structure_cfg.trading_universe_tickers

    if reg_cfg:
        if reg_cfg.calibration_method:
            effective_calibrate = reg_cfg.calibration_method
        if reg_cfg.c_grid:
            effective_c_grid = reg_cfg.c_grid
        if reg_cfg.selection_objective:
            effective_selection_objective = reg_cfg.selection_objective

    if split_cfg and split_cfg.test_window_weeks is not None:
        effective_test_weeks = split_cfg.test_window_weeks

    if payload.weight_col_strategy:
        if payload.weight_col_strategy == "uniform":
            effective_weight_col = "uniform"
        elif payload.weight_col_strategy in {"weight_final", "sample_weight_final"}:
            effective_weight_col = payload.weight_col_strategy
        elif payload.weight_col_strategy == "auto" and not effective_weight_col:
            effective_weight_col = "weight_final"

    if weighting_cfg:
        if weighting_cfg.base_weight_source == "uniform":
            effective_weight_col = "uniform"
        if weighting_cfg.group_equalization and not effective_group_reweight:
            effective_group_reweight = "chain_snapshot"

    if bootstrap_cfg:
        if bootstrap_cfg.bootstrap_ci is not None:
            effective_bootstrap_ci = bootstrap_cfg.bootstrap_ci
        if bootstrap_cfg.bootstrap_b is not None:
            effective_bootstrap_b = bootstrap_cfg.bootstrap_b
        if bootstrap_cfg.bootstrap_seed is not None:
            effective_bootstrap_seed = bootstrap_cfg.bootstrap_seed
        if bootstrap_cfg.bootstrap_group:
            effective_bootstrap_group = bootstrap_cfg.bootstrap_group
        if bootstrap_cfg.allow_iid_bootstrap is not None:
            effective_allow_iid_bootstrap = bootstrap_cfg.allow_iid_bootstrap

    tdays_allowed = _parse_tdays_allowed(payload.tdays_allowed)
    tdays_allowed_arg = str(tdays_allowed) if tdays_allowed is not None else None

    if effective_group_reweight:
        group_reweight = str(effective_group_reweight).strip().lower()
        if group_reweight == "chain":
            group_reweight = "chain_snapshot"
        effective_group_reweight = group_reweight

    strict_args = bool(payload.strict_args)

    return {
        "config_schema_version": 2,
        "run_mode": "manual",
        "csv": str(dataset_path),
        "out_dir": str(out_dir),
        "target_col": payload.target_col,
        "week_col": effective_week_col,
        "ticker_col": effective_ticker_col,
        "weight_col": effective_weight_col,
        "weight_col_strategy": payload.weight_col_strategy,
        "foundation_tickers": effective_foundation_tickers,
        "foundation_weight": effective_foundation_weight,
        "ticker_intercepts": effective_ticker_intercepts,
        "ticker_x_interactions": effective_ticker_x_interactions,
        "ticker_min_support": effective_ticker_min_support,
        "ticker_min_support_interactions": effective_ticker_min_support_interactions,
        "train_tickers": effective_train_tickers,
        "tdays_allowed": tdays_allowed_arg,
        "asof_dow_allowed": payload.asof_dow_allowed,
        "features": payload.features,
        "categorical_features": payload.categorical_features,
        "add_interactions": payload.add_interactions,
        "calibrate": effective_calibrate,
        "c_grid": effective_c_grid,
        "train_decay_half_life_weeks": payload.train_decay_half_life_weeks,
        "calib_frac_of_train": payload.calib_frac_of_train,
        "fit_weight_renorm": payload.fit_weight_renorm,
        "test_weeks": effective_test_weeks,
        "val_windows": payload.val_windows,
        "val_window_weeks": payload.val_window_weeks,
        "val_split_mode": payload.val_split_mode,
        "val_weeks": payload.val_weeks,
        "n_bins": payload.n_bins,
        "eceq_bins": payload.eceq_bins,
        "metrics_top_tickers": payload.metrics_top_tickers,
        "random_state": effective_random_state,
        "selection_objective": effective_selection_objective,
        "strict_args": strict_args,
        "allow_defaults": bool(payload.allow_defaults),
        "allow_iid_bootstrap": bool(effective_allow_iid_bootstrap),
        "fallback_to_baseline_if_worse": payload.fallback_to_baseline_if_worse,
        "auto_drop_near_constant": payload.auto_drop_near_constant,
        "enable_x_abs_m": payload.enable_x_abs_m,
        "group_reweight": effective_group_reweight,
        "max_abs_logm": payload.max_abs_logm,
        "drop_prn_extremes": payload.drop_prn_extremes,
        "prn_eps": payload.prn_eps,
        "prn_below": payload.prn_below,
        "prn_above": payload.prn_above,
        "bootstrap_ci": effective_bootstrap_ci,
        "bootstrap_B": effective_bootstrap_b,
        "bootstrap_seed": effective_bootstrap_seed,
        "bootstrap_group": effective_bootstrap_group,
        "split": {
            "strategy": split_cfg.strategy if split_cfg else None,
            "window_mode": split_cfg.window_mode if split_cfg else None,
            "train_window_weeks": split_cfg.train_window_weeks if split_cfg else None,
            "validation_folds": split_cfg.validation_folds if split_cfg else None,
            "validation_window_weeks": split_cfg.validation_window_weeks if split_cfg else None,
            "test_window_weeks": split_cfg.test_window_weeks if split_cfg else None,
            "embargo_days": split_cfg.embargo_days if split_cfg else None,
        },
        "regularization": {
            "c_grid": reg_cfg.c_grid if reg_cfg else effective_c_grid,
            "calibration_method": reg_cfg.calibration_method if reg_cfg else effective_calibrate,
            "selection_objective": reg_cfg.selection_objective if reg_cfg else effective_selection_objective,
        },
        "model_structure": {
            "trading_universe_tickers": structure_cfg.trading_universe_tickers if structure_cfg else None,
            "train_tickers": structure_cfg.train_tickers if structure_cfg else effective_train_tickers,
            "foundation_tickers": structure_cfg.foundation_tickers if structure_cfg else effective_foundation_tickers,
            "foundation_weight": structure_cfg.foundation_weight if structure_cfg else effective_foundation_weight,
            "ticker_intercepts": structure_cfg.ticker_intercepts if structure_cfg else effective_ticker_intercepts,
            "ticker_x_interactions": (
                structure_cfg.ticker_x_interactions if structure_cfg else effective_ticker_x_interactions
            ),
            "ticker_min_support": structure_cfg.ticker_min_support if structure_cfg else effective_ticker_min_support,
            "ticker_min_support_interactions": (
                structure_cfg.ticker_min_support_interactions
                if structure_cfg else effective_ticker_min_support_interactions
            ),
        },
        "weighting": {
            "base_weight_source": weighting_cfg.base_weight_source if weighting_cfg else None,
            "grouping_key": weighting_cfg.grouping_key if weighting_cfg else None,
            "group_equalization": weighting_cfg.group_equalization if weighting_cfg else None,
            "renorm": weighting_cfg.renorm if weighting_cfg else None,
            "trading_universe_upweight": weighting_cfg.trading_universe_upweight if weighting_cfg else None,
            "ticker_balance_mode": weighting_cfg.ticker_balance_mode if weighting_cfg else None,
        },
        "bootstrap": {
            "bootstrap_ci": bootstrap_cfg.bootstrap_ci if bootstrap_cfg else effective_bootstrap_ci,
            "bootstrap_group": bootstrap_cfg.bootstrap_group if bootstrap_cfg else effective_bootstrap_group,
            "bootstrap_b": bootstrap_cfg.bootstrap_b if bootstrap_cfg else effective_bootstrap_b,
            "bootstrap_seed": bootstrap_cfg.bootstrap_seed if bootstrap_cfg else effective_bootstrap_seed,
            "ci_level": bootstrap_cfg.ci_level if bootstrap_cfg else None,
            "per_split_reporting": bootstrap_cfg.per_split_reporting if bootstrap_cfg else None,
            "per_fold_reporting": bootstrap_cfg.per_fold_reporting if bootstrap_cfg else None,
            "allow_iid_bootstrap": bool(effective_allow_iid_bootstrap),
        },
        "diagnostics": {
            "split_timeline": diagnostics_cfg.split_timeline if diagnostics_cfg else None,
            "per_fold_delta_chart": diagnostics_cfg.per_fold_delta_chart if diagnostics_cfg else None,
            "per_group_delta_distribution": (
                diagnostics_cfg.per_group_delta_distribution if diagnostics_cfg else None
            ),
        },
    }


def _write_temp_config(payload: Dict[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        json.dump(payload, handle, indent=2)
        return handle.name


def _prepare_manual_run(payload: CalibrateModelRunRequest) -> Tuple[List[str], Path, str]:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Calibration script not found at {SCRIPT_PATH}")

    # Validate payload against v2.0 CLI contract
    validation_errors = validate_payload(payload)
    if validation_errors:
        raise ValidationError("; ".join(validation_errors))

    if payload.run_mode == "auto_search":
        raise ValueError("run_mode='auto_search' is currently disabled. Use manual mode.")
    if payload.two_stage_mode:
        raise ValueError("two_stage_mode is currently disabled for calibration jobs.")

    dataset_path = _resolve_project_path(payload.csv)
    _ensure_dataset_path(dataset_path)

    default_name = f"calibration-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    requested_name = _sanitize_name(payload.out_name or "")
    final_name = requested_name or default_name
    out_dir = MODELS_DIR / final_name
    config_payload = _build_config_payload(payload, dataset_path=dataset_path, out_dir=out_dir)
    strict_args = bool(payload.strict_args)

    config_json_path = _write_temp_config(config_payload)

    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_PATH),
        "--csv",
        str(dataset_path),
        "--out-dir",
        str(out_dir),
        "--model-kind",
        "calibrate",
        "--config-json",
        config_json_path,
    ]
    if strict_args:
        cmd.append("--strict-args")

    return cmd, out_dir, config_json_path


def _prepare_auto_run(payload: AutoModelRunRequest) -> Tuple[List[str], Path, str]:
    if not AUTO_SCRIPT_PATH.exists():
        raise RuntimeError(f"Auto selector script not found at {AUTO_SCRIPT_PATH}")

    # Resolve base configuration for auto-search
    base_config = payload.base_config
    if base_config is None:
        # Legacy fallback: build a minimal base config from old fields
        base_config = CalibrateModelRunRequest(
            csv=payload.csv,
            out_name=payload.run_name,
            tdays_allowed=payload.tdays_allowed,
            asof_dow_allowed=payload.asof_dow_allowed,
            foundation_tickers=payload.foundation_tickers,
            foundation_weight=payload.foundation_weight,
            bootstrap_ci=payload.bootstrap_ci,
            bootstrap_b=payload.bootstrap_b,
            bootstrap_seed=payload.bootstrap_seed,
            bootstrap_group=payload.bootstrap_group,
        )

    dataset_path = _resolve_project_path(base_config.csv)
    _ensure_dataset_path(dataset_path)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_auto_run_name(payload)
    run_name = payload.run_name or _default_auto_run_name()
    out_dir = MODELS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config_payload = _build_config_payload(
        base_config,
        dataset_path=dataset_path,
        out_dir=out_dir,
    )

    search_payload = payload.search.dict() if payload.search else {}
    auto_config_payload = {
        "base_config": base_config_payload,
        "search": search_payload,
        "run_name": run_name,
        "seed": payload.seed,
        "parallel": payload.parallel,
    }

    config_json_path = _write_temp_config(auto_config_payload)

    cmd: List[str] = [
        sys.executable,
        str(AUTO_SCRIPT_PATH),
        "--config-json",
        config_json_path,
        "--out-dir",
        str(out_dir),
    ]
    _add_value(cmd, "--calibrator-script", str(SCRIPT_PATH))
    _add_value(cmd, "--seed", payload.seed)
    _add_value(cmd, "--parallel", payload.parallel)
    if payload.search and payload.search.max_trials is not None:
        _add_value(cmd, "--max-trials", payload.search.max_trials)

    return cmd, out_dir, config_json_path


def _build_calibration_run_response(
    out_dir: Path,
    *,
    cmd: List[str],
    duration_s: float,
    stdout: str,
    stderr: str,
    ok: bool,
    payload: Optional[CalibrateModelRunRequest] = None,
) -> CalibrateModelRunResponse:
    effective_dir = _resolve_effective_model_dir(out_dir)
    is_auto_run = _is_auto_run_dir(out_dir)
    files = []
    if effective_dir.exists():
        files = sorted([item.name for item in effective_dir.iterdir() if item.is_file()])

    metrics_summary = _build_metrics_summary(effective_dir / "metrics.csv")
    if not metrics_summary:
        metrics_summary = None

    split_row_counts, split_group_counts = _build_split_counts(effective_dir)

    features = _read_features(effective_dir)
    two_stage_metrics = _load_two_stage_metrics(effective_dir)

    is_two_stage = bool(payload.two_stage_mode) if payload else False
    two_stage_meta_exists = (effective_dir / "two_stage_metadata.json").exists()

    if is_two_stage and two_stage_meta_exists:
        stage1_equation_spec = _build_model_equation_spec(effective_dir)
        two_stage_equation_spec = _build_two_stage_equation_spec(effective_dir)
        combined_p_hat_equation_spec = _build_combined_p_hat_equation_spec(effective_dir)
        stage1_equation = stage1_equation_spec.get("compact_latex") if stage1_equation_spec else None
        two_stage_equation = two_stage_equation_spec.get("compact_latex") if two_stage_equation_spec else None
        combined_p_hat_equation = (
            combined_p_hat_equation_spec.get("compact_latex")
            if combined_p_hat_equation_spec else None
        )
        model_equation = None
        model_equation_spec = None
    else:
        model_equation_spec = _build_model_equation_spec(effective_dir)
        model_equation = model_equation_spec.get("compact_latex") if model_equation_spec else None
        stage1_equation = None
        stage1_equation_spec = None
        two_stage_equation = None
        two_stage_equation_spec = None
        combined_p_hat_equation = None
        combined_p_hat_equation_spec = None

    artifact_manifest: List[Dict[str, str]] = []
    if is_auto_run:
        for entry in _collect_curated_file_entries(out_dir):
            path = entry["path"]
            rel = str(entry.get("relative_path") or path.name)
            suffix = path.suffix.lower()
            if suffix == ".json":
                artifact_type = "json"
            elif suffix == ".csv":
                artifact_type = "csv"
            elif suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                artifact_type = "plot"
            elif suffix in {".joblib", ".pkl"}:
                artifact_type = "model"
            else:
                artifact_type = "artifact"
            artifact_manifest.append(
                {
                    "name": path.name,
                    "type": artifact_type,
                    "path": f"{out_dir.relative_to(BASE_DIR)}/{rel}",
                    "relative_path": rel,
                    "section": str(entry.get("section") or "legacy_root"),
                }
            )
    else:
        for name in files:
            suffix = Path(name).suffix.lower()
            if suffix == ".json":
                artifact_type = "json"
            elif suffix == ".csv":
                artifact_type = "csv"
            elif suffix in {".png", ".jpg", ".jpeg", ".svg"}:
                artifact_type = "plot"
            elif suffix in {".joblib", ".pkl"}:
                artifact_type = "model"
            else:
                artifact_type = "artifact"
            artifact_manifest.append({"name": name, "type": artifact_type, "path": f"{out_dir.relative_to(BASE_DIR)}/{name}"})

    diagnostics_available = {
        "split_timeline": (effective_dir / "split_timeline.json").exists(),
        "per_fold_delta_chart": (effective_dir / "fold_deltas.csv").exists(),
        "per_group_delta_distribution": (effective_dir / "group_delta_distribution.csv").exists(),
    }
    run_warnings: List[str] = []
    metadata = _load_json(effective_dir / "metadata.json")
    if isinstance(metadata, dict):
        ignored = metadata.get("unsupported_controls_ignored")
        if isinstance(ignored, dict) and ignored:
            run_warnings.append(
                "Some controls were ignored by the trainer: "
                + ", ".join(sorted(str(k) for k in ignored.keys()))
            )
        meta_warnings = metadata.get("warnings")
        if isinstance(meta_warnings, list):
            run_warnings.extend([str(w) for w in meta_warnings if str(w).strip()])

    return CalibrateModelRunResponse(
        ok=ok,
        out_dir=str(out_dir.relative_to(BASE_DIR)),
        files=files,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
        command=cmd,
        metrics_summary=metrics_summary,
        split_row_counts=split_row_counts or None,
        split_group_counts=split_group_counts or None,
        auto_out_dir=None,
        features=features,
        model_equation=model_equation,
        model_equation_spec=model_equation_spec,
        two_stage_metrics=two_stage_metrics,
        is_two_stage=is_two_stage and two_stage_meta_exists,
        stage1_equation=stage1_equation,
        stage1_equation_spec=stage1_equation_spec,
        two_stage_equation=two_stage_equation,
        two_stage_equation_spec=two_stage_equation_spec,
        combined_p_hat_equation=combined_p_hat_equation,
        combined_p_hat_equation_spec=combined_p_hat_equation_spec,
        artifact_manifest=artifact_manifest,
        diagnostics_available=diagnostics_available,
        warnings=run_warnings or None,
    )


def run_calibration(payload: CalibrateModelRunRequest) -> CalibrateModelRunResponse:
    cmd, out_dir, config_json_path = _prepare_manual_run(payload)

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=str(BASE_DIR),
            env=_build_python_env(),
        )
    finally:
        try:
            Path(config_json_path).unlink(missing_ok=True)
        except Exception:
            pass
    duration_s = round(time.monotonic() - start, 3)

    return _build_calibration_run_response(
        out_dir,
        cmd=cmd,
        duration_s=duration_s,
        stdout=result.stdout,
        stderr=result.stderr,
        ok=result.returncode == 0,
        payload=payload,
    )


def run_auto_model_selection(payload: AutoModelRunRequest) -> CalibrateModelRunResponse:
    if not _try_acquire_auto_run_lock():
        raise RuntimeError("Another job is already running (auto calibration).")

    try:
        cmd, out_dir, config_json_path = _prepare_auto_run(payload)

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                cwd=str(BASE_DIR),
                env=_build_python_env(),
            )
        finally:
            try:
                Path(config_json_path).unlink(missing_ok=True)
            except Exception:
                pass
        duration_s = round(time.monotonic() - start, 3)

        if result.returncode != 0:
            try:
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass

        return _build_calibration_run_response(
            out_dir,
            cmd=cmd,
            duration_s=duration_s,
            stdout=result.stdout,
            stderr=result.stderr,
            ok=result.returncode == 0,
            payload=None,
        )
    finally:
        _release_auto_run_lock()


class CalibrationJob:
    def __init__(
        self,
        job_id: str,
        mode: str,
        payload: CalibrateModelRunRequest | AutoModelRunRequest,
    ) -> None:
        self.job_id = job_id
        self.mode = mode
        self.payload = payload
        self.status = "queued"
        self.result: Optional[CalibrateModelRunResponse] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._cancel_requested = False
        self._proc: Optional[subprocess.Popen] = None
        self._proc_handle: Optional[ManagedProcessHandle] = None
        self._out_dir: Optional[Path] = None
        self._config_json_path: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._auto_lock_held = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        if self.status in {"finished", "failed", "cancelled"}:
            return
        proc = self._proc
        if proc and proc.poll() is not None:
            return
        self._cancel_requested = True
        handle = self._proc_handle
        if handle and proc and proc.poll() is None:
            terminate_managed_process(handle, term_timeout_s=3.0, kill_timeout_s=3.0)
            return
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def targets_model_dir(self, model_id: str) -> bool:
        if self._out_dir is not None and self._out_dir.name == model_id:
            return True
        if self.mode == "auto" and isinstance(self.payload, AutoModelRunRequest):
            run_name = str(self.payload.run_name or "").strip()
            return bool(run_name and run_name == model_id)
        if self.mode != "auto" and isinstance(self.payload, CalibrateModelRunRequest):
            out_name = str(self.payload.out_name or "").strip()
            return bool(out_name and _sanitize_name(out_name) == model_id)
        return False

    def to_status(self) -> CalibrationJobStatus:
        progress = None
        if self.mode == "auto" and isinstance(self.payload, AutoModelRunRequest):
            if self.payload.run_name:
                progress = _read_progress(MODELS_DIR / self.payload.run_name)
        return CalibrationJobStatus(
            job_id=self.job_id,
            status=self.status,
            mode="auto" if self.mode == "auto" else "manual",
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
            progress=progress,
        )

    def _cleanup_run_dir(self) -> None:
        if not self._out_dir:
            return
        clear_runtime_file(self._out_dir)
        try:
            if self._out_dir.exists():
                shutil.rmtree(self._out_dir, ignore_errors=True)
        except Exception:
            pass

    def _cleanup_config(self) -> None:
        if not self._config_json_path:
            return
        try:
            Path(self._config_json_path).unlink(missing_ok=True)
        except Exception:
            pass

    def _run(self) -> None:
        try:
            try:
                if self.mode == "auto":
                    cmd, out_dir, config_json_path = _prepare_auto_run(
                        self.payload  # type: ignore[arg-type]
                    )
                else:
                    cmd, out_dir, config_json_path = _prepare_manual_run(
                        self.payload  # type: ignore[arg-type]
                    )
            except Exception as exc:
                self.status = "failed"
                self.error = str(exc)
                if self.mode == "auto" and isinstance(self.payload, AutoModelRunRequest):
                    run_name = str(self.payload.run_name or "").strip()
                    if run_name:
                        self._out_dir = MODELS_DIR / run_name
                        self._cleanup_run_dir()
                self.finished_at = datetime.utcnow()
                return

            self._out_dir = out_dir
            self._config_json_path = config_json_path
            self.started_at = datetime.utcnow()
            self.status = "running"

            if self._cancel_requested:
                self.status = "cancelled"
                self.error = "Calibration run cancelled."
                self._cleanup_run_dir()
                self._cleanup_config()
                self.finished_at = datetime.utcnow()
                return

            env = _build_python_env()
            env["PYTHONUNBUFFERED"] = "1"
            start = time.monotonic()
            stdout_lines: List[str] = []
            stderr_lines: List[str] = []
            try:
                handle = spawn_managed_process(
                    cmd,
                    job_id=self.job_id,
                    service=f"calibrate_{self.mode}",
                    run_dir=self._out_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    cwd=str(BASE_DIR),
                    env=env,
                )
                proc = handle.process
                if proc is None:
                    raise RuntimeError("Failed to start calibration process.")
                self._proc_handle = handle
                self._proc = proc

                lines_queue: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()

                def _reader(stream, kind: str) -> None:
                    for line in iter(stream.readline, ""):
                        lines_queue.put((kind, line))
                    lines_queue.put((kind, None))

                threads = [
                    threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True),
                    threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True),
                ]
                for thread in threads:
                    thread.start()

                done_streams = 0
                cancel_signal_sent = False
                while done_streams < 2:
                    kind, line = lines_queue.get()
                    if line is None:
                        done_streams += 1
                        continue
                    if kind == "stdout":
                        stdout_lines.append(line)
                    else:
                        stderr_lines.append(line)
                    if self._cancel_requested and proc.poll() is None and not cancel_signal_sent:
                        terminate_managed_process(handle, term_timeout_s=3.0, kill_timeout_s=3.0)
                        cancel_signal_sent = True

                for thread in threads:
                    thread.join()

                return_code = proc.wait()
                duration_s = round(time.monotonic() - start, 3)
            except Exception as exc:
                self._cleanup_config()
                self.status = "failed"
                self.error = str(exc)
                if self.mode == "auto":
                    self._cleanup_run_dir()
                self.finished_at = datetime.utcnow()
                return

            self._cleanup_config()

            if self._cancel_requested:
                self.status = "cancelled"
                self.error = "Calibration run cancelled."
                self.result = None
                self._cleanup_run_dir()
            else:
                ok = return_code == 0
                if self.mode == "auto" and not ok:
                    self._cleanup_run_dir()
                self.result = _build_calibration_run_response(
                    out_dir,
                    cmd=cmd,
                    duration_s=duration_s,
                    stdout="".join(stdout_lines),
                    stderr="".join(stderr_lines),
                    ok=ok,
                    payload=self.payload if self.mode != "auto" else None,  # type: ignore[arg-type]
                )
                if self.result and self.result.ok:
                    self.status = "finished"
                else:
                    self.status = "failed"
                    if self.result:
                        stderr_text = (self.result.stderr or "").strip()
                        stdout_text = (self.result.stdout or "").strip()
                        self.error = stderr_text or stdout_text or "Calibration run failed."

            self.finished_at = datetime.utcnow()
        finally:
            if self._out_dir:
                clear_runtime_file(self._out_dir)
            self._proc = None
            self._proc_handle = None
            if self._auto_lock_held:
                _release_auto_run_lock()
                self._auto_lock_held = False


class CalibrationJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, CalibrationJob] = {}
        self._lock = threading.Lock()

    def start_job(
        self,
        payload: CalibrateModelRunRequest | AutoModelRunRequest,
        *,
        mode: str,
    ) -> str:
        auto_lock_held = False
        if mode == "auto":
            if not _try_acquire_auto_run_lock():
                raise RuntimeError("Another job is already running (auto calibration).")
            auto_lock_held = True

        job_id = uuid4().hex
        job = CalibrationJob(job_id, mode, payload)
        if auto_lock_held:
            job._auto_lock_held = True
        with self._lock:
            self._jobs[job_id] = job
        try:
            job.start()
        except Exception:
            with self._lock:
                self._jobs.pop(job_id, None)
            if auto_lock_held:
                _release_auto_run_lock()
            raise
        return job_id

    def get_status(self, job_id: str) -> CalibrationJobStatus:
        job = self._get_job(job_id)
        return job.to_status()

    def list_jobs(self) -> List[CalibrationJobStatus]:
        with self._lock:
            return [job.to_status() for job in self._jobs.values()]

    def cancel_job(self, job_id: str) -> CalibrationJobStatus:
        job = self._get_job(job_id)
        job.cancel()
        return job.to_status()

    def cancel_jobs_for_model(self, model_id: str) -> List[str]:
        with self._lock:
            matching = [
                job
                for job in self._jobs.values()
                if job.targets_model_dir(model_id) and job.status in {"queued", "running"}
            ]
        for job in matching:
            job.cancel()
        return [job.job_id for job in matching]

    def _get_job(self, job_id: str) -> CalibrationJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


CALIBRATION_JOB_MANAGER = CalibrationJobManager()


def start_calibration_job(payload: CalibrateModelRunRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    return CALIBRATION_JOB_MANAGER.start_job(payload, mode="manual")


def start_auto_calibration_job(payload: AutoModelRunRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    _ensure_auto_run_name(payload)
    return CALIBRATION_JOB_MANAGER.start_job(payload, mode="auto")


def get_calibration_job(job_id: str) -> CalibrationJobStatus:
    try:
        status = CALIBRATION_JOB_MANAGER.get_status(job_id)
        LOGGER.info(
            "job_lookup_manager_hit job_id=%s status=%s mode=%s",
            job_id,
            status.status,
            status.mode,
        )
        return status
    except KeyError:
        runtime_status = _runtime_backed_status(job_id)
        if runtime_status is not None:
            return runtime_status
        raise


def list_calibration_jobs() -> List[CalibrationJobStatus]:
    statuses = CALIBRATION_JOB_MANAGER.list_jobs()
    seen_job_ids = {status.job_id for status in statuses}
    for run_dir, payload in _iter_calibration_runtime_dirs():
        runtime_job_id = payload.get("job_id")
        if runtime_job_id is None:
            continue
        runtime_job_id_str = str(runtime_job_id)
        if runtime_job_id_str in seen_job_ids:
            continue
        pid = payload.get("pid")
        if not is_process_alive(pid):
            LOGGER.info(
                "job_lookup_runtime_stale job_id=%s path=%s pid=%s",
                runtime_job_id_str,
                run_dir,
                pid,
            )
            clear_runtime_file(run_dir)
            continue
        LOGGER.info(
            "job_lookup_runtime_hit job_id=%s path=%s pid=%s",
            runtime_job_id_str,
            run_dir,
            pid,
        )
        statuses.append(
            _build_runtime_backed_job_status(
                runtime_job_id_str,
                payload,
                run_dir,
                status="running",
            )
        )
        seen_job_ids.add(runtime_job_id_str)
    return statuses


def cancel_calibration_job(job_id: str) -> CalibrationJobStatus:
    try:
        return CALIBRATION_JOB_MANAGER.cancel_job(job_id)
    except KeyError:
        runtime = _find_runtime_job(job_id)
        if runtime is None:
            raise
        run_dir, payload = runtime
        handle = managed_handle_from_runtime_payload(run_dir, payload)
        if handle is None:
            clear_runtime_file(run_dir)
            raise KeyError(job_id)

        if not is_process_alive(handle.pid):
            LOGGER.info(
                "job_lookup_runtime_stale job_id=%s path=%s pid=%s",
                job_id,
                run_dir,
                handle.pid,
            )
            clear_runtime_file(run_dir)
            raise KeyError(job_id)

        result = terminate_managed_process(
            handle,
            term_timeout_s=DELETE_TERM_TIMEOUT_S,
            kill_timeout_s=DELETE_KILL_TIMEOUT_S,
        )
        LOGGER.info(
            "job_cancel_runtime_fallback job_id=%s path=%s pid=%s pgid=%s ok=%s reason=%s",
            job_id,
            run_dir,
            result.pid,
            result.pgid,
            result.ok,
            result.reason,
        )
        if not result.ok:
            raise RuntimeError(
                f"Failed to cancel calibration job '{job_id}' via runtime fallback: {result.reason}"
            )

        clear_runtime_file(run_dir)
        return _build_runtime_backed_job_status(
            job_id,
            payload,
            run_dir,
            status="cancelled",
            finished_at=datetime.now(timezone.utc),
        )


# v2.0 Model Training
SCRIPT_V2_PATH = BASE_DIR / "src" / "scripts" / "03-calibrate-logit-model-v2.0.py"


def run_calibration_v2(payload) -> CalibrateModelRunResponse:
    """
    Run v2.0 model training with PM+options integration.

    Args:
        payload: TrainModelV2Request with training configuration

    Returns:
        CalibrateModelRunResponse with results, metrics, and edge predictions
    """
    from app.models.calibrate_models import TrainModelV2Request

    if not isinstance(payload, TrainModelV2Request):
        raise ValueError("Invalid payload type for v2.0 training")

    if not SCRIPT_V2_PATH.exists():
        raise RuntimeError(f"v2.0 calibration script not found at {SCRIPT_V2_PATH}")

    prn_dataset_path = _resolve_project_path(payload.prn_csv)
    _ensure_dataset_path(prn_dataset_path)

    if payload.pm_csv:
        pm_dataset_path = _resolve_project_path(payload.pm_csv)
        if not pm_dataset_path.exists():
            raise ValueError(f"Polymarket dataset not found: {pm_dataset_path}")
    else:
        pm_dataset_path = None

    out_dir = _resolve_project_path(payload.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_V2_PATH),
        "--csv",
        str(prn_dataset_path),
        "--out-dir",
        str(out_dir),
        "--training-mode",
        payload.training_mode,
        "--feature-sources",
        payload.feature_sources,
        "--pm-overlap-window",
        payload.overlap_window,
    ]

    # Add two-stage mode flags for backward compatibility
    if payload.training_mode == "two_stage" and pm_dataset_path:
        cmd.append("--two-stage-mode")
        cmd.extend(["--two-stage-pm-csv", str(pm_dataset_path)])
        if payload.label_col:
            cmd.extend(["--two-stage-label-col", payload.label_col])

    # Add edge computation flag
    if payload.compute_edge:
        cmd.append("--compute-edge")

    # Add optional overrides
    if payload.numeric_features:
        cmd.extend(["--numeric-features", ",".join(payload.numeric_features)])
    if payload.pm_features:
        cmd.extend(["--pm-features", ",".join(payload.pm_features)])

    # Add standard calibration arguments
    _add_value(cmd, "--test-weeks", payload.test_weeks)
    _add_value(cmd, "--random-state", payload.random_state)
    _add_value(cmd, "--calibrate", payload.calibrate)

    start = time.monotonic()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(BASE_DIR),
        env=_build_python_env(),
    )
    duration_s = round(time.monotonic() - start, 3)

    files = []
    if out_dir.exists():
        files = sorted([item.name for item in out_dir.iterdir() if item.is_file()])

    metrics_summary = _build_metrics_summary(out_dir / "metrics.csv")
    if not metrics_summary:
        # Try two_stage_metrics.csv
        metrics_summary = _build_metrics_summary(out_dir / "two_stage_metrics.csv")
    if not metrics_summary:
        metrics_summary = None

    features = _read_features(out_dir)
    model_equation_spec = _build_model_equation_spec(out_dir)
    model_equation = model_equation_spec.get("compact_latex") if model_equation_spec else None
    two_stage_metrics = _load_two_stage_metrics(out_dir)
    is_two_stage = (out_dir / "two_stage_metadata.json").exists()
    stage1_equation_spec = _build_model_equation_spec(out_dir) if is_two_stage else None
    two_stage_equation_spec = _build_two_stage_equation_spec(out_dir) if is_two_stage else None
    combined_p_hat_equation_spec = _build_combined_p_hat_equation_spec(out_dir) if is_two_stage else None
    stage1_equation = stage1_equation_spec.get("compact_latex") if stage1_equation_spec else None
    two_stage_equation = two_stage_equation_spec.get("compact_latex") if two_stage_equation_spec else None
    combined_p_hat_equation = (
        combined_p_hat_equation_spec.get("compact_latex")
        if combined_p_hat_equation_spec else None
    )
    if is_two_stage:
        model_equation = None
        model_equation_spec = None

    return CalibrateModelRunResponse(
        ok=result.returncode == 0,
        out_dir=str(out_dir.relative_to(BASE_DIR)),
        files=files,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=duration_s,
        command=cmd,
        metrics_summary=metrics_summary,
        auto_out_dir=None,
        features=features,
        model_equation=model_equation,
        model_equation_spec=model_equation_spec,
        two_stage_metrics=two_stage_metrics,
        is_two_stage=is_two_stage,
        stage1_equation=stage1_equation,
        stage1_equation_spec=stage1_equation_spec,
        two_stage_equation=two_stage_equation,
        two_stage_equation_spec=two_stage_equation_spec,
        combined_p_hat_equation=combined_p_hat_equation,
        combined_p_hat_equation_spec=combined_p_hat_equation_spec,
    )


def get_edge_predictions(model_id: str):
    """
    Retrieve edge predictions for a trained v2.0 model.

    Args:
        model_id: Model identifier (folder name)

    Returns:
        EdgePredictionsResponse with edge estimates and confidence intervals
    """
    from app.models.calibrate_models import EdgePrediction, EdgePredictionsResponse

    model_dir = MODELS_DIR / model_id
    if not model_dir.exists():
        raise KeyError(f"Model directory not found: {model_id}")

    edge_file = model_dir / "edge_predictions.csv"
    if not edge_file.exists():
        raise KeyError(f"No edge predictions found for model: {model_id}")

    try:
        df = pd.read_csv(edge_file)
    except Exception as exc:
        raise ValueError(f"Failed to read edge predictions: {exc}") from exc

    predictions = []
    for _, row in df.iterrows():
        pred = EdgePrediction(
            ticker=str(row.get("ticker", "")),
            threshold=float(row.get("threshold", 0.0)),
            expiry_date=str(row.get("expiry_date", "")),
            snapshot_date=str(row.get("snapshot_date", "")),
            p_base=float(row.get("p_base", 0.0)),
            p_pm=float(row["p_pm"]) if pd.notna(row.get("p_pm")) else None,
            p_final=float(row.get("p_final", 0.0)),
            edge=float(row["edge"]) if pd.notna(row.get("edge")) else None,
            edge_lower=float(row["edge_lower"]) if pd.notna(row.get("edge_lower")) else None,
            edge_upper=float(row["edge_upper"]) if pd.notna(row.get("edge_upper")) else None,
        )
        predictions.append(pred)

    return EdgePredictionsResponse(
        model_id=model_id,
        count=len(predictions),
        predictions=predictions,
    )
