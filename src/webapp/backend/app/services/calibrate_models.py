from __future__ import annotations

import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from app.models.calibrate_models import (
    CalibrateModelRunRequest,
    CalibrateModelRunResponse,
    DatasetFileSummary,
    DatasetListResponse,
    ModelListResponse,
    ModelRunSummary,
)

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "2-calibrate-logit-model-v1.5.py"
DATASET_DIR_PRIMARY = BASE_DIR / "src" / "data" / "raw" / "option-chains"
DATASET_DIR_FALLBACK = BASE_DIR / "src" / "data" / "raw" / "option-chain"
MODELS_DIR = BASE_DIR / "src" / "data" / "models"


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.resolve()
    try:
        path.relative_to(BASE_DIR)
    except ValueError as exc:
        raise ValueError("Path must be inside the project root.") from exc
    return path


def _is_valid_dataset(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() != ".csv":
        return False
    lowered = path.name.lower()
    if "drop" in lowered:
        return False
    return True


def _select_dataset_dirs() -> List[Path]:
    dirs: List[Path] = []
    if DATASET_DIR_PRIMARY.exists():
        dirs.append(DATASET_DIR_PRIMARY)
    if DATASET_DIR_FALLBACK.exists():
        dirs.append(DATASET_DIR_FALLBACK)
    return dirs


def list_datasets() -> DatasetListResponse:
    dataset_dirs = _select_dataset_dirs()
    if not dataset_dirs:
        return DatasetListResponse(
            base_dir=str(DATASET_DIR_PRIMARY.relative_to(BASE_DIR)),
            datasets=[],
        )

    datasets: List[DatasetFileSummary] = []
    for dataset_dir in dataset_dirs:
        for item in dataset_dir.rglob("*.csv"):
            if not _is_valid_dataset(item):
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


def list_models() -> ModelListResponse:
    if not MODELS_DIR.exists():
        return ModelListResponse(base_dir=str(MODELS_DIR.relative_to(BASE_DIR)), models=[])

    models: List[ModelRunSummary] = []
    for item in MODELS_DIR.iterdir():
        if not item.is_dir():
            continue
        metadata_path = item / "metadata.json"
        metrics_path = item / "metrics.csv"
        mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()
        models.append(
            ModelRunSummary(
                id=item.name,
                path=str(item.relative_to(BASE_DIR)),
                last_modified=mtime,
                has_metadata=metadata_path.exists(),
                has_metrics=metrics_path.exists(),
            )
        )
    models.sort(key=lambda m: m.last_modified or "", reverse=True)
    return ModelListResponse(
        base_dir=str(MODELS_DIR.relative_to(BASE_DIR)),
        models=models,
    )


def _model_summary_from_path(item: Path) -> ModelRunSummary:
    metadata_path = item / "metadata.json"
    metrics_path = item / "metrics.csv"
    mtime = datetime.fromtimestamp(item.stat().st_mtime, tz=timezone.utc).isoformat()
    return ModelRunSummary(
        id=item.name,
        path=str(item.relative_to(BASE_DIR)),
        last_modified=mtime,
        has_metadata=metadata_path.exists(),
        has_metrics=metrics_path.exists(),
    )


def delete_model(model_id: str) -> ModelRunSummary:
    target = (MODELS_DIR / model_id).resolve()
    try:
        target.relative_to(MODELS_DIR.resolve())
    except Exception:
        raise KeyError(model_id)
    if not target.exists() or not target.is_dir():
        raise KeyError(model_id)
    summary = _model_summary_from_path(target)
    shutil.rmtree(target)
    return summary


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


def _sanitize_name(value: str) -> str:
    cleaned = "".join(ch for ch in value.strip() if ch.isalnum() or ch in ("-", "_", "."))
    return cleaned.strip(".-_")


def run_calibration(payload: CalibrateModelRunRequest) -> CalibrateModelRunResponse:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Calibration script not found at {SCRIPT_PATH}")

    dataset_path = _resolve_project_path(payload.csv)
    valid_dirs = _select_dataset_dirs()
    if not valid_dirs:
        raise ValueError("No dataset directories found under src/data/raw.")
    if not any(dataset_path.is_relative_to(d) for d in valid_dirs):
        raise ValueError("Dataset must be in src/data/raw/option-chains.")
    if not _is_valid_dataset(dataset_path):
        raise ValueError("Dataset must be a CSV and cannot be a drops file.")

    default_name = f"calibration-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    requested_name = _sanitize_name(payload.out_name or "")
    final_name = requested_name or default_name
    out_dir = MODELS_DIR / final_name

    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_PATH),
        "--csv",
        str(dataset_path),
        "--out-dir",
        str(out_dir),
    ]

    _add_value(cmd, "--target-col", payload.target_col)
    _add_value(cmd, "--week-col", payload.week_col)
    _add_value(cmd, "--ticker-col", payload.ticker_col)
    _add_value(cmd, "--weight-col", payload.weight_col)
    _add_value(cmd, "--features", payload.features)
    _add_value(cmd, "--categorical-features", payload.categorical_features)
    _add_flag(cmd, "--add-interactions", payload.add_interactions)
    _add_value(cmd, "--calibrate", payload.calibrate)
    _add_value(cmd, "--C-grid", payload.c_grid)
    _add_value(cmd, "--train-decay-half-life-weeks", payload.train_decay_half_life_weeks)
    _add_value(cmd, "--calib-frac-of-train", payload.calib_frac_of_train)
    _add_value(cmd, "--fit-weight-renorm", payload.fit_weight_renorm)
    _add_value(cmd, "--test-weeks", payload.test_weeks)
    _add_value(cmd, "--val-windows", payload.val_windows)
    _add_value(cmd, "--val-window-weeks", payload.val_window_weeks)
    _add_value(cmd, "--n-bins", payload.n_bins)
    _add_value(cmd, "--random-state", payload.random_state)

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    duration_s = round(time.monotonic() - start, 3)

    return CalibrateModelRunResponse(
        ok=result.returncode == 0,
        out_dir=str(out_dir.relative_to(BASE_DIR)),
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=duration_s,
        command=cmd,
    )
