from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd

from app.models.calibrate_models import (
    AutoModelRunRequest,
    CalibrateModelRunRequest,
    CalibrateModelRunResponse,
    CalibrationJobStatus,
    DatasetFileSummary,
    DatasetListResponse,
    ModelFileContentResponse,
    ModelFilesListResponse,
    ModelFileSummary,
    RegimePreviewRequest,
    RegimePreviewResponse,
    ModelDetailResponse,
    ModelListResponse,
    ModelRunSummary,
    SplitMetricSummary,
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
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "2-calibrate-logit-model-v1.5.py"
AUTO_SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "2-auto-calibrate-logit-model.py"
CALIBRATE_DATASET_DIRS = _unique_dirs(
    [
        BASE_DIR / "src" / "data" / "raw" / "option-chains",
        BASE_DIR / "data" / "raw" / "option-chains",
        BASE_DIR / "src" / "data" / "raw" / "option-chain",
        BASE_DIR / "data" / "raw" / "option-chain",
    ],
)
MODELS_DIR = BASE_DIR / "src" / "data" / "models"


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


def _find_calibrate_dataset_base(path: Path) -> Path:
    for base in CALIBRATE_DATASET_DIRS:
        try:
            path.relative_to(base)
            return base
        except ValueError:
            continue
    raise ValueError(
        "Dataset must be under src/data/raw/option-chains or src/data/raw/option-chain "
        "or their data/raw equivalents."
    )


def _select_dataset_dirs() -> List[Path]:
    existing_dirs = [path for path in CALIBRATE_DATASET_DIRS if path.exists()]
    return existing_dirs or [CALIBRATE_DATASET_DIRS[0]]


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
IMPORTANT_FILES = [
    "metadata.json",
    "metrics.csv",
    "metrics_summary.json",
    "feature_manifest.json",
    "best_config.json",
    "best_model_report.md",
    "leaderboard.csv",
    "reliability_bins.csv",
    "rolling_summary.csv",
    "rolling_windows.csv",
    "metrics_groups.csv",
]
MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB max for viewing


def list_model_files(model_id: str) -> ModelFilesListResponse:
    """List files in a model directory that can be viewed."""
    target = (MODELS_DIR / model_id).resolve()
    try:
        target.relative_to(MODELS_DIR.resolve())
    except Exception:
        raise KeyError(model_id)
    if not target.exists() or not target.is_dir():
        raise KeyError(model_id)

    files: List[ModelFileSummary] = []
    for fname in IMPORTANT_FILES:
        fpath = target / fname
        if fpath.exists() and fpath.is_file():
            size = fpath.stat().st_size
            is_viewable = size <= MAX_FILE_SIZE_BYTES
            files.append(ModelFileSummary(name=fname, size_bytes=size, is_viewable=is_viewable))

    return ModelFilesListResponse(model_id=model_id, files=files)


def get_model_file_content(model_id: str, filename: str) -> ModelFileContentResponse:
    """Get the content of a file in a model directory."""
    target = (MODELS_DIR / model_id).resolve()
    try:
        target.relative_to(MODELS_DIR.resolve())
    except Exception:
        raise KeyError(model_id)
    if not target.exists() or not target.is_dir():
        raise KeyError(model_id)

    # Only allow known important files
    if filename not in IMPORTANT_FILES:
        raise ValueError(f"File '{filename}' is not in the list of viewable files.")

    file_path = target / filename
    if not file_path.exists() or not file_path.is_file():
        raise KeyError(f"{model_id}/{filename}")

    size = file_path.stat().st_size
    truncated = size > MAX_FILE_SIZE_BYTES

    # Determine content type
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        content_type = "json"
    elif suffix == ".csv":
        content_type = "csv"
    elif suffix == ".md":
        content_type = "markdown"
    else:
        content_type = "text"

    # Read content (possibly truncated)
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        if truncated:
            content = f.read(MAX_FILE_SIZE_BYTES)
        else:
            content = f.read()

    return ModelFileContentResponse(
        model_id=model_id,
        filename=filename,
        content=content,
        content_type=content_type,
        truncated=truncated,
    )


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


def _ensure_dataset_path(dataset_path: Path) -> None:
    valid_dirs = _select_dataset_dirs()
    if not valid_dirs:
        raise ValueError("No dataset directories found under src/data/raw.")
    try:
        _find_calibrate_dataset_base(dataset_path)
    except ValueError as exc:
        raise ValueError(
            "Dataset must be under src/data/raw/option-chains or src/data/raw/option-chain "
            "or their data/raw equivalents."
        ) from exc
    if not _is_valid_dataset(dataset_path):
        raise ValueError("Dataset must be a CSV and cannot be a drops file.")


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

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_ABBREV_TO_INDEX = {name[:3].lower(): idx for idx, name in enumerate(DAY_NAMES)}


def _parse_tdays_allowed(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return None
    parsed: List[int] = []
    for token in tokens:
        try:
            tdays = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid T_days value '{token}'.") from exc
        if tdays < 0:
            raise ValueError(f"T_days must be >= 0 (got {tdays}).")
        if tdays not in parsed:
            parsed.append(tdays)
    return parsed


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


def _build_model_equation(out_dir: Path) -> Optional[str]:
    metadata_path = out_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        with metadata_path.open() as f:
            metadata = json.load(f)

        coefficients = metadata.get("coefficients")
        intercept = metadata.get("intercept")
        features = metadata.get("features")

        if not coefficients or not features:
            return None

        terms: List[str] = []
        if intercept is not None:
            terms.append(f"{intercept:.4f}")

        placeholder_added = False
        for feat, coef in zip(features, coefficients):
            feat_name = str(feat)
            if feat_name.startswith("_ticker_intercept"):
                placeholder_added = True
                continue

            formatted_feat = _format_feature_latex(feat_name)
            if coef >= 0:
                terms.append(f"+ {coef:.4f} \\cdot {formatted_feat}")
            else:
                terms.append(f"- {abs(coef):.4f} \\cdot {formatted_feat}")

        if placeholder_added:
            terms.append(f"+ \\text{{{_escape_latex_text('ticker_intercept')}}}")

        if terms:
            equation = "\\displaystyle \\hat{p} = " + " ".join(terms)
            return equation
    except Exception:
        pass

    return None


def preview_regime(payload: RegimePreviewRequest) -> RegimePreviewResponse:
    dataset_path = _resolve_project_path(payload.csv)
    _ensure_dataset_path(dataset_path)

    df = pd.read_csv(dataset_path)
    tdays_allowed = _parse_tdays_allowed(payload.tdays_allowed)
    asof_dow_allowed = _parse_asof_dow_allowed(payload.asof_dow_allowed)

    rows_before = int(len(df))
    by_tdays: Dict[str, int] = {}

    if tdays_allowed or asof_dow_allowed:
        missing_filter_cols = []
        if "T_days" not in df.columns:
            missing_filter_cols.append("T_days")
        asof_col = _resolve_asof_date_column(df)
        expiry_col = _resolve_expiry_date_column(df)
        if not asof_col:
            missing_filter_cols.append("asof_date (or equivalent)")
        if not expiry_col:
            missing_filter_cols.append("expiry_date (or equivalent)")
        if missing_filter_cols:
            raise ValueError(
                f"Regime filtering requires columns: {missing_filter_cols}. "
                "Ensure T_days, asof_date, and expiry_date are present."
            )

        df["T_days"] = pd.to_numeric(df["T_days"], errors="coerce")
        df[asof_col] = pd.to_datetime(df[asof_col], errors="coerce", utc=True)
        df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce", utc=True)
        _add_asof_dow_column(df, asof_col)

        if tdays_allowed:
            df = df[df["T_days"].isin(tdays_allowed)]
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


def get_model_detail(model_id: str) -> ModelDetailResponse:
    target = (MODELS_DIR / model_id).resolve()
    try:
        target.relative_to(MODELS_DIR.resolve())
    except Exception as exc:
        raise KeyError(model_id) from exc
    if not target.exists() or not target.is_dir():
        raise KeyError(model_id)

    summary = _model_summary_from_path(target)
    metadata = _load_json(target / "metadata.json")
    feature_manifest = _load_json(target / "feature_manifest.json")

    metrics_summary = _build_metrics_summary(target / "metrics.csv")
    if not metrics_summary:
        metrics_summary = None

    model_equation = _build_model_equation(target)
    files = sorted([item.name for item in target.iterdir() if item.is_file()])

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
        model_equation=model_equation,
        metadata=metadata if isinstance(metadata, dict) else None,
        feature_manifest=feature_manifest if isinstance(feature_manifest, dict) else None,
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


def run_calibration(payload: CalibrateModelRunRequest) -> CalibrateModelRunResponse:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Calibration script not found at {SCRIPT_PATH}")

    dataset_path = _resolve_project_path(payload.csv)
    _ensure_dataset_path(dataset_path)

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
    _add_value(cmd, "--foundation-tickers", payload.foundation_tickers)
    _add_value(cmd, "--foundation-weight", payload.foundation_weight)
    _add_value(cmd, "--mode", payload.mode)
    _add_value(cmd, "--ticker-intercepts", payload.ticker_intercepts)
    _add_flag(cmd, "--ticker-x-interactions", payload.ticker_x_interactions)
    _add_value(cmd, "--ticker-min-support", payload.ticker_min_support)
    _add_value(cmd, "--ticker-min-support-interactions", payload.ticker_min_support_interactions)
    _add_value(cmd, "--train-tickers", payload.train_tickers)
    _add_value(cmd, "--tdays-allowed", payload.tdays_allowed)
    _add_value(cmd, "--asof-dow-allowed", payload.asof_dow_allowed)
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
    _add_value(cmd, "--eceq-bins", payload.eceq_bins)
    _add_value(cmd, "--metrics-top-tickers", payload.metrics_top_tickers)
    _add_value(cmd, "--random-state", payload.random_state)
    _add_value(cmd, "--selection-objective", payload.selection_objective)
    if payload.fallback_to_baseline_if_worse is False:
        cmd.append("--no-fallback-to-baseline-if-worse")
    elif payload.fallback_to_baseline_if_worse is True:
        cmd.append("--fallback-to-baseline-if-worse")
    if payload.auto_drop_near_constant is False:
        cmd.append("--no-auto-drop-near-constant")
    elif payload.auto_drop_near_constant is True:
        cmd.append("--auto-drop-near-constant")

    # New features and filters
    _add_flag(cmd, "--enable-x-abs-m", payload.enable_x_abs_m)
    if payload.group_reweight and payload.group_reweight != "none":
        _add_value(cmd, "--group-reweight", payload.group_reweight)
    _add_value(cmd, "--max-abs-logm", payload.max_abs_logm)
    _add_flag(cmd, "--drop-prn-extremes", payload.drop_prn_extremes)
    if payload.drop_prn_extremes:
        _add_value(cmd, "--prn-eps", payload.prn_eps)

    # Bootstrap confidence intervals
    _add_flag(cmd, "--bootstrap-ci", payload.bootstrap_ci)
    if payload.bootstrap_ci:
        _add_value(cmd, "--bootstrap-B", payload.bootstrap_b)
        _add_value(cmd, "--bootstrap-seed", payload.bootstrap_seed)
        _add_value(cmd, "--bootstrap-group", payload.bootstrap_group)

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
        metrics_summary = None

    features = _read_features(out_dir)
    model_equation = _build_model_equation(out_dir)

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
    )


def run_auto_model_selection(payload: AutoModelRunRequest) -> CalibrateModelRunResponse:
    if not AUTO_SCRIPT_PATH.exists():
        raise RuntimeError(f"Auto selector script not found at {AUTO_SCRIPT_PATH}")

    dataset_path = _resolve_project_path(payload.csv)
    _ensure_dataset_path(dataset_path)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Use run_name to determine output directory
    default_name = f"auto-run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_name = payload.run_name or default_name
    out_dir = MODELS_DIR / run_name

    cmd: List[str] = [
        sys.executable,
        str(AUTO_SCRIPT_PATH),
        "--csv",
        str(dataset_path),
        "--out-dir",
        str(out_dir),
    ]
    _add_value(cmd, "--calibrator-script", str(SCRIPT_PATH))
    _add_value(cmd, "--objective", payload.objective)
    _add_value(cmd, "--max-trials", payload.max_trials)
    _add_value(cmd, "--seed", payload.seed)
    _add_value(cmd, "--parallel", payload.parallel)
    # Build baseline_args including any bootstrap flags
    baseline_args_parts: List[str] = []
    if payload.baseline_args:
        baseline_args_parts.append(payload.baseline_args)
    if payload.bootstrap_ci:
        baseline_args_parts.append("--bootstrap-ci")
        if payload.bootstrap_b is not None:
            baseline_args_parts.append(f"--bootstrap-B {payload.bootstrap_b}")
        if payload.bootstrap_seed is not None:
            baseline_args_parts.append(f"--bootstrap-seed {payload.bootstrap_seed}")
        if payload.bootstrap_group:
            baseline_args_parts.append(f"--bootstrap-group {payload.bootstrap_group}")
    combined_baseline_args = " ".join(baseline_args_parts) if baseline_args_parts else None

    _add_value(cmd, "--baseline-args", combined_baseline_args)
    _add_value(cmd, "--tdays-allowed", payload.tdays_allowed)
    _add_value(cmd, "--asof-dow-allowed", payload.asof_dow_allowed)
    _add_value(cmd, "--foundation-tickers", payload.foundation_tickers)
    _add_value(cmd, "--foundation-weight", payload.foundation_weight)

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

    # The best model is exported directly to out_dir (no longer in a "best_model" subdirectory)
    target_dir = out_dir

    artifact_files = []
    if target_dir.exists():
        artifact_files = sorted([item.name for item in target_dir.iterdir() if item.is_file()])

    metrics_summary = _build_metrics_summary(target_dir / "metrics.csv")
    if not metrics_summary:
        metrics_summary = None

    features = _read_features(target_dir)
    model_equation = _build_model_equation(target_dir)

    return CalibrateModelRunResponse(
        ok=result.returncode == 0,
        out_dir=str(target_dir.relative_to(BASE_DIR)),
        files=artifact_files,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=duration_s,
        command=cmd,
        metrics_summary=metrics_summary,
        auto_out_dir=None,
        features=features,
        model_equation=model_equation,
    )


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
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def to_status(self) -> CalibrationJobStatus:
        return CalibrationJobStatus(
            job_id=self.job_id,
            status=self.status,
            mode="auto" if self.mode == "auto" else "manual",
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    def _run(self) -> None:
        self.started_at = datetime.utcnow()
        self.status = "running"
        try:
            if self.mode == "auto":
                self.result = run_auto_model_selection(
                    self.payload  # type: ignore[arg-type]
                )
            else:
                self.result = run_calibration(
                    self.payload  # type: ignore[arg-type]
                )
            if self.result and self.result.ok:
                self.status = "finished"
            else:
                self.status = "failed"
                if self.result:
                    self.error = (self.result.stderr or "").strip() or "Calibration run failed."
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
        finally:
            self.finished_at = datetime.utcnow()


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
        job_id = uuid4().hex
        job = CalibrationJob(job_id, mode, payload)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> CalibrationJobStatus:
        job = self._get_job(job_id)
        return job.to_status()

    def list_jobs(self) -> List[CalibrationJobStatus]:
        with self._lock:
            return [job.to_status() for job in self._jobs.values()]

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
    return CALIBRATION_JOB_MANAGER.start_job(payload, mode="auto")


def get_calibration_job(job_id: str) -> CalibrationJobStatus:
    return CALIBRATION_JOB_MANAGER.get_status(job_id)


def list_calibration_jobs() -> List[CalibrationJobStatus]:
    return CALIBRATION_JOB_MANAGER.list_jobs()
