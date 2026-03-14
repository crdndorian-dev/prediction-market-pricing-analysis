from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from feature_engineering.option_chain.weighting_v3 import (
    LEGACY_WEIGHT_COLUMNS,
    V3_WEIGHT_COLUMNS,
    apply_weighting_v3,
    drop_weight_columns,
)
from option_chain.quality_flags import counted_quality_flag_columns


BUILD_META_NAME = "dataset_build_meta.json"
TRAINING_META_NAME = "training_selection.json"
DEFAULT_TRADE_FOCUS_TICKERS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "META",
    "AMZN",
    "PLTR",
    "NVDA",
    "NFLX",
    "OPEN",
    "TSLA",
]
QUALITY_BUCKETS = ("clean", "watch", "noisy")
_CSV_PURPOSE_PREFIXES = ("training-", "snapshot-", "prn-view-", "legacy-", "drops-")
_CLEANED_VARIANT_RE = re.compile(r"-cleaned(?:-\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class CleanupCriteria:
    quality_buckets: List[str] = field(default_factory=lambda: ["noisy"])
    min_quality_issue_count: Optional[int] = 3
    flag_columns: List[str] = field(default_factory=list)
    flag_match_mode: Optional[str] = "any"
    min_rel_spread_median: Optional[float] = None
    max_n_chain_used: Optional[float] = None


@dataclass(frozen=True)
class CleanupSampleRow:
    row_id: Optional[str] = None
    ticker: Optional[str] = None
    asof_date: Optional[str] = None
    expiry_date: Optional[str] = None
    K: Optional[float] = None
    pRN: Optional[float] = None
    quality_issue_count: Optional[float] = None
    rel_spread_median: Optional[float] = None
    n_chain_used: Optional[float] = None
    flags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CleanupFlagSummary:
    name: str
    count: int
    share: float


@dataclass(frozen=True)
class CleanupPreviewResult:
    training_path: Path
    rows_before: int
    rows_to_drop: int
    rows_after: int
    drop_share: float
    used_defaults: bool
    would_drop_all: bool
    dropped_bucket_counts: Dict[str, int]
    matched_flag_counts: List[CleanupFlagSummary]
    sample_rows: List[CleanupSampleRow]
    message: str


@dataclass(frozen=True)
class CleanupApplyResult:
    training_path: Path
    cleaned_run_dir: Path
    cleaned_path: Path
    rows_before: int
    rows_to_drop: int
    rows_after: int
    drop_share: float
    used_defaults: bool
    would_drop_all: bool
    dropped_bucket_counts: Dict[str, int]
    matched_flag_counts: List[CleanupFlagSummary]
    sample_rows: List[CleanupSampleRow]
    message: str


@dataclass(frozen=True)
class CleanupEvaluation:
    issue_count: pd.Series
    quality_bucket: pd.Series
    flag_columns: List[str]
    drop_mask: pd.Series


def _normalize_criteria(criteria: CleanupCriteria) -> CleanupCriteria:
    quality_buckets = [
        bucket
        for bucket in dict.fromkeys(
            str(bucket).strip().lower() for bucket in criteria.quality_buckets
        )
        if bucket in QUALITY_BUCKETS
    ]
    flag_columns = [
        value
        for value in dict.fromkeys(
            str(column).strip() for column in criteria.flag_columns if str(column).strip()
        )
    ]
    flag_match_mode = (
        "any"
        if criteria.flag_match_mode is None
        else str(criteria.flag_match_mode).strip().lower() or "any"
    )
    if flag_match_mode not in {"any", "all"}:
        raise ValueError("flag_match_mode must be 'any' or 'all'.")
    return CleanupCriteria(
        quality_buckets=quality_buckets,
        min_quality_issue_count=criteria.min_quality_issue_count,
        flag_columns=flag_columns,
        flag_match_mode=flag_match_mode,
        min_rel_spread_median=criteria.min_rel_spread_median,
        max_n_chain_used=criteria.max_n_chain_used,
    )


def _coerce_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    if str(series.dtype) == "bool":
        return series.fillna(False)
    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            }
        )
    )
    if normalized.notna().any():
        return normalized.fillna(False).astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0).astype(float).ne(0.0)


def _safe_float(value: object) -> Optional[float]:
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def _quality_bucket_from_issue_count(value: object) -> str:
    numeric = _safe_float(value)
    if numeric is None or numeric <= 0:
        return "clean"
    if numeric <= 2:
        return "watch"
    return "noisy"


def _parse_cmd_value(cmd: Sequence[str], flag: str) -> Optional[str]:
    values = list(cmd)
    if flag in values:
        idx = values.index(flag)
        if idx + 1 < len(values):
            return str(values[idx + 1])
    prefix = f"{flag}="
    for item in values:
        if str(item).startswith(prefix):
            return str(item)[len(prefix):]
    return None


def _resolve_weighting_params_from_run_dir(
    run_dir: Path,
    *,
    allow_defaults: bool = False,
) -> tuple[Dict[str, object], bool]:
    params: Dict[str, object] = {
        "ticker_reweight_mode": "none",
        "ticker_reweight_alpha_min": 0.5,
        "ticker_reweight_alpha_max": 2.0,
        "trade_focus_beta": 1.0,
        "trade_focus_tickers": None,
    }
    used_defaults = False
    meta_path = run_dir / BUILD_META_NAME
    payload: Optional[Dict[str, object]] = None
    command: Optional[List[str]] = None

    if meta_path.exists():
        try:
            parsed = json.loads(meta_path.read_text())
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            if isinstance(parsed.get("payload"), dict):
                payload = parsed["payload"]
            command_value = parsed.get("command")
            if isinstance(command_value, list):
                command = [str(item) for item in command_value]
    if payload is None and command is None:
        if not allow_defaults:
            raise ValueError(
                "Dataset build metadata is missing. Enable allow_defaults to proceed."
            )
        used_defaults = True

    if payload:
        if payload.get("ticker_reweight_mode"):
            params["ticker_reweight_mode"] = str(payload["ticker_reweight_mode"])
        if payload.get("ticker_reweight_alpha_min") is not None:
            params["ticker_reweight_alpha_min"] = float(payload["ticker_reweight_alpha_min"])
        if payload.get("ticker_reweight_alpha_max") is not None:
            params["ticker_reweight_alpha_max"] = float(payload["ticker_reweight_alpha_max"])
        if payload.get("trade_focus_beta") is not None:
            params["trade_focus_beta"] = float(payload["trade_focus_beta"])
        if payload.get("trade_focus_tickers"):
            params["trade_focus_tickers"] = str(payload["trade_focus_tickers"])

    if command:
        mode = _parse_cmd_value(command, "--ticker-reweight-mode")
        if mode:
            params["ticker_reweight_mode"] = mode
        alpha_min = _parse_cmd_value(command, "--ticker-reweight-alpha-min")
        if alpha_min:
            params["ticker_reweight_alpha_min"] = float(alpha_min)
        alpha_max = _parse_cmd_value(command, "--ticker-reweight-alpha-max")
        if alpha_max:
            params["ticker_reweight_alpha_max"] = float(alpha_max)
        trade_beta = _parse_cmd_value(command, "--trade-focus-beta")
        if trade_beta:
            params["trade_focus_beta"] = float(trade_beta)
        trade_tickers = _parse_cmd_value(command, "--trade-focus-tickers")
        if trade_tickers:
            params["trade_focus_tickers"] = trade_tickers

    if params.get("trade_focus_tickers") in {None, ""}:
        beta = float(params.get("trade_focus_beta", 1.0))
        if beta != 1.0:
            params["trade_focus_tickers"] = ",".join(DEFAULT_TRADE_FOCUS_TICKERS)

    return params, used_defaults


def resolve_training_file(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    for candidate in sorted(run_dir.glob("training-*.csv")):
        if candidate.is_file() and not _is_cleaned_variant_stem(candidate.stem):
            return candidate
    legacy_target = run_dir / f"{run_dir.name}.csv"
    if legacy_target.exists() and legacy_target.is_file():
        return legacy_target
    selection_path = run_dir / "training_selection.json"
    if selection_path.exists():
        try:
            selection = json.loads(selection_path.read_text())
        except Exception:
            selection = None
        if isinstance(selection, dict) and selection.get("training_file"):
            candidate = run_dir / str(selection["training_file"])
            if candidate.exists() and candidate.is_file():
                return candidate
    default_train_view = run_dir / "train_view.csv"
    if default_train_view.exists() and default_train_view.is_file():
        return default_train_view
    for candidate in sorted(run_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() == ".csv" and "drop" not in candidate.name.lower():
            return candidate
    raise FileNotFoundError("Training dataset not found in the selected run directory.")


def _resolve_issue_count(df: pd.DataFrame, flag_columns: List[str]) -> pd.Series:
    if "quality_issue_count" in df.columns:
        return pd.to_numeric(df["quality_issue_count"], errors="coerce").fillna(0.0)
    if flag_columns:
        quality_flag_columns = counted_quality_flag_columns(flag_columns)
        return pd.DataFrame(
            {
                column: _coerce_bool_series(df, column).astype(int)
                for column in quality_flag_columns
            }
        ).sum(axis=1)
    raise ValueError(
        "Dataset cleanup requires a quality_issue_count column or at least one flag_* column."
    )


def evaluate_cleanup(df: pd.DataFrame, criteria: CleanupCriteria) -> CleanupEvaluation:
    normalized = _normalize_criteria(criteria)
    flag_columns = sorted(column for column in df.columns if column.startswith("flag_"))
    issue_count = _resolve_issue_count(df, flag_columns)
    quality_bucket = issue_count.map(_quality_bucket_from_issue_count)
    drop_mask = pd.Series(True, index=df.index, dtype=bool)

    if normalized.quality_buckets:
        drop_mask &= quality_bucket.isin(normalized.quality_buckets)

    if normalized.min_quality_issue_count is not None:
        drop_mask &= issue_count.ge(float(normalized.min_quality_issue_count))

    if normalized.flag_columns:
        missing_flags = [column for column in normalized.flag_columns if column not in flag_columns]
        if missing_flags:
            raise ValueError(
                "Unknown cleanup flag columns: " + ", ".join(sorted(missing_flags))
            )
        flag_frame = pd.DataFrame(
            {
                column: _coerce_bool_series(df, column)
                for column in normalized.flag_columns
            }
        )
        flag_mask = flag_frame.all(axis=1) if normalized.flag_match_mode == "all" else flag_frame.any(axis=1)
        drop_mask &= flag_mask

    if normalized.min_rel_spread_median is not None:
        if "rel_spread_median" not in df.columns:
            raise ValueError("Dataset does not contain rel_spread_median.")
        rel_spread = pd.to_numeric(df["rel_spread_median"], errors="coerce")
        drop_mask &= rel_spread.ge(float(normalized.min_rel_spread_median)).fillna(False)

    if normalized.max_n_chain_used is not None:
        if "n_chain_used" not in df.columns:
            raise ValueError("Dataset does not contain n_chain_used.")
        n_chain_used = pd.to_numeric(df["n_chain_used"], errors="coerce")
        drop_mask &= n_chain_used.le(float(normalized.max_n_chain_used)).fillna(False)

    return CleanupEvaluation(
        issue_count=issue_count,
        quality_bucket=quality_bucket,
        flag_columns=flag_columns,
        drop_mask=drop_mask.fillna(False).astype(bool),
    )


def _rank_sample_rows(
    df: pd.DataFrame,
    evaluation: CleanupEvaluation,
    *,
    limit: int = 12,
) -> List[CleanupSampleRow]:
    dropped = df.loc[evaluation.drop_mask].copy()
    if dropped.empty:
        return []
    dropped["__issue_count"] = evaluation.issue_count.loc[dropped.index]
    dropped["__rel_spread_median"] = (
        pd.to_numeric(dropped["rel_spread_median"], errors="coerce").fillna(-1.0)
        if "rel_spread_median" in dropped.columns
        else -1.0
    )
    dropped["__n_chain_used"] = (
        pd.to_numeric(dropped["n_chain_used"], errors="coerce").fillna(1e9)
        if "n_chain_used" in dropped.columns
        else 1e9
    )
    for column in evaluation.flag_columns:
        dropped[column] = _coerce_bool_series(dropped, column)
    dropped = dropped.sort_values(
        ["__issue_count", "__rel_spread_median", "__n_chain_used"],
        ascending=[False, False, True],
    ).head(limit)

    out: List[CleanupSampleRow] = []
    for _, row in dropped.iterrows():
        active_flags = [
            column for column in evaluation.flag_columns if bool(row.get(column, False))
        ]
        out.append(
            CleanupSampleRow(
                row_id=str(row["row_id"]) if "row_id" in row.index and pd.notna(row["row_id"]) else None,
                ticker=str(row["ticker"]) if "ticker" in row.index and pd.notna(row["ticker"]) else None,
                asof_date=str(row["asof_date"]) if "asof_date" in row.index and pd.notna(row["asof_date"]) else None,
                expiry_date=str(row["expiry_date"]) if "expiry_date" in row.index and pd.notna(row["expiry_date"]) else None,
                K=_safe_float(row["K"]) if "K" in row.index else None,
                pRN=_safe_float(row["pRN"]) if "pRN" in row.index else None,
                quality_issue_count=_safe_float(row["__issue_count"]),
                rel_spread_median=_safe_float(row["__rel_spread_median"]),
                n_chain_used=_safe_float(row["__n_chain_used"]),
                flags=active_flags,
            )
        )
    return out


def _summarize_matched_flags(
    df: pd.DataFrame,
    evaluation: CleanupEvaluation,
) -> List[CleanupFlagSummary]:
    dropped = df.loc[evaluation.drop_mask]
    if dropped.empty or not evaluation.flag_columns:
        return []
    summaries: List[CleanupFlagSummary] = []
    rows_to_drop = len(dropped)
    for column in evaluation.flag_columns:
        active = _coerce_bool_series(dropped, column)
        count = int(active.sum())
        if count <= 0:
            continue
        summaries.append(
            CleanupFlagSummary(
                name=column,
                count=count,
                share=round(count / rows_to_drop, 6) if rows_to_drop else 0.0,
            )
        )
    summaries.sort(key=lambda item: (-item.count, item.name))
    return summaries


def build_cleanup_preview(
    training_path: Path,
    criteria: CleanupCriteria,
    *,
    used_defaults: bool = False,
) -> CleanupPreviewResult:
    training_path = training_path.resolve()
    if not training_path.exists() or not training_path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {training_path}")
    if training_path.suffix.lower() != ".csv":
        raise ValueError("Dataset cleanup currently supports CSV files only.")

    df = pd.read_csv(training_path, low_memory=False)
    evaluation = evaluate_cleanup(df, criteria)
    rows_before = int(len(df))
    rows_to_drop = int(evaluation.drop_mask.sum())
    rows_after = max(0, rows_before - rows_to_drop)
    would_drop_all = rows_before > 0 and rows_to_drop == rows_before
    drop_share = round((rows_to_drop / rows_before) if rows_before else 0.0, 6)

    dropped_quality_bucket = evaluation.quality_bucket.loc[evaluation.drop_mask]
    dropped_bucket_counts = {
        bucket: int((dropped_quality_bucket == bucket).sum()) for bucket in QUALITY_BUCKETS
    }
    matched_flag_counts = _summarize_matched_flags(df, evaluation)
    sample_rows = _rank_sample_rows(df, evaluation)

    if rows_to_drop == 0:
        message = "No rows match the selected cleanup criteria."
    elif would_drop_all:
        message = (
            "Selected cleanup criteria would drop all rows. Narrow the filters before applying."
        )
    else:
        message = (
            f"Cleanup will drop {rows_to_drop} rows and keep {rows_after} rows "
            f"({drop_share * 100:.1f}% dropped)."
        )

    return CleanupPreviewResult(
        training_path=training_path,
        rows_before=rows_before,
        rows_to_drop=rows_to_drop,
        rows_after=rows_after,
        drop_share=drop_share,
        used_defaults=used_defaults,
        would_drop_all=would_drop_all,
        dropped_bucket_counts=dropped_bucket_counts,
        matched_flag_counts=matched_flag_counts,
        sample_rows=sample_rows,
        message=message,
    )


def _reweight_dataframe(df: pd.DataFrame, weighting_params: Dict[str, object]) -> pd.DataFrame:
    cleaned = drop_weight_columns(df)
    return apply_weighting_v3(
        cleaned,
        ticker_reweight_mode=str(weighting_params["ticker_reweight_mode"]),
        ticker_reweight_alpha_min=float(weighting_params["ticker_reweight_alpha_min"]),
        ticker_reweight_alpha_max=float(weighting_params["ticker_reweight_alpha_max"]),
        trade_focus_beta=float(weighting_params["trade_focus_beta"]),
        trade_focus_tickers=weighting_params.get("trade_focus_tickers"),
        strict=True,
    )


def _is_cleaned_variant_stem(stem: str) -> bool:
    return bool(_CLEANED_VARIANT_RE.search(str(stem).strip()))


def _build_cleaned_run_dir(run_dir: Path) -> Path:
    base_name = f"{run_dir.name}-cleaned"
    candidate = run_dir.parent / base_name
    index = 0
    while candidate.exists():
        index += 1
        candidate = run_dir.parent / f"{base_name}-{index}"
    return candidate


def _renamed_csv_name(
    name: str,
    *,
    source_run_name: str,
    target_run_name: str,
) -> str:
    lowered = name.lower()
    for prefix in _CSV_PURPOSE_PREFIXES:
        if lowered.startswith(prefix):
            return f"{prefix}{target_run_name}.csv"
    if lowered == "train_view.csv":
        return f"training-{target_run_name}.csv"
    if lowered == "snapshot.csv":
        return f"snapshot-{target_run_name}.csv"
    if lowered == "prn_view.csv":
        return f"prn-view-{target_run_name}.csv"
    if "drop" in lowered:
        return f"drops-{target_run_name}.csv"
    if lowered == f"{source_run_name.lower()}.csv":
        return f"training-{target_run_name}.csv"
    return name


def _is_cleanup_byproduct_csv(path: Path, *, source_run_name: str) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    if not _is_cleaned_variant_stem(path.stem):
        return False
    if path.stem.lower().endswith(source_run_name.lower()):
        return False
    return True


def _copy_run_dir(
    source_run_dir: Path,
    staging_dir: Path,
) -> None:
    source_run_name = source_run_dir.name

    def _ignore(current_dir: str, names: List[str]) -> Set[str]:
        ignored: Set[str] = set()
        current_path = Path(current_dir)
        for name in names:
            candidate = current_path / name
            if _is_cleanup_byproduct_csv(candidate, source_run_name=source_run_name):
                ignored.add(name)
        return ignored

    shutil.copytree(
        source_run_dir,
        staging_dir,
        dirs_exist_ok=True,
        ignore=_ignore,
        copy_function=shutil.copy2,
    )


def _rename_standard_csvs(
    run_dir: Path,
    *,
    source_run_name: str,
    target_run_name: str,
) -> None:
    for item in sorted(run_dir.iterdir()):
        if not item.is_file() or item.suffix.lower() != ".csv":
            continue
        new_name = _renamed_csv_name(
            item.name,
            source_run_name=source_run_name,
            target_run_name=target_run_name,
        )
        if new_name == item.name:
            continue
        target = run_dir / new_name
        if target.exists():
            raise FileExistsError(f"Target CSV already exists in cleaned run staging dir: {target}")
        item.rename(target)


def _write_dataframe_atomic(path: Path, df: pd.DataFrame) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.stem}-",
        suffix=path.suffix or ".csv",
        dir=str(path.parent),
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _filter_row_id_csv(
    csv_path: Path,
    *,
    keep_row_ids: Set[str],
    weighting_params: Dict[str, object],
) -> None:
    df = pd.read_csv(csv_path, low_memory=False)
    if "row_id" not in df.columns:
        return
    row_ids = df["row_id"].astype("string")
    filtered = df.loc[row_ids.isin(keep_row_ids)].copy()
    if any(column in df.columns for column in (*LEGACY_WEIGHT_COLUMNS, *V3_WEIGHT_COLUMNS)):
        filtered = _reweight_dataframe(filtered, weighting_params)
    _write_dataframe_atomic(csv_path, filtered)


def _rewrite_training_selection(
    run_dir: Path,
    *,
    source_run_name: str,
    target_run_name: str,
    training_file_name: str,
) -> None:
    path = run_dir / TRAINING_META_NAME
    payload: Dict[str, object] = {}
    if path.exists():
        try:
            parsed = json.loads(path.read_text())
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    payload["dataset_name"] = target_run_name
    payload["training_file"] = training_file_name
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    for field in ("train_view_file", "legacy_file"):
        existing = payload.get(field)
        if isinstance(existing, str) and existing.strip():
            payload[field] = _renamed_csv_name(
                existing,
                source_run_name=source_run_name,
                target_run_name=target_run_name,
            )
    path.write_text(json.dumps(payload, indent=2))


def _rewrite_build_meta(
    run_dir: Path,
    *,
    source_run_name: str,
    target_run_name: str,
) -> None:
    path = run_dir / BUILD_META_NAME
    if not path.exists():
        return
    try:
        parsed = json.loads(path.read_text())
    except Exception:
        return
    if not isinstance(parsed, dict):
        return

    def _rename_if_present(value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            return value
        return _renamed_csv_name(
            value,
            source_run_name=source_run_name,
            target_run_name=target_run_name,
        )

    parsed["out_name"] = _rename_if_present(parsed.get("out_name"))
    parsed["drops_name"] = _rename_if_present(parsed.get("drops_name"))
    payload = parsed.get("payload")
    if isinstance(payload, dict):
        payload["dataset_name"] = target_run_name
        payload["run_dir_name"] = target_run_name
        for field in ("out_name", "train_view_name", "drops_name"):
            payload[field] = _rename_if_present(payload.get(field))
    path.write_text(json.dumps(parsed, indent=2))


def apply_cleanup(
    run_dir: Path,
    training_path: Path,
    criteria: CleanupCriteria,
    weighting_params: Dict[str, object],
    *,
    used_defaults: bool = False,
) -> CleanupApplyResult:
    run_dir = run_dir.resolve()
    training_path = training_path.resolve()
    preview = build_cleanup_preview(training_path, criteria, used_defaults=used_defaults)
    if preview.rows_to_drop <= 0:
        raise ValueError("No rows match the selected cleanup criteria.")
    if preview.would_drop_all:
        raise ValueError(
            "Selected cleanup criteria would drop all rows. Narrow the filters before applying."
        )

    df = pd.read_csv(training_path, low_memory=False)
    evaluation = evaluate_cleanup(df, criteria)
    kept_rows = df.loc[~evaluation.drop_mask].copy()
    keep_row_ids = {
        str(value).strip()
        for value in kept_rows["row_id"].astype("string").tolist()
        if str(value).strip() and str(value).strip().lower() != "<na>"
    }
    cleaned_run_dir = _build_cleaned_run_dir(run_dir)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{cleaned_run_dir.name}.__tmp__",
            dir=str(run_dir.parent),
        )
    )
    cleaned_training_name = _renamed_csv_name(
        training_path.name,
        source_run_name=run_dir.name,
        target_run_name=cleaned_run_dir.name,
    )
    cleaned_staging_training_path = staging_dir / cleaned_training_name
    try:
        _copy_run_dir(run_dir, staging_dir)
        _rename_standard_csvs(
            staging_dir,
            source_run_name=run_dir.name,
            target_run_name=cleaned_run_dir.name,
        )
        if not cleaned_staging_training_path.exists():
            raise FileNotFoundError(
                f"Cleaned training artifact missing from staging directory: {cleaned_staging_training_path}"
            )
        for csv_path in sorted(staging_dir.glob("*.csv")):
            _filter_row_id_csv(
                csv_path,
                keep_row_ids=keep_row_ids,
                weighting_params=weighting_params,
            )
        _rewrite_training_selection(
            staging_dir,
            source_run_name=run_dir.name,
            target_run_name=cleaned_run_dir.name,
            training_file_name=cleaned_training_name,
        )
        _rewrite_build_meta(
            staging_dir,
            source_run_name=run_dir.name,
            target_run_name=cleaned_run_dir.name,
        )
        staging_dir.rename(cleaned_run_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return CleanupApplyResult(
        training_path=training_path,
        cleaned_run_dir=cleaned_run_dir,
        cleaned_path=cleaned_run_dir / cleaned_training_name,
        rows_before=preview.rows_before,
        rows_to_drop=preview.rows_to_drop,
        rows_after=preview.rows_after,
        drop_share=preview.drop_share,
        used_defaults=preview.used_defaults,
        would_drop_all=preview.would_drop_all,
        dropped_bucket_counts=preview.dropped_bucket_counts,
        matched_flag_counts=preview.matched_flag_counts,
        sample_rows=preview.sample_rows,
        message=(
            f"Cleanup complete: {preview.rows_to_drop} rows dropped from "
            f"{training_path.name}; created cleaned run {cleaned_run_dir.name} "
            f"with {preview.rows_after} rows in {cleaned_training_name}."
        ),
    )


def _result_to_jsonable(result: CleanupPreviewResult | CleanupApplyResult) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "training_path": str(result.training_path),
        "rows_before": result.rows_before,
        "rows_to_drop": result.rows_to_drop,
        "rows_after": result.rows_after,
        "drop_share": result.drop_share,
        "used_defaults": result.used_defaults,
        "would_drop_all": result.would_drop_all,
        "dropped_bucket_counts": result.dropped_bucket_counts,
        "matched_flag_counts": [asdict(item) for item in result.matched_flag_counts],
        "sample_rows": [asdict(item) for item in result.sample_rows],
        "message": result.message,
    }
    if isinstance(result, CleanupApplyResult):
        payload.update(
            {
                "cleaned_run_dir": str(result.cleaned_run_dir),
                "cleaned_path": str(result.cleaned_path),
            }
        )
    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or apply noisy-row cleanup for an option-chain training dataset.",
    )
    parser.add_argument("--run-dir", required=True, help="Dataset run directory.")
    parser.add_argument(
        "--training-file",
        default=None,
        help="Optional explicit training CSV path. Defaults to the run's selected training file.",
    )
    parser.add_argument(
        "--quality-bucket",
        action="append",
        choices=list(QUALITY_BUCKETS),
        dest="quality_buckets",
        help="Quality bucket to drop. Repeat to select multiple buckets.",
    )
    parser.add_argument(
        "--min-quality-issue-count",
        type=int,
        default=3,
        help="Minimum quality_issue_count required for a row to be dropped.",
    )
    parser.add_argument(
        "--flag-column",
        action="append",
        default=[],
        help="Flag column that must match for a row to be dropped. Repeat to select multiple flags.",
    )
    parser.add_argument(
        "--flag-match-mode",
        choices=["any", "all"],
        default="any",
        help="How selected flag columns are combined.",
    )
    parser.add_argument(
        "--min-rel-spread-median",
        type=float,
        default=None,
        help="Optional minimum rel_spread_median threshold.",
    )
    parser.add_argument(
        "--max-n-chain-used",
        type=float,
        default=None,
        help="Optional maximum n_chain_used threshold.",
    )
    parser.add_argument(
        "--allow-defaults",
        action="store_true",
        help="Allow cleanup to use default weighting settings when build metadata is missing.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create a cleaned cloned run directory. Without this flag the script prints a preview only.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    training_path = (
        Path(args.training_file).expanduser().resolve()
        if args.training_file
        else resolve_training_file(run_dir)
    )
    criteria = CleanupCriteria(
        quality_buckets=args.quality_buckets or ["noisy"],
        min_quality_issue_count=args.min_quality_issue_count,
        flag_columns=list(args.flag_column or []),
        flag_match_mode=args.flag_match_mode,
        min_rel_spread_median=args.min_rel_spread_median,
        max_n_chain_used=args.max_n_chain_used,
    )
    weighting_params, used_defaults = _resolve_weighting_params_from_run_dir(
        run_dir,
        allow_defaults=bool(args.allow_defaults),
    )
    if args.apply:
        result = apply_cleanup(
            run_dir,
            training_path,
            criteria,
            weighting_params,
            used_defaults=used_defaults,
        )
    else:
        result = build_cleanup_preview(
            training_path,
            criteria,
            used_defaults=used_defaults,
        )
    print(json.dumps(_result_to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
