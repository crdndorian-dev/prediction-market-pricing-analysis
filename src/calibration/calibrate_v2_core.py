#!/usr/bin/env python3
"""
calibrate_v2_core.py

In-process calibration core for the v2 logit trainer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration.calibrate_common import (
    EPS,
    FinalModelBundle,
    apply_platt,
    build_chain_group_id,
    ece_equal_mass,
    ensure_engineered_features,
    fit_platt_on_logits,
)

SCRIPT_VERSION = "v2.0.0"

# Default feature sets
DEFAULT_PRN_FEATURES = [
    "pRN",
    "x_logit_prn",
    "log_m_fwd",
    "abs_log_m_fwd",
    "T_days",
    "rv20",
    "rv20_sqrtT",
    "rel_spread_median",
]
PRN_ASOF_DATE_CANDIDATES = ["asof_date", "asof_target", "asof_ts", "asof_time", "asof_datetime"]
PRN_EXPIRY_CANDIDATES = ["expiry_close_date_used", "option_expiration_used", "option_expiration_requested", "expiry_date"]
PRN_STRIKE_CANDIDATES = ["K", "threshold"]


@dataclass
class CalibrationCache:
    df_base: pd.DataFrame
    target_col: str
    week_col: str
    ticker_col: str
    train_idx: pd.Index
    test_idx: pd.Index
    train_fit_idx: pd.Index
    val_idx: pd.Index
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_fit_df: pd.DataFrame
    val_df: pd.DataFrame
    train_pos: pd.Series
    fit_pos: np.ndarray
    val_pos: np.ndarray
    split_group_series: Optional[pd.Series]
    split_group_key: Optional[str]
    split_group_dropped_train_rows: int
    split_group_dropped_train_fit_rows: int
    embargo_mode: str
    embargo_date_col_used: Optional[str]
    embargo_rows_dropped_train: int
    embargo_rows_dropped_train_fit: int
    walk_forward_folds: List[Dict[str, Any]]
    val_split_info: Dict[str, Any]
    split_overlap: Dict[str, Any]
    split_ranges: Dict[str, Any]
    split_composition_rows: List[Dict[str, Any]]
    train_weights_raw: np.ndarray
    weight_source: str
    group_key: Optional[pd.Series]
    group_key_source: Optional[str]
    foundation_set: set[str]
    active_filters: Dict[str, Any]
    trainer_warnings: List[str]
    requested_numeric_features: List[str]
    requested_categorical_features: List[str]


@dataclass
class RunResult:
    exit_code: int
    out_dir: Path
    metrics_rows: List[Dict[str, Any]]


def _file_fingerprint(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime}".encode()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "fingerprint": hashlib.sha256(payload).hexdigest(),
    }


def _get_git_info() -> Dict[str, Optional[str]]:
    commit = None
    commit_dt = None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=True,
        )
        commit = res.stdout.strip() or None
    except Exception:
        pass
    try:
        res = subprocess.run(
            ["git", "show", "-s", "--format=%cI", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=True,
        )
        commit_dt = res.stdout.strip() or None
    except Exception:
        pass
    return {"git_commit": commit, "git_commit_datetime": commit_dt}


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_first_column(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"{label}: none of {candidates} present in dataset.")


def _to_date_series(df: pd.DataFrame, col: str, label: str) -> pd.Series:
    series = pd.to_datetime(df[col], errors="coerce", utc=True)
    if series.isna().all():
        raise ValueError(f"{label}: column {col} could not be parsed to datetime.")
    return series.dt.date


def _compute_week_friday(dates: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dates, errors="coerce")
    if dt.isna().all():
        return pd.Series([pd.NaT] * len(dates))
    weekday = dt.dt.weekday
    offset = (4 - weekday) % 7
    return (dt + pd.to_timedelta(offset, unit="D")).dt.normalize()


def _select_label_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isin([0, 1]).any():
            return col
    return None


def _drop_all_nan_features(df: pd.DataFrame, features: List[str], label: str) -> List[str]:
    kept: List[str] = []
    dropped: List[str] = []
    for feat in features:
        if feat not in df.columns:
            dropped.append(feat)
            continue
        series = pd.to_numeric(df[feat], errors="coerce")
        if series.notna().any():
            kept.append(feat)
        else:
            dropped.append(feat)
    if dropped:
        print(f"[{label}] Dropping all-NaN or missing features: {sorted(set(dropped))}")
    return kept


def _load_prn_full(path: Path) -> pd.DataFrame:
    df = _load_dataset(path)
    if df.empty:
        raise ValueError("pRN dataset is empty.")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("string").str.upper()
    strike_col = _resolve_first_column(df, PRN_STRIKE_CANDIDATES, "pRN strike")
    expiry_col = _resolve_first_column(df, PRN_EXPIRY_CANDIDATES, "pRN expiry date")
    asof_col = _resolve_first_column(df, PRN_ASOF_DATE_CANDIDATES, "pRN asof date")

    df["threshold"] = pd.to_numeric(df[strike_col], errors="coerce")
    df["expiry_date"] = _to_date_series(df, expiry_col, "pRN expiry date")
    df["snapshot_date"] = _to_date_series(df, asof_col, "pRN asof date")

    if "week_friday" in df.columns:
        df["week_friday"] = pd.to_datetime(df["week_friday"], errors="coerce").dt.normalize()
    else:
        df["week_friday"] = _compute_week_friday(df["snapshot_date"])

    if df["snapshot_date"].isna().any():
        raise ValueError("pRN dataset has NaT snapshot_date values; cannot merge.")
    return df


def _dedupe_on_keys(df: pd.DataFrame, keys: List[str], label: str) -> pd.DataFrame:
    if df.duplicated(subset=keys).any():
        dupes = int(df.duplicated(subset=keys).sum())
        print(f"[{label}] Dropping {dupes} duplicate key rows.")
        df = df.sort_values(keys).drop_duplicates(subset=keys, keep="last")
    return df


def _compute_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    y_true = y_true.astype(float)
    return float(np.mean((p - y_true) ** 2))


def _coerce_positive_bin_count(value: Optional[int], *, default: int, arg_name: str) -> int:
    if value is None:
        return int(default)
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"{arg_name} must be a positive integer (got {value!r}).") from exc
    if parsed <= 0:
        raise ValueError(f"{arg_name} must be > 0 (got {parsed}).")
    return parsed


def _compute_ece(y_true: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using equal-width bins."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y_true = np.asarray(y_true, dtype=float)
    n = len(p)
    if n == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # np.digitize returns bins in [1, n_bins+1]; clamp to [0, n_bins-1].
    bin_idx = np.digitize(p, bins[1:-1], right=False)
    ece = 0.0

    for i in range(int(n_bins)):
        mask = bin_idx == i
        if not np.any(mask):
            continue
        avg_pred = float(np.mean(p[mask]))
        avg_true = float(np.mean(y_true[mask]))
        ece += (float(mask.sum()) / float(n)) * abs(avg_pred - avg_true)
    return float(ece)


def _compute_ece_q(y_true: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> float:
    """Compute equal-mass ECE (ECE-Q)."""
    return float(ece_equal_mass(np.asarray(y_true, dtype=float), np.asarray(p, dtype=float), n_bins=int(n_bins)))


def _compute_logloss(y_true: np.ndarray, p: np.ndarray) -> float:
    """Binary logloss that is stable for single-class samples (e.g., bootstrap resamples)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def _compute_metrics(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    eceq_bins: int = 10,
) -> Dict[str, float]:
    """Compute comprehensive metrics including logloss, Brier, ECE, and ECE-Q."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return {
        "logloss": _compute_logloss(y_true, p),
        "brier": _compute_brier(y_true, p),
        "ece": _compute_ece(y_true, p, n_bins=n_bins),
        "ece_q": _compute_ece_q(y_true, p, n_bins=eceq_bins),
    }


def _resolve_bootstrap_day_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Best-effort per-row day key for grouped bootstrap."""
    for col in [
        "snapshot_date",
        "asof_date",
        "snapshot_time_utc",
        "asof_datetime_utc",
        "asof_datetime",
        "snapshot_datetime_utc",
        "week_friday",
    ]:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().any():
            return parsed.dt.strftime("%Y-%m-%d")
    return None


def _resolve_bootstrap_groups(
    df: pd.DataFrame,
    *,
    ticker_col: str,
    requested_group: str,
    allow_iid_bootstrap: bool = False,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Resolve bootstrap grouping labels for a split.

    Returns (labels, resolved_group_name). labels=None means iid row bootstrap.
    """
    req = (requested_group or "auto").strip().lower()
    if req not in {"auto", "ticker_day", "day", "iid", "contract_id", "group_id"}:
        req = "auto"

    if req == "iid":
        return None, "iid"

    day_series = _resolve_bootstrap_day_series(df)
    has_day = day_series is not None

    ticker_series: Optional[pd.Series] = None
    if ticker_col in df.columns:
        ticker_series = df[ticker_col].astype("string").fillna("UNKNOWN").str.upper()
        if not ticker_series.notna().any():
            ticker_series = None

    def _fallback_or_fail(reason: str) -> Tuple[Optional[np.ndarray], str]:
        if allow_iid_bootstrap:
            return None, "iid"
        raise ValueError(
            f"Requested bootstrap_group={req!r} is unavailable ({reason}). "
            "Set allow_iid_bootstrap=true to permit iid fallback."
        )

    if req == "ticker_day":
        if has_day and ticker_series is not None:
            labels = (
                ticker_series.astype("string").fillna("UNKNOWN")
                + "|"
                + day_series.fillna("NA").astype("string")
            )
            return labels.to_numpy(dtype=object), "ticker_day"
        if not has_day:
            return _fallback_or_fail("missing date column")
        return _fallback_or_fail(f"missing ticker column {ticker_col!r}")

    if req == "day":
        if has_day:
            return day_series.fillna("NA").to_numpy(dtype=object), "day"
        return _fallback_or_fail("missing date column")

    if req in {"contract_id", "group_id"}:
        if req in df.columns:
            labels = df[req].astype("string").fillna("NA")
            if labels.notna().any():
                return labels.to_numpy(dtype=object), req
            return _fallback_or_fail(f"column {req!r} has no usable values")
        return _fallback_or_fail(f"column {req!r} missing")

    # auto: prefer ticker_day, then day, else iid
    if has_day and ticker_series is not None:
        labels = (
            ticker_series.astype("string").fillna("UNKNOWN")
            + "|"
            + day_series.fillna("NA").astype("string")
        )
        return labels.to_numpy(dtype=object), "ticker_day"
    if has_day:
        return day_series.fillna("NA").to_numpy(dtype=object), "day"
    return None, "iid"


def _bootstrap_delta_metric_cis(
    *,
    y_true: np.ndarray,
    p_model: np.ndarray,
    p_baseline: np.ndarray,
    group_labels: Optional[np.ndarray],
    bootstrap_B: int,
    bootstrap_seed: int,
    n_bins: int,
    eceq_bins: int,
    ci_level: int = 95,
) -> Dict[str, Optional[float]]:
    """
    Compute bootstrap confidence intervals for delta metrics (model - baseline).

    Returns CI columns plus bootstrap metadata expected by the webapp parser.
    """
    out: Dict[str, Optional[float]] = {
        "delta_logloss_ci_lo": None,
        "delta_logloss_ci_hi": None,
        "delta_brier_ci_lo": None,
        "delta_brier_ci_hi": None,
        "delta_ece_ci_lo": None,
        "delta_ece_ci_hi": None,
        "delta_ece_q_ci_lo": None,
        "delta_ece_q_ci_hi": None,
        "bootstrap_n_groups": None,
        "bootstrap_B": None,
    }
    if bootstrap_B <= 0:
        return out

    y = np.asarray(y_true, dtype=float)
    p_m = np.asarray(p_model, dtype=float)
    p_b = np.asarray(p_baseline, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p_m) & np.isfinite(p_b)
    y = y[valid]
    p_m = np.clip(p_m[valid], EPS, 1.0 - EPS)
    p_b = np.clip(p_b[valid], EPS, 1.0 - EPS)
    if len(y) < 2:
        return out

    rng = np.random.RandomState(int(bootstrap_seed))

    if group_labels is None:
        n = len(y)
        out["bootstrap_n_groups"] = int(n)
        out["bootstrap_B"] = int(bootstrap_B)
        delta_logloss_vals: List[float] = []
        delta_brier_vals: List[float] = []
        delta_ece_vals: List[float] = []
        delta_ece_q_vals: List[float] = []
        for _ in range(int(bootstrap_B)):
            idx = rng.choice(n, size=n, replace=True)
            y_s = y[idx]
            p_m_s = p_m[idx]
            p_b_s = p_b[idx]
            delta_logloss_vals.append(float(_compute_logloss(y_s, p_m_s) - _compute_logloss(y_s, p_b_s)))
            delta_brier_vals.append(float(_compute_brier(y_s, p_m_s) - _compute_brier(y_s, p_b_s)))
            delta_ece_vals.append(float(_compute_ece(y_s, p_m_s, n_bins=n_bins) - _compute_ece(y_s, p_b_s, n_bins=n_bins)))
            delta_ece_q_vals.append(
                float(
                    _compute_ece_q(y_s, p_m_s, n_bins=eceq_bins)
                    - _compute_ece_q(y_s, p_b_s, n_bins=eceq_bins)
                )
            )
    else:
        labels = np.asarray(group_labels, dtype=object)[valid]
        if len(labels) != len(y):
            return out
        codes, uniques = pd.factorize(labels, sort=False)
        if len(uniques) < 1:
            return out
        group_idx = [np.where(codes == g)[0] for g in range(len(uniques))]
        group_idx = [idx for idx in group_idx if len(idx) > 0]
        if not group_idx:
            return out
        n_groups = len(group_idx)
        out["bootstrap_n_groups"] = int(n_groups)
        out["bootstrap_B"] = int(bootstrap_B)
        delta_logloss_vals = []
        delta_brier_vals = []
        delta_ece_vals = []
        delta_ece_q_vals = []
        for _ in range(int(bootstrap_B)):
            sampled = rng.choice(n_groups, size=n_groups, replace=True)
            sample_idx = np.concatenate([group_idx[g] for g in sampled])
            y_s = y[sample_idx]
            p_m_s = p_m[sample_idx]
            p_b_s = p_b[sample_idx]
            delta_logloss_vals.append(float(_compute_logloss(y_s, p_m_s) - _compute_logloss(y_s, p_b_s)))
            delta_brier_vals.append(float(_compute_brier(y_s, p_m_s) - _compute_brier(y_s, p_b_s)))
            delta_ece_vals.append(float(_compute_ece(y_s, p_m_s, n_bins=n_bins) - _compute_ece(y_s, p_b_s, n_bins=n_bins)))
            delta_ece_q_vals.append(
                float(
                    _compute_ece_q(y_s, p_m_s, n_bins=eceq_bins)
                    - _compute_ece_q(y_s, p_b_s, n_bins=eceq_bins)
                )
            )

    alpha = max(0.1, min(49.9, (100.0 - float(ci_level)) / 2.0))
    low_q = alpha
    high_q = 100.0 - alpha
    if delta_logloss_vals:
        lo, hi = np.percentile(np.asarray(delta_logloss_vals, dtype=float), [low_q, high_q])
        out["delta_logloss_ci_lo"] = float(lo)
        out["delta_logloss_ci_hi"] = float(hi)
    if delta_brier_vals:
        lo, hi = np.percentile(np.asarray(delta_brier_vals, dtype=float), [low_q, high_q])
        out["delta_brier_ci_lo"] = float(lo)
        out["delta_brier_ci_hi"] = float(hi)
    if delta_ece_vals:
        lo, hi = np.percentile(np.asarray(delta_ece_vals, dtype=float), [low_q, high_q])
        out["delta_ece_ci_lo"] = float(lo)
        out["delta_ece_ci_hi"] = float(hi)
    if delta_ece_q_vals:
        lo, hi = np.percentile(np.asarray(delta_ece_q_vals, dtype=float), [low_q, high_q])
        out["delta_ece_q_ci_lo"] = float(lo)
        out["delta_ece_q_ci_hi"] = float(hi)
    return out


def _append_split_metrics_rows(
    metrics_rows: List[Dict[str, Any]],
    *,
    split_name: str,
    split_df: pd.DataFrame,
    y_true: np.ndarray,
    p_model: np.ndarray,
    ticker_col: str,
    bootstrap_ci: bool,
    bootstrap_B: int,
    bootstrap_seed: int,
    bootstrap_group: str,
    allow_iid_bootstrap: bool,
    n_bins: int,
    eceq_bins: int,
    ci_level: int = 95,
) -> None:
    """Append baseline + model rows for one split, optionally with CI columns."""
    if split_df.empty or len(y_true) == 0:
        return
    y = np.asarray(y_true, dtype=float)
    p_m = np.asarray(p_model, dtype=float)
    if len(y) != len(split_df) or len(p_m) != len(split_df):
        return

    valid_model = np.isfinite(y) & np.isfinite(p_m)
    p_baseline_raw = pd.to_numeric(
        split_df.get("pRN", pd.Series(dtype=float)), errors="coerce"
    ).to_numpy(dtype=float)
    valid_base = np.isfinite(p_baseline_raw)
    shared_mask = valid_model
    if valid_base.any():
        shared_mask = valid_model & valid_base
    if shared_mask.sum() <= 0:
        return

    if valid_base.any():
        metrics_rows.append({
            "split": split_name,
            "model": "baseline_pRN",
            "n": int(shared_mask.sum()),
            **_compute_metrics(
                y[shared_mask],
                p_baseline_raw[shared_mask],
                n_bins=n_bins,
                eceq_bins=eceq_bins,
            ),
        })

    model_row: Dict[str, Any] = {
        "split": split_name,
        "model": "logit",
        "n": int(shared_mask.sum()),
        **_compute_metrics(
            y[shared_mask],
            p_m[shared_mask],
            n_bins=n_bins,
            eceq_bins=eceq_bins,
        ),
    }

    if bootstrap_ci and valid_base.any():
        labels, _resolved_group = _resolve_bootstrap_groups(
            split_df,
            ticker_col=ticker_col,
            requested_group=bootstrap_group,
            allow_iid_bootstrap=bool(allow_iid_bootstrap),
        )
        if labels is not None:
            labels = np.asarray(labels, dtype=object)[shared_mask]
        model_row["bootstrap_group_requested"] = str(bootstrap_group)
        model_row["bootstrap_group_resolved"] = str(_resolved_group)
        if _resolved_group == "iid" and str(bootstrap_group).strip().lower() not in {"iid", "auto"}:
            model_row["bootstrap_warning"] = (
                "requested group key unavailable; fell back to iid bootstrap (unsafe default)"
            )
        model_row.update(
            _bootstrap_delta_metric_cis(
                y_true=y[shared_mask],
                p_model=p_m[shared_mask],
                p_baseline=p_baseline_raw[shared_mask],
                group_labels=labels,
                bootstrap_B=int(bootstrap_B),
                bootstrap_seed=int(bootstrap_seed),
                n_bins=n_bins,
                eceq_bins=eceq_bins,
                ci_level=int(ci_level),
            )
        )
    metrics_rows.append(model_row)


def _parse_args(calibrate_args: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Train probabilistic option-chain calibration model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-kind",
        choices=["calibrate"],
        default="calibrate",
        help="Model runner mode.",
    )

    if calibrate_args is None:
        calibrate_args = sys.argv[1:]
    return parser.parse_known_args(calibrate_args)


def _flag_has_value(args: List[str], flag: str) -> bool:
    if flag not in args:
        return False
    idx = args.index(flag)
    return idx + 1 < len(args)


def _get_arg_value(args: List[str], flag: str, *, default: Optional[str] = None) -> Optional[str]:
    if flag not in args:
        return default
    idx = args.index(flag)
    if idx + 1 >= len(args):
        return default
    return args[idx + 1]


def _replace_arg_value(args: List[str], flag: str, value: str) -> List[str]:
    if flag in args:
        idx = args.index(flag)
        if idx + 1 < len(args):
            args[idx + 1] = value
            return args
    return args + [flag, value]


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _config_set(args: argparse.Namespace, attr: str, value: Any) -> None:
    if value is not None:
        setattr(args, attr, value)


def _load_config_json(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("--config-json must point to a JSON object.")
    return payload


def _apply_config_json_overrides(args: argparse.Namespace, payload: Dict[str, Any]) -> None:
    # Top-level controls.
    for key, attr in [
        ("csv", "csv"),
        ("out_dir", "out_dir"),
        ("features", "features"),
        ("categorical_features", "categorical_features"),
        ("target_col", "target_col"),
        ("week_col", "week_col"),
        ("ticker_col", "ticker_col"),
        ("weight_col", "weight_col"),
        ("foundation_tickers", "foundation_tickers"),
        ("foundation_weight", "foundation_weight"),
        ("tdays_allowed", "tdays_allowed"),
        ("asof_dow_allowed", "asof_dow_allowed"),
        ("calibrate", "calibrate"),
        ("c_grid", "C_grid"),
        ("test_weeks", "test_weeks"),
        ("random_state", "random_state"),
        ("ticker_intercepts", "ticker_intercepts"),
        ("group_reweight", "group_reweight"),
        ("grouping_key", "grouping_key"),
        ("train_tickers", "train_tickers"),
        ("max_abs_logm", "max_abs_logm"),
        ("prn_eps", "prn_eps"),
        ("prn_below", "prn_below"),
        ("prn_above", "prn_above"),
        ("selection_objective", "selection_objective"),
        ("bootstrap_group", "bootstrap_group"),
        ("bootstrap_B", "bootstrap_B"),
        ("bootstrap_seed", "bootstrap_seed"),
        ("ci_level", "ci_level"),
        ("run_mode", "run_mode"),
        ("weight_col_strategy", "weight_col_strategy"),
        ("val_split_mode", "val_split_mode"),
        ("val_weeks", "val_weeks"),
        ("calib_frac_of_train", "calib_frac_of_train"),
        ("split_strategy", "split_strategy"),
        ("window_mode", "window_mode"),
        ("train_window_weeks", "train_window_weeks"),
        ("validation_folds", "validation_folds"),
        ("validation_window_weeks", "validation_window_weeks"),
        ("embargo_days", "embargo_days"),
        ("trading_universe_tickers", "trading_universe_tickers"),
        ("trading_universe_upweight", "trading_universe_upweight"),
        ("ticker_balance_mode", "ticker_balance_mode"),
        ("ticker_min_support", "ticker_min_support"),
        ("ticker_min_support_interactions", "ticker_min_support_interactions"),
    ]:
        _config_set(args, attr, payload.get(key))

    for key, attr in [
        ("strict_args", "strict_args"),
        ("add_interactions", "add_interactions"),
        ("drop_prn_extremes", "drop_prn_extremes"),
        ("enable_x_abs_m", "enable_x_abs_m"),
        ("ticker_x_interactions", "ticker_x_interactions"),
        ("group_equalization", "group_equalization"),
        ("split_timeline", "split_timeline"),
        ("per_fold_delta_chart", "per_fold_delta_chart"),
        ("per_group_delta_distribution", "per_group_delta_distribution"),
        ("per_split_reporting", "per_split_reporting"),
        ("per_fold_reporting", "per_fold_reporting"),
        ("bootstrap_ci", "bootstrap_ci"),
        ("skip_test_metrics", "skip_test_metrics"),
        ("allow_defaults", "allow_defaults"),
        ("allow_iid_bootstrap", "allow_iid_bootstrap"),
    ]:
        parsed = _coerce_optional_bool(payload.get(key))
        if parsed is not None:
            setattr(args, attr, parsed)
    fallback_flag = _coerce_optional_bool(payload.get("fallback_to_baseline_if_worse"))
    if fallback_flag is True:
        args.fallback_to_baseline_if_worse = True
        args.no_fallback_to_baseline_if_worse = False
    elif fallback_flag is False:
        args.fallback_to_baseline_if_worse = False
        args.no_fallback_to_baseline_if_worse = True
    auto_drop_flag = _coerce_optional_bool(payload.get("auto_drop_near_constant"))
    if auto_drop_flag is True:
        args.auto_drop_near_constant = True
        args.no_auto_drop_near_constant = False
    elif auto_drop_flag is False:
        args.auto_drop_near_constant = False
        args.no_auto_drop_near_constant = True

    split_cfg = payload.get("split") if isinstance(payload.get("split"), dict) else {}
    reg_cfg = payload.get("regularization") if isinstance(payload.get("regularization"), dict) else {}
    structure_cfg = payload.get("model_structure") if isinstance(payload.get("model_structure"), dict) else {}
    weighting_cfg = payload.get("weighting") if isinstance(payload.get("weighting"), dict) else {}
    bootstrap_cfg = payload.get("bootstrap") if isinstance(payload.get("bootstrap"), dict) else {}
    diagnostics_cfg = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}

    # Nested config values have precedence over top-level aliases.
    for key, attr in [
        ("strategy", "split_strategy"),
        ("window_mode", "window_mode"),
        ("train_window_weeks", "train_window_weeks"),
        ("validation_folds", "validation_folds"),
        ("validation_window_weeks", "validation_window_weeks"),
        ("test_window_weeks", "test_weeks"),
        ("embargo_days", "embargo_days"),
    ]:
        _config_set(args, attr, split_cfg.get(key))

    for key, attr in [
        ("c_grid", "C_grid"),
        ("calibration_method", "calibrate"),
        ("selection_objective", "selection_objective"),
    ]:
        _config_set(args, attr, reg_cfg.get(key))

    for key, attr in [
        ("trading_universe_tickers", "trading_universe_tickers"),
        ("train_tickers", "train_tickers"),
        ("foundation_tickers", "foundation_tickers"),
        ("foundation_weight", "foundation_weight"),
        ("ticker_intercepts", "ticker_intercepts"),
        ("ticker_min_support", "ticker_min_support"),
        ("ticker_min_support_interactions", "ticker_min_support_interactions"),
    ]:
        _config_set(args, attr, structure_cfg.get(key))
    ticker_x_interactions = _coerce_optional_bool(structure_cfg.get("ticker_x_interactions"))
    if ticker_x_interactions is not None:
        args.ticker_x_interactions = ticker_x_interactions

    for key, attr in [
        ("grouping_key", "grouping_key"),
        ("trading_universe_upweight", "trading_universe_upweight"),
        ("ticker_balance_mode", "ticker_balance_mode"),
    ]:
        _config_set(args, attr, weighting_cfg.get(key))
    group_equalization = _coerce_optional_bool(weighting_cfg.get("group_equalization"))
    if group_equalization is True:
        args.group_equalization = True
        args.no_group_equalization = False
    elif group_equalization is False:
        args.group_equalization = False
        args.no_group_equalization = True

    base_weight_source = str(weighting_cfg.get("base_weight_source") or "").strip().lower()
    if base_weight_source == "uniform":
        args.weight_col = "uniform"

    for key, attr in [
        ("bootstrap_group", "bootstrap_group"),
        ("bootstrap_b", "bootstrap_B"),
        ("bootstrap_seed", "bootstrap_seed"),
        ("ci_level", "ci_level"),
    ]:
        _config_set(args, attr, bootstrap_cfg.get(key))
    for key, attr in [
        ("bootstrap_ci", "bootstrap_ci"),
        ("per_split_reporting", "per_split_reporting"),
        ("per_fold_reporting", "per_fold_reporting"),
        ("allow_iid_bootstrap", "allow_iid_bootstrap"),
    ]:
        parsed = _coerce_optional_bool(bootstrap_cfg.get(key))
        if parsed is not None:
            setattr(args, attr, parsed)

    for key, attr in [
        ("split_timeline", "split_timeline"),
        ("per_fold_delta_chart", "per_fold_delta_chart"),
        ("per_group_delta_distribution", "per_group_delta_distribution"),
    ]:
        parsed = _coerce_optional_bool(diagnostics_cfg.get(key))
        if parsed is not None:
            setattr(args, attr, parsed)

    strategy = str(getattr(args, "weight_col_strategy", "") or "").strip().lower()
    if strategy in {"uniform", "weight_final", "sample_weight_final"}:
        args.weight_col = strategy
    elif strategy == "auto" and not getattr(args, "weight_col", None):
        args.weight_col = "weight_final"


def build_args_from_config(
    payload: Dict[str, Any],
    *,
    calibrate_args: Optional[List[str]] = None,
) -> Tuple[argparse.Namespace, List[str]]:
    parser = _build_calibration_arg_parser()
    args, unknown = parser.parse_known_args(calibrate_args or [])
    _apply_config_json_overrides(args, payload)
    return args, unknown


def _normalize_csv_like_string(value: Any) -> str:
    return ",".join([token.strip() for token in str(value).split(",") if token.strip()])


def _canonicalize_arg_value(attr: str, value: Any) -> Any:
    if value is None:
        return None
    bool_attrs = {
        "strict_args",
        "add_interactions",
        "drop_prn_extremes",
        "enable_x_abs_m",
        "ticker_x_interactions",
        "group_equalization",
        "split_timeline",
        "per_fold_delta_chart",
        "per_group_delta_distribution",
        "per_split_reporting",
        "per_fold_reporting",
        "bootstrap_ci",
        "skip_test_metrics",
        "allow_defaults",
        "allow_iid_bootstrap",
    }
    int_attrs = {
        "test_weeks",
        "random_state",
        "bootstrap_B",
        "bootstrap_seed",
        "ci_level",
        "val_weeks",
        "train_window_weeks",
        "validation_folds",
        "validation_window_weeks",
        "embargo_days",
        "ticker_min_support",
        "ticker_min_support_interactions",
    }
    float_attrs = {
        "foundation_weight",
        "max_abs_logm",
        "prn_eps",
        "prn_below",
        "prn_above",
        "calib_frac_of_train",
        "trading_universe_upweight",
    }
    if attr in bool_attrs:
        parsed = _coerce_optional_bool(value)
        return bool(parsed) if parsed is not None else None
    if attr in int_attrs:
        try:
            return int(value)
        except Exception:
            return value
    if attr in float_attrs:
        try:
            parsed = float(value)
            return parsed if np.isfinite(parsed) else value
        except Exception:
            return value
    if attr == "C_grid":
        return _normalize_csv_like_string(value)
    if attr in {"features", "categorical_features"}:
        return _normalize_csv_like_string(value)
    if isinstance(value, str):
        return value.strip()
    return value


def _collect_requested_arg_expectations(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    expected: Dict[str, Dict[str, Any]] = {}

    def _set(attr: str, value: Any, path: str) -> None:
        if value is None:
            return
        expected[attr] = {
            "path": path,
            "requested": _canonicalize_arg_value(attr, value),
        }

    top_level_map = [
        ("csv", "csv"),
        ("out_dir", "out_dir"),
        ("features", "features"),
        ("categorical_features", "categorical_features"),
        ("target_col", "target_col"),
        ("week_col", "week_col"),
        ("ticker_col", "ticker_col"),
        ("weight_col", "weight_col"),
        ("weight_col_strategy", "weight_col_strategy"),
        ("foundation_tickers", "foundation_tickers"),
        ("foundation_weight", "foundation_weight"),
        ("train_tickers", "train_tickers"),
        ("calibrate", "calibrate"),
        ("c_grid", "C_grid"),
        ("test_weeks", "test_weeks"),
        ("random_state", "random_state"),
        ("ticker_intercepts", "ticker_intercepts"),
        ("group_reweight", "group_reweight"),
        ("grouping_key", "grouping_key"),
        ("max_abs_logm", "max_abs_logm"),
        ("prn_eps", "prn_eps"),
        ("prn_below", "prn_below"),
        ("prn_above", "prn_above"),
        ("selection_objective", "selection_objective"),
        ("bootstrap_group", "bootstrap_group"),
        ("bootstrap_B", "bootstrap_B"),
        ("bootstrap_seed", "bootstrap_seed"),
        ("ci_level", "ci_level"),
        ("run_mode", "run_mode"),
        ("split_strategy", "split_strategy"),
        ("window_mode", "window_mode"),
        ("train_window_weeks", "train_window_weeks"),
        ("validation_folds", "validation_folds"),
        ("validation_window_weeks", "validation_window_weeks"),
        ("embargo_days", "embargo_days"),
        ("val_split_mode", "val_split_mode"),
        ("val_weeks", "val_weeks"),
        ("calib_frac_of_train", "calib_frac_of_train"),
        ("trading_universe_tickers", "trading_universe_tickers"),
        ("trading_universe_upweight", "trading_universe_upweight"),
        ("ticker_balance_mode", "ticker_balance_mode"),
        ("ticker_min_support", "ticker_min_support"),
        ("ticker_min_support_interactions", "ticker_min_support_interactions"),
    ]
    for key, attr in top_level_map:
        if key in payload:
            _set(attr, payload.get(key), key)

    top_level_bool_map = [
        ("strict_args", "strict_args"),
        ("add_interactions", "add_interactions"),
        ("drop_prn_extremes", "drop_prn_extremes"),
        ("enable_x_abs_m", "enable_x_abs_m"),
        ("ticker_x_interactions", "ticker_x_interactions"),
        ("group_equalization", "group_equalization"),
        ("split_timeline", "split_timeline"),
        ("per_fold_delta_chart", "per_fold_delta_chart"),
        ("per_group_delta_distribution", "per_group_delta_distribution"),
        ("per_split_reporting", "per_split_reporting"),
        ("per_fold_reporting", "per_fold_reporting"),
        ("bootstrap_ci", "bootstrap_ci"),
        ("skip_test_metrics", "skip_test_metrics"),
        ("allow_defaults", "allow_defaults"),
        ("allow_iid_bootstrap", "allow_iid_bootstrap"),
    ]
    for key, attr in top_level_bool_map:
        if key in payload:
            _set(attr, _coerce_optional_bool(payload.get(key)), key)

    split_cfg = payload.get("split") if isinstance(payload.get("split"), dict) else {}
    for key, attr in [
        ("strategy", "split_strategy"),
        ("window_mode", "window_mode"),
        ("train_window_weeks", "train_window_weeks"),
        ("validation_folds", "validation_folds"),
        ("validation_window_weeks", "validation_window_weeks"),
        ("test_window_weeks", "test_weeks"),
        ("embargo_days", "embargo_days"),
    ]:
        if key in split_cfg:
            _set(attr, split_cfg.get(key), f"split.{key}")

    reg_cfg = payload.get("regularization") if isinstance(payload.get("regularization"), dict) else {}
    for key, attr in [
        ("c_grid", "C_grid"),
        ("calibration_method", "calibrate"),
        ("selection_objective", "selection_objective"),
    ]:
        if key in reg_cfg:
            _set(attr, reg_cfg.get(key), f"regularization.{key}")

    structure_cfg = payload.get("model_structure") if isinstance(payload.get("model_structure"), dict) else {}
    for key, attr in [
        ("trading_universe_tickers", "trading_universe_tickers"),
        ("train_tickers", "train_tickers"),
        ("foundation_tickers", "foundation_tickers"),
        ("foundation_weight", "foundation_weight"),
        ("ticker_intercepts", "ticker_intercepts"),
        ("ticker_min_support", "ticker_min_support"),
        ("ticker_min_support_interactions", "ticker_min_support_interactions"),
    ]:
        if key in structure_cfg:
            _set(attr, structure_cfg.get(key), f"model_structure.{key}")
    if "ticker_x_interactions" in structure_cfg:
        _set(
            "ticker_x_interactions",
            _coerce_optional_bool(structure_cfg.get("ticker_x_interactions")),
            "model_structure.ticker_x_interactions",
        )

    weighting_cfg = payload.get("weighting") if isinstance(payload.get("weighting"), dict) else {}
    for key, attr in [
        ("grouping_key", "grouping_key"),
        ("trading_universe_upweight", "trading_universe_upweight"),
        ("ticker_balance_mode", "ticker_balance_mode"),
    ]:
        if key in weighting_cfg:
            _set(attr, weighting_cfg.get(key), f"weighting.{key}")
    if "group_equalization" in weighting_cfg:
        _set(
            "group_equalization",
            _coerce_optional_bool(weighting_cfg.get("group_equalization")),
            "weighting.group_equalization",
        )
    if str(weighting_cfg.get("base_weight_source") or "").strip().lower() == "uniform":
        _set("weight_col", "uniform", "weighting.base_weight_source")

    bootstrap_cfg = payload.get("bootstrap") if isinstance(payload.get("bootstrap"), dict) else {}
    for key, attr in [
        ("bootstrap_group", "bootstrap_group"),
        ("bootstrap_b", "bootstrap_B"),
        ("bootstrap_seed", "bootstrap_seed"),
        ("ci_level", "ci_level"),
    ]:
        if key in bootstrap_cfg:
            _set(attr, bootstrap_cfg.get(key), f"bootstrap.{key}")
    for key, attr in [
        ("bootstrap_ci", "bootstrap_ci"),
        ("per_split_reporting", "per_split_reporting"),
        ("per_fold_reporting", "per_fold_reporting"),
        ("allow_iid_bootstrap", "allow_iid_bootstrap"),
    ]:
        if key in bootstrap_cfg:
            _set(attr, _coerce_optional_bool(bootstrap_cfg.get(key)), f"bootstrap.{key}")

    diagnostics_cfg = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), dict) else {}
    for key, attr in [
        ("split_timeline", "split_timeline"),
        ("per_fold_delta_chart", "per_fold_delta_chart"),
        ("per_group_delta_distribution", "per_group_delta_distribution"),
    ]:
        if key in diagnostics_cfg:
            _set(attr, _coerce_optional_bool(diagnostics_cfg.get(key)), f"diagnostics.{key}")

    return expected


def _resolved_args_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "csv": args.csv,
        "out_dir": args.out_dir,
        "features": _normalize_csv_like_string(args.features),
        "categorical_features": _normalize_csv_like_string(args.categorical_features),
        "target_col": args.target_col,
        "week_col": args.week_col,
        "ticker_col": args.ticker_col,
        "weight_col": args.weight_col,
        "weight_col_strategy": args.weight_col_strategy,
        "run_mode": args.run_mode,
        "split": {
            "strategy": args.split_strategy,
            "window_mode": args.window_mode,
            "train_window_weeks": int(args.train_window_weeks),
            "validation_folds": int(args.validation_folds),
            "validation_window_weeks": int(args.validation_window_weeks),
            "test_window_weeks": int(args.test_weeks),
            "embargo_days": int(args.embargo_days),
            "val_split_mode": args.val_split_mode,
            "val_weeks": int(args.val_weeks),
        },
        "regularization": {
            "c_grid": _normalize_csv_like_string(args.C_grid),
            "calibration_method": args.calibrate,
            "selection_objective": args.selection_objective,
        },
        "model_structure": {
            "trading_universe_tickers": args.trading_universe_tickers,
            "train_tickers": args.train_tickers,
            "foundation_tickers": args.foundation_tickers,
            "foundation_weight": float(args.foundation_weight),
            "ticker_intercepts": args.ticker_intercepts,
            "ticker_x_interactions": bool(args.ticker_x_interactions),
            "ticker_min_support": args.ticker_min_support,
            "ticker_min_support_interactions": args.ticker_min_support_interactions,
        },
        "weighting": {
            "group_reweight": args.group_reweight,
            "grouping_key": args.grouping_key,
            "group_equalization": bool(args.group_equalization),
            "trading_universe_upweight": float(args.trading_universe_upweight),
            "ticker_balance_mode": args.ticker_balance_mode,
        },
        "bootstrap": {
            "bootstrap_ci": bool(args.bootstrap_ci),
            "bootstrap_group": args.bootstrap_group,
            "bootstrap_B": int(args.bootstrap_B),
            "bootstrap_seed": int(args.bootstrap_seed),
            "ci_level": int(args.ci_level),
            "allow_iid_bootstrap": bool(args.allow_iid_bootstrap),
            "per_split_reporting": bool(args.per_split_reporting),
            "per_fold_reporting": bool(args.per_fold_reporting),
        },
        "diagnostics": {
            "split_timeline": bool(args.split_timeline),
            "per_fold_delta_chart": bool(args.per_fold_delta_chart),
            "per_group_delta_distribution": bool(args.per_group_delta_distribution),
            "skip_test_metrics": bool(args.skip_test_metrics),
        },
        "allow_defaults": bool(args.allow_defaults),
    }


def _values_match(requested: Any, resolved: Any, *, tol: float = 1e-12) -> bool:
    if requested is None and resolved is None:
        return True
    if isinstance(requested, (float, int)) and isinstance(resolved, (float, int)):
        req = float(requested)
        res = float(resolved)
        if not (np.isfinite(req) and np.isfinite(res)):
            return requested == resolved
        return abs(req - res) <= tol
    return requested == resolved


def _diff_requested_vs_resolved_config(
    payload: Dict[str, Any],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    expected = _collect_requested_arg_expectations(payload)
    diffs: List[Dict[str, Any]] = []
    for attr, meta in expected.items():
        requested = meta.get("requested")
        resolved = _canonicalize_arg_value(attr, getattr(args, attr, None))
        if _values_match(requested, resolved):
            continue
        diffs.append(
            {
                "field": meta.get("path"),
                "arg": attr,
                "requested": requested,
                "resolved": resolved,
            }
        )
    return diffs


def _warn_or_fail_unconsumed_args(
    *,
    unknown_args: List[str],
    strict: bool,
) -> None:
    if not unknown_args:
        return
    msg = (
        "[base-model] Unconsumed calibrator args (possible no-op settings): "
        + " ".join(str(x) for x in unknown_args)
    )
    if strict:
        raise ValueError(msg)
    print(f"[WARN] {msg}")


def _resolve_first_available_datetime_col(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    for col in candidates:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if parsed.notna().any():
            return col
    return None


def _resolve_embargo_date_col(df: pd.DataFrame) -> Optional[str]:
    return _resolve_first_available_datetime_col(
        df,
        [
            "asof_date",
            "snapshot_date",
            "asof_datetime_utc",
            "asof_datetime",
            "snapshot_time_utc",
            "timestamp_utc",
            "asof_ts",
            "asof_time",
            "snapshot_time",
        ],
    )


def _apply_embargo_by_days(
    train_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    *,
    embargo_days: int,
    date_col: str,
) -> Tuple[pd.DataFrame, int]:
    if train_df.empty or boundary_df.empty or embargo_days <= 0:
        return train_df, 0
    if date_col not in train_df.columns or date_col not in boundary_df.columns:
        return train_df, 0
    boundary_dates = pd.to_datetime(boundary_df[date_col], errors="coerce", utc=True)
    if boundary_dates.notna().sum() == 0:
        return train_df, 0
    boundary_start = boundary_dates.min()
    cutoff = boundary_start - pd.Timedelta(days=int(embargo_days))
    train_dates = pd.to_datetime(train_df[date_col], errors="coerce", utc=True)
    keep_mask = train_dates < cutoff
    dropped = int((~keep_mask).sum())
    if not keep_mask.any():
        return train_df.iloc[:0].copy(), dropped
    return train_df.loc[keep_mask].copy(), dropped


def _apply_embargo_by_weeks(
    train_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    *,
    embargo_days: int,
    week_col: str,
) -> Tuple[pd.DataFrame, int]:
    if train_df.empty or boundary_df.empty or embargo_days <= 0:
        return train_df, 0
    if week_col not in train_df.columns or week_col not in boundary_df.columns:
        return train_df, 0
    boundary_weeks = pd.to_datetime(boundary_df[week_col], errors="coerce").dt.normalize()
    if boundary_weeks.notna().sum() == 0:
        return train_df, 0
    embargo_weeks = int(np.ceil(float(embargo_days) / 7.0))
    if embargo_weeks <= 0:
        return train_df, 0
    cutoff = boundary_weeks.min() - pd.Timedelta(weeks=embargo_weeks)
    train_weeks = pd.to_datetime(train_df[week_col], errors="coerce").dt.normalize()
    keep_mask = train_weeks < cutoff
    dropped = int((~keep_mask).sum())
    if not keep_mask.any():
        return train_df.iloc[:0].copy(), dropped
    return train_df.loc[keep_mask].copy(), dropped


def _parse_asof_dow_allowed(value: Optional[str]) -> List[int]:
    if not value:
        return []
    mapping = {
        "mon": 0,
        "monday": 0,
        "tue": 1,
        "tues": 1,
        "tuesday": 1,
        "wed": 2,
        "wednesday": 2,
        "thu": 3,
        "thur": 3,
        "thurs": 3,
        "thursday": 3,
    }
    out: List[int] = []
    for token in str(value).split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if tok.isdigit():
            wd = int(tok)
            if wd < 0 or wd > 3:
                raise ValueError(
                    f"Invalid asof_dow value '{tok}' (allowed: Mon=0, Tue=1, Wed=2, Thu=3)."
                )
            out.append(wd)
            continue
        if tok not in mapping:
            raise ValueError(
                f"Invalid asof_dow value '{tok}' (allowed names: monday, tuesday, wednesday, thursday)."
            )
        out.append(mapping[tok])
    uniq = sorted(set(out))
    if len(uniq) > 1:
        raise ValueError("Only one as-of weekday can be selected for a calibration run.")
    return uniq


def _sort_base_calibration_df(
    df: pd.DataFrame,
    *,
    week_col: str,
    ticker_col: str,
) -> pd.DataFrame:
    frame = df.copy()
    date_candidates = [
        "asof_date",
        "snapshot_date",
        "asof_datetime_utc",
        "asof_datetime",
        "snapshot_time_utc",
        "timestamp_utc",
        week_col,
    ]
    for col in date_candidates:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", utc=True)
    sort_cols: List[str] = []
    for col in date_candidates:
        if col in frame.columns and col not in sort_cols:
            sort_cols.append(col)
    for col in [ticker_col, "ticker", "K", "threshold"]:
        if col in frame.columns and col not in sort_cols:
            sort_cols.append(col)
    if sort_cols:
        frame = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
    return frame


def _series_overlap_count(left: Optional[pd.Series], right: Optional[pd.Series]) -> Optional[int]:
    if left is None or right is None:
        return None
    left_vals = set(pd.Series(left).dropna().astype(str).tolist())
    right_vals = set(pd.Series(right).dropna().astype(str).tolist())
    return int(len(left_vals & right_vals))


def _build_contract_id_series(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    ticker_col = "ticker" if "ticker" in df.columns else None
    strike_col = None
    for cand in ["K", "threshold"]:
        if cand in df.columns:
            strike_col = cand
            break
    expiry_col = None
    for cand in [
        "expiry_date",
        "option_expiration_used",
        "expiry_close_date_used",
        "option_expiration_requested",
        "event_end_date",
    ]:
        if cand in df.columns:
            expiry_col = cand
            break
    if ticker_col is None or strike_col is None or expiry_col is None:
        return None
    ticker = df[ticker_col].astype("string").str.upper().fillna("UNKNOWN")
    expiry = pd.to_datetime(df[expiry_col], errors="coerce").astype("string").fillna("NaT")
    strike = pd.to_numeric(df[strike_col], errors="coerce").round(8).astype("string").fillna("NaN")
    return ticker + "|" + expiry + "|" + strike


def _week_series_for_overlap(df: pd.DataFrame, week_col: str) -> Optional[pd.Series]:
    if week_col not in df.columns:
        return None
    return pd.to_datetime(df[week_col], errors="coerce").dt.strftime("%Y-%m-%d")


def _build_split_overlap_diagnostics(
    *,
    train_fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    week_col: str,
    split_group_key: Optional[str] = None,
    split_group_series: Optional[pd.Series] = None,
) -> Dict[str, Optional[int]]:
    train_fit_contract = _build_contract_id_series(train_fit_df)
    val_contract = _build_contract_id_series(val_df)
    train_contract = _build_contract_id_series(train_df)
    test_contract = _build_contract_id_series(test_df)
    train_fit_group = train_fit_df["group_id"] if "group_id" in train_fit_df.columns else None
    val_group = val_df["group_id"] if "group_id" in val_df.columns else None
    train_group = train_df["group_id"] if "group_id" in train_df.columns else None
    test_group = test_df["group_id"] if "group_id" in test_df.columns else None
    overlaps = {
        "contract_id_train_val": _series_overlap_count(train_fit_contract, val_contract),
        "contract_id_train_test": _series_overlap_count(train_contract, test_contract),
        "group_id_train_val": _series_overlap_count(train_fit_group, val_group),
        "group_id_train_test": _series_overlap_count(train_group, test_group),
        "week_train_val": _series_overlap_count(
            _week_series_for_overlap(train_fit_df, week_col),
            _week_series_for_overlap(val_df, week_col),
        ),
        "week_train_test": _series_overlap_count(
            _week_series_for_overlap(train_df, week_col),
            _week_series_for_overlap(test_df, week_col),
        ),
    }
    if split_group_key and split_group_series is not None:
        overlaps["split_group_key"] = split_group_key
        overlaps["split_group_key_train_val"] = _series_overlap_count(
            split_group_series.loc[train_fit_df.index] if not train_fit_df.empty else None,
            split_group_series.loc[val_df.index] if not val_df.empty else None,
        )
        overlaps["split_group_key_train_test"] = _series_overlap_count(
            split_group_series.loc[train_df.index] if not train_df.empty else None,
            split_group_series.loc[test_df.index] if not test_df.empty else None,
        )
    return overlaps


def _week_range(frame: pd.DataFrame, week_col: str) -> Optional[List[str]]:
    if frame is None or frame.empty or week_col not in frame.columns:
        return None
    weeks = pd.to_datetime(frame[week_col], errors="coerce").dropna().sort_values()
    if weeks.empty:
        return None
    return [
        weeks.iloc[0].date().isoformat(),
        weeks.iloc[-1].date().isoformat(),
    ]


def _count_unique_weeks(frame: pd.DataFrame, week_col: str) -> Optional[int]:
    if frame is None or frame.empty or week_col not in frame.columns:
        return None
    weeks = pd.to_datetime(frame[week_col], errors="coerce").dt.normalize().dropna()
    if weeks.empty:
        return None
    return int(weeks.nunique())


def _compute_embargo_gap_days_actual(
    *,
    train_frame: pd.DataFrame,
    boundary_frame: pd.DataFrame,
    embargo_mode: str,
    embargo_date_col: Optional[str],
    week_col: str,
) -> Optional[int]:
    if train_frame.empty or boundary_frame.empty:
        return None
    if embargo_mode == "days" and embargo_date_col:
        if embargo_date_col not in train_frame.columns or embargo_date_col not in boundary_frame.columns:
            return None
        train_dates = pd.to_datetime(train_frame[embargo_date_col], errors="coerce", utc=True).dropna()
        boundary_dates = pd.to_datetime(boundary_frame[embargo_date_col], errors="coerce", utc=True).dropna()
        if train_dates.empty or boundary_dates.empty:
            return None
        return int((boundary_dates.min() - train_dates.max()).days)
    if week_col not in train_frame.columns or week_col not in boundary_frame.columns:
        return None
    train_weeks = pd.to_datetime(train_frame[week_col], errors="coerce").dt.normalize().dropna()
    boundary_weeks = pd.to_datetime(boundary_frame[week_col], errors="coerce").dt.normalize().dropna()
    if train_weeks.empty or boundary_weeks.empty:
        return None
    return int((boundary_weeks.min() - train_weeks.max()).days)


def _date_range_from_candidates(frame: pd.DataFrame, candidates: List[str]) -> Optional[List[str]]:
    if frame is None or frame.empty:
        return None
    col = _resolve_first_available_datetime_col(frame, candidates)
    if not col:
        return None
    vals = pd.to_datetime(frame[col], errors="coerce", utc=True).dropna().sort_values()
    if vals.empty:
        return None
    return [
        vals.iloc[0].isoformat(),
        vals.iloc[-1].isoformat(),
    ]


def _resolve_base_weight_vector(df: pd.DataFrame, weight_col: Optional[str]) -> Tuple[np.ndarray, str]:
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=float), "none"
    if not weight_col:
        return np.ones(n, dtype=float), "none"
    selected_col = str(weight_col)
    if selected_col.strip().lower() in {"uniform", "none"}:
        return np.ones(n, dtype=float), "uniform"
    if selected_col not in df.columns:
        if selected_col == "weight_final" and "sample_weight_final" in df.columns:
            selected_col = "sample_weight_final"
            print(
                "[WARN] [base-model] weight_col 'weight_final' not found; "
                "falling back to legacy 'sample_weight_final'."
            )
        else:
            print(f"[WARN] [base-model] weight_col {weight_col!r} not found; using uniform weights.")
            return np.ones(n, dtype=float), "missing"

    weights = pd.to_numeric(df[selected_col], errors="coerce").to_numpy(dtype=float)
    invalid = ~np.isfinite(weights) | (weights <= 0)
    if invalid.any():
        print(
            f"[WARN] [base-model] weight_col {selected_col!r} has {int(invalid.sum())} invalid rows; "
            "replacing with 1.0."
        )
        weights = weights.copy()
        weights[invalid] = 1.0
    if selected_col != str(weight_col):
        return weights, f"fallback:{selected_col}"
    return weights, "column"


def _renorm_weights_mean_one(weights: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    arr = np.asarray(weights, dtype=float).copy()
    if arr.size == 0:
        return arr
    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        return arr
    mean_w = float(np.mean(arr[valid]))
    if mean_w <= tol:
        return arr
    arr[valid] = arr[valid] / mean_w
    return arr


def _equalize_group_sums_to_one(
    weights: np.ndarray,
    group_key: Optional[pd.Series],
) -> np.ndarray:
    arr = np.asarray(weights, dtype=float).copy()
    if group_key is None or arr.size == 0 or len(group_key) != len(arr):
        return arr
    groups = group_key.astype("string").fillna("NA")
    grouped = pd.DataFrame({"group": groups, "w": arr})
    totals = grouped.groupby("group", dropna=False)["w"].sum()
    if totals.empty:
        return arr
    scales = totals.replace(0.0, np.nan)
    mapped = groups.map(scales).astype(float).to_numpy(dtype=float)
    valid = np.isfinite(arr) & np.isfinite(mapped) & (mapped > 0)
    if valid.any():
        arr[valid] = arr[valid] / mapped[valid]
    return arr


def _fit_pipe_with_optional_weights(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> Pipeline:
    fit_kwargs: Dict[str, Any] = {}
    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=float).ravel()
        if len(sw) == len(y):
            fit_kwargs["clf__sample_weight"] = sw
    pipe.fit(X, y, **fit_kwargs)
    return pipe


def _compute_weighted_logloss(
    y_true: np.ndarray,
    p: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    if sample_weight is None:
        return _compute_logloss(y, p)
    w = np.asarray(sample_weight, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return _compute_logloss(y, p)
    yv = y[valid]
    pv = p[valid]
    wv = w[valid]
    denom = float(np.sum(wv))
    if denom <= 0:
        return _compute_logloss(yv, pv)
    return float(-np.sum(wv * (yv * np.log(pv) + (1.0 - yv) * np.log(1.0 - pv))) / denom)


def _parse_and_validate_c_grid(value: str) -> List[float]:
    tokens = [t.strip() for t in str(value).split(",") if t.strip()]
    if not tokens:
        raise ValueError("C-grid cannot be empty.")
    parsed: List[float] = []
    for token in tokens:
        try:
            c_val = float(token)
        except Exception as exc:
            raise ValueError(f"Invalid C-grid value '{token}'.") from exc
        if not np.isfinite(c_val):
            raise ValueError(f"C-grid value '{token}' must be finite.")
        if c_val <= 0:
            raise ValueError(f"C-grid value '{token}' must be > 0.")
        parsed.append(c_val)
    return sorted(set(parsed))


def _class_count(y: np.ndarray) -> int:
    vals = np.asarray(y, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0
    return int(np.unique(vals).size)


def _collect_unsupported_compat_args(args: argparse.Namespace) -> Dict[str, Any]:
    ignored: Dict[str, Any] = {}
    if args.train_decay_half_life_weeks is not None:
        ignored["train_decay_half_life_weeks"] = args.train_decay_half_life_weeks
    if args.fit_weight_renorm is not None:
        ignored["fit_weight_renorm"] = args.fit_weight_renorm
    if args.metrics_top_tickers is not None:
        ignored["metrics_top_tickers"] = args.metrics_top_tickers
    if bool(args.add_interactions):
        ignored["add_interactions"] = True
    if bool(args.auto_drop_near_constant):
        ignored["auto_drop_near_constant"] = True
    if bool(args.no_auto_drop_near_constant):
        ignored["no_auto_drop_near_constant"] = True
    if bool(args.fallback_to_baseline_if_worse):
        ignored["fallback_to_baseline_if_worse"] = True
    if bool(args.no_fallback_to_baseline_if_worse):
        ignored["no_fallback_to_baseline_if_worse"] = True
    return ignored


def _build_group_key_for_reweight(
    df: pd.DataFrame,
    preferred_key: Optional[str] = None,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    if df.empty:
        return None, None
    if preferred_key and preferred_key in df.columns:
        preferred = df[preferred_key].astype("string").fillna("NA")
        if preferred.notna().any():
            return preferred, preferred_key
    if "group_id" in df.columns:
        series = df["group_id"].astype("string").fillna("NA")
        if series.notna().any():
            return series, "group_id"
    fallback = build_chain_group_id(df)
    if fallback is not None:
        return fallback.astype("string").fillna("NA"), "build_chain_group_id"
    return None, None


def _resolve_group_key_for_splits(
    df: pd.DataFrame,
    preferred_key: Optional[str] = None,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    if df.empty:
        return None, None
    if preferred_key:
        key = str(preferred_key).strip()
        if key == "contract_id":
            contract = _build_contract_id_series(df)
            if contract is not None:
                return pd.Series(contract).astype("string").fillna("NA"), "contract_id"
        if key in df.columns:
            series = df[key].astype("string").fillna("NA")
            if series.notna().any():
                return series, key
    if "group_id" in df.columns:
        series = df["group_id"].astype("string").fillna("NA")
        if series.notna().any():
            return series, "group_id"
    contract = _build_contract_id_series(df)
    if contract is not None:
        return pd.Series(contract).astype("string").fillna("NA"), "contract_id"
    fallback = build_chain_group_id(df)
    if fallback is not None:
        return fallback.astype("string").fillna("NA"), "build_chain_group_id"
    return None, None


def _normalize_group_reweight_mode(value: Optional[str]) -> str:
    if not value:
        return "none"
    mode = str(value).strip().lower()
    if mode == "chain":
        return "chain_snapshot"
    if mode not in {"none", "chain_snapshot"}:
        raise ValueError(f"Invalid --group-reweight={value!r}; expected none|chain_snapshot.")
    return mode


def _apply_asof_dow_filter(df: pd.DataFrame, asof_dow_allowed: List[int]) -> Tuple[pd.DataFrame, Optional[str]]:
    if not asof_dow_allowed:
        return df, None
    asof_col = _resolve_first_available_datetime_col(
        df,
        ["asof_date", "asof_target", "snapshot_date", "asof_datetime_utc", "asof_datetime", "snapshot_time_utc"],
    )
    if not asof_col:
        print("[WARN] [base-model] --asof-dow-allowed ignored: no asof date column found.")
        return df, None
    parsed = pd.to_datetime(df[asof_col], errors="coerce", utc=True)
    mask = parsed.dt.weekday.isin(asof_dow_allowed)
    return df[mask.fillna(False)].copy(), asof_col


def _resolve_val_split(
    train_df: pd.DataFrame,
    *,
    week_col: str,
    val_split_mode: str,
    val_weeks_override: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "mode": val_split_mode,
        "used_week_groups": False,
        "val_weeks": None,
        "fallback_reason": None,
    }
    if train_df.empty:
        return train_df.copy(), train_df.copy(), info

    if (
        val_split_mode == "week_group"
        and week_col in train_df.columns
        and pd.to_datetime(train_df[week_col], errors="coerce").notna().any()
    ):
        weeks = pd.to_datetime(train_df[week_col], errors="coerce").dt.normalize()
        train_df_local = train_df.copy()
        train_df_local[week_col] = weeks
        uniq = train_df_local[week_col].dropna().sort_values().unique()
        if len(uniq) >= 2:
            if val_weeks_override is not None and val_weeks_override > 0:
                n_val_weeks = min(len(uniq) - 1, int(val_weeks_override))
            else:
                n_val_weeks = max(1, int(round(len(uniq) * 0.2)))
                n_val_weeks = min(n_val_weeks, len(uniq) - 1)
            val_week_set = set(uniq[-n_val_weeks:])
            val_mask = train_df_local[week_col].isin(val_week_set)
            train_fit_df = train_df_local[~val_mask].copy()
            val_df = train_df_local[val_mask].copy()
            if not train_fit_df.empty and not val_df.empty:
                info["used_week_groups"] = True
                info["val_weeks"] = int(n_val_weeks)
                return train_fit_df, val_df, info
            info["fallback_reason"] = "empty_train_or_val_after_week_split"
            train_df = train_df_local
        else:
            info["fallback_reason"] = "insufficient_unique_weeks"
            train_df = train_df_local
    elif val_split_mode == "week_group":
        info["fallback_reason"] = "missing_week_col"

    # Fallback: row-tail split
    val_cut = max(1, int(len(train_df) * 0.8))
    if val_cut >= len(train_df):
        val_cut = max(0, len(train_df) - 1)
    train_fit_df = train_df.iloc[:val_cut].copy()
    val_df = train_df.iloc[val_cut:].copy()
    if train_fit_df.empty and not val_df.empty:
        train_fit_df = train_df.iloc[: max(0, len(train_df) - 1)].copy()
        val_df = train_df.iloc[max(0, len(train_df) - 1):].copy()
    info["mode"] = "row_tail"
    return train_fit_df, val_df, info


def _resolve_walk_forward_folds(
    train_df: pd.DataFrame,
    *,
    week_col: str,
    train_window_weeks: int,
    validation_window_weeks: int,
    validation_folds: int,
    window_mode: str,
    embargo_days: int,
    split_group_series: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    if train_df.empty or week_col not in train_df.columns:
        raise ValueError("Walk-forward split requires non-empty data and a valid week column.")

    weeks = pd.to_datetime(train_df[week_col], errors="coerce").dt.normalize()
    if weeks.notna().sum() == 0:
        raise ValueError("Walk-forward split requires a valid week column with parseable dates.")

    unique_weeks = sorted(pd.Series(weeks.dropna().unique()).tolist())
    if len(unique_weeks) < 3:
        raise ValueError("Walk-forward split requires at least 3 unique weeks.")

    val_weeks = max(1, int(validation_window_weeks))
    folds = max(1, int(validation_folds))
    train_weeks = max(1, int(train_window_weeks))
    min_train_weeks = 8
    embargo_days = max(0, int(embargo_days))
    embargo_date_col = _resolve_embargo_date_col(train_df) if embargo_days > 0 else None
    embargo_mode = "days" if (embargo_days > 0 and embargo_date_col) else "weeks"
    embargo_weeks = int(np.ceil(float(embargo_days) / 7.0)) if (embargo_days > 0 and embargo_mode == "weeks") else 0

    out: List[Dict[str, Any]] = []
    right = len(unique_weeks)
    for fold_idx in range(folds):
        val_end = right
        if val_end <= 0:
            raise ValueError(
                f"Unable to satisfy requested folds ({folds}) with available history."
            )
        val_start = max(0, val_end - val_weeks)
        val_weeks_used = val_end - val_start
        if val_weeks_used < 1:
            raise ValueError(
                f"Unable to satisfy requested folds ({folds}) with available history."
            )

        # Adjust validation window to ensure minimum train window is met.
        while True:
            train_end = val_start - embargo_weeks
            if window_mode == "expanding":
                train_start = 0
            else:
                train_start = max(0, train_end - train_weeks)
            train_weeks_used = train_end - train_start
            if train_weeks_used >= min_train_weeks:
                break
            if val_weeks_used <= 1:
                break
            val_start += 1
            val_weeks_used = val_end - val_start

        if train_weeks_used < min_train_weeks:
            raise ValueError(
                "Unable to satisfy requested folds with min train window (8 weeks)."
            )

        train_week_set = set(unique_weeks[train_start:train_end])
        val_week_set = set(unique_weeks[val_start:val_end])
        train_idx = train_df.index[weeks.isin(train_week_set)].tolist()
        val_idx = train_df.index[weeks.isin(val_week_set)].tolist()
        embargo_rows_dropped = 0
        if split_group_series is not None and val_idx:
            val_groups = set(split_group_series.loc[val_idx].dropna().unique().tolist())
            if val_groups:
                train_idx = [idx for idx in train_idx if split_group_series.loc[idx] not in val_groups]
        if embargo_days > 0 and val_idx and embargo_mode == "days" and embargo_date_col:
            boundary_df = train_df.loc[val_idx]
            if embargo_date_col:
                filtered_train, embargo_rows_dropped = _apply_embargo_by_days(
                    train_df.loc[train_idx],
                    boundary_df,
                    embargo_days=embargo_days,
                    date_col=embargo_date_col,
                )
                train_idx = filtered_train.index.tolist()
        embargo_gap_days_actual: Optional[int] = None
        if train_idx and val_idx:
            if embargo_mode == "days" and embargo_date_col:
                train_dates = pd.to_datetime(train_df.loc[train_idx, embargo_date_col], errors="coerce", utc=True).dropna()
                val_dates = pd.to_datetime(train_df.loc[val_idx, embargo_date_col], errors="coerce", utc=True).dropna()
                if not train_dates.empty and not val_dates.empty:
                    embargo_gap_days_actual = int((val_dates.min() - train_dates.max()).days)
            else:
                train_weeks_actual = pd.to_datetime(train_df.loc[train_idx, week_col], errors="coerce").dropna()
                val_weeks_actual = pd.to_datetime(train_df.loc[val_idx, week_col], errors="coerce").dropna()
                if not train_weeks_actual.empty and not val_weeks_actual.empty:
                    embargo_gap_days_actual = int((val_weeks_actual.min() - train_weeks_actual.max()).days)
        if not train_idx or not val_idx:
            raise ValueError(
                f"Fold {fold_idx + 1} became empty after group/embargo filtering; "
                "unable to honor requested fold count."
            )
        out.append(
            {
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_start": str(unique_weeks[train_start].date()),
                "train_end": str(unique_weeks[train_end - 1].date()),
                "val_start": str(unique_weeks[val_start].date()),
                "val_end": str(unique_weeks[val_end - 1].date()),
                "train_window_weeks_used": int(train_weeks_used),
                "val_window_weeks_used": int(val_weeks_used),
                "train_window_weeks_target": int(train_weeks),
                "val_window_weeks_target": int(val_weeks),
                "embargo_days": int(embargo_days),
                "embargo_mode": str(embargo_mode),
                "embargo_rows_dropped_train": int(embargo_rows_dropped),
                "embargo_gap_days_actual": embargo_gap_days_actual,
            }
        )
        right = val_start

    out.reverse()
    if len(out) != folds:
        raise ValueError(
            f"Unable to build requested fold count ({folds}); built {len(out)}."
        )
    return out


def build_calibration_cache(
    df: pd.DataFrame,
    args: argparse.Namespace,
    *,
    trainer_warnings: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    fast_trial: Optional[bool] = None,
) -> Optional[CalibrationCache]:
    trainer_warnings = list(trainer_warnings or [])
    if fast_trial is None:
        fast_trial = bool(getattr(args, "fast_trial", False))
    fast_trial = bool(fast_trial)

    # Parse feature lists (for engineered feature prep only).
    if numeric_features is None:
        numeric_features = [f.strip() for f in str(args.features).split(",") if f.strip()]
    else:
        numeric_features = [f.strip() for f in numeric_features if str(f).strip()]
    if bool(getattr(args, "enable_x_abs_m", False)) and "x_abs_m" not in numeric_features:
        numeric_features.append("x_abs_m")
    if categorical_features is None:
        categorical_features = [
            f.strip()
            for f in (args.categorical_features or "").split(",")
            if f.strip() and f.strip().lower() not in {"none", "null", "false"}
        ]
    else:
        categorical_features = [f.strip() for f in categorical_features if str(f).strip()]

    ticker_col = args.ticker_col
    foundation_set = {t.strip().upper() for t in str(args.foundation_tickers).split(",") if t.strip()}

    try:
        asof_dow_allowed = _parse_asof_dow_allowed(args.asof_dow_allowed)
    except ValueError as exc:
        print(f"[base-model] ERROR: {exc}")
        return None

    active_filters: Dict[str, Any] = {
        "tdays_allowed": [],
        "asof_dow_allowed": asof_dow_allowed,
        "asof_dow_col": None,
        "drop_prn_extremes": bool(args.drop_prn_extremes),
        "prn_eps": float(args.prn_eps) if args.prn_eps is not None else None,
        "prn_below": float(args.prn_below) if args.prn_below is not None else None,
        "prn_above": float(args.prn_above) if args.prn_above is not None else None,
        "max_abs_logm": float(args.max_abs_logm) if args.max_abs_logm is not None else None,
        "train_tickers": [],
    }

    df = ensure_engineered_features(df, numeric_features)

    if args.drop_prn_extremes:
        if "pRN" in df.columns:
            prn_below = float(args.prn_below) if args.prn_below is not None else None
            prn_above = float(args.prn_above) if args.prn_above is not None else None
            if prn_below is None and prn_above is None:
                prn_eps = float(args.prn_eps) if args.prn_eps is not None else 1e-3
                prn_below = prn_eps
                prn_above = 1.0 - prn_eps
            elif prn_below is None:
                prn_below = 1e-6
            elif prn_above is None:
                prn_above = 1.0 - 1e-6
            if not (0.0 <= float(prn_below) < float(prn_above) <= 1.0):
                print(
                    "[base-model] ERROR: invalid pRN bounds for --drop-prn-extremes "
                    f"(below={prn_below}, above={prn_above})."
                )
                return None
            p = pd.to_numeric(df["pRN"], errors="coerce")
            df = df[(p > float(prn_below)) & (p < float(prn_above))].copy()
            active_filters["prn_below"] = float(prn_below)
            active_filters["prn_above"] = float(prn_above)
        else:
            print("[WARN] [base-model] --drop-prn-extremes ignored: pRN column missing.")

    if args.max_abs_logm is not None:
        abs_bound = float(args.max_abs_logm)
        m_col = None
        if "log_m_fwd" in df.columns:
            m_col = "log_m_fwd"
        elif "log_m" in df.columns:
            m_col = "log_m"
        if m_col is not None:
            mvals = pd.to_numeric(df[m_col], errors="coerce").abs()
            df = df[mvals <= abs_bound].copy()
        else:
            print("[WARN] [base-model] --max-abs-logm ignored: no log-moneyness column found.")

    tdays_single: Optional[int] = None
    if args.tdays_allowed:
        tdays_tokens = [t.strip() for t in str(args.tdays_allowed).split(",") if t.strip()]
        if len(tdays_tokens) != 1:
            print(
                "[base-model] ERROR: only one T_days regime is allowed. "
                "Choose exactly one of: 1,2,3,4."
            )
            return None
        try:
            tdays_single = int(tdays_tokens[0])
        except Exception:
            print(f"[base-model] ERROR: invalid T_days value {tdays_tokens[0]!r}.")
            return None
        if tdays_single not in {1, 2, 3, 4}:
            print(f"[base-model] ERROR: T_days must be one of 1,2,3,4 (got {tdays_single}).")
            return None
        if "T_days" in df.columns:
            df = df[pd.to_numeric(df["T_days"], errors="coerce") == float(tdays_single)].copy()
            active_filters["tdays_allowed"] = [int(tdays_single)]
        else:
            print("[WARN] [base-model] --tdays-allowed ignored: T_days column missing.")

    if asof_dow_allowed:
        df, asof_dow_col = _apply_asof_dow_filter(df, asof_dow_allowed)
        active_filters["asof_dow_col"] = asof_dow_col

    if args.train_tickers:
        train_tickers = {t.strip().upper() for t in str(args.train_tickers).split(",") if t.strip()}
        if train_tickers:
            if ticker_col in df.columns:
                before_rows = len(df)
                df = df[df[ticker_col].astype("string").str.upper().isin(train_tickers)].copy()
                active_filters["train_tickers"] = sorted(train_tickers)
                active_filters["train_tickers_rows_filtered"] = int(before_rows - len(df))
            else:
                print(f"[WARN] [base-model] --train-tickers ignored: {ticker_col!r} column missing.")

    target_col = args.target_col
    if not target_col:
        for cand in ["label", "outcome_ST_gt_K", "outcome"]:
            if cand in df.columns:
                target_col = cand
                break
    if not target_col or target_col not in df.columns:
        print("[base-model] ERROR: no target column found (tried label, outcome_ST_gt_K, outcome).")
        return None

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    if df.empty:
        print("[base-model] ERROR: no valid rows after label filter.")
        return None

    week_col = args.week_col
    if week_col not in df.columns:
        for date_cand in ["asof_date", "snapshot_time_utc", "asof_datetime_utc", "snapshot_date"]:
            if date_cand in df.columns:
                df[week_col] = _compute_week_friday(df[date_cand])
                break

    df = _sort_base_calibration_df(df, week_col=week_col, ticker_col=ticker_col)

    if week_col in df.columns:
        df[week_col] = pd.to_datetime(df[week_col], errors="coerce").dt.normalize()
        all_weeks = df[week_col].dropna().sort_values().unique()
        if len(all_weeks) > args.test_weeks:
            test_week_set = set(all_weeks[-args.test_weeks:])
            test_mask = df[week_col].isin(test_week_set)
        else:
            test_mask = pd.Series(False, index=df.index)
    else:
        n = len(df)
        cutoff = max(1, int(n * 0.8))
        test_mask = pd.Series([i >= cutoff for i in range(n)], index=df.index)

    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()

    split_group_series, split_group_key = _resolve_group_key_for_splits(df, preferred_key=args.grouping_key)
    split_group_dropped_train_rows = 0
    split_group_dropped_train_fit_rows = 0
    if split_group_series is not None and not train_df.empty and not test_df.empty:
        test_groups = set(split_group_series.loc[test_df.index].dropna().unique().tolist())
        if test_groups:
            drop_mask = split_group_series.loc[train_df.index].isin(test_groups)
            split_group_dropped_train_rows = int(drop_mask.sum())
            if split_group_dropped_train_rows:
                train_df = train_df.loc[~drop_mask].copy()
                trainer_warnings.append(
                    f"Dropped {split_group_dropped_train_rows} train rows to enforce group-disjoint test split ({split_group_key})."
                )

    embargo_days_requested = max(0, int(args.embargo_days or 0))
    embargo_date_candidate = _resolve_embargo_date_col(df)
    embargo_mode = "days" if (embargo_days_requested > 0 and embargo_date_candidate) else "weeks"
    embargo_date_col_used = embargo_date_candidate if embargo_mode == "days" else None
    embargo_rows_dropped_train = 0
    embargo_rows_dropped_train_fit = 0
    if embargo_days_requested > 0:
        if embargo_mode == "days" and embargo_date_col_used:
            filtered_train, embargo_rows_dropped_train = _apply_embargo_by_days(
                train_df,
                test_df,
                embargo_days=embargo_days_requested,
                date_col=embargo_date_col_used,
            )
        else:
            filtered_train, embargo_rows_dropped_train = _apply_embargo_by_weeks(
                train_df,
                test_df,
                embargo_days=embargo_days_requested,
                week_col=week_col,
            )
        train_df = filtered_train

    if train_df.empty:
        print("[base-model] ERROR: training set empty after time split.")
        return None

    val_weeks_override = int(args.val_weeks) if int(args.val_weeks or 0) > 0 else None
    walk_forward_folds: List[Dict[str, Any]] = []
    if str(args.split_strategy) == "walk_forward":
        try:
            walk_forward_folds = _resolve_walk_forward_folds(
                train_df,
                week_col=week_col,
                train_window_weeks=max(1, int(args.train_window_weeks)),
                validation_window_weeks=max(1, int(args.validation_window_weeks)),
                validation_folds=max(1, int(args.validation_folds)),
                window_mode=str(args.window_mode),
                embargo_days=embargo_days_requested,
                split_group_series=split_group_series.loc[train_df.index] if split_group_series is not None else None,
            )
        except ValueError as exc:
            print(f"[base-model] ERROR: {exc}")
            return None
    if walk_forward_folds:
        latest_fold = walk_forward_folds[-1]
        train_fit_df = train_df.loc[latest_fold["train_idx"]].copy()
        val_df = train_df.loc[latest_fold["val_idx"]].copy()
        val_split_info = {
            "mode": "walk_forward",
            "used_week_groups": True,
            "val_weeks": int(args.validation_window_weeks),
            "fold_count": len(walk_forward_folds),
        }
        shortened_train = any(
            int(fold.get("train_window_weeks_used", int(args.train_window_weeks))) < int(args.train_window_weeks)
            for fold in walk_forward_folds
        )
        shortened_val = any(
            int(fold.get("val_window_weeks_used", int(args.validation_window_weeks))) < int(args.validation_window_weeks)
            for fold in walk_forward_folds
        )
        if shortened_train or shortened_val:
            trainer_warnings.append(
                "Walk-forward folds used shorter windows to preserve requested fold count."
            )
    else:
        train_fit_df, val_df, val_split_info = _resolve_val_split(
            train_df,
            week_col=week_col,
            val_split_mode=str(args.val_split_mode),
            val_weeks_override=val_weeks_override,
        )
        if split_group_series is not None and not val_df.empty and not train_fit_df.empty:
            val_groups = set(split_group_series.loc[val_df.index].dropna().unique().tolist())
            if val_groups:
                drop_mask = split_group_series.loc[train_fit_df.index].isin(val_groups)
                split_group_dropped_train_fit_rows = int(drop_mask.sum())
                if split_group_dropped_train_fit_rows:
                    train_fit_df = train_fit_df.loc[~drop_mask].copy()
                    trainer_warnings.append(
                        f"Dropped {split_group_dropped_train_fit_rows} train-fit rows to enforce group-disjoint validation split ({split_group_key})."
                    )
        if embargo_days_requested > 0 and embargo_mode == "days" and embargo_date_col_used:
            filtered_fit, embargo_rows_dropped_train_fit = _apply_embargo_by_days(
                train_fit_df,
                val_df,
                embargo_days=embargo_days_requested,
                date_col=embargo_date_col_used,
            )
            train_fit_df = filtered_fit
        elif embargo_days_requested > 0 and embargo_mode == "weeks":
            filtered_fit, embargo_rows_dropped_train_fit = _apply_embargo_by_weeks(
                train_fit_df,
                val_df,
                embargo_days=embargo_days_requested,
                week_col=week_col,
            )
            train_fit_df = filtered_fit

    if train_fit_df.empty:
        if embargo_rows_dropped_train_fit > 0:
            print("[base-model] ERROR: train_fit split empty after embargo; reduce embargo_days.")
        else:
            print("[base-model] ERROR: train_fit split is empty.")
        return None
    if val_df.empty:
        if int(args.embargo_days or 0) > 0:
            print("[WARN] [base-model] validation split empty after embargo; tuning will use default C.")
        else:
            print("[WARN] [base-model] validation split is empty; tuning will use default C.")

    train_pos = pd.Series(np.arange(len(train_df)), index=train_df.index)
    fit_pos = train_pos.loc[train_fit_df.index].to_numpy(dtype=int)
    val_pos = train_pos.loc[val_df.index].to_numpy(dtype=int)

    train_weights_raw, weight_source = _resolve_base_weight_vector(train_df, args.weight_col)

    group_key, group_key_source = _build_group_key_for_reweight(train_df, preferred_key=args.grouping_key)

    if fast_trial:
        split_overlap: Dict[str, Any] = {}
        split_ranges: Dict[str, Any] = {}
        split_composition_rows: List[Dict[str, Any]] = []
    else:
        split_overlap = _build_split_overlap_diagnostics(
            train_fit_df=train_fit_df,
            val_df=val_df,
            train_df=train_df,
            test_df=test_df,
            week_col=week_col,
            split_group_key=split_group_key,
            split_group_series=split_group_series,
        )
        split_ranges = {
            "train_fit_weeks_range": _week_range(train_fit_df, week_col),
            "val_weeks_range": _week_range(val_df, week_col),
            "test_weeks_range": _week_range(test_df, week_col),
            "train_rows_range": _date_range_from_candidates(
                train_df, ["asof_date", "snapshot_date", "asof_datetime_utc", "snapshot_time_utc"]
            ),
            "val_rows_range": _date_range_from_candidates(
                val_df, ["asof_date", "snapshot_date", "asof_datetime_utc", "snapshot_time_utc"]
            ),
            "test_rows_range": _date_range_from_candidates(
                test_df, ["asof_date", "snapshot_date", "asof_datetime_utc", "snapshot_time_utc"]
            ),
        }
        split_composition_rows = [
            _split_composition_row(split_name="train_fit", frame=train_fit_df, target_col=target_col, week_col=week_col),
            _split_composition_row(split_name="val", frame=val_df, target_col=target_col, week_col=week_col),
            _split_composition_row(split_name="train", frame=train_df, target_col=target_col, week_col=week_col),
            _split_composition_row(split_name="test", frame=test_df, target_col=target_col, week_col=week_col),
        ]
        for fold_idx, fold in enumerate(walk_forward_folds):
            fold_train = train_df.loc[fold["train_idx"]]
            fold_val = train_df.loc[fold["val_idx"]]
            split_composition_rows.append(
                _split_composition_row(
                    split_name=f"fold_{fold_idx + 1}_train",
                    frame=fold_train,
                    target_col=target_col,
                    week_col=week_col,
                )
            )
            split_composition_rows.append(
                _split_composition_row(
                    split_name=f"fold_{fold_idx + 1}_val",
                    frame=fold_val,
                    target_col=target_col,
                    week_col=week_col,
                )
            )

    return CalibrationCache(
        df_base=df,
        target_col=target_col,
        week_col=week_col,
        ticker_col=ticker_col,
        train_idx=train_df.index.copy(),
        test_idx=test_df.index.copy(),
        train_fit_idx=train_fit_df.index.copy(),
        val_idx=val_df.index.copy(),
        train_df=train_df,
        test_df=test_df,
        train_fit_df=train_fit_df,
        val_df=val_df,
        train_pos=train_pos,
        fit_pos=fit_pos,
        val_pos=val_pos,
        split_group_series=split_group_series,
        split_group_key=split_group_key,
        split_group_dropped_train_rows=split_group_dropped_train_rows,
        split_group_dropped_train_fit_rows=split_group_dropped_train_fit_rows,
        embargo_mode=embargo_mode,
        embargo_date_col_used=embargo_date_col_used,
        embargo_rows_dropped_train=embargo_rows_dropped_train,
        embargo_rows_dropped_train_fit=embargo_rows_dropped_train_fit,
        walk_forward_folds=walk_forward_folds,
        val_split_info=val_split_info,
        split_overlap=split_overlap,
        split_ranges=split_ranges,
        split_composition_rows=split_composition_rows,
        train_weights_raw=train_weights_raw,
        weight_source=weight_source,
        group_key=group_key,
        group_key_source=group_key_source,
        foundation_set=foundation_set,
        active_filters=active_filters,
        trainer_warnings=trainer_warnings,
        requested_numeric_features=numeric_features,
        requested_categorical_features=categorical_features,
    )


def run_calibration_from_cache(
    cache: CalibrationCache,
    args: argparse.Namespace,
    out_dir: Path,
    *,
    config_payload: Optional[Dict[str, Any]] = None,
    unknown: Optional[List[str]] = None,
    unsupported_controls: Optional[Dict[str, Any]] = None,
    n_bins: int,
    eceq_bins: int,
    fast_trial: bool,
    skip_test_metrics: bool,
) -> RunResult:
    args = argparse.Namespace(**vars(args))
    args.out_dir = str(out_dir)
    args.fast_trial = bool(fast_trial)
    args.skip_test_metrics = bool(skip_test_metrics)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_args_payload = _resolved_args_snapshot(args)
    resolved_args_path = out_dir / "resolved_args.json"
    resolved_args_path.write_text(json.dumps(resolved_args_payload, indent=2))

    if config_payload and not bool(getattr(args, "allow_defaults", False)):
        diffs = _diff_requested_vs_resolved_config(config_payload, args)
        if diffs:
            diff_payload = {
                "allow_defaults": bool(getattr(args, "allow_defaults", False)),
                "mismatch_count": int(len(diffs)),
                "mismatches": diffs,
            }
            (out_dir / "resolved_args_diff.json").write_text(json.dumps(diff_payload, indent=2))
            preview = "; ".join(
                f"{item.get('field')}: requested={item.get('requested')} resolved={item.get('resolved')}"
                for item in diffs[:5]
            )
            print(
                "[base-model] ERROR: Requested config does not match resolved args. "
                "Set allow_defaults=true to bypass strict matching. "
                f"Diff preview: {preview}"
            )
            return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])

    trainer_warnings: List[str] = list(cache.trainer_warnings)
    active_filters = dict(cache.active_filters)
    unknown = list(unknown or [])
    unsupported_controls = dict(unsupported_controls or {})

    target_col = cache.target_col
    week_col = cache.week_col
    ticker_col = cache.ticker_col
    split_group_series = cache.split_group_series
    split_group_key = cache.split_group_key
    split_group_dropped_train_rows = cache.split_group_dropped_train_rows
    split_group_dropped_train_fit_rows = cache.split_group_dropped_train_fit_rows
    embargo_mode = cache.embargo_mode
    embargo_date_col_used = cache.embargo_date_col_used
    embargo_rows_dropped_train = cache.embargo_rows_dropped_train
    embargo_rows_dropped_train_fit = cache.embargo_rows_dropped_train_fit
    walk_forward_folds = cache.walk_forward_folds
    val_split_info = cache.val_split_info

    numeric_features = [f.strip() for f in str(args.features).split(",") if f.strip()]
    if bool(getattr(args, "enable_x_abs_m", False)) and "x_abs_m" not in numeric_features:
        numeric_features.append("x_abs_m")
    enable_x_abs_m_effective = "x_abs_m" in numeric_features
    categorical_features = [
        f.strip()
        for f in (args.categorical_features or "").split(",")
        if f.strip() and f.strip().lower() not in {"none", "null", "false"}
    ]

    train_df = cache.train_df
    test_df = cache.test_df
    train_fit_df = cache.train_fit_df
    val_df = cache.val_df

    avail_numeric = [f for f in numeric_features if f in train_df.columns]
    avail_numeric = _drop_all_nan_features(train_df, avail_numeric, "base-model")
    avail_cat = [f for f in categorical_features if f in train_df.columns]
    if not avail_numeric:
        print(f"[base-model] ERROR: none of {numeric_features} found in dataset.")
        return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])

    foundation_set = {t.strip().upper() for t in str(args.foundation_tickers).split(",") if t.strip()}
    foundation_tickers_list: Optional[List[str]] = sorted(foundation_set) if foundation_set else None

    ticker_feature_col: Optional[str] = None
    ticker_support_values: Optional[List[str]] = None
    ticker_other_label = "OTHER"
    if args.ticker_intercepts != "none" and ticker_col in train_df.columns:
        ticker_feature_col = "_ticker_feature"

        def _make_ticker_col(frame: pd.DataFrame) -> "pd.Series[str]":
            raw = frame[ticker_col].astype(str).str.upper()
            if args.ticker_intercepts == "non_foundation" and foundation_set:
                return raw.where(~raw.isin(foundation_set), "FOUNDATION")
            return raw

        train_df = train_df.copy()
        train_fit_df = train_fit_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        train_df[ticker_feature_col] = _make_ticker_col(train_df)
        train_fit_df[ticker_feature_col] = _make_ticker_col(train_fit_df)
        if not val_df.empty:
            val_df[ticker_feature_col] = _make_ticker_col(val_df)
        if not test_df.empty:
            test_df[ticker_feature_col] = _make_ticker_col(test_df)

        if args.ticker_min_support is not None and int(args.ticker_min_support) > 1:
            min_support = int(args.ticker_min_support)
            counts = train_df[ticker_feature_col].value_counts(dropna=False)
            keep = set(counts[counts >= min_support].index.astype(str).tolist())
            if args.ticker_intercepts == "non_foundation" and foundation_set:
                keep.add("FOUNDATION")

            train_df[ticker_feature_col] = train_df[ticker_feature_col].astype(str)
            train_df[ticker_feature_col] = train_df[ticker_feature_col].where(
                train_df[ticker_feature_col].isin(keep),
                ticker_other_label,
            )
            train_fit_df[ticker_feature_col] = train_fit_df[ticker_feature_col].astype(str)
            train_fit_df[ticker_feature_col] = train_fit_df[ticker_feature_col].where(
                train_fit_df[ticker_feature_col].isin(keep),
                ticker_other_label,
            )
            if not val_df.empty:
                val_df[ticker_feature_col] = val_df[ticker_feature_col].astype(str)
                val_df[ticker_feature_col] = val_df[ticker_feature_col].where(
                    val_df[ticker_feature_col].isin(keep),
                    ticker_other_label,
                )
            if not test_df.empty:
                test_df[ticker_feature_col] = test_df[ticker_feature_col].astype(str)
                test_df[ticker_feature_col] = test_df[ticker_feature_col].where(
                    test_df[ticker_feature_col].isin(keep),
                    ticker_other_label,
                )

            ticker_support_values = sorted(keep)
            active_filters["ticker_min_support"] = min_support
            active_filters["ticker_support_kept"] = ticker_support_values

        avail_cat = avail_cat + [ticker_feature_col]

    interaction_cols_added: List[str] = []
    if args.ticker_x_interactions and ticker_feature_col and "x_logit_prn" in train_df.columns:
        min_support_inter = int(args.ticker_min_support_interactions or args.ticker_min_support or 1000)
        counts = train_df[ticker_feature_col].value_counts(dropna=False)
        keep_interactions = sorted(
            [str(k) for k, v in counts.items() if int(v) >= max(1, min_support_inter)]
        )
        if keep_interactions:
            for ticker_label in keep_interactions:
                safe = re.sub(r"[^A-Za-z0-9]+", "_", ticker_label).strip("_") or "UNK"
                col_name = f"x_ticker_{safe}"
                train_df[col_name] = (
                    (train_df[ticker_feature_col].astype(str) == ticker_label).astype(float)
                    * pd.to_numeric(train_df["x_logit_prn"], errors="coerce").fillna(0.0)
                )
                train_fit_df[col_name] = (
                    (train_fit_df[ticker_feature_col].astype(str) == ticker_label).astype(float)
                    * pd.to_numeric(train_fit_df["x_logit_prn"], errors="coerce").fillna(0.0)
                )
                if not val_df.empty:
                    val_df[col_name] = (
                        (val_df[ticker_feature_col].astype(str) == ticker_label).astype(float)
                        * pd.to_numeric(val_df["x_logit_prn"], errors="coerce").fillna(0.0)
                    )
                if not test_df.empty:
                    test_df[col_name] = (
                        (test_df[ticker_feature_col].astype(str) == ticker_label).astype(float)
                        * pd.to_numeric(test_df["x_logit_prn"], errors="coerce").fillna(0.0)
                    )
                interaction_cols_added.append(col_name)
            avail_numeric = avail_numeric + interaction_cols_added

    all_feat_cols = avail_numeric + avail_cat

    train_weights_base = np.asarray(cache.train_weights_raw, dtype=float).copy()
    weight_source = cache.weight_source
    weight_invariant_tol = 1e-9
    foundation_weight_applied_rows = 0
    group_reweight_mode = _normalize_group_reweight_mode(args.group_reweight)
    group_equalization_enabled = bool(args.group_equalization)
    if args.no_group_equalization:
        group_equalization_enabled = False
    if group_reweight_mode == "chain_snapshot":
        group_equalization_enabled = True

    group_key_source: Optional[str] = None
    group_key = None
    if group_equalization_enabled:
        group_key = cache.group_key
        group_key_source = cache.group_key_source
        if group_key is None:
            group_key, group_key_source = _build_group_key_for_reweight(
                train_df, preferred_key=args.grouping_key
            )

    weights_equalized = train_weights_base.copy()
    if group_equalization_enabled:
        if group_key is not None:
            weights_equalized = _equalize_group_sums_to_one(train_weights_base, group_key)
            group_reweight_mode = "chain_snapshot"
        else:
            print("[WARN] [base-model] Group equalization requested but no grouping key could be resolved.")
            group_equalization_enabled = False
            group_reweight_mode = "none"

    group_sum_unit_before_multipliers_passed = None
    if group_equalization_enabled and group_key is not None:
        grouped_before = _describe_group_sums(weights_equalized, group_key)
        if grouped_before is not None:
            group_sum_unit_before_multipliers_passed = bool(
                abs(float(grouped_before["min"]) - 1.0) <= weight_invariant_tol
                and abs(float(grouped_before["mean"]) - 1.0) <= weight_invariant_tol
                and abs(float(grouped_before["max"]) - 1.0) <= weight_invariant_tol
            )
            if not group_sum_unit_before_multipliers_passed:
                print(
                    "[base-model] ERROR: Group-equalization invariant failed "
                    "(group sums must be 1 before multipliers)."
                )
                return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])

    train_weights_policy = weights_equalized.copy()
    if foundation_set and ticker_col in train_df.columns and args.foundation_weight is not None:
        f_weight = float(args.foundation_weight)
        if np.isfinite(f_weight) and f_weight > 0 and abs(f_weight - 1.0) > 1e-12:
            foundation_mask = train_df[ticker_col].astype("string").str.upper().isin(foundation_set).to_numpy()
            if foundation_mask.any():
                train_weights_policy[foundation_mask] = train_weights_policy[foundation_mask] * f_weight
                foundation_weight_applied_rows = int(foundation_mask.sum())

    trading_universe_upweight_applied_rows = 0
    if (
        args.trading_universe_tickers
        and ticker_col in train_df.columns
        and args.trading_universe_upweight is not None
        and float(args.trading_universe_upweight) > 0
        and abs(float(args.trading_universe_upweight) - 1.0) > 1e-12
    ):
        trading_set = {t.strip().upper() for t in str(args.trading_universe_tickers).split(",") if t.strip()}
        if trading_set:
            up_mask = train_df[ticker_col].astype("string").str.upper().isin(trading_set).to_numpy()
            if up_mask.any():
                train_weights_policy[up_mask] = train_weights_policy[up_mask] * float(args.trading_universe_upweight)
                trading_universe_upweight_applied_rows = int(up_mask.sum())

    ticker_balance_mode_used = str(args.ticker_balance_mode or "none").strip().lower()
    if ticker_balance_mode_used == "sqrt_inv_clipped" and ticker_col in train_df.columns:
        ticker_series = train_df[ticker_col].astype("string").str.upper().fillna("UNKNOWN")
        counts = ticker_series.value_counts(dropna=False)
        if len(counts) > 0:
            mean_count = float(counts.mean())
            factors = {
                key: float(np.clip(np.sqrt(mean_count / max(1.0, float(count))), 0.5, 2.0))
                for key, count in counts.items()
            }
            train_weights_policy = train_weights_policy * ticker_series.map(factors).astype(float).to_numpy(dtype=float)

    weight_audit_payload: Dict[str, Any] = {
        "weight_col_requested": args.weight_col,
        "weight_source": weight_source,
        "group_reweight_mode": group_reweight_mode,
        "group_key_source": group_key_source,
        "grouping_key_requested": args.grouping_key,
        "group_equalization_enabled": bool(group_equalization_enabled),
        "raw": _describe_weight_vector(train_weights_base),
        "equalized_before_multipliers": _describe_weight_vector(weights_equalized),
        "after_multipliers_train_weights": _describe_weight_vector(train_weights_policy),
        "raw_group_sums": _describe_group_sums(train_weights_base, group_key),
        "equalized_group_sums_before_multipliers": _describe_group_sums(weights_equalized, group_key),
        "after_multipliers_group_sums": _describe_group_sums(train_weights_policy, group_key),
    }

    train_pos = cache.train_pos
    fit_pos = cache.fit_pos
    val_pos = cache.val_pos
    w_train = _renorm_weights_mean_one(train_weights_policy, tol=weight_invariant_tol)
    w_fit = _renorm_weights_mean_one(train_weights_policy[fit_pos], tol=weight_invariant_tol) if len(fit_pos) else np.zeros(0, dtype=float)
    w_val = _renorm_weights_mean_one(train_weights_policy[val_pos], tol=weight_invariant_tol) if len(val_pos) else np.zeros(0, dtype=float)

    test_weights_raw, _ = _resolve_base_weight_vector(test_df, args.weight_col)
    mean1_train_fit_passed = True
    if w_fit.size > 0:
        mean1_train_fit_passed = abs(float(np.mean(w_fit)) - 1.0) <= weight_invariant_tol
        if not mean1_train_fit_passed:
            print("[base-model] ERROR: train-fit weights failed mean-1 invariant.")
            return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])

    weight_audit_payload.update(
        {
            "final_train_weights": _describe_weight_vector(w_train),
            "final_group_sums": _describe_group_sums(w_train, group_key),
            "split_weight_means": {
                "train_fit": float(np.mean(w_fit)) if w_fit.size > 0 else None,
                "val": float(np.mean(w_val)) if w_val.size > 0 else None,
                "test": float(np.mean(test_weights_raw)) if test_weights_raw.size > 0 else None,
            },
            "invariants": {
                "weight_tol": float(weight_invariant_tol),
                "group_sum_unit_before_multipliers_passed": group_sum_unit_before_multipliers_passed,
                "mean1_train_fit_passed": bool(mean1_train_fit_passed),
            },
            "calibration_weight_policy": "subset_mean1_from_train_policy",
        }
    )

    try:
        c_grid = _parse_and_validate_c_grid(args.C_grid)
    except ValueError as exc:
        print(f"[base-model] ERROR: invalid --C-grid: {exc}")
        return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])

    def _make_pipe(c: float) -> Pipeline:
        transformers: List = []
        if avail_numeric:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    avail_numeric,
                )
            )
        if avail_cat:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "onehot",
                                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
                            ),
                        ]
                    ),
                    avail_cat,
                )
            )
        pre = ColumnTransformer(transformers, remainder="drop")
        clf = LogisticRegression(C=c, solver="lbfgs", max_iter=4000, random_state=args.random_state)
        return Pipeline([("pre", pre), ("clf", clf)])

    X_train = train_df[all_feat_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_fit = train_fit_df[all_feat_cols]
    y_fit = train_fit_df[target_col].to_numpy(dtype=float)
    X_val = val_df[all_feat_cols]
    y_val = val_df[target_col].to_numpy(dtype=float)
    train_fit_class_count = _class_count(y_fit)
    train_class_count = _class_count(y_train)
    val_class_count = _class_count(y_val)

    if train_fit_class_count < 2:
        print("[base-model] ERROR: train_fit split has fewer than two classes; cannot fit logistic model.")
        return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])
    if train_class_count < 2:
        print("[base-model] ERROR: training split has fewer than two classes; cannot fit final model.")
        return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])

    best_c = c_grid[len(c_grid) // 2]
    best_score = float("inf")
    objective_name = str(args.selection_objective or "logloss")
    c_search_stats: List[Dict[str, Any]] = []
    c_selection_rule = "midpoint_default"

    if walk_forward_folds:
        for c in c_grid:
            fold_scores: List[float] = []
            for fold in walk_forward_folds:
                fold_train = train_df.loc[fold["train_idx"]]
                fold_val = train_df.loc[fold["val_idx"]]
                if fold_train.empty or fold_val.empty:
                    continue
                y_fold_train = fold_train[target_col].to_numpy(dtype=float)
                y_fold_val = fold_val[target_col].to_numpy(dtype=float)
                if _class_count(y_fold_train) < 2 or _class_count(y_fold_val) < 2:
                    continue
                fold_train_pos = train_pos.loc[fold_train.index].to_numpy(dtype=int)
                w_fold_train = _renorm_weights_mean_one(
                    train_weights_policy[fold_train_pos],
                    tol=weight_invariant_tol,
                )
                pipe = _make_pipe(c)
                pipe = _fit_pipe_with_optional_weights(
                    pipe,
                    fold_train[all_feat_cols],
                    y_fold_train,
                    sample_weight=w_fold_train,
                )
                p_fold = pipe.predict_proba(fold_val[all_feat_cols])[:, 1]
                fold_scores.append(
                    _score_for_objective(
                        objective=objective_name,
                        y_true=y_fold_val,
                        p_pred=p_fold,
                        n_bins=n_bins,
                        eceq_bins=eceq_bins,
                    )
                )
            if fold_scores:
                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
                se_score = float(std_score / np.sqrt(len(fold_scores))) if len(fold_scores) > 0 else 0.0
                c_search_stats.append(
                    {
                        "C": float(c),
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "se_score": se_score,
                        "n_folds_used": int(len(fold_scores)),
                    }
                )
        if c_search_stats:
            min_row = min(c_search_stats, key=lambda row: float(row["mean_score"]))
            threshold = float(min_row["mean_score"]) + float(min_row["se_score"])
            eligible = [row for row in c_search_stats if float(row["mean_score"]) <= threshold + 1e-12]
            chosen = min(eligible, key=lambda row: float(row["C"])) if eligible else min_row
            best_c = float(chosen["C"])
            best_score = float(chosen["mean_score"])
            c_selection_rule = "one_se"
        else:
            c_selection_rule = "midpoint_no_valid_fold_scores"
    elif len(y_val) >= 10 and val_class_count >= 2:
        for c in c_grid:
            try:
                pipe = _make_pipe(c)
                pipe = _fit_pipe_with_optional_weights(pipe, X_fit, y_fit, sample_weight=w_fit)
                p = pipe.predict_proba(X_val)[:, 1]
                score = _score_for_objective(
                    objective=objective_name,
                    y_true=y_val,
                    p_pred=p,
                    n_bins=n_bins,
                    eceq_bins=eceq_bins,
                )
                c_search_stats.append(
                    {
                        "C": float(c),
                        "mean_score": float(score),
                        "std_score": None,
                        "se_score": None,
                        "n_folds_used": 1,
                    }
                )
                if score < best_score:
                    best_score = score
                    best_c = c
                    c_selection_rule = "single_holdout_min"
            except Exception:
                continue
    elif len(y_val) > 0 and val_class_count < 2:
        print("[WARN] [base-model] validation split has a single class; using midpoint C without tuning.")
        trainer_warnings.append("Validation split has a single class; C-grid tuning skipped.")
        c_selection_rule = "midpoint_single_class_val"

    val_pipe = _make_pipe(best_c)
    val_pipe = _fit_pipe_with_optional_weights(val_pipe, X_fit, y_fit, sample_weight=w_fit)

    final_pipe = _make_pipe(best_c)
    final_pipe = _fit_pipe_with_optional_weights(final_pipe, X_train, y_train, sample_weight=w_train)

    platt_cal = None
    platt_skip_reason: Optional[str] = None
    if args.calibrate == "platt":
        if len(y_val) < 50:
            platt_skip_reason = "insufficient_val_rows"
        elif val_class_count < 2:
            platt_skip_reason = "single_class_val"
        else:
            logits = val_pipe.decision_function(X_val)
            platt_cal = fit_platt_on_logits(
                logits=logits,
                y=y_val,
                w_fit=w_val if len(w_val) == len(y_val) else None,
                random_state=args.random_state,
            )
        if platt_skip_reason:
            print(f"[WARN] [base-model] Platt calibration skipped: {platt_skip_reason}.")
            trainer_warnings.append(f"Platt calibration skipped: {platt_skip_reason}.")

    fold_delta_rows: List[Dict[str, Any]] = []
    val_eval_df = val_df
    val_eval_y = y_val
    val_eval_pred: Optional[np.ndarray] = None
    split_timeline_payload: Dict[str, Any] = {
        "split_strategy": str(args.split_strategy),
        "window_mode": str(args.window_mode),
        "embargo_days_requested": int(args.embargo_days),
        "embargo_mode": embargo_mode,
        "embargo_date_col_used": embargo_date_col_used,
        "train_window_weeks_requested": int(args.train_window_weeks),
        "validation_window_weeks_requested": int(args.validation_window_weeks),
        "test_window_weeks_requested": int(args.test_weeks),
        "train_window_weeks_actual": _count_unique_weeks(train_fit_df, week_col),
        "validation_window_weeks_actual": _count_unique_weeks(val_df, week_col),
        "test_window_weeks_actual": _count_unique_weeks(test_df, week_col),
        "folds": [],
    }

    if walk_forward_folds:
        fold_frames: List[pd.DataFrame] = []
        fold_preds: List[np.ndarray] = []
        for fold_idx, fold in enumerate(walk_forward_folds):
            fold_train = train_df.loc[fold["train_idx"]]
            fold_val = train_df.loc[fold["val_idx"]]
            if fold_train.empty or fold_val.empty:
                continue
            y_fold_train = fold_train[target_col].to_numpy(dtype=float)
            y_fold_val = fold_val[target_col].to_numpy(dtype=float)
            if _class_count(y_fold_train) < 2:
                continue
            fold_train_pos = train_pos.loc[fold_train.index].to_numpy(dtype=int)
            w_fold_train = _renorm_weights_mean_one(
                train_weights_policy[fold_train_pos],
                tol=weight_invariant_tol,
            )
            fold_pipe = _make_pipe(best_c)
            fold_pipe = _fit_pipe_with_optional_weights(
                fold_pipe,
                fold_train[all_feat_cols],
                y_fold_train,
                sample_weight=w_fold_train,
            )
            p_fold = fold_pipe.predict_proba(fold_val[all_feat_cols])[:, 1]
            fold_frames.append(fold_val.copy())
            fold_preds.append(p_fold)

            p_baseline_fold = pd.to_numeric(
                fold_val.get("pRN", pd.Series(dtype=float)), errors="coerce"
            ).to_numpy(dtype=float)
            valid_base = np.isfinite(p_baseline_fold)
            if valid_base.any():
                model_metrics = _compute_metrics(
                    y_fold_val[valid_base], p_fold[valid_base], n_bins=n_bins, eceq_bins=eceq_bins
                )
                baseline_metrics = _compute_metrics(
                    y_fold_val[valid_base], p_baseline_fold[valid_base], n_bins=n_bins, eceq_bins=eceq_bins
                )
                fold_delta_rows.append(
                    {
                        "fold": int(fold_idx + 1),
                        "val_start": fold["val_start"],
                        "val_end": fold["val_end"],
                        "n_rows": int(len(y_fold_val)),
                        "delta_logloss": float(model_metrics["logloss"] - baseline_metrics["logloss"]),
                        "delta_brier": float(model_metrics["brier"] - baseline_metrics["brier"]),
                        "delta_ece": float(model_metrics["ece"] - baseline_metrics["ece"]),
                        "delta_ece_q": float(model_metrics["ece_q"] - baseline_metrics["ece_q"]),
                    }
                )
            split_timeline_payload["folds"].append(
                {
                    "fold": int(fold_idx + 1),
                    "train_start": fold["train_start"],
                    "train_end": fold["train_end"],
                    "val_start": fold["val_start"],
                    "val_end": fold["val_end"],
                    "n_train_rows": int(len(fold_train)),
                    "n_val_rows": int(len(fold_val)),
                    "train_window_weeks_used": int(fold.get("train_window_weeks_used", 0)),
                    "val_window_weeks_used": int(fold.get("val_window_weeks_used", 0)),
                    "train_window_weeks_target": int(fold.get("train_window_weeks_target", args.train_window_weeks)),
                    "val_window_weeks_target": int(fold.get("val_window_weeks_target", args.validation_window_weeks)),
                    "embargo_mode": fold.get("embargo_mode", embargo_mode),
                    "embargo_gap_days_actual": fold.get("embargo_gap_days_actual"),
                }
            )

        if fold_frames and fold_preds:
            val_eval_df = pd.concat(fold_frames, axis=0)
            val_eval_pred = np.concatenate(fold_preds).astype(float)
            val_eval_y = val_eval_df[target_col].to_numpy(dtype=float)
    else:
        split_timeline_payload["embargo_gap_days_actual"] = _compute_embargo_gap_days_actual(
            train_frame=train_fit_df,
            boundary_frame=val_df,
            embargo_mode=embargo_mode,
            embargo_date_col=embargo_date_col_used,
            week_col=week_col,
        )

    if fast_trial:
        split_overlap = {}
        split_ranges: Dict[str, Any] = {}
        split_composition_rows: List[Dict[str, Any]] = []
    else:
        if cache.split_ranges and cache.split_overlap and cache.split_composition_rows:
            split_overlap = dict(cache.split_overlap)
            split_ranges = dict(cache.split_ranges)
            split_composition_rows = list(cache.split_composition_rows)
            contract_overlap_test = split_overlap.get("contract_id_train_test")
            if contract_overlap_test and contract_overlap_test > 0 and week_col in train_df.columns:
                overlap_msg = (
                    "[base-model] non-zero contract overlap across train/test under week split: "
                    f"{contract_overlap_test}"
                )
                print(f"[WARN] {overlap_msg}")
                trainer_warnings.append(overlap_msg)
        else:
            split_overlap = _build_split_overlap_diagnostics(
                train_fit_df=train_fit_df,
                val_df=val_df,
                train_df=train_df,
                test_df=test_df,
                week_col=week_col,
                split_group_key=split_group_key,
                split_group_series=split_group_series,
            )
            contract_overlap_test = split_overlap.get("contract_id_train_test")
            if contract_overlap_test and contract_overlap_test > 0 and week_col in train_df.columns:
                overlap_msg = (
                    "[base-model] non-zero contract overlap across train/test under week split: "
                    f"{contract_overlap_test}"
                )
                print(f"[WARN] {overlap_msg}")
                trainer_warnings.append(overlap_msg)
            split_ranges = {
                "train_fit_weeks_range": _week_range(train_fit_df, week_col),
                "val_weeks_range": _week_range(val_df, week_col),
                "test_weeks_range": _week_range(test_df, week_col),
                "train_rows_range": _date_range_from_candidates(
                    train_df, ["asof_date", "snapshot_date", "asof_datetime_utc", "snapshot_time_utc"]
                ),
                "val_rows_range": _date_range_from_candidates(
                    val_df, ["asof_date", "snapshot_date", "asof_datetime_utc", "snapshot_time_utc"]
                ),
                "test_rows_range": _date_range_from_candidates(
                    test_df, ["asof_date", "snapshot_date", "asof_datetime_utc", "snapshot_time_utc"]
                ),
            }
            split_composition_rows = [
                _split_composition_row(
                    split_name="train_fit", frame=train_fit_df, target_col=target_col, week_col=week_col
                ),
                _split_composition_row(split_name="val", frame=val_df, target_col=target_col, week_col=week_col),
                _split_composition_row(split_name="train", frame=train_df, target_col=target_col, week_col=week_col),
                _split_composition_row(split_name="test", frame=test_df, target_col=target_col, week_col=week_col),
            ]
            for fold_idx, fold in enumerate(walk_forward_folds):
                fold_train = train_df.loc[fold["train_idx"]]
                fold_val = train_df.loc[fold["val_idx"]]
                split_composition_rows.append(
                    _split_composition_row(
                        split_name=f"fold_{fold_idx + 1}_train",
                        frame=fold_train,
                        target_col=target_col,
                        week_col=week_col,
                    )
                )
                split_composition_rows.append(
                    _split_composition_row(
                        split_name=f"fold_{fold_idx + 1}_val",
                        frame=fold_val,
                        target_col=target_col,
                        week_col=week_col,
                    )
                )

    if not fast_trial:
        bundle = FinalModelBundle(
            kind="logit+platt" if platt_cal is not None else "logit",
            mode="pooled",
            numeric_features=avail_numeric,
            categorical_features=avail_cat,
            ticker_col=ticker_col,
            ticker_feature_col=ticker_feature_col,
            ticker_intercepts=args.ticker_intercepts,
            foundation_tickers=foundation_tickers_list,
            foundation_label="FOUNDATION",
            ticker_support=ticker_support_values,
            ticker_other_label=ticker_other_label,
            base_pipeline=final_pipe,
            platt_calibrator=platt_cal,
        )
        joblib.dump(bundle, out_dir / "final_model.joblib")
        joblib.dump(final_pipe, out_dir / "base_pipeline.joblib")

    metrics_rows: List[Dict[str, Any]] = []
    if len(val_eval_y) > 0 and not val_eval_df.empty:
        if val_eval_pred is None:
            if platt_cal is not None:
                val_logits = val_pipe.decision_function(X_val)
                p_val = apply_platt(platt_cal, val_logits)
            else:
                p_val = val_pipe.predict_proba(X_val)[:, 1]
        else:
            p_val = val_eval_pred
        try:
            _append_split_metrics_rows(
                metrics_rows,
                split_name="val",
                split_df=val_eval_df,
                y_true=val_eval_y,
                p_model=p_val,
                ticker_col=ticker_col,
                bootstrap_ci=bool(args.bootstrap_ci),
                bootstrap_B=max(0, int(args.bootstrap_B)),
                bootstrap_seed=int(args.bootstrap_seed),
                bootstrap_group=str(args.bootstrap_group),
                allow_iid_bootstrap=bool(args.allow_iid_bootstrap),
                n_bins=n_bins,
                eceq_bins=eceq_bins,
                ci_level=int(args.ci_level),
            )
        except ValueError as exc:
            print(f"[base-model] ERROR: {exc}")
            return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])
    if not args.skip_test_metrics and not test_df.empty:
        X_test = test_df[all_feat_cols]
        y_test = test_df[target_col].to_numpy(dtype=float)
        if platt_cal is not None:
            test_logits = final_pipe.decision_function(X_test)
            p_test = apply_platt(platt_cal, test_logits)
        else:
            p_test = final_pipe.predict_proba(X_test)[:, 1]
        try:
            _append_split_metrics_rows(
                metrics_rows,
                split_name="test",
                split_df=test_df,
                y_true=y_test,
                p_model=p_test,
                ticker_col=ticker_col,
                bootstrap_ci=bool(args.bootstrap_ci),
                bootstrap_B=max(0, int(args.bootstrap_B)),
                bootstrap_seed=int(args.bootstrap_seed),
                bootstrap_group=str(args.bootstrap_group),
                allow_iid_bootstrap=bool(args.allow_iid_bootstrap),
                n_bins=n_bins,
                eceq_bins=eceq_bins,
                ci_level=int(args.ci_level),
            )
        except ValueError as exc:
            print(f"[base-model] ERROR: {exc}")
            return RunResult(exit_code=1, out_dir=out_dir, metrics_rows=[])
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df["metrics_schema_version"] = 2
        metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    if not fast_trial:
        _build_base_metrics_summary(
            metrics_rows=metrics_rows,
            split_ranges=split_ranges,
            out_dir=out_dir,
        )
        if split_composition_rows:
            pd.DataFrame(split_composition_rows).to_csv(out_dir / "audit_split_composition.csv", index=False)
        (out_dir / "audit_overlap.json").write_text(
            json.dumps(
                {
                    "split_overlap": split_overlap,
                    "split_ranges": split_ranges,
                },
                indent=2,
            )
        )
        (out_dir / "audit_weight_distribution.json").write_text(json.dumps(weight_audit_payload, indent=2))

    if args.per_fold_delta_chart and fold_delta_rows:
        pd.DataFrame(fold_delta_rows).to_csv(out_dir / "fold_deltas.csv", index=False)
    if args.split_timeline:
        if split_timeline_payload.get("train_window_weeks_actual") is None:
            split_timeline_payload["train_window_weeks_actual"] = _count_unique_weeks(train_fit_df, week_col)
        if split_timeline_payload.get("validation_window_weeks_actual") is None:
            split_timeline_payload["validation_window_weeks_actual"] = _count_unique_weeks(val_df, week_col)
        if split_timeline_payload.get("test_window_weeks_actual") is None:
            split_timeline_payload["test_window_weeks_actual"] = _count_unique_weeks(test_df, week_col)
        split_timeline_payload["test_range"] = split_ranges.get("test_rows_range")
        split_timeline_payload["train_range"] = split_ranges.get("train_rows_range")
        split_timeline_payload["val_range"] = split_ranges.get("val_rows_range")
        split_timeline_payload["fold_count"] = len(split_timeline_payload.get("folds", []))
        (out_dir / "split_timeline.json").write_text(json.dumps(split_timeline_payload, indent=2))
    if (
        not args.skip_test_metrics
        and args.per_group_delta_distribution
        and "group_id" in test_df.columns
        and not test_df.empty
    ):
        p_baseline_test = pd.to_numeric(test_df.get("pRN", pd.Series(dtype=float)), errors="coerce")
        group_df = pd.DataFrame(
            {
                "group_id": test_df["group_id"].astype("string"),
                "y_true": y_test,
                "p_model": p_test,
                "p_baseline": p_baseline_test.to_numpy(dtype=float),
            }
        )
        group_df = group_df[np.isfinite(group_df["p_model"]) & np.isfinite(group_df["p_baseline"])]
        if not group_df.empty:
            grouped = (
                group_df.groupby("group_id", dropna=False)
                .apply(
                    lambda g: pd.Series(
                        {
                            "n_rows": int(len(g)),
                            "delta_logloss": float(
                                _compute_logloss(
                                    g["y_true"].to_numpy(dtype=float),
                                    g["p_model"].to_numpy(dtype=float),
                                )
                                - _compute_logloss(
                                    g["y_true"].to_numpy(dtype=float),
                                    g["p_baseline"].to_numpy(dtype=float),
                                )
                            ),
                        }
                    )
                )
                .reset_index()
            )
            grouped.to_csv(out_dir / "group_delta_distribution.csv", index=False)

    clf_step = final_pipe.named_steps["clf"]
    try:
        feat_names_out = final_pipe.named_steps["pre"].get_feature_names_out().tolist()
    except Exception:
        feat_names_out = avail_numeric + avail_cat

    x_logit_prn_coef = None
    for feat_name, coef_val in zip(feat_names_out, clf_step.coef_[0].tolist()):
        if str(feat_name).endswith("x_logit_prn") or str(feat_name) == "x_logit_prn":
            x_logit_prn_coef = float(coef_val)
            break
    if x_logit_prn_coef is not None and x_logit_prn_coef <= 0:
        monotonic_msg = (
            f"x_logit_prn coefficient is non-positive ({x_logit_prn_coef:.6g}); "
            "check leakage, feature definitions, and split safety."
        )
        print(f"[WARN] [base-model] {monotonic_msg}")
        trainer_warnings.append(monotonic_msg)

    dataset_fingerprint: Optional[Dict[str, Any]]
    try:
        dataset_fingerprint = _file_fingerprint(Path(args.csv))
    except Exception:
        dataset_fingerprint = None

    executed_config: Dict[str, Any] = {
        "config_schema_version": 2,
        "run_mode": str(args.run_mode),
        "csv": str(Path(args.csv).resolve()) if args.csv else None,
        "out_dir": str(out_dir.resolve()),
        "features": ",".join(avail_numeric),
        "categorical_features": ",".join(avail_cat),
        "target_col": target_col,
        "week_col": week_col,
        "ticker_col": ticker_col,
        "weight_col": str(args.weight_col),
        "weight_col_strategy": args.weight_col_strategy,
        "split": {
            "strategy": str(args.split_strategy),
            "window_mode": str(args.window_mode),
            "train_window_weeks": int(args.train_window_weeks),
            "validation_folds": int(args.validation_folds),
            "validation_window_weeks": int(args.validation_window_weeks),
            "test_window_weeks": int(args.test_weeks),
            "embargo_days": int(args.embargo_days),
        },
        "regularization": {
            "c_grid": str(args.C_grid),
            "calibration_method": str(args.calibrate),
            "selection_objective": str(args.selection_objective),
            "best_c": float(best_c),
            "selection_rule": str(c_selection_rule),
            "cv_scores": c_search_stats,
        },
        "model_structure": {
            "train_tickers": args.train_tickers,
            "foundation_tickers": args.foundation_tickers,
            "foundation_weight": float(args.foundation_weight),
            "ticker_intercepts": str(args.ticker_intercepts),
            "ticker_x_interactions": bool(args.ticker_x_interactions),
            "ticker_min_support": args.ticker_min_support,
            "ticker_min_support_interactions": args.ticker_min_support_interactions,
        },
        "weighting": {
            "group_reweight": str(group_reweight_mode),
            "grouping_key": args.grouping_key,
            "group_equalization": bool(group_equalization_enabled),
            "trading_universe_tickers": args.trading_universe_tickers,
            "trading_universe_upweight": float(args.trading_universe_upweight),
            "ticker_balance_mode": str(ticker_balance_mode_used),
            "audit_weight_distribution": "audit_weight_distribution.json",
        },
        "bootstrap": {
            "bootstrap_ci": bool(args.bootstrap_ci),
            "bootstrap_group": str(args.bootstrap_group),
            "bootstrap_B": int(args.bootstrap_B),
            "bootstrap_seed": int(args.bootstrap_seed),
            "ci_level": int(args.ci_level),
            "per_split_reporting": bool(args.per_split_reporting),
            "per_fold_reporting": bool(args.per_fold_reporting),
        },
        "diagnostics": {
            "split_timeline": bool(args.split_timeline),
            "per_fold_delta_chart": bool(args.per_fold_delta_chart),
            "per_group_delta_distribution": bool(args.per_group_delta_distribution),
            "skip_test_metrics": bool(args.skip_test_metrics),
        },
        "filters": active_filters,
        "config_json_source": str(args.config_json) if args.config_json else None,
        "raw_config_json": config_payload,
    }
    if not fast_trial:
        (out_dir / "config.executed.json").write_text(json.dumps(executed_config, indent=2))

    git_info = _get_git_info()
    meta: Dict[str, Any] = {
        "metadata_schema_version": 2,
        "script_version": SCRIPT_VERSION,
        "calibrator_version": "inline-v2.0",
        "best_C": best_c,
        "c_selection_rule": c_selection_rule,
        "c_search_stats": c_search_stats,
        "best_score": best_score if np.isfinite(best_score) else None,
        "features": avail_numeric,
        "features_used_final": avail_numeric,
        "categorical_features": avail_cat,
        "categorical_features_used": avail_cat,
        "ticker_intercepts": args.ticker_intercepts,
        "ticker_col": ticker_col,
        "ticker_feature_col": ticker_feature_col,
        "ticker_support": ticker_support_values,
        "ticker_other_label": ticker_other_label if ticker_support_values else None,
        "foundation_label": "FOUNDATION",
        "foundation_tickers": foundation_tickers_list,
        "foundation_weight": float(args.foundation_weight),
        "target_col": target_col,
        "train_fit_rows": int(len(train_fit_df)),
        "val_rows": int(len(val_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "csv": str(Path(args.csv).resolve()) if args.csv else None,
        "coefficients": clf_step.coef_[0].tolist(),
        "intercept": float(clf_step.intercept_[0]),
        "feature_names_out": feat_names_out,
        "calibration": args.calibrate,
        "calibration_requested": args.calibrate,
        "calibration_used": "platt" if platt_cal is not None else "none",
        "platt_skip_reason": platt_skip_reason,
        "selection_objective": args.selection_objective,
        "split_strategy": args.split_strategy,
        "window_mode": args.window_mode,
        "train_window_weeks": args.train_window_weeks,
        "validation_folds": args.validation_folds,
        "validation_window_weeks": args.validation_window_weeks,
        "embargo_days": args.embargo_days,
        "fold_count_requested": int(args.validation_folds) if str(args.split_strategy) == "walk_forward" else None,
        "fold_count_actual": int(len(walk_forward_folds)) if walk_forward_folds else 1,
        "fold_count_enforced": bool(str(args.split_strategy) == "walk_forward"),
        "val_split_mode_requested": args.val_split_mode,
        "val_split_mode_used": val_split_info.get("mode"),
        "split_group_key": split_group_key,
        "split_group_dropped_train_rows": split_group_dropped_train_rows,
        "split_group_dropped_train_fit_rows": split_group_dropped_train_fit_rows,
        "embargo_date_col_used": embargo_date_col_used,
        "embargo_mode": embargo_mode,
        "embargo_rows_dropped_train": embargo_rows_dropped_train,
        "embargo_rows_dropped_train_fit": embargo_rows_dropped_train_fit,
        "class_support": {
            "train_fit_classes": train_fit_class_count,
            "val_classes": val_class_count,
            "train_classes": train_class_count,
        },
        "splits": split_ranges,
        "split_overlap": split_overlap,
        "weighting": {
            "weight_col": args.weight_col,
            "weight_source": weight_source,
            "group_reweight": group_reweight_mode,
            "group_key_source": group_key_source,
            "foundation_weight": float(args.foundation_weight),
            "foundation_weight_applied_rows": foundation_weight_applied_rows,
            "trading_universe_upweight": float(args.trading_universe_upweight),
            "trading_universe_upweight_applied_rows": trading_universe_upweight_applied_rows,
            "ticker_balance_mode": ticker_balance_mode_used,
            "grouping_key_requested": args.grouping_key,
        },
        "fit_weights": {
            "weight_col": args.weight_col,
            "weight_source": weight_source,
            "group_reweight": group_reweight_mode,
            "group_key_source": group_key_source,
            "foundation_weight": float(args.foundation_weight),
            "foundation_weight_applied_rows": foundation_weight_applied_rows,
            "train_decay_half_life_weeks": args.train_decay_half_life_weeks,
            "fit_weight_renorm": args.fit_weight_renorm,
            "trading_universe_upweight": float(args.trading_universe_upweight),
            "trading_universe_upweight_applied_rows": trading_universe_upweight_applied_rows,
            "ticker_balance_mode": ticker_balance_mode_used,
            "grouping_key_requested": args.grouping_key,
        },
        "filters": active_filters,
        "optional_filters": active_filters,
        "group_reweight": group_reweight_mode,
        "enable_x_abs_m": enable_x_abs_m_effective,
        "ticker_x_interactions": bool(args.ticker_x_interactions),
        "ticker_interaction_cols": interaction_cols_added,
        "n_bins": n_bins,
        "eceq_bins": eceq_bins,
        "ci_level": int(args.ci_level),
        "diagnostics": {
            "split_timeline": bool(args.split_timeline),
            "per_fold_delta_chart": bool(args.per_fold_delta_chart),
            "per_group_delta_distribution": bool(args.per_group_delta_distribution),
            "per_split_reporting": bool(args.per_split_reporting),
            "per_fold_reporting": bool(args.per_fold_reporting),
        },
        "unsupported_controls_ignored": unsupported_controls,
        "warnings": trainer_warnings,
        "compat_args": {
            "val_windows": args.val_windows,
            "val_window_weeks": args.val_window_weeks,
            "train_decay_half_life_weeks": args.train_decay_half_life_weeks,
            "fit_weight_renorm": args.fit_weight_renorm,
            "n_bins": n_bins,
            "eceq_bins": eceq_bins,
            "metrics_top_tickers": args.metrics_top_tickers,
        },
        "unconsumed_args": unknown,
        "run_mode": args.run_mode,
        "dataset_fingerprint": dataset_fingerprint,
        "config_executed_path": str((out_dir / "config.executed.json").resolve()),
        "git_commit": git_info["git_commit"],
        "git_commit_datetime": git_info["git_commit_datetime"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "x_logit_prn_coef": x_logit_prn_coef,
    }
    if not fast_trial:
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[base-model] C={best_c:.3g}  train_fit={len(train_fit_df)}  val={len(val_df)}  "
        f"train={len(train_df)}  test={len(test_df)}  "
        f"features={len(avail_numeric)}num+{len(avail_cat)}cat  out={out_dir}"
    )
    return RunResult(exit_code=0, out_dir=out_dir, metrics_rows=metrics_rows)


def _score_for_objective(
    *,
    objective: str,
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int,
    eceq_bins: int,
) -> float:
    metrics = _compute_metrics(y_true, p_pred, n_bins=n_bins, eceq_bins=eceq_bins)
    obj = (objective or "logloss").strip().lower()
    if obj == "brier":
        return float(metrics["brier"])
    if obj == "ece_q":
        return float(metrics["ece_q"])
    return float(metrics["logloss"])


def _describe_weight_vector(weights: np.ndarray) -> Dict[str, Optional[float]]:
    arr = np.asarray(weights, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {
            "n": int(len(arr)),
            "min": None,
            "mean": None,
            "max": None,
            "p01": None,
            "p99": None,
        }
    return {
        "n": int(len(arr)),
        "min": float(np.min(valid)),
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
        "p01": float(np.percentile(valid, 1)),
        "p99": float(np.percentile(valid, 99)),
    }


def _describe_group_sums(weights: np.ndarray, group_key: Optional[pd.Series]) -> Optional[Dict[str, Any]]:
    if group_key is None:
        return None
    if len(group_key) != len(weights):
        return None
    grouped = pd.DataFrame(
        {"group": group_key.astype("string"), "w": np.asarray(weights, dtype=float)}
    ).dropna(subset=["group", "w"])
    if grouped.empty:
        return None
    sums = grouped.groupby("group", dropna=False)["w"].sum()
    if sums.empty:
        return None
    return {
        "n_groups": int(len(sums)),
        "min": float(sums.min()),
        "mean": float(sums.mean()),
        "max": float(sums.max()),
    }


def _split_composition_row(
    *,
    split_name: str,
    frame: pd.DataFrame,
    target_col: str,
    week_col: str,
) -> Dict[str, Any]:
    y = pd.to_numeric(frame[target_col], errors="coerce") if target_col in frame.columns else pd.Series(dtype=float)
    n_pos = int((y == 1).sum()) if not y.empty else 0
    n_neg = int((y == 0).sum()) if not y.empty else 0
    class_count = int(y.dropna().nunique()) if not y.empty else 0
    row: Dict[str, Any] = {
        "split": split_name,
        "n_rows": int(len(frame)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "class_count": class_count,
        "week_start": None,
        "week_end": None,
        "n_tickers": int(frame["ticker"].nunique(dropna=True)) if "ticker" in frame.columns else None,
        "n_group_id": int(frame["group_id"].nunique(dropna=True)) if "group_id" in frame.columns else None,
        "n_contract_id": None,
    }
    week_range = _week_range(frame, week_col)
    if week_range:
        row["week_start"] = week_range[0]
        row["week_end"] = week_range[1]
    contract_id = _build_contract_id_series(frame)
    if contract_id is not None:
        row["n_contract_id"] = int(pd.Series(contract_id).nunique(dropna=True))
    return row


def _build_base_metrics_summary(
    *,
    metrics_rows: List[Dict[str, Any]],
    split_ranges: Dict[str, Any],
    out_dir: Path,
) -> None:
    def _find_metric(split_name: str, model_name: str, field: str) -> Optional[float]:
        for row in metrics_rows:
            if row.get("split") == split_name and row.get("model") == model_name:
                try:
                    return float(row.get(field)) if row.get(field) is not None else None
                except Exception:
                    return None
        return None

    val_logloss = _find_metric("val", "logit", "logloss")
    val_baseline_logloss = _find_metric("val", "baseline_pRN", "logloss")
    summary: Dict[str, Any] = {
        "val_logloss_mean": val_logloss,
        "val_logloss_std": 0.0 if val_logloss is not None else None,
        "val_logloss_by_window": None,
        "split_ranges": split_ranges,
    }
    if val_logloss is not None:
        window_row = {
            "window": "val_single",
            "window_start": (split_ranges.get("val_rows_range") or [None, None])[0],
            "window_end": (split_ranges.get("val_rows_range") or [None, None])[1],
            "logloss": val_logloss,
            "baseline_logloss": val_baseline_logloss,
        }
        summary["val_logloss_by_window"] = [window_row]
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2))


def _build_calibration_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", required=False)
    parser.add_argument("--out-dir", required=False)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--run-mode", default="manual", choices=["manual", "auto_search"])
    parser.add_argument(
        "--weight-col-strategy",
        default=None,
        choices=["auto", "weight_final", "sample_weight_final", "uniform"],
    )
    parser.add_argument(
        "--features",
        default=(
            "x_logit_prn,log_m_fwd,abs_log_m_fwd,"
            "rv20,rv20_sqrtT,log_m_fwd_over_volT,log_rel_spread,"
            "had_fallback,had_intrinsic_drop,had_band_clip,prn_raw_gap,dividend_yield"
        ),
    )
    parser.add_argument("--categorical-features", default="")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--week-col", default="week_friday")
    parser.add_argument("--ticker-col", default="ticker")
    parser.add_argument("--calibrate", default="none", choices=["none", "platt"])
    parser.add_argument("--C-grid", default="0.05,0.1,0.2,0.5,1,2,5")
    parser.add_argument("--test-weeks", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument(
        "--ticker-intercepts",
        default="non_foundation",
        choices=["none", "all", "non_foundation"],
    )
    parser.add_argument("--foundation-tickers", default="")
    parser.add_argument("--foundation-weight", type=float, default=1.0)
    parser.add_argument("--tdays-allowed", default="")
    parser.add_argument("--asof-dow-allowed", default="")
    parser.add_argument("--weight-col", default="weight_final")
    parser.add_argument("--group-reweight", default="none")
    parser.add_argument(
        "--split-strategy",
        default="walk_forward",
        choices=["walk_forward", "single_holdout"],
    )
    parser.add_argument(
        "--window-mode",
        default="rolling",
        choices=["rolling", "expanding"],
    )
    parser.add_argument("--train-window-weeks", type=int, default=52)
    parser.add_argument("--validation-folds", type=int, default=4)
    parser.add_argument("--validation-window-weeks", type=int, default=8)
    parser.add_argument("--embargo-days", type=int, default=2)
    parser.add_argument(
        "--val-split-mode",
        default="week_group",
        choices=["week_group", "row_tail"],
    )
    parser.add_argument("--val-weeks", type=int, default=0)
    parser.add_argument("--strict-args", action="store_true")
    parser.add_argument("--calib-frac-of-train", type=float, default=0.20)
    parser.add_argument("--selection-objective", default="logloss")
    parser.add_argument("--val-window-weeks", type=int, default=None)
    parser.add_argument("--val-windows", type=int, default=None)
    parser.add_argument("--train-decay-half-life-weeks", type=float, default=None)
    parser.add_argument("--fit-weight-renorm", default=None)
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--eceq-bins", type=int, default=None)
    parser.add_argument("--metrics-top-tickers", type=int, default=None)
    parser.add_argument("--ticker-x-interactions", action="store_true")
    parser.add_argument("--add-interactions", action="store_true")
    parser.add_argument("--ticker-min-support", type=int, default=None)
    parser.add_argument("--ticker-min-support-interactions", type=int, default=None)
    parser.add_argument("--grouping-key", default=None)
    parser.add_argument("--group-equalization", action="store_true")
    parser.add_argument("--no-group-equalization", action="store_true")
    parser.add_argument("--trading-universe-tickers", default=None)
    parser.add_argument("--trading-universe-upweight", type=float, default=1.0)
    parser.add_argument(
        "--ticker-balance-mode",
        default="none",
        choices=["none", "sqrt_inv_clipped"],
    )
    parser.add_argument("--train-tickers", default=None)
    parser.add_argument("--max-abs-logm", type=float, default=None)
    parser.add_argument("--drop-prn-extremes", action="store_true")
    parser.add_argument("--prn-eps", type=float, default=None)
    parser.add_argument("--prn-below", type=float, default=None)
    parser.add_argument("--prn-above", type=float, default=None)
    parser.add_argument("--enable-x-abs-m", action="store_true")
    parser.add_argument("--auto-drop-near-constant", action="store_true")
    parser.add_argument("--no-auto-drop-near-constant", action="store_true")
    parser.add_argument("--fallback-to-baseline-if-worse", action="store_true")
    parser.add_argument("--no-fallback-to-baseline-if-worse", action="store_true")
    parser.add_argument("--split-timeline", action="store_true")
    parser.add_argument("--per-fold-delta-chart", action="store_true")
    parser.add_argument("--per-group-delta-distribution", action="store_true")
    parser.add_argument("--per-split-reporting", action="store_true")
    parser.add_argument("--per-fold-reporting", action="store_true")
    parser.add_argument("--skip-test-metrics", action="store_true")
    parser.add_argument("--fast-trial", action="store_true")
    parser.add_argument("--allow-defaults", action="store_true")
    parser.add_argument("--allow-iid-bootstrap", action="store_true")
    parser.add_argument("--ci-level", type=int, default=95)
    parser.add_argument("--bootstrap-ci", action="store_true")
    parser.add_argument("--bootstrap-B", dest="bootstrap_B", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", dest="bootstrap_seed", type=int, default=0)
    parser.add_argument(
        "--bootstrap-group",
        dest="bootstrap_group",
        default="auto",
        choices=["auto", "ticker_day", "day", "iid", "contract_id", "group_id"],
    )
    return parser


def _train_base_model(calibrate_args: List[str]) -> int:
    """
    Self-contained base model trainer. Replaces the v1.5 subprocess call.
    Trains a logistic regression on option-chain data and saves FinalModelBundle
    plus standard artifacts (final_model.joblib, metadata.json, metrics.csv).
    Returns 0 on success, 1 on failure.
    """
    parser = _build_calibration_arg_parser()
    args, unknown = parser.parse_known_args(calibrate_args)
    config_payload: Optional[Dict[str, Any]] = None
    if args.config_json:
        try:
            config_payload = _load_config_json(str(args.config_json))
            _apply_config_json_overrides(args, config_payload)
        except Exception as exc:
            print(f"[base-model] ERROR: failed to read --config-json: {exc}")
            return 1

    if str(getattr(args, "run_mode", "manual")).strip().lower() == "auto_search":
        print("[base-model] ERROR: run_mode='auto_search' is disabled for this trainer endpoint.")
        return 1
    if not args.csv:
        print("[base-model] ERROR: missing required --csv argument.")
        return 1
    if not args.out_dir:
        print("[base-model] ERROR: missing required --out-dir argument.")
        return 1

    try:
        _warn_or_fail_unconsumed_args(unknown_args=unknown, strict=bool(args.strict_args))
    except ValueError as exc:
        print(f"[base-model] ERROR: {exc}")
        return 1

    try:
        n_bins = _coerce_positive_bin_count(args.n_bins, default=10, arg_name="--n-bins")
        eceq_bins = _coerce_positive_bin_count(args.eceq_bins, default=n_bins, arg_name="--eceq-bins")
    except ValueError as exc:
        print(f"[base-model] ERROR: {exc}")
        return 1

    trainer_warnings: List[str] = []
    unsupported_controls = _collect_unsupported_compat_args(args)
    if unsupported_controls:
        unsupported_msg = (
            "[base-model] Unsupported manual controls were provided and ignored: "
            + ", ".join(f"{k}={v}" for k, v in unsupported_controls.items())
        )
        if args.strict_args:
            print(f"[base-model] ERROR: {unsupported_msg}")
            return 1
        print(f"[WARN] {unsupported_msg}")
        trainer_warnings.append(unsupported_msg)

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
        print("[base-model] ERROR: --group-equalization and --no-group-equalization are mutually exclusive.")
        return 1
    if int(args.ci_level) not in {90, 95, 99}:
        print("[base-model] ERROR: --ci-level must be one of 90, 95, 99.")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = _load_dataset(Path(args.csv))
    except Exception as exc:
        print(f"[base-model] ERROR loading dataset: {exc}")
        return 1

    cache = build_calibration_cache(df, args, trainer_warnings=trainer_warnings, fast_trial=bool(args.fast_trial))
    if cache is None:
        return 1
    result = run_calibration_from_cache(
        cache,
        args,
        out_dir,
        config_payload=config_payload,
        unknown=unknown,
        unsupported_controls=unsupported_controls,
        n_bins=n_bins,
        eceq_bins=eceq_bins,
        fast_trial=bool(args.fast_trial),
        skip_test_metrics=bool(args.skip_test_metrics),
    )
    return result.exit_code


def _run_calibrate(calibrate_args: List[str]) -> int:
    """Train the base model inline. Returns 0 on success, non-zero on failure."""
    return _train_base_model(calibrate_args)


def main(entry_script: Optional[str] = None) -> None:
    print("RUNNING SCRIPT:", entry_script or __file__)
    print("VERSION:", SCRIPT_VERSION)
    _args, calibrate_args = _parse_args()
    calibrate_args = list(calibrate_args)
    if not _flag_has_value(calibrate_args, "--csv") or not _flag_has_value(calibrate_args, "--out-dir"):
        print("[calibrate] Missing required --csv and/or --out-dir arguments for calibration.")
        sys.exit(2)
    rc = _run_calibrate(calibrate_args)
    sys.exit(int(rc))


if __name__ == "__main__":
    main()
