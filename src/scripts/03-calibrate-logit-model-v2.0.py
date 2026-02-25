#!/usr/bin/env python3
"""
03-calibrate-logit-model-v2.0.py

Hierarchical probabilistic model trainer combining:
- pRN-derived features (option chain history, 2 years)
- Polymarket features (implied probabilities + momentum, 3 months)

Training modes:
- pretrain: Base model on options-only (long history)
- finetune: Meta-model on PM+options (overlap window)
- joint: Single model on PM+options (overlap window)
- two_stage: Backward-compatible v1.6 mode

Outputs: P(outcome) predictions + edge estimates (vs PM implied)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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
    TwoStageBundle,
    apply_platt,
    ensure_engineered_features,
    fit_platt_on_logits,
)

MIXED_SCRIPT = REPO_ROOT / "src" / "scripts" / "03-train-mixed-model-v1.0.py"

DEFAULT_MIXED_FEATURES = (
    REPO_ROOT
    / "src"
    / "data"
    / "models"
    / "polymarket"
    / "decision_features.parquet"
)
DEFAULT_MIXED_OUT_DIR = REPO_ROOT / "src" / "data" / "models" / "mixed"

SCRIPT_VERSION = "v2.0.0"

# New v2.0 constants
OVERLAP_WINDOW_DAYS = 90  # Default PM overlap window
DEFAULT_EDGE_OUTPUT = "edge_predictions.csv"

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
DEFAULT_PM_FEATURES = [
    "pm_mid",
    "x_logit_pm",
    "pm_spread",
    "pm_liquidity_proxy",
    "pm_momentum_1h",
    "pm_momentum_1d",
    "pm_time_to_resolution",
]

# Two-stage (Polymarket overlay) helpers
PRN_ASOF_DATE_CANDIDATES = ["asof_date", "asof_target", "asof_ts", "asof_time", "asof_datetime"]
PRN_EXPIRY_CANDIDATES = ["expiry_close_date_used", "option_expiration_used", "option_expiration_requested", "expiry_date"]
PRN_STRIKE_CANDIDATES = ["K", "threshold"]

PM_SNAPSHOT_DATE_CANDIDATES = ["snapshot_date", "decision_date", "timestamp_utc", "snapshot_time_utc", "asof_date", "asof_ts"]
PM_EXPIRY_CANDIDATES = ["expiry_date", "expiry_date_utc", "event_endDate", "expiry_ts_utc", "expiry_time_utc"]
PM_STRIKE_CANDIDATES = ["threshold", "K"]

PM_FEATURE_CANDIDATES = [
    "pm_mid",
    "pm_last",
    "pm_bid",
    "pm_ask",
    "pm_spread",
    "pm_liquidity_proxy",
    "pm_momentum_5m",
    "pm_momentum_1h",
    "pm_momentum_1d",
    "pm_volatility",
    "pm_time_to_resolution",
    "pPM_mid",
    "pPM_buy",
    "pPM_last",
    "pPM_bid",
    "pPM_ask",
    "yes_spread",
    "no_spread",
]
PM_PRIMARY_CANDIDATES = ["pm_mid", "pPM_mid", "pPM_buy", "pm_last", "pPM_last", "pm_bid", "pPM_bid"]
MIN_STAGE2_CALIB_ROWS = 50


def _build_env() -> dict:
    env = os.environ.copy()
    root = str(REPO_ROOT)
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH")
    base = f"{root}{os.pathsep}{src}"
    env["PYTHONPATH"] = f"{base}{os.pathsep}{existing}" if existing else base
    return env


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


def _select_pm_feature_cols(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in PM_FEATURE_CANDIDATES:
        if c in df.columns:
            series = pd.to_numeric(df[c], errors="coerce")
            if series.notna().any():
                cols.append(c)
    # Ensure pm_mid exists if pPM_mid provided
    if "pm_mid" not in cols and "pPM_mid" in df.columns:
        df["pm_mid"] = pd.to_numeric(df["pPM_mid"], errors="coerce")
        if df["pm_mid"].notna().any():
            cols.insert(0, "pm_mid")
    return cols


def _select_pm_primary_col(df: pd.DataFrame) -> Optional[str]:
    for c in PM_PRIMARY_CANDIDATES:
        if c in df.columns:
            return c
    return None


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


def _load_pm_dataset(path: Path) -> pd.DataFrame:
    df = _load_dataset(path)
    if df.empty:
        raise ValueError("Polymarket dataset is empty.")
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("string").str.upper()
    strike_col = _resolve_first_column(df, PM_STRIKE_CANDIDATES, "Polymarket strike")
    expiry_col = _resolve_first_column(df, PM_EXPIRY_CANDIDATES, "Polymarket expiry date")
    snapshot_col = _resolve_first_column(df, PM_SNAPSHOT_DATE_CANDIDATES, "Polymarket snapshot date")

    df["threshold"] = pd.to_numeric(df[strike_col], errors="coerce")
    df["expiry_date"] = _to_date_series(df, expiry_col, "Polymarket expiry date")
    df["snapshot_date"] = _to_date_series(df, snapshot_col, "Polymarket snapshot date")

    if "pm_mid" not in df.columns and "pPM_mid" in df.columns:
        df["pm_mid"] = pd.to_numeric(df["pPM_mid"], errors="coerce")
    return df


def _dedupe_on_keys(df: pd.DataFrame, keys: List[str], label: str) -> pd.DataFrame:
    if df.duplicated(subset=keys).any():
        dupes = int(df.duplicated(subset=keys).sum())
        print(f"[{label}] Dropping {dupes} duplicate key rows.")
        df = df.sort_values(keys).drop_duplicates(subset=keys, keep="last")
    return df


def _build_overlap(
    prn: pd.DataFrame,
    pm: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["ticker", "threshold", "expiry_date", "snapshot_date"]
    for col in keys:
        if col not in prn.columns:
            raise ValueError(f"pRN dataset missing merge key: {col}")
        if col not in pm.columns:
            raise ValueError(f"Polymarket dataset missing merge key: {col}")

    prn = _dedupe_on_keys(prn, keys, "pRN")
    pm = _dedupe_on_keys(pm, keys, "Polymarket")

    merged = prn.merge(pm, on=keys, how="inner", suffixes=("", "_pm"))
    if merged.empty:
        raise ValueError("No overlap between pRN dataset and Polymarket dataset (strict daily join).")

    # Drop duplicate columns coming from Polymarket that collide with pRN columns
    dup_cols = [c for c in merged.columns if c.endswith("_pm") and c[:-3] in prn.columns]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)
    return merged


def _build_stage2_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    *,
    random_state: int,
) -> Pipeline:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            )
        )

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=4000,
        random_state=int(random_state),
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def _compute_brier(y_true: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    y_true = y_true.astype(float)
    return float(np.mean((p - y_true) ** 2))


def _compute_ece(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error using equal-mass bins."""
    p = np.clip(p, EPS, 1.0 - EPS)
    y_true = y_true.astype(float)

    # Sort by predicted probability
    sorted_idx = np.argsort(p)
    p_sorted = p[sorted_idx]
    y_sorted = y_true[sorted_idx]

    # Create equal-mass bins
    n = len(p)
    bin_size = n // n_bins
    ece = 0.0

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else n

        if start_idx >= end_idx:
            continue

        bin_p = p_sorted[start_idx:end_idx]
        bin_y = y_sorted[start_idx:end_idx]

        if len(bin_p) == 0:
            continue

        avg_pred = np.mean(bin_p)
        avg_true = np.mean(bin_y)
        bin_weight = len(bin_p) / n

        ece += bin_weight * abs(avg_pred - avg_true)

    return float(ece)


def _compute_metrics(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive metrics including logloss, Brier, and ECE."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return {
        "logloss": float(log_loss(y_true, p)),
        "brier": _compute_brier(y_true, p),
        "ece": _compute_ece(y_true, p),
    }


def _compute_edge(
    df: pd.DataFrame,
    *,
    p_final_col: str = "p_final",
    pm_col: str = "pm_mid",
    bootstrap_iters: int = 1000,
) -> pd.DataFrame:
    """Compute edge with confidence intervals using bootstrap."""
    result = df.copy()
    p_final = pd.to_numeric(result[p_final_col], errors="coerce").to_numpy()
    p_pm = pd.to_numeric(result[pm_col], errors="coerce").to_numpy()

    edge = p_final - p_pm
    result["edge"] = edge

    # Bootstrap confidence intervals
    edges_boot = []
    rng = np.random.RandomState(42)
    valid_mask = np.isfinite(edge)
    edge_valid = edge[valid_mask]

    if len(edge_valid) > 0:
        for _ in range(bootstrap_iters):
            idx = rng.choice(len(edge_valid), size=len(edge_valid), replace=True)
            edges_boot.append(np.mean(edge_valid[idx]))

        result["edge_lower"] = float(np.percentile(edges_boot, 2.5))
        result["edge_upper"] = float(np.percentile(edges_boot, 97.5))
    else:
        result["edge_lower"] = np.nan
        result["edge_upper"] = np.nan

    return result


def _save_edge_predictions(
    df: pd.DataFrame,
    out_path: Path,
    *,
    keys: List[str] = ["ticker", "threshold", "expiry_date", "snapshot_date"],
    value_cols: List[str] = ["p_base", "p_pm", "p_final", "edge", "edge_lower", "edge_upper"],
) -> None:
    """Save edge predictions to CSV."""
    output_cols = keys + value_cols
    available_cols = [c for c in output_cols if c in df.columns]

    if "p_pm" not in df.columns and "pm_mid" in df.columns:
        df["p_pm"] = df["pm_mid"]
        available_cols = [c for c in output_cols if c in df.columns]

    edge_df = df[available_cols].copy()
    edge_df.to_csv(out_path, index=False)
    print(f"[edge] Saved edge predictions to {out_path}")


def _parse_overlap_window(window_str: str) -> int:
    """Parse overlap window string to days."""
    if window_str.endswith("days"):
        return int(window_str.replace("days", ""))
    if window_str.endswith("months"):
        return int(window_str.replace("months", "")) * 30
    raise ValueError(f"Invalid overlap window format: {window_str}")


def _validate_time_safety(
    pretrain_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    overlap_start: Any,
) -> None:
    """Validate no time leakage between pretrain and overlap."""
    pretrain_max = pretrain_df["snapshot_date"].max()
    overlap_min = overlap_df["snapshot_date"].min()

    if pretrain_max >= overlap_start:
        raise ValueError(
            f"Pretrain data leaks into overlap window: "
            f"pretrain_max={pretrain_max}, overlap_start={overlap_start}"
        )
    print(f"[validation] Time safety check passed: pretrain_max={pretrain_max} < overlap_start={overlap_start}")


def _parse_args() -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Train probabilistic model v2.0 with PM+pRN integration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # New v2.0 training mode arguments
    parser.add_argument(
        "--training-mode",
        choices=["pretrain", "finetune", "joint", "two_stage"],
        default="two_stage",
        help="Training workflow mode.",
    )
    parser.add_argument(
        "--feature-sources",
        choices=["options", "pm", "both"],
        default="both",
        help="Which feature sets to use.",
    )
    parser.add_argument(
        "--compute-edge",
        action="store_true",
        default=False,
        help="Compute and save edge predictions.",
    )
    parser.add_argument(
        "--pm-overlap-window",
        default="90days",
        help="PM overlap window (e.g., '90days', '3months').",
    )
    parser.add_argument(
        "--numeric-features",
        default=None,
        help="Override numeric features (comma-separated).",
    )
    parser.add_argument(
        "--pm-features",
        default=None,
        help="Override PM features (comma-separated).",
    )
    parser.add_argument(
        "--edge-output-path",
        default=None,
        help="Path for edge predictions CSV.",
    )

    # Legacy v1.6 arguments (backward compatibility)
    parser.add_argument(
        "--model-kind",
        choices=["calibrate", "mixed", "both"],
        default="calibrate",
        help="Which model(s) to run (legacy v1.6 mode).",
    )

    # Mixed-model arguments (prefixed to avoid collisions with calibrator flags)
    parser.add_argument(
        "--mixed-features",
        default=str(DEFAULT_MIXED_FEATURES),
        help="Decision features path.",
    )
    parser.add_argument(
        "--mixed-out-dir",
        default=str(DEFAULT_MIXED_OUT_DIR),
        help="Output root directory for mixed model runs.",
    )
    parser.add_argument(
        "--mixed-run-id",
        default=None,
        help="Optional run id for mixed model.",
    )
    parser.add_argument(
        "--mixed-model",
        default="residual",
        choices=["residual", "blend"],
        help="Mixed model type.",
    )
    parser.add_argument("--mixed-pm-col", default="pm_mid", help="Polymarket column.")
    parser.add_argument("--mixed-prn-col", default="pRN", help="pRN column.")
    parser.add_argument("--mixed-label-col", default="label", help="Label column.")
    parser.add_argument(
        "--mixed-features-cols",
        default=None,
        help="Comma-separated feature columns (optional).",
    )
    parser.add_argument("--mixed-train-frac", type=float, default=0.7, help="Train fraction.")
    parser.add_argument("--mixed-walk-forward", action="store_true", help="Enable walk-forward splits.")
    parser.add_argument("--mixed-wf-train-days", type=int, default=180, help="WF train window (days).")
    parser.add_argument("--mixed-wf-test-days", type=int, default=30, help="WF test window (days).")
    parser.add_argument("--mixed-wf-step-days", type=int, default=30, help="WF step size (days).")
    parser.add_argument("--mixed-max-splits", type=int, default=6, help="Max walk-forward splits.")
    parser.add_argument("--mixed-embargo-days", type=int, default=2, help="Embargo window (days).")
    parser.add_argument(
        "--mixed-min-time-to-resolution-days",
        type=float,
        default=0.0,
        help="Minimum time to resolution (days).",
    )
    parser.add_argument("--mixed-alpha", type=float, default=1.0, help="Ridge alpha.")

    # Two-stage overlay (Polymarket) arguments
    parser.add_argument(
        "--two-stage-mode",
        action="store_true",
        help="Enable two-stage overlay: base pRN model + Polymarket meta-model.",
    )
    parser.add_argument(
        "--two-stage-prn-csv",
        default=None,
        help="Override pRN dataset for Stage A (defaults to --csv).",
    )
    parser.add_argument(
        "--two-stage-pm-csv",
        default=None,
        help="Polymarket dataset CSV/Parquet for Stage B (decision_features).",
    )
    parser.add_argument(
        "--two-stage-label-col",
        default=None,
        help="Override label column for Stage B (default: label if present).",
    )

    return parser.parse_known_args()


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


def _train_base_model(calibrate_args: List[str]) -> int:
    """
    Self-contained base model trainer. Replaces the v1.5 subprocess call.
    Trains a logistic regression on option-chain data and saves FinalModelBundle
    plus standard artifacts (final_model.joblib, metadata.json, metrics.csv).
    Returns 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--features",
        default=(
            "x_logit_prn,log_m_fwd,abs_log_m_fwd,T_days,sqrt_T_years,"
            "rv20,rv20_sqrtT,log_m_fwd_over_volT,log_rel_spread,"
            "had_fallback,had_intrinsic_drop,had_band_clip,prn_raw_gap,dividend_yield"
        ),
    )
    parser.add_argument("--categorical-features", default="spot_scale_used")
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
    args, _ = parser.parse_known_args(calibrate_args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = _load_dataset(Path(args.csv))
    except Exception as exc:
        print(f"[base-model] ERROR loading dataset: {exc}")
        return 1

    # Parse feature lists
    numeric_features = [f.strip() for f in args.features.split(",") if f.strip()]
    categorical_features = [
        f.strip() for f in (args.categorical_features or "").split(",") if f.strip()
    ]
    ticker_col = args.ticker_col

    # Apply engineered features
    df = ensure_engineered_features(df, numeric_features)

    # Optional T_days filter
    if args.tdays_allowed:
        tdays = [int(t.strip()) for t in args.tdays_allowed.split(",") if t.strip()]
        if tdays and "T_days" in df.columns:
            df = df[pd.to_numeric(df["T_days"], errors="coerce").isin(tdays)].copy()

    # Resolve target column
    target_col = args.target_col
    if not target_col:
        for cand in ["label", "outcome_ST_gt_K", "outcome"]:
            if cand in df.columns:
                target_col = cand
                break
    if not target_col or target_col not in df.columns:
        print("[base-model] ERROR: no target column found (tried label, outcome_ST_gt_K, outcome).")
        return 1

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    if df.empty:
        print("[base-model] ERROR: no valid rows after label filter.")
        return 1

    # Derive week_friday if missing
    week_col = args.week_col
    if week_col not in df.columns:
        for date_cand in ["asof_date", "snapshot_time_utc", "asof_datetime_utc", "snapshot_date"]:
            if date_cand in df.columns:
                df[week_col] = _compute_week_friday(df[date_cand])
                break

    # Time-based split
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

    if train_df.empty:
        print("[base-model] ERROR: training set empty after time split.")
        return 1

    # Resolve which features exist in the data
    avail_numeric = [f for f in numeric_features if f in train_df.columns]
    avail_numeric = _drop_all_nan_features(train_df, avail_numeric, "base-model")
    avail_cat = [f for f in categorical_features if f in train_df.columns]
    if not avail_numeric:
        print(f"[base-model] ERROR: none of {numeric_features} found in dataset.")
        return 1

    # Ticker intercepts: encode ticker as a categorical feature
    ticker_feature_col: Optional[str] = None
    foundation_tickers_list: Optional[List[str]] = None
    if args.ticker_intercepts != "none" and ticker_col in train_df.columns:
        ticker_feature_col = "_ticker_feature"
        foundation_set = {
            t.strip().upper()
            for t in args.foundation_tickers.split(",")
            if t.strip()
        }
        if foundation_set:
            foundation_tickers_list = list(foundation_set)

        def _make_ticker_col(frame: pd.DataFrame) -> "pd.Series[str]":
            raw = frame[ticker_col].astype(str).str.upper()
            if args.ticker_intercepts == "non_foundation" and foundation_set:
                return raw.where(~raw.isin(foundation_set), "FOUNDATION")
            return raw

        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df[ticker_feature_col] = _make_ticker_col(train_df)
        if not test_df.empty:
            test_df[ticker_feature_col] = _make_ticker_col(test_df)
        avail_cat = avail_cat + [ticker_feature_col]

    all_feat_cols = avail_numeric + avail_cat

    # C-grid search using a simple val split (last 20 % of train)
    try:
        c_grid = [float(c.strip()) for c in args.C_grid.split(",") if c.strip()]
    except Exception:
        c_grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    def _make_pipe(c: float) -> Pipeline:
        transformers: List = []
        if avail_numeric:
            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                avail_numeric,
            ))
        if avail_cat:
            transformers.append((
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore", drop="first", sparse_output=False,
                    )),
                ]),
                avail_cat,
            ))
        pre = ColumnTransformer(transformers, remainder="drop")
        clf = LogisticRegression(
            C=c, solver="lbfgs", max_iter=4000, random_state=args.random_state,
        )
        return Pipeline([("pre", pre), ("clf", clf)])

    X_train = train_df[all_feat_cols]
    y_train = train_df[target_col].to_numpy(dtype=float)
    val_cut = max(1, int(len(X_train) * 0.8))
    X_fit, X_val = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
    y_fit, y_val = y_train[:val_cut], y_train[val_cut:]

    best_c = c_grid[len(c_grid) // 2]
    best_ll = float("inf")
    if len(y_val) >= 10:
        for c in c_grid:
            try:
                p = _make_pipe(c).fit(X_fit, y_fit).predict_proba(X_val)[:, 1]
                ll = log_loss(y_val, np.clip(p, EPS, 1.0 - EPS))
                if ll < best_ll:
                    best_ll, best_c = ll, c
            except Exception:
                continue

    # Train final model on all training data
    final_pipe = _make_pipe(best_c)
    final_pipe.fit(X_train, y_train)

    # Optional Platt calibration on val split
    platt_cal = None
    if args.calibrate == "platt" and len(y_val) >= 50:
        logits = _make_pipe(best_c).fit(X_fit, y_fit).decision_function(X_val)
        platt_cal = fit_platt_on_logits(
            logits=logits, y=y_val, w_fit=None, random_state=args.random_state,
        )

    # Save FinalModelBundle
    bundle = FinalModelBundle(
        kind="logit+platt" if platt_cal is not None else "logit",
        mode="pooled",
        numeric_features=avail_numeric,
        categorical_features=avail_cat,
        ticker_col=ticker_col,
        ticker_feature_col=ticker_feature_col,
        ticker_intercepts=args.ticker_intercepts,
        foundation_tickers=foundation_tickers_list,
        base_pipeline=final_pipe,
        platt_calibrator=platt_cal,
    )
    joblib.dump(bundle, out_dir / "final_model.joblib")
    joblib.dump(final_pipe, out_dir / "base_pipeline.joblib")

    # Compute test metrics
    metrics_rows: List[Dict[str, Any]] = []
    if not test_df.empty:
        X_test = test_df[all_feat_cols]
        y_test = test_df[target_col].to_numpy(dtype=float)
        p_test = final_pipe.predict_proba(X_test)[:, 1]
        p_baseline_raw = pd.to_numeric(
            test_df.get("pRN", pd.Series(dtype=float)), errors="coerce"
        ).to_numpy(dtype=float)
        valid_base = np.isfinite(p_baseline_raw)
        if valid_base.any():
            metrics_rows.append({
                "split": "test", "model": "baseline_pRN",
                "n": int(valid_base.sum()),
                **_compute_metrics(
                    y_test[valid_base],
                    np.clip(p_baseline_raw[valid_base], EPS, 1.0 - EPS),
                ),
            })
        metrics_rows.append({
            "split": "test", "model": "logit",
            "n": int(len(y_test)),
            **_compute_metrics(y_test, p_test),
        })
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)

    # Save metadata
    clf_step = final_pipe.named_steps["clf"]
    try:
        feat_names_out = final_pipe.named_steps["pre"].get_feature_names_out().tolist()
    except Exception:
        feat_names_out = avail_numeric + avail_cat

    git_info = _get_git_info()
    meta: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "calibrator_version": "inline-v2.0",
        "best_C": best_c,
        "features": avail_numeric,
        "features_used_final": avail_numeric,
        "categorical_features": avail_cat,
        "categorical_features_used": avail_cat,
        "ticker_intercepts": args.ticker_intercepts,
        "ticker_col": ticker_col,
        "ticker_feature_col": ticker_feature_col,
        "target_col": target_col,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "csv": str(Path(args.csv).resolve()),
        "coefficients": clf_step.coef_[0].tolist(),
        "intercept": float(clf_step.intercept_[0]),
        "feature_names_out": feat_names_out,
        "calibration": args.calibrate,
        "git_commit": git_info["git_commit"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(
        f"[base-model] C={best_c:.3g}  train={len(train_df)}  test={len(test_df)}  "
        f"features={len(avail_numeric)}num+{len(avail_cat)}cat  out={out_dir}"
    )
    return 0


def _run_calibrate(calibrate_args: List[str]) -> int:
    """Train the base model inline. Returns 0 on success, non-zero on failure."""
    return _train_base_model(calibrate_args)


def _run_mixed(args: argparse.Namespace) -> subprocess.CompletedProcess:
    if not MIXED_SCRIPT.exists():
        raise FileNotFoundError(f"Mixed model script not found at {MIXED_SCRIPT}")

    # Resolve features file: use explicit arg if it exists, otherwise fall back
    # to the two-stage PM CSV (which _load_features can read as CSV).
    features_path = str(args.mixed_features)
    if not Path(features_path).exists():
        fallback = getattr(args, "two_stage_pm_csv", None)
        if fallback and Path(str(fallback)).exists():
            print(f"[mixed] features file not found at {features_path!r}; "
                  f"falling back to two-stage PM CSV: {fallback!r}")
            features_path = str(fallback)
        else:
            print(f"[mixed] WARNING: features file not found: {features_path!r}")

    cmd = [
        sys.executable,
        str(MIXED_SCRIPT),
        "--features",
        features_path,
        "--out-dir",
        str(args.mixed_out_dir),
        "--model",
        args.mixed_model,
        "--pm-col",
        args.mixed_pm_col,
        "--prn-col",
        args.mixed_prn_col,
        "--label-col",
        args.mixed_label_col,
        "--train-frac",
        str(args.mixed_train_frac),
        "--wf-train-days",
        str(args.mixed_wf_train_days),
        "--wf-test-days",
        str(args.mixed_wf_test_days),
        "--wf-step-days",
        str(args.mixed_wf_step_days),
        "--max-splits",
        str(args.mixed_max_splits),
        "--embargo-days",
        str(args.mixed_embargo_days),
        "--min-time-to-resolution-days",
        str(args.mixed_min_time_to_resolution_days),
        "--alpha",
        str(args.mixed_alpha),
    ]

    if args.mixed_run_id:
        cmd.extend(["--run-id", args.mixed_run_id])
    if args.mixed_features_cols:
        cmd.extend(["--features-cols", args.mixed_features_cols])
    if args.mixed_walk_forward:
        cmd.append("--walk-forward")

    print("[mixed] command:", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=_build_env(), check=False)


def _run_two_stage_overlay(
    *,
    calibrate_args: List[str],
    prn_csv: Path,
    pm_csv: Path,
    out_dir: Path,
    label_override: Optional[str],
    compute_edge: bool = False,
) -> None:
    """
    Run two-stage overlay training (backward-compatible v1.6 mode).
    Enhanced with edge computation when requested.
    """
    if not prn_csv.is_absolute():
        prn_csv = (REPO_ROOT / prn_csv).resolve()
    if not pm_csv.is_absolute():
        pm_csv = (REPO_ROOT / pm_csv).resolve()
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[two-stage] loading datasets...")

    prn_full = _load_prn_full(prn_csv)
    pm_df = _load_pm_dataset(pm_csv)

    overlap = _build_overlap(prn_full, pm_df)
    if "week_friday" not in overlap.columns or overlap["week_friday"].isna().all():
        overlap["week_friday"] = _compute_week_friday(overlap["snapshot_date"])

    label_col = label_override
    if label_col:
        if label_col not in overlap.columns:
            raise ValueError(f"Stage B label override {label_col!r} not in dataset.")
        values = pd.to_numeric(overlap[label_col], errors="coerce")
        if not values.isin([0, 1]).any():
            raise ValueError(f"Stage B label override {label_col!r} has no valid {{0,1}} values.")
    else:
        label_col = _select_label_column(overlap, ["label", "outcome_ST_gt_K", "outcome"])
        if not label_col:
            raise ValueError("Stage B requires a label column with values in {0,1}.")

    overlap[label_col] = pd.to_numeric(overlap[label_col], errors="coerce")
    overlap = overlap[overlap[label_col].isin([0, 1])].copy()
    if overlap.empty:
        raise ValueError("No valid Stage B rows after filtering labels {0,1}.")

    overlap_start = min(overlap["snapshot_date"])
    pre_overlap = prn_full[prn_full["snapshot_date"] < overlap_start].copy()
    if pre_overlap.empty:
        raise ValueError("No pre-overlap rows available for time-safe base predictions.")
    if pre_overlap["snapshot_date"].max() >= overlap_start:
        raise ValueError("Pre-overlap filter failed (leakage risk).")

    # Validate time safety
    _validate_time_safety(pre_overlap, overlap, overlap_start)

    pre_overlap_path = out_dir / "two_stage_prn_pre_overlap.csv"
    pre_overlap.to_csv(pre_overlap_path, index=False)

    oos_dir = out_dir / "two_stage_base_oos"
    oos_args = list(calibrate_args)
    oos_args = _replace_arg_value(oos_args, "--csv", str(pre_overlap_path))
    oos_args = _replace_arg_value(oos_args, "--out-dir", str(oos_dir))

    print("[two-stage] training time-safe base model on pre-overlap data...")
    rc = _run_calibrate(oos_args)
    if rc != 0:
        raise RuntimeError("Two-stage base model (pre-overlap) training failed.")

    oos_model_path = oos_dir / "final_model.joblib"
    if not oos_model_path.exists():
        raise FileNotFoundError(f"Missing OOS base model at {oos_model_path}")

    base_full_path = out_dir / "final_model.joblib"
    if not base_full_path.exists():
        raise FileNotFoundError(f"Missing base model at {base_full_path}")

    base_oos = joblib.load(oos_model_path)
    if not isinstance(base_oos, FinalModelBundle):
        raise ValueError("OOS base model artifact did not resolve to FinalModelBundle.")

    base_full = joblib.load(base_full_path)
    if not isinstance(base_full, FinalModelBundle):
        raise ValueError("Base model artifact did not resolve to FinalModelBundle.")

    overlap = ensure_engineered_features(overlap, base_oos.numeric_features)
    p_base = base_oos.predict_proba_from_df(overlap)
    overlap["p_base"] = p_base

    pm_features_override = _get_arg_value(calibrate_args, "--pm-features", default=None)
    if pm_features_override:
        pm_features = [f.strip() for f in pm_features_override.split(",") if f.strip()]
        pm_features = _drop_all_nan_features(overlap, pm_features, "two-stage")
    else:
        pm_features = _select_pm_feature_cols(overlap)

    pm_features = [
        c
        for c in pm_features
        if not c.endswith("_time") and not c.endswith("_ts") and not c.endswith("_utc")
    ]
    pm_primary_col = _select_pm_primary_col(overlap)
    if not pm_primary_col:
        raise ValueError("Could not locate Polymarket implied probability column for Stage B.")
    if pm_primary_col not in pm_features:
        pm_features.insert(0, pm_primary_col)
    if not pm_features:
        raise ValueError("No Polymarket feature columns available for Stage B.")

    numeric_features = ["p_base"] + pm_features
    if "T_days" in overlap.columns:
        numeric_features.append("T_days")

    categorical_features: List[str] = []
    if "ticker" in overlap.columns:
        categorical_features.append("ticker")
    if "snapshot_date" in overlap.columns:
        overlap["snapshot_dow"] = pd.to_datetime(overlap["snapshot_date"]).dt.day_name()
        categorical_features.append("snapshot_dow")

    stage2_df = overlap.copy()
    stage2_df["p_base"] = p_base
    for col in pm_features:
        if col in stage2_df.columns:
            stage2_df[col] = pd.to_numeric(stage2_df[col], errors="coerce")
    stage2_df = stage2_df[stage2_df[label_col].isin([0, 1])].copy()
    if stage2_df.empty:
        raise ValueError("Stage B has no rows after label filtering.")

    pm_available = stage2_df[pm_primary_col].notna()
    if pm_available.sum() == 0:
        raise ValueError("Stage B requires Polymarket implied probability values (none found).")

    stage2_df = stage2_df.copy()
    stage2_df["week_friday"] = pd.to_datetime(stage2_df["week_friday"], errors="coerce").dt.normalize()

    test_weeks_val = _get_arg_value(calibrate_args, "--test-weeks", default="20")
    try:
        test_weeks = int(test_weeks_val or 20)
    except Exception:
        test_weeks = 20

    all_weeks = stage2_df["week_friday"].dropna().sort_values().unique()
    if len(all_weeks) <= test_weeks:
        print("[two-stage] Not enough weeks for test_weeks; falling back to last 20% by snapshot_date.")
        cutoff_idx = max(1, int(len(stage2_df) * 0.8))
        cutoff_date = stage2_df.sort_values("snapshot_date")["snapshot_date"].iloc[cutoff_idx - 1]
        test_mask_all = stage2_df["snapshot_date"] >= cutoff_date
    else:
        test_set = set(all_weeks[-test_weeks:])
        test_mask_all = stage2_df["week_friday"].isin(test_set)

    train_mask_all = ~test_mask_all
    train_df_all = stage2_df[train_mask_all & pm_available].copy()
    test_df_all = stage2_df[test_mask_all].copy()

    if train_df_all.empty:
        raise ValueError("Stage B training set is empty after time split.")

    numeric_features = _drop_all_nan_features(train_df_all, numeric_features, "two-stage")
    if not numeric_features:
        raise ValueError("Stage B has no usable numeric features after dropping all-NaN.")

    stage2_features = numeric_features + categorical_features

    random_state_val = _get_arg_value(calibrate_args, "--random-state", default="7")
    try:
        random_state = int(random_state_val or 7)
    except Exception:
        random_state = 7

    pipeline = _build_stage2_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )

    calibrate_mode = (_get_arg_value(calibrate_args, "--calibrate", default="none") or "none").strip().lower()
    use_platt = calibrate_mode == "platt"
    calib_frac_val = _get_arg_value(calibrate_args, "--calib-frac-of-train", default="0.20")
    try:
        calib_frac = float(calib_frac_val or 0.20)
    except Exception:
        calib_frac = 0.20

    platt_cal = None
    train_fit_df = train_df_all
    calib_df = pd.DataFrame()
    if use_platt and 0.0 < calib_frac < 1.0:
        train_weeks = train_df_all["week_friday"].dropna().sort_values().unique()
        n_calib_weeks = max(1, int(len(train_weeks) * calib_frac))
        calib_weeks = set(train_weeks[-n_calib_weeks:])
        calib_df = train_df_all[train_df_all["week_friday"].isin(calib_weeks)].copy()
        train_fit_df = train_df_all[~train_df_all["week_friday"].isin(calib_weeks)].copy()
        if train_fit_df.empty:
            train_fit_df = train_df_all
            calib_df = pd.DataFrame()

    pipeline.fit(train_fit_df[stage2_features], train_fit_df[label_col].to_numpy(dtype=float))

    if use_platt and not calib_df.empty and len(calib_df) >= MIN_STAGE2_CALIB_ROWS:
        logits = pipeline.decision_function(calib_df[stage2_features])
        platt_cal = fit_platt_on_logits(
            logits=logits,
            y=calib_df[label_col].to_numpy(dtype=float),
            w_fit=None,
            random_state=random_state,
        )

    # Evaluation on test split
    y_test = test_df_all[label_col].to_numpy(dtype=float)
    p_base_test = test_df_all["p_base"].to_numpy(dtype=float)

    pm_test = pd.to_numeric(test_df_all.get(pm_primary_col), errors="coerce").to_numpy(dtype=float)
    pm_test_mask = np.isfinite(pm_test)

    p_final_test = p_base_test.copy()
    if pm_test_mask.any():
        test_stage2 = test_df_all.iloc[np.where(pm_test_mask)[0]]
        if platt_cal is not None:
            logits = pipeline.decision_function(test_stage2[stage2_features])
            p_stage2 = apply_platt(platt_cal, logits)
        else:
            p_stage2 = pipeline.predict_proba(test_stage2[stage2_features])[:, 1]
        p_final_test[pm_test_mask] = p_stage2

    metrics_rows: List[Dict[str, Any]] = []
    metrics_rows.append({
        "split": "test",
        "model": "stage_a",
        "n": int(len(y_test)),
        **_compute_metrics(y_test, p_base_test),
    })
    if pm_test_mask.any():
        metrics_rows.append({
            "split": "test",
            "model": "pm_baseline",
            "n": int(pm_test_mask.sum()),
            **_compute_metrics(y_test[pm_test_mask], pm_test[pm_test_mask]),
        })
    metrics_rows.append({
        "split": "test",
        "model": "two_stage",
        "n": int(len(y_test)),
        **_compute_metrics(y_test, p_final_test),
    })

    # Compute edge statistics if PM available
    if pm_test_mask.any():
        edge_test = p_final_test[pm_test_mask] - pm_test[pm_test_mask]
        edge_mean = float(np.mean(edge_test))
        edge_std = float(np.std(edge_test))
        # Add edge stats to two_stage row
        metrics_rows[-1]["edge_mean"] = edge_mean
        metrics_rows[-1]["edge_std"] = edge_std

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "two_stage_metrics.csv", index=False)
    (out_dir / "two_stage_metrics_summary.json").write_text(
        json.dumps({"rows": metrics_rows}, indent=2)
    )

    bundle = TwoStageBundle(
        base_bundle=base_full,
        stage2_pipeline=pipeline,
        stage2_feature_cols=stage2_features,
        pm_primary_col=pm_primary_col,
        platt_calibrator=platt_cal,
    )
    joblib.dump(bundle, out_dir / "two_stage_model.joblib")

    # Compute and save edge predictions if requested
    if compute_edge:
        test_df_all["p_final"] = p_final_test
        test_df_all["p_pm"] = pm_test
        edge_df = _compute_edge(test_df_all, p_final_col="p_final", pm_col="p_pm")
        edge_output_path = out_dir / DEFAULT_EDGE_OUTPUT
        _save_edge_predictions(edge_df, edge_output_path)

    git_info = _get_git_info()
    pm_coverage_pct = float(pm_available.sum() / len(stage2_df)) if len(stage2_df) > 0 else 0.0

    # Extract stage2 model coefficients and intercept
    stage2_classifier = pipeline.named_steps["clf"]
    stage2_coefficients = stage2_classifier.coef_[0].tolist()
    stage2_intercept = float(stage2_classifier.intercept_[0])

    # Extract feature names after preprocessing (handles categorical encoding)
    stage2_feature_names = stage2_features.copy()
    if categorical_features:
        # If one-hot encoding was applied, get transformed feature names
        try:
            preprocessor = pipeline.named_steps.get("pre")
            if preprocessor and hasattr(preprocessor, "get_feature_names_out"):
                stage2_feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            pass  # Keep original feature names if extraction fails

    meta = {
        "script_version": SCRIPT_VERSION,
        "training_mode": "two_stage",
        "feature_sources": "both",
        "two_stage_mode": True,
        "prn_csv": str(prn_csv),
        "pm_csv": str(pm_csv),
        "label_col": label_col,
        "overlap_rows": int(len(overlap)),
        "overlap_start_date": str(overlap_start),
        "pm_coverage_pct": pm_coverage_pct,
        "pm_primary_col": pm_primary_col,
        "pm_features": pm_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "stage2_features": stage2_features,
        "stage2_feature_names": stage2_feature_names,
        "stage2_coefficients": stage2_coefficients,
        "stage2_intercept": stage2_intercept,
        "stage2_model_type": "logistic_regression",
        "train_rows": int(len(train_df_all)),
        "test_rows": int(len(test_df_all)),
        "pm_available_rows": int(pm_available.sum()),
        "calibration": calibrate_mode,
        "base_model_full": str(base_full_path),
        "base_model_oos": str(oos_model_path),
        "dataset_fingerprints": {
            "prn": _file_fingerprint(prn_csv),
            "polymarket": _file_fingerprint(pm_csv),
        },
        "git_commit": git_info["git_commit"],
        "git_commit_datetime": git_info["git_commit_datetime"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "two_stage_metadata.json").write_text(json.dumps(meta, indent=2))

    print("[two-stage] overlap rows:", len(overlap))
    print("[two-stage] PM coverage:", f"{pm_coverage_pct:.1%}")
    print("[two-stage] metrics saved to two_stage_metrics.csv")

    try:
        pre_overlap_path.unlink()
    except Exception:
        pass


def main() -> None:
    print("RUNNING SCRIPT:", __file__)
    print("VERSION:", SCRIPT_VERSION)
    args, calibrate_args = _parse_args()
    calibrate_args = list(calibrate_args)

    # Determine effective training mode
    if args.two_stage_mode:
        effective_mode = "two_stage"
    else:
        effective_mode = args.training_mode

    print(f"[v2.0] Training mode: {effective_mode}")
    print(f"[v2.0] Feature sources: {args.feature_sources}")
    print(f"[v2.0] Compute edge: {args.compute_edge}")

    if effective_mode == "two_stage":
        # Run backward-compatible two-stage mode
        if args.two_stage_mode and args.model_kind == "mixed":
            print("[two-stage] --two-stage-mode requires calibrate or both (not mixed only).")
            sys.exit(2)

        if args.two_stage_prn_csv:
            calibrate_args = _replace_arg_value(calibrate_args, "--csv", args.two_stage_prn_csv)

        exit_code = 0
        if args.model_kind in {"calibrate", "both"} or args.two_stage_mode:
            if not _flag_has_value(calibrate_args, "--csv") or not _flag_has_value(calibrate_args, "--out-dir"):
                print("[calibrate] Missing required --csv and/or --out-dir arguments for calibration.")
                exit_code = 2
            else:
                rc = _run_calibrate(calibrate_args)
                if rc != 0:
                    exit_code = rc or 1
                elif args.two_stage_mode:
                    pm_csv_val = args.two_stage_pm_csv
                    prn_csv_val = _get_arg_value(calibrate_args, "--csv")
                    out_dir_val = _get_arg_value(calibrate_args, "--out-dir")
                    if not pm_csv_val:
                        print("[two-stage] Missing --two-stage-pm-csv for overlay.")
                        exit_code = 2
                    elif not prn_csv_val or not out_dir_val:
                        print("[two-stage] Missing --csv/--out-dir for overlay.")
                        exit_code = 2
                    else:
                        try:
                            _run_two_stage_overlay(
                                calibrate_args=calibrate_args,
                                prn_csv=Path(prn_csv_val),
                                pm_csv=Path(pm_csv_val),
                                out_dir=Path(out_dir_val),
                                label_override=args.two_stage_label_col,
                                compute_edge=args.compute_edge,
                            )
                        except Exception as exc:
                            print(f"[two-stage] FAILED: {exc}")
                            import traceback
                            traceback.print_exc()
                            exit_code = 1

        if args.model_kind in {"mixed", "both"}:
            result = _run_mixed(args)
            if result.returncode != 0 and exit_code == 0:
                exit_code = result.returncode or 1

        sys.exit(exit_code)

    else:
        # Future: Implement pretrain, finetune, joint modes
        print(f"[v2.0] Training mode '{effective_mode}' not yet fully implemented.")
        print("[v2.0] Falling back to two_stage mode for now.")
        print("[v2.0] To use, add --two-stage-mode flag.")
        sys.exit(1)


if __name__ == "__main__":
    main()
