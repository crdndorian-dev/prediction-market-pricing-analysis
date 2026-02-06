#!/usr/bin/env python3
"""
2_calibrate_logit_model_v1.5.py

Weekly pHAT calibrator (simple, robust, time-safe) with:
- Logistic regression with L2 regularization
- Numeric features + ticker intercepts (one-hot, drop='first')
- Optional interactions
- Optional Platt calibration (fit only on a CALIB slice, never on VAL/TEST)
- Recency decay on TRAIN_FIT sample weights (fit-time only; metrics use raw weights)
- Rolling-window validation (multiple contiguous windows) for model selection

Key changes vs v1.4:
--------------------
(1) Ticker-based model: adds categorical 'ticker' with OneHotEncoder(handle_unknown="ignore", drop="first")
    -> learns a per-ticker intercept adjustment (relative to a reference ticker).
(2) Keeps recency decay: exp(-ln2 * age_weeks / half_life_weeks) applied only to TRAIN_FIT weights.
(3) Replaces single VAL selection with rolling validation windows:
    - Reserve last --test-weeks as final TEST (never used for selection).
    - Build --val-windows contiguous windows of --val-window-weeks each from the period just before TEST.
    - For each window, train on all weeks strictly before the window start (time-safe), then evaluate on the window.
    - Choose C minimizing average rolling-window logloss.
(4) Adds horizon- and quality-aware engineered features (vol-time scaling, moneyness scaling, liquidity/coverage ratios).
(5) Fixes calibration split when --calibrate=none (uses all training data) and avoids VAL_POOL leakage
    by fitting a separate OOS model for VAL_POOL metrics.

New in v1.5 (foundation training):
----------------------------------
- Foundation tickers allow a stable global pRN -> pHAT mapping via ETF-only training rows.
- Pooled mode learns global coefficients + per-ticker intercepts (and optional ticker*x interactions).
- Two-stage mode fits:
  * Stage 1 (foundation only): global mapping on ETFs.
  * Stage 2 (non-foundation): offset logistic correction in logit space.
- Foundation weighting applies only to TRAIN_FIT rows.

Example runs:
-------------
Pooled:
python 02__pRN__calibrate__model__v1.5.0.py --csv data.csv --foundation-tickers SPY,QQQ,IWM --mode pooled --ticker-intercepts non_foundation

Two-stage:
python 02__pRN__calibrate__model__v1.5.0.py --csv data.csv --foundation-tickers SPY,QQQ,IWM --mode two_stage --ticker-intercepts non_foundation

Outputs:
--------
- metrics.csv (includes rolling averages + final test)
- rolling_windows.csv (per-window metrics baseline vs model)
- reliability_bins.csv (val_pool combined + test)
- base_pipeline.joblib
- calibrator.joblib (if platt)
- final_model.joblib
- metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scipy.special import expit

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from calibrate_common import (
    FOUNDATION_LABEL,
    EPS,
    FinalModelBundle,
    LogisticRegressionOffset,
    apply_group_reweight,
    apply_platt,
    build_chain_group_id,
    dedupe_preserve_order,
    ece_equal_mass,
    ensure_engineered_features,
    filter_forbidden_features,
    fit_platt_on_logits,
    find_near_constant_features,
    make_pipeline,
    make_preprocessor,
    normalize_ticker,
    resolve_moneyness_column,
    scope_tickers_by_support,
    tickers_meeting_support,
    _logit,
)
MIN_CALIB_ROWS = 50
MIN_FILTER_ROWS = 200
CALIBRATOR_VERSION = "v1.5.0"

DERIVED_FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "x_logit_prn": f"logit(pRN) clipped to range {EPS}..{1.0 - EPS}",
    "log_m": "log(K / S) using the available settled spot",
    "abs_log_m": "absolute of log_m",
    "log_m_fwd": "log(K / forward_price) when forward_price is available",
    "abs_log_m_fwd": "absolute of log_m_fwd",
    "log_T_days": "log1p(T_days)",
    "sqrt_T_years": "sqrt(T_days / 365)",
    "rv20_sqrtT": "rv20 * sqrt_T_years",
    "log_m_over_volT": "log_m / (rv20 * sqrt_T_years)",
    "abs_log_m_over_volT": "absolute of log_m_over_volT",
    "log_m_fwd_over_volT": "log_m_fwd / (rv20 * sqrt_T_years)",
    "abs_log_m_fwd_over_volT": "absolute of log_m_fwd_over_volT",
    "log_rel_spread": "log1p(rel_spread_median)",
    "fallback_any": "binary fallback indicator from asof/expiry fallback days",
    "had_fallback": "binary indicator for any fallback usage (asof or expiry)",
    "had_intrinsic_drop": "binary indicator for intrinsic-drop guardrails",
    "had_band_clip": "binary indicator for band coverage < 1",
    "prn_raw_gap": "pRN - pRN_raw",
    "x_prn_x_tdays": "x_logit_prn * T_days",
    "x_prn_x_rv20": "x_logit_prn * rv20",
    "x_prn_x_logm": "x_logit_prn * log_m",
    "x_m": "x_logit_prn * moneyness (log_m_fwd or log_m)",
    "x_abs_m": "x_logit_prn * abs(moneyness)",
    "asof_dow": "Day-of-week of as-of timestamp (Mon..Sun, UTC)",
}

def get_git_info() -> Dict[str, Optional[str]]:
    repo_root = Path(__file__).resolve().parents[1]
    commit = None
    commit_dt = None
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
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
            cwd=repo_root,
            check=True,
        )
        commit_dt = res.stdout.strip() or None
    except Exception:
        pass
    return {"git_commit": commit, "git_commit_datetime": commit_dt}

def build_required_columns(
    *,
    numeric_features: List[str],
    categorical_features: List[str],
    ticker_col: str,
    ticker_feature_col: Optional[str],
    interaction_ticker_col: Optional[str],
) -> List[str]:
    base = [
        ticker_col,
        "pRN",
        "pRN_raw",
        "K",
        "S",
        "T_days",
        "event_endDate",
        "snapshot_time_utc",
        "r",
        "dividend_yield",
        "forward_price",
        "rv20",
    ]
    extras = [ticker_feature_col, interaction_ticker_col]
    candidates = [col for col in numeric_features + categorical_features + base + extras if col]
    return dedupe_preserve_order(candidates)




# -----------------------------
# Numerics / Metrics
# -----------------------------

def weighted_brier(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray]) -> float:
    y = y.astype(float)
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    if w is None:
        return float(np.mean((p - y) ** 2))
    w = w.astype(float)
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0:
        return float(np.mean((p - y) ** 2))
    return float(np.sum(w * (p - y) ** 2) / sw)


def ece_score(y: np.ndarray, p: np.ndarray, *, n_bins: int, w: Optional[np.ndarray]) -> float:
    y = y.astype(float)
    p = np.clip(p.astype(float), 0.0, 1.0)
    if w is None:
        w = np.ones_like(p, dtype=float)
    else:
        w = w.astype(float)

    total_w = float(np.sum(w))
    if not np.isfinite(total_w) or total_w <= 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        if b1 == 1.0:
            m = (p >= b0) & (p <= b1)
        else:
            m = (p >= b0) & (p < b1)
        if not np.any(m):
            continue
        wb = w[m]
        swb = float(np.sum(wb))
        if swb <= 0:
            continue
        avg_p = float(np.sum(wb * p[m]) / swb)
        avg_y = float(np.sum(wb * y[m]) / swb)
        ece += (swb / total_w) * abs(avg_p - avg_y)
    return float(ece)


def reliability_table(y: np.ndarray, p: np.ndarray, *, n_bins: int, w: Optional[np.ndarray], label: str) -> pd.DataFrame:
    y = y.astype(float)
    p = np.clip(p.astype(float), 0.0, 1.0)
    if w is None:
        w = np.ones_like(p, dtype=float)
    else:
        w = w.astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:]), start=1):
        if b1 == 1.0:
            m = (p >= b0) & (p <= b1)
        else:
            m = (p >= b0) & (p < b1)

        if not np.any(m):
            rows.append({"label": label, "bin": i, "bin_lo": b0, "bin_hi": b1,
                         "n": 0, "weight_sum": 0.0, "avg_pred": np.nan, "emp_rate": np.nan})
            continue

        wb = w[m]
        swb = float(np.sum(wb))
        avg_p = float(np.sum(wb * p[m]) / swb) if swb > 0 else np.nan
        avg_y = float(np.sum(wb * y[m]) / swb) if swb > 0 else np.nan
        rows.append({"label": label, "bin": i, "bin_lo": b0, "bin_hi": b1,
                     "n": int(m.sum()), "weight_sum": swb, "avg_pred": avg_p, "emp_rate": avg_y})

    return pd.DataFrame(rows)


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    *,
    w: Optional[np.ndarray],
    n_bins: int,
    n_bins_q: int,
) -> Dict[str, float]:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return {
        "logloss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
        "brier": weighted_brier(y, p, w),
        "ece": ece_score(y, p, n_bins=n_bins, w=w),
        "ece_q": ece_equal_mass(y, p, n_bins=n_bins_q, sample_weight=w),
    }


def group_metrics_rows(
    *,
    split: str,
    group_type: str,
    group_value: str,
    y: np.ndarray,
    p_base: np.ndarray,
    p_model: np.ndarray,
    w: Optional[np.ndarray],
    n_bins: int,
    n_bins_q: int,
    model_label: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rows.append({
        "model": "baseline_pRN",
        "split": split,
        "group_type": group_type,
        "group_value": group_value,
        **evaluate(y, p_base, w=w, n_bins=n_bins, n_bins_q=n_bins_q),
        "n": int(len(y)),
        "weight_sum": float(np.sum(w)) if w is not None else float(len(y)),
    })
    rows.append({
        "model": model_label,
        "split": split,
        "group_type": group_type,
        "group_value": group_value,
        **evaluate(y, p_model, w=w, n_bins=n_bins, n_bins_q=n_bins_q),
        "n": int(len(y)),
        "weight_sum": float(np.sum(w)) if w is not None else float(len(y)),
    })
    return rows


# ------------------------------------------------
# Bootstrap confidence intervals for delta metrics
# ------------------------------------------------

def _build_bootstrap_group_keys(
    df_split: pd.DataFrame,
    *,
    ticker_col: str,
    asof_date_col: Optional[str],
    expiry_date_col: Optional[str],
    week_col: str = "week_friday",
    strategy: str = "auto",
) -> np.ndarray:
    """Build an array of group keys for block bootstrap resampling.

    Each unique key defines one resampling block.
    Returns np.ndarray of strings, same length as df_split.
    """
    n = len(df_split)
    if strategy == "iid":
        return np.arange(n).astype(str)

    has_ticker = ticker_col in df_split.columns
    has_asof = asof_date_col is not None and asof_date_col in df_split.columns
    has_expiry = expiry_date_col is not None and expiry_date_col in df_split.columns
    has_week = week_col in df_split.columns

    def _col_str(col: str) -> np.ndarray:
        s = df_split[col]
        dt = pd.to_datetime(s, errors="coerce")
        if dt.notna().any():
            return dt.dt.strftime("%Y-%m-%d").fillna("NA").to_numpy(dtype=str)
        return s.astype(str).fillna("NA").to_numpy(dtype=str)

    if strategy == "auto":
        if has_ticker and has_asof and has_expiry:
            t = df_split[ticker_col].astype(str).fillna("NA").to_numpy(dtype=str)
            a = _col_str(asof_date_col)
            e = _col_str(expiry_date_col)
            return np.char.add(np.char.add(np.char.add(t, "|"), np.char.add(a, "|")), e)
        if has_ticker and has_asof:
            t = df_split[ticker_col].astype(str).fillna("NA").to_numpy(dtype=str)
            a = _col_str(asof_date_col)
            print("[WARN] Bootstrap auto: no expiry column; falling back to ticker+asof_date.")
            return np.char.add(np.char.add(t, "|"), a)
        if has_asof:
            print("[WARN] Bootstrap auto: no ticker or expiry column; falling back to asof_date.")
            return _col_str(asof_date_col)
        if has_week:
            print("[WARN] Bootstrap auto: no asof_date; falling back to week column.")
            return _col_str(week_col)
        print("[WARN] Bootstrap auto: no grouping columns; falling back to iid.")
        return np.arange(n).astype(str)

    if strategy == "ticker_day":
        if has_ticker and has_asof:
            t = df_split[ticker_col].astype(str).fillna("NA").to_numpy(dtype=str)
            a = _col_str(asof_date_col)
            return np.char.add(np.char.add(t, "|"), a)
        print("[WARN] Bootstrap ticker_day: missing columns; falling back to iid.")
        return np.arange(n).astype(str)

    if strategy == "day":
        if has_asof:
            return _col_str(asof_date_col)
        if has_week:
            print("[WARN] Bootstrap day: no asof_date; falling back to week column.")
            return _col_str(week_col)
        print("[WARN] Bootstrap day: no date columns; falling back to iid.")
        return np.arange(n).astype(str)

    # Unknown strategy: iid
    return np.arange(n).astype(str)


def bootstrap_delta_ci(
    y: np.ndarray,
    p_base: np.ndarray,
    p_model: np.ndarray,
    w: Optional[np.ndarray],
    *,
    group_keys: np.ndarray,
    n_bins: int,
    n_bins_q: int,
    B: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Block bootstrap percentile CIs for delta metrics (model minus baseline).

    Returns dict keyed by metric name, each value is:
        {"ci_lo": float|None, "ci_hi": float|None, "n_groups": int, "B": int}
    """
    unique_groups = np.unique(group_keys)
    n_groups = len(unique_groups)

    if n_groups < 5:
        print(f"[WARN] Bootstrap: only {n_groups} groups; CIs may be unreliable.")

    # Build index: group_key -> array of row positions
    group_indices: Dict[str, np.ndarray] = {}
    for g in unique_groups:
        group_indices[g] = np.where(group_keys == g)[0]

    metric_keys = ["logloss", "brier", "ece", "ece_q"]
    deltas = {mk: np.empty(B, dtype=float) for mk in metric_keys}

    rng = np.random.default_rng(seed)

    for b in range(B):
        sampled = rng.choice(unique_groups, size=n_groups, replace=True)
        idx = np.concatenate([group_indices[g] for g in sampled])
        if len(idx) == 0:
            for mk in metric_keys:
                deltas[mk][b] = float("nan")
            continue
        y_b = y[idx]
        p_base_b = p_base[idx]
        p_model_b = p_model[idx]
        w_b = w[idx] if w is not None else None
        met_base = evaluate(y_b, p_base_b, w=w_b, n_bins=n_bins, n_bins_q=n_bins_q)
        met_model = evaluate(y_b, p_model_b, w=w_b, n_bins=n_bins, n_bins_q=n_bins_q)
        for mk in metric_keys:
            deltas[mk][b] = met_model[mk] - met_base[mk]

    lo_q = 100.0 * alpha / 2.0
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for mk in metric_keys:
        d = deltas[mk]
        finite_mask = np.isfinite(d)
        if not np.any(finite_mask):
            result[mk] = {"ci_lo": None, "ci_hi": None, "n_groups": n_groups, "B": B}
            continue
        d_finite = d[finite_mask]
        ci_lo = float(np.percentile(d_finite, lo_q))
        ci_hi = float(np.percentile(d_finite, hi_q))
        if ci_lo > ci_hi:
            ci_lo, ci_hi = ci_hi, ci_lo
        result[mk] = {"ci_lo": ci_lo, "ci_hi": ci_hi, "n_groups": n_groups, "B": B}

    return result


# -----------------------------
# Weights utilities
# -----------------------------

def renorm_fit_weights_mean1(w: np.ndarray) -> np.ndarray:
    """Renormalize weights so sum(w) == n (mean==1). Use ONLY for FIT weights."""
    w = w.astype(float).copy()
    if np.any(w < 0):
        raise ValueError("Negative sample weights found; abort.")
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w)
    return w * (len(w) / s)


def apply_recency_decay(
    w: np.ndarray,
    weeks: np.ndarray,
    *,
    half_life_weeks: float,
) -> np.ndarray:
    """Multiply weights by exp(-ln2 * age_weeks / half_life)."""
    if half_life_weeks <= 0:
        return w

    weeks_s = pd.to_datetime(pd.Series(weeks), errors="coerce")
    if weeks_s.isna().any():
        return w  # fail-safe

    last = weeks_s.max()
    age_weeks = (last - weeks_s).dt.days.to_numpy(dtype=float) / 7.0
    decay = np.exp(-math.log(2.0) * age_weeks / float(half_life_weeks))
    return w.astype(float) * decay.astype(float)


def build_fit_weights(
    w_raw: np.ndarray,
    weeks: np.ndarray,
    *,
    foundation_mask: Optional[np.ndarray],
    foundation_weight: float,
    decay_half_life_weeks: float,
    fit_weight_renorm: str,
    group_id: Optional[pd.Series] = None,
    enable_group_reweight: bool = False,
) -> np.ndarray:
    w_fit = w_raw.astype(float).copy()
    if foundation_mask is not None and foundation_weight != 1.0:
        w_fit = w_fit * np.where(foundation_mask, foundation_weight, 1.0)
    if decay_half_life_weeks and decay_half_life_weeks > 0:
        w_fit = apply_recency_decay(w_fit, weeks, half_life_weeks=decay_half_life_weeks)
    # Apply group reweighting if enabled
    if enable_group_reweight and group_id is not None:
        mask = np.ones(len(w_fit), dtype=bool)  # Reweight all rows in this subset
        w_fit = apply_group_reweight(w_fit, group_id, mask)
    if fit_weight_renorm == "mean1":
        w_fit = renorm_fit_weights_mean1(w_fit)
    return w_fit


# -----------------------------
# Feature parsing + sanity
# -----------------------------

def pick_target_column(df: pd.DataFrame, user_target: Optional[str]) -> str:
    if user_target:
        if user_target not in df.columns:
            raise ValueError(f"--target-col {user_target} not found.")
        return user_target
    for cand in ["outcome_ST_gt_K", "target", "y", "label"]:
        if cand in df.columns:
            return cand
    raise ValueError("Could not detect target column; pass --target-col.")


def pick_weight_column(df: pd.DataFrame, user_weight: Optional[str]) -> Optional[str]:
    if user_weight:
        if user_weight.lower() in ("none", "null", "no"):
            return None
        if user_weight not in df.columns:
            raise ValueError(f"--weight-col {user_weight} not found.")
        return user_weight
    for cand in ["sample_weight_final", "quality_weight"]:
        if cand in df.columns:
            return cand
    return None


def parse_feature_list(features_arg: str) -> List[str]:
    return [s.strip() for s in features_arg.split(",") if s.strip()]


def parse_categorical_list(features_arg: str) -> List[str]:
    if not features_arg:
        return []
    if features_arg.strip().lower() in ("none", "null", "no"):
        return []
    return [s.strip() for s in features_arg.split(",") if s.strip()]


def parse_ticker_list(tickers_arg: str) -> List[str]:
    if not tickers_arg:
        return []
    if tickers_arg.strip().lower() in ("none", "null", "no"):
        return []
    return [s.strip() for s in tickers_arg.split(",") if s.strip()]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAY_NAME_TO_INDEX = {name.lower(): idx for idx, name in enumerate(DAY_NAMES)}
DAY_ABBREV_TO_INDEX = {name[:3].lower(): idx for idx, name in enumerate(DAY_NAMES)}
TIME_REP_FEATURES = ["T_days", "log_T_days", "sqrt_T_years", "T_years"]
ABSOLUTE_TIME_FEATURES = {
    "asof_date",
    "snapshot_time_utc",
    "week_friday",
    "week_monday",
    "expiry_close_date_used",
    "expiry_date",
    "event_endDate",
}


def parse_tdays_allowed(value: str) -> Optional[List[int]]:
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
            raise ValueError(f"Invalid T_days value '{token}' in --tdays-allowed.") from exc
        if tdays < 0:
            raise ValueError(f"T_days must be >= 0 (got {tdays}).")
        if tdays not in parsed:
            parsed.append(tdays)
    return parsed


def parse_asof_dow_allowed(value: str) -> Optional[List[str]]:
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
            if key in DAY_ABBREV_TO_INDEX:
                name = DAY_NAMES[DAY_ABBREV_TO_INDEX[key]]
            else:
                raise ValueError(f"Invalid day-of-week token '{token}'. Use Mon..Sun or 0..6.")
        if name not in allowed:
            allowed.append(name)
    return allowed


def resolve_asof_date_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["asof_date", "snapshot_time_utc", "snapshot_time", "asof_datetime_utc"]:
        if cand in df.columns:
            return cand
    return None


def resolve_expiry_date_column(df: pd.DataFrame) -> Optional[str]:
    for cand in ["expiry_date", "expiry_close_date_used", "event_endDate"]:
        if cand in df.columns:
            return cand
    return None


def add_asof_dow_column(df: pd.DataFrame, asof_col: str) -> None:
    dt = pd.to_datetime(df[asof_col], errors="coerce", utc=True)
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"{asof_col} has {bad} NaT values; cannot derive asof_dow.")
    dow_idx = dt.dt.weekday.astype(int)
    df["asof_dow"] = dow_idx.map(lambda idx: DAY_NAMES[idx])


def build_feature_spec(
    df: pd.DataFrame,
    *,
    numeric_features: List[str],
    categorical_features: List[str],
    asof_dow_col: Optional[str],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    numeric = dedupe_preserve_order(numeric_features)
    categorical = dedupe_preserve_order(categorical_features)

    if "asof_dow" in numeric:
        raise ValueError("asof_dow must be categorical (not numeric) to avoid time leakage.")

    banned_numeric = sorted(set(numeric) & ABSOLUTE_TIME_FEATURES)
    if banned_numeric:
        raise ValueError(f"Absolute time features are not allowed: {banned_numeric}")
    banned_categorical = sorted(set(categorical) & ABSOLUTE_TIME_FEATURES)
    if banned_categorical:
        raise ValueError(f"Absolute time features are not allowed: {banned_categorical}")

    tdays_constant = False
    if "T_days" in df.columns:
        tdays_vals = pd.to_numeric(df["T_days"], errors="coerce").dropna().unique()
        tdays_constant = len(tdays_vals) <= 1

    asof_dow_constant = False
    if asof_dow_col and asof_dow_col in df.columns:
        asof_dow_constant = df[asof_dow_col].dropna().nunique() <= 1

    dropped_numeric: List[str] = []
    dropped_categorical: List[str] = []
    if tdays_constant:
        for feat in TIME_REP_FEATURES:
            if feat in numeric:
                numeric.remove(feat)
                dropped_numeric.append(feat)
        if "x_prn_x_tdays" in numeric:
            numeric.remove("x_prn_x_tdays")
            dropped_numeric.append("x_prn_x_tdays")
    else:
        if "T_days" in numeric:
            for feat in ["log_T_days", "sqrt_T_years", "T_years"]:
                if feat in numeric:
                    numeric.remove(feat)
                    dropped_numeric.append(feat)
        elif "log_T_days" in numeric:
            for feat in ["T_days", "sqrt_T_years", "T_years"]:
                if feat in numeric:
                    numeric.remove(feat)
                    dropped_numeric.append(feat)
        elif "sqrt_T_years" in numeric:
            for feat in ["T_days", "log_T_days", "T_years"]:
                if feat in numeric:
                    numeric.remove(feat)
                    dropped_numeric.append(feat)
        elif "T_years" in numeric:
            for feat in ["T_days", "log_T_days", "sqrt_T_years"]:
                if feat in numeric:
                    numeric.remove(feat)
                    dropped_numeric.append(feat)

    if asof_dow_constant and asof_dow_col and asof_dow_col in categorical:
        categorical.remove(asof_dow_col)
        dropped_categorical.append(asof_dow_col)

    derived_features: List[str] = []
    if asof_dow_col and asof_dow_col in df.columns:
        derived_features.append(asof_dow_col)

    return numeric, categorical, {
        "tdays_constant": tdays_constant,
        "asof_dow_constant": asof_dow_constant,
        "dropped_numeric": dropped_numeric,
        "dropped_categorical": dropped_categorical,
        "derived_features": derived_features,
    }



def feature_presence_report(df: pd.DataFrame, feats: List[str], *, kind: str) -> pd.DataFrame:
    rows = []
    for c in feats:
        present = c in df.columns
        dtype = str(df[c].dtype) if present else "MISSING"
        nn = int(df[c].notna().sum()) if present else 0
        nunique = int(df[c].nunique(dropna=True)) if present else 0
        rows.append({
            "feature": c,
            "kind": kind,
            "present": present,
            "dtype": dtype,
            "non_missing": nn,
            "n_unique": nunique,
        })
    return pd.DataFrame(rows)


def apply_ticker_scope_column(
    df: pd.DataFrame,
    *,
    ticker_col: str,
    is_foundation: np.ndarray,
    scope: str,
    out_col: str,
    base_label: str,
) -> Optional[str]:
    if scope == "none":
        return None
    if scope == "all" or not np.any(is_foundation):
        df[out_col] = df[ticker_col]
        return out_col
    df[out_col] = np.where(is_foundation, base_label, df[ticker_col])
    return out_col




# -----------------------------
# Rolling validation windows
# -----------------------------

@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_end_week: pd.Timestamp   # strictly before window_start
    window_start: pd.Timestamp
    window_end: pd.Timestamp       # inclusive
    n_weeks: int


def build_rolling_windows(
    uniq_weeks_sorted: np.ndarray,
    *,
    test_weeks: int,
    val_windows: int,
    val_window_weeks: int,
) -> Tuple[np.ndarray, np.ndarray, List[WindowSpec]]:
    """
    uniq_weeks_sorted: sorted unique weeks (pd.Timestamp-like)
    Returns:
      - pretest_weeks: all weeks strictly before test block
      - test_block_weeks: last test_weeks weeks
      - windows: list of validation windows taken from the tail of pretest
    """
    weeks = pd.to_datetime(pd.Series(uniq_weeks_sorted)).sort_values().to_numpy()
    n = len(weeks)
    if n < (test_weeks + val_windows * val_window_weeks + 10):
        raise ValueError(
            f"Not enough weeks for test+rolling val. "
            f"Have {n}, need at least {test_weeks + val_windows * val_window_weeks + 10}."
        )

    test_block = weeks[-test_weeks:]
    pretest = weeks[:-test_weeks]

    # Take the last (val_windows * val_window_weeks) weeks of pretest as the "val pool"
    need = val_windows * val_window_weeks
    val_pool = pretest[-need:]
    # Split into contiguous windows (oldest -> newest)
    windows: List[WindowSpec] = []
    for i in range(val_windows):
        w = val_pool[i * val_window_weeks:(i + 1) * val_window_weeks]
        ws = pd.Timestamp(w[0])
        we = pd.Timestamp(w[-1])
        windows.append(WindowSpec(
            name=f"roll_val_{i+1}",
            train_end_week=pd.Timestamp(ws),  # train uses weeks < window_start
            window_start=ws,
            window_end=we,
            n_weeks=len(w),
        ))

    return pretest, test_block, windows


def mask_weeks(df: pd.DataFrame, week_col: str, weeks: np.ndarray) -> np.ndarray:
    s = pd.to_datetime(df[week_col], errors="coerce")
    return s.isin(pd.to_datetime(pd.Series(weeks))).to_numpy()


def mask_weeks_before(df: pd.DataFrame, week_col: str, strictly_before: pd.Timestamp) -> np.ndarray:
    s = pd.to_datetime(df[week_col], errors="coerce")
    return (s < strictly_before).to_numpy()


def split_train_calib_weeks(
    train_weeks_sorted: np.ndarray,
    calib_frac_of_train: float,
    *,
    use_calib: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Given sorted weeks used for training, take last fraction as calib weeks (optional)."""
    w = pd.to_datetime(pd.Series(train_weeks_sorted)).sort_values().to_numpy()
    if (not use_calib) or (calib_frac_of_train <= 0):
        return w, w[:0]
    n = len(w)
    if n < 2:
        return w, w[:0]
    n_calib = max(1, int(round(n * calib_frac_of_train)))
    n_calib = min(n_calib, n - 1)  # ensure at least 1 week train_fit
    train_fit_weeks = w[:-n_calib]
    calib_weeks = w[-n_calib:]
    return train_fit_weeks, calib_weeks


# -----------------------------
# Reporting helpers
# -----------------------------

def _extract_pre_and_coefs(pipe: Pipeline) -> Tuple[ColumnTransformer, np.ndarray, float]:
    clf = pipe.named_steps["clf"]
    pre: ColumnTransformer = pipe.named_steps["pre"]
    coefs = np.asarray(clf.coef_).ravel()
    intercept = float(np.asarray(clf.intercept_).ravel()[0])
    return pre, coefs, intercept


def _simplify_feature_name(raw_name: str) -> str:
    """Drop sklearn-style prefixes so the equation references human-friendly feature names."""
    name = str(raw_name)
    for prefix in ("num__", "cat__", "ticker_x__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def print_numeric_coeffs_only(pre: ColumnTransformer, coefs: np.ndarray, intercept: float) -> None:
    """Print only the numeric part coefficients (on standardized numeric features)."""
    feature_names = list(pre.get_feature_names_out())

    # numeric features come out as: "num__<col>"
    rows = []
    for fname, c in zip(feature_names, coefs):
        if fname.startswith("num__"):
            rows.append((fname.replace("num__", ""), float(c)))

    dfc = pd.DataFrame(rows, columns=["feature", "coef"])
    dfc["abs"] = dfc["coef"].abs()
    dfc = dfc.sort_values("abs", ascending=False).drop(columns=["abs"])

    print("\n=== COEFFICIENTS (numeric only; standardized) ===")
    print(f"intercept: {intercept:.6f}")
    if len(dfc) == 0:
        print("(no numeric coefficients found?)")
    else:
        print(dfc.to_string(index=False))
    print("\n(note) Numeric coefs correspond to standardized (z-scored) features due to StandardScaler.")


def print_top_ticker_adjustments(pre: ColumnTransformer, coefs: np.ndarray, ticker_col: str, top_k: int = 10) -> None:
    """
    Approximate ticker intercept adjustments from one-hot coefficients.
    With drop='first', the reference ticker has 0 adjustment; others are relative.
    """
    feature_names = list(pre.get_feature_names_out())

    # cat onehots come out like: "cat__ticker_<LEVEL>" depending on sklearn version
    rows = []
    for fname, c in zip(feature_names, coefs):
        if fname.startswith("cat__"):
            rows.append((fname.replace("cat__", ""), float(c)))

    if not rows:
        print("\n=== TICKER ADJUSTMENTS ===")
        print("(no categorical coefficients found?)")
        return

    dft = pd.DataFrame(rows, columns=["onehot", "coef"])
    # Keep only those that mention ticker_col
    dft = dft[dft["onehot"].str.contains(str(ticker_col))]
    if dft.empty:
        print("\n=== TICKER ADJUSTMENTS ===")
        print("(could not isolate ticker onehots; sklearn naming changed?)")
        return

    dft = dft.sort_values("coef", ascending=False)
    print("\n=== TICKER INTERCEPT ADJUSTMENTS (relative to reference ticker; standardized space) ===")
    print("Top positive (model pushes prob UP vs reference ticker):")
    print(dft.head(top_k).to_string(index=False))
    print("\nTop negative (model pushes prob DOWN vs reference ticker):")
    print(dft.tail(top_k).sort_values("coef").to_string(index=False))


def print_top_interactions(pre: ColumnTransformer, coefs: np.ndarray, top_k: int = 10) -> None:
    feature_names = list(pre.get_feature_names_out())
    rows = []
    for fname, c in zip(feature_names, coefs):
        if fname.startswith("ticker_x__"):
            rows.append((fname, float(c)))
    if not rows:
        print("\n=== TICKER x x_logit_prn INTERACTIONS ===")
        print("(no interaction coefficients found?)")
        return

    dfi = pd.DataFrame(rows, columns=["interaction", "coef"]).sort_values("coef", ascending=False)
    print("\n=== TICKER x x_logit_prn INTERACTIONS (standardized space) ===")
    print("Top positive interactions:")
    print(dfi.head(top_k).to_string(index=False))
    print("\nTop negative interactions:")
    print(dfi.tail(top_k).sort_values("coef").to_string(index=False))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    print("RUNNING SCRIPT:", __file__)

    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--target-col", default=None)
    ap.add_argument("--week-col", default="week_friday")
    ap.add_argument("--ticker-col", default="ticker")
    ap.add_argument("--weight-col", default=None)
    ap.add_argument("--foundation-tickers", default="")
    ap.add_argument("--foundation-weight", type=float, default=1.0)
    ap.add_argument("--mode", choices=["baseline", "pooled", "two_stage"], default="pooled")
    ap.add_argument("--ticker-intercepts", choices=["none", "all", "non_foundation"], default="non_foundation")
    ap.add_argument("--ticker-x-interactions", action="store_true")
    ap.add_argument("--ticker-min-support", type=int, default=300)
    ap.add_argument("--ticker-min-support-interactions", type=int, default=1000)
    ap.add_argument("--train-tickers", default="")
    ap.add_argument(
        "--tdays-allowed",
        default="",
        help="Comma-separated list of allowed T_days values (ints).",
    )
    ap.add_argument(
        "--asof-dow-allowed",
        default="",
        help="Comma-separated list of allowed as-of weekdays (Mon..Sun or 0..6 where 0=Mon).",
    )

    ap.add_argument(
        "--features",
        default="x_logit_prn,log_m_fwd,abs_log_m_fwd,T_days,sqrt_T_years,rv20,rv20_sqrtT,log_m_fwd_over_volT,log_rel_spread,had_fallback,had_intrinsic_drop,had_band_clip,prn_raw_gap,dividend_yield",
    )
    ap.add_argument("--categorical-features", default="spot_scale_used")
    ap.add_argument("--add-interactions", action="store_true")
    ap.add_argument("--calibrate", choices=["none", "platt"], default="none")

    # New interaction features
    ap.add_argument("--enable-x-abs-m", action="store_true", help="Enable x_abs_m interaction feature (default: off)")

    # Group reweighting
    ap.add_argument("--group-reweight", choices=["none", "chain"], default="none", help="Apply group reweighting to TRAIN_FIT (default: none)")

    # Optional data filters (OFF by default)
    ap.add_argument("--max-abs-logm", type=float, default=None, help="Maximum absolute log-moneyness filter (default: None)")
    ap.add_argument("--drop-prn-extremes", action="store_true", help="Drop pRN extremes near 0 or 1 (default: off)")
    ap.add_argument("--prn-eps", type=float, default=1e-4, help="Epsilon for pRN extremes filter (default: 1e-4)")

    ap.add_argument("--C-grid", default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10")

    ap.add_argument("--train-decay-half-life-weeks", type=float, default=0.0)
    ap.add_argument("--calib-frac-of-train", type=float, default=0.20)
    ap.add_argument("--fit-weight-renorm", choices=["none", "mean1"], default="mean1")

    # Rolling validation settings (step 3)
    ap.add_argument("--test-weeks", type=int, default=20)
    ap.add_argument("--val-windows", type=int, default=4)
    ap.add_argument("--val-window-weeks", type=int, default=10)

    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--eceq-bins", type=int, default=10)
    ap.add_argument("--selection-objective", choices=["delta_vs_baseline"], default="delta_vs_baseline")
    ap.add_argument("--fallback-to-baseline-if-worse", action="store_true", default=True)
    ap.add_argument("--no-fallback-to-baseline-if-worse", dest="fallback_to_baseline_if_worse", action="store_false")
    ap.add_argument("--auto-drop-near-constant", action="store_true", default=True)
    ap.add_argument("--no-auto-drop-near-constant", dest="auto_drop_near_constant", action="store_false")
    ap.add_argument("--metrics-top-tickers", type=int, default=10)
    ap.add_argument("--random-state", type=int, default=7)

    # Bootstrap confidence intervals (off by default)
    ap.add_argument("--bootstrap-ci", action="store_true", default=False,
        help="Compute bootstrap confidence intervals for delta metrics on val_pool and test.")
    ap.add_argument("--bootstrap-B", type=int, default=2000,
        help="Number of bootstrap resamples (default: 2000).")
    ap.add_argument("--bootstrap-seed", type=int, default=0,
        help="Random seed for bootstrap resampling (default: 0).")
    ap.add_argument("--bootstrap-group", choices=["auto", "ticker_day", "day", "iid"], default="auto",
        help="Bootstrap grouping strategy (default: auto).")

    args = ap.parse_args()
    calibration_requested = args.calibrate
    if args.test_weeks <= 0:
        raise ValueError("--test-weeks must be >= 1.")
    if args.val_windows <= 0 or args.val_window_weeks <= 0:
        raise ValueError("--val-windows and --val-window-weeks must be >= 1.")
    if args.n_bins < 2:
        raise ValueError("--n-bins must be >= 2.")
    if args.eceq_bins < 2:
        raise ValueError("--eceq-bins must be >= 2.")
    if not (0.0 <= args.calib_frac_of_train < 1.0):
        raise ValueError("--calib-frac-of-train must be in [0,1).")
    if args.foundation_weight <= 0:
        raise ValueError("--foundation-weight must be > 0.")
    if args.metrics_top_tickers < 0:
        raise ValueError("--metrics-top-tickers must be >= 0.")
    if args.ticker_min_support < 1:
        raise ValueError("--ticker-min-support must be >= 1.")
    if args.ticker_min_support_interactions < 1:
        raise ValueError("--ticker-min-support-interactions must be >= 1.")
    if args.bootstrap_ci and args.bootstrap_B < 100:
        raise ValueError("--bootstrap-B must be >= 100.")

    use_calib = args.calibrate == "platt"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    foundation_list = parse_ticker_list(args.foundation_tickers)
    train_ticker_list = parse_ticker_list(args.train_tickers)

    if args.mode == "baseline":
        if args.ticker_intercepts != "none":
            print("[WARN] --mode baseline: forcing --ticker-intercepts none.")
            args.ticker_intercepts = "none"
        if args.ticker_x_interactions:
            print("[WARN] --mode baseline: disabling --ticker-x-interactions.")
            args.ticker_x_interactions = False

    if args.ticker_intercepts == "none" and args.ticker_x_interactions:
        print("[WARN] --ticker-x-interactions requires ticker intercepts; disabling interactions.")
        args.ticker_x_interactions = False

    # Parse dates if present
    for c in ["asof_date", "week_friday", "week_monday", "expiry_close_date_used"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    tdays_allowed = parse_tdays_allowed(args.tdays_allowed)
    asof_dow_allowed = parse_asof_dow_allowed(args.asof_dow_allowed)
    asof_date_col = resolve_asof_date_column(df)
    expiry_date_col = resolve_expiry_date_column(df)
    extra_cat = parse_categorical_list(args.categorical_features)
    asof_dow_requested = "asof_dow" in extra_cat

    if tdays_allowed or asof_dow_allowed:
        missing_filter_cols = []
        if "T_days" not in df.columns:
            missing_filter_cols.append("T_days")
        if not asof_date_col:
            missing_filter_cols.append("asof_date (or equivalent)")
        if not expiry_date_col:
            missing_filter_cols.append("expiry_date (or equivalent)")
        if missing_filter_cols:
            raise ValueError(
                f"Regime filtering requires columns: {missing_filter_cols}. "
                "Ensure T_days, asof_date, and expiry_date are present."
            )
    if (asof_dow_allowed or asof_dow_requested) and not asof_date_col:
        raise ValueError("asof_dow feature requires asof_date (or equivalent) column.")

    # Required columns
    missing_required = []
    for c in [args.week_col, args.ticker_col, "pRN"]:
        if c not in df.columns:
            missing_required.append(c)
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    target_col = pick_target_column(df, args.target_col)
    weight_col = pick_weight_column(df, args.weight_col)

    # Target cleanup
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No rows left after filtering valid target values {0,1}.")

    forward_fallback_rows = 0
    if "asof_fallback_days" in df.columns:
        df["asof_fallback_days"] = pd.to_numeric(df["asof_fallback_days"], errors="coerce")
        forward_mask = df["asof_fallback_days"] > 0
        forward_fallback_rows = int(forward_mask.sum())
        if forward_fallback_rows > 0:
            print(
                f"[WARN] Dropping {forward_fallback_rows} rows with forward fallback "
                "(asof_fallback_days > 0) to avoid look-ahead leakage."
            )
            df = df.loc[~forward_mask].copy()
            if df.empty:
                raise ValueError("No rows left after dropping forward-fallback rows.")
        if (df["asof_fallback_days"] > 0).any():
            raise ValueError("Forward fallback rows remain after filtering; aborting.")

    # Week sanity
    df[args.week_col] = pd.to_datetime(df[args.week_col], errors="coerce")
    if df[args.week_col].isna().any():
        bad = int(df[args.week_col].isna().sum())
        raise ValueError(f"{args.week_col} has {bad} NaT values; cannot proceed.")
    df = df[df[args.week_col].notna()].copy()

    # Regime filtering (time-to-expiry + as-of day-of-week)
    if tdays_allowed or asof_dow_allowed or asof_dow_requested:
        if "T_days" in df.columns:
            df["T_days"] = pd.to_numeric(df["T_days"], errors="coerce")
        if asof_date_col:
            df[asof_date_col] = pd.to_datetime(df[asof_date_col], errors="coerce", utc=True)
        if expiry_date_col:
            df[expiry_date_col] = pd.to_datetime(df[expiry_date_col], errors="coerce", utc=True)
        if asof_date_col:
            add_asof_dow_column(df, asof_date_col)

        before_rows = int(len(df))
        if tdays_allowed:
            df = df[df["T_days"].isin(tdays_allowed)].copy()
        if asof_dow_allowed:
            df = df[df["asof_dow"].isin(asof_dow_allowed)].copy()
        after_rows = int(len(df))

        print("\n=== REGIME FILTER SUMMARY ===")
        print(f"tdays_allowed: {tdays_allowed if tdays_allowed else 'ALL'}")
        print(f"asof_dow_allowed: {asof_dow_allowed if asof_dow_allowed else 'ALL'}")
        print(f"rows_before: {before_rows} rows_after: {after_rows}")
        if after_rows == 0:
            raise ValueError("No rows left after applying regime filters. Loosen filters or provide more data.")

        if "T_days" in df.columns:
            by_tdays = df["T_days"].value_counts(dropna=True).sort_index()
            print("\nRows by T_days:")
            print(by_tdays.to_string())
        if args.ticker_col in df.columns:
            by_ticker = df[args.ticker_col].value_counts(dropna=True)
            print("\nRows by ticker:")
            print(by_ticker.to_string())

        if after_rows < MIN_FILTER_ROWS:
            raise ValueError(
                f"Regime filtering left {after_rows} rows (< {MIN_FILTER_ROWS}). "
                "Loosen filters or provide more data."
            )

    # Ticker sanity
    df[args.ticker_col] = df[args.ticker_col].astype("string").fillna("UNKNOWN")
    n_tickers = int(df[args.ticker_col].nunique())
    print(f"\n=== TICKER SANITY ===\nunique_tickers: {n_tickers}")
    if n_tickers < 2:
        raise ValueError("Need at least 2 tickers for ticker-based model.")
    if n_tickers > 5000:
        print("[WARN] Very high ticker cardinality; one-hot may be too large. Consider restricting universe.")

    # Foundation / training tickers
    df["_ticker_norm"] = df[args.ticker_col].map(normalize_ticker)
    foundation_set = {normalize_ticker(t) for t in foundation_list}
    train_ticker_set = {normalize_ticker(t) for t in train_ticker_list}

    if foundation_set:
        found = sorted(set(df["_ticker_norm"].unique()) & foundation_set)
        missing_found = sorted(foundation_set - set(df["_ticker_norm"].unique()))
        print(f"\n=== FOUNDATION TICKERS ===\nrequested: {sorted(foundation_set)}")
        print(f"found: {found}")
        if missing_found:
            print(f"[WARN] Foundation tickers missing from data: {missing_found}")
        if args.mode == "two_stage" and not found:
            raise ValueError("Mode two_stage requires foundation tickers present in data.")
    elif args.mode == "two_stage":
        raise ValueError("Mode two_stage requires non-empty --foundation-tickers.")

    df["_is_foundation"] = df["_ticker_norm"].isin(foundation_set)
    if train_ticker_set:
        keep_set = set(train_ticker_set) | set(foundation_set)
        df["_train_ticker_keep"] = df["_ticker_norm"].isin(keep_set)
        extra_keep = sorted(set(foundation_set) - set(train_ticker_set))
        if extra_keep:
            print(f"[INFO] Keeping foundation tickers outside --train-tickers: {extra_keep}")
        missing_train = sorted(set(train_ticker_set) - set(df["_ticker_norm"].unique()))
        if missing_train:
            print(f"[WARN] Train tickers missing from data: {missing_train}")
    else:
        df["_train_ticker_keep"] = True

    # pRN required
    df["pRN"] = pd.to_numeric(df["pRN"], errors="coerce").clip(EPS, 1.0 - EPS)
    df = df[np.isfinite(df["pRN"])].copy()
    if df.empty:
        raise ValueError("No rows left after filtering finite pRN.")
    df["x_logit_prn"] = _logit(df["pRN"].to_numpy(dtype=float))

    # Optional data filters (OFF by default, transparent logging)
    print("\n=== OPTIONAL DATA FILTERS ===")
    rows_before_filters = len(df)

    # Filter pRN extremes
    if args.drop_prn_extremes:
        prn_before = len(df)
        df = df[(df["pRN"] >= args.prn_eps) & (df["pRN"] <= 1.0 - args.prn_eps)].copy()
        prn_dropped = prn_before - len(df)
        print(f"[FILTER] drop_prn_extremes: dropped {prn_dropped} rows with pRN < {args.prn_eps} or > {1.0 - args.prn_eps}")
        if df.empty:
            raise ValueError("No rows left after filtering pRN extremes.")
    else:
        print("[FILTER] drop_prn_extremes: OFF")

    # Filter max absolute moneyness
    if args.max_abs_logm is not None:
        # Need to ensure moneyness column exists first
        temp_df = ensure_engineered_features(df.copy(), ["log_m_fwd", "log_m"])
        m_col = resolve_moneyness_column(temp_df)
        if m_col:
            m_before = len(df)
            m_vals = pd.to_numeric(temp_df[m_col], errors="coerce")
            keep_mask = m_vals.abs() <= args.max_abs_logm
            df = df[keep_mask].copy()
            m_dropped = m_before - len(df)
            print(f"[FILTER] max_abs_logm: dropped {m_dropped} rows with |{m_col}| > {args.max_abs_logm}")
            if df.empty:
                raise ValueError("No rows left after filtering max absolute log-moneyness.")
        else:
            print(f"[WARN] max_abs_logm filter requested but no moneyness column found; skipping filter")
    else:
        print("[FILTER] max_abs_logm: OFF")

    total_filtered = rows_before_filters - len(df)
    print(f"[FILTER] Total rows dropped by optional filters: {total_filtered} (remaining: {len(df)})")
    print()

    # Track pRN training range after filters
    prn_train_min = float(df["pRN"].min()) if not df.empty else None
    prn_train_max = float(df["pRN"].max()) if not df.empty else None
    if prn_train_min is not None and prn_train_max is not None:
        print(f"[INFO] pRN training range: [{prn_train_min:.6f}, {prn_train_max:.6f}]")

    # Features
    requested = parse_feature_list(args.features)
    if "x_logit_prn" not in requested:
        print("[INFO] Forcing inclusion of x_logit_prn in features.")
        requested.insert(0, "x_logit_prn")
    tdays_constant_hint = False
    if "T_days" in df.columns:
        tdays_vals = pd.to_numeric(df["T_days"], errors="coerce").dropna().unique()
        tdays_constant_hint = len(tdays_vals) <= 1
    if args.add_interactions:
        if (
            not tdays_constant_hint
            and ("T_days" in df.columns or "T_days" in requested)
            and "x_prn_x_tdays" not in requested
        ):
            requested.append("x_prn_x_tdays")
        if ("rv20" in df.columns or "rv20" in requested) and "x_prn_x_rv20" not in requested:
            requested.append("x_prn_x_rv20")
        if ("log_m" in df.columns or "log_m" in requested) and "x_prn_x_logm" not in requested:
            requested.append("x_prn_x_logm")

    # Add x_m and x_abs_m interaction features if enabled
    moneyness_col = resolve_moneyness_column(df)
    if moneyness_col:
        print(f"[INFO] Resolved moneyness column: {moneyness_col}")
        # Always add moneyness column and x_m if moneyness exists
        if moneyness_col not in requested:
            requested.append(moneyness_col)
        if "x_m" not in requested:
            requested.append("x_m")
        # Add x_abs_m only if explicitly enabled
        if args.enable_x_abs_m and "x_abs_m" not in requested:
            requested.append("x_abs_m")
            print("[INFO] Adding x_abs_m interaction feature")
    else:
        print("[WARN] No moneyness column found; x_m and x_abs_m will not be created")

    requested, forbidden_removed = filter_forbidden_features(requested)
    if forbidden_removed:
        print(f"[INFO] Removing forbidden numeric features: {forbidden_removed}")
        if any("fallback" in feat.lower() for feat in forbidden_removed) and "had_fallback" not in requested:
            requested.append("had_fallback")
        if any("drop_intrinsic" in feat.lower() for feat in forbidden_removed) and "had_intrinsic_drop" not in requested:
            requested.append("had_intrinsic_drop")
        if any("band" in feat.lower() for feat in forbidden_removed) and "had_band_clip" not in requested:
            requested.append("had_band_clip")
        requested = dedupe_preserve_order(requested)

    requested, extra_cat, feature_gate_info = build_feature_spec(
        df,
        numeric_features=requested,
        categorical_features=extra_cat,
        asof_dow_col="asof_dow" if (asof_dow_allowed or asof_dow_requested) else None,
    )

    if feature_gate_info["dropped_numeric"]:
        print(f"[INFO] Dropping constant/redundant time features: {feature_gate_info['dropped_numeric']}")
    if feature_gate_info["dropped_categorical"]:
        print(f"[INFO] Dropping constant categorical features: {feature_gate_info['dropped_categorical']}")

    df = ensure_engineered_features(df, requested)

    # Validate feature availability against schema contract
    print("\n=== VALIDATING FEATURE AVAILABILITY ===")
    try:
        from calibrate_common import validate_feature_availability
        valid_feats, nan_prone_feats, unknown_feats = validate_feature_availability(
            requested,
            warn_on_nan_prone=True
        )
        if nan_prone_feats:
            print(f"[INFO] Model includes {len(nan_prone_feats)} features that depend on NaN-prone columns.")
            print(f"       These features will likely be NaN in snapshot-only inference:")
            for f in nan_prone_feats:
                print(f"       - {f}")
            print(f"\n[RECOMMENDATION] Consider training a snapshot-only model variant without these features")
            print(f"                 for production inference on live Polymarket snapshots.\n")
    except Exception as e:
        print(f"[WARN] Could not validate features against schema contract: {e}")
        print(f"       Continuing with training, but inference may fail if features are unavailable.")

    # Ticker scope columns
    ticker_intercept_col = apply_ticker_scope_column(
        df,
        ticker_col=args.ticker_col,
        is_foundation=df["_is_foundation"].to_numpy(),
        scope=args.ticker_intercepts,
        out_col="_ticker_intercept",
        base_label=FOUNDATION_LABEL,
    )

    interaction_ticker_col = None
    if args.ticker_x_interactions:
        interaction_ticker_col = apply_ticker_scope_column(
            df,
            ticker_col=args.ticker_col,
            is_foundation=df["_is_foundation"].to_numpy(),
            scope=args.ticker_intercepts,
            out_col="_ticker_interaction",
            base_label=FOUNDATION_LABEL,
        )

    # Categorical features (ticker intercepts + optional extras)
    cat_features = dedupe_preserve_order(([ticker_intercept_col] if ticker_intercept_col else []) + extra_cat)
    stage1_cat_features = dedupe_preserve_order(extra_cat)

    # Ensure no overlap between numeric and categorical
    overlap = sorted(set(requested) & set(cat_features))
    if overlap:
        raise ValueError(f"Features overlap numeric & categorical lists: {overlap}")

    pres_num = feature_presence_report(df, requested, kind="numeric")
    pres_cat = feature_presence_report(df, cat_features, kind="categorical")

    print("\n=== FEATURE PRESENCE (numeric) ===")
    print(pres_num.to_string(index=False))
    missing = pres_num.loc[~pres_num["present"], "feature"].tolist()
    if missing:
        raise ValueError(f"Requested features missing from CSV: {missing}")
    all_missing = pres_num.loc[pres_num["non_missing"] == 0, "feature"].tolist()
    if all_missing:
        print(f"[WARN] Dropping all-missing features (entire dataset): {all_missing}")
        requested = [c for c in requested if c not in all_missing]
    if not requested:
        raise ValueError("No numeric features left after dropping all-missing features.")

    print("\n=== FEATURE PRESENCE (categorical) ===")
    print(pres_cat.to_string(index=False))
    missing_cat = pres_cat.loc[~pres_cat["present"], "feature"].tolist()
    if missing_cat:
        raise ValueError(f"Requested categorical features missing from CSV: {missing_cat}")

    # Ensure categorical columns are string and non-missing
    for c in cat_features:
        df[c] = df[c].astype("string").fillna("UNKNOWN")
    if interaction_ticker_col and interaction_ticker_col not in cat_features:
        df[interaction_ticker_col] = df[interaction_ticker_col].astype("string").fillna("UNKNOWN")

    # Weights
    if weight_col is None:
        df["_w"] = 1.0
        weight_col = "_w"
    else:
        df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
        if (df[weight_col] < 0).any():
            raise ValueError("Negative weights found.")

    # Clean numeric feature matrix
    for c in requested:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Weeks sorted
    uniq_weeks = np.array(sorted(pd.unique(df[args.week_col])))
    pretest_weeks, test_block_weeks, windows = build_rolling_windows(
        uniq_weeks,
        test_weeks=args.test_weeks,
        val_windows=args.val_windows,
        val_window_weeks=args.val_window_weeks,
    )

    is_foundation = df["_is_foundation"].to_numpy()
    train_ticker_keep = df["_train_ticker_keep"].to_numpy()

    m_pretest = mask_weeks(df, args.week_col, pretest_weeks)
    m_pretest_train = m_pretest & train_ticker_keep
    train_weeks_for_drop = np.array(sorted(pd.unique(df.loc[m_pretest_train, args.week_col])))
    train_fit_weeks_for_drop, _ = split_train_calib_weeks(
        train_weeks_for_drop,
        args.calib_frac_of_train,
        use_calib=use_calib,
    )
    m_trainfit_for_drop = m_pretest_train & mask_weeks(df, args.week_col, train_fit_weeks_for_drop)

    near_constant_removed: List[str] = []
    near_constant_variances: Dict[str, float] = {}
    if args.auto_drop_near_constant:
        near_constant_removed, near_constant_variances = find_near_constant_features(
            df,
            requested,
            train_fit_mask=m_trainfit_for_drop,
        )
        if near_constant_removed:
            print(f"[INFO] Dropping near-constant features (TRAIN_FIT var<1e-12): {near_constant_removed}")
            requested = [c for c in requested if c not in near_constant_removed]

    feature_gate_info["forbidden_removed"] = forbidden_removed
    feature_gate_info["near_constant_removed"] = near_constant_removed

    ticker_support_info = None
    interaction_support_info = None
    if ticker_intercept_col:
        ticker_intercept_col, ticker_support_info = scope_tickers_by_support(
            df,
            ticker_col=ticker_intercept_col,
            train_fit_mask=m_trainfit_for_drop,
            min_support=args.ticker_min_support,
            out_col="_ticker_intercept_scoped",
            preserve=[FOUNDATION_LABEL] if args.ticker_intercepts == "non_foundation" else None,
        )

    if args.ticker_x_interactions and interaction_ticker_col:
        interaction_ticker_col, interaction_support_info = scope_tickers_by_support(
            df,
            ticker_col=interaction_ticker_col,
            train_fit_mask=m_trainfit_for_drop,
            min_support=args.ticker_min_support_interactions,
            out_col="_ticker_interaction_scoped",
            preserve=[FOUNDATION_LABEL] if args.ticker_intercepts == "non_foundation" else None,
        )

    cat_features = dedupe_preserve_order(([ticker_intercept_col] if ticker_intercept_col else []) + extra_cat)
    stage1_cat_features = dedupe_preserve_order(extra_cat)

    for c in cat_features:
        df[c] = df[c].astype("string").fillna("UNKNOWN")
    if interaction_ticker_col and interaction_ticker_col not in cat_features:
        df[interaction_ticker_col] = df[interaction_ticker_col].astype("string").fillna("UNKNOWN")

    interaction_scope_tickers: Optional[List[str]] = None
    if args.ticker_x_interactions and interaction_ticker_col:
        support_tickers = tickers_meeting_support(
            df,
            ticker_col=interaction_ticker_col,
            train_fit_mask=m_trainfit_for_drop,
            min_support=args.ticker_min_support_interactions,
            preserve=[FOUNDATION_LABEL] if args.ticker_intercepts == "non_foundation" else None,
        )
        support_tickers = [t for t in support_tickers if t not in ("OTHER", "UNKNOWN")]
        if args.ticker_intercepts == "non_foundation" and foundation_set:
            support_tickers = [t for t in support_tickers if t != FOUNDATION_LABEL]
        unique_train = int(df.loc[m_trainfit_for_drop, interaction_ticker_col].nunique())
        min_required = max(2, int(0.2 * unique_train)) if unique_train > 0 else 0
        if len(support_tickers) < min_required:
            print(
                "[WARN] Ticker interactions disabled: too few tickers meet support threshold "
                f"({len(support_tickers)} < {min_required})."
            )
            args.ticker_x_interactions = False
            interaction_ticker_col = None
            support_tickers = []
        interaction_scope_tickers = support_tickers or None

    ticker_support_tickers: Optional[List[str]] = None
    if ticker_intercept_col:
        ticker_support_tickers = tickers_meeting_support(
            df,
            ticker_col=ticker_intercept_col,
            train_fit_mask=m_trainfit_for_drop,
            min_support=args.ticker_min_support,
            preserve=[FOUNDATION_LABEL] if args.ticker_intercepts == "non_foundation" else None,
        )

    interaction_extra_cols: List[str] = []
    if args.ticker_x_interactions and interaction_ticker_col:
        interaction_extra_cols = dedupe_preserve_order([interaction_ticker_col, "x_logit_prn"])

    model_cols_main = dedupe_preserve_order(requested + cat_features + interaction_extra_cols)
    stage1_cols = dedupe_preserve_order(requested + stage1_cat_features)
    stage2_cols = dedupe_preserve_order(requested + cat_features + interaction_extra_cols)

    # Build chain-snapshot group ID for reweighting
    print("\n=== GROUP REWEIGHTING ===")
    enable_group_reweight = args.group_reweight == "chain"
    chain_group_id = None
    group_reweight_info = {"mode": args.group_reweight}

    if enable_group_reweight:
        chain_group_id = build_chain_group_id(df)
        if chain_group_id is not None:
            n_groups = chain_group_id.nunique()
            group_counts = chain_group_id.value_counts()
            median_size = int(group_counts.median())
            print(f"[GROUP-REWEIGHT] Chain reweighting enabled")
            print(f"                 Total unique chain snapshots: {n_groups}")
            print(f"                 Median group size: {median_size} rows")
            group_reweight_info.update({
                "enabled": True,
                "n_groups": n_groups,
                "median_group_size": median_size,
            })
        else:
            print("[WARN] Chain reweighting requested but cannot build group ID (missing columns)")
            print("       Falling back to no reweighting")
            enable_group_reweight = False
            group_reweight_info["enabled"] = False
            group_reweight_info["fallback_reason"] = "missing_columns"
    else:
        print(f"[GROUP-REWEIGHT] Disabled (mode={args.group_reweight})")
        group_reweight_info["enabled"] = False

    print("\n=== ROLLING WINDOW SPECS ===")
    for w in windows:
        print(f"{w.name}: window=[{w.window_start.date()}..{w.window_end.date()}] ({w.n_weeks} wks), train uses weeks < {w.window_start.date()}")
    print(f"FINAL TEST block: [{pd.Timestamp(test_block_weeks[0]).date()}..{pd.Timestamp(test_block_weeks[-1]).date()}] ({len(test_block_weeks)} wks)")

    # Helper to subset rows
    def subset(mask: np.ndarray, cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        Xsub = df.loc[mask, cols].copy()
        ysub = df.loc[mask, target_col].astype(int).to_numpy()
        wsub = df.loc[mask, weight_col].astype(float).to_numpy()
        weeksub = df.loc[mask, args.week_col].to_numpy()
        return Xsub, ysub, wsub, weeksub

    # Baseline (pRN) predictions helper
    def baseline_pred(mask: np.ndarray) -> np.ndarray:
        return df.loc[mask, "pRN"].to_numpy(dtype=float)

    # C grid
    C_grid = [float(x) for x in args.C_grid.split(",") if x.strip()]
    if not C_grid:
        raise ValueError("Empty --C-grid.")

    def predict_two_stage(
        *,
        mask: np.ndarray,
        stage1_pipe: Pipeline,
        stage1_cols_use: List[str],
        stage2_pre: ColumnTransformer,
        stage2_model: LogisticRegressionOffset,
        stage2_cols_use: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X1 = df.loc[mask, stage1_cols_use].copy()
        p_base = stage1_pipe.predict_proba(X1)[:, 1]
        logits = _logit(p_base)

        is_fnd_local = is_foundation[mask]
        if np.any(~is_fnd_local):
            X2 = df.loc[mask, stage2_cols_use].copy()
            X2_nf = X2.loc[~is_fnd_local]
            z2 = stage2_pre.transform(X2_nf)
            g = stage2_model.decision_function(z2)
            logits[~is_fnd_local] = logits[~is_fnd_local] + g
        p = expit(logits)
        return logits, p

    # Rolling evaluation for each C
    roll_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    best_C = None
    best_avg_ll = float("inf")
    best_avg_delta = float("inf")

    for C in C_grid:
        lls_model: List[float] = []
        lls_base: List[float] = []

        for wspec in windows:
            # Train mask: weeks strictly before window_start (and also strictly before test block)
            m_train = mask_weeks_before(df, args.week_col, wspec.window_start)
            # Evaluate mask: weeks in this window
            m_val = (df[args.week_col] >= wspec.window_start) & (df[args.week_col] <= wspec.window_end)
            m_val = m_val.to_numpy()

            # Ensure train is only pretest
            m_pretest = mask_weeks(df, args.week_col, pretest_weeks)
            m_train = m_train & m_pretest & train_ticker_keep
            m_val = m_val & m_pretest

            # If no data, skip
            if m_train.sum() < 100 or m_val.sum() < 50:
                continue

            # Train weeks list
            train_weeks_used = np.array(sorted(pd.unique(df.loc[m_train, args.week_col])))

            # Split train_fit/calib weeks (calib only used if platt)
            use_platt_window = use_calib
            train_fit_weeks, calib_weeks = split_train_calib_weeks(
                train_weeks_used,
                args.calib_frac_of_train,
                use_calib=use_platt_window,
            )

            m_trainfit = m_train & mask_weeks(df, args.week_col, train_fit_weeks)
            m_calib = m_train & mask_weeks(df, args.week_col, calib_weeks)

            if use_platt_window and m_calib.sum() < MIN_CALIB_ROWS:
                # Fall back to full training set if calib slice is too small.
                use_platt_window = False
                m_trainfit = m_train
                m_calib = np.zeros_like(m_train, dtype=bool)

            if args.mode == "two_stage":
                m_trainfit_fnd = m_trainfit & is_foundation
                m_trainfit_non = m_trainfit & (~is_foundation)
                if m_trainfit_fnd.sum() < 50 or m_trainfit_non.sum() < 50:
                    continue

                X_fnd, y_fnd, w_fnd_raw, wk_fnd = subset(m_trainfit_fnd, stage1_cols)
                X_non_stage1, y_non, w_non_raw, wk_non = subset(m_trainfit_non, stage1_cols)
                X_non_stage2, _, _, _ = subset(m_trainfit_non, stage2_cols)

                w_fit_fnd = build_fit_weights(
                    w_fnd_raw,
                    wk_fnd,
                    foundation_mask=np.ones_like(w_fnd_raw, dtype=bool),
                    foundation_weight=args.foundation_weight,
                    decay_half_life_weeks=args.train_decay_half_life_weeks,
                    fit_weight_renorm=args.fit_weight_renorm,
                    group_id=chain_group_id[m_trainfit_fnd] if chain_group_id is not None else None,
                    enable_group_reweight=enable_group_reweight,
                )
                stage1_pipe = make_pipeline(
                    numeric_features=requested,
                    categorical_features=stage1_cat_features,
                    interaction_ticker_col=None,
                    interaction_x_col="x_logit_prn",
                    interaction_scope_tickers=None,
                    enable_interactions=False,
                    C=C,
                    random_state=args.random_state,
                )
                stage1_pipe.fit(X_fnd, y_fnd, clf__sample_weight=w_fit_fnd)

                p_base_non = stage1_pipe.predict_proba(X_non_stage1)[:, 1]
                logit_base_non = _logit(p_base_non)

                w_fit_non = build_fit_weights(
                    w_non_raw,
                    wk_non,
                    foundation_mask=None,
                    foundation_weight=args.foundation_weight,
                    decay_half_life_weeks=args.train_decay_half_life_weeks,
                    fit_weight_renorm=args.fit_weight_renorm,
                    group_id=chain_group_id[m_trainfit_non] if chain_group_id is not None else None,
                    enable_group_reweight=enable_group_reweight,
                )
                stage2_pre = make_preprocessor(
                    numeric_features=requested,
                    categorical_features=cat_features,
                    interaction_ticker_col=interaction_ticker_col,
                    interaction_x_col="x_logit_prn",
                    interaction_scope_tickers=interaction_scope_tickers,
                    enable_interactions=args.ticker_x_interactions,
                )
                z_train = stage2_pre.fit_transform(X_non_stage2)
                stage2_model = LogisticRegressionOffset(C=C, max_iter=2000, tol=1e-6)
                stage2_model.fit(z_train, y_non, sample_weight=w_fit_non, offset=logit_base_non)

                _, y_val, w_val, _ = subset(m_val, stage1_cols)
                logits_val, p_val_model = predict_two_stage(
                    mask=m_val,
                    stage1_pipe=stage1_pipe,
                    stage1_cols_use=stage1_cols,
                    stage2_pre=stage2_pre,
                    stage2_model=stage2_model,
                    stage2_cols_use=stage2_cols,
                )
                p_val_base = baseline_pred(m_val)

                if use_platt_window:
                    _, y_calib, w_calib_raw, _ = subset(m_calib, stage1_cols)
                    logits_cal, _ = predict_two_stage(
                        mask=m_calib,
                        stage1_pipe=stage1_pipe,
                        stage1_cols_use=stage1_cols,
                        stage2_pre=stage2_pre,
                        stage2_model=stage2_model,
                        stage2_cols_use=stage2_cols,
                    )
                    w_cal_fit = w_calib_raw.copy()
                    if args.fit_weight_renorm == "mean1":
                        w_cal_fit = renorm_fit_weights_mean1(w_cal_fit)
                    cal = fit_platt_on_logits(logits_cal, y_calib, w_cal_fit, random_state=args.random_state)
                    p_val_model = apply_platt(cal, logits_val)
            else:
                X_trainfit, y_trainfit, w_trainfit_raw, wk_trainfit = subset(m_trainfit, model_cols_main)
                X_calib, y_calib, w_calib_raw, wk_calib = subset(m_calib, model_cols_main)
                X_val, y_val, w_val, wk_val = subset(m_val, model_cols_main)

                w_fit = build_fit_weights(
                    w_trainfit_raw,
                    wk_trainfit,
                    foundation_mask=is_foundation[m_trainfit],
                    foundation_weight=args.foundation_weight,
                    decay_half_life_weeks=args.train_decay_half_life_weeks,
                    fit_weight_renorm=args.fit_weight_renorm,
                    group_id=chain_group_id[m_trainfit] if chain_group_id is not None else None,
                    enable_group_reweight=enable_group_reweight,
                )

                pipe = make_pipeline(
                    numeric_features=requested,
                    categorical_features=cat_features,
                    interaction_ticker_col=interaction_ticker_col,
                    interaction_x_col="x_logit_prn",
                    interaction_scope_tickers=interaction_scope_tickers,
                    enable_interactions=args.ticker_x_interactions,
                    C=C,
                    random_state=args.random_state,
                )
                pipe.fit(X_trainfit, y_trainfit, clf__sample_weight=w_fit)

                p_val_model = pipe.predict_proba(X_val)[:, 1]
                p_val_base = baseline_pred(m_val)

                if use_platt_window:
                    z_cal = pipe.decision_function(X_calib)
                    w_cal_fit = w_calib_raw.copy()
                    if args.fit_weight_renorm == "mean1":
                        w_cal_fit = renorm_fit_weights_mean1(w_cal_fit)
                    cal = fit_platt_on_logits(z_cal, y_calib, w_cal_fit, random_state=args.random_state)
                    z_val = pipe.decision_function(X_val)
                    p_val_model = apply_platt(cal, z_val)

            # Window metrics
            ll_model = float(log_loss(y_val, np.clip(p_val_model, EPS, 1.0 - EPS), labels=[0, 1], sample_weight=w_val))
            ll_base = float(log_loss(y_val, np.clip(p_val_base, EPS, 1.0 - EPS), labels=[0, 1], sample_weight=w_val))

            lls_model.append(ll_model)
            lls_base.append(ll_base)

            roll_rows.append({
                "C": C,
                "window": wspec.name,
                "window_start": str(wspec.window_start.date()),
                "window_end": str(wspec.window_end.date()),
                "n_val": int(len(y_val)),
                "w_sum_val": float(np.sum(w_val)),
                "logloss_baseline": ll_base,
                "logloss_model": ll_model,
                "delta_model_minus_baseline": ll_model - ll_base,
            })

        if lls_model:
            avg_ll = float(np.mean(lls_model))
            avg_ll_base = float(np.mean(lls_base)) if lls_base else float("nan")
            avg_delta = avg_ll - avg_ll_base
            summary_rows.append({
                "C": C,
                "avg_roll_logloss_model": avg_ll,
                "avg_roll_logloss_baseline": avg_ll_base,
                "avg_roll_delta": avg_delta,
                "n_windows_used": int(len(lls_model)),
            })
            if avg_delta < best_avg_delta:
                best_avg_delta = avg_delta
                best_avg_ll = avg_ll
                best_C = C

    if best_C is None:
        raise ValueError("No rolling windows produced usable evaluations (train/val too small).")

    rolling_df = pd.DataFrame(roll_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("avg_roll_delta").reset_index(drop=True)

    print("\n=== ROLLING SELECTION SUMMARY (best by avg delta vs baseline) ===")
    print(summary_df.to_string(index=False))
    print(
        f"\nSelected C by rolling avg delta: {best_C:g}  "
        f"(avg_roll_delta={best_avg_delta:.6f}, avg_roll_logloss_model={best_avg_ll:.6f})"
    )

    fallback_to_baseline = False
    if args.fallback_to_baseline_if_worse and best_avg_delta > 0:
        fallback_to_baseline = True
        print(
            "[WARN] Best rolling avg delta is worse than baseline. "
            "Will deploy baseline-only bundle."
        )

    # -----------------------------
    # Final fit: train on ALL pretest weeks (strictly before test block)
    # -----------------------------
    # Masks for final fit / eval
    m_pretest = mask_weeks(df, args.week_col, pretest_weeks)  # all weeks before TEST
    m_test = mask_weeks(df, args.week_col, test_block_weeks)
    m_pretest_train = m_pretest & train_ticker_keep

    # Train weeks used for final fit: all pretest weeks (train tickers only)
    train_weeks_used = np.array(sorted(pd.unique(df.loc[m_pretest_train, args.week_col])))
    train_fit_weeks, calib_weeks = split_train_calib_weeks(
        train_weeks_used,
        args.calib_frac_of_train,
        use_calib=use_calib,
    )

    # Val pool: the same last block of pretest weeks used by rolling windows
    val_pool_weeks = pretest_weeks[-(args.val_windows * args.val_window_weeks):]
    m_val_pool = mask_weeks(df, args.week_col, val_pool_weeks)

    m_trainfit = m_pretest_train & mask_weeks(df, args.week_col, train_fit_weeks)
    m_calib = m_pretest_train & mask_weeks(df, args.week_col, calib_weeks)

    if use_calib and m_calib.sum() < MIN_CALIB_ROWS:
        print(f"[WARN] CALIB rows={int(m_calib.sum())} < {MIN_CALIB_ROWS}; refitting without calibration.")
        use_calib = False
        args.calibrate = "none"
        train_fit_weeks = train_weeks_used
        calib_weeks = train_weeks_used[:0]
        m_trainfit = m_pretest_train
        m_calib = np.zeros_like(m_pretest_train, dtype=bool)

    # -----------------------------
    # Sanity checks: non-missing counts by split, and drop all-missing-in-trainfit features
    # -----------------------------
    def _nonmissing_counts(mask: np.ndarray, feats: List[str]) -> Dict[str, int]:
        return {c: int(pd.to_numeric(df.loc[mask, c], errors="coerce").notna().sum()) for c in feats}

    nm_trainfit = _nonmissing_counts(m_trainfit, requested)
    nm_calib = _nonmissing_counts(m_calib, requested)
    nm_val = _nonmissing_counts(m_val_pool, requested)
    nm_test = _nonmissing_counts(m_test, requested)

    print("\n=== SPLIT NON-MISSING COUNTS (before dropping) ===")
    tmp_rows = []
    for c in requested:
        tmp_rows.append({
            "feature": c,
            "trainfit_nonmissing": nm_trainfit.get(c, 0),
            "calib_nonmissing": nm_calib.get(c, 0),
            "valpool_nonmissing": nm_val.get(c, 0),
            "test_nonmissing": nm_test.get(c, 0),
        })
    print(pd.DataFrame(tmp_rows).to_string(index=False))

    drop_cols = [c for c, nn in nm_trainfit.items() if nn == 0]
    if drop_cols:
        print(f"\n[WARN] Dropping all-missing-in-TRAIN_FIT features: {drop_cols}")
        requested = [c for c in requested if c not in drop_cols]

        # Rebuild column lists with updated features
        interaction_extra_cols = []
        if args.ticker_x_interactions and interaction_ticker_col:
            interaction_extra_cols = dedupe_preserve_order([interaction_ticker_col, "x_logit_prn"])
        model_cols_main = dedupe_preserve_order(requested + cat_features + interaction_extra_cols)
        stage1_cols = dedupe_preserve_order(requested + stage1_cat_features)
        stage2_cols = dedupe_preserve_order(requested + cat_features + interaction_extra_cols)

    requested, forbidden_late = filter_forbidden_features(requested)
    if forbidden_late:
        print(f"\n[WARN] Dropping forbidden features before fit: {forbidden_late}")
        feature_gate_info["forbidden_removed"] = sorted(
            set(feature_gate_info.get("forbidden_removed", [])) | set(forbidden_late)
        )
        interaction_extra_cols = []
        if args.ticker_x_interactions and interaction_ticker_col:
            interaction_extra_cols = dedupe_preserve_order([interaction_ticker_col, "x_logit_prn"])
        model_cols_main = dedupe_preserve_order(requested + cat_features + interaction_extra_cols)
        stage1_cols = dedupe_preserve_order(requested + stage1_cat_features)
        stage2_cols = dedupe_preserve_order(requested + cat_features + interaction_extra_cols)

    print("\n=== FINAL FEATURES USED FOR FIT ===")
    print(requested)
    print("\n=== FINAL CATEGORICAL FEATURES USED FOR FIT ===")
    print(cat_features)

    remaining_forbidden = filter_forbidden_features(requested)[1]
    if remaining_forbidden:
        raise ValueError(f"Forbidden numeric features remain in final feature set: {remaining_forbidden}")

    # -----------------------------
    # Fit final model
    # -----------------------------
    platt_cal = None
    pipe_final: Optional[Pipeline] = None
    stage1_final: Optional[Pipeline] = None
    stage2_pre_final: Optional[ColumnTransformer] = None
    stage2_model_final: Optional[LogisticRegressionOffset] = None

    if args.mode == "two_stage":
        m_trainfit_fnd = m_trainfit & is_foundation
        m_trainfit_non = m_trainfit & (~is_foundation)
        if m_trainfit_fnd.sum() < 50 or m_trainfit_non.sum() < 50:
            raise ValueError("Not enough TRAIN_FIT rows for two-stage model (foundation or non-foundation too small).")

        X_fnd, y_fnd, w_fnd_raw, wk_fnd = subset(m_trainfit_fnd, stage1_cols)
        X_non_stage1, y_non, w_non_raw, wk_non = subset(m_trainfit_non, stage1_cols)
        X_non_stage2, _, _, _ = subset(m_trainfit_non, stage2_cols)

        w_fit_fnd = build_fit_weights(
            w_fnd_raw,
            wk_fnd,
            foundation_mask=np.ones_like(w_fnd_raw, dtype=bool),
            foundation_weight=args.foundation_weight,
            decay_half_life_weeks=args.train_decay_half_life_weeks,
            fit_weight_renorm=args.fit_weight_renorm,
            group_id=chain_group_id[m_trainfit_fnd] if chain_group_id is not None else None,
            enable_group_reweight=enable_group_reweight,
        )

        stage1_final = make_pipeline(
            numeric_features=requested,
            categorical_features=stage1_cat_features,
            interaction_ticker_col=None,
            interaction_x_col="x_logit_prn",
            interaction_scope_tickers=None,
            enable_interactions=False,
            C=best_C,
            random_state=args.random_state,
        )
        stage1_final.fit(X_fnd, y_fnd, clf__sample_weight=w_fit_fnd)

        p_base_non = stage1_final.predict_proba(X_non_stage1)[:, 1]
        logit_base_non = _logit(p_base_non)

        w_fit_non = build_fit_weights(
            w_non_raw,
            wk_non,
            foundation_mask=None,
            foundation_weight=args.foundation_weight,
            decay_half_life_weeks=args.train_decay_half_life_weeks,
            fit_weight_renorm=args.fit_weight_renorm,
            group_id=chain_group_id[m_trainfit_non] if chain_group_id is not None else None,
            enable_group_reweight=enable_group_reweight,
        )
        stage2_pre_final = make_preprocessor(
            numeric_features=requested,
            categorical_features=cat_features,
            interaction_ticker_col=interaction_ticker_col,
            interaction_x_col="x_logit_prn",
            interaction_scope_tickers=interaction_scope_tickers,
            enable_interactions=args.ticker_x_interactions,
        )
        z_train = stage2_pre_final.fit_transform(X_non_stage2)
        stage2_model_final = LogisticRegressionOffset(C=best_C, max_iter=2000, tol=1e-6)
        stage2_model_final.fit(z_train, y_non, sample_weight=w_fit_non, offset=logit_base_non)

        # Optional Platt calibrator (fit ONLY on CALIB)
        if use_calib:
            _, y_calib, w_calib_raw, _ = subset(m_calib, stage1_cols)
            logits_cal, _ = predict_two_stage(
                mask=m_calib,
                stage1_pipe=stage1_final,
                stage1_cols_use=stage1_cols,
                stage2_pre=stage2_pre_final,
                stage2_model=stage2_model_final,
                stage2_cols_use=stage2_cols,
            )
            w_cal_fit = w_calib_raw.copy()
            if args.fit_weight_renorm == "mean1":
                w_cal_fit = renorm_fit_weights_mean1(w_cal_fit)
            platt_cal = fit_platt_on_logits(logits_cal, y_calib, w_cal_fit, random_state=args.random_state)

        print("\n=== FIT WEIGHT DEBUG (two_stage) ===")
        print(f"STAGE1 TRAIN_FIT: n={len(w_fnd_raw)} sum_raw={float(np.sum(w_fnd_raw)):.6f} sum_fit={float(np.sum(w_fit_fnd)):.6f}")
        print(f"STAGE2 TRAIN_FIT: n={len(w_non_raw)} sum_raw={float(np.sum(w_non_raw)):.6f} sum_fit={float(np.sum(w_fit_non)):.6f}")
        print(f"CALIB            : n={int(m_calib.sum())} sum_raw={float(np.sum(df.loc[m_calib, weight_col])):.6f}")
        if args.train_decay_half_life_weeks and args.train_decay_half_life_weeks > 0:
            print(f"TRAIN_FIT recency decay half-life weeks: {args.train_decay_half_life_weeks:g}")
        fit_weight_meta = {
            "fit_weight_renorm": args.fit_weight_renorm,
            "train_decay_half_life_weeks": float(args.train_decay_half_life_weeks),
            "foundation_weight": float(args.foundation_weight),
            "stage1_trainfit_n": int(len(w_fnd_raw)),
            "stage1_trainfit_sum_raw": float(np.sum(w_fnd_raw)),
            "stage1_trainfit_sum_fit": float(np.sum(w_fit_fnd)),
            "stage2_trainfit_n": int(len(w_non_raw)),
            "stage2_trainfit_sum_raw": float(np.sum(w_non_raw)),
            "stage2_trainfit_sum_fit": float(np.sum(w_fit_non)),
            "calib_n": int(m_calib.sum()),
            "calib_sum_raw": float(np.sum(df.loc[m_calib, weight_col])),
        }
    else:
        X_trainfit, y_trainfit, w_trainfit_raw, wk_trainfit = subset(m_trainfit, model_cols_main)
        X_calib, y_calib, w_calib_raw, wk_calib = subset(m_calib, model_cols_main)

        w_fit = build_fit_weights(
            w_trainfit_raw,
            wk_trainfit,
            foundation_mask=is_foundation[m_trainfit],
            foundation_weight=args.foundation_weight,
            decay_half_life_weeks=args.train_decay_half_life_weeks,
            fit_weight_renorm=args.fit_weight_renorm,
            group_id=chain_group_id[m_trainfit] if chain_group_id is not None else None,
            enable_group_reweight=enable_group_reweight,
        )

        print("\n=== FIT WEIGHT DEBUG ===")
        print(f"TRAIN_FIT: n={len(w_trainfit_raw)} sum_raw={float(np.sum(w_trainfit_raw)):.6f} sum_fit={float(np.sum(w_fit)):.6f}")
        print(f"CALIB    : n={len(w_calib_raw)} sum_raw={float(np.sum(w_calib_raw)):.6f}")
        if args.train_decay_half_life_weeks and args.train_decay_half_life_weeks > 0:
            print(f"TRAIN_FIT recency decay half-life weeks: {args.train_decay_half_life_weeks:g}")
        fit_weight_meta = {
            "fit_weight_renorm": args.fit_weight_renorm,
            "train_decay_half_life_weeks": float(args.train_decay_half_life_weeks),
            "foundation_weight": float(args.foundation_weight),
            "trainfit_n": int(len(w_trainfit_raw)),
            "trainfit_sum_raw": float(np.sum(w_trainfit_raw)),
            "trainfit_sum_fit": float(np.sum(w_fit)),
            "calib_n": int(len(w_calib_raw)),
            "calib_sum_raw": float(np.sum(w_calib_raw)),
        }

        pipe_final = make_pipeline(
            numeric_features=requested,
            categorical_features=cat_features,
            interaction_ticker_col=interaction_ticker_col,
            interaction_x_col="x_logit_prn",
            interaction_scope_tickers=interaction_scope_tickers,
            enable_interactions=args.ticker_x_interactions,
            C=best_C,
            random_state=args.random_state,
        )
        pipe_final.fit(X_trainfit, y_trainfit, clf__sample_weight=w_fit)

        # Optional Platt calibrator (fit ONLY on CALIB)
        if use_calib:
            z_cal = pipe_final.decision_function(X_calib)
            w_cal_fit = w_calib_raw.copy()
            if args.fit_weight_renorm == "mean1":
                w_cal_fit = renorm_fit_weights_mean1(w_cal_fit)
            platt_cal = fit_platt_on_logits(z_cal, y_calib, w_cal_fit, random_state=args.random_state)

    # Bundle model for saving/inference
    if args.mode == "two_stage":
        model_kind = "two_stage+platt" if use_calib else "two_stage"
        final_bundle = FinalModelBundle(
            kind=model_kind,
            mode=args.mode,
            numeric_features=requested,
            categorical_features=cat_features,
            ticker_col=args.ticker_col,
            ticker_feature_col=ticker_intercept_col,
            interaction_ticker_col=interaction_ticker_col,
            foundation_tickers=foundation_list,
            ticker_intercepts=args.ticker_intercepts,
            ticker_x_interactions=bool(args.ticker_x_interactions),
            ticker_support=ticker_support_tickers,
            interaction_ticker_support=interaction_scope_tickers,
            stage1_pipeline=stage1_final,
            stage2_preprocessor=stage2_pre_final,
            stage2_model=stage2_model_final,
            stage1_categorical_features=stage1_cat_features,
            stage2_categorical_features=cat_features,
            platt_calibrator=platt_cal,
        )
    else:
        model_kind = "logit+platt" if use_calib else "logit"
        final_bundle = FinalModelBundle(
            kind=model_kind,
            mode=args.mode,
            numeric_features=requested,
            categorical_features=cat_features,
            ticker_col=args.ticker_col,
            ticker_feature_col=ticker_intercept_col,
            interaction_ticker_col=interaction_ticker_col,
            foundation_tickers=foundation_list,
            ticker_intercepts=args.ticker_intercepts,
            ticker_x_interactions=bool(args.ticker_x_interactions),
            ticker_support=ticker_support_tickers,
            interaction_ticker_support=interaction_scope_tickers,
            base_pipeline=pipe_final,
            platt_calibrator=platt_cal,
        )

    trained_bundle = final_bundle
    deployment_model_kind = model_kind
    deploy_bundle = final_bundle
    if fallback_to_baseline:
        deploy_bundle = FinalModelBundle(
            kind="baseline_pRN",
            mode="baseline",
            numeric_features=[],
            categorical_features=[],
            ticker_col=args.ticker_col,
            ticker_feature_col=None,
            interaction_ticker_col=None,
            foundation_tickers=foundation_list,
            ticker_intercepts="none",
            ticker_x_interactions=False,
        )
        deployment_model_kind = "baseline_pRN"

    # -----------------------------
    # OOS model for VAL_POOL metrics (train strictly before val_pool)
    # -----------------------------
    eval_bundle = trained_bundle
    model_kind_val = model_kind
    train_weeks_eval: Optional[np.ndarray] = None
    use_calib_eval_used: Optional[bool] = None
    val_pool_start = pd.Timestamp(val_pool_weeks[0])
    m_oos_train = mask_weeks_before(df, args.week_col, val_pool_start) & m_pretest & train_ticker_keep
    if m_oos_train.sum() < 100:
        print("[WARN] Not enough pre-val data for OOS VAL_POOL; using final model (in-sample) for val_pool metrics.")
    else:
        train_weeks_eval = np.array(sorted(pd.unique(df.loc[m_oos_train, args.week_col])))
        use_calib_eval = use_calib
        train_fit_weeks_eval, calib_weeks_eval = split_train_calib_weeks(
            train_weeks_eval,
            args.calib_frac_of_train,
            use_calib=use_calib_eval,
        )
        m_trainfit_eval = m_oos_train & mask_weeks(df, args.week_col, train_fit_weeks_eval)
        m_calib_eval = m_oos_train & mask_weeks(df, args.week_col, calib_weeks_eval)

        if use_calib_eval and m_calib_eval.sum() < MIN_CALIB_ROWS:
            use_calib_eval = False
            m_trainfit_eval = m_oos_train
            m_calib_eval = np.zeros_like(m_oos_train, dtype=bool)

        if args.mode == "two_stage":
            m_trainfit_eval_fnd = m_trainfit_eval & is_foundation
            m_trainfit_eval_non = m_trainfit_eval & (~is_foundation)
            if m_trainfit_eval_fnd.sum() < 50 or m_trainfit_eval_non.sum() < 50:
                print("[WARN] Not enough pre-val data for two-stage OOS; using final model (in-sample) for val_pool metrics.")
            else:
                X_fnd, y_fnd, w_fnd_raw, wk_fnd = subset(m_trainfit_eval_fnd, stage1_cols)
                X_non_stage1, y_non, w_non_raw, wk_non = subset(m_trainfit_eval_non, stage1_cols)
                X_non_stage2, _, _, _ = subset(m_trainfit_eval_non, stage2_cols)

                w_fit_fnd = build_fit_weights(
                    w_fnd_raw,
                    wk_fnd,
                    foundation_mask=np.ones_like(w_fnd_raw, dtype=bool),
                    foundation_weight=args.foundation_weight,
                    decay_half_life_weeks=args.train_decay_half_life_weeks,
                    fit_weight_renorm=args.fit_weight_renorm,
                    group_id=chain_group_id[m_trainfit_eval_fnd] if chain_group_id is not None else None,
                    enable_group_reweight=enable_group_reweight,
                )
                stage1_eval = make_pipeline(
                    numeric_features=requested,
                    categorical_features=stage1_cat_features,
                    interaction_ticker_col=None,
                    interaction_x_col="x_logit_prn",
                    interaction_scope_tickers=None,
                    enable_interactions=False,
                    C=best_C,
                    random_state=args.random_state,
                )
                stage1_eval.fit(X_fnd, y_fnd, clf__sample_weight=w_fit_fnd)

                p_base_non = stage1_eval.predict_proba(X_non_stage1)[:, 1]
                logit_base_non = _logit(p_base_non)

                w_fit_non = build_fit_weights(
                    w_non_raw,
                    wk_non,
                    foundation_mask=None,
                    foundation_weight=args.foundation_weight,
                    decay_half_life_weeks=args.train_decay_half_life_weeks,
                    fit_weight_renorm=args.fit_weight_renorm,
                    group_id=chain_group_id[m_trainfit_eval_non] if chain_group_id is not None else None,
                    enable_group_reweight=enable_group_reweight,
                )
                stage2_pre_eval = make_preprocessor(
                    numeric_features=requested,
                    categorical_features=cat_features,
                    interaction_ticker_col=interaction_ticker_col,
                    interaction_x_col="x_logit_prn",
                    interaction_scope_tickers=interaction_scope_tickers,
                    enable_interactions=args.ticker_x_interactions,
                )
                z_train = stage2_pre_eval.fit_transform(X_non_stage2)
                stage2_eval = LogisticRegressionOffset(C=best_C, max_iter=2000, tol=1e-6)
                stage2_eval.fit(z_train, y_non, sample_weight=w_fit_non, offset=logit_base_non)

                platt_eval = None
                if use_calib_eval:
                    _, y_calib_eval, w_calib_eval_raw, _ = subset(m_calib_eval, stage1_cols)
                    logits_cal, _ = predict_two_stage(
                        mask=m_calib_eval,
                        stage1_pipe=stage1_eval,
                        stage1_cols_use=stage1_cols,
                        stage2_pre=stage2_pre_eval,
                        stage2_model=stage2_eval,
                        stage2_cols_use=stage2_cols,
                    )
                    w_cal_fit = w_calib_eval_raw.copy()
                    if args.fit_weight_renorm == "mean1":
                        w_cal_fit = renorm_fit_weights_mean1(w_cal_fit)
                    platt_eval = fit_platt_on_logits(logits_cal, y_calib_eval, w_cal_fit, random_state=args.random_state)

                model_kind_val = "two_stage+platt" if use_calib_eval else "two_stage"
                use_calib_eval_used = use_calib_eval
                eval_bundle = FinalModelBundle(
                    kind=model_kind_val,
                    mode=args.mode,
                    numeric_features=requested,
                    categorical_features=cat_features,
                    ticker_col=args.ticker_col,
                    ticker_feature_col=ticker_intercept_col,
                    interaction_ticker_col=interaction_ticker_col,
                    foundation_tickers=foundation_list,
                    ticker_intercepts=args.ticker_intercepts,
                    ticker_x_interactions=bool(args.ticker_x_interactions),
                    ticker_support=ticker_support_tickers,
                    interaction_ticker_support=interaction_scope_tickers,
                    stage1_pipeline=stage1_eval,
                    stage2_preprocessor=stage2_pre_eval,
                    stage2_model=stage2_eval,
                    stage1_categorical_features=stage1_cat_features,
                    stage2_categorical_features=cat_features,
                    platt_calibrator=platt_eval,
                )
        else:
            X_trainfit_eval, y_trainfit_eval, w_trainfit_eval_raw, wk_trainfit_eval = subset(m_trainfit_eval, model_cols_main)
            X_calib_eval, y_calib_eval, w_calib_eval_raw, wk_calib_eval = subset(m_calib_eval, model_cols_main)

            w_fit_eval = build_fit_weights(
                w_trainfit_eval_raw,
                wk_trainfit_eval,
                foundation_mask=is_foundation[m_trainfit_eval],
                foundation_weight=args.foundation_weight,
                decay_half_life_weeks=args.train_decay_half_life_weeks,
                fit_weight_renorm=args.fit_weight_renorm,
                group_id=chain_group_id[m_trainfit_eval] if chain_group_id is not None else None,
                enable_group_reweight=enable_group_reweight,
            )

            pipe_eval = make_pipeline(
                numeric_features=requested,
                categorical_features=cat_features,
                interaction_ticker_col=interaction_ticker_col,
                interaction_x_col="x_logit_prn",
                interaction_scope_tickers=interaction_scope_tickers,
                enable_interactions=args.ticker_x_interactions,
                C=best_C,
                random_state=args.random_state,
            )
            pipe_eval.fit(X_trainfit_eval, y_trainfit_eval, clf__sample_weight=w_fit_eval)

            platt_eval = None
            if use_calib_eval:
                z_cal = pipe_eval.decision_function(X_calib_eval)
                w_cal_fit = w_calib_eval_raw.copy()
                if args.fit_weight_renorm == "mean1":
                    w_cal_fit = renorm_fit_weights_mean1(w_cal_fit)
                platt_eval = fit_platt_on_logits(z_cal, y_calib_eval, w_cal_fit, random_state=args.random_state)

            model_kind_val = "logit+platt" if use_calib_eval else "logit"
            use_calib_eval_used = use_calib_eval
            eval_bundle = FinalModelBundle(
                kind=model_kind_val,
                mode=args.mode,
                numeric_features=requested,
                categorical_features=cat_features,
                ticker_col=args.ticker_col,
                ticker_feature_col=ticker_intercept_col,
                interaction_ticker_col=interaction_ticker_col,
                foundation_tickers=foundation_list,
                ticker_intercepts=args.ticker_intercepts,
                ticker_x_interactions=bool(args.ticker_x_interactions),
                ticker_support=ticker_support_tickers,
                interaction_ticker_support=interaction_scope_tickers,
                base_pipeline=pipe_eval,
                platt_calibrator=platt_eval,
            )

    # -----------------------------
    # Predictions + Metrics (VAL_POOL + TEST)
    # -----------------------------
    _, y_valpool, w_valpool, _ = subset(m_val_pool, stage1_cols)
    _, y_test, w_test, _ = subset(m_test, stage1_cols)

    p_base_val = df.loc[m_val_pool, "pRN"].to_numpy(dtype=float)
    p_base_test = df.loc[m_test, "pRN"].to_numpy(dtype=float)

    p_model_val = eval_bundle.predict_proba_from_df(df.loc[m_val_pool].copy())
    p_model_test = trained_bundle.predict_proba_from_df(df.loc[m_test].copy())

    # Metrics
    met_base_val = evaluate(y_valpool, p_base_val, w=w_valpool, n_bins=args.n_bins, n_bins_q=args.eceq_bins)
    met_base_test = evaluate(y_test, p_base_test, w=w_test, n_bins=args.n_bins, n_bins_q=args.eceq_bins)

    met_model_val = evaluate(y_valpool, p_model_val, w=w_valpool, n_bins=args.n_bins, n_bins_q=args.eceq_bins)
    met_model_test = evaluate(y_test, p_model_test, w=w_test, n_bins=args.n_bins, n_bins_q=args.eceq_bins)

    # Rolling summary row
    sel_row = summary_df.loc[summary_df["C"] == best_C].head(1)
    if not sel_row.empty:
        avg_roll_ll = float(sel_row["avg_roll_logloss_model"].iloc[0])
        avg_roll_ll_base = float(sel_row["avg_roll_logloss_baseline"].iloc[0])
        avg_roll_delta = float(sel_row["avg_roll_delta"].iloc[0])
        n_win_used = int(sel_row["n_windows_used"].iloc[0])
    else:
        avg_roll_ll = float("nan")
        avg_roll_ll_base = float("nan")
        avg_roll_delta = float("nan")
        n_win_used = 0

    # Metrics table for saving
    metrics_rows = [
        {"model": "baseline_pRN", "split": "val_pool", **met_base_val, "n": int(len(y_valpool)), "weight_sum": float(np.sum(w_valpool))},
        {"model": "baseline_pRN", "split": "test", **met_base_test, "n": int(len(y_test)), "weight_sum": float(np.sum(w_test))},
        {"model": model_kind_val + f"_C={best_C:g}", "split": "val_pool", **met_model_val, "n": int(len(y_valpool)), "weight_sum": float(np.sum(w_valpool))},
        {"model": model_kind + f"_C={best_C:g}", "split": "test", **met_model_test, "n": int(len(y_test)), "weight_sum": float(np.sum(w_test))},
        {"model": "rolling_selection", "split": "avg_over_windows",
         "logloss": avg_roll_ll, "brier": float("nan"), "ece": float("nan"),
         "n": n_win_used, "weight_sum": float("nan"),
         "baseline_logloss": avg_roll_ll_base, "delta_model_minus_baseline": avg_roll_delta},
    ]

    # Bootstrap confidence intervals (optional, only when --bootstrap-ci is set)
    bootstrap_ci_results: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    if args.bootstrap_ci:
        print(f"\n=== BOOTSTRAP CONFIDENCE INTERVALS ===")
        print(f"  B={args.bootstrap_B}, seed={args.bootstrap_seed}, group={args.bootstrap_group}")

        for split_name, split_mask, y_split, p_base_split, p_model_split, w_split in [
            ("val_pool", m_val_pool, y_valpool, p_base_val, p_model_val, w_valpool),
            ("test", m_test, y_test, p_base_test, p_model_test, w_test),
        ]:
            df_split = df.loc[split_mask]
            group_keys = _build_bootstrap_group_keys(
                df_split,
                ticker_col=args.ticker_col,
                asof_date_col=asof_date_col,
                expiry_date_col=expiry_date_col,
                week_col=args.week_col,
                strategy=args.bootstrap_group,
            )
            n_groups = len(np.unique(group_keys))
            print(f"  {split_name}: {len(y_split)} rows, {n_groups} groups")

            ci = bootstrap_delta_ci(
                y_split, p_base_split, p_model_split, w_split,
                group_keys=group_keys,
                n_bins=args.n_bins,
                n_bins_q=args.eceq_bins,
                B=args.bootstrap_B,
                seed=args.bootstrap_seed,
            )
            bootstrap_ci_results[split_name] = ci

            for metric_key, ci_data in ci.items():
                lo = ci_data["ci_lo"]
                hi = ci_data["ci_hi"]
                if lo is not None and hi is not None:
                    print(f"    delta_{metric_key} CI95: [{lo:.6f}, {hi:.6f}]")
                else:
                    print(f"    delta_{metric_key} CI95: [NA, NA]")

        # Inject CI columns into model rows of metrics_rows
        _CI_METRIC_KEYS = ["logloss", "brier", "ece", "ece_q"]
        for row in metrics_rows:
            split = row.get("split")
            model_tag = str(row.get("model", ""))
            if model_tag.startswith("baseline") or model_tag.startswith("rolling"):
                for mk in _CI_METRIC_KEYS:
                    row[f"delta_{mk}_ci_lo"] = None
                    row[f"delta_{mk}_ci_hi"] = None
                row["bootstrap_n_groups"] = None
                row["bootstrap_B"] = None
                continue
            ci = bootstrap_ci_results.get(split, {})
            for mk in _CI_METRIC_KEYS:
                ci_data = ci.get(mk, {})
                row[f"delta_{mk}_ci_lo"] = ci_data.get("ci_lo")
                row[f"delta_{mk}_ci_hi"] = ci_data.get("ci_hi")
            first_ci = next(iter(ci.values()), {})
            row["bootstrap_n_groups"] = first_ci.get("n_groups")
            row["bootstrap_B"] = first_ci.get("B")

    metrics_df = pd.DataFrame(metrics_rows)

    # Group diagnostics: foundation vs non-foundation, and top tickers
    group_rows: List[Dict[str, object]] = []
    model_label_val = model_kind_val + f"_C={best_C:g}"
    model_label_test = model_kind + f"_C={best_C:g}"

    is_fnd_val = is_foundation[m_val_pool]
    is_fnd_test = is_foundation[m_test]
    if foundation_set:
        if np.any(is_fnd_val):
            group_rows += group_metrics_rows(
                split="val_pool",
                group_type="foundation",
                group_value="foundation",
                y=y_valpool[is_fnd_val],
                p_base=p_base_val[is_fnd_val],
                p_model=p_model_val[is_fnd_val],
                w=w_valpool[is_fnd_val],
                n_bins=args.n_bins,
                n_bins_q=args.eceq_bins,
                model_label=model_label_val,
            )
        if np.any(~is_fnd_val):
            group_rows += group_metrics_rows(
                split="val_pool",
                group_type="foundation",
                group_value="non_foundation",
                y=y_valpool[~is_fnd_val],
                p_base=p_base_val[~is_fnd_val],
                p_model=p_model_val[~is_fnd_val],
                w=w_valpool[~is_fnd_val],
                n_bins=args.n_bins,
                n_bins_q=args.eceq_bins,
                model_label=model_label_val,
            )
        if np.any(is_fnd_test):
            group_rows += group_metrics_rows(
                split="test",
                group_type="foundation",
                group_value="foundation",
                y=y_test[is_fnd_test],
                p_base=p_base_test[is_fnd_test],
                p_model=p_model_test[is_fnd_test],
                w=w_test[is_fnd_test],
                n_bins=args.n_bins,
                n_bins_q=args.eceq_bins,
                model_label=model_label_test,
            )
        if np.any(~is_fnd_test):
            group_rows += group_metrics_rows(
                split="test",
                group_type="foundation",
                group_value="non_foundation",
                y=y_test[~is_fnd_test],
                p_base=p_base_test[~is_fnd_test],
                p_model=p_model_test[~is_fnd_test],
                w=w_test[~is_fnd_test],
                n_bins=args.n_bins,
                n_bins_q=args.eceq_bins,
                model_label=model_label_test,
            )

    if args.metrics_top_tickers > 0:
        tickers_val = df.loc[m_val_pool, args.ticker_col].astype("string").to_numpy()
        tickers_test = df.loc[m_test, args.ticker_col].astype("string").to_numpy()
        top_val = pd.Series(tickers_val).value_counts().head(args.metrics_top_tickers).index.tolist()
        top_test = pd.Series(tickers_test).value_counts().head(args.metrics_top_tickers).index.tolist()

        for t in top_val:
            m = tickers_val == t
            if np.any(m):
                group_rows += group_metrics_rows(
                    split="val_pool",
                    group_type="ticker",
                    group_value=str(t),
                    y=y_valpool[m],
                    p_base=p_base_val[m],
                    p_model=p_model_val[m],
                    w=w_valpool[m],
                    n_bins=args.n_bins,
                    n_bins_q=args.eceq_bins,
                    model_label=model_label_val,
                )
        for t in top_test:
            m = tickers_test == t
            if np.any(m):
                group_rows += group_metrics_rows(
                    split="test",
                    group_type="ticker",
                    group_value=str(t),
                    y=y_test[m],
                    p_base=p_base_test[m],
                    p_model=p_model_test[m],
                    w=w_test[m],
                    n_bins=args.n_bins,
                    n_bins_q=args.eceq_bins,
                    model_label=model_label_test,
                )

    if "T_days" in df.columns:
        tdays_val = pd.to_numeric(df.loc[m_val_pool, "T_days"], errors="coerce")
        tdays_test = pd.to_numeric(df.loc[m_test, "T_days"], errors="coerce")
        for t in sorted(tdays_val.dropna().unique()):
            m = (tdays_val == t).to_numpy()
            if np.any(m):
                group_rows += group_metrics_rows(
                    split="val_pool",
                    group_type="T_days",
                    group_value=str(int(t)) if float(t).is_integer() else str(t),
                    y=y_valpool[m],
                    p_base=p_base_val[m],
                    p_model=p_model_val[m],
                    w=w_valpool[m],
                    n_bins=args.n_bins,
                    n_bins_q=args.eceq_bins,
                    model_label=model_label_val,
                )
        for t in sorted(tdays_test.dropna().unique()):
            m = (tdays_test == t).to_numpy()
            if np.any(m):
                group_rows += group_metrics_rows(
                    split="test",
                    group_type="T_days",
                    group_value=str(int(t)) if float(t).is_integer() else str(t),
                    y=y_test[m],
                    p_base=p_base_test[m],
                    p_model=p_model_test[m],
                    w=w_test[m],
                    n_bins=args.n_bins,
                    n_bins_q=args.eceq_bins,
                    model_label=model_label_test,
                )

    if "asof_dow" in df.columns:
        asof_val = df.loc[m_val_pool, "asof_dow"].astype("string")
        asof_test = df.loc[m_test, "asof_dow"].astype("string")
        for d in sorted(asof_val.dropna().unique()):
            m = (asof_val == d).to_numpy()
            if np.any(m):
                group_rows += group_metrics_rows(
                    split="val_pool",
                    group_type="asof_dow",
                    group_value=str(d),
                    y=y_valpool[m],
                    p_base=p_base_val[m],
                    p_model=p_model_val[m],
                    w=w_valpool[m],
                    n_bins=args.n_bins,
                    n_bins_q=args.eceq_bins,
                    model_label=model_label_val,
                )
        for d in sorted(asof_test.dropna().unique()):
            m = (asof_test == d).to_numpy()
            if np.any(m):
                group_rows += group_metrics_rows(
                    split="test",
                    group_type="asof_dow",
                    group_value=str(d),
                    y=y_test[m],
                    p_base=p_base_test[m],
                    p_model=p_model_test[m],
                    w=w_test[m],
                    n_bins=args.n_bins,
                    n_bins_q=args.eceq_bins,
                    model_label=model_label_test,
                )

    group_metrics_df = pd.DataFrame(group_rows)

    print("\n=== FINAL FIT SUMMARY ===")
    print(f"best_C (rolling avg): {best_C:g}")
    print(f"mode: {args.mode}")
    print(f"calibration: {args.calibrate}")
    if fallback_to_baseline:
        print(f"deployment_model: {deployment_model_kind} (fallback_to_baseline=True)")
    if model_kind_val != model_kind:
        print(f"val_pool_model_kind: {model_kind_val} (OOS)")
    print("\n=== METRICS (VAL_POOL + TEST) ===")
    # Sort for readability
    show_cols = [c for c in ["model", "split", "logloss", "brier", "ece", "ece_q", "n", "weight_sum",
                            "baseline_logloss", "delta_model_minus_baseline"] if c in metrics_df.columns]
    print(metrics_df[show_cols].to_string(index=False))

    # Coefficients inspection
    if args.mode == "two_stage":
        if stage1_final is not None:
            pre1, coef1, intercept1 = _extract_pre_and_coefs(stage1_final)
            print("\n=== STAGE 1 (foundation-only) ===")
            print_numeric_coeffs_only(pre1, coef1, intercept1)
        if stage2_pre_final is not None and stage2_model_final is not None:
            coef2 = np.asarray(stage2_model_final.coef_).ravel()
            intercept2 = float(np.asarray(stage2_model_final.intercept_).ravel()[0])
            print("\n=== STAGE 2 (correction on non-foundation) ===")
            print_numeric_coeffs_only(stage2_pre_final, coef2, intercept2)
            if args.ticker_intercepts != "none" and ticker_intercept_col:
                print_top_ticker_adjustments(stage2_pre_final, coef2, ticker_intercept_col, top_k=10)
            if args.ticker_x_interactions:
                print_top_interactions(stage2_pre_final, coef2, top_k=10)
    else:
        if pipe_final is not None:
            pre, coef, intercept = _extract_pre_and_coefs(pipe_final)
            print_numeric_coeffs_only(pre, coef, intercept)
            if args.ticker_intercepts != "none" and ticker_intercept_col:
                print_top_ticker_adjustments(pre, coef, ticker_intercept_col, top_k=10)
            if args.ticker_x_interactions:
                print_top_interactions(pre, coef, top_k=10)

    # -----------------------------
    # Reliability bins: baseline + model on val_pool and test
    # -----------------------------
    rel = pd.concat([
        reliability_table(y_valpool, p_base_val, n_bins=args.n_bins, w=w_valpool, label="val_pool__baseline_pRN"),
        reliability_table(y_valpool, p_model_val, n_bins=args.n_bins, w=w_valpool, label=f"val_pool__{model_kind_val}_C={best_C:g}"),
        reliability_table(y_test, p_base_test, n_bins=args.n_bins, w=w_test, label="test__baseline_pRN"),
        reliability_table(y_test, p_model_test, n_bins=args.n_bins, w=w_test, label=f"test__{model_kind}_C={best_C:g}"),
    ], ignore_index=True)

    # -----------------------------
    # Save artifacts
    # -----------------------------
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    rolling_df.to_csv(out_dir / "rolling_windows.csv", index=False)
    summary_df.to_csv(out_dir / "rolling_summary.csv", index=False)
    rel.to_csv(out_dir / "reliability_bins.csv", index=False)
    group_metrics_df.to_csv(out_dir / "metrics_groups.csv", index=False)

    # Save model objects
    if pipe_final is not None:
        joblib.dump(pipe_final, out_dir / "base_pipeline.joblib")
    elif stage1_final is not None:
        joblib.dump(stage1_final, out_dir / "base_pipeline.joblib")
    if platt_cal is not None:
        joblib.dump(platt_cal, out_dir / "platt_calibrator.joblib")
    joblib.dump(deploy_bundle, out_dir / "final_model.joblib")

    # Metadata
    meta = {
        "script": Path(__file__).name,
        "csv": str(Path(args.csv)),
        "out_dir": str(out_dir),
        "target_col": target_col,
        "week_col": args.week_col,
        "ticker_col": args.ticker_col,
        "mode": args.mode,
        "foundation_tickers": foundation_list,
        "foundation_weight": float(args.foundation_weight),
        "train_tickers": train_ticker_list,
        "tdays_allowed": tdays_allowed,
        "asof_dow_allowed": asof_dow_allowed,
        "asof_date_col": asof_date_col,
        "expiry_date_col": expiry_date_col,
        "tdays_constant": feature_gate_info["tdays_constant"],
        "asof_dow_constant": feature_gate_info["asof_dow_constant"],
        "ticker_intercepts": args.ticker_intercepts,
        "ticker_x_interactions": bool(args.ticker_x_interactions),
        "ticker_feature_col": ticker_intercept_col,
        "interaction_ticker_col": interaction_ticker_col,
        "metrics_top_tickers": int(args.metrics_top_tickers),
        "categorical_features_requested": parse_categorical_list(args.categorical_features),
        "categorical_features_used": cat_features,
        "weight_col": weight_col,
        "features_requested": parse_feature_list(args.features),
        "features_used_final": requested,
        "forbidden_features_removed": forbidden_removed,
        "near_constant_features_removed": near_constant_removed,
        "forward_fallback_rows_dropped": forward_fallback_rows,
        "add_interactions": bool(args.add_interactions),
        "calibration_requested": calibration_requested,
        "calibration_used": args.calibrate,
        "selection_objective": args.selection_objective,
        "fallback_to_baseline_if_worse": bool(args.fallback_to_baseline_if_worse),
        "fallback_to_baseline_triggered": bool(fallback_to_baseline),
        "model_kind_val_pool": model_kind_val,
        "model_kind_test": model_kind,
        "model_kind_deployed": deployment_model_kind,
        "C_grid": C_grid,
        "best_C": best_C,
        "rolling": {
            "test_weeks": args.test_weeks,
            "val_windows": args.val_windows,
            "val_window_weeks": args.val_window_weeks,
            "n_windows_used": n_win_used,
            "avg_roll_logloss_model": avg_roll_ll,
            "avg_roll_logloss_baseline": avg_roll_ll_base,
            "avg_roll_delta": avg_roll_delta,
        },
        "two_stage": {
            "stage1_categorical_features": stage1_cat_features if args.mode == "two_stage" else None,
            "stage2_categorical_features": cat_features if args.mode == "two_stage" else None,
        },
        "ticker_support": ticker_support_info,
        "interaction_support": interaction_support_info,
        "ticker_support_tickers": ticker_support_tickers,
        "interaction_support_tickers": interaction_scope_tickers,
        "ticker_min_support": int(args.ticker_min_support),
        "ticker_min_support_interactions": int(args.ticker_min_support_interactions),
        "auto_drop_near_constant": bool(args.auto_drop_near_constant),
        "splits": {
            "n_weeks_total": int(len(uniq_weeks)),
            "n_weeks_pretest": int(len(pretest_weeks)),
            "n_weeks_test": int(len(test_block_weeks)),
            "train_fit_weeks_range": [str(pd.Timestamp(train_fit_weeks[0]).date()), str(pd.Timestamp(train_fit_weeks[-1]).date())] if len(train_fit_weeks) else None,
            "calib_weeks_range": [str(pd.Timestamp(calib_weeks[0]).date()), str(pd.Timestamp(calib_weeks[-1]).date())] if len(calib_weeks) else None,
            "val_pool_weeks_range": [str(pd.Timestamp(val_pool_weeks[0]).date()), str(pd.Timestamp(val_pool_weeks[-1]).date())] if len(val_pool_weeks) else None,
            "test_weeks_range": [str(pd.Timestamp(test_block_weeks[0]).date()), str(pd.Timestamp(test_block_weeks[-1]).date())],
        },
        "val_pool_oos": {
            "used": train_weeks_eval is not None,
            "train_weeks_range": [str(pd.Timestamp(train_weeks_eval[0]).date()), str(pd.Timestamp(train_weeks_eval[-1]).date())] if isinstance(train_weeks_eval, np.ndarray) and len(train_weeks_eval) else None,
            "use_calib": use_calib_eval_used,
            "model_kind": model_kind_val,
            "mode": args.mode,
        },
        "fit_weights": fit_weight_meta,
        "group_reweight": group_reweight_info,
        "moneyness_column_used": moneyness_col,
        "enable_x_abs_m": bool(args.enable_x_abs_m),
        "optional_filters": {
            "max_abs_logm": args.max_abs_logm,
            "drop_prn_extremes": bool(args.drop_prn_extremes),
            "prn_eps": args.prn_eps,
            "rows_filtered": total_filtered,
        },
        "min_prn_train": prn_train_min,
        "max_prn_train": prn_train_max,
        "prn_train_range": [prn_train_min, prn_train_max] if prn_train_min is not None and prn_train_max is not None else None,
        "random_state": int(args.random_state),
        "n_bins": int(args.n_bins),
        "eceq_bins": int(args.eceq_bins),
        "bootstrap_ci": bool(args.bootstrap_ci),
        "bootstrap_B": int(args.bootstrap_B) if args.bootstrap_ci else None,
        "bootstrap_seed": int(args.bootstrap_seed) if args.bootstrap_ci else None,
        "bootstrap_group": args.bootstrap_group if args.bootstrap_ci else None,
    }
    def _safe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(result):
            return None
        return result

    best_windows = rolling_df[rolling_df["C"] == best_C].sort_values("window_start")
    val_losses = (
        best_windows["logloss_model"].to_numpy(dtype=float)
        if not best_windows.empty
        else np.array([], dtype=float)
    )
    val_window_entries: List[Dict[str, Any]] = []
    for _, row in best_windows.iterrows():
        val_window_entries.append({
            "window": row["window"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
            "logloss": float(row["logloss_model"]),
            "baseline_logloss": float(row["logloss_baseline"]),
        })
    test_metrics_row = metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["model"] == model_label_test)
    ]
    test_metrics: Dict[str, Optional[float]] = {
        "logloss": None,
        "brier": None,
        "ece": None,
        "ece_q": None,
        "baseline_logloss": None,
    }
    if not test_metrics_row.empty:
        row = test_metrics_row.iloc[0]
        test_metrics = {
            "logloss": _safe_float(row.get("logloss")),
            "brier": _safe_float(row.get("brier")),
            "ece": _safe_float(row.get("ece")),
            "ece_q": _safe_float(row.get("ece_q")),
            "baseline_logloss": _safe_float(row.get("baseline_logloss")),
        }
    metrics_summary = {
        "val_logloss_mean": float(best_avg_ll),
        "val_logloss_std": float(np.std(val_losses)) if val_losses.size else None,
        "val_logloss_by_window": val_window_entries,
        "val_windows_count": int(best_windows.shape[0]),
        "test": test_metrics,
    }
    if args.bootstrap_ci and bootstrap_ci_results:
        metrics_summary["bootstrap"] = {
            "B": args.bootstrap_B,
            "seed": args.bootstrap_seed,
            "group_strategy": args.bootstrap_group,
            "val_pool": bootstrap_ci_results.get("val_pool", {}),
            "test": bootstrap_ci_results.get("test", {}),
        }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2))
    now = datetime.now(timezone.utc)
    git_info = get_git_info()
    training_ts = now.isoformat()
    required_columns = build_required_columns(
        numeric_features=requested,
        categorical_features=cat_features,
        ticker_col=args.ticker_col,
        ticker_feature_col=ticker_intercept_col,
        interaction_ticker_col=interaction_ticker_col,
    )
    feature_manifest = {
        "script": Path(__file__).name,
        "model_version": CALIBRATOR_VERSION,
        "training_timestamp": training_ts,
        "git_commit": git_info["git_commit"],
        "git_commit_datetime": git_info["git_commit_datetime"],
        "tdays_allowed": tdays_allowed,
        "asof_dow_allowed": asof_dow_allowed,
        "asof_date_col": asof_date_col,
        "expiry_date_col": expiry_date_col,
        "tdays_constant": feature_gate_info["tdays_constant"],
        "asof_dow_constant": feature_gate_info["asof_dow_constant"],
        "ticker_col": args.ticker_col,
        "numeric_features": requested,
        "categorical_features": cat_features,
        "required_columns": required_columns,
        "derived_features": dict(DERIVED_FEATURE_DESCRIPTIONS),
        "notes": "Missing columns such as rv20/dividend_yield/forward_price stay NaN until joined from historical data.",
    }
    manifest_path = out_dir / "feature_manifest.json"
    artifact_path = out_dir / "model.joblib"
    train_end_date = str(pd.Timestamp(train_weeks_used[-1]).date()) if len(train_weeks_used) else None
    meta.update({
        "model_version": CALIBRATOR_VERSION,
        "training_timestamp": training_ts,
        "git_commit": git_info["git_commit"],
        "git_commit_datetime": git_info["git_commit_datetime"],
        "required_columns": required_columns,
        "feature_manifest_path": manifest_path.name,
        "model_artifact": artifact_path.name,
        "train_end_date": train_end_date,
    })
    coefficient_names: List[str] = []
    coefficient_values: List[float] = []
    intercept_value: Optional[float] = None

    if pipe_final is not None:
        pre, coefs, intercept = _extract_pre_and_coefs(pipe_final)
        coefficient_names = [_simplify_feature_name(name) for name in pre.get_feature_names_out()]
        coefficient_values = [float(c) for c in coefs]
        intercept_value = intercept
    elif stage2_pre_final is not None and stage2_model_final is not None:
        names = stage2_pre_final.get_feature_names_out()
        coefficient_names = [_simplify_feature_name(name) for name in names]
        coefficient_values = [float(c) for c in np.asarray(stage2_model_final.coef_).ravel()]
        intercept_value = float(np.asarray(stage2_model_final.intercept_).ravel()[0])

    if coefficient_names and coefficient_values and len(coefficient_names) == len(coefficient_values) and intercept_value is not None:
        meta.update({
            "features": coefficient_names,
            "coefficients": coefficient_values,
            "intercept": intercept_value,
        })
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    manifest_path.write_text(json.dumps(feature_manifest, indent=2))
    artifact_data = {
        "bundle": deploy_bundle,
        "metadata": {
            "model_version": CALIBRATOR_VERSION,
            "training_timestamp": training_ts,
            "git_commit": git_info["git_commit"],
            "git_commit_datetime": git_info["git_commit_datetime"],
            "tdays_allowed": tdays_allowed,
            "asof_dow_allowed": asof_dow_allowed,
            "asof_date_col": asof_date_col,
            "expiry_date_col": expiry_date_col,
            "tdays_constant": feature_gate_info["tdays_constant"],
            "asof_dow_constant": feature_gate_info["asof_dow_constant"],
            "forbidden_features_removed": forbidden_removed,
            "near_constant_features_removed": near_constant_removed,
            "forward_fallback_rows_dropped": forward_fallback_rows,
            "ticker_support": ticker_support_info,
            "interaction_support": interaction_support_info,
            "ticker_support_tickers": ticker_support_tickers,
            "interaction_support_tickers": interaction_scope_tickers,
            "selection_objective": args.selection_objective,
            "fallback_to_baseline_triggered": bool(fallback_to_baseline),
            "model_kind_deployed": deployment_model_kind,
            "min_prn_train": prn_train_min,
            "max_prn_train": prn_train_max,
            "prn_train_range": [prn_train_min, prn_train_max] if prn_train_min is not None and prn_train_max is not None else None,
            "feature_manifest": feature_manifest,
        },
    }
    joblib.dump(artifact_data, artifact_path)

    print(f"\nSaved artifacts to: {out_dir.resolve()}")
    print("Files: metrics.csv, metrics_groups.csv, rolling_windows.csv, rolling_summary.csv, reliability_bins.csv, base_pipeline.joblib, final_model.joblib, model.joblib, feature_manifest.json, metadata.json")


if __name__ == "__main__":
    main()
