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
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-6
MIN_CALIB_ROWS = 50


# -----------------------------
# Numerics / Metrics
# -----------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


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


def evaluate(y: np.ndarray, p: np.ndarray, *, w: Optional[np.ndarray], n_bins: int) -> Dict[str, float]:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return {
        "logloss": float(log_loss(y, p, labels=[0, 1], sample_weight=w)),
        "brier": weighted_brier(y, p, w),
        "ece": ece_score(y, p, n_bins=n_bins, w=w),
    }


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


# -----------------------------
# Model / Calibration
# -----------------------------

def make_pipeline(
    *,
    numeric_features: List[str],
    categorical_features: List[str],
    C: float,
    random_state: int,
) -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
    ])

    transformers = [("num", num_pipe, numeric_features)]
    if categorical_features:
        transformers.append(("cat", cat_pipe, categorical_features))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,  # OK either way
    )

    clf = LogisticRegression(
        penalty="l2",
        C=float(C),
        solver="lbfgs",
        max_iter=4000,
        random_state=int(random_state),
    )
    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def fit_platt_on_logits(
    logits: np.ndarray,
    y: np.ndarray,
    w_fit: Optional[np.ndarray],
    *,
    random_state: int,
) -> LogisticRegression:
    Xc = logits.reshape(-1, 1)
    cal = LogisticRegression(
        penalty="l2",
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=int(random_state),
    )
    cal.fit(Xc, y, sample_weight=w_fit)
    return cal


def apply_platt(cal: LogisticRegression, logits: np.ndarray) -> np.ndarray:
    return cal.predict_proba(logits.reshape(-1, 1))[:, 1]


# -----------------------------
# Final model bundle (saved)
# -----------------------------

@dataclass
class FinalModelBundle:
    kind: str  # "baseline_pRN" | "logit" | "logit+platt"
    numeric_features: List[str]
    ticker_col: str
    categorical_features: List[str]
    base_pipeline: Optional[Pipeline] = None
    platt_calibrator: Optional[LogisticRegression] = None
    eps: float = EPS

    def predict_proba_from_df(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "baseline_pRN":
            if "pRN" not in df.columns:
                raise ValueError("baseline_pRN requires pRN column.")
            p = pd.to_numeric(df["pRN"], errors="coerce").to_numpy(dtype=float)
            return np.clip(p, self.eps, 1.0 - self.eps)

        if self.base_pipeline is None:
            raise ValueError("Missing base_pipeline.")

        cat_cols = dedupe_preserve_order([self.ticker_col] + list(self.categorical_features))
        X = df[self.numeric_features + cat_cols].copy()
        for c in self.numeric_features:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        for c in cat_cols:
            X[c] = X[c].astype("string").fillna("UNKNOWN")

        p = self.base_pipeline.predict_proba(X)[:, 1]

        if self.kind == "logit+platt":
            if self.platt_calibrator is None:
                raise ValueError("Missing platt_calibrator.")
            logits = self.base_pipeline.decision_function(X)
            p = apply_platt(self.platt_calibrator, logits)

        return np.clip(p.astype(float), self.eps, 1.0 - self.eps)


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


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


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


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def ensure_engineered_features(df: pd.DataFrame, requested_features: List[str]) -> pd.DataFrame:
    df = df.copy()

    for c in [
        "T_days",
        "T_years",
        "rv20",
        "r",
        "log_m",
        "abs_log_m",
        "dividend_yield",
        "forward_price",
        "log_m_fwd",
        "abs_log_m_fwd",
        "rel_spread_median",
        "n_chain_raw",
        "n_chain_used",
        "n_band_raw",
        "n_band_inside",
        "dropped_intrinsic",
        "asof_fallback_days",
        "expiry_fallback_days",
        "pRN",
        "pRN_raw",
        "x_logit_prn",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Compute log_m if requested and missing (safe: uses as-of spot only).
    if "log_m" in requested_features and "log_m" not in df.columns:
        if ("K" in df.columns) and ("S_asof_close" in df.columns):
            K = _numeric_series(df, "K").to_numpy(dtype=float)
            S0 = _numeric_series(df, "S_asof_close").to_numpy(dtype=float)
            df["log_m"] = np.log(np.clip(K, 1e-12, None) / np.clip(S0, 1e-12, None))

    if "abs_log_m" in requested_features and "abs_log_m" not in df.columns:
        if "log_m" in df.columns:
            df["abs_log_m"] = pd.to_numeric(df["log_m"], errors="coerce").abs()

    need_forward = any(
        f in requested_features for f in ["forward_price", "log_m_fwd", "abs_log_m_fwd", "log_m_fwd_over_volT", "abs_log_m_fwd_over_volT"]
    )
    if need_forward and "forward_price" not in df.columns:
        if ("S_asof_close" in df.columns) and ("r" in df.columns) and ("dividend_yield" in df.columns):
            S0 = _numeric_series(df, "S_asof_close")
            r = _numeric_series(df, "r")
            q = _numeric_series(df, "dividend_yield")
            if "T_years" in df.columns:
                T_years = _numeric_series(df, "T_years")
            else:
                T_years = _numeric_series(df, "T_days") / 365.0
            df["forward_price"] = S0 * np.exp((r - q) * T_years)

    if "log_m_fwd" in requested_features and "log_m_fwd" not in df.columns:
        if ("K" in df.columns) and ("forward_price" in df.columns):
            K = _numeric_series(df, "K").to_numpy(dtype=float)
            F = _numeric_series(df, "forward_price").to_numpy(dtype=float)
            df["log_m_fwd"] = np.log(np.clip(K, 1e-12, None) / np.clip(F, 1e-12, None))

    if "abs_log_m_fwd" in requested_features and "abs_log_m_fwd" not in df.columns:
        if "log_m_fwd" in df.columns:
            df["abs_log_m_fwd"] = pd.to_numeric(df["log_m_fwd"], errors="coerce").abs()

    # Horizon transforms
    if "log_T_days" in requested_features:
        T_days = _numeric_series(df, "T_days")
        df["log_T_days"] = np.log1p(T_days.clip(lower=0))

    need_sqrt_T = any(
        f in requested_features for f in ["sqrt_T_years", "rv20_sqrtT", "log_m_over_volT", "abs_log_m_over_volT", "log_m_fwd_over_volT", "abs_log_m_fwd_over_volT"]
    )
    if need_sqrt_T:
        if "T_years" in df.columns:
            T_years = _numeric_series(df, "T_years")
        else:
            T_years = _numeric_series(df, "T_days") / 365.0
        df["sqrt_T_years"] = np.sqrt(T_years.clip(lower=0))

    if "rv20_sqrtT" in requested_features:
        df["rv20_sqrtT"] = _numeric_series(df, "rv20") * _numeric_series(df, "sqrt_T_years")

    if ("log_m_over_volT" in requested_features) or ("abs_log_m_over_volT" in requested_features):
        denom = _numeric_series(df, "rv20") * _numeric_series(df, "sqrt_T_years")
        denom = denom.replace(0, np.nan)
        if "log_m_over_volT" in requested_features:
            df["log_m_over_volT"] = _numeric_series(df, "log_m") / denom
        if "abs_log_m_over_volT" in requested_features:
            df["abs_log_m_over_volT"] = _numeric_series(df, "abs_log_m") / denom

    if ("log_m_fwd_over_volT" in requested_features) or ("abs_log_m_fwd_over_volT" in requested_features):
        denom = _numeric_series(df, "rv20") * _numeric_series(df, "sqrt_T_years")
        denom = denom.replace(0, np.nan)
        if "log_m_fwd_over_volT" in requested_features:
            df["log_m_fwd_over_volT"] = _numeric_series(df, "log_m_fwd") / denom
        if "abs_log_m_fwd_over_volT" in requested_features:
            df["abs_log_m_fwd_over_volT"] = _numeric_series(df, "abs_log_m_fwd") / denom

    # Liquidity / quality transforms
    if "log_rel_spread" in requested_features:
        rs = _numeric_series(df, "rel_spread_median").clip(lower=0)
        df["log_rel_spread"] = np.log1p(rs)

    if "chain_used_frac" in requested_features:
        df["chain_used_frac"] = _safe_divide(_numeric_series(df, "n_chain_used"), _numeric_series(df, "n_chain_raw"))

    if "band_inside_frac" in requested_features:
        df["band_inside_frac"] = _safe_divide(_numeric_series(df, "n_band_inside"), _numeric_series(df, "n_band_raw"))

    if "drop_intrinsic_frac" in requested_features:
        df["drop_intrinsic_frac"] = _safe_divide(_numeric_series(df, "dropped_intrinsic"), _numeric_series(df, "n_chain_raw"))

    if "fallback_any" in requested_features:
        a = _numeric_series(df, "asof_fallback_days").fillna(0)
        e = _numeric_series(df, "expiry_fallback_days").fillna(0)
        df["fallback_any"] = ((a > 0) | (e > 0)).astype(float)

    if "prn_raw_gap" in requested_features:
        df["prn_raw_gap"] = _numeric_series(df, "pRN") - _numeric_series(df, "pRN_raw")

    # Interactions with logit(pRN) (optional)
    if "x_prn_x_tdays" in requested_features and "x_prn_x_tdays" not in df.columns:
        df["x_prn_x_tdays"] = _numeric_series(df, "x_logit_prn") * _numeric_series(df, "T_days")

    if "x_prn_x_rv20" in requested_features and "x_prn_x_rv20" not in df.columns:
        df["x_prn_x_rv20"] = _numeric_series(df, "x_logit_prn") * _numeric_series(df, "rv20")

    if "x_prn_x_logm" in requested_features and "x_prn_x_logm" not in df.columns:
        df["x_prn_x_logm"] = _numeric_series(df, "x_logit_prn") * _numeric_series(df, "log_m")

    return df


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

def print_numeric_coeffs_only(pipe: Pipeline, numeric_features: List[str]) -> None:
    """Print only the numeric part coefficients (on standardized numeric features)."""
    clf: LogisticRegression = pipe.named_steps["clf"]
    pre: ColumnTransformer = pipe.named_steps["pre"]
    feature_names = list(pre.get_feature_names_out())

    coefs = clf.coef_.ravel()
    intercept = float(clf.intercept_[0])

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


def print_top_ticker_adjustments(pipe: Pipeline, ticker_col: str, top_k: int = 10) -> None:
    """
    Approximate ticker intercept adjustments from one-hot coefficients.
    With drop='first', the reference ticker has 0 adjustment; others are relative.
    """
    clf: LogisticRegression = pipe.named_steps["clf"]
    pre: ColumnTransformer = pipe.named_steps["pre"]
    feature_names = list(pre.get_feature_names_out())
    coefs = clf.coef_.ravel()

    # cat onehots come out like: "cat__ticker_<LEVEL>" depending on sklearn version
    rows = []
    for fname, c in zip(feature_names, coefs):
        if fname.startswith("cat__"):
            # fname example: cat__ticker_AAPL or cat__ticker_col_AAPL depending on sklearn
            # We'll just strip the prefix and keep the tail
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

    ap.add_argument(
        "--features",
        default="x_logit_prn,log_m_fwd,abs_log_m_fwd,T_days,sqrt_T_years,rv20,rv20_sqrtT,log_m_fwd_over_volT,log_rel_spread,chain_used_frac,band_inside_frac,drop_intrinsic_frac,asof_fallback_days,split_events_in_preload_range,prn_raw_gap,dividend_yield",
    )
    ap.add_argument("--categorical-features", default="spot_scale_used")
    ap.add_argument("--add-interactions", action="store_true")
    ap.add_argument("--calibrate", choices=["none", "platt"], default="none")

    ap.add_argument("--C-grid", default="0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10")

    ap.add_argument("--train-decay-half-life-weeks", type=float, default=0.0)
    ap.add_argument("--calib-frac-of-train", type=float, default=0.20)
    ap.add_argument("--fit-weight-renorm", choices=["none", "mean1"], default="mean1")

    # Rolling validation settings (step 3)
    ap.add_argument("--test-weeks", type=int, default=20)
    ap.add_argument("--val-windows", type=int, default=4)
    ap.add_argument("--val-window-weeks", type=int, default=10)

    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--random-state", type=int, default=7)

    args = ap.parse_args()
    calibration_requested = args.calibrate
    if args.test_weeks <= 0:
        raise ValueError("--test-weeks must be >= 1.")
    if args.val_windows <= 0 or args.val_window_weeks <= 0:
        raise ValueError("--val-windows and --val-window-weeks must be >= 1.")
    if args.n_bins < 2:
        raise ValueError("--n-bins must be >= 2.")
    if not (0.0 <= args.calib_frac_of_train < 1.0):
        raise ValueError("--calib-frac-of-train must be in [0,1).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Parse dates if present
    for c in ["asof_date", "week_friday", "week_monday", "expiry_close_date_used"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Required columns
    if args.week_col not in df.columns:
        raise ValueError(f"Missing week column: {args.week_col}")
    if args.ticker_col not in df.columns:
        raise ValueError(f"Missing ticker column: {args.ticker_col}")

    target_col = pick_target_column(df, args.target_col)
    weight_col = pick_weight_column(df, args.weight_col)

    # Target cleanup
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])].copy()
    if df.empty:
        raise ValueError("No rows left after filtering valid target values {0,1}.")

    # Week sanity
    df[args.week_col] = pd.to_datetime(df[args.week_col], errors="coerce")
    if df[args.week_col].isna().any():
        bad = int(df[args.week_col].isna().sum())
        raise ValueError(f"{args.week_col} has {bad} NaT values; cannot proceed.")
    df = df[df[args.week_col].notna()].copy()

    # Ticker sanity
    df[args.ticker_col] = df[args.ticker_col].astype("string").fillna("UNKNOWN")
    n_tickers = int(df[args.ticker_col].nunique())
    print(f"\n=== TICKER SANITY ===\nunique_tickers: {n_tickers}")
    if n_tickers < 2:
        raise ValueError("Need at least 2 tickers for ticker-based model.")
    if n_tickers > 5000:
        print("[WARN] Very high ticker cardinality; one-hot may be too large. Consider restricting universe.")

    # pRN required
    if "pRN" not in df.columns:
        raise ValueError("CSV missing pRN.")
    df["pRN"] = pd.to_numeric(df["pRN"], errors="coerce").clip(EPS, 1.0 - EPS)
    df = df[np.isfinite(df["pRN"])].copy()
    if df.empty:
        raise ValueError("No rows left after filtering finite pRN.")
    df["x_logit_prn"] = _logit(df["pRN"].to_numpy(dtype=float))

    # Features
    requested = parse_feature_list(args.features)
    if args.add_interactions:
        if ("T_days" in df.columns or "T_days" in requested) and "x_prn_x_tdays" not in requested:
            requested.append("x_prn_x_tdays")
        if ("rv20" in df.columns or "rv20" in requested) and "x_prn_x_rv20" not in requested:
            requested.append("x_prn_x_rv20")
        if ("log_m" in df.columns or "log_m" in requested) and "x_prn_x_logm" not in requested:
            requested.append("x_prn_x_logm")

    df = ensure_engineered_features(df, requested)

    # Categorical features (ticker + optional extras)
    extra_cat = parse_categorical_list(args.categorical_features)
    cat_features = dedupe_preserve_order([args.ticker_col] + extra_cat)

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

    print("\n=== ROLLING WINDOW SPECS ===")
    for w in windows:
        print(f"{w.name}: window=[{w.window_start.date()}..{w.window_end.date()}] ({w.n_weeks} wks), train uses weeks < {w.window_start.date()}")
    print(f"FINAL TEST block: [{pd.Timestamp(test_block_weeks[0]).date()}..{pd.Timestamp(test_block_weeks[-1]).date()}] ({len(test_block_weeks)} wks)")

    # Helper to subset rows
    def subset(mask: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        Xsub = df.loc[mask, requested + cat_features].copy()
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
    use_calib = args.calibrate == "platt"

    # Rolling evaluation for each C
    roll_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    best_C = None
    best_avg_ll = float("inf")

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
            m_train = m_train & m_pretest
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

            X_trainfit, y_trainfit, w_trainfit_raw, wk_trainfit = subset(m_trainfit)
            X_calib, y_calib, w_calib_raw, wk_calib = subset(m_calib)
            X_val, y_val, w_val, wk_val = subset(m_val)
            if use_platt_window and len(y_calib) < MIN_CALIB_ROWS:
                # Fall back to full training set if calib slice is too small.
                use_platt_window = False
                m_trainfit = m_train
                m_calib = np.zeros_like(m_train, dtype=bool)
                X_trainfit, y_trainfit, w_trainfit_raw, wk_trainfit = subset(m_trainfit)
                X_calib, y_calib, w_calib_raw, wk_calib = subset(m_calib)

            # Fit weights: decay + renorm (fit-time only)
            w_fit = w_trainfit_raw.copy()
            if args.train_decay_half_life_weeks and args.train_decay_half_life_weeks > 0:
                w_fit = apply_recency_decay(w_fit, wk_trainfit, half_life_weeks=args.train_decay_half_life_weeks)
            if args.fit_weight_renorm == "mean1":
                w_fit = renorm_fit_weights_mean1(w_fit)

            pipe = make_pipeline(
                numeric_features=requested,
                categorical_features=cat_features,
                C=C,
                random_state=args.random_state,
            )
            pipe.fit(X_trainfit, y_trainfit, clf__sample_weight=w_fit)

            # Predictions
            p_val_model = pipe.predict_proba(X_val)[:, 1]
            p_val_base = baseline_pred(m_val)

            # Optional Platt (fit on CALIB slice only)
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
            summary_rows.append({
                "C": C,
                "avg_roll_logloss_model": avg_ll,
                "avg_roll_logloss_baseline": avg_ll_base,
                "avg_roll_delta": avg_ll - avg_ll_base,
                "n_windows_used": int(len(lls_model)),
            })
            if avg_ll < best_avg_ll:
                best_avg_ll = avg_ll
                best_C = C

    if best_C is None:
        raise ValueError("No rolling windows produced usable evaluations (train/val too small).")

    rolling_df = pd.DataFrame(roll_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("avg_roll_logloss_model").reset_index(drop=True)

    print("\n=== ROLLING SELECTION SUMMARY (best by avg model logloss) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSelected C by rolling avg: {best_C:g}  (avg_roll_logloss_model={best_avg_ll:.6f})")

    # -----------------------------
    # Final fit: train on ALL pretest weeks (strictly before test block)
    # -----------------------------
    # Masks for final fit / eval
    m_pretest = mask_weeks(df, args.week_col, pretest_weeks)  # all weeks before TEST
    m_test = mask_weeks(df, args.week_col, test_block_weeks)

    # Train weeks used for final fit: all pretest weeks
    train_weeks_used = np.array(sorted(pd.unique(df.loc[m_pretest, args.week_col])))
    train_fit_weeks, calib_weeks = split_train_calib_weeks(
        train_weeks_used,
        args.calib_frac_of_train,
        use_calib=use_calib,
    )

    # Val pool: the same last block of pretest weeks used by rolling windows
    val_pool_weeks = pretest_weeks[-(args.val_windows * args.val_window_weeks):]
    m_val_pool = mask_weeks(df, args.week_col, val_pool_weeks)

    m_trainfit = m_pretest & mask_weeks(df, args.week_col, train_fit_weeks)
    m_calib = m_pretest & mask_weeks(df, args.week_col, calib_weeks)

    # Subsets
    X_trainfit, y_trainfit, w_trainfit_raw, wk_trainfit = subset(m_trainfit)
    X_calib, y_calib, w_calib_raw, wk_calib = subset(m_calib)
    X_valpool, y_valpool, w_valpool, wk_valpool = subset(m_val_pool)
    X_test, y_test, w_test, wk_test = subset(m_test)

    if use_calib and len(y_calib) < MIN_CALIB_ROWS:
        print(f"[WARN] CALIB rows={len(y_calib)} < {MIN_CALIB_ROWS}; refitting without calibration.")
        use_calib = False
        args.calibrate = "none"
        train_fit_weeks = train_weeks_used
        calib_weeks = train_weeks_used[:0]
        m_trainfit = m_pretest
        m_calib = np.zeros_like(m_pretest, dtype=bool)
        X_trainfit, y_trainfit, w_trainfit_raw, wk_trainfit = subset(m_trainfit)
        X_calib, y_calib, w_calib_raw, wk_calib = subset(m_calib)

    # -----------------------------
    # Sanity checks: non-missing counts by split, and drop all-missing-in-trainfit features
    # -----------------------------
    def _nonmissing_counts(X: pd.DataFrame, feats: List[str]) -> Dict[str, int]:
        return {c: int(pd.to_numeric(X[c], errors="coerce").notna().sum()) for c in feats}

    nm_trainfit = _nonmissing_counts(X_trainfit, requested)
    nm_calib = _nonmissing_counts(X_calib, requested)
    nm_val = _nonmissing_counts(X_valpool, requested)
    nm_test = _nonmissing_counts(X_test, requested)

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

        # Rebuild subsets with updated feature list
        X_trainfit = df.loc[m_trainfit, requested + cat_features].copy()
        X_calib = df.loc[m_calib, requested + cat_features].copy()
        X_valpool = df.loc[m_val_pool, requested + cat_features].copy()
        X_test = df.loc[m_test, requested + cat_features].copy()

        # Ensure numeric types remain clean
        for c in requested:
            X_trainfit[c] = pd.to_numeric(X_trainfit[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_calib[c] = pd.to_numeric(X_calib[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_valpool[c] = pd.to_numeric(X_valpool[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    print("\n=== FINAL FEATURES USED FOR FIT ===")
    print(requested)
    print("\n=== FINAL CATEGORICAL FEATURES USED FOR FIT ===")
    print(cat_features)

    # -----------------------------
    # Fit weights: decay + renorm (fit-time only)
    # -----------------------------
    w_fit = w_trainfit_raw.copy()
    if args.train_decay_half_life_weeks and args.train_decay_half_life_weeks > 0:
        w_fit = apply_recency_decay(w_fit, wk_trainfit, half_life_weeks=args.train_decay_half_life_weeks)

    if args.fit_weight_renorm == "mean1":
        w_fit = renorm_fit_weights_mean1(w_fit)

    print("\n=== FIT WEIGHT DEBUG ===")
    print(f"TRAIN_FIT: n={len(w_trainfit_raw)} sum_raw={float(np.sum(w_trainfit_raw)):.6f} sum_fit={float(np.sum(w_fit)):.6f}")
    print(f"CALIB    : n={len(w_calib_raw)} sum_raw={float(np.sum(w_calib_raw)):.6f}")
    if args.train_decay_half_life_weeks and args.train_decay_half_life_weeks > 0:
        print(f"TRAIN_FIT recency decay half-life weeks: {args.train_decay_half_life_weeks:g}")

    # -----------------------------
    # Fit final base pipeline (ticker intercepts + numeric)
    # -----------------------------
    pipe_final = make_pipeline(
        numeric_features=requested,
        categorical_features=cat_features,
        C=best_C,
        random_state=args.random_state,
    )
    pipe_final.fit(X_trainfit, y_trainfit, clf__sample_weight=w_fit)

    # Optional Platt calibrator (fit ONLY on CALIB)
    platt_cal = None
    if use_calib:
        z_cal = pipe_final.decision_function(X_calib)
        w_cal_fit = w_calib_raw.copy()
        if args.fit_weight_renorm == "mean1":
            w_cal_fit = renorm_fit_weights_mean1(w_cal_fit)
        platt_cal = fit_platt_on_logits(z_cal, y_calib, w_cal_fit, random_state=args.random_state)

    # Bundle model for saving/inference
    model_kind = "logit+platt" if use_calib else "logit"
    final_bundle = FinalModelBundle(
        kind=model_kind,
        numeric_features=requested,
        ticker_col=args.ticker_col,
        categorical_features=cat_features,
        base_pipeline=pipe_final,
        platt_calibrator=platt_cal,
    )

    # -----------------------------
    # OOS model for VAL_POOL metrics (train strictly before val_pool)
    # -----------------------------
    eval_bundle = final_bundle
    model_kind_val = model_kind
    train_weeks_eval: Optional[np.ndarray] = None
    use_calib_eval_used: Optional[bool] = None
    val_pool_start = pd.Timestamp(val_pool_weeks[0])
    m_oos_train = mask_weeks_before(df, args.week_col, val_pool_start) & m_pretest
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

        X_trainfit_eval, y_trainfit_eval, w_trainfit_eval_raw, wk_trainfit_eval = subset(m_trainfit_eval)
        X_calib_eval, y_calib_eval, w_calib_eval_raw, wk_calib_eval = subset(m_calib_eval)

        if use_calib_eval and len(y_calib_eval) < MIN_CALIB_ROWS:
            use_calib_eval = False
            m_trainfit_eval = m_oos_train
            m_calib_eval = np.zeros_like(m_oos_train, dtype=bool)
            X_trainfit_eval, y_trainfit_eval, w_trainfit_eval_raw, wk_trainfit_eval = subset(m_trainfit_eval)
            X_calib_eval, y_calib_eval, w_calib_eval_raw, wk_calib_eval = subset(m_calib_eval)

        w_fit_eval = w_trainfit_eval_raw.copy()
        if args.train_decay_half_life_weeks and args.train_decay_half_life_weeks > 0:
            w_fit_eval = apply_recency_decay(w_fit_eval, wk_trainfit_eval, half_life_weeks=args.train_decay_half_life_weeks)
        if args.fit_weight_renorm == "mean1":
            w_fit_eval = renorm_fit_weights_mean1(w_fit_eval)

        pipe_eval = make_pipeline(
            numeric_features=requested,
            categorical_features=cat_features,
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
            numeric_features=requested,
            ticker_col=args.ticker_col,
            categorical_features=cat_features,
            base_pipeline=pipe_eval,
            platt_calibrator=platt_eval,
        )

    # -----------------------------
    # Predictions + Metrics (VAL_POOL + TEST)
    # -----------------------------
    p_base_val = df.loc[m_val_pool, "pRN"].to_numpy(dtype=float)
    p_base_test = df.loc[m_test, "pRN"].to_numpy(dtype=float)

    p_model_val = eval_bundle.predict_proba_from_df(df.loc[m_val_pool, requested + cat_features + ["pRN"]])
    p_model_test = final_bundle.predict_proba_from_df(df.loc[m_test, requested + cat_features + ["pRN"]])

    # Metrics
    met_base_val = evaluate(y_valpool, p_base_val, w=w_valpool, n_bins=args.n_bins)
    met_base_test = evaluate(y_test, p_base_test, w=w_test, n_bins=args.n_bins)

    met_model_val = evaluate(y_valpool, p_model_val, w=w_valpool, n_bins=args.n_bins)
    met_model_test = evaluate(y_test, p_model_test, w=w_test, n_bins=args.n_bins)

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
    metrics_df = pd.DataFrame(metrics_rows)

    print("\n=== FINAL FIT SUMMARY ===")
    print(f"best_C (rolling avg): {best_C:g}")
    print(f"calibration: {args.calibrate}")
    if model_kind_val != model_kind:
        print(f"val_pool_model_kind: {model_kind_val} (OOS)")
    print("\n=== METRICS (VAL_POOL + TEST) ===")
    # Sort for readability
    show_cols = [c for c in ["model", "split", "logloss", "brier", "ece", "n", "weight_sum",
                            "baseline_logloss", "delta_model_minus_baseline"] if c in metrics_df.columns]
    print(metrics_df[show_cols].to_string(index=False))

    # Coefficients inspection
    print_numeric_coeffs_only(pipe_final, requested)
    print_top_ticker_adjustments(pipe_final, args.ticker_col, top_k=10)

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

    # Save model objects
    joblib.dump(pipe_final, out_dir / "base_pipeline.joblib")
    if platt_cal is not None:
        joblib.dump(platt_cal, out_dir / "platt_calibrator.joblib")
    joblib.dump(final_bundle, out_dir / "final_model.joblib")

    # Metadata
    meta = {
        "script": Path(__file__).name,
        "csv": str(Path(args.csv)),
        "out_dir": str(out_dir),
        "target_col": target_col,
        "week_col": args.week_col,
        "ticker_col": args.ticker_col,
        "categorical_features_requested": parse_categorical_list(args.categorical_features),
        "categorical_features_used": cat_features,
        "weight_col": weight_col,
        "features_requested": parse_feature_list(args.features),
        "features_used_final": requested,
        "add_interactions": bool(args.add_interactions),
        "calibration_requested": calibration_requested,
        "calibration_used": args.calibrate,
        "model_kind_val_pool": model_kind_val,
        "model_kind_test": model_kind,
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
        },
        "fit_weights": {
            "fit_weight_renorm": args.fit_weight_renorm,
            "train_decay_half_life_weeks": float(args.train_decay_half_life_weeks),
            "trainfit_n": int(len(w_trainfit_raw)),
            "trainfit_sum_raw": float(np.sum(w_trainfit_raw)),
            "trainfit_sum_fit": float(np.sum(w_fit)),
            "calib_n": int(len(w_calib_raw)),
            "calib_sum_raw": float(np.sum(w_calib_raw)),
        },
        "random_state": int(args.random_state),
        "n_bins": int(args.n_bins),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved artifacts to: {out_dir.resolve()}")
    print("Files: metrics.csv, rolling_windows.csv, rolling_summary.csv, reliability_bins.csv, base_pipeline.joblib, final_model.joblib, metadata.json")


if __name__ == "__main__":
    main()
