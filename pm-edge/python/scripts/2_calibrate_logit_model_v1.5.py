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
    ticker_col: str,
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

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, [ticker_col]),
        ],
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

        X = df[self.numeric_features + [self.ticker_col]].copy()
        for c in self.numeric_features:
            X[c] = pd.to_numeric(X[c], errors="coerce")

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


def feature_presence_report(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    rows = []
    for c in feats:
        present = c in df.columns
        dtype = str(df[c].dtype) if present else "MISSING"
        nn = int(df[c].notna().sum()) if present else 0
        rows.append({"feature": c, "present": present, "dtype": dtype, "non_missing": nn})
    return pd.DataFrame(rows)


def ensure_engineered_features(df: pd.DataFrame, requested_features: List[str]) -> pd.DataFrame:
    df = df.copy()

    for c in ["T_days", "rv20", "abs_log_m", "x_logit_prn"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "x_prn_x_tdays" in requested_features and "x_prn_x_tdays" not in df.columns:
        df["x_prn_x_tdays"] = df.get("x_logit_prn", np.nan) * df.get("T_days", np.nan)

    if "x_prn_x_rv20" in requested_features and "x_prn_x_rv20" not in df.columns:
        df["x_prn_x_rv20"] = df.get("x_logit_prn", np.nan) * df.get("rv20", np.nan)

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Given sorted weeks used for training, take last fraction as calib weeks."""
    w = pd.to_datetime(pd.Series(train_weeks_sorted)).sort_values().to_numpy()
    n = len(w)
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

    ap.add_argument("--features", default="x_logit_prn,T_days,abs_log_m,rv20")
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
    df["x_logit_prn"] = _logit(df["pRN"].to_numpy(dtype=float))

    # Features
    requested = parse_feature_list(args.features)
    if args.add_interactions:
        if "T_days" in df.columns and "x_prn_x_tdays" not in requested:
            requested.append("x_prn_x_tdays")
        if "rv20" in df.columns and "x_prn_x_rv20" not in requested:
            requested.append("x_prn_x_rv20")

    df = ensure_engineered_features(df, requested)

    pres = feature_presence_report(df, requested)
    print("\n=== FEATURE PRESENCE (numeric) ===")
    print(pres.to_string(index=False))
    missing = pres.loc[~pres["present"], "feature"].tolist()
    if missing:
        raise ValueError(f"Requested features missing from CSV: {missing}")

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
        Xsub = df.loc[mask, requested + [args.ticker_col]].copy()
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
            window_weeks = pd.date_range(wspec.window_start, wspec.window_end, freq="7D")  # not used for filtering
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
            train_fit_weeks, calib_weeks = split_train_calib_weeks(train_weeks_used, args.calib_frac_of_train)

            m_trainfit = m_train & mask_weeks(df, args.week_col, train_fit_weeks)
            m_calib = m_train & mask_weeks(df, args.week_col, calib_weeks)

            X_trainfit, y_trainfit, w_trainfit_raw, wk_trainfit = subset(m_trainfit)
            X_calib, y_calib, w_calib_raw, wk_calib = subset(m_calib)
            X_val, y_val, w_val, wk_val = subset(m_val)

            # Fit weights: decay + renorm (fit-time only)
            w_fit = w_trainfit_raw.copy()
            if args.train_decay_half_life_weeks and args.train_decay_half_life_weeks > 0:
                w_fit = apply_recency_decay(w_fit, wk_trainfit, half_life_weeks=args.train_decay_half_life_weeks)
            if args.fit_weight_renorm == "mean1":
                w_fit = renorm_fit_weights_mean1(w_fit)

            pipe = make_pipeline(
                numeric_features=requested,
                ticker_col=args.ticker_col,
                C=C,
                random_state=args.random_state,
            )
            pipe.fit(X_trainfit, y_trainfit, clf__sample_weight=w_fit)

            # Predictions
            p_val_model = pipe.predict_proba(X_val)[:, 1]
            p_val_base = baseline_pred(m_val)

            # Optional Platt (fit on CALIB slice only)
            if args.calibrate == "platt":
                if len(y_calib) < 50:
                    # not enough calib; fall back to base
                    pass
                else:
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
    test_start = pd.Timestamp(test_block_weeks[0])

    # Masks for final fit / eval
    m_pretest = mask_weeks(df, args.week_col, pretest_weeks)  # all weeks before TEST
    m_test = mask_weeks(df, args.week_col, test_block_weeks)

    # Train weeks used for final fit: all pretest weeks
    train_weeks_used = np.array(sorted(pd.unique(df.loc[m_pretest, args.week_col])))
    train_fit_weeks, calib_weeks = split_train_calib_weeks(train_weeks_used, args.calib_frac_of_train)

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
        X_trainfit = df.loc[m_trainfit, requested + [args.ticker_col]].copy()
        X_calib = df.loc[m_calib, requested + [args.ticker_col]].copy()
        X_valpool = df.loc[m_val_pool, requested + [args.ticker_col]].copy()
        X_test = df.loc[m_test, requested + [args.ticker_col]].copy()

        # Ensure numeric types remain clean
        for c in requested:
            X_trainfit[c] = pd.to_numeric(X_trainfit[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_calib[c] = pd.to_numeric(X_calib[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_valpool[c] = pd.to_numeric(X_valpool[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            X_test[c] = pd.to_numeric(X_test[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    print("\n=== FINAL FEATURES USED FOR FIT ===")
    print(requested)

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
        ticker_col=args.ticker_col,
        C=best_C,
        random_state=args.random_state,
    )
    pipe_final.fit(X_trainfit, y_trainfit, clf__sample_weight=w_fit)

    # Optional Platt calibrator (fit ONLY on CALIB)
    platt_cal = None
    if args.calibrate == "platt":
        if len(y_calib) < 50:
            print("[WARN] Not enough CALIB rows for Platt; skipping calibration.")
            args.calibrate = "none"
        else:
            z_cal = pipe_final.decision_function(X_calib)
            w_cal_fit = w_calib_raw.copy()
            if args.fit_weight_renorm == "mean1":
                w_cal_fit = renorm_fit_weights_mean1(w_cal_fit)
            platt_cal = fit_platt_on_logits(z_cal, y_calib, w_cal_fit, random_state=args.random_state)

    # Bundle model for saving/inference
    model_kind = "logit+platt" if (args.calibrate == "platt") else "logit"
    final_bundle = FinalModelBundle(
        kind=model_kind,
        numeric_features=requested,
        ticker_col=args.ticker_col,
        base_pipeline=pipe_final,
        platt_calibrator=platt_cal,
    )

    # -----------------------------
    # Predictions + Metrics (VAL_POOL + TEST)
    # -----------------------------
    p_base_val = df.loc[m_val_pool, "pRN"].to_numpy(dtype=float)
    p_base_test = df.loc[m_test, "pRN"].to_numpy(dtype=float)

    p_model_val = final_bundle.predict_proba_from_df(df.loc[m_val_pool, requested + [args.ticker_col, "pRN"]])
    p_model_test = final_bundle.predict_proba_from_df(df.loc[m_test, requested + [args.ticker_col, "pRN"]])

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
        {"model": model_kind + f"_C={best_C:g}", "split": "val_pool", **met_model_val, "n": int(len(y_valpool)), "weight_sum": float(np.sum(w_valpool))},
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
        reliability_table(y_valpool, p_model_val, n_bins=args.n_bins, w=w_valpool, label=f"val_pool__{model_kind}_C={best_C:g}"),
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
        "weight_col": weight_col,
        "features_requested": parse_feature_list(args.features),
        "features_used_final": requested,
        "add_interactions": bool(args.add_interactions),
        "calibration": args.calibrate,
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


