from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1e-6
FOUNDATION_LABEL = "0_FOUNDATION"

FORBIDDEN_NUMERIC_FEATURE_PATTERNS = [
    r".*fallback.*",
    r".*_used.*",
    r".*inside_frac.*",
    r".*drop_.*",
    r".*band_.*",
    r".*n_calls_.*",
    r".*calls_k_.*",
    r".*deltaK.*",
    r".*split_events.*",
]
FORBIDDEN_NUMERIC_FEATURE_EXCEPTIONS = {
    "fallback_any",
    "had_fallback",
    "had_intrinsic_drop",
    "had_band_clip",
}
FORBIDDEN_NUMERIC_FEATURE_REGEX = [
    re.compile(pattern, re.IGNORECASE) for pattern in FORBIDDEN_NUMERIC_FEATURE_PATTERNS
]


def _logit(p: np.ndarray) -> np.ndarray:
    p_clipped = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return np.log(p_clipped / (1.0 - p_clipped))


class TickerInteractionTransformer(BaseEstimator, TransformerMixin):
    """Creates ticker-based sparse interactions with another numeric column."""

    def __init__(
        self,
        *,
        ticker_col: str,
        x_col: str,
        scope_tickers: Optional[List[str]] = None,
    ) -> None:
        self.ticker_col = ticker_col
        self.x_col = x_col
        self.scope_tickers = [s for s in scope_tickers] if scope_tickers else None
        self.encoder = OneHotEncoder(handle_unknown="ignore", drop="first")
        self._keep_mask: Optional[np.ndarray] = None
        self._feature_names_out: List[str] = []

    def _split(self, X: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(X, pd.DataFrame):
            tickers = X[self.ticker_col].astype("string").fillna("UNKNOWN").to_numpy()
            x = pd.to_numeric(X[self.x_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            tickers = pd.Series(X[:, 0]).astype("string").fillna("UNKNOWN").to_numpy()
            x = pd.to_numeric(pd.Series(X[:, 1]), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return tickers, x

    def fit(self, X: Any, y: Optional[np.ndarray] = None) -> "TickerInteractionTransformer":
        tickers, _ = self._split(X)
        self.encoder.fit(tickers.reshape(-1, 1))
        names = list(self.encoder.get_feature_names_out([self.ticker_col]))
        if self.scope_tickers is None:
            self._keep_mask = None
            self._feature_names_out = names
            return self

        scope = {str(s).strip().upper() for s in self.scope_tickers}
        keep = []
        for name in names:
            raw = name
            if raw.startswith(f"{self.ticker_col}_"):
                raw = raw[len(self.ticker_col) + 1 :]
            keep.append(str(raw).strip().upper() in scope)
        self._keep_mask = np.array(keep, dtype=bool)
        self._feature_names_out = [n for n, k in zip(names, keep) if k]
        return self

    def transform(self, X: Any) -> sparse.csr_matrix:
        tickers, x = self._split(X)
        mat = self.encoder.transform(tickers.reshape(-1, 1))
        if self._keep_mask is not None:
            if not np.any(self._keep_mask):
                return sparse.csr_matrix((len(tickers), 0))
            mat = mat[:, self._keep_mask]
        mat = mat.multiply(x.reshape(-1, 1))
        return mat

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> np.ndarray:
        return np.array(self._feature_names_out, dtype=object)


class LogisticRegressionOffset(BaseEstimator):
    """Logistic regression that accepts an external logit offset."""

    def __init__(self, *, C: float = 1.0, max_iter: int = 2000, tol: float = 1e-6) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.n_iter_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> "LogisticRegressionOffset":
        y = np.asarray(y, dtype=float).ravel()
        if np.any(~np.isin(y, [0.0, 1.0])):
            raise ValueError("LogisticRegressionOffset expects binary y in {0,1}.")

        Xs = X.tocsr() if sparse.issparse(X) else np.asarray(X, dtype=float)
        n_samples, n_features = Xs.shape
        self.n_features_in_ = int(n_features)

        if sample_weight is None:
            sw = np.ones(n_samples, dtype=float)
        else:
            sw = np.asarray(sample_weight, dtype=float).ravel()
            if len(sw) != n_samples:
                raise ValueError("sample_weight length mismatch.")

        if offset is None:
            off = np.zeros(n_samples, dtype=float)
        else:
            off = np.asarray(offset, dtype=float).ravel()
            if len(off) != n_samples:
                raise ValueError("offset length mismatch.")

        alpha = 1.0 / self.C if self.C > 0 else 0.0

        def _loss_grad(wb: np.ndarray) -> Tuple[float, np.ndarray]:
            w = wb[:-1]
            b = wb[-1]
            eta = (Xs @ w) + b + off
            loss = float(np.sum(sw * (np.logaddexp(0.0, eta) - y * eta)))
            if alpha > 0:
                loss += 0.5 * alpha * float(np.dot(w, w))
            p = expit(eta)
            diff = (p - y) * sw
            if sparse.issparse(Xs):
                grad_w = (Xs.T @ diff).ravel()
            else:
                grad_w = Xs.T.dot(diff)
            if alpha > 0:
                grad_w += alpha * w
            grad_b = float(np.sum(diff))
            return loss, np.concatenate([grad_w, [grad_b]])

        w0 = np.zeros(self.n_features_in_ + 1, dtype=float)
        res = minimize(
            _loss_grad,
            w0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )
        if not res.success:
            print(f"[WARN] LogisticRegressionOffset did not fully converge: {res.message}")

        wb = res.x
        self.coef_ = wb[:-1].reshape(1, -1)
        self.intercept_ = np.array([wb[-1]], dtype=float)
        self.n_iter_ = np.array([res.nit], dtype=int)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if sparse.issparse(X):
            Xs = X.tocsr()
            return (Xs @ self.coef_.ravel()) + float(self.intercept_[0])
        Xs = np.asarray(X, dtype=float)
        return Xs.dot(self.coef_.ravel()) + float(self.intercept_[0])

    def predict_proba(self, X: np.ndarray, *, offset: Optional[np.ndarray] = None) -> np.ndarray:
        if offset is None:
            offset = np.zeros(X.shape[0], dtype=float)
        logits = self.decision_function(X) + np.asarray(offset, dtype=float).ravel()
        p1 = expit(logits)
        return np.column_stack([1.0 - p1, p1])


def make_preprocessor(
    *,
    numeric_features: List[str],
    categorical_features: List[str],
    interaction_ticker_col: Optional[str],
    interaction_x_col: str,
    interaction_scope_tickers: Optional[List[str]],
    enable_interactions: bool,
) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    transformers = [("num", num_pipe, numeric_features)]
    if categorical_features:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ])
        transformers.append(("cat", cat_pipe, categorical_features))
    if enable_interactions and interaction_ticker_col:
        inter = TickerInteractionTransformer(
            ticker_col=interaction_ticker_col,
            x_col=interaction_x_col,
            scope_tickers=interaction_scope_tickers,
        )
        inter_pipe = Pipeline(steps=[
            ("inter", inter),
            ("scaler", StandardScaler(with_mean=False)),
        ])
        transformers.append(("ticker_x", inter_pipe, [interaction_ticker_col, interaction_x_col]))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


def make_pipeline(
    *,
    numeric_features: List[str],
    categorical_features: List[str],
    interaction_ticker_col: Optional[str],
    interaction_x_col: str,
    interaction_scope_tickers: Optional[List[str]],
    enable_interactions: bool,
    C: float,
    random_state: int,
) -> Pipeline:
    pre = make_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        interaction_ticker_col=interaction_ticker_col,
        interaction_x_col=interaction_x_col,
        interaction_scope_tickers=interaction_scope_tickers,
        enable_interactions=enable_interactions,
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


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def resolve_moneyness_column(df: pd.DataFrame) -> Optional[str]:
    """
    Resolve which moneyness column to use for interactions.

    Prefer log_m_fwd if present; else fallback to log_m if present.
    Returns None if neither exists.
    """
    if "log_m_fwd" in df.columns:
        valid = pd.to_numeric(df["log_m_fwd"], errors="coerce")
        if valid.notna().any():
            return "log_m_fwd"
    if "log_m" in df.columns:
        valid = pd.to_numeric(df["log_m"], errors="coerce")
        if valid.notna().any():
            return "log_m"
    return None


def build_chain_group_id(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Build a chain snapshot group identifier for reweighting.

    Uses: ticker + asof_date + expiry_date (preferred)
    Fallback: ticker + week_friday + T_days if asof_date not available

    Returns None if required columns are missing.
    """
    # Prefer: ticker + asof_date + expiry_date
    if "ticker" in df.columns and "asof_date" in df.columns and "expiry_date" in df.columns:
        ticker = df["ticker"].astype(str).fillna("UNKNOWN")
        asof = pd.to_datetime(df["asof_date"], errors="coerce").astype(str).fillna("NaT")
        expiry = pd.to_datetime(df["expiry_date"], errors="coerce").astype(str).fillna("NaT")
        return ticker + "|" + asof + "|" + expiry

    # Fallback: ticker + week_friday + T_days
    if "ticker" in df.columns and "week_friday" in df.columns and "T_days" in df.columns:
        ticker = df["ticker"].astype(str).fillna("UNKNOWN")
        week = pd.to_datetime(df["week_friday"], errors="coerce").astype(str).fillna("NaT")
        tdays = pd.to_numeric(df["T_days"], errors="coerce").fillna(-1).astype(int).astype(str)
        return ticker + "|" + week + "|T" + tdays

    # Cannot build group id
    return None


def apply_group_reweight(
    weights: np.ndarray,
    group_id: pd.Series,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply group reweighting to sample weights within the masked subset.

    Each group in mask=True receives equal total weight.
    Weights outside mask are unchanged.

    Args:
        weights: Original sample weights (length N)
        group_id: Group identifier series (length N)
        mask: Boolean mask indicating which rows to reweight (length N)

    Returns:
        Reweighted array of same length as weights
    """
    weights = np.asarray(weights, dtype=float).copy()

    if not np.any(mask):
        return weights

    # Get groups within mask
    masked_groups = group_id[mask]
    masked_weights = weights[mask]

    # Compute group totals
    group_totals = pd.Series(masked_weights, index=masked_groups.values).groupby(level=0).sum()

    # Reweight: divide each weight by its group total
    # This makes each group contribute total weight = number of groups
    group_total_map = group_totals.to_dict()

    reweighted = weights.copy()
    for i, (g, w) in enumerate(zip(masked_groups, masked_weights)):
        total = group_total_map[g]
        if total > 0:
            reweighted[np.where(mask)[0][i]] = w / total

    # Renormalize to keep mean weight ~ 1 within mask
    mean_before = float(np.mean(weights[mask]))
    mean_after = float(np.mean(reweighted[mask]))
    if mean_after > 0 and mean_before > 0:
        reweighted[mask] *= (mean_before / mean_after)

    return reweighted


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

    if "log_m" in requested_features and "log_m" not in df.columns:
        if ("K" in df.columns) and ("S_asof_close" in df.columns):
            K = _numeric_series(df, "K").to_numpy(dtype=float)
            S0 = _numeric_series(df, "S_asof_close").to_numpy(dtype=float)
            df["log_m"] = np.log(np.clip(K, 1e-12, None) / np.clip(S0, 1e-12, None))

    if "abs_log_m" in requested_features and "abs_log_m" not in df.columns:
        if "log_m" in df.columns:
            df["abs_log_m"] = pd.to_numeric(df["log_m"], errors="coerce").abs()

    need_forward = any(
        f in requested_features
        for f in ["forward_price", "log_m_fwd", "abs_log_m_fwd", "log_m_fwd_over_volT", "abs_log_m_fwd_over_volT"]
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

    if "log_T_days" in requested_features:
        T_days = _numeric_series(df, "T_days")
        df["log_T_days"] = np.log1p(T_days.clip(lower=0))

    need_sqrt_T = any(
        f in requested_features
        for f in [
            "sqrt_T_years",
            "rv20_sqrtT",
            "log_m_over_volT",
            "abs_log_m_over_volT",
            "log_m_fwd_over_volT",
            "abs_log_m_fwd_over_volT",
        ]
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
        df["fallback_any"] = ((a != 0) | (e != 0)).astype(float)

    if "had_fallback" in requested_features:
        a = _numeric_series(df, "asof_fallback_days").fillna(0)
        e = _numeric_series(df, "expiry_fallback_days").fillna(0)
        df["had_fallback"] = ((a != 0) | (e != 0)).astype(float)

    if "had_intrinsic_drop" in requested_features:
        if "drop_intrinsic_frac" in df.columns:
            src = _numeric_series(df, "drop_intrinsic_frac")
        else:
            src = _safe_divide(_numeric_series(df, "dropped_intrinsic"), _numeric_series(df, "n_chain_raw"))
        df["had_intrinsic_drop"] = (src.fillna(0) > 0).astype(float)

    if "had_band_clip" in requested_features:
        if "band_inside_frac" in df.columns:
            src = _numeric_series(df, "band_inside_frac")
        else:
            src = _safe_divide(_numeric_series(df, "n_band_inside"), _numeric_series(df, "n_band_raw"))
        df["had_band_clip"] = (src.fillna(1.0) < 1.0).astype(float)

    if "prn_raw_gap" in requested_features:
        df["prn_raw_gap"] = _numeric_series(df, "pRN") - _numeric_series(df, "pRN_raw")

    if "x_prn_x_tdays" in requested_features:
        df["x_prn_x_tdays"] = _numeric_series(df, "x_logit_prn") * _numeric_series(df, "T_days")

    if "x_prn_x_rv20" in requested_features:
        df["x_prn_x_rv20"] = _numeric_series(df, "x_logit_prn") * _numeric_series(df, "rv20")

    if "x_prn_x_logm" in requested_features:
        df["x_prn_x_logm"] = _numeric_series(df, "x_logit_prn") * _numeric_series(df, "log_m")

    # x_m and x_abs_m: moneyness interaction features
    if "x_m" in requested_features or "x_abs_m" in requested_features:
        m_col = resolve_moneyness_column(df)
        if m_col:
            x_logit = _numeric_series(df, "x_logit_prn")
            m_val = _numeric_series(df, m_col)
            if "x_m" in requested_features:
                df["x_m"] = x_logit * m_val
            if "x_abs_m" in requested_features:
                df["x_abs_m"] = x_logit * m_val.abs()

    return df


def normalize_ticker(value: Any) -> str:
    return str(value).strip().upper()


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def filter_forbidden_features(feature_names: List[str]) -> Tuple[List[str], List[str]]:
    filtered: List[str] = []
    removed: List[str] = []
    for feat in feature_names:
        if feat in FORBIDDEN_NUMERIC_FEATURE_EXCEPTIONS:
            filtered.append(feat)
            continue
        if any(regex.search(feat) for regex in FORBIDDEN_NUMERIC_FEATURE_REGEX):
            removed.append(feat)
            continue
        filtered.append(feat)
    return dedupe_preserve_order(filtered), sorted(set(removed))


def find_near_constant_features(
    df: pd.DataFrame,
    feature_names: List[str],
    *,
    train_fit_mask: np.ndarray,
    var_eps: float = 1e-12,
) -> Tuple[List[str], Dict[str, float]]:
    dropped: List[str] = []
    variances: Dict[str, float] = {}
    if train_fit_mask is None or len(feature_names) == 0:
        return dropped, variances
    for feat in feature_names:
        values = pd.to_numeric(df.loc[train_fit_mask, feat], errors="coerce").to_numpy(dtype=float)
        if values.size == 0:
            dropped.append(feat)
            variances[feat] = float("nan")
            continue
        var = float(np.nanvar(values))
        variances[feat] = var
        if not np.isfinite(var) or var < var_eps:
            dropped.append(feat)
    return dropped, variances


def scope_tickers_by_support(
    df: pd.DataFrame,
    *,
    ticker_col: str,
    train_fit_mask: np.ndarray,
    min_support: int,
    out_col: str = "_ticker_scoped",
    other_label: str = "OTHER",
    preserve: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, int]]:
    if min_support <= 1:
        df[out_col] = df[ticker_col].map(normalize_ticker)
        return out_col, {
            "min_support": int(min_support),
            "train_unique": int(df.loc[train_fit_mask, ticker_col].nunique()),
            "kept_unique": int(df.loc[train_fit_mask, ticker_col].nunique()),
            "mapped_train_rows": 0,
        }
    preserve_set = {normalize_ticker(x) for x in (preserve or [])}
    tickers = df.loc[train_fit_mask, ticker_col].map(normalize_ticker)
    counts = tickers.value_counts()
    keep = set(counts[counts >= min_support].index)
    keep |= preserve_set
    df[out_col] = df[ticker_col].map(normalize_ticker)
    df[out_col] = df[out_col].where(df[out_col].isin(keep), other_label)
    mapped_train_rows = int((~tickers.isin(keep)).sum())
    return out_col, {
        "min_support": int(min_support),
        "train_unique": int(counts.shape[0]),
        "kept_unique": int(len(keep)),
        "mapped_train_rows": mapped_train_rows,
    }


def tickers_meeting_support(
    df: pd.DataFrame,
    *,
    ticker_col: str,
    train_fit_mask: np.ndarray,
    min_support: int,
    preserve: Optional[List[str]] = None,
) -> List[str]:
    if min_support <= 1:
        return sorted({normalize_ticker(x) for x in df.loc[train_fit_mask, ticker_col].tolist()})
    preserve_set = {normalize_ticker(x) for x in (preserve or [])}
    tickers = df.loc[train_fit_mask, ticker_col].map(normalize_ticker)
    counts = tickers.value_counts()
    keep = set(counts[counts >= min_support].index)
    keep |= preserve_set
    return sorted(keep)


def ece_equal_mass(
    y: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if sample_weight is None:
        w = np.ones_like(p, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)

    total_w = float(np.sum(w))
    if not np.isfinite(total_w) or total_w <= 0:
        return float("nan")

    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    w_sorted = w[order]

    target = total_w / float(n_bins)
    ece = 0.0
    start = 0
    n = len(p_sorted)
    for b in range(n_bins):
        if start >= n:
            break
        cum_w = 0.0
        end = start
        while end < n and (cum_w + w_sorted[end] <= target or b == n_bins - 1):
            cum_w += w_sorted[end]
            end += 1
            if b < n_bins - 1 and cum_w >= target:
                break
        if end == start:
            end = min(start + 1, n)
            cum_w = float(np.sum(w_sorted[start:end]))
        wb = w_sorted[start:end]
        swb = float(np.sum(wb))
        if swb > 0:
            avg_p = float(np.sum(wb * p_sorted[start:end]) / swb)
            avg_y = float(np.sum(wb * y_sorted[start:end]) / swb)
            ece += (swb / total_w) * abs(avg_p - avg_y)
        start = end

    return float(ece)


def load_schema_contract(config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Load the pipeline schema contract from config directory."""
    if config_dir is None:
        # Try to find config directory relative to this file
        import os
        script_dir = Path(__file__).resolve().parent
        config_dir = script_dir / "config"
        if not config_dir.exists():
            # Try parent directory
            config_dir = script_dir.parent / "config"

    contract_path = config_dir / "pipeline_schema_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Schema contract not found at {contract_path}")

    import json
    with open(contract_path, "r") as f:
        return json.load(f)


def validate_feature_availability(
    requested_features: List[str],
    *,
    warn_on_nan_prone: bool = True,
    config_dir: Optional[Path] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Validate that requested features can be computed from baseline schema.

    Returns:
        (valid_features, nan_prone_features, unknown_features)

    Raises:
        ValueError if any requested feature is unknown and cannot be computed
    """
    contract = load_schema_contract(config_dir)

    guaranteed_derived = set(contract["guaranteed_derived_features"]["features"])
    conditional_derived = set(contract["conditional_derived_features"]["features"].keys())
    baseline_columns = set(contract["guaranteed_baseline_columns"]["columns"])
    nan_prone_columns = set(contract["known_limitations"]["columns"])

    valid = []
    nan_prone = []
    unknown = []

    for feat in requested_features:
        # Check if it's a baseline column (directly available)
        if feat in baseline_columns:
            if feat in nan_prone_columns:
                nan_prone.append(feat)
            else:
                valid.append(feat)
        # Check if it's a guaranteed derived feature
        elif feat in guaranteed_derived:
            valid.append(feat)
        # Check if it's a conditional derived feature
        elif feat in conditional_derived:
            # Check if its dependencies are nan-prone
            deps = contract["conditional_derived_features"]["features"][feat].get("requires", [])
            if any(dep in nan_prone_columns for dep in deps):
                nan_prone.append(feat)
            else:
                valid.append(feat)
        else:
            unknown.append(feat)

    if unknown:
        raise ValueError(
            f"Unknown features requested that cannot be computed from baseline schema: {unknown}\n"
            f"Please check config/pipeline_schema_contract.json for available features."
        )

    if warn_on_nan_prone and nan_prone:
        print(f"\n[WARNING] Features depending on NaN-prone columns: {nan_prone}")
        print(f"These columns are not available in snapshot-only mode: {nan_prone_columns}")
        print(f"Predictions will have NaN for these features unless historical data is joined.")
        print(f"Consider training a snapshot-only model variant without these features.\n")

    return valid, nan_prone, unknown


@dataclass
class FinalModelBundle:
    kind: str
    mode: str
    numeric_features: List[str]
    categorical_features: List[str]
    ticker_col: str
    ticker_feature_col: Optional[str] = None
    interaction_ticker_col: Optional[str] = None
    foundation_tickers: Optional[List[str]] = None
    foundation_label: str = FOUNDATION_LABEL
    ticker_intercepts: str = "none"
    ticker_x_interactions: bool = False
    ticker_support: Optional[List[str]] = None
    interaction_ticker_support: Optional[List[str]] = None
    ticker_other_label: str = "OTHER"
    base_pipeline: Optional[Pipeline] = None
    platt_calibrator: Optional[LogisticRegression] = None
    stage1_pipeline: Optional[Pipeline] = None
    stage2_preprocessor: Optional[ColumnTransformer] = None
    stage2_model: Optional[LogisticRegressionOffset] = None
    stage1_categorical_features: Optional[List[str]] = None
    stage2_categorical_features: Optional[List[str]] = None
    eps: float = EPS

    def predict_proba_from_df(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "baseline_pRN":
            if "pRN" not in df.columns:
                raise ValueError("baseline_pRN requires pRN column.")
            p = pd.to_numeric(df["pRN"], errors="coerce").to_numpy(dtype=float)
            return np.clip(p, self.eps, 1.0 - self.eps)

        data = df.copy()
        if self.ticker_col not in data.columns:
            raise ValueError(f"Missing ticker column: {self.ticker_col}")
        data[self.ticker_col] = data[self.ticker_col].astype("string").fillna("UNKNOWN")

        if "x_logit_prn" in self.numeric_features or (self.interaction_ticker_col is not None):
            if "pRN" not in data.columns:
                raise ValueError("Missing pRN required to compute x_logit_prn.")
            data["pRN"] = pd.to_numeric(data["pRN"], errors="coerce").clip(self.eps, 1.0 - self.eps)
            data["x_logit_prn"] = _logit(data["pRN"].to_numpy(dtype=float))

        foundation_set = {normalize_ticker(t) for t in (self.foundation_tickers or [])}
        if foundation_set:
            is_foundation = data[self.ticker_col].map(normalize_ticker).isin(foundation_set).to_numpy()
        else:
            is_foundation = np.zeros(len(data), dtype=bool)

        if self.ticker_feature_col and self.ticker_feature_col not in data.columns:
            if self.ticker_intercepts == "non_foundation" and foundation_set:
                base = np.where(is_foundation, self.foundation_label, data[self.ticker_col])
            else:
                base = data[self.ticker_col]
            if self.ticker_support:
                support = {normalize_ticker(t) for t in self.ticker_support}
                base_norm = pd.Series(base).map(normalize_ticker)
                scoped = np.where(base_norm.isin(support), base, self.ticker_other_label)
                data[self.ticker_feature_col] = scoped
            else:
                data[self.ticker_feature_col] = base

        if self.interaction_ticker_col and self.interaction_ticker_col not in data.columns:
            if self.ticker_intercepts == "non_foundation" and foundation_set:
                base = np.where(is_foundation, self.foundation_label, data[self.ticker_col])
            else:
                base = data[self.ticker_col]
            if self.interaction_ticker_support:
                support = {normalize_ticker(t) for t in self.interaction_ticker_support}
                base_norm = pd.Series(base).map(normalize_ticker)
                scoped = np.where(base_norm.isin(support), base, self.ticker_other_label)
                data[self.interaction_ticker_col] = scoped
            else:
                data[self.interaction_ticker_col] = base

        if self.mode in ("baseline", "pooled"):
            if self.base_pipeline is None:
                raise ValueError("Missing base_pipeline.")

            extra_cols: List[str] = []
            if self.interaction_ticker_col:
                extra_cols.append(self.interaction_ticker_col)
            cols = dedupe_preserve_order(self.numeric_features + list(self.categorical_features) + extra_cols)
            X = data[cols].copy()
            for c in self.numeric_features:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            for c in self.categorical_features + extra_cols:
                if c in X.columns:
                    X[c] = X[c].astype("string").fillna("UNKNOWN")
            p = self.base_pipeline.predict_proba(X)[:, 1]
            if self.kind == "logit+platt":
                if self.platt_calibrator is None:
                    raise ValueError("Missing platt_calibrator.")
                logits = self.base_pipeline.decision_function(X)
                p = apply_platt(self.platt_calibrator, logits)
            return np.clip(p.astype(float), self.eps, 1.0 - self.eps)

        if self.mode == "two_stage":
            if self.stage1_pipeline is None or self.stage2_preprocessor is None or self.stage2_model is None:
                raise ValueError("Missing two-stage components.")

            stage1_cat = self.stage1_categorical_features or []
            stage2_cat = self.stage2_categorical_features or []
            stage1_cols = dedupe_preserve_order(self.numeric_features + list(stage1_cat))
            extra_cols = []
            if self.interaction_ticker_col:
                extra_cols.append(self.interaction_ticker_col)
            stage2_cols = dedupe_preserve_order(self.numeric_features + list(stage2_cat) + extra_cols)

            X1 = data[stage1_cols].copy()
            for c in self.numeric_features:
                X1[c] = pd.to_numeric(X1[c], errors="coerce")
            for c in stage1_cat:
                X1[c] = X1[c].astype("string").fillna("UNKNOWN")

            p_base = self.stage1_pipeline.predict_proba(X1)[:, 1]
            logits = _logit(p_base)
            if np.any(~is_foundation):
                X2 = data[stage2_cols].copy()
                for c in self.numeric_features:
                    X2[c] = pd.to_numeric(X2[c], errors="coerce")
                for c in stage2_cat + extra_cols:
                    if c in X2.columns:
                        X2[c] = X2[c].astype("string").fillna("UNKNOWN")
                X2_nf = X2.loc[~is_foundation]
                z2 = self.stage2_preprocessor.transform(X2_nf)
                g = self.stage2_model.decision_function(z2)
                logits[~is_foundation] = logits[~is_foundation] + g
            p = expit(logits)
            if self.kind == "two_stage+platt":
                if self.platt_calibrator is None:
                    raise ValueError("Missing platt_calibrator.")
                p = apply_platt(self.platt_calibrator, logits)
            return np.clip(p.astype(float), self.eps, 1.0 - self.eps)

        raise ValueError(f"Unknown mode: {self.mode}")


def validate_snapshot_schema(
    df: pd.DataFrame,
    *,
    required_features: Optional[List[str]] = None,
    config_dir: Optional[Path] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validate that a snapshot DataFrame conforms to the guaranteed baseline schema.

    Args:
        df: Snapshot dataframe to validate
        required_features: Optional list of features required by model (for additional checking)
        config_dir: Path to config directory
        strict: If True, raise ValueError on missing baseline columns. If False, return diagnostics.

    Returns:
        Dictionary with validation results:
        {
            "valid": bool,
            "missing_baseline": List[str],
            "nan_fraction": Dict[str, float],
            "warnings": List[str],
            "errors": List[str]
        }
    """
    contract = load_schema_contract(config_dir)
    baseline_cols = contract["guaranteed_baseline_columns"]["columns"]
    nan_prone_cols = set(contract["known_limitations"]["columns"])

    missing_baseline = [col for col in baseline_cols if col not in df.columns]
    warnings = []
    errors = []

    if missing_baseline:
        msg = f"Missing guaranteed baseline columns: {missing_baseline}"
        if strict:
            raise ValueError(msg)
        errors.append(msg)

    # Check NaN fractions for key columns
    nan_fraction = {}
    for col in baseline_cols:
        if col in df.columns:
            frac = df[col].isna().mean()
            nan_fraction[col] = float(frac)
            if frac > 0.9 and col not in nan_prone_cols:
                warnings.append(f"Column '{col}' is >90% NaN ({frac:.1%})")

    # Check if required features can be computed
    if required_features:
        for feat in required_features:
            if feat in df.columns:
                frac = df[feat].isna().mean()
                if frac > 0.5:
                    warnings.append(f"Required feature '{feat}' is >{frac:.1%} NaN")
            else:
                # Check if it can be computed
                guaranteed = set(contract["guaranteed_derived_features"]["features"])
                conditional = set(contract["conditional_derived_features"]["features"].keys())
                if feat not in guaranteed and feat not in conditional and feat not in baseline_cols:
                    errors.append(f"Required feature '{feat}' cannot be computed from baseline schema")

    valid = len(errors) == 0 and len(missing_baseline) == 0

    return {
        "valid": valid,
        "missing_baseline": missing_baseline,
        "nan_fraction": nan_fraction,
        "warnings": warnings,
        "errors": errors,
    }
