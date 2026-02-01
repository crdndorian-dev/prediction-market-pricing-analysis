#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

# =============================
# Math helpers
# =============================

EPS = 1e-6


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def softplus(z: np.ndarray) -> np.ndarray:
    # stable softplus: log(1+exp(z))
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def brier_w(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    denom = float(np.sum(w))
    if denom <= 0:
        return float(np.mean((y - p) ** 2))
    return float(np.sum(w * (y - p) ** 2) / denom)


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def logloss_w(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    denom = float(np.sum(w))
    if denom <= 0:
        return logloss(y, p)
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    ce = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(np.sum(w * ce) / denom)


def auc_roc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue
        w = float(np.mean(m))
        e += w * abs(float(np.mean(p[m]) - np.mean(y[m])))
    return float(e)


def ece_w(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total_w = float(np.sum(w))
    if total_w <= 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue
        wb = w[m]
        swb = float(np.sum(wb))
        if swb <= 0:
            continue
        pred = float(np.sum(wb * p[m]) / swb)
        obs = float(np.sum(wb * y[m]) / swb)
        frac = swb / total_w
        e += frac * abs(pred - obs)
    return float(e)


# =============================
# Data helpers
# =============================

def compute_log_moneyness(df: pd.DataFrame) -> np.ndarray:
    K = pd.to_numeric(df["K"], errors="coerce").to_numpy(dtype=float)
    S0 = pd.to_numeric(df["S_asof_close"], errors="coerce").to_numpy(dtype=float)
    out = np.log(np.clip(K, 1e-12, None) / np.clip(S0, 1e-12, None))
    out[~np.isfinite(out)] = 0.0
    return out


def time_split(df: pd.DataFrame, test_weeks: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep your existing "week-based holdout" split.
    NOTE: For your dataset (Mon-Thu snapshots for a given Fri expiry),
    this is acceptable if 'asof_date' is the snapshot date and week identity
    is stable. If you also have 'expiry_date', you may prefer grouping by expiry week.
    """
    df = df.copy()
    df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")
    df = df.dropna(subset=["asof_date"])
    df["week"] = df["asof_date"].dt.to_period("W-MON").astype(str)
    weeks = sorted(df["week"].unique())
    if len(weeks) <= test_weeks:
        return df, df.iloc[0:0].copy()
    test_set = set(weeks[-test_weeks:])
    train = df[~df["week"].isin(test_set)].copy()
    test = df[df["week"].isin(test_set)].copy()
    return train, test


# =============================
# Isotonic post-calibration
# =============================

def fit_isotonic_curve(p: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    ir.fit(p, y)
    return np.asarray(ir.X_thresholds_, dtype=float), np.asarray(ir.y_thresholds_, dtype=float)


def apply_isotonic(p: np.ndarray, pk: np.ndarray, yi: np.ndarray) -> np.ndarray:
    if pk.size == 0:
        return np.clip(p, 0.0, 1.0)
    return np.interp(p, pk, yi, left=yi[0], right=yi[-1])


# =============================
# Parsing helpers
# =============================

def _parse_ticker_list(s: str) -> List[str]:
    if not s:
        return []
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def _parse_ticker_weights(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip().upper()
        try:
            fv = float(v.strip())
        except Exception:
            continue
        if k and np.isfinite(fv) and fv > 0:
            out[k] = fv
    return out


# =============================
# Weighting (FIXED)
# =============================

def build_sample_weights(
    df: pd.DataFrame,
    *,
    weight_col: str,
    focus_tickers: List[str],
    focus_weight: float,
    other_weight: float,
    ticker_weights: Dict[str, float],
) -> np.ndarray:
    n = len(df)
    w = np.ones(n, dtype=float)

    # base weight column
    if weight_col:
        if weight_col in df.columns:
            wc = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)
            wc = np.where(np.isfinite(wc) & (wc > 0), wc, 1.0)
            w *= wc
        else:
            print(f"[Weights] ⚠️ weight_col='{weight_col}' not found; ignoring.")

    # focus scaling
    if focus_tickers:
        ft = set([t.upper() for t in focus_tickers])
        tcol = df["ticker"].astype(str).str.upper().to_numpy()
        mask = np.fromiter((t in ft for t in tcol), dtype=bool, count=n)
        w *= np.where(mask, float(focus_weight), float(other_weight))

    # per-ticker multipliers
    if ticker_weights:
        tcol = df["ticker"].astype(str).str.upper().to_numpy()
        mult = np.ones(n, dtype=float)
        for i, t in enumerate(tcol):
            if t in ticker_weights:
                mult[i] = float(ticker_weights[t])
        w *= mult

    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    return w


def infer_group_keys(df: pd.DataFrame) -> List[str]:
    """
    Try to infer a "chain" identity so multiple strikes sharing the same realized outcome
    don't dominate. Preference: ticker + asof_date + expiry/exp_date/expiration + T_days.
    """
    keys = ["ticker", "asof_date"]
    for cand in ["expiry", "exp_date", "expiration", "expiry_date", "date_expiry"]:
        if cand in df.columns:
            keys.append(cand)
            break
    if "T_days" in df.columns:
        keys.append("T_days")
    return keys


def apply_group_reweight(df: pd.DataFrame, w: np.ndarray, group_keys: List[str]) -> np.ndarray:
    """
    Make each group contribute ~1 total weight by dividing by group total.
    This is the simplest high-ROI fix for row dependence in strike grids.
    """
    if not group_keys:
        return w
    tmp = df[group_keys].copy()
    for c in group_keys:
        tmp[c] = tmp[c].astype(str)
    g = tmp.groupby(group_keys, sort=False).ngroup().to_numpy(dtype=int)
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if w.sum() <= 0:
        return w
    group_sum = np.bincount(g, weights=w)
    denom = group_sum[g]
    denom = np.where(denom > 0, denom, 1.0)
    return w / denom


# =============================
# Features (UPDATED: time-conditioned)
# =============================

def _standardize_train_stats(train_vals: np.ndarray) -> Tuple[float, float]:
    mu = float(np.nanmean(train_vals))
    sd = float(np.nanstd(train_vals))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return mu, sd


def _get_time_col(df: pd.DataFrame, time_col: str) -> np.ndarray:
    """
    Returns a strictly-positive time-to-expiry vector for feature building.
    Supports 'T_years' or 'T_days' (or any numeric column).
    """
    if time_col not in df.columns:
        raise ValueError(f"Missing time_col='{time_col}' in dataframe.")
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    # time must be > 0 for log(time)
    t = np.where(np.isfinite(t), t, np.nan)
    # if there are zeros/negatives, clip to a tiny positive value (better than dropping late)
    t = np.clip(t, 1e-6, None)
    return t


def build_global_features(
    df: pd.DataFrame,
    *,
    use_moneyness: bool,
    use_interaction: bool,
    use_time: bool,
    time_col: str,
    use_xt: bool,
    use_logtime: bool,
    use_xlogt: bool,
    use_hinge: bool,
    hinge_knot: float,
    t_mu: float,
    t_sd: float,
    tlog_mu: float,
    tlog_sd: float,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Global (dense) feature block, includes intercept.

    Core: intercept + x where x=logit(pRN)
    Optional: moneyness, x*m
    Time-conditioned (NEW): t_std, logt_std, and interactions x*t_std and x*logt_std
    """
    x = df["x_logit_prn"].to_numpy(dtype=float)
    feats = [np.ones(len(df), dtype=float), x]
    names = ["intercept", "x"]

    if use_hinge:
        xp = np.maximum(0.0, x - hinge_knot)
        xn = np.maximum(0.0, hinge_knot - x)
        feats += [xp, xn]
        names += [f"x_hinge_pos@{hinge_knot:.3f}", f"x_hinge_neg@{hinge_knot:.3f}"]

    # moneyness
    if use_moneyness:
        m = df["m_logm"].to_numpy(dtype=float)
        feats.append(m)
        names.append("m")
    else:
        m = np.zeros(len(df), dtype=float)

    if use_interaction:
        feats.append(x * m)
        names.append("x*m")

    # time (UPDATED)
    if use_time:
        t = _get_time_col(df, time_col=time_col)
        t_std = (t - t_mu) / t_sd
        feats.append(t_std)
        names.append(f"{time_col}_std")

        if use_xt:
            feats.append(x * t_std)
            names.append(f"x*{time_col}_std")

        if use_logtime:
            tlog = np.log(t + 1e-6)
            tlog_std = (tlog - tlog_mu) / tlog_sd
            feats.append(tlog_std)
            names.append(f"log_{time_col}_std")

            if use_xlogt:
                feats.append(x * tlog_std)
                names.append(f"x*log_{time_col}_std")

    Xg = np.column_stack(feats)
    return Xg, names, x


# =============================
# Hier options + Sparse design
# =============================

@dataclass
class HierOptions:
    random_slope_x: bool = True
    random_slope_m: bool = False
    random_slope_xm: bool = False


def build_ticker_matrix(df: pd.DataFrame, ticker_to_idx: Dict[str, int]) -> sparse.csr_matrix:
    """
    Sparse one-hot matrix for tickers. Unknown tickers map to __UNK__ index.
    """
    tcol = df["ticker"].astype(str).str.upper().to_numpy()
    unk = ticker_to_idx.get("__UNK__", None)
    idx = np.empty(len(tcol), dtype=int)
    for i, t in enumerate(tcol):
        idx[i] = ticker_to_idx.get(t, unk)
    if unk is None:
        raise ValueError("ticker_to_idx must include '__UNK__' for robustness.")
    rows = np.arange(len(tcol), dtype=int)
    data = np.ones(len(tcol), dtype=float)
    T = len(ticker_to_idx)
    return sparse.csr_matrix((data, (rows, idx)), shape=(len(tcol), T))


def build_design_and_penalty_sparse(
    mode: str,
    df: pd.DataFrame,
    ticker_to_idx: Dict[str, int],
    *,
    use_moneyness: bool,
    use_interaction: bool,
    use_time: bool,
    time_col: str,
    use_xt: bool,
    use_logtime: bool,
    use_xlogt: bool,
    use_hinge: bool,
    hinge_knot: float,
    t_mu: float,
    t_sd: float,
    tlog_mu: float,
    tlog_sd: float,
    hier: HierOptions,
    l2_global: float,
    l2_beta_t: float,
    l2_beta_xt: float,
    l2_beta_logt: float,
    l2_beta_xlogt: float,
    l2_hinge: float,
    l2_alpha_dev: float,
    l2_beta_x_dev: float,
    l2_beta_m_dev: float,
    l2_beta_xm_dev: float,
) -> Tuple[sparse.csr_matrix, np.ndarray, List[str]]:
    """
    Returns sparse X, diagonal penalty vector, and column names.
    """
    Xg, gnames, x_vec = build_global_features(
        df,
        use_moneyness=use_moneyness,
        use_interaction=use_interaction,
        use_time=use_time,
        time_col=time_col,
        use_xt=use_xt,
        use_logtime=use_logtime,
        use_xlogt=use_xlogt,
        use_hinge=use_hinge,
        hinge_knot=hinge_knot,
        t_mu=t_mu,
        t_sd=t_sd,
        tlog_mu=tlog_mu,
        tlog_sd=tlog_sd,
    )

    # penalties for global block
    pen_g = np.zeros(Xg.shape[1], dtype=float)
    for j, nm in enumerate(gnames):
        if nm == "intercept":
            pen_g[j] = 0.0
        elif nm == "x":
            pen_g[j] = l2_global
        elif nm.startswith("x_hinge"):
            pen_g[j] = l2_hinge
        elif nm.endswith("_std") and nm.startswith(time_col):
            pen_g[j] = l2_beta_t
        elif nm.startswith("x*") and nm.endswith(f"{time_col}_std"):
            pen_g[j] = l2_beta_xt
        elif nm == f"log_{time_col}_std":
            pen_g[j] = l2_beta_logt
        elif nm == f"x*log_{time_col}_std":
            pen_g[j] = l2_beta_xlogt
        else:
            pen_g[j] = l2_global

    if mode == "global":
        X = sparse.csr_matrix(Xg)
        return X, pen_g, gnames

    Xt = build_ticker_matrix(df, ticker_to_idx)
    Tn = Xt.shape[1]

    if mode == "pooled":
        # pooled ticker intercepts + global features excluding intercept
        X = sparse.hstack([Xt, sparse.csr_matrix(Xg[:, 1:])], format="csr")
        colnames = [f"alpha_{t}" for t in ticker_to_idx.keys()] + gnames[1:]
        pen = np.concatenate([np.full(Tn, 1.0, dtype=float), pen_g[1:]])
        return X, pen, colnames

    # hier:
    parts = [sparse.csr_matrix(Xg), Xt]
    colnames = gnames + [f"alpha_dev_{t}" for t in ticker_to_idx.keys()]
    pen_parts = [pen_g, np.full(Tn, l2_alpha_dev, dtype=float)]

    # slope devs
    if hier.random_slope_x:
        X_dx = Xt.multiply(x_vec[:, None])
        parts.append(X_dx)
        colnames += [f"beta_x_dev_{t}" for t in ticker_to_idx.keys()]
        pen_parts.append(np.full(Tn, l2_beta_x_dev, dtype=float))

    if hier.random_slope_m and use_moneyness:
        m = df["m_logm"].to_numpy(dtype=float)
        X_dm = Xt.multiply(m[:, None])
        parts.append(X_dm)
        colnames += [f"beta_m_dev_{t}" for t in ticker_to_idx.keys()]
        pen_parts.append(np.full(Tn, l2_beta_m_dev, dtype=float))

    if hier.random_slope_xm and use_interaction:
        m = df["m_logm"].to_numpy(dtype=float) if use_moneyness else np.zeros(len(df), dtype=float)
        xm = x_vec * m
        X_dxm = Xt.multiply(xm[:, None])
        parts.append(X_dxm)
        colnames += [f"beta_xm_dev_{t}" for t in ticker_to_idx.keys()]
        pen_parts.append(np.full(Tn, l2_beta_xm_dev, dtype=float))

    X = sparse.hstack(parts, format="csr")
    pen = np.concatenate(pen_parts)
    return X, pen, colnames


# =============================
# Optimization: stable NLL + gradient + L-BFGS-B
# =============================

def nll_and_grad(
    params: np.ndarray,
    X: sparse.csr_matrix,
    y: np.ndarray,
    pen_diag: np.ndarray,
    w: Optional[np.ndarray],
) -> Tuple[float, np.ndarray]:
    z = X @ params
    if w is None:
        data_nll = float(np.sum(softplus(z) - y * z))
        p = sigmoid(z)
        r = (p - y)
        grad = (X.T @ r)
    else:
        ww = np.asarray(w, dtype=float)
        ww = np.where(np.isfinite(ww) & (ww > 0), ww, 0.0)
        data_nll = float(np.sum(ww * (softplus(z) - y * z)))
        p = sigmoid(z)
        r = ww * (p - y)
        grad = (X.T @ r)

    if pen_diag is not None and np.any(pen_diag > 0):
        data_nll += 0.5 * float(np.sum(pen_diag * (params ** 2)))
        grad = np.asarray(grad).reshape(-1) + pen_diag * params
    else:
        grad = np.asarray(grad).reshape(-1)

    return float(data_nll), grad


def fit_mle_lbfgs(
    X: sparse.csr_matrix,
    y: np.ndarray,
    pen_diag: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    x0 = np.zeros(X.shape[1], dtype=float)

    def fun(p: np.ndarray) -> float:
        val, _ = nll_and_grad(p, X, y, pen_diag, w)
        return val

    def jac(p: np.ndarray) -> np.ndarray:
        _, g = nll_and_grad(p, X, y, pen_diag, w)
        return g

    res = minimize(
        fun=fun,
        x0=x0,
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    info = {
        "success": float(bool(res.success)),
        "nll": float(res.fun),
        "iters": float(res.nit),
        "w_sum": float(np.sum(w)) if w is not None else float(len(y)),
    }
    return res.x.astype(float), info


def predict(X: sparse.csr_matrix, params: np.ndarray) -> np.ndarray:
    return sigmoid(X @ params)


# =============================
# Calibration bins
# =============================

def calibration_bins_df(
    y: np.ndarray,
    p: np.ndarray,
    bins: int = 10,
    w: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if w is None:
        w = np.ones_like(p, dtype=float)
    w = np.asarray(w, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)

    rows = []
    for b in range(bins):
        m = idx == b
        if not np.any(m):
            continue
        ww = w[m]
        ws = float(np.sum(ww))
        if ws <= 0:
            continue
        p_mean = float(np.sum(ww * p[m]) / ws)
        y_mean = float(np.sum(ww * y[m]) / ws)
        rows.append({
            "bin": b,
            "n": int(np.sum(m)),
            "w_sum": ws,
            "p_mean": p_mean,
            "y_mean": y_mean,
            "abs_gap": float(abs(p_mean - y_mean)),
            "p_min": float(np.min(p[m])),
            "p_max": float(np.max(p[m])),
        })
    return pd.DataFrame(rows)
                "week_friday": week_friday.isoformat(),
                "asof_target": asof_target.isoformat(),
                "asof_date": asof_used.isoformat(),
                "expiry_close_date_used": expiry_close_used.isoformat(),

                # option chain expiries (requested Fri, maybe Sat fallback)
                "option_expiration_requested": expiration_requested.isoformat(),
                "option_expiration_used": expiration_used.isoformat(),
                "expiry_convention": expiry_convention,

                # horizons (to Friday)
                "T_days": int(T_days),
                "T_years": float(np.round(float(T_years), 9)),
                "r": float(cfg.risk_free_rate),

                # raw/adj audit
                "S_asof_close_raw": float(np.round(float(S0_raw), 7)),
                "S_expiry_close_raw": float(np.round(float(ST_raw), 7)),
                "S_asof_close_adj": float(np.round(float(S0_adj), 7)),
                "S_expiry_close_adj": float(np.round(float(ST_adj), 7)),
                "split_events_in_preload_range": split_n,
                "split_adjustment_applied": bool(cfg.apply_split_adjustment),

                # chosen scale
                "spot_scale_used": spot_scale_used,
                "spot_scale_score_raw": float(score_raw),
                "spot_scale_score_adj": float(score_adj),

                # used (consistent with chosen scale)
                "S_asof_close": float(np.round(float(S0_used), 7)),
                "S_expiry_close": float(np.round(float(ST_used), 7)),

                "asof_fallback_days": int(asof_fwd),
                "expiry_fallback_days": int(exp_bwd),

                # strike + moneyness
                "K": float(np.round(float(K), 7)),
                "log_m": float(np.round(np.log(float(K) / float(S0_used)), 9)),
                "abs_log_m": float(np.round(abs(np.log(float(K) / float(S0_used))), 9)),

                # vol proxy
                "rv20": float(np.round(rv20_used, 8)) if np.isfinite(rv20_used) else np.nan,

                # pRN (+ audit raw targets)
                "pRN": p,
                "qRN": float(np.round(1.0 - p, 7)),
                "pRN_raw": p_raw,
                "qRN_raw": float(np.round(1.0 - p_raw, 7)) if np.isfinite(p_raw) else np.nan,

                # realized outcome label (close used at/near Friday)
                "outcome_ST_gt_K": 1 if float(ST_used) > float(K) else 0,

                # band diagnostics
                "max_abs_logm_start": float(cfg.max_abs_logm),
                "max_abs_logm_cap": float(cfg.max_abs_logm_cap),
                "used_max_abs_logm": float(np.round(used_abslogm, 6)),
                "n_band_raw": int(n_band_raw),
                "n_band_inside": int(n_band_inside),
                "calls_k_min": float(np.round(k_min, 7)),
                "calls_k_max": float(np.round(k_max, 7)),

                # quote/curve diagnostics
                "theta_quote_source": diag_curve.get("quote_source"),
                "n_chain_raw": diag_curve.get("n_raw"),
                "n_chain_used": diag_curve.get("n_used"),
                "rel_spread_median": diag_curve.get("rel_spread_median"),
                "dropped_liquidity": diag_curve.get("dropped_liquidity"),
                "dropped_intrinsic": diag_curve.get("dropped_intrinsic"),
                "dropped_insane": diag_curve.get("dropped_insane"),
                "prn_monotone_adj_intervals": bool(diag_prn.get("monotone_adjusted_intervals")),
                "prn_monotone_adj_targets": bool(diag_prn.get("monotone_adjusted_targets")),
            }
        )

    if len(tmp_rows) < int(cfg.min_strikes_in_prn_band):
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "too_few_in_prn_band",
            "detail": f"kept={len(tmp_rows)} need={cfg.min_strikes_in_prn_band} inside={n_band_inside} used_abslogm={used_abslogm:.4f} spot_scale={spot_scale_used}",
        }

    # Group id (per ticker + snapshot day + week)
    group_id = f"{ticker}|{asof_used.isoformat()}|{week_friday.isoformat()}"
    med_dk, min_dk = strike_spacing_stats(np.array([r["K"] for r in tmp_rows], dtype=float))

    for rr in tmp_rows:
        rr["group_id"] = group_id
        rr["median_dK"] = float(np.round(med_dk, 6)) if np.isfinite(med_dk) else np.nan
        rr["min_dK"] = float(np.round(min_dk, 6)) if np.isfinite(min_dk) else np.nan

        if cfg.use_soft_quality_weight:
            rr["quality_weight"] = compute_quality_weight(
                quote_source=str(rr.get("theta_quote_source") or ""),
                rel_spread_median=rr.get("rel_spread_median"),
                prn_adj_intervals=bool(rr.get("prn_monotone_adj_intervals")),
                prn_adj_targets=bool(rr.get("prn_monotone_adj_targets")),
            )
        else:
            rr["quality_weight"] = 1.0

    return tmp_rows, None


# ----------------------------
# Optional sanity report/drop (group-level)
# ----------------------------

def _sanity_report_and_optional_drop(out_df: pd.DataFrame, drops: List[dict], cfg: Config) -> pd.DataFrame:
    if out_df is None or out_df.empty:
        return out_df
    if "group_id" not in out_df.columns:
        return out_df
    needed = {"K", "S_asof_close", "abs_log_m", "ticker", "week_monday", "week_friday", "asof_date"}
    if not needed.issubset(set(out_df.columns)):
        return out_df

    tmp = out_df.copy()
    tmp["K"] = pd.to_numeric(tmp["K"], errors="coerce")
    tmp["S_asof_close"] = pd.to_numeric(tmp["S_asof_close"], errors="coerce")
    tmp["abs_log_m"] = pd.to_numeric(tmp["abs_log_m"], errors="coerce")
    tmp["K_over_S"] = tmp["K"] / tmp["S_asof_close"]
    tmp = tmp.replace([np.inf, -np.inf], np.nan)

    g = tmp.groupby("group_id").agg(
        ticker=("ticker", "first"),
        week_monday=("week_monday", "first"),
        week_friday=("week_friday", "first"),
        asof_date=("asof_date", "first"),
        n=("K", "count"),
        med_abs_log_m=("abs_log_m", "median"),
        med_K_over_S=("K_over_S", "median"),
        min_K_over_S=("K_over_S", "min"),
        max_K_over_S=("K_over_S", "max"),
    ).reset_index()
    g["med_K_over_S_dist1"] = (g["med_K_over_S"] - 1.0).abs()

    if cfg.sanity_report:
        worst_abs = g.sort_values("med_abs_log_m", ascending=False).head(5)
        worst_ks = g.sort_values("med_K_over_S_dist1", ascending=False).head(5)
        print("\n[SANITY] Worst 5 groups by median abs_log_m:")
        for _, row in worst_abs.iterrows():
            print(
                f"  {row['group_id']} | {row['ticker']} | asof={row['asof_date']} | {row['week_monday']}→{row['week_friday']} | "
                f"n={int(row['n'])} | med_abs_log_m={row['med_abs_log_m']:.4f} | "
                f"med_K/S={row['med_K_over_S']:.4f} | K/S=[{row['min_K_over_S']:.4f},{row['max_K_over_S']:.4f}]"
            )
        print("\n[SANITY] Worst 5 groups by |median(K/S)-1|:")
        for _, row in worst_ks.iterrows():
            print(
                f"  {row['group_id']} | {row['ticker']} | asof={row['asof_date']} | {row['week_monday']}→{row['week_friday']} | "
                f"n={int(row['n'])} | med_K/S={row['med_K_over_S']:.4f} (dist={row['med_K_over_S_dist1']:.4f}) | "
                f"med_abs_log_m={row['med_abs_log_m']:.4f}"
            )
        print("")

    if not cfg.sanity_drop:
        return out_df

    bad_groups = set(
        g[(~np.isfinite(g["med_abs_log_m"])) | (g["med_abs_log_m"] > float(cfg.sanity_abs_logm_max))]["group_id"].tolist()
    ) | set(
        g[
            (~np.isfinite(g["med_K_over_S"]))
            | (g["med_K_over_S"] < float(cfg.sanity_k_over_s_min))
            | (g["med_K_over_S"] > float(cfg.sanity_k_over_s_max))
        ]["group_id"].tolist()
    )

    if not bad_groups:
        return out_df

    bad_mask = out_df["group_id"].isin(bad_groups)
    print(f"[SANITY] Dropping obviously broken groups: groups={len(bad_groups)} rows={int(bad_mask.sum())}")

    for gid in sorted(bad_groups):
        row = g[g["group_id"] == gid].head(1)
        if len(row) == 1:
            r = row.iloc[0]
            drops.append(
                {
                    "ticker": str(r["ticker"]),
                    "week_monday": str(r["week_monday"]),
                    "week_friday": str(r["week_friday"]),
                    "asof_date": str(r["asof_date"]),
                    "drop_reason": "sanity_bad_strike_or_scale",
                    "detail": f"group_id={gid} med_abs_log_m={float(r['med_abs_log_m']):.4f} med_K_over_S={float(r['med_K_over_S']):.4f}",
                }
            )

    return out_df.loc[~bad_mask].reset_index(drop=True)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out-dir", type=str, default="./data/history")
    ap.add_argument("--out-name", type=str, default="pRN__history__mon_thu__PM10__v1.6.0.csv")

    ap.add_argument("--tickers", type=str, default=",".join(PM10_TICKERS))
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD (range used to generate Mondays)")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD (range used to generate Mondays)")

    ap.add_argument("--theta-base-url", type=str, default=Config().theta_base_url)
    ap.add_argument("--stock-source", type=str, default=Config().stock_source, choices=["yfinance", "theta", "auto"])
    ap.add_argument("--timeout-s", type=int, default=Config().timeout_s)
    ap.add_argument("--r", type=float, default=Config().risk_free_rate)

    # Band + thresholds
    ap.add_argument("--max-abs-logm", type=float, default=Config().max_abs_logm)
    ap.add_argument("--max-abs-logm-cap", type=float, default=Config().max_abs_logm_cap)
    ap.add_argument("--band-widen-step", type=float, default=Config().band_widen_step)
    ap.add_argument("--no-adaptive-band", action="store_true")
    ap.add_argument("--max-band-strikes", type=int, default=Config().max_band_strikes)

    ap.add_argument("--min-band-strikes", type=int, default=Config().min_strikes_for_curve)
    ap.add_argument("--min-band-prn-strikes", type=int, default=Config().min_strikes_in_prn_band)

    # Option chain / expiry
    ap.add_argument("--strike-range", type=int, default=Config().option_strike_range)
    ap.add_argument("--no-retry-full-chain", action="store_true")
    ap.add_argument("--no-sat-expiry-fallback", action="store_true")
    ap.add_argument("--threads", type=int, default=6)

    # Quote/liquidity
    ap.add_argument("--prefer-bidask", action=argparse.BooleanOptionalAction, default=Config().prefer_bidask)
    ap.add_argument("--min-trade-count", type=int, default=0)
    ap.add_argument("--min-volume", type=int, default=0)

    # Hard filters
    ap.add_argument("--min-chain-used-hard", type=int, default=0)
    ap.add_argument("--max-rel-spread-median-hard", type=float, default=1e9)
    ap.add_argument("--hard-drop-close-fallback", action="store_true")

    # Training band
    ap.add_argument("--min-prn-train", type=float, default=0.10)
    ap.add_argument("--max-prn-train", type=float, default=0.90)

    # Split adjustment
    ap.add_argument("--no-split-adjust", action="store_true")

    # Weights
    ap.add_argument("--no-group-weights", action="store_true")
    ap.add_argument("--no-ticker-weights", action="store_true")
    ap.add_argument("--no-soft-quality-weight", action="store_true")

    # Vol proxy
    ap.add_argument("--rv-lookback-days", type=int, default=20)

    # Cache
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=Config().use_cache)

    # Drops
    ap.add_argument("--write-drops", action="store_true")
    ap.add_argument("--drops-name", type=str, default="pRN__history__mon_thu__drops__v1.6.0.csv")

    # Sanity
    ap.add_argument("--sanity-report", action="store_true")
    ap.add_argument("--sanity-drop", action="store_true")
    ap.add_argument("--sanity-abs-logm-max", type=float, default=Config().sanity_abs_logm_max)
    ap.add_argument("--sanity-k-over-s-min", type=float, default=Config().sanity_k_over_s_min)
    ap.add_argument("--sanity-k-over-s-max", type=float, default=Config().sanity_k_over_s_max)

    ap.add_argument("--verbose-skips", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided.")

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        raise SystemExit("--end must be >= --start")

    cfg = Config(
        theta_base_url=str(args.theta_base_url),
        timeout_s=int(args.timeout_s),
        risk_free_rate=float(args.r),

        option_strike_range=int(args.strike_range),
        retry_full_chain_if_band_thin=(not bool(args.no_retry_full_chain)),
        try_saturday_expiry_fallback=(not bool(args.no_sat_expiry_fallback)),

        max_abs_logm=float(args.max_abs_logm),
        max_abs_logm_cap=float(args.max_abs_logm_cap),
        band_widen_step=float(args.band_widen_step),
        adaptive_band=(not bool(args.no_adaptive_band)),
        max_band_strikes=int(args.max_band_strikes),

        min_strikes_for_curve=int(args.min_band_strikes),
        min_strikes_in_prn_band=int(args.min_band_prn_strikes),

        prefer_bidask=bool(args.prefer_bidask),
        min_trade_count=int(args.min_trade_count),
        min_volume=int(args.min_volume),

        min_chain_used_hard=int(args.min_chain_used_hard),
        max_rel_spread_median_hard=float(args.max_rel_spread_median_hard),
        hard_drop_close_fallback=bool(args.hard_drop_close_fallback),

        min_prn_train=float(args.min_prn_train),
        max_prn_train=float(args.max_prn_train),

        apply_split_adjustment=(not bool(args.no_split_adjust)),

        add_group_weights=(not bool(args.no_group_weights)),
        add_ticker_weights=(not bool(args.no_ticker_weights)),
        use_soft_quality_weight=(not bool(args.no_soft_quality_weight)),

        rv_lookback_days=int(args.rv_lookback_days),

        use_cache=bool(args.cache),

        stock_source=str(args.stock_source),

        sanity_report=bool(args.sanity_report),
        sanity_drop=bool(args.sanity_drop),
        sanity_abs_logm_max=float(args.sanity_abs_logm_max),
        sanity_k_over_s_min=float(args.sanity_k_over_s_min),
        sanity_k_over_s_max=float(args.sanity_k_over_s_max),
    )

    # basic parameter guards
    if not (0.0 < cfg.min_prn_train < cfg.max_prn_train < 1.0):
        raise SystemExit("Require 0 < --min-prn-train < --max-prn-train < 1")
    if cfg.max_abs_logm_cap < cfg.max_abs_logm:
        raise SystemExit("--max-abs-logm-cap must be >= --max-abs-logm")
    if cfg.band_widen_step <= 0:
        raise SystemExit("--band-widen-step must be > 0")
    if cfg.min_strikes_for_curve < 3:
        print("[WARN] min_strikes_for_curve < 3 is likely too low.")
    if cfg.min_strikes_in_prn_band < 1:
        raise SystemExit("--min-band-prn-strikes must be >= 1")

    theta = ThetaClient(cfg.theta_base_url, timeout_s=cfg.timeout_s, verbose=bool(args.verbose_skips))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)
    drops_path = os.path.join(args.out_dir, args.drops_name)

    mondays = mondays_in_range(start, end)
    print(f"[PLAN] Weeks={len(mondays)} (Mondays) range={start}..{end} tickers={len(tickers)} snapshots/day=4 (Mon-Thu)")
    print(f"[UNIVERSE] {tickers}")
    print(f"[CFG] pRN train band: [{cfg.min_prn_train}, {cfg.max_prn_train}]")
    print(f"[CFG] band start={cfg.max_abs_logm} cap={cfg.max_abs_logm_cap} step={cfg.band_widen_step} adaptive={cfg.adaptive_band}")
    print(f"[CFG] min strikes: curve={cfg.min_strikes_for_curve} after_pRN={cfg.min_strikes_in_prn_band}")
    print(f"[CFG] expiry fallback Fri->Sat: {cfg.try_saturday_expiry_fallback}")
    print(f"[CFG] split_adjustment: {cfg.apply_split_adjustment}")
    print(f"[CFG] prefer_bidask: {cfg.prefer_bidask}")
    print(f"[CFG] threads={args.threads} cache={cfg.use_cache} stock_source={cfg.stock_source}")
    if cfg.sanity_report or cfg.sanity_drop:
        print(
            f"[CFG] sanity_report={cfg.sanity_report} sanity_drop={cfg.sanity_drop} "
            f"abs_logm_max={cfg.sanity_abs_logm_max} K/S in [{cfg.sanity_k_over_s_min},{cfg.sanity_k_over_s_max}]"
        )

    # Preload stock closes for the whole range + a small cushion (because we also use Thu and exp-close fallback)
    preload_start = start
    preload_end = end + timedelta(days=4)
    print(f"[STOCK] Preloading closes for {preload_start}..{preload_end} ...")
    raw_closes_by_ticker, adj_closes_by_ticker, split_counts = preload_stock_closes(
        theta=theta,
        tickers=tickers,
        start=preload_start,
        end=preload_end,
        cfg=cfg,
        stock_source=args.stock_source,
    )

    missing = [t for t in tickers if len(raw_closes_by_ticker.get(t, {})) == 0]
    if missing:
        print(f"[STOCK] ⚠️ No close data for: {missing}")

    option_chain_cache: Dict[Tuple[str, date, date, Optional[int]], pd.DataFrame] = {}
    cache_lock = threading.Lock()

    rows: List[dict] = []
    drops: List[dict] = []

    # Jobs = for each week, snapshot days Mon/Tue/Wed/Thu, for each ticker (expiry always that week's Friday)
    def jobs():
        for mon in mondays:
            fri = iso_week_friday(mon)
            for asof_target in asof_days_mon_to_thu(mon):
                for t in tickers:
                    yield (t, mon, fri, asof_target)

    total_jobs = len(mondays) * 4 * len(tickers)
    done = 0
    kept_groups = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.threads))) as ex:
        futures = {
            ex.submit(
                process_one,
                theta=theta,
                cfg=cfg,
                ticker=t,
                asof_target=asof_target,
                week_monday=mon,
                week_friday=fri,
                raw_closes_by_ticker=raw_closes_by_ticker,
                adj_closes_by_ticker=adj_closes_by_ticker,
                split_event_counts=split_counts,
                option_chain_cache=option_chain_cache,
                cache_lock=cache_lock,
            ): (t, mon, asof_target)
            for (t, mon, fri, asof_target) in jobs()
        }

        for fut in as_completed(futures):
            t, mon, asof_target = futures[fut]
            done += 1
            try:
                rws, drop_log = fut.result()
            except Exception as e:
                rws, drop_log = [], {
                    "ticker": t,
                    "week_monday": mon.isoformat(),
                    "week_friday": iso_week_friday(mon).isoformat(),
                    "asof_target": asof_target.isoformat(),
                    "drop_reason": "exception",
                    "detail": str(e),
                }
                if args.verbose_skips:
                    print(f"[ERR] {t} week={mon} asof_target={asof_target}: {e}")

            if rws:
                kept_groups += 1
                rows.extend(rws)
            if drop_log is not None:
                drops.append(drop_log)

            if done % 100 == 0 or done == total_jobs:
                print(
                    f"[PROGRESS] {done}/{total_jobs} jobs | groups_kept={kept_groups} | rows={len(rows)} | last={t} week={mon} asof_target={asof_target}"
                )

    if not rows:
        print("[RESULT] No rows produced.")
        if args.write_drops and drops:
            pd.DataFrame(drops).to_csv(drops_path, index=False)
            print(f"[WRITE] drops: {drops_path}")
        return

    out_df = pd.DataFrame(rows)

    # parse + sort
    out_df["asof_date"] = pd.to_datetime(out_df["asof_date"], errors="coerce")
    out_df = out_df.dropna(subset=["asof_date"]).copy()
    out_df = out_df.sort_values(["asof_date", "ticker", "K"]).reset_index(drop=True)

    # optional sanity (report and/or drop)
    out_df = _sanity_report_and_optional_drop(out_df, drops, cfg)

    # Group weights: each group sums to ~1 across its rows
    if cfg.add_group_weights and "group_id" in out_df.columns:
        gsize = out_df.groupby("group_id")["K"].transform("count").astype(float)
        out_df["group_size"] = gsize
        out_df["group_weight"] = (1.0 / gsize).astype(float)
    else:
        out_df["group_size"] = np.nan
        out_df["group_weight"] = 1.0

    # Ticker weights: each ticker sums to ~1 across its groups
    if cfg.add_ticker_weights and "group_id" in out_df.columns:
        gcount = out_df.groupby("ticker")["group_id"].nunique()
        out_df["ticker_group_count"] = out_df["ticker"].map(gcount).astype(float)
        out_df["ticker_weight"] = (1.0 / out_df["ticker_group_count"]).astype(float)
    else:
        out_df["ticker_group_count"] = np.nan
        out_df["ticker_weight"] = 1.0

    # Final sample weight
    qw = pd.to_numeric(out_df.get("quality_weight", 1.0), errors="coerce").fillna(1.0).clip(0.01, 1.0)
    out_df["quality_weight"] = qw
    out_df["sample_weight_final"] = (out_df["group_weight"] * out_df["ticker_weight"] * out_df["quality_weight"]).astype(float)
    out_df["sample_weight_final"] = (
        pd.to_numeric(out_df["sample_weight_final"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Summary
    weeks = out_df["asof_date"].dt.to_period("W-MON").nunique()
    groups = out_df["group_id"].nunique() if "group_id" in out_df.columns else np.nan
    print(f"[SUMMARY] rows={len(out_df)} tickers={out_df['ticker'].nunique()} weeks={weeks} groups={groups}")
    if "group_id" in out_df.columns:
        print("[SUMMARY] per-ticker groups:")
        print(out_df.groupby("ticker")["group_id"].nunique().sort_values(ascending=False).to_string())

    out_df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}")

    if args.write_drops:
        drops_df = pd.DataFrame(drops) if drops else pd.DataFrame(columns=["ticker", "week_monday", "week_friday", "asof_target", "drop_reason", "detail"])
        drops_df.to_csv(drops_path, index=False)
        print(f"[WRITE] drops: {drops_path}")


if __name__ == "__main__":
    main()


