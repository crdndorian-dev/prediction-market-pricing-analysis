from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from calibration.calibrate_common import EPS, _logit


def infer_asof_fallback_days(spot_source: Optional[str]) -> Optional[int]:
    if not spot_source:
        return None
    src = spot_source.strip().lower()
    if src in ("intraday_1m", "intraday_prepost_1m"):
        return 0
    if src == "prev_close":
        return 1
    return None


def compute_forward_price(S: float, r: float, q: float, T_years: float) -> Optional[float]:
    """
    Compute forward price using: F = S * exp((r - q) * T)

    Args:
        S: Spot price
        r: Risk-free rate
        q: Continuous dividend yield
        T_years: Time to maturity in years

    Returns: Forward price, or None if computation fails
    """
    try:
        if S <= 0 or T_years <= 0:
            return None
        if not (0.0 <= r <= 1.0) or not (0.0 <= q <= 1.0):
            return None

        forward = S * math.exp((r - q) * T_years)
        if np.isfinite(forward) and forward > 0:
            return float(forward)
        return None
    except Exception:
        return None


def enrich_snapshot_features(
    df: pd.DataFrame,
    cfg: Any,
    manifest: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Enrich snapshot with all guaranteed baseline columns and derived features.

    This function GUARANTEES that all columns from pipeline_schema_contract.json
    baseline schema will exist (even if NaN). This ensures schema compatibility
    with any downstream calibration model.
    """
    out = df.copy()

    expiry_src = out["expiry_ts_utc"] if "expiry_ts_utc" in out.columns else out["event_endDate"]
    expiry_dt = pd.to_datetime(expiry_src, utc=True, errors="coerce")
    out["expiry_date"] = expiry_dt.dt.strftime("%Y-%m-%d")
    out["expiry_close_date_used"] = expiry_dt.dt.tz_convert(ZoneInfo(cfg.tz_name)).dt.date.astype(str)
    if "snapshot_time_utc" in out.columns:
        asof_dt = pd.to_datetime(out["snapshot_time_utc"], utc=True, errors="coerce")
        out["asof_date"] = asof_dt.dt.tz_convert(ZoneInfo(cfg.tz_name)).dt.date.astype(str)
    else:
        out["asof_date"] = np.nan
    out["S_asof_close"] = pd.to_numeric(out.get("S", np.nan), errors="coerce")
    out["S_asof_close_adj"] = out["S_asof_close"]
    out["S_expiry_close"] = out["S_asof_close"]
    out["S_expiry_close_adj"] = out["S_asof_close"]
    out["r"] = cfg.risk_free_rate

    baseline_defaults = {
        "rv20": cfg.rv20_fallback,
        "rv20_source": "fallback",
        "rv20_window": 0,
        "is_missing_rv20": True,
        "dividend_yield": 0.0,
        "rel_spread_median": np.nan,
        "n_chain_raw": np.nan,
        "n_chain_used": np.nan,
        "n_band_raw": np.nan,
        "n_band_inside": np.nan,
        "dropped_intrinsic": np.nan,
        "asof_fallback_days": np.nan,
        "expiry_fallback_days": 0,
        "split_events_in_preload_range": np.nan,
        "spot_scale_used": "raw",
    }
    for col, default in baseline_defaults.items():
        if col not in out.columns:
            out[col] = default
    if "spot_scale_used" in out.columns:
        out["spot_scale_used"] = out["spot_scale_used"].fillna("raw").astype(str)
    if "asof_fallback_days" in out.columns and out["asof_fallback_days"].isna().all() and "spot_source" in out.columns:
        out["asof_fallback_days"] = out["spot_source"].map(infer_asof_fallback_days)

    out["T_days"] = pd.to_numeric(out["T_days"], errors="coerce")
    out["T_years"] = out["T_days"] / 365.0
    out["sqrt_T_years"] = np.sqrt(out["T_years"].clip(lower=0))
    out["log_T_days"] = np.log1p(out["T_days"].clip(lower=0))

    K = pd.to_numeric(out["K"], errors="coerce")
    S = pd.to_numeric(out["S"], errors="coerce")
    out["log_m"] = np.log(np.clip(K, 1e-12, None) / np.clip(S, 1e-12, None))
    out["abs_log_m"] = out["log_m"].abs()

    prn = pd.to_numeric(out["pRN"], errors="coerce")
    prn_clipped = prn.clip(EPS, 1.0 - EPS)
    out["x_logit_prn"] = _logit(prn_clipped.to_numpy(dtype=float))
    out["pRN"] = prn

    if "pRN_raw" not in out.columns or pd.isna(out["pRN_raw"]).all():
        out["pRN_raw"] = out["pRN"]
    out["prn_raw_gap"] = prn - pd.to_numeric(out["pRN_raw"], errors="coerce")

    out["x_prn_x_tdays"] = out["x_logit_prn"] * out["T_days"]
    out["x_prn_x_rv20"] = out["x_logit_prn"] * pd.to_numeric(out.get("rv20", np.nan), errors="coerce")
    out["x_prn_x_logm"] = out["x_logit_prn"] * out["log_m"]

    rv20 = pd.to_numeric(out.get("rv20", np.nan), errors="coerce")
    vol_denom = rv20 * out["sqrt_T_years"]
    vol_denom = vol_denom.replace(0, np.nan)
    out["rv20_sqrtT"] = rv20 * out["sqrt_T_years"]
    out["log_m_over_volT"] = out["log_m"] / vol_denom
    out["abs_log_m_over_volT"] = out["log_m"].abs() / vol_denom

    if "dividend_yield" not in out.columns:
        out["dividend_yield"] = 0.0
    out["dividend_yield"] = pd.to_numeric(out["dividend_yield"], errors="coerce").fillna(0.0)

    if "forward_price" not in out.columns:
        out["forward_price"] = np.nan
    F = pd.to_numeric(out.get("forward_price", np.nan), errors="coerce")

    if pd.isna(F).any():
        if "forward_price" not in out.columns:
            out["forward_price"] = np.nan
        for i, row in out.iterrows():
            if np.isfinite(row.get("forward_price", np.nan)):
                continue
            S_val = row.get("S")
            T_years = row.get("T_years")
            if S_val is None or not np.isfinite(S_val) or T_years is None or not np.isfinite(T_years):
                continue
            div_yield = row.get("dividend_yield")
            if div_yield is None or not np.isfinite(div_yield):
                div_yield = 0.0
            forward = compute_forward_price(float(S_val), float(cfg.risk_free_rate), float(div_yield), float(T_years))
            out.at[i, "forward_price"] = forward

    F = pd.to_numeric(out.get("forward_price", np.nan), errors="coerce")
    out["log_m_fwd"] = np.log(np.clip(out["K"], 1e-12, None) / np.clip(F, 1e-12, None))
    out["abs_log_m_fwd"] = out["log_m_fwd"].abs()

    out["log_m_fwd_over_volT"] = out["log_m_fwd"] / vol_denom
    out["abs_log_m_fwd_over_volT"] = out["abs_log_m_fwd"] / vol_denom

    return out
