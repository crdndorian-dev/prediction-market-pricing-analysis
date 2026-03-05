from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

WEIGHTING_VERSION = "v3"

WEEKDAY_ABBR = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
WEEKDAY_TO_NUM = {name: idx for idx, name in enumerate(WEEKDAY_ABBR)}

LEGACY_WEIGHT_COLUMNS = (
    "quality_weight",
    "group_size",
    "group_weight",
    "ticker_group_count",
    "ticker_weight",
    "sample_weight_final",
)

V3_WEIGHT_COLUMNS = (
    "cluster_week",
    "cluster_snapshot",
    "cluster_strike",
    "weight_group_key",
    "weight_group_size",
    "weight_group_w",
    "weight_ticker_group_count",
    "weight_ticker_w_raw",
    "weight_trade_focus_mult",
    "weight_raw",
    "weight_final",
    "weighting_version",
)


def _date_to_ymd(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.strftime("%Y-%m-%d")


def _coalesce_date(
    df: pd.DataFrame,
    candidates: Sequence[str],
) -> Tuple[pd.Series, Optional[str]]:
    for col in candidates:
        if col in df.columns:
            as_ymd = _date_to_ymd(df[col])
            if as_ymd.notna().any():
                return as_ymd, col
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="string"), None


def _normalize_snapshot_dow(
    series: Optional[pd.Series],
    fallback_dates: pd.Series,
) -> pd.Series:
    out = pd.Series([pd.NA] * len(fallback_dates), index=fallback_dates.index, dtype="string")
    if series is not None:
        raw = series.astype("string").str.strip()
        numeric = pd.to_numeric(raw, errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            mapped = numeric[numeric_mask].astype(int).map(lambda x: WEEKDAY_ABBR[x] if 0 <= x <= 6 else pd.NA)
            out.loc[numeric_mask] = mapped.astype("string")
        text_mask = ~numeric_mask & raw.notna()
        if text_mask.any():
            normalized = raw[text_mask].str.upper().str.slice(0, 3)
            out.loc[text_mask] = normalized.astype("string")
    invalid = ~out.isin(WEEKDAY_ABBR)
    out.loc[invalid] = pd.NA
    missing = out.isna()
    if missing.any():
        dt = pd.to_datetime(fallback_dates, errors="coerce")
        derived = dt.dt.dayofweek.map(lambda x: WEEKDAY_ABBR[int(x)] if pd.notna(x) else pd.NA)
        out.loc[missing] = derived.loc[missing].astype("string")
    return out


def _resolve_week_id(df: pd.DataFrame, expiry_date: pd.Series) -> pd.Series:
    if "week_id" in df.columns:
        existing = _date_to_ymd(df["week_id"])
        if existing.notna().any():
            return existing
    if "week_friday" in df.columns:
        existing = _date_to_ymd(df["week_friday"])
        if existing.notna().any():
            return existing
    exp_ts = pd.to_datetime(expiry_date, errors="coerce")
    week_friday = exp_ts + pd.to_timedelta((4 - exp_ts.dt.dayofweek) % 7, unit="D")
    return week_friday.dt.strftime("%Y-%m-%d")


def parse_trade_focus_tickers(raw: Optional[Sequence[str] | str]) -> set[str]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        tokens = [t.strip().upper() for t in raw.split(",") if t.strip()]
        return set(tokens)
    return {str(v).strip().upper() for v in raw if str(v).strip()}


def drop_weight_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in (*LEGACY_WEIGHT_COLUMNS, *V3_WEIGHT_COLUMNS) if c in df.columns]
    if not cols:
        return df
    return df.drop(columns=cols)


def _ensure_required_base_columns(df: pd.DataFrame) -> None:
    if "ticker" not in df.columns:
        raise ValueError("Missing required column: ticker")


def _resolve_strike_series(df: pd.DataFrame) -> pd.Series:
    for col in ("K", "threshold", "strike"):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                return vals
    raise ValueError("Missing strike column. Need one of: K, threshold, strike")


def _coerce_cluster_strings(
    ticker: pd.Series,
    expiry_date: pd.Series,
    snapshot_date: pd.Series,
    snapshot_dow: pd.Series,
    strike: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    t = ticker.astype("string").fillna("UNKNOWN").str.upper()
    e = expiry_date.astype("string").fillna("NA")
    s = snapshot_date.astype("string").fillna("NA")
    d = snapshot_dow.astype("string").fillna("NA")
    strike_arr = pd.to_numeric(strike, errors="coerce").round(7).to_numpy(dtype=float)
    strike_txt = pd.Series(
        np.where(np.isfinite(strike_arr), np.char.mod("%.7f", strike_arr), "NA"),
        index=t.index,
        dtype="string",
    )
    cluster_week = t + "|" + e
    cluster_snapshot = t + "|" + e + "|" + s + "|" + d
    cluster_strike = t + "|" + e + "|" + strike_txt
    return cluster_week, cluster_snapshot, cluster_strike


def apply_weighting_v3(
    df: pd.DataFrame,
    *,
    ticker_reweight_mode: str = "none",
    ticker_reweight_alpha_min: float = 0.5,
    ticker_reweight_alpha_max: float = 2.0,
    trade_focus_beta: float = 1.0,
    trade_focus_tickers: Optional[Sequence[str] | str] = None,
    strict: bool = True,
) -> pd.DataFrame:
    if df is None:
        raise ValueError("Dataframe is required.")
    out = df.copy()
    if out.empty:
        out["weighting_version"] = WEIGHTING_VERSION
        return out

    _ensure_required_base_columns(out)

    snapshot_date, snapshot_col = _coalesce_date(out, ("snapshot_date", "asof_date", "asof_target", "asof_ts"))
    if snapshot_col is None:
        raise ValueError("Cannot resolve snapshot_date from: snapshot_date, asof_date, asof_target, asof_ts")
    out["snapshot_date"] = snapshot_date

    expiry_date, expiry_col = _coalesce_date(
        out,
        ("expiry_date", "option_expiration_used", "option_expiration_requested", "expiry_close_date_used", "week_friday"),
    )
    if expiry_col is None:
        raise ValueError(
            "Cannot resolve expiry_date from: expiry_date, option_expiration_used, option_expiration_requested, "
            "expiry_close_date_used, week_friday"
        )
    out["expiry_date"] = expiry_date

    snapshot_dow_source = out["snapshot_dow"] if "snapshot_dow" in out.columns else None
    out["snapshot_dow"] = _normalize_snapshot_dow(snapshot_dow_source, out["snapshot_date"])
    out["week_id"] = _resolve_week_id(out, out["expiry_date"])

    strike = _resolve_strike_series(out)

    critical_missing = out["snapshot_date"].isna() | out["expiry_date"].isna() | out["snapshot_dow"].isna() | strike.isna()
    if critical_missing.any():
        msg = (
            "Cannot compute weighting keys for all rows. Missing among snapshot_date/expiry_date/"
            f"snapshot_dow/strike in {int(critical_missing.sum())} rows."
        )
        if strict:
            raise ValueError(msg)
        out = out.loc[~critical_missing].copy()
        strike = strike.loc[out.index]
        if out.empty:
            raise ValueError(msg)

    cluster_week, cluster_snapshot, cluster_strike = _coerce_cluster_strings(
        out["ticker"],
        out["expiry_date"],
        out["snapshot_date"],
        out["snapshot_dow"],
        strike,
    )
    out["cluster_week"] = cluster_week
    out["cluster_snapshot"] = cluster_snapshot
    out["cluster_strike"] = cluster_strike
    out["weight_group_key"] = out["cluster_snapshot"]
    out["group_id"] = out["cluster_snapshot"]

    group_size = out.groupby("weight_group_key", dropna=False)["weight_group_key"].transform("size").astype(float)
    out["weight_group_size"] = group_size
    out["weight_group_w"] = 1.0 / group_size

    ticker_upper = out["ticker"].astype("string").fillna("UNKNOWN").str.upper()
    ticker_group_count_map = out.groupby(ticker_upper, dropna=False)["weight_group_key"].nunique().astype(float)
    out["weight_ticker_group_count"] = ticker_upper.map(ticker_group_count_map).astype(float)

    mode = str(ticker_reweight_mode or "none").strip().lower()
    if mode not in {"none", "sqrt_inv"}:
        raise ValueError("ticker_reweight_mode must be one of: none, sqrt_inv")

    if mode == "none":
        out["weight_ticker_w_raw"] = 1.0
    else:
        mean_groups = float(ticker_group_count_map.mean()) if len(ticker_group_count_map) > 0 else 1.0
        group_counts = out["weight_ticker_group_count"].clip(lower=1.0)
        alpha = np.sqrt(mean_groups / group_counts)
        alpha = alpha.clip(lower=float(ticker_reweight_alpha_min), upper=float(ticker_reweight_alpha_max))
        out["weight_ticker_w_raw"] = alpha.astype(float)

    focus_set = parse_trade_focus_tickers(trade_focus_tickers)
    beta = float(trade_focus_beta)
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("trade_focus_beta must be finite and > 0")
    out["weight_trade_focus_mult"] = np.where(ticker_upper.isin(focus_set), beta, 1.0).astype(float)

    out["weight_raw"] = (
        pd.to_numeric(out["weight_group_w"], errors="coerce")
        * pd.to_numeric(out["weight_ticker_w_raw"], errors="coerce")
        * pd.to_numeric(out["weight_trade_focus_mult"], errors="coerce")
    )
    invalid_raw = ~np.isfinite(out["weight_raw"]) | (out["weight_raw"] <= 0)
    if invalid_raw.any():
        raise ValueError(f"weight_raw has invalid rows: {int(invalid_raw.sum())}")

    raw_mean = float(out["weight_raw"].mean())
    if not np.isfinite(raw_mean) or raw_mean <= 0:
        raise ValueError(f"weight_raw mean must be positive finite; got {raw_mean}")

    out["weight_final"] = out["weight_raw"] / raw_mean
    invalid_final = ~np.isfinite(out["weight_final"]) | (out["weight_final"] <= 0)
    if invalid_final.any():
        raise ValueError(f"weight_final has invalid rows: {int(invalid_final.sum())}")

    out["weighting_version"] = WEIGHTING_VERSION
    return out
