from __future__ import annotations

from datetime import date, time as dt_time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]

PRN_COL_CANDIDATES = [
    "pRN",
    "qRN",
    "pRN_raw",
    "qRN_raw",
    "rv20",
    "log_m",
    "abs_log_m",
    "log_m_fwd",
    "abs_log_m_fwd",
    "x_logit_prn",
    "T_days",
    "S_asof_close",
    "forward_price",
    "dividend_yield",
]


def _safe_zoneinfo(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except Exception:
        print(f"[WARN] Invalid prn_asof_tz '{tz_name}', falling back to UTC.")
        return ZoneInfo("UTC")


def _parse_close_time(value: str) -> dt_time:
    if not value:
        return dt_time(16, 0)
    raw = value.strip()
    parts = raw.split(":")
    try:
        if len(parts) == 1:
            return dt_time(int(parts[0]), 0)
        if len(parts) == 2:
            return dt_time(int(parts[0]), int(parts[1]))
        if len(parts) >= 3:
            return dt_time(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        pass
    print(f"[WARN] Invalid prn_asof_close_time '{value}', using 16:00.")
    return dt_time(16, 0)


def _has_non_midnight(ts: pd.Series) -> bool:
    if ts.empty:
        return False
    ts = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts.isna().all():
        return False
    return bool(
        ((ts.dt.hour != 0) | (ts.dt.minute != 0) | (ts.dt.second != 0) | (ts.dt.microsecond != 0)).any()
    )


def _asof_time_from_date(
    dates: pd.Series,
    *,
    tz_name: str,
    close_time: str,
) -> pd.Series:
    tz = _safe_zoneinfo(tz_name)
    close_t = _parse_close_time(close_time)
    date_vals = pd.to_datetime(dates, errors="coerce").dt.date.astype(str)
    naive = pd.to_datetime(date_vals + " " + close_t.strftime("%H:%M:%S"), errors="coerce")
    localized = naive.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    return localized.dt.tz_convert("UTC")


def asof_time_from_date(
    dates: pd.Series,
    *,
    tz_name: str,
    close_time: str,
) -> pd.Series:
    return _asof_time_from_date(dates, tz_name=tz_name, close_time=close_time)


def derive_prn_asof_time(
    df: pd.DataFrame,
    *,
    tz_name: str,
    close_time: str,
) -> pd.Series:
    for col in ("asof_ts", "asof_time", "asof_datetime"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            if _has_non_midnight(ts):
                return ts
            return _asof_time_from_date(ts.dt.date, tz_name=tz_name, close_time=close_time)

    for col in ("asof_date", "asof_target"):
        if col in df.columns:
            return _asof_time_from_date(df[col], tz_name=tz_name, close_time=close_time)

    raise KeyError("pRN dataset missing asof_date/asof_target/asof_ts column.")


def find_latest_prn_dataset() -> Optional[Path]:
    base = REPO_ROOT / "src" / "data" / "raw" / "option-chain"
    candidates = list(base.rglob("dataset-n1.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def normalize_threshold(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return round(val, 6)


def normalize_threshold_series(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    vals = vals.where(np.isfinite(vals), np.nan)
    return vals.round(6)


def _filter_prn_chunk(
    df: pd.DataFrame,
    *,
    expiry_col: str,
    tickers: Optional[set[str]],
    thresholds: Optional[set[float]],
    expiry_dates: Optional[set[date]],
    date_start: Optional[date],
    date_end: Optional[date],
    prn_asof_tz: str,
    prn_asof_close_time: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    df["ticker"] = df["ticker"].astype(str).str.upper()
    if tickers:
        df = df[df["ticker"].isin(tickers)]
    if df.empty:
        return df

    if "K" in df.columns:
        df["threshold"] = normalize_threshold_series(df["K"])
        if thresholds:
            df = df[df["threshold"].isin(thresholds)]
    if df.empty:
        return df

    df["expiry_date"] = pd.to_datetime(df[expiry_col], errors="coerce").dt.date
    if expiry_dates:
        df = df[df["expiry_date"].isin(expiry_dates)]
    if df.empty:
        return df

    if date_start or date_end:
        date_series = None
        if "asof_date" in df.columns:
            date_series = pd.to_datetime(df["asof_date"], errors="coerce").dt.date
        elif "asof_target" in df.columns:
            date_series = pd.to_datetime(df["asof_target"], errors="coerce").dt.date
        elif "asof_ts" in df.columns:
            date_series = pd.to_datetime(df["asof_ts"], utc=True, errors="coerce").dt.date
        elif "asof_time" in df.columns:
            date_series = pd.to_datetime(df["asof_time"], utc=True, errors="coerce").dt.date
        elif "asof_datetime" in df.columns:
            date_series = pd.to_datetime(df["asof_datetime"], utc=True, errors="coerce").dt.date

        if date_series is not None:
            mask = pd.Series(True, index=df.index)
            if date_start:
                mask &= (date_series >= date_start)
            if date_end:
                mask &= (date_series <= date_end)
            df = df[mask]
            if df.empty:
                return df

    df["asof_time"] = derive_prn_asof_time(
        df,
        tz_name=prn_asof_tz,
        close_time=prn_asof_close_time,
    )
    df["snapshot_date"] = pd.to_datetime(df["asof_time"], utc=True, errors="coerce").dt.date

    if date_start or date_end:
        mask = pd.Series(True, index=df.index)
        if date_start:
            mask &= (df["snapshot_date"] >= date_start)
        if date_end:
            mask &= (df["snapshot_date"] <= date_end)
        df = df[mask]

    return df


def load_prn_dataset(
    path: Path,
    *,
    prn_asof_tz: str,
    prn_asof_close_time: str,
    tickers: Optional[set[str]] = None,
    thresholds: Optional[set[float]] = None,
    expiry_dates: Optional[set[date]] = None,
    date_start: Optional[date] = None,
    date_end: Optional[date] = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    try:
        header = pd.read_csv(path, nrows=0)
        cols = list(header.columns)
    except Exception:
        cols = []

    if cols:
        expiry_candidates = ["expiry_close_date_used", "option_expiration_used", "option_expiration_requested", "expiry_date"]
        expiry_col = next((c for c in expiry_candidates if c in cols), None)
        if not expiry_col:
            raise KeyError("pRN dataset missing expiry date column.")
        asof_cols = [c for c in ("asof_ts", "asof_time", "asof_datetime", "asof_date", "asof_target") if c in cols]
        if not asof_cols:
            raise KeyError("pRN dataset missing asof_date/asof_target/asof_ts column.")
        usecols = {"ticker", "K", expiry_col}
        usecols.update(asof_cols)
        usecols.update([c for c in PRN_COL_CANDIDATES if c in cols])
    else:
        expiry_col = None
        usecols = None

    tickers = set(tickers) if tickers else None
    thresholds = set(thresholds) if thresholds else None
    expiry_dates = set(expiry_dates) if expiry_dates else None

    if not cols:
        df = pd.read_csv(path)
        if df.empty:
            return df
        expiry_col = None
        for col in ["expiry_close_date_used", "option_expiration_used", "option_expiration_requested", "expiry_date"]:
            if col in df.columns:
                expiry_col = col
                break
        if not expiry_col:
            raise KeyError("pRN dataset missing expiry date column.")
        df = _filter_prn_chunk(
            df,
            expiry_col=expiry_col,
            tickers=tickers,
            thresholds=thresholds,
            expiry_dates=expiry_dates,
            date_start=date_start,
            date_end=date_end,
            prn_asof_tz=prn_asof_tz,
            prn_asof_close_time=prn_asof_close_time,
        )
        if df.empty:
            return df
    else:
        if not expiry_col:
            raise KeyError("pRN dataset missing expiry date column.")
        reader = pd.read_csv(path, usecols=list(usecols), chunksize=chunksize)
        frames = []
        total_rows = 0
        kept_rows = 0
        chunk_idx = 0
        for chunk in reader:
            chunk_idx += 1
            total_rows += len(chunk)
            filtered = _filter_prn_chunk(
                chunk,
                expiry_col=expiry_col,
                tickers=tickers,
                thresholds=thresholds,
                expiry_dates=expiry_dates,
                date_start=date_start,
                date_end=date_end,
                prn_asof_tz=prn_asof_tz,
                prn_asof_close_time=prn_asof_close_time,
            )
            if not filtered.empty:
                frames.append(filtered)
                kept_rows += len(filtered)
            if chunk_idx == 1 or chunk_idx % 5 == 0:
                print(f"[features] pRN rows kept {kept_rows}/{total_rows} after {chunk_idx} chunks")
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

    if df.empty:
        return df

    keep_cols = ["ticker", "threshold", "expiry_date", "asof_time", "snapshot_date"]
    for col in PRN_COL_CANDIDATES:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols]
    df = df.dropna(subset=["ticker", "threshold", "expiry_date", "asof_time"])

    key_cols = ["ticker", "threshold", "expiry_date", "snapshot_date"]
    if not df.empty and df.duplicated(subset=key_cols).any():
        print("[WARN] pRN dataset has duplicate rows for the same (ticker, threshold, expiry_date, snapshot_date).")
        df = df.sort_values(key_cols + ["asof_time"]).drop_duplicates(subset=key_cols, keep="last")

    return df
