#!/usr/bin/env python3
"""
02-polymarket-build-features-v1.0.py

Build a time-safe decision dataset by combining Polymarket bars with
pRN features. Enforces anti-leak checks.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from polymarket.subgraph_client import SubgraphClient

SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION = "pm_features_v1.0"

DEFAULT_DIM_PATH = REPO_ROOT / "src" / "data" / "models" / "polymarket" / "dim_market.parquet"
DEFAULT_BARS_DIR = REPO_ROOT / "src" / "data" / "analysis" / "polymarket" / "bars"
DEFAULT_BARS_HISTORY_DIR = REPO_ROOT / "src" / "data" / "analysis" / "polymarket" / "bars_history"
DEFAULT_OUT_DIR = REPO_ROOT / "src" / "data" / "models" / "polymarket"

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


# ----------------------------
# Profiling infrastructure
# ----------------------------

@dataclass
class ProfileStats:
    """Track performance metrics for the pipeline."""
    stage_times: Dict[str, float] = field(default_factory=dict)
    stage_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    api_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record_stage(self, stage: str, duration: float) -> None:
        self.stage_times[stage] = duration

    def increment_count(self, key: str, value: int = 1) -> None:
        self.stage_counts[key] += value

    def print_summary(self) -> None:
        total = time.perf_counter() - self.start_time
        print("\n" + "="*70)
        print("[PROFILE] Performance Summary")
        print("="*70)
        print(f"Total runtime: {total:.2f}s")
        print("\nStage breakdown:")
        for stage, duration in sorted(self.stage_times.items(), key=lambda x: -x[1]):
            pct = (duration / total * 100) if total > 0 else 0
            print(f"  {stage:40s} {duration:8.2f}s  ({pct:5.1f}%)")
        print("\nCounts:")
        for key, count in sorted(self.stage_counts.items()):
            print(f"  {key:40s} {count:>12,}")
        if self.api_calls:
            print(f"  {'API calls':40s} {self.api_calls:>12,}")
        if self.cache_hits or self.cache_misses:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            print(f"  {'Cache hit rate':40s} {hit_rate:>11.1f}%")
        print("="*70 + "\n")


class ProfileContext:
    """Context manager for timing pipeline stages."""
    def __init__(self, stats: Optional[ProfileStats], stage: str):
        self.stats = stats
        self.stage = stage
        self.start = None

    def __enter__(self):
        if self.stats:
            self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.stats and self.start:
            duration = time.perf_counter() - self.start
            self.stats.record_stage(self.stage, duration)
            print(f"[PROFILE] {self.stage}: {duration:.2f}s")


@dataclass
class Config:
    dim_market_path: Path = DEFAULT_DIM_PATH
    bars_dir: Path = DEFAULT_BARS_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    prn_dataset: Optional[Path] = None
    decision_freq: str = "1h"
    spread_bps: float = 200.0
    vol_window: int = 24
    vol_min: int = 6
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fail_on_leak: bool = True
    prn_asof_tz: str = "America/New_York"
    prn_asof_close_time: str = "16:00"
    profile: bool = False
    profile_output: Optional[Path] = None


# ----------------------------
# Helpers
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

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


def _is_daily_freq(freq: str) -> bool:
    if not freq:
        return False
    return freq.strip().lower() in {"1d", "1day", "1-day"}


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


def _apply_daily_decision_time(
    df: pd.DataFrame,
    *,
    tz_name: str,
    close_time: str,
    decision_offset_s: int = 1,
) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["decision_date"] = ts.dt.date
    decision_ts = _asof_time_from_date(df["decision_date"], tz_name=tz_name, close_time=close_time)
    if decision_offset_s:
        decision_ts = decision_ts + pd.to_timedelta(decision_offset_s, unit="s")
    df["timestamp_utc"] = decision_ts
    return df


def _derive_prn_asof_time(
    df: pd.DataFrame,
    *,
    tz_name: str,
    close_time: str,
) -> pd.Series:
    # Prefer explicit asof timestamps if they include time-of-day.
    for col in ("asof_ts", "asof_time", "asof_datetime"):
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            if _has_non_midnight(ts):
                return ts
            # If only midnight stamps exist, treat them as date-only snapshots.
            return _asof_time_from_date(ts.dt.date, tz_name=tz_name, close_time=close_time)

    for col in ("asof_date", "asof_target"):
        if col in df.columns:
            return _asof_time_from_date(df[col], tz_name=tz_name, close_time=close_time)

    raise KeyError("pRN dataset missing asof_date/asof_target/asof_ts column.")


def _find_latest_prn_dataset() -> Optional[Path]:
    base = REPO_ROOT / "src" / "data" / "raw" / "option-chain"
    candidates = list(base.rglob("dataset-n1.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_dim_market(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"dim_market not found: {path}")
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    return pd.read_csv(path)


def _parse_date_folder(path: Path) -> Optional[str]:
    name = path.name
    if name.startswith("date="):
        return name.replace("date=", "")
    return None


def _iter_bar_files(
    bars_dir: Path,
    freq: str,
    market_ids: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Path]:
    base = bars_dir / freq
    if not base.exists():
        return []

    paths: List[Path] = []
    if market_ids:
        for market_id in market_ids:
            glob = base.glob(f"market_id={market_id}/date=*/bars.csv")
            paths.extend(list(glob))
    else:
        paths = list(base.glob("market_id=*/date=*/bars.csv"))

    out: List[Path] = []
    for path in sorted(paths):
        date_part = _parse_date_folder(path.parent)
        if not date_part:
            continue
        if start_date and date_part < start_date:
            continue
        if end_date and date_part > end_date:
            continue
        out.append(path)
    return out


def _load_bars(
    bars_dir: Path,
    freq: str,
    market_ids: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    files = _iter_bar_files(bars_dir, freq, market_ids, start_date, end_date)
    if not files:
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    read_errors = 0
    missing_cols = 0
    required_cols = {"timestamp_utc", "market_id", "close"}
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            read_errors += 1
            if read_errors <= 3:
                print(f"[WARN] Failed to read bars file {path}: {exc}")
            continue
        if df.empty:
            continue
        missing = required_cols - set(df.columns)
        if missing:
            missing_cols += 1
            if missing_cols <= 3:
                print(f"[WARN] Bars file missing columns {sorted(missing)}: {path}")
            continue
        frames.append(df)

    if not frames:
        if read_errors:
            print(f"[features] Skipped {read_errors} bars files due to read errors.")
        if missing_cols:
            print(f"[features] Skipped {missing_cols} bars files due to missing columns.")
        return pd.DataFrame()

    if read_errors:
        print(f"[features] Skipped {read_errors} bars files due to read errors.")
    if missing_cols:
        print(f"[features] Skipped {missing_cols} bars files due to missing columns.")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df["open"] = pd.to_numeric(df.get("open"), errors="coerce")
    df["high"] = pd.to_numeric(df.get("high"), errors="coerce")
    df["low"] = pd.to_numeric(df.get("low"), errors="coerce")
    df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce")
    df["trade_count"] = pd.to_numeric(df.get("trade_count"), errors="coerce")
    df = df.dropna(subset=["close"])
    if "market_id" in df.columns:
        df = df.drop_duplicates(subset=["market_id", "timestamp_utc"], keep="last")
    return df


def _normalize_threshold(x: Any) -> Optional[float]:
    try:
        val = float(x)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return round(val, 6)

def _normalize_threshold_series(values: pd.Series) -> pd.Series:
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
        df["threshold"] = _normalize_threshold_series(df["K"])
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

    df["asof_time"] = _derive_prn_asof_time(
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


def _load_prn_dataset(
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
        expiry_candidates = ["expiry_close_date_used", "option_expiration_used", "option_expiration_requested"]
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
        for col in ["expiry_close_date_used", "option_expiration_used", "option_expiration_requested"]:
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
        used_filters = any([tickers, thresholds, expiry_dates, date_start, date_end])
        if used_filters:
            try:
                import pyarrow.dataset as ds  # type: ignore

                dataset = ds.dataset(str(path), format="csv")
                columns = sorted(usecols)

                filter_expr = None
                if tickers:
                    filter_expr = ds.field("ticker").isin(sorted(tickers))
                if expiry_dates:
                    expiry_vals = [d.isoformat() for d in sorted(expiry_dates)]
                    expr = ds.field(expiry_col).isin(expiry_vals)
                    filter_expr = expr if filter_expr is None else filter_expr & expr
                if date_start or date_end:
                    date_col = None
                    for candidate in ("asof_date", "asof_target"):
                        if candidate in columns:
                            date_col = candidate
                            break
                    if date_col and date_start:
                        expr = ds.field(date_col) >= date_start.isoformat()
                        filter_expr = expr if filter_expr is None else filter_expr & expr
                    if date_col and date_end:
                        expr = ds.field(date_col) <= date_end.isoformat()
                        filter_expr = expr if filter_expr is None else filter_expr & expr

                print("[features] pRN scan: pyarrow")
                table = dataset.to_table(columns=columns, filter=filter_expr, use_threads=True)
                df = table.to_pandas()
                if not df.empty:
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
            except Exception as exc:
                print(f"[features] pRN pyarrow scan failed, falling back to chunked pandas: {exc}")
                df = pd.DataFrame()

            if df.empty:
                reader = pd.read_csv(path, usecols=sorted(usecols), chunksize=chunksize)
                frames: List[pd.DataFrame] = []
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
        else:
            reader = pd.read_csv(path, usecols=sorted(usecols), chunksize=chunksize)
            frames: List[pd.DataFrame] = []
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


def _build_base_features(df: pd.DataFrame, spread_bps: float, vol_window: int, vol_min: int) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values(["market_id", "timestamp_utc"]).copy()
    g = df.groupby("market_id")

    df["pm_last"] = g["close"].shift(1)
    df["pm_last_time"] = g["timestamp_utc"].shift(1)

    df["pm_mid"] = df["pm_last"]
    spread = spread_bps / 10_000.0
    df["pm_bid"] = (df["pm_mid"] * (1 - spread)).clip(0, 1)
    df["pm_ask"] = (df["pm_mid"] * (1 + spread)).clip(0, 1)
    df["pm_spread"] = df["pm_ask"] - df["pm_bid"]

    df["pm_liquidity_proxy"] = g["volume"].shift(1)

    df["ret_1h"] = g["close"].pct_change()
    df["pm_momentum_1h"] = g["ret_1h"].shift(1)
    df["pm_momentum_1h_time"] = g["timestamp_utc"].shift(1)

    df["log_ret"] = np.log(df["close"] / g["close"].shift(1))
    df["log_ret"] = df["log_ret"].replace([np.inf, -np.inf], np.nan)
    rolling = (
        df.groupby("market_id")["log_ret"]
        .rolling(window=vol_window, min_periods=vol_min)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["pm_volatility"] = rolling
    df["pm_volatility"] = df.groupby("market_id")["pm_volatility"].shift(1)
    df["pm_volatility_time"] = g["timestamp_utc"].shift(1)

    return df


def _merge_asof_feature(
    base: pd.DataFrame,
    other: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    if base.empty or other.empty:
        return base
    # merge_asof requires global sorting on the 'on' key (timestamp_utc)
    base_sorted = base.sort_values(["timestamp_utc", "market_id"]).reset_index(drop=True)
    other_sorted = other.sort_values(["timestamp_utc", "market_id"]).reset_index(drop=True)
    # Drop rows with NaT in timestamp_utc or null market_id to ensure valid merge
    base_sorted = base_sorted.dropna(subset=["timestamp_utc", "market_id"])
    other_sorted = other_sorted.dropna(subset=["timestamp_utc", "market_id"])
    if other_sorted.empty:
        return base_sorted
    merged = pd.merge_asof(
        base_sorted,
        other_sorted[["timestamp_utc", "market_id"] + feature_cols],
        on="timestamp_utc",
        by="market_id",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def _build_momentum_5m(bars_5m: pd.DataFrame) -> pd.DataFrame:
    if bars_5m.empty:
        return bars_5m
    bars_5m = bars_5m.sort_values(["market_id", "timestamp_utc"]).copy()
    g = bars_5m.groupby("market_id")
    bars_5m["ret_5m"] = g["close"].pct_change()
    bars_5m["pm_momentum_5m"] = g["ret_5m"].shift(1)
    bars_5m["pm_momentum_5m_time"] = g["timestamp_utc"].shift(1)
    return bars_5m


def _build_momentum_1d(bars_1h: pd.DataFrame) -> pd.DataFrame:
    if bars_1h.empty:
        return bars_1h
    df = bars_1h.copy()
    df["date"] = df["timestamp_utc"].dt.date
    # Take the last bar of each day for each market
    df = df.sort_values(["market_id", "timestamp_utc"]).groupby(["market_id", "date"]).tail(1).copy()
    # Keep the original timestamp instead of artificially setting to 23:59:59
    # This preserves the actual last bar timestamp and avoids duplicate timestamps
    df = df.sort_values(["market_id", "timestamp_utc"]).reset_index(drop=True)
    g = df.groupby("market_id", sort=False)
    df["ret_1d"] = g["close"].pct_change()
    df["pm_momentum_1d"] = g["ret_1d"].shift(1)
    df["pm_momentum_1d_time"] = g["timestamp_utc"].shift(1)
    return df


def _attach_dim_market(base: pd.DataFrame, dim_market: pd.DataFrame) -> pd.DataFrame:
    dim = dim_market.copy()
    dim["ticker"] = dim["ticker"].astype(str).str.upper()
    dim["threshold"] = dim["threshold"].apply(_normalize_threshold)
    dim["expiry_date_utc"] = pd.to_datetime(dim["expiry_date_utc"], utc=True, errors="coerce")
    dim["resolution_time_utc"] = pd.to_datetime(dim["resolution_time_utc"], utc=True, errors="coerce")

    keep_cols = [
        "market_id",
        "condition_id",
        "ticker",
        "threshold",
        "expiry_date_utc",
        "resolution_time_utc",
    ]
    dim = dim[[c for c in keep_cols if c in dim.columns]]

    merged = base.merge(dim, on="market_id", how="left")
    merged["expiry_date"] = merged["expiry_date_utc"].dt.date

    return merged


def _attach_prn_features(base: pd.DataFrame, prn: pd.DataFrame) -> pd.DataFrame:
    if base.empty or prn.empty:
        base["prn_asof_time"] = pd.NaT
        return base

    prn = prn.sort_values(["ticker", "threshold", "expiry_date", "asof_time"])

    base = base.sort_values(["ticker", "threshold", "expiry_date", "timestamp_utc"])

    merged = pd.merge_asof(
        base,
        prn,
        left_on="timestamp_utc",
        right_on="asof_time",
        by=["ticker", "threshold", "expiry_date"],
        direction="backward",
        allow_exact_matches=False,
    )

    merged = merged.rename(columns={"asof_time": "prn_asof_time"})
    return merged


def _attach_prn_features_daily(base: pd.DataFrame, prn: pd.DataFrame) -> pd.DataFrame:
    if base.empty or prn.empty:
        return base

    prn_cols = [c for c in PRN_COL_CANDIDATES if c in prn.columns]
    keep_cols = ["ticker", "threshold", "expiry_date", "snapshot_date", "asof_time"] + prn_cols
    prn = prn[keep_cols].rename(columns={"asof_time": "prn_asof_time"})
    prn = prn.dropna(subset=["ticker", "threshold", "expiry_date", "snapshot_date", "prn_asof_time"])

    base = base.dropna(subset=["ticker", "threshold", "expiry_date", "decision_date"])
    merged = base.merge(
        prn,
        left_on=["ticker", "threshold", "expiry_date", "decision_date"],
        right_on=["ticker", "threshold", "expiry_date", "snapshot_date"],
        how="inner",
    )
    return merged


def _load_pnl_conditions(run_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if run_dir:
        entities = SubgraphClient.entities_from_run(run_dir)
        df = pd.DataFrame(entities)
        return df if not df.empty else None
    return None


def _load_positions(run_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if run_dir:
        entities = SubgraphClient.entities_from_run(run_dir)
        df = pd.DataFrame(entities)
        return df if not df.empty else None
    return None


def _fetch_pnl_and_positions() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        client = SubgraphClient()
        pnl_run = client.pull("pnlConditions")
        pnl = pd.DataFrame(client.entities_from_run(pnl_run.run_dir))
    except Exception:
        pnl = None

    try:
        client = SubgraphClient()
        pos_run = client.pull("positions")
        pos = pd.DataFrame(client.entities_from_run(pos_run.run_dir))
    except Exception:
        pos = None

    return pnl if pnl is not None and not pnl.empty else None, pos if pos is not None and not pos.empty else None


def _chunk_list(values: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [values]
    return [values[i : i + size] for i in range(0, len(values), size)]


def _fetch_labels_filtered(dim_market: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        from polymarket.subgraph_client import SubgraphClient
        from polymarket.graphql_queries import get_query
    except Exception as exc:
        print(f"[features] Subgraph import failed: {exc}")
        return None, None

    condition_ids = (
        dim_market.get("condition_id")
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
        if "condition_id" in dim_market.columns
        else []
    )
    condition_ids = sorted({c for c in condition_ids if c})

    token_ids: List[str] = []
    for col in ("outcome_yes_token_id", "outcome_no_token_id"):
        if col in dim_market.columns:
            token_ids.extend(
                dim_market[col].dropna().astype(str).str.strip().tolist()
            )
    token_ids = sorted({t for t in token_ids if t})

    if not condition_ids and not token_ids:
        print("[features] No condition/token IDs available for filtered label fetch.")
        return None, None

    try:
        client = SubgraphClient()
    except Exception as exc:
        print(f"[features] Subgraph not configured: {exc}")
        return None, None

    pnl_df: Optional[pd.DataFrame] = None
    positions_df: Optional[pd.DataFrame] = None

    if condition_ids:
        try:
            sq = get_query("pnlConditionsById")
            chunks = _chunk_list(condition_ids, 200)
            pnl_rows: List[dict] = []
            for idx, chunk in enumerate(chunks, 1):
                pnl_rows.extend(
                    client.fetch_all(sq, variable_overrides={"conditionIds": chunk})
                )
                if idx == 1 or idx % 5 == 0 or idx == len(chunks):
                    print(f"[features] pnlConditions chunks {idx}/{len(chunks)}")
            if pnl_rows:
                pnl_df = pd.DataFrame(pnl_rows)
        except Exception as exc:
            print(f"[features] Filtered pnlConditions fetch failed: {exc}")

    if token_ids:
        try:
            sq = get_query("positionsByTokenIds")
            chunks = _chunk_list(token_ids, 200)
            pos_rows: List[dict] = []
            for idx, chunk in enumerate(chunks, 1):
                pos_rows.extend(
                    client.fetch_all(sq, variable_overrides={"tokenIds": chunk})
                )
                if idx == 1 or idx % 5 == 0 or idx == len(chunks):
                    print(f"[features] positions chunks {idx}/{len(chunks)}")
            if pos_rows:
                positions_df = pd.DataFrame(pos_rows)
        except Exception as exc:
            print(f"[features] Filtered positions fetch failed: {exc}")

    if pnl_df is not None and not pnl_df.empty:
        print(f"[features] Filtered pnlConditions rows={len(pnl_df)}")
    if positions_df is not None and not positions_df.empty:
        print(f"[features] Filtered positions rows={len(positions_df)}")

    return pnl_df if pnl_df is not None and not pnl_df.empty else None, positions_df if positions_df is not None and not positions_df.empty else None


def _build_label_map(dim_market: pd.DataFrame, pnl: Optional[pd.DataFrame], positions: Optional[pd.DataFrame]) -> Dict[str, Optional[int]]:
    label_map: Dict[str, Optional[int]] = {}
    if pnl is None or pnl.empty:
        return label_map
    if "condition_id" not in dim_market.columns:
        print("[WARN] dim_market missing condition_id; cannot build labels.")
        return label_map

    pos_map: Dict[Tuple[str, str], Optional[int]] = {}
    if positions is not None and not positions.empty:
        for _, row in positions.iterrows():
            condition = str(row.get("condition") or "").strip()
            token_id = str(row.get("id") or "").strip()
            outcome_idx = row.get("outcomeIndex")
            if condition and token_id:
                try:
                    pos_map[(condition, token_id)] = int(outcome_idx)
                except Exception:
                    pos_map[(condition, token_id)] = None

    dim = dim_market.copy()
    dim["condition_id"] = dim["condition_id"].astype(str)
    dim_groups = {k: v for k, v in dim.groupby("condition_id", sort=False)}

    for _, row in pnl.iterrows():
        condition_id = str(row.get("id") or "").strip()
        if not condition_id:
            continue
        payout_num = row.get("payoutNumerators")
        payout_den = row.get("payoutDenominator")
        if not isinstance(payout_num, list) or payout_den in (None, 0, "0"):
            continue
        try:
            payout_den = float(payout_den)
        except Exception:
            continue
        if payout_den == 0:
            continue

        subset = dim_groups.get(condition_id)
        if subset is None:
            continue
        for _, mrow in subset.iterrows():
            yes_token = str(mrow.get("outcome_yes_token_id") or "").strip()
            no_token = str(mrow.get("outcome_no_token_id") or "").strip()
            idx_yes = pos_map.get((condition_id, yes_token)) if pos_map else None
            idx_no = pos_map.get((condition_id, no_token)) if pos_map else None
            if idx_yes is None or idx_no is None:
                # Fallback: assume yes is index 0, no is index 1
                idx_yes, idx_no = 0, 1
            if idx_yes >= len(payout_num) or idx_no >= len(payout_num):
                continue
            payout_yes = float(payout_num[idx_yes]) / payout_den
            payout_no = float(payout_num[idx_no]) / payout_den
            if payout_yes == payout_no:
                label = None
            else:
                label = 1 if payout_yes > payout_no else 0
            label_map[str(mrow.get("market_id"))] = label

    return label_map


def _write_outputs(df: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(out_dir)

    features_path = out_dir / "decision_features.parquet"
    if not df.empty:
        try:
            df.to_parquet(features_path, index=False)
        except Exception:
            features_path = out_dir / "decision_features.csv"
            df.to_csv(features_path, index=False)

    feature_cols = [c for c in df.columns if c.startswith("pm_") or c in PRN_COL_CANDIDATES]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "decision_time_col": "timestamp_utc",
        "label_col": "label",
        "features": feature_cols,
        "categorical_features": ["ticker"],
        "numeric_features": [c for c in feature_cols if c not in {"ticker"}],
    }
    manifest_path = out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return features_path, manifest_path


def _leak_check_allow_na(
    df: pd.DataFrame,
    decision_col: str,
    feature_time_cols: List[str],
) -> pd.Series:
    t_decision = pd.to_datetime(df[decision_col], utc=True, errors="coerce")
    mask = pd.Series(True, index=df.index)
    for col in feature_time_cols:
        t_feat = pd.to_datetime(df[col], utc=True, errors="coerce")
        mask &= (t_feat < t_decision) | t_feat.isna()
    return mask


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build time-safe Polymarket + pRN decision dataset.")
    parser.add_argument("--dim-market", type=str, default=str(DEFAULT_DIM_PATH), help="dim_market path.")
    parser.add_argument("--bars-dir", type=str, default=str(DEFAULT_BARS_DIR), help="bars directory.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory.")
    parser.add_argument("--prn-dataset", type=str, default=None, help="Path to pRN dataset CSV.")
    parser.add_argument("--decision-freq", type=str, default="1d", help="Decision freq (daily only, use 1d).")
    parser.add_argument("--spread-bps", type=float, default=200.0, help="Spread proxy bps for bid/ask.")
    parser.add_argument("--vol-window", type=int, default=24, help="Volatility rolling window (bars).")
    parser.add_argument("--vol-min", type=int, default=6, help="Volatility min periods.")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD.")
    parser.add_argument("--allow-leak", action="store_true", help="Do not fail if leak check fails.")
    parser.add_argument("--pnl-run-dir", type=str, default=None, help="Optional pnlConditions run dir.")
    parser.add_argument("--positions-run-dir", type=str, default=None, help="Optional positions run dir.")
    parser.add_argument(
        "--skip-subgraph-labels",
        action="store_true",
        help="Skip fetching labels from the subgraph (labels will be NaN).",
    )
    parser.add_argument("--prn-asof-tz", type=str, default="America/New_York", help="Timezone for pRN asof_date.")
    parser.add_argument(
        "--prn-asof-close-time",
        type=str,
        default="16:00",
        help="Local market close time for pRN asof_date (HH:MM or HH:MM:SS).",
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode with detailed timing.")
    parser.add_argument("--profile-output", type=str, default=None, help="Write cProfile stats to file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not _is_daily_freq(args.decision_freq):
        raise ValueError(
            "Daily-only mode: decision_freq must be '1d' because pRN snapshots are daily."
        )

    cfg = Config(
        dim_market_path=Path(args.dim_market),
        bars_dir=Path(args.bars_dir),
        out_dir=Path(args.out_dir),
        prn_dataset=Path(args.prn_dataset) if args.prn_dataset else None,
        decision_freq=args.decision_freq,
        spread_bps=args.spread_bps,
        vol_window=args.vol_window,
        vol_min=args.vol_min,
        start_date=args.start_date,
        end_date=args.end_date,
        fail_on_leak=not args.allow_leak,
        prn_asof_tz=args.prn_asof_tz,
        prn_asof_close_time=args.prn_asof_close_time,
        profile=args.profile,
        profile_output=Path(args.profile_output) if args.profile_output else None,
    )

    # Initialize profiling
    stats = ProfileStats() if cfg.profile else None
    profiler = cProfile.Profile() if cfg.profile_output else None
    if profiler:
        profiler.enable()

    total_steps = 6

    def _progress(step: int, label: str, step_id: str) -> None:
        print(
            f"[features] PROGRESS {step}/{total_steps} step={step_id} label={label}",
            flush=True,
        )

    print(f"[features] Loading dim_market: {cfg.dim_market_path}")
    with ProfileContext(stats, "load_dim_market"):
        dim_market = _load_dim_market(cfg.dim_market_path)
        market_ids = [str(m) for m in dim_market["market_id"].dropna().unique() if str(m).strip()]
    if stats:
        stats.increment_count("dim_market_rows", len(dim_market))
    _progress(1, "dim_market_loaded", "dim_market")

    prn_path = cfg.prn_dataset or _find_latest_prn_dataset()
    if prn_path is None or not prn_path.exists():
        raise ValueError("pRN dataset is required. Provide --prn-dataset.")

    user_start = pd.to_datetime(cfg.start_date).date() if cfg.start_date else None
    user_end = pd.to_datetime(cfg.end_date).date() if cfg.end_date else None

    filter_tickers = None
    if "ticker" in dim_market.columns:
        tickers_raw = dim_market["ticker"].dropna().astype(str).str.upper().tolist()
        tickers_raw = [t for t in tickers_raw if t and t != "NAN"]
        filter_tickers = set(tickers_raw) if tickers_raw else None

    filter_thresholds = None
    if "threshold" in dim_market.columns:
        thresholds = dim_market["threshold"].apply(_normalize_threshold).dropna().tolist()
        filter_thresholds = set(thresholds) if thresholds else None

    filter_expiries = None
    for col in ("expiry_date_utc", "expiry_date", "resolution_time_utc"):
        if col in dim_market.columns:
            expiries = pd.to_datetime(dim_market[col], errors="coerce").dt.date.dropna().tolist()
            if expiries:
                filter_expiries = set(expiries)
                break

    if filter_tickers or filter_thresholds or filter_expiries or user_start or user_end:
        print(
            "[features] pRN filters: "
            f"tickers={len(filter_tickers) if filter_tickers else 0} "
            f"thresholds={len(filter_thresholds) if filter_thresholds else 0} "
            f"expiries={len(filter_expiries) if filter_expiries else 0} "
            f"dates={user_start or '--'}->{user_end or '--'}"
        )
    print(f"[features] Loading pRN dataset: {prn_path}")
    with ProfileContext(stats, "load_prn_dataset"):
        prn = _load_prn_dataset(
            prn_path,
            prn_asof_tz=cfg.prn_asof_tz,
            prn_asof_close_time=cfg.prn_asof_close_time,
            tickers=filter_tickers,
            thresholds=filter_thresholds,
            expiry_dates=filter_expiries,
            date_start=user_start,
            date_end=user_end,
        )
    if stats:
        stats.increment_count("prn_rows_loaded", len(prn))
    if prn.empty:
        raise ValueError("pRN dataset is empty after loading. Cannot proceed.")
    _progress(2, "prn_loaded", "prn_dataset")

    prn_dates = pd.to_datetime(prn["snapshot_date"], errors="coerce").dt.date.dropna().unique().tolist()
    if not prn_dates:
        raise ValueError("pRN dataset has no snapshot dates to align.")
    prn_min, prn_max = min(prn_dates), max(prn_dates)

    effective_start = max(filter(None, [user_start, prn_min])) if (user_start or prn_min) else None
    effective_end = min(filter(None, [user_end, prn_max])) if (user_end or prn_max) else None
    if effective_start and effective_end and effective_start > effective_end:
        raise ValueError(
            "No overlapping dates between requested range and pRN snapshots. "
            f"Requested: {cfg.start_date} -> {cfg.end_date}; "
            f"pRN: {prn_min} -> {prn_max}."
        )

    if cfg.bars_dir == DEFAULT_BARS_DIR and DEFAULT_BARS_HISTORY_DIR.exists():
        print(f"[features] Using bars_history for daily freq: {DEFAULT_BARS_HISTORY_DIR}")
        cfg = Config(
            dim_market_path=cfg.dim_market_path,
            bars_dir=DEFAULT_BARS_HISTORY_DIR,
            out_dir=cfg.out_dir,
            prn_dataset=cfg.prn_dataset,
            decision_freq=cfg.decision_freq,
            spread_bps=cfg.spread_bps,
            vol_window=cfg.vol_window,
            vol_min=cfg.vol_min,
            start_date=cfg.start_date,
            end_date=cfg.end_date,
            fail_on_leak=cfg.fail_on_leak,
            prn_asof_tz=cfg.prn_asof_tz,
            prn_asof_close_time=cfg.prn_asof_close_time,
            profile=cfg.profile,
            profile_output=cfg.profile_output,
        )

    bars_start = effective_start.isoformat() if effective_start else cfg.start_date
    bars_end = effective_end.isoformat() if effective_end else cfg.end_date

    prn = prn[prn["snapshot_date"].between(effective_start or prn_min, effective_end or prn_max)].copy()

    print(f"[features] Loading bars: {cfg.bars_dir} ({bars_start or '--'} -> {bars_end or '--'})")
    with ProfileContext(stats, "load_bars"):
        bars_base = _load_bars(cfg.bars_dir, cfg.decision_freq, market_ids, bars_start, bars_end)
    if stats:
        stats.increment_count("bars_rows_loaded", len(bars_base))
    _progress(3, "bars_loaded", "bars")
    if bars_base.empty:
        print("[features] No base bars found.")
        return

    with ProfileContext(stats, "apply_daily_decision_time"):
        bars_base = _apply_daily_decision_time(
            bars_base,
            tz_name=cfg.prn_asof_tz,
            close_time=cfg.prn_asof_close_time,
            decision_offset_s=1,
        )

    with ProfileContext(stats, "build_base_features"):
        base = _build_base_features(bars_base, cfg.spread_bps, cfg.vol_window, cfg.vol_min)

    base["pm_momentum_5m"] = np.nan
    base["pm_momentum_5m_time"] = pd.NaT
    if "pm_momentum_1h" in base.columns:
        base["pm_momentum_1d"] = base["pm_momentum_1h"]
        base["pm_momentum_1d_time"] = base.get("pm_momentum_1h_time")
    else:
        base["pm_momentum_1d"] = np.nan
        base["pm_momentum_1d_time"] = pd.NaT

    base = _attach_dim_market(base, dim_market)
    _progress(4, "base_features_built", "base_features")

    base["pm_time_to_resolution"] = (
        base["resolution_time_utc"] - base["timestamp_utc"]
    ).dt.total_seconds() / 86_400.0

    base["decision_date"] = pd.to_datetime(base["timestamp_utc"], utc=True, errors="coerce").dt.date
    base_tickers = base["ticker"].dropna().astype(str).str.upper().unique().tolist()
    base_thresholds = base["threshold"].dropna().unique().tolist()
    base_expiries = base["expiry_date"].dropna().unique().tolist()
    if base_tickers or base_thresholds or base_expiries:
        mask = pd.Series(True, index=prn.index)
        if base_tickers:
            mask &= prn["ticker"].isin(base_tickers)
        if base_thresholds:
            mask &= prn["threshold"].isin(base_thresholds)
        if base_expiries:
            mask &= prn["expiry_date"].isin(base_expiries)
        prn = prn[mask]

    prn_dates = set(prn["snapshot_date"].dropna().unique())
    base_dates = set(base["decision_date"].dropna().unique())
    common_dates = sorted(prn_dates & base_dates)
    if not common_dates:
        prn_range = (min(prn_dates), max(prn_dates)) if prn_dates else (None, None)
        base_range = (min(base_dates), max(base_dates)) if base_dates else (None, None)
        raise ValueError(
            "No overlapping dates between pRN snapshots and Polymarket bars. "
            f"pRN dates: {prn_range[0]} -> {prn_range[1]}; "
            f"bars dates: {base_range[0]} -> {base_range[1]}."
        )

    base = base[base["decision_date"].isin(common_dates)].copy()
    prn = prn[prn["snapshot_date"].isin(common_dates)].copy()

    print("[features] Attaching pRN snapshots to bars...")
    with ProfileContext(stats, "attach_prn_features"):
        base = _attach_prn_features_daily(base, prn)
    if stats:
        stats.increment_count("features_rows_after_merge", len(base))
    if base.empty:
        raise ValueError(
            "No overlapping events after merging pRN and Polymarket data on common dates. "
            "Check ticker/threshold/expiry alignment."
        )

    with ProfileContext(stats, "fetch_labels"):
        pnl = _load_pnl_conditions(Path(args.pnl_run_dir)) if args.pnl_run_dir else None
        positions = _load_positions(Path(args.positions_run_dir)) if args.positions_run_dir else None
        if (pnl is None or positions is None) and not args.skip_subgraph_labels:
            if stats:
                stats.api_calls += 1
            filtered_pnl, filtered_positions = _fetch_labels_filtered(dim_market)
            pnl = pnl or filtered_pnl
            positions = positions or filtered_positions
            if pnl is None or positions is None:
                if stats:
                    stats.api_calls += 1
                fetched_pnl, fetched_positions = _fetch_pnl_and_positions()
                pnl = pnl or fetched_pnl
                positions = positions or fetched_positions
        elif args.skip_subgraph_labels:
            print("[features] Skipping subgraph label fetch (labels will be NaN).")

    with ProfileContext(stats, "build_label_map"):
        label_map = _build_label_map(dim_market, pnl, positions)
    if label_map:
        base["label"] = base["market_id"].map(label_map)
    else:
        base["label"] = np.nan

    base["schema_version"] = SCHEMA_VERSION

    feature_time_cols = [
        "pm_last_time",
        "pm_momentum_5m_time",
        "pm_momentum_1h_time",
        "pm_momentum_1d_time",
        "pm_volatility_time",
        "prn_asof_time",
    ]
    feature_time_cols = [c for c in feature_time_cols if c in base.columns]
    if feature_time_cols:
        base["leak_check_passed"] = _leak_check_allow_na(
            base,
            decision_col="timestamp_utc",
            feature_time_cols=feature_time_cols,
        )
    else:
        base["leak_check_passed"] = True

    if cfg.fail_on_leak and not bool(base["leak_check_passed"].all()):
        bad = base.loc[~base["leak_check_passed"]]
        raise ValueError(
            f"Anti-leak violation: {len(bad):,} / {len(base):,} rows have feature times >= decision time."
        )

    _progress(5, "labels_ready", "labels")

    with ProfileContext(stats, "write_outputs"):
        features_path, manifest_path = _write_outputs(base, cfg.out_dir)
    _progress(6, "outputs_written", "write_outputs")

    print("[features] rows=", len(base))
    print("[features] output=", features_path)
    print("[features] manifest=", manifest_path)

    # Profiling summary
    if profiler:
        profiler.disable()
        profile_stats = pstats.Stats(profiler)
        profile_stats.dump_stats(str(cfg.profile_output))
        print(f"[PROFILE] cProfile stats written to {cfg.profile_output}")

    if stats:
        stats.increment_count("final_feature_rows", len(base))
        stats.print_summary()


if __name__ == "__main__":
    main()
