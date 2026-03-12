from __future__ import annotations

import fcntl
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

FREQ_ALIASES = {
    "60m": "1h",
    "1h": "1h",
    "1d": "1D",
    "1D": "1D",
}

MASTER_BAR_FILE_NAMES = {
    "1h": "hourly_master.csv",
    "1d": "daily_master.csv",
}

BAR_STORAGE_COLUMNS = [
    "timestamp_utc",
    "market_id",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "schema_version",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_df_to_csv_with_schema(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    file_exists = path.exists() and path.stat().st_size > 0
    if not file_exists:
        df.to_csv(path, index=False)
        return

    existing_cols = list(pd.read_csv(path, nrows=0).columns)
    new_cols = [c for c in df.columns if c not in existing_cols]
    if not new_cols:
        df = df.reindex(columns=existing_cols)
        df.to_csv(path, mode="a", header=False, index=False)
        return

    existing = pd.read_csv(path)
    for col in new_cols:
        existing[col] = np.nan
    new_order = existing_cols + new_cols
    temp_path = path.with_suffix(path.suffix + ".tmp")
    existing = existing.reindex(columns=new_order)
    existing.to_csv(temp_path, index=False)
    df = df.reindex(columns=new_order)
    df.to_csv(temp_path, mode="a", header=False, index=False)
    os.replace(temp_path, path)


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


def canonical_bar_freq(freq: str) -> str:
    value = str(freq).strip().lower()
    if value in {"60m", "1h", "hourly"}:
        return "1h"
    if value in {"1d", "1day", "1-day", "daily"}:
        return "1d"
    raise ValueError(f"Unsupported bars frequency: {freq}")


def resolve_bars_master_path(bars_dir: Path, freq: str) -> Path:
    return bars_dir / MASTER_BAR_FILE_NAMES[canonical_bar_freq(freq)]


def resolve_bars_master_paths(
    bars_dir: Path,
    freqs: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    selected = freqs if freqs is not None else MASTER_BAR_FILE_NAMES.keys()
    out: Dict[str, Path] = {}
    for freq in selected:
        canonical = canonical_bar_freq(freq)
        out[canonical] = resolve_bars_master_path(bars_dir, canonical)
    return out


def _empty_bars_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_STORAGE_COLUMNS)


def _normalize_bars_for_storage(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return _empty_bars_frame()
    if "timestamp_utc" not in bars.columns or "market_id" not in bars.columns:
        raise ValueError("Bars frame must include timestamp_utc and market_id.")

    out = bars.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp_utc"])
    out["market_id"] = out["market_id"].astype(str).str.strip()
    out = out[out["market_id"] != ""]
    if out.empty:
        return _empty_bars_frame()

    out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out.reindex(columns=BAR_STORAGE_COLUMNS)
    out = out.drop_duplicates(subset=["market_id", "timestamp_utc"], keep="last")
    out = out.sort_values(["market_id", "timestamp_utc"], kind="mergesort").reset_index(drop=True)
    return out


def _read_existing_master(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return _empty_bars_frame()
    return pd.read_csv(path, dtype={"market_id": str}).reindex(columns=BAR_STORAGE_COLUMNS)


def _atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


@contextmanager
def _locked_master_file(path: Path) -> Iterator[None]:
    ensure_dir(path.parent)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def stage_bars(bars: pd.DataFrame, stage_path: Path) -> int:
    staged = _normalize_bars_for_storage(bars)
    if staged.empty:
        return 0
    ensure_dir(stage_path.parent)
    header = not stage_path.exists() or stage_path.stat().st_size == 0
    staged.to_csv(stage_path, mode="a", header=header, index=False)
    return len(staged)




def merge_staged_bars(stage_path: Path, bars_dir: Path, freq: str) -> int:
    if not stage_path.exists() or stage_path.stat().st_size == 0:
        return 0
    staged = pd.read_csv(stage_path, dtype={"market_id": str})
    return write_bars(staged, bars_dir, freq)


def read_master_bars_filtered(
    bars_dir: Path,
    freq: str,
    market_ids: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
    *,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    master_path = resolve_bars_master_path(bars_dir, freq)
    if not master_path.exists():
        return _empty_bars_frame()

    market_filter = {str(m) for m in market_ids} if market_ids else None
    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(master_path, chunksize=chunksize, dtype={"market_id": str}):
        if chunk.empty:
            continue
        chunk = chunk.reindex(columns=BAR_STORAGE_COLUMNS)
        if market_filter is not None:
            chunk["market_id"] = chunk["market_id"].astype(str)
            chunk = chunk[chunk["market_id"].isin(market_filter)]
            if chunk.empty:
                continue
        if start_date or end_date:
            dates = chunk["timestamp_utc"].astype(str).str[:10]
            mask = pd.Series(True, index=chunk.index)
            if start_date:
                mask &= dates >= start_date
            if end_date:
                mask &= dates <= end_date
            chunk = chunk[mask]
            if chunk.empty:
                continue
        frames.append(chunk)

    if not frames:
        return _empty_bars_frame()
    return pd.concat(frames, ignore_index=True)


def build_master_from_legacy_partitions(
    bars_dir: Path,
    freq: str,
    *,
    overwrite: bool = False,
) -> Dict[str, Any]:
    canonical = canonical_bar_freq(freq)
    legacy_root = bars_dir / canonical
    master_path = resolve_bars_master_path(bars_dir, canonical)
    files = sorted(legacy_root.glob("market_id=*/date=*/bars.csv"))

    if master_path.exists() and not overwrite:
        raise FileExistsError(f"Master bars file already exists: {master_path}")

    if not files:
        return {
            "freq": canonical,
            "legacy_root": str(legacy_root),
            "legacy_files": 0,
            "legacy_rows": 0,
            "unique_rows": 0,
            "master_path": str(master_path),
        }

    frames: List[pd.DataFrame] = []
    legacy_rows = 0
    for path in files:
        frame = pd.read_csv(path, dtype={"market_id": str})
        legacy_rows += len(frame)
        if not frame.empty:
            frames.append(frame)

    combined = _normalize_bars_for_storage(
        pd.concat(frames, ignore_index=True) if frames else _empty_bars_frame()
    )
    with _locked_master_file(master_path):
        _atomic_write_dataframe(combined, master_path)

    written = _normalize_bars_for_storage(_read_existing_master(master_path))
    if len(written) != len(combined):
        raise RuntimeError(
            f"Migrated master row count mismatch for {canonical}: "
            f"expected {len(combined)}, wrote {len(written)}."
        )

    return {
        "freq": canonical,
        "legacy_root": str(legacy_root),
        "legacy_files": len(files),
        "legacy_rows": legacy_rows,
        "unique_rows": len(combined),
        "master_path": str(master_path),
    }


def cleanup_legacy_partition_dirs(bars_dir: Path, freqs: Optional[Iterable[str]] = None) -> None:
    selected = freqs if freqs is not None else MASTER_BAR_FILE_NAMES.keys()
    for freq in selected:
        legacy_root = bars_dir / canonical_bar_freq(freq)
        if legacy_root.exists():
            shutil.rmtree(legacy_root)


def payload_to_history_df(payload: dict, token_id: str, schema_version: str) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame()
    history = payload.get("history") or []
    if not isinstance(history, list) or not history:
        return pd.DataFrame()
    df = pd.DataFrame(history)
    if df.empty:
        return df
    df = df.rename(columns={"t": "timestamp", "p": "price"})
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["timestamp", "price"])
    df = df[(df["price"] >= 0) & (df["price"] <= 1)]
    if df.empty:
        return df
    df["timestamp_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df["token_id"] = token_id
    df["schema_version"] = schema_version
    return df[["timestamp_utc", "price", "token_id", "schema_version"]]


def fetch_price_history(
    session: requests.Session,
    token_id: str,
    cfg: Any,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    *,
    schema_version: str,
) -> pd.DataFrame:
    base_params: Dict[str, Any] = {
        "market": token_id,
        "fidelity": int(cfg.clob_fidelity_min),
    }

    if not (start_dt or end_dt):
        params = dict(base_params)
        params.update({"interval": "max"})
        resp = session.get(cfg.clob_price_history_url, params=params, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        return payload_to_history_df(resp.json(), token_id, schema_version)

    start_ts = int(start_dt.timestamp()) if start_dt else 0
    end_ts = int(end_dt.timestamp()) if end_dt else int(datetime.now(timezone.utc).timestamp())
    if end_ts < start_ts:
        return pd.DataFrame()

    max_days = max(1, int(cfg.clob_max_range_days))
    max_span = max_days * 86400 - 1
    if max_span <= 0:
        max_span = 1

    frames: List[pd.DataFrame] = []
    cur_start = start_ts
    while cur_start <= end_ts:
        cur_end = min(cur_start + max_span, end_ts)
        params = dict(base_params)
        params.update({"startTs": int(cur_start), "endTs": int(cur_end)})
        resp = session.get(cfg.clob_price_history_url, params=params, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        frame = payload_to_history_df(resp.json(), token_id, schema_version)
        if not frame.empty:
            frames.append(frame)
        if getattr(cfg, "sleep_between_requests_s", 0) > 0:
            time.sleep(cfg.sleep_between_requests_s)
        cur_start = cur_end + 1

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    if start_dt is not None:
        out = out[out["timestamp_utc"] >= start_dt]
    if end_dt is not None:
        out = out[out["timestamp_utc"] <= end_dt]
    return out


def clean_price_history(
    df: pd.DataFrame,
    despike: bool,
    jump_threshold: float,
    revert_threshold: float,
) -> Tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0

    df = df.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
    df = df.reset_index(drop=True)

    if not despike or len(df) < 3:
        return df, 0

    prices = df["price"].to_numpy(dtype=float)
    cleaned = prices.copy()
    adjusted = 0

    for i in range(1, len(prices) - 1):
        prev_price = prices[i - 1]
        curr_price = prices[i]
        next_price = prices[i + 1]
        if (
            abs(curr_price - prev_price) >= jump_threshold
            and abs(curr_price - next_price) >= jump_threshold
            and abs(next_price - prev_price) <= revert_threshold
        ):
            cleaned[i] = 0.5 * (prev_price + next_price)
            adjusted += 1

    if adjusted:
        df["price_raw"] = prices
        df["price"] = np.clip(cleaned, 0.0, 1.0)

    return df, adjusted


def build_bars_from_prices(
    df: pd.DataFrame,
    freq: str,
    *,
    schema_version: str,
) -> pd.DataFrame:
    if df.empty:
        return df

    freq_alias = FREQ_ALIASES.get(freq, freq)
    df = df.sort_values(["market_id", "timestamp_utc"]).copy()
    df = df.set_index("timestamp_utc")

    ohlc = (
        df.groupby("market_id")["price"]
        .resample(freq_alias)
        .ohlc()
        .reset_index()
    )

    if ohlc.empty:
        return ohlc

    ohlc["volume"] = np.nan
    ohlc["trade_count"] = np.nan
    ohlc["schema_version"] = schema_version
    return ohlc


def write_bars(bars: pd.DataFrame, bars_dir: Path, freq: str) -> int:
    incoming = _normalize_bars_for_storage(bars)
    if incoming.empty:
        return 0

    master_path = resolve_bars_master_path(bars_dir, freq)
    with _locked_master_file(master_path):
        existing = _read_existing_master(master_path)
        combined = pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
        combined = _normalize_bars_for_storage(combined)
        _atomic_write_dataframe(combined, master_path)
    return len(incoming)
