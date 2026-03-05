from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

FREQ_ALIASES = {
    "60m": "1h",
    "1h": "1h",
    "1d": "1D",
    "1D": "1D",
}


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
    if bars.empty:
        return 0

    bars = bars.copy()
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp_utc"], utc=True, errors="coerce")
    bars = bars.dropna(subset=["timestamp_utc"])
    bars["bar_date"] = bars["timestamp_utc"].dt.strftime("%Y-%m-%d")
    bars["timestamp_utc"] = bars["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    cols = [
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

    count = 0
    for (market_id, bar_date), part in bars.groupby(["market_id", "bar_date"]):
        path = bars_dir / freq / f"market_id={market_id}" / f"date={bar_date}" / "bars.csv"
        part = part.reindex(columns=cols)
        append_df_to_csv_with_schema(part, path)
        count += 1
    return count
