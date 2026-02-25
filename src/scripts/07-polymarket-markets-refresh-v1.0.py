#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION_PRICES = "pm_weekly_prices_v1.0"
SCHEMA_VERSION_MARKETS = "pm_weekly_markets_v1.0"
SCHEMA_VERSION_BARS = "pm_bars_history_v1.0"
SCHEMA_VERSION_PRN = "pm_markets_prn_hourly_v1.0"

# Column names (keep in one place to avoid drift).
COL_PM_BUY = "polymarket_buy"
COL_PM_BID = "polymarket_bid"
COL_PM_ASK = "polymarket_ask"

PRN_COLUMNS = [
    "timestamp_utc",
    "week_monday",
    "week_friday",
    "week_sunday",
    "ticker",
    "threshold",
    "market_id",
    "event_id",
    "event_endDate",
    COL_PM_BUY,
    COL_PM_BID,
    COL_PM_ASK,
    "spot",
    "pRN",
    "pRN_raw",
    "rn_method",
    "spot_source",
    "spot_asof_utc",
    "rn_asof_utc",
    "rv20",
    "rv20_source",
    "rv20_window",
    "dividend_yield",
    "forward_price",
    "schema_version",
    "run_id",
]

WEEKLY_HISTORY_DIR = REPO_ROOT / "src" / "data" / "raw" / "polymarket" / "weekly_history"
RUNS_DIR = WEEKLY_HISTORY_DIR / "runs"
LATEST_POINTER_PATH = WEEKLY_HISTORY_DIR / "latest.json"
BARS_HISTORY_DIR = REPO_ROOT / "src" / "data" / "analysis" / "polymarket" / "bars_history"

CLOB_PRICE_HISTORY = "https://clob.polymarket.com/prices-history"


# -----------------------------
# Snapshot helper import
# -----------------------------


def _load_snapshot_module() -> Any:
    snapshot_path = REPO_ROOT / "src" / "scripts" / "05-polymarket-fetch-snapshot-v1.0.py"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot script not found: {snapshot_path}")
    spec = importlib.util.spec_from_file_location("polymarket_snapshot", snapshot_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load snapshot module spec.")
    module = importlib.util.module_from_spec(spec)
    # Ensure dataclasses can resolve module globals during exec_module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SNAPSHOT = _load_snapshot_module()
SnapshotConfig = SNAPSHOT.Config
build_weekly_slug = SNAPSHOT.build_weekly_slug
fetch_event_by_slug = SNAPSHOT.fetch_event_by_slug
market_tradeable = SNAPSHOT.market_tradeable
discover_finishweek_event_slug = SNAPSHOT.discover_finishweek_event_slug
extract_strike_K_from_question = SNAPSHOT.extract_strike_K_from_question
compute_rv20_robust = SNAPSHOT.compute_rv20_robust
fetch_dividend_yield = SNAPSHOT.fetch_dividend_yield
risk_neutral_prob_bs_tail = SNAPSHOT.risk_neutral_prob_bs_tail
compute_forward_price = SNAPSHOT.compute_forward_price
_local_date = SNAPSHOT._local_date


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class MarketsConfig:
    tz_name: str = "America/New_York"
    risk_free_rate: float = SnapshotConfig().risk_free_rate
    clob_fidelity_min: int = 60
    clob_max_range_days: int = 15
    request_timeout_s: int = 30
    sleep_between_requests_s: float = 0.15
    clob_max_workers: int = 4
    yf_max_workers: int = 2
    bidask_spread_bps: float = 200.0
    cache_dir: Path = REPO_ROOT / "src" / "data" / "cache" / "markets"


# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except Exception as exc:
        raise ValueError(f"Invalid date '{value}' (expected YYYY-MM-DD).") from exc


def date_to_utc_start(d: date) -> datetime:
    return datetime.combine(d, dt_time(0, 0), tzinfo=timezone.utc)


def date_to_utc_end(d: date) -> datetime:
    return datetime.combine(d, dt_time(23, 59, 59), tzinfo=timezone.utc)


def round_down_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def finish_week_friday_markets(today_local: date) -> date:
    weekday = today_local.weekday()  # Mon=0 .. Sun=6
    if weekday >= 5:
        # Sat/Sun -> previous Friday
        return today_local - timedelta(days=weekday - 4)
    return today_local + timedelta(days=(4 - weekday))


def trading_week_bounds(week_friday: date) -> Tuple[date, date, date]:
    monday = week_friday - timedelta(days=4)
    sunday = monday + timedelta(days=6)
    return monday, week_friday, sunday


def compute_cutoff_utc(now_utc: datetime, tz_name: str, week_friday: date) -> datetime:
    tz = ZoneInfo(tz_name)
    now_local = now_utc.astimezone(tz)
    weekday = now_local.weekday()
    if weekday >= 5:
        # weekend -> cutoff at Friday 16:00 local
        cutoff_local = datetime.combine(week_friday, dt_time(16, 0), tzinfo=tz)
        return round_down_hour(cutoff_local.astimezone(timezone.utc))
    return round_down_hour(now_utc)


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


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


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp_path, path)


def _safe_iso(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_tickers(tickers: Optional[str]) -> List[str]:
    if not tickers:
        return []
    return [t.strip().upper() for t in tickers.split(",") if t.strip()]


def progress(stage: str, current: int, total: int) -> None:
    print(f"[Markets] PROGRESS stage={stage} current={current} total={total}", flush=True)


# -----------------------------
# Market selection
# -----------------------------


def _is_weekly_question(question: str, slug: str) -> bool:
    if not question and not slug:
        return False
    q = (question or "").lower()
    if "week" in q and ("finish" in q or "finishes" in q or "ending" in q or "close" in q):
        if "above" in q or "over" in q or "$" in q:
            return True
    if slug and "-above-on-" in slug:
        return True
    return False


def _market_end_local(market: Dict[str, Any], event: Dict[str, Any], tz_name: str) -> Optional[date]:
    for key in ("endDate", "endDateIso", "end_time", "endDateTime"):
        local = _local_date(market.get(key), tz_name)
        if local is not None:
            return local
    for key in ("endDate", "endDateIso", "end_time", "endDateTime"):
        local = _local_date(event.get(key), tz_name)
        if local is not None:
            return local
    return None


def _event_end_local(event: Dict[str, Any], tz_name: str) -> Optional[date]:
    for key in ("endDate", "endDateIso", "end_time", "endDateTime"):
        local = _local_date(event.get(key), tz_name)
        if local is not None:
            return local
    return None


def _load_latest_run_id() -> Optional[str]:
    if not LATEST_POINTER_PATH.exists():
        return None
    try:
        payload = json.loads(LATEST_POINTER_PATH.read_text())
    except Exception:
        return None
    return payload.get("run_id") if isinstance(payload, dict) else None


def _read_weekly_markets(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_weekly_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["event_id", "event_slug", "event_title", "event_endDate"])
    return pd.read_csv(path)


def _build_dim_market(markets: pd.DataFrame) -> pd.DataFrame:
    if markets.empty:
        return markets

    df = markets.copy()
    df = df.rename(
        columns={
            "market_question": "question",
            "market_slug": "slug",
            "threshold": "threshold",
        }
    )
    df["source"] = "gamma"
    df["mapping_confidence"] = 1.0
    df["outcome_yes_token_id"] = df["yes_token_id"]
    df["outcome_no_token_id"] = df["no_token_id"]
    df["schema_version"] = "pm_dim_market_weekly_v1.0"

    keep = [
        "market_id",
        "condition_id",
        "question",
        "ticker",
        "threshold",
        "expiry_date_utc",
        "resolution_time_utc",
        "outcome_yes_token_id",
        "outcome_no_token_id",
        "slug",
        "source",
        "mapping_confidence",
        "ticker_source",
        "schema_version",
    ]
    return df.reindex(columns=keep)


def _ensure_weekly_markets(
    run_dir: Path,
    week_friday: date,
    tickers: List[str],
    cfg: MarketsConfig,
    session: requests.Session,
) -> pd.DataFrame:
    markets_path = run_dir / "weekly_markets.csv"
    events_path = run_dir / "weekly_events.csv"
    dim_market_path = run_dir / "dim_market_weekly.csv"

    markets_df = _read_weekly_markets(markets_path)
    events_df = _read_weekly_events(events_path)

    week_key = week_friday.isoformat()
    if "ticker" in markets_df.columns:
        markets_df["ticker"] = markets_df["ticker"].astype(str).str.upper()
    if "week_friday" in markets_df.columns:
        week_markets = markets_df[markets_df["week_friday"] == week_key]
    else:
        week_markets = pd.DataFrame()
    existing_tickers = set(
        week_markets.get("ticker", pd.Series(dtype=str)).dropna().astype(str).str.upper().tolist()
    )
    missing = [t for t in tickers if t.upper() not in existing_tickers]

    if not missing:
        return markets_df

    tz_name = cfg.tz_name
    week_monday, _, week_sunday = trading_week_bounds(week_friday)

    new_rows: List[Dict[str, Any]] = []
    new_events: List[Dict[str, Any]] = []

    for ticker in missing:
        slug = build_weekly_slug(ticker, week_friday)
        used_slug = slug
        try:
            event = fetch_event_by_slug(session, slug, SnapshotConfig())
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                event = None
            else:
                raise

        if event is None:
            fallback = discover_finishweek_event_slug(
                ticker=ticker,
                expected_friday=week_friday,
                cfg=SnapshotConfig(),
                session=session,
            )
            if fallback:
                used_slug = fallback
                event = fetch_event_by_slug(session, fallback, SnapshotConfig())

        if not isinstance(event, dict):
            print(f"[Markets] Missing event for {ticker} week_friday={week_key}")
            continue

        end_local = _event_end_local(event, tz_name)
        if end_local != week_friday:
            print(f"[Markets] Event end mismatch for {ticker}: {end_local} != {week_friday}")
            continue

        event_id = str(event.get("id") or "").strip() or None
        event_title = event.get("title") or event.get("question")
        event_slug = str(event.get("slug") or used_slug).strip() or None

        existing_event_ids = set(
            events_df.get("event_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
        )
        if event_id and event_id not in existing_event_ids:
            new_events.append(
                {
                    "event_id": event_id,
                    "event_slug": event_slug,
                    "event_title": event_title,
                    "event_endDate": week_friday.isoformat(),
                }
            )

        markets = event.get("markets") or []
        if not isinstance(markets, list):
            continue

        for market in markets:
            ok, _reason = market_tradeable(market)
            if not ok:
                continue

            question = str(market.get("question") or market.get("title") or "").strip()
            market_slug = str(market.get("slug") or "").strip() or None
            if not _is_weekly_question(question, market_slug or event_slug or ""):
                continue

            m_end_local = _market_end_local(market, event, tz_name)
            if m_end_local != week_friday:
                continue

            K = extract_strike_K_from_question(question)
            if K is None:
                continue

            token_ids = market.get("clobTokenIds")
            if isinstance(token_ids, str):
                try:
                    token_ids = json.loads(token_ids)
                except Exception:
                    token_ids = None
            token_ids = token_ids if isinstance(token_ids, list) else []
            yes_token = str(token_ids[0]) if len(token_ids) >= 1 and token_ids[0] else None
            no_token = str(token_ids[1]) if len(token_ids) >= 2 and token_ids[1] else None

            new_rows.append(
                {
                    "event_id": event_id,
                    "event_slug": event_slug,
                    "event_title": event_title,
                    "event_endDate": week_friday.isoformat(),
                    "market_id": str(market.get("id") or "").strip() or None,
                    "condition_id": str(market.get("conditionId") or market.get("condition_id") or "").strip() or None,
                    "market_slug": market_slug or event_slug,
                    "market_question": question or None,
                    "ticker": ticker,
                    "ticker_source": "slug",
                    "threshold": float(K),
                    "week_monday": week_monday.isoformat(),
                    "week_friday": week_friday.isoformat(),
                    "week_sunday": week_sunday.isoformat(),
                    "expiry_date_utc": date_to_utc_start(week_friday).isoformat().replace("+00:00", "Z"),
                    "resolution_time_utc": date_to_utc_end(week_friday).isoformat().replace("+00:00", "Z"),
                    "yes_token_id": yes_token,
                    "no_token_id": no_token,
                    "enable_order_book": market.get("enableOrderBook"),
                    "active": market.get("active"),
                    "closed": market.get("closed"),
                    "schema_version": SCHEMA_VERSION_MARKETS,
                }
            )

    if not new_rows:
        return markets_df

    new_df = pd.DataFrame(new_rows)
    markets_df = pd.concat([markets_df, new_df], ignore_index=True)
    markets_df = markets_df.drop_duplicates(subset=["market_id"], keep="last")

    append_df_to_csv_with_schema(new_df, markets_path)

    if new_events:
        events_append = pd.DataFrame(new_events)
        append_df_to_csv_with_schema(events_append, events_path)

    if dim_market_path.exists():
        dim_existing = pd.read_csv(dim_market_path)
    else:
        dim_existing = pd.DataFrame()
    dim_new = _build_dim_market(new_df)
    dim_all = pd.concat([dim_existing, dim_new], ignore_index=True)
    dim_all = dim_all.drop_duplicates(subset=["market_id"], keep="last")
    ensure_dir(dim_market_path.parent)
    dim_all.to_csv(dim_market_path, index=False)

    return markets_df


# -----------------------------
# CLOB price history
# -----------------------------


def _payload_to_history_df(payload: dict, token_id: str) -> pd.DataFrame:
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
    df["schema_version"] = SCHEMA_VERSION_PRICES
    return df[["timestamp_utc", "price", "token_id", "schema_version"]]


def fetch_price_history(
    session: requests.Session,
    token_id: str,
    cfg: MarketsConfig,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> pd.DataFrame:
    base_params: Dict[str, Any] = {
        "market": token_id,
        "fidelity": cfg.clob_fidelity_min,
    }

    if not (start_dt or end_dt):
        params = dict(base_params)
        params.update({"interval": "max"})
        resp = session.get(CLOB_PRICE_HISTORY, params=params, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        return _payload_to_history_df(resp.json(), token_id)

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
        resp = session.get(CLOB_PRICE_HISTORY, params=params, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        frame = _payload_to_history_df(resp.json(), token_id)
        if not frame.empty:
            frames.append(frame)
        if cfg.sleep_between_requests_s > 0:
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


def build_hourly_series(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "price"])
    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc")
    if df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "price"])

    df = df.set_index("timestamp_utc")
    hourly_index = pd.date_range(start=start, end=end, freq="1h", tz=timezone.utc)
    series = df["price"].resample("1h").last().reindex(hourly_index)
    series = series.ffill()
    out = series.reset_index()
    out.columns = ["timestamp_utc", "price"]
    out = out.dropna(subset=["price"])
    return out


def _build_bars_from_prices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(["market_id", "timestamp_utc"]).copy()
    df = df.set_index("timestamp_utc")
    ohlc = (
        df.groupby("market_id")["price"]
        .resample("1h")
        .ohlc()
        .reset_index()
    )
    if ohlc.empty:
        return ohlc
    ohlc["volume"] = np.nan
    ohlc["trade_count"] = np.nan
    ohlc["schema_version"] = SCHEMA_VERSION_BARS
    return ohlc


def _write_bars(bars: pd.DataFrame, bars_dir: Path) -> int:
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
        path = bars_dir / "1h" / f"market_id={market_id}" / f"date={bar_date}" / "bars.csv"
        part = part.reindex(columns=cols)
        append_df_to_csv_with_schema(part, path)
        count += 1
    return count


def _scan_last_timestamp(
    price_history_path: Path,
    market_ids: Iterable[str],
    token_role: str = "yes",
) -> Dict[str, Optional[datetime]]:
    if not price_history_path.exists():
        return {mid: None for mid in market_ids}

    target = set(market_ids)
    max_map: Dict[str, Optional[datetime]] = {mid: None for mid in target}

    usecols = ["timestamp_utc", "market_id", "token_role"]
    try:
        for chunk in pd.read_csv(price_history_path, usecols=usecols, chunksize=100_000):
            chunk = chunk[chunk["market_id"].isin(target) & (chunk["token_role"] == token_role)]
            if chunk.empty:
                continue
            ts = pd.to_datetime(chunk["timestamp_utc"], utc=True, errors="coerce")
            chunk = chunk.assign(timestamp_utc=ts)
            for market_id, grp in chunk.groupby("market_id"):
                latest = grp["timestamp_utc"].max()
                if pd.isna(latest):
                    continue
                prev = max_map.get(market_id)
                if prev is None or latest > prev:
                    max_map[market_id] = latest.to_pydatetime()
    except Exception:
        return {mid: None for mid in market_ids}

    return max_map


# -----------------------------
# yfinance helpers
# -----------------------------


def _yf_cache_path(cache_dir: Path, ticker: str, interval: str, cache_date: str) -> Path:
    safe = ticker.replace("/", "-")
    return cache_dir / f"yf_{safe}_{interval}_{cache_date}.csv"


def _read_yf_csv(path: Path) -> Optional[pd.DataFrame]:
    """Read a yfinance-saved CSV, handling both flat and multi-level (new yfinance) headers.

    Newer yfinance writes a 3-row multi-level header:
        Price,Close,High,Low,Open,Volume
        Ticker,AAPL,AAPL,AAPL,AAPL,AAPL
        Datetime,,,,,
        2026-02-10 14:30:00+00:00,...

    The standard pd.read_csv(parse_dates=True, index_col=0) misreads this as an object
    index containing "Ticker" and "Datetime" strings, causing tz_localize to fail.
    """
    with path.open() as fh:
        first_line = fh.readline()
        second_line = fh.readline()

    first_field = first_line.split(",")[0].strip()
    second_field = second_line.split(",")[0].strip()
    is_new_format = first_field == "Price" and second_field == "Ticker"

    if is_new_format:
        # Skip the "Ticker" (row 1) and "Datetime" (row 2) rows; use the "Price" row
        # (row 0) as column names.  Column 0 becomes the DatetimeIndex.
        df = pd.read_csv(path, header=0, index_col=0, skiprows=[1, 2])
    else:
        df = pd.read_csv(path, index_col=0)

    # Explicit datetime parsing avoids the pandas "Could not infer format" UserWarning
    # and guarantees a proper UTC DatetimeIndex regardless of the offset notation used.
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[df.index.notna()]
    return df if not df.empty else None


def _load_yf_cached(ticker: str, interval: str, period: str, cache_dir: Path) -> pd.DataFrame:
    ensure_dir(cache_dir)
    cache_date = datetime.now(timezone.utc).date().isoformat()
    path = _yf_cache_path(cache_dir, ticker, interval, cache_date)
    if path.exists():
        try:
            result = _read_yf_csv(path)
            if result is not None:
                return result
        except Exception:
            pass

    df = yf.download(ticker, interval=interval, period=period, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns (new yfinance) to a single level before saving so that
    # subsequent cache reads use the simple flat-header path.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    try:
        df.to_csv(path)
    except Exception:
        pass
    return df


def _prep_yf_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    df = df.sort_index()
    return df


def _build_spot_series(
    hourly_index: pd.DatetimeIndex,
    intraday: pd.DataFrame,
    daily: pd.DataFrame,
    tz_name: str,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if intraday is not None and not intraday.empty and "Close" in intraday.columns:
        intraday = _prep_yf_index(intraday)
        close = intraday["Close"].dropna()
        if not close.empty:
            spot = close.reindex(hourly_index, method="ffill")
            spot_asof = pd.Series(close.index, index=close.index).reindex(hourly_index, method="ffill")
            source = pd.Series("yf_1h", index=hourly_index)
            return spot, spot_asof, source

    # fallback to daily close (prev close)
    if daily is None or daily.empty or "Close" not in daily.columns:
        empty = pd.Series(index=hourly_index, dtype=float)
        return empty, empty, pd.Series("none", index=hourly_index)

    daily = daily.copy()
    idx = daily.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    daily.index = idx
    daily = daily.sort_index()
    close = daily["Close"].dropna()
    if close.empty:
        empty = pd.Series(index=hourly_index, dtype=float)
        return empty, empty, pd.Series("none", index=hourly_index)

    tz = ZoneInfo(tz_name)
    local_dates = pd.Series(close.index.tz_convert(tz).date, index=close.index)
    date_to_price = {}
    date_to_ts = {}
    for ts, d in local_dates.items():
        date_to_price[d] = float(close.loc[ts])
        date_to_ts[d] = ts

    spot_vals = []
    spot_ts = []
    for ts in hourly_index:
        local_date = ts.tz_convert(tz).date()
        prev_dates = [d for d in date_to_price.keys() if d < local_date]
        if not prev_dates:
            spot_vals.append(np.nan)
            spot_ts.append(pd.NaT)
            continue
        last_date = max(prev_dates)
        spot_vals.append(date_to_price[last_date])
        spot_ts.append(date_to_ts[last_date])
    spot = pd.Series(spot_vals, index=hourly_index, dtype=float)
    spot_asof = pd.Series(spot_ts, index=hourly_index)
    source = pd.Series("yf_1d", index=hourly_index)
    return spot, spot_asof, source


# -----------------------------
# pRN computation
# -----------------------------


def _rv20_by_date(
    ticker: str,
    dates: Iterable[date],
    tz_name: str,
    cfg: MarketsConfig,
) -> Dict[date, Any]:
    result: Dict[date, Any] = {}
    snap_cfg = SnapshotConfig()
    for d in sorted(set(dates)):
        asof_utc = datetime.combine(d, dt_time(16, 0), tzinfo=ZoneInfo(tz_name)).astimezone(timezone.utc)
        result[d] = compute_rv20_robust(ticker, asof_utc, tz_name, snap_cfg)
    return result


def _compute_prn_rows(
    market_row: pd.Series,
    hourly_index: pd.DatetimeIndex,
    spot: pd.Series,
    spot_asof: pd.Series,
    spot_source: pd.Series,
    rv20_map: Dict[date, Any],
    div_yield: Optional[float],
    cfg: MarketsConfig,
    run_id: str,
    week_bounds: Tuple[date, date, date],
) -> pd.DataFrame:
    if hourly_index.empty:
        return pd.DataFrame()

    tz = ZoneInfo(cfg.tz_name)
    week_monday, week_friday, week_sunday = week_bounds

    expiry_ts = market_row.get("resolution_time_utc") or market_row.get("expiry_date_utc")
    expiry_dt = pd.to_datetime(expiry_ts, utc=True, errors="coerce")
    if pd.isna(expiry_dt):
        return pd.DataFrame()

    K = market_row.get("threshold")
    if K is None or not np.isfinite(float(K)):
        return pd.DataFrame()
    K = float(K)

    if div_yield is None or not np.isfinite(div_yield):
        div_yield = 0.0

    rows: List[Dict[str, Any]] = []
    for ts in hourly_index:
        spot_val = spot.get(ts)
        spot_ts = spot_asof.get(ts)
        source = spot_source.get(ts) if ts in spot_source.index else None
        if spot_ts is pd.NaT:
            spot_ts = None

        local_date = ts.tz_convert(tz).date()
        rv20_result = rv20_map.get(local_date)
        rv20 = rv20_result.rv20 if rv20_result is not None else np.nan
        rv20_source = rv20_result.source if rv20_result is not None else None
        rv20_window = rv20_result.window if rv20_result is not None else None

        if spot_val is None or not np.isfinite(spot_val):
            prn = np.nan
            rn_method = "no_spot"
            forward_price = np.nan
        else:
            T_years = (expiry_dt.to_pydatetime() - ts.to_pydatetime()).total_seconds() / (365.25 * 86400)
            T_years = max(T_years, 1e-6)
            prn_val = risk_neutral_prob_bs_tail(float(spot_val), K, T_years, cfg.risk_free_rate, float(rv20))
            prn = float(np.clip(prn_val, 0.0, 1.0)) if prn_val is not None else np.nan
            rn_method = "bs_rv20"
            forward_price = compute_forward_price(float(spot_val), cfg.risk_free_rate, div_yield, T_years)
            forward_price = forward_price if forward_price is not None else np.nan

        rows.append(
            {
                "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "week_monday": week_monday.isoformat(),
                "week_friday": week_friday.isoformat(),
                "week_sunday": week_sunday.isoformat(),
                "ticker": market_row.get("ticker"),
                "threshold": K,
                "market_id": market_row.get("market_id"),
                "event_id": market_row.get("event_id"),
                "event_endDate": week_friday.isoformat(),
                COL_PM_BUY: np.nan,
                # Bid/ask placeholders (filled after Polymarket join).
                COL_PM_BID: np.nan,
                COL_PM_ASK: np.nan,
                "spot": float(spot_val) if spot_val is not None and np.isfinite(spot_val) else np.nan,
                "pRN": prn,
                "pRN_raw": prn,
                "rn_method": rn_method,
                "spot_source": source,
                "spot_asof_utc": _safe_iso(spot_ts if isinstance(spot_ts, datetime) else None),
                "rn_asof_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rv20": rv20,
                "rv20_source": rv20_source,
                "rv20_window": rv20_window,
                "dividend_yield": div_yield,
                "forward_price": forward_price,
                "schema_version": SCHEMA_VERSION_PRN,
                "run_id": run_id,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh weekly Polymarket markets with hourly curves.")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--week-friday", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--tz", type=str, default=MarketsConfig().tz_name)
    parser.add_argument("--fidelity", type=int, default=MarketsConfig().clob_fidelity_min)
    parser.add_argument(
        "--bidask-spread-bps",
        type=float,
        default=MarketsConfig().bidask_spread_bps,
        help="Spread proxy (bps) used to derive polymarket_bid from polymarket_buy.",
    )
    parser.add_argument("--cache-dir", type=str, default=str(MarketsConfig().cache_dir))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = MarketsConfig(
        tz_name=args.tz,
        clob_fidelity_min=int(args.fidelity),
        bidask_spread_bps=float(args.bidask_spread_bps),
        cache_dir=Path(args.cache_dir),
    )

    run_id = args.run_id or _load_latest_run_id()
    if not run_id:
        raise RuntimeError("No run_id provided and latest.json missing.")

    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    tz = ZoneInfo(cfg.tz_name)
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)

    week_friday = _parse_iso_date(args.week_friday) if args.week_friday else finish_week_friday_markets(now_local.date())
    week_monday, week_friday, week_sunday = trading_week_bounds(week_friday)

    cutoff_utc = compute_cutoff_utc(now_utc, cfg.tz_name, week_friday)

    tickers = _normalize_tickers(args.tickers)
    if not tickers:
        tickers = SNAPSHOT.DEFAULT_TICKERS_WEEKLY.copy()

    print(f"[Markets] run_id={run_id}")
    print(f"[Markets] week_friday={week_friday.isoformat()} week_monday={week_monday.isoformat()}")
    print(f"[Markets] cutoff_utc={cutoff_utc.isoformat()}")

    session = make_session()

    # Ensure weekly markets exist
    progress("discover", 0, 1)
    markets_df = _ensure_weekly_markets(run_dir, week_friday, tickers, cfg, session)
    progress("discover", 1, 1)

    week_key = week_friday.isoformat()
    markets_df = markets_df.copy()
    if "ticker" in markets_df.columns:
        markets_df["ticker"] = markets_df["ticker"].astype(str).str.upper()
    markets_week = markets_df[(markets_df["week_friday"] == week_key) & (markets_df["ticker"].isin(tickers))]
    if markets_week.empty:
        raise RuntimeError(f"No weekly markets found for week_friday={week_key}")

    markets_week = markets_week.drop_duplicates(subset=["market_id"], keep="last")
    markets_week = markets_week.reset_index(drop=True)

    market_ids = markets_week["market_id"].dropna().astype(str).tolist()

    # Load refresh index
    refresh_index_path = run_dir / "markets_refresh_index.json"
    refresh_index: Dict[str, Any] = {}
    if refresh_index_path.exists():
        try:
            refresh_index = json.loads(refresh_index_path.read_text())
        except Exception:
            refresh_index = {}
    index_markets = refresh_index.get("markets", {}) if isinstance(refresh_index, dict) else {}

    price_history_path = run_dir / "price_history.csv"
    last_ts_map = _scan_last_timestamp(price_history_path, market_ids, token_role="yes")

    # Merge with index
    for mid, ts_str in index_markets.items():
        if not ts_str:
            continue
        try:
            ts = pd.to_datetime(ts_str, utc=True, errors="coerce")
        except Exception:
            continue
        if pd.isna(ts):
            continue
        prev = last_ts_map.get(mid)
        if prev is None or ts.to_pydatetime() > prev:
            last_ts_map[mid] = ts.to_pydatetime()

    # Fetch CLOB hourly prices
    total_markets = len(markets_week)
    price_rows: List[pd.DataFrame] = []
    price_hourly_map: Dict[str, pd.DataFrame] = {}
    bar_partitions = 0

    for idx, row in markets_week.iterrows():
        market_id = str(row.get("market_id"))
        yes_token = row.get("yes_token_id")
        ticker = row.get("ticker")
        threshold = row.get("threshold")

        progress("prices", idx + 1, total_markets)

        if not yes_token or not market_id:
            continue

        last_ts = last_ts_map.get(market_id)
        start_dt = last_ts + timedelta(seconds=1) if last_ts else date_to_utc_start(week_monday)
        if args.force_refresh:
            start_dt = date_to_utc_start(week_monday)

        end_dt = cutoff_utc
        if start_dt >= end_dt:
            continue

        try:
            history = fetch_price_history(session, str(yes_token), cfg, start_dt, end_dt)
        except Exception as exc:
            print(f"[Markets] Failed CLOB history for {ticker} {threshold}: {exc}")
            continue

        if history.empty:
            continue

        history = history.copy()
        history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True, errors="coerce")
        history = history.dropna(subset=["timestamp_utc"])

        hourly = build_hourly_series(history, start_dt, end_dt)
        if hourly.empty:
            continue

        hourly["market_id"] = market_id
        hourly["ticker"] = ticker
        hourly["threshold"] = threshold
        hourly["token_role"] = "yes"
        hourly["fidelity_min"] = cfg.clob_fidelity_min
        hourly["token_id"] = yes_token
        hourly["schema_version"] = SCHEMA_VERSION_PRICES

        price_rows.append(hourly)
        price_hourly_map[market_id] = hourly[["timestamp_utc", "price"]].copy()

        bars_in = hourly[["timestamp_utc", "price"]].copy()
        bars_in["market_id"] = market_id
        bars = _build_bars_from_prices(bars_in)
        if not args.dry_run:
            bar_partitions += _write_bars(bars, BARS_HISTORY_DIR)

    price_history_appended = 0
    if price_rows and not args.dry_run:
        prices_out = pd.concat(price_rows, ignore_index=True)
        prices_out["timestamp_utc"] = prices_out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        append_df_to_csv_with_schema(prices_out, price_history_path)
        price_history_appended = len(prices_out)

    # yfinance data for pRN
    unique_tickers = sorted({str(t) for t in markets_week["ticker"].dropna().astype(str).tolist()})
    progress("yfinance", 0, max(1, len(unique_tickers)))

    ticker_spot_series: Dict[str, Tuple[pd.Series, pd.Series, pd.Series]] = {}
    ticker_div_yield: Dict[str, Optional[float]] = {}
    ticker_rv20_map: Dict[str, Dict[date, Any]] = {}

    hourly_start = date_to_utc_start(week_monday)
    hourly_end = cutoff_utc
    hourly_index = pd.date_range(start=hourly_start, end=hourly_end, freq="1h", tz=timezone.utc)

    for idx, ticker in enumerate(unique_tickers):
        progress("yfinance", idx + 1, len(unique_tickers))
        intraday = _load_yf_cached(ticker, "1h", "7d", cfg.cache_dir)
        daily = _load_yf_cached(ticker, "1d", "400d", cfg.cache_dir)

        spot, spot_asof, source = _build_spot_series(hourly_index, intraday, daily, cfg.tz_name)
        ticker_spot_series[ticker] = (spot, spot_asof, source)

        try:
            ticker_div_yield[ticker] = fetch_dividend_yield(ticker, cutoff_utc, cfg.tz_name)
        except Exception:
            ticker_div_yield[ticker] = None

        local_dates = sorted({ts.tz_convert(tz).date() for ts in hourly_index})
        ticker_rv20_map[ticker] = _rv20_by_date(ticker, local_dates, cfg.tz_name, cfg)

    # Compute pRN and join polymarket
    prn_rows: List[pd.DataFrame] = []
    progress("prn", 0, total_markets)

    for idx, row in markets_week.iterrows():
        market_id = str(row.get("market_id"))
        ticker = str(row.get("ticker"))

        progress("prn", idx + 1, total_markets)

        # Limit per-market hours by expiry
        expiry_ts = row.get("resolution_time_utc") or row.get("expiry_date_utc")
        expiry_dt = pd.to_datetime(expiry_ts, utc=True, errors="coerce")
        if pd.isna(expiry_dt):
            continue
        market_end = min(expiry_dt.to_pydatetime(), cutoff_utc)
        market_hourly_index = hourly_index[hourly_index <= market_end]
        if market_hourly_index.empty:
            continue

        spot, spot_asof, source = ticker_spot_series.get(ticker, (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=str)))
        spot = spot.reindex(market_hourly_index)
        spot_asof = spot_asof.reindex(market_hourly_index)
        source = source.reindex(market_hourly_index)

        rv20_map = ticker_rv20_map.get(ticker, {})
        div_yield = ticker_div_yield.get(ticker)

        prn_df = _compute_prn_rows(
            row,
            market_hourly_index,
            spot,
            spot_asof,
            source,
            rv20_map,
            div_yield,
            cfg,
            run_id,
            (week_monday, week_friday, week_sunday),
        )
        if prn_df.empty:
            continue

        # Join polymarket prices
        if market_id in price_hourly_map:
            pm_prices = price_hourly_map[market_id].copy()
            pm_prices["timestamp_utc"] = pm_prices["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            prn_df = prn_df.merge(
                pm_prices.rename(columns={"price": COL_PM_BUY}),
                on="timestamp_utc",
                how="left",
            )
            prn_df[COL_PM_BUY] = prn_df[f"{COL_PM_BUY}_y"].combine_first(prn_df[f"{COL_PM_BUY}_x"])
            prn_df = prn_df.drop(columns=[c for c in prn_df.columns if c.endswith("_x") or c.endswith("_y")])
            # Bid/ask proxy: Polymarket history exposes trade prices (no historical top-of-book),
            # so treat polymarket_buy as the ask and derive a synthetic bid from a spread proxy.
            spread = cfg.bidask_spread_bps / 10_000.0
            prn_df[COL_PM_ASK] = prn_df[COL_PM_BUY]
            prn_df[COL_PM_BID] = (prn_df[COL_PM_ASK] * (1 - spread)).clip(0, 1)

        prn_rows.append(prn_df)

    if not prn_rows:
        prn_out = pd.DataFrame(columns=PRN_COLUMNS)
    else:
        prn_out = pd.concat(prn_rows, ignore_index=True)

        # Invariants
        prn_out["timestamp_utc"] = pd.to_datetime(prn_out["timestamp_utc"], utc=True, errors="coerce")
        prn_out["spot_asof_utc"] = pd.to_datetime(prn_out["spot_asof_utc"], utc=True, errors="coerce")
        prn_out["rn_asof_utc"] = pd.to_datetime(prn_out["rn_asof_utc"], utc=True, errors="coerce")

        prn_out = prn_out[prn_out["timestamp_utc"] <= cutoff_utc]
        prn_out = prn_out[(prn_out["spot_asof_utc"].isna()) | (prn_out["spot_asof_utc"] <= prn_out["timestamp_utc"])]
        prn_out = prn_out[(prn_out["rn_asof_utc"].isna()) | (prn_out["rn_asof_utc"] <= prn_out["timestamp_utc"])]

        prn_out["timestamp_utc"] = prn_out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        prn_out["spot_asof_utc"] = prn_out["spot_asof_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        prn_out["rn_asof_utc"] = prn_out["rn_asof_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Deduplicate by (timestamp_utc, ticker, threshold, event_endDate)
    prn_path = run_dir / "markets_prn_hourly.csv"

    # On force-refresh: purge this week's rows so new data fully replaces old / errored data
    if args.force_refresh and prn_path.exists():
        try:
            existing_prn = pd.read_csv(prn_path)
            if "week_friday" in existing_prn.columns:
                existing_prn = existing_prn[existing_prn["week_friday"] != week_key]
                existing_prn.to_csv(prn_path, index=False)
            else:
                prn_path.unlink()
            print(f"[Markets] force-refresh: cleared existing rows for week {week_key}", flush=True)
        except Exception as exc:
            print(f"[Markets] force-refresh: warning — could not clear prn rows: {exc}", file=sys.stderr)

    if not prn_out.empty and prn_path.exists():
        existing_max: Dict[Tuple[str, float, str], datetime] = {}
        usecols = ["timestamp_utc", "ticker", "threshold", "event_endDate"]
        for chunk in pd.read_csv(prn_path, usecols=usecols, chunksize=100_000):
            chunk["timestamp_utc"] = pd.to_datetime(chunk["timestamp_utc"], utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["timestamp_utc", "ticker", "threshold", "event_endDate"])
            for (ticker, threshold, end_date), grp in chunk.groupby(["ticker", "threshold", "event_endDate"]):
                latest = grp["timestamp_utc"].max()
                if pd.isna(latest):
                    continue
                key = (str(ticker), float(threshold), str(end_date))
                prev = existing_max.get(key)
                if prev is None or latest > prev:
                    existing_max[key] = latest.to_pydatetime()

        def _is_new_row(row: pd.Series) -> bool:
            key = (str(row.get("ticker")), float(row.get("threshold")), str(row.get("event_endDate")))
            latest = existing_max.get(key)
            if latest is None:
                return True
            ts = pd.to_datetime(row.get("timestamp_utc"), utc=True, errors="coerce")
            if pd.isna(ts):
                return False
            return ts.to_pydatetime() > latest

        mask = prn_out.apply(_is_new_row, axis=1)
        prn_out = prn_out[mask]

    progress("append", 0, 1)

    if prn_out.empty:
        print("[Markets] No new pRN rows to append.")
        if not args.dry_run and not prn_path.exists():
            append_df_to_csv_with_schema(prn_out, prn_path)
    elif not args.dry_run:
        append_df_to_csv_with_schema(prn_out, prn_path)

    progress("append", 1, 1)

    # Update refresh index
    refresh_markets = dict(index_markets) if isinstance(index_markets, dict) else {}
    for market_id, df in price_hourly_map.items():
        if df.empty:
            continue
        latest = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce").max()
        if pd.isna(latest):
            continue
        refresh_markets[market_id] = latest.strftime("%Y-%m-%dT%H:%M:%SZ")

    refresh_payload = {
        "run_id": run_id,
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "week_friday": week_friday.isoformat(),
        "markets": refresh_markets,
    }

    if not args.dry_run:
        atomic_write_json(refresh_index_path, refresh_payload)

    # Manifest
    manifest = {
        "run_id": run_id,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "week_friday": week_friday.isoformat(),
        "tickers": tickers,
        "cutoff_utc": cutoff_utc.isoformat(),
        "markets": len(markets_week),
        "price_rows_appended": price_history_appended,
        "prn_rows_appended": 0 if prn_out is None else len(prn_out),
        "bars_partitions": bar_partitions,
    }

    if not args.dry_run:
        atomic_write_json(run_dir / "markets_refresh.json", manifest)

    progress("done", 1, 1)
    print("[Markets] Refresh complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)
        sys.exit(1)
