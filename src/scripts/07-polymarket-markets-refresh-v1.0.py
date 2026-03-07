#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from polymarket.prn_loader import (
    PRN_COL_CANDIDATES,
    find_latest_prn_dataset,
    load_prn_dataset,
    normalize_threshold,
)
from polymarket.snapshot_enrichment import enrich_snapshot_features
from polymarket.weekly_history_io import (
    append_df_to_csv_with_schema,
    build_bars_from_prices,
    clean_price_history,
    fetch_price_history,
)

SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION_PRICES = "pm_weekly_prices_v1.0"
SCHEMA_VERSION_MARKETS = "pm_weekly_markets_v1.0"
SCHEMA_VERSION_BARS = "pm_bars_history_v1.0"
SCHEMA_VERSION_PRN = "pm_markets_prn_hourly_v1.0"
SCHEMA_VERSION_SNAPSHOT_DAILY = "pPM_polymarket_snapshot_daily_v1.0"

BAR_FREQS = ("1h", "1d")

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
    "expiry_date",
    "market_id",
    "event_id",
    "event_endDate",
    COL_PM_BUY,
    COL_PM_BID,
    COL_PM_ASK,
    "spot",
    "spot_asof_utc",
    "rn_asof_utc",
    "prn_asof_time",
    "snapshot_date",
    "rn_method",
    "spot_source",
]
for _col in PRN_COL_CANDIDATES + ["schema_version", "run_id"]:
    if _col not in PRN_COLUMNS:
        PRN_COLUMNS.append(_col)

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
_local_date = SNAPSHOT._local_date


# -----------------------------
# Config
# -----------------------------


@dataclass(frozen=True)
class MarketsConfig:
    tz_name: str = "America/New_York"
    prn_asof_tz: str = "America/New_York"
    prn_asof_close_time: str = "16:00"
    risk_free_rate: float = SnapshotConfig().risk_free_rate
    rv20_fallback: float = SnapshotConfig().rv20_fallback
    clob_fidelity_min: int = 60
    clob_max_range_days: int = 15
    request_timeout_s: int = 30
    sleep_between_requests_s: float = 0.15
    clob_max_workers: int = 4
    bidask_spread_bps: float = 200.0
    despike_enabled: bool = False
    despike_jump: float = 0.25
    despike_revert: float = 0.1
    clob_price_history_url: str = CLOB_PRICE_HISTORY


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


def _normalize_event_end_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.notna(ts):
            return ts.date().isoformat()
    except Exception:
        pass
    text = str(value).strip()
    return text if text else None


def date_to_utc_start(d: date) -> datetime:
    return datetime.combine(d, dt_time(0, 0), tzinfo=timezone.utc)


def date_to_utc_end(d: date) -> datetime:
    return datetime.combine(d, dt_time(23, 59, 59), tzinfo=timezone.utc)


def round_down_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


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
    print(f"[Markets] WARNING invalid close time '{value}', using 16:00.", flush=True)
    return dt_time(16, 0)


def latest_completed_close_utc(
    now_utc: datetime,
    *,
    tz_name: str,
    close_time: str,
) -> datetime:
    tz = ZoneInfo(tz_name)
    now_local = now_utc.astimezone(tz)
    close_t = _parse_close_time(close_time)
    close_local = datetime.combine(now_local.date(), close_t, tzinfo=tz)
    if now_local < close_local:
        close_local = close_local - timedelta(days=1)
    while close_local.weekday() >= 5:
        close_local = close_local - timedelta(days=1)
    return close_local.astimezone(timezone.utc)


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


def _build_spot_source(spot: pd.Series) -> pd.Series:
    source = pd.Series(pd.NA, index=spot.index, dtype="string")
    source.loc[spot.notna()] = "option_chain"
    return source


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


def _fallback_latest_run_dir() -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"No weekly history runs found: {RUNS_DIR}")
    run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No weekly history runs found: {RUNS_DIR}")
    return run_dirs[0]


def _resolve_run_dir(run_id: Optional[str]) -> Tuple[Path, Optional[str]]:
    if run_id:
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir, None

    latest_id = _load_latest_run_id()
    if latest_id:
        candidate = RUNS_DIR / latest_id
        if candidate.exists():
            return candidate, None
        fallback = _fallback_latest_run_dir()
        warning = (
            f"[Markets] WARNING latest.json run_id={latest_id} not found; "
            f"falling back to latest existing run_id={fallback.name}"
        )
        return fallback, warning

    fallback = _fallback_latest_run_dir()
    warning = (
        f"[Markets] WARNING latest.json missing or invalid; "
        f"falling back to latest existing run_id={fallback.name}"
    )
    return fallback, warning


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


def _scan_last_timestamp(
    price_history_path: Path,
    market_ids: Iterable[str],
    token_role: str = "yes",
) -> Dict[str, Optional[datetime]]:
    if not price_history_path.exists():
        return {mid: None for mid in market_ids}

    target = set(market_ids)
    max_map: Dict[str, Optional[datetime]] = {mid: None for mid in target}

    try:
        header = pd.read_csv(price_history_path, nrows=0)
        cols = set(header.columns)
    except Exception:
        cols = set()
    usecols = [c for c in ["timestamp_utc", "market_id", "token_role"] if c in cols]
    if not usecols:
        usecols = None

    try:
        for chunk in pd.read_csv(price_history_path, usecols=usecols, chunksize=100_000):
            if "market_id" not in chunk.columns:
                continue
            chunk["market_id"] = chunk["market_id"].astype(str)
            chunk = chunk[chunk["market_id"].isin(target)]
            if chunk.empty:
                continue
            if "token_role" in chunk.columns:
                chunk["token_role"] = chunk["token_role"].astype(str).str.lower()
                chunk = chunk[chunk["token_role"] == token_role]
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


def _load_price_history_slice(
    price_history_path: Path,
    market_ids: Iterable[str],
    *,
    token_roles: Optional[Iterable[str]] = None,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
) -> pd.DataFrame:
    if not price_history_path.exists():
        return pd.DataFrame()

    target = set(str(mid) for mid in market_ids)
    roles = None
    if token_roles:
        roles = {r.strip().lower() for r in token_roles if r}

    try:
        header = pd.read_csv(price_history_path, nrows=0)
        cols = set(header.columns)
    except Exception:
        cols = set()
    base_cols = ["timestamp_utc", "price", "market_id", "token_role", "token_id"]
    usecols = [c for c in base_cols if c in cols]
    if not usecols:
        usecols = None
    frames: List[pd.DataFrame] = []

    try:
        for chunk in pd.read_csv(price_history_path, usecols=usecols, chunksize=200_000):
            if "market_id" not in chunk.columns:
                continue
            chunk["market_id"] = chunk["market_id"].astype(str)
            chunk = chunk[chunk["market_id"].isin(target)]
            if chunk.empty:
                continue
            if "token_role" in chunk.columns:
                chunk["token_role"] = chunk["token_role"].astype(str).str.lower()
            else:
                chunk["token_role"] = "yes"
            if roles:
                chunk = chunk[chunk["token_role"].isin(roles)]
                if chunk.empty:
                    continue
            chunk["timestamp_utc"] = pd.to_datetime(chunk["timestamp_utc"], utc=True, errors="coerce")
            chunk = chunk.dropna(subset=["timestamp_utc"])
            if start_dt is not None:
                chunk = chunk[chunk["timestamp_utc"] >= start_dt]
            if end_dt is not None:
                chunk = chunk[chunk["timestamp_utc"] <= end_dt]
            if not chunk.empty:
                frames.append(chunk)
    except Exception as exc:
        print(f"[Markets] WARNING failed to load price_history slice: {exc}", flush=True)
        return pd.DataFrame()

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_hourly_series(history: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    if hourly_index.empty:
        return pd.DataFrame(columns=["timestamp_utc", "price"])

    base = pd.DataFrame({"timestamp_utc": hourly_index})
    if history.empty:
        base["price"] = np.nan
        return base

    hist = history.sort_values("timestamp_utc")
    merged = pd.merge_asof(
        base,
        hist[["timestamp_utc", "price"]],
        on="timestamp_utc",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged


def write_bars_replace(bars: pd.DataFrame, bars_dir: Path, freq: str) -> int:
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
        ensure_dir(path.parent)
        part = part.reindex(columns=cols)
        part.to_csv(path, index=False)
        count += 1
    return count


def append_snapshot_daily(snapshot: pd.DataFrame, snapshot_path: Path) -> int:
    if snapshot.empty:
        return 0
    ensure_dir(snapshot_path.parent)
    if not snapshot_path.exists():
        if "market_id" in snapshot.columns:
            snapshot["market_id"] = snapshot["market_id"].astype(str)
        snapshot.to_csv(snapshot_path, index=False)
        return len(snapshot)

    existing = pd.read_csv(snapshot_path)
    if "market_id" in existing.columns:
        existing["market_id"] = existing["market_id"].astype(str)
    if "market_id" in snapshot.columns:
        snapshot["market_id"] = snapshot["market_id"].astype(str)
    combined = pd.concat([existing, snapshot], ignore_index=True)
    if {"snapshot_time_utc", "market_id"}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset=["snapshot_time_utc", "market_id"], keep="last")
    combined.to_csv(snapshot_path, index=False)
    return len(combined) - len(existing)


# -----------------------------
# CLI
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
    parser.add_argument("--prn-dataset", type=str, default=None, help="Path to option-chain pRN dataset.")
    parser.add_argument("--prn-asof-tz", type=str, default=MarketsConfig().prn_asof_tz)
    parser.add_argument("--prn-asof-close-time", type=str, default=MarketsConfig().prn_asof_close_time)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = MarketsConfig(
        tz_name=args.tz,
        clob_fidelity_min=int(args.fidelity),
        bidask_spread_bps=float(args.bidask_spread_bps),
        prn_asof_tz=args.prn_asof_tz,
        prn_asof_close_time=args.prn_asof_close_time,
    )

    run_dir, run_resolution_warning = _resolve_run_dir(args.run_id)
    run_id = run_dir.name
    if run_resolution_warning:
        print(run_resolution_warning, flush=True)

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

    markets_week = markets_week.drop_duplicates(subset=["market_id"], keep="last").reset_index(drop=True)

    markets_week["threshold"] = markets_week["threshold"].apply(normalize_threshold)
    markets_week = markets_week.dropna(subset=["market_id", "threshold"])

    expiry_source = None
    for col in ("expiry_date_utc", "resolution_time_utc", "event_endDate"):
        if col in markets_week.columns:
            expiry_source = col
            break
    if expiry_source:
        markets_week["expiry_date"] = pd.to_datetime(markets_week[expiry_source], utc=True, errors="coerce").dt.date
    else:
        markets_week["expiry_date"] = week_friday
    markets_week = markets_week.dropna(subset=["expiry_date"])

    if "event_endDate" in markets_week.columns:
        markets_week["event_endDate"] = markets_week["event_endDate"].apply(_normalize_event_end_date)
    else:
        markets_week["event_endDate"] = week_friday.isoformat()

    market_ids = markets_week["market_id"].dropna().astype(str).tolist()
    if not market_ids:
        raise RuntimeError("No weekly markets with market_id to refresh.")

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
    if args.force_refresh and not args.dry_run and price_history_path.exists():
        price_history_path.unlink()

    last_ts_yes = _scan_last_timestamp(price_history_path, market_ids, token_role="yes")
    last_ts_no = _scan_last_timestamp(price_history_path, market_ids, token_role="no")

    for mid, payload in (index_markets or {}).items():
        if not payload:
            continue
        if isinstance(payload, dict):
            for role, target in [("yes", last_ts_yes), ("no", last_ts_no)]:
                ts_val = payload.get(role) or payload.get(role.upper())
                if not ts_val:
                    continue
                try:
                    ts = pd.to_datetime(ts_val, utc=True, errors="coerce")
                except Exception:
                    continue
                if pd.isna(ts):
                    continue
                prev = target.get(mid)
                if prev is None or ts.to_pydatetime() > prev:
                    target[mid] = ts.to_pydatetime()
        else:
            try:
                ts = pd.to_datetime(payload, utc=True, errors="coerce")
            except Exception:
                continue
            if pd.isna(ts):
                continue
            prev = last_ts_yes.get(mid)
            if prev is None or ts.to_pydatetime() > prev:
                last_ts_yes[mid] = ts.to_pydatetime()

    # Fetch CLOB raw prices (YES + NO)
    total_markets = len(markets_week)
    price_rows: List[pd.DataFrame] = []
    despike_adjusted = 0

    for idx, row in markets_week.iterrows():
        market_id = str(row.get("market_id"))
        ticker = row.get("ticker")
        threshold = row.get("threshold")

        progress("prices", idx + 1, total_markets)

        for token_role, token_id, last_map in [
            ("yes", row.get("yes_token_id"), last_ts_yes),
            ("no", row.get("no_token_id"), last_ts_no),
        ]:
            if not token_id or not market_id:
                continue

            last_ts = last_map.get(market_id)
            start_dt = last_ts + timedelta(seconds=1) if last_ts else date_to_utc_start(week_monday)
            if args.force_refresh:
                start_dt = date_to_utc_start(week_monday)

            end_dt = cutoff_utc
            if start_dt >= end_dt:
                continue

            try:
                history = fetch_price_history(
                    session,
                    str(token_id),
                    cfg,
                    start_dt,
                    end_dt,
                    schema_version=SCHEMA_VERSION_PRICES,
                )
            except Exception as exc:
                print(f"[Markets] Failed CLOB history for {ticker} {threshold} ({token_role}): {exc}")
                continue

            if history.empty:
                continue

            history = history.copy()
            history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True, errors="coerce")
            history = history.dropna(subset=["timestamp_utc"])
            history["market_id"] = market_id
            history["ticker"] = ticker
            history["threshold"] = threshold
            history["token_role"] = token_role
            history["fidelity_min"] = cfg.clob_fidelity_min

            history, adjusted = clean_price_history(
                history,
                cfg.despike_enabled,
                cfg.despike_jump,
                cfg.despike_revert,
            )
            despike_adjusted += adjusted

            price_rows.append(history)
            if cfg.sleep_between_requests_s:
                time.sleep(cfg.sleep_between_requests_s)

    price_history_appended = 0
    if price_rows and not args.dry_run:
        prices_out = pd.concat(price_rows, ignore_index=True)
        prices_out["timestamp_utc"] = prices_out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        append_df_to_csv_with_schema(prices_out, price_history_path)
        price_history_appended = len(prices_out)

    # Load price history slice for bars + snapshot
    history_start = date_to_utc_start(week_monday)
    price_hist = _load_price_history_slice(
        price_history_path,
        market_ids,
        token_roles=["yes", "no"],
        start_dt=history_start,
        end_dt=cutoff_utc,
    )

    bar_partitions = 0
    if not args.dry_run and not price_hist.empty:
        price_yes = price_hist[price_hist["token_role"] == "yes"].copy()
        if not price_yes.empty:
            bars_source = price_yes[["timestamp_utc", "price", "market_id"]].copy()
            for freq in BAR_FREQS:
                bars = build_bars_from_prices(bars_source, freq, schema_version=SCHEMA_VERSION_BARS)
                bar_partitions += write_bars_replace(bars, BARS_HISTORY_DIR, freq)

    # Load pRN dataset (option-chain snapshots)
    prn_path = Path(args.prn_dataset) if args.prn_dataset else find_latest_prn_dataset()
    prn_df = pd.DataFrame()
    prn_missing = False
    if prn_path is None or not prn_path.exists():
        prn_missing = True
        print("[Markets] WARNING no option-chain pRN dataset found; pRN curves will be empty.", flush=True)
    else:
        progress("prn_load", 0, 1)
        prn_df = load_prn_dataset(
            prn_path,
            prn_asof_tz=cfg.prn_asof_tz,
            prn_asof_close_time=cfg.prn_asof_close_time,
            tickers=set(markets_week["ticker"].dropna().astype(str).str.upper().tolist()),
            thresholds=set(markets_week["threshold"].dropna().tolist()),
            expiry_dates=set(markets_week["expiry_date"].dropna().tolist()),
            date_start=week_monday,
            date_end=week_friday,
        )
        progress("prn_load", 1, 1)
        if prn_df.empty:
            prn_missing = True
            print("[Markets] WARNING pRN dataset loaded but has no rows for this week.", flush=True)

    if not prn_df.empty:
        if "pRN" not in prn_df.columns and "qRN" in prn_df.columns:
            prn_df["pRN"] = 1.0 - pd.to_numeric(prn_df["qRN"], errors="coerce")
        if "pRN_raw" not in prn_df.columns and "qRN_raw" in prn_df.columns:
            prn_df["pRN_raw"] = 1.0 - pd.to_numeric(prn_df["qRN_raw"], errors="coerce")

    # Build hourly base and attach pRN
    hourly_start = round_down_hour(history_start)
    hourly_end = round_down_hour(cutoff_utc)
    hourly_index = pd.date_range(start=hourly_start, end=hourly_end, freq="1h", tz=timezone.utc)

    yes_groups: Dict[str, pd.DataFrame] = {}
    if not price_hist.empty:
        price_hist["market_id"] = price_hist["market_id"].astype(str)
        price_hist["token_role"] = price_hist["token_role"].astype(str).str.lower()
        price_yes = price_hist[price_hist["token_role"] == "yes"].copy()
        if not price_yes.empty:
            for mid, grp in price_yes.groupby("market_id"):
                yes_groups[str(mid)] = grp.sort_values("timestamp_utc")

    base_rows: List[pd.DataFrame] = []
    pm_hourly_rows: List[pd.DataFrame] = []
    progress("prn", 0, total_markets)

    for idx, row in markets_week.iterrows():
        progress("prn", idx + 1, total_markets)
        market_id = str(row.get("market_id"))
        ticker = str(row.get("ticker") or "").upper()
        threshold = row.get("threshold")

        expiry_ts = row.get("resolution_time_utc") or row.get("expiry_date_utc")
        expiry_dt = pd.to_datetime(expiry_ts, utc=True, errors="coerce")
        if pd.isna(expiry_dt):
            expiry_dt = date_to_utc_end(week_friday)
        market_end = min(expiry_dt.to_pydatetime(), cutoff_utc)
        market_hours = hourly_index[hourly_index <= market_end]
        if market_hours.empty:
            continue

        base = pd.DataFrame({"timestamp_utc": market_hours})
        base["week_monday"] = week_monday.isoformat()
        base["week_friday"] = week_friday.isoformat()
        base["week_sunday"] = week_sunday.isoformat()
        base["ticker"] = ticker
        base["threshold"] = threshold
        base["expiry_date"] = row.get("expiry_date")
        base["market_id"] = market_id
        base["event_id"] = row.get("event_id")
        base["event_endDate"] = row.get("event_endDate") or week_friday.isoformat()
        base_rows.append(base)

        hist_yes = yes_groups.get(market_id, pd.DataFrame(columns=["timestamp_utc", "price"]))
        hourly_prices = build_hourly_series(hist_yes, market_hours)
        hourly_prices["market_id"] = market_id
        pm_hourly_rows.append(hourly_prices)

    if not base_rows:
        prn_out = pd.DataFrame(columns=PRN_COLUMNS)
    else:
        hourly_base = pd.concat(base_rows, ignore_index=True)
        prn_out = hourly_base.copy()

        if not prn_df.empty:
            prn_df = prn_df.copy()
            prn_df["ticker"] = prn_df["ticker"].astype(str).str.upper()
            prn_df["threshold"] = pd.to_numeric(prn_df["threshold"], errors="coerce").round(6)
            prn_df["expiry_date"] = pd.to_datetime(prn_df["expiry_date"], errors="coerce").dt.date
            prn_df["asof_time"] = pd.to_datetime(prn_df["asof_time"], utc=True, errors="coerce")
            prn_df = prn_df.dropna(subset=["ticker", "threshold", "expiry_date", "asof_time"])

            prn_out = prn_out.sort_values(["ticker", "threshold", "expiry_date", "timestamp_utc"])
            prn_df = prn_df.sort_values(["ticker", "threshold", "expiry_date", "asof_time"])

            prn_out = pd.merge_asof(
                prn_out,
                prn_df,
                left_on="timestamp_utc",
                right_on="asof_time",
                by=["ticker", "threshold", "expiry_date"],
                direction="backward",
                allow_exact_matches=True,
                suffixes=("", "_prn"),
            )
            prn_out = prn_out.rename(columns={"asof_time": "prn_asof_time"})
        else:
            prn_out["prn_asof_time"] = pd.NaT

        if "pRN" not in prn_out.columns and "qRN" in prn_out.columns:
            prn_out["pRN"] = 1.0 - pd.to_numeric(prn_out["qRN"], errors="coerce")
        if "pRN_raw" not in prn_out.columns:
            if "qRN_raw" in prn_out.columns:
                prn_out["pRN_raw"] = 1.0 - pd.to_numeric(prn_out["qRN_raw"], errors="coerce")
            else:
                prn_out["pRN_raw"] = prn_out.get("pRN")
        if "pRN" in prn_out.columns:
            prn_out["pRN"] = pd.to_numeric(prn_out["pRN"], errors="coerce")
        if "pRN_raw" in prn_out.columns:
            prn_out["pRN_raw"] = pd.to_numeric(prn_out["pRN_raw"], errors="coerce")

        if "spot" not in prn_out.columns:
            if "S_asof_close" in prn_out.columns:
                prn_out["spot"] = prn_out["S_asof_close"]
            elif "S" in prn_out.columns:
                prn_out["spot"] = prn_out["S"]
            else:
                prn_out["spot"] = np.nan
        if "spot" in prn_out.columns:
            prn_out["spot"] = pd.to_numeric(prn_out["spot"], errors="coerce")

        if "rn_method" not in prn_out.columns:
            prn_out["rn_method"] = "option_chain"
        if "spot_source" not in prn_out.columns:
            prn_out["spot_source"] = _build_spot_source(prn_out["spot"])

        prn_out["spot_asof_utc"] = prn_out["prn_asof_time"]
        prn_out["rn_asof_utc"] = prn_out["prn_asof_time"]

        if pm_hourly_rows:
            pm_hourly = pd.concat(pm_hourly_rows, ignore_index=True)
            pm_hourly = pm_hourly.rename(columns={"price": COL_PM_BUY})
            prn_out = prn_out.merge(pm_hourly, on=["timestamp_utc", "market_id"], how="left")

        if COL_PM_BUY not in prn_out.columns:
            prn_out[COL_PM_BUY] = np.nan

        spread = cfg.bidask_spread_bps / 10_000.0
        prn_out[COL_PM_ASK] = pd.to_numeric(prn_out[COL_PM_BUY], errors="coerce")
        prn_out[COL_PM_BID] = (prn_out[COL_PM_ASK] * (1 - spread)).clip(0, 1)

        prn_out["schema_version"] = SCHEMA_VERSION_PRN
        prn_out["run_id"] = run_id

        # Leak checks
        prn_out["timestamp_utc"] = pd.to_datetime(prn_out["timestamp_utc"], utc=True, errors="coerce")
        prn_out["prn_asof_time"] = pd.to_datetime(prn_out["prn_asof_time"], utc=True, errors="coerce")
        prn_out["spot_asof_utc"] = pd.to_datetime(prn_out["spot_asof_utc"], utc=True, errors="coerce")
        prn_out["rn_asof_utc"] = pd.to_datetime(prn_out["rn_asof_utc"], utc=True, errors="coerce")

        leak_errors = []
        if (prn_out["timestamp_utc"] > cutoff_utc).any():
            leak_errors.append("timestamp_utc exceeds cutoff")
        if (prn_out["prn_asof_time"] > prn_out["timestamp_utc"]).any():
            leak_errors.append("prn_asof_time > timestamp_utc")
        if (prn_out["spot_asof_utc"] > prn_out["timestamp_utc"]).any():
            leak_errors.append("spot_asof_utc > timestamp_utc")
        if (prn_out["rn_asof_utc"] > prn_out["timestamp_utc"]).any():
            leak_errors.append("rn_asof_utc > timestamp_utc")
        if leak_errors:
            raise RuntimeError(f"Leak check failed: {', '.join(leak_errors)}")

        prn_out["timestamp_utc"] = prn_out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        prn_out["spot_asof_utc"] = prn_out["spot_asof_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        prn_out["rn_asof_utc"] = prn_out["rn_asof_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "prn_asof_time" in prn_out.columns:
            prn_out["prn_asof_time"] = prn_out["prn_asof_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        if "snapshot_date" in prn_out.columns:
            prn_out["snapshot_date"] = prn_out["snapshot_date"].astype(str)

    # Deduplicate by market_id + timestamp_utc
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
        if "market_id" in prn_out.columns:
            existing_max: Dict[str, datetime] = {}
            usecols = ["timestamp_utc", "market_id"]
            for chunk in pd.read_csv(prn_path, usecols=usecols, chunksize=100_000):
                if "market_id" not in chunk.columns:
                    continue
                chunk["timestamp_utc"] = pd.to_datetime(chunk["timestamp_utc"], utc=True, errors="coerce")
                chunk = chunk.dropna(subset=["timestamp_utc", "market_id"])
                for market_id, grp in chunk.groupby("market_id"):
                    latest = grp["timestamp_utc"].max()
                    if pd.isna(latest):
                        continue
                    prev = existing_max.get(str(market_id))
                    if prev is None or latest > prev:
                        existing_max[str(market_id)] = latest.to_pydatetime()

            if existing_max:
                max_series = pd.Series(existing_max)
                ts = pd.to_datetime(prn_out["timestamp_utc"], utc=True, errors="coerce")
                max_per_row = prn_out["market_id"].astype(str).map(max_series)
                mask = max_per_row.isna() | (ts > max_per_row)
                prn_out = prn_out[mask]
        else:
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

    # Daily snapshot export
    snapshot_rows_appended = 0
    last_snapshot_date: Optional[str] = None
    snapshot_ready = False
    snapshot_time_utc = None

    if not args.dry_run:
        snapshot_close_utc = latest_completed_close_utc(
            now_utc,
            tz_name=cfg.prn_asof_tz,
            close_time=cfg.prn_asof_close_time,
        )
        snapshot_local = snapshot_close_utc.astimezone(ZoneInfo(cfg.prn_asof_tz))
        snapshot_date = snapshot_local.date()
        if snapshot_date > week_friday:
            snapshot_date = week_friday
            snapshot_close_utc = datetime.combine(
                snapshot_date,
                _parse_close_time(cfg.prn_asof_close_time),
                tzinfo=ZoneInfo(cfg.prn_asof_tz),
            ).astimezone(timezone.utc)
        if snapshot_date < week_monday:
            print("[Markets] No completed close within the selected week; skipping snapshot_daily.", flush=True)
        elif prn_df.empty:
            print("[Markets] No pRN data available; skipping snapshot_daily.", flush=True)
        else:
            prn_snapshot = prn_df[prn_df["snapshot_date"] == snapshot_date].copy()
            if prn_snapshot.empty:
                print(f"[Markets] pRN snapshot missing for {snapshot_date}; skipping snapshot_daily.", flush=True)
            else:
                snapshot_ready = True
                snapshot_time_utc = snapshot_close_utc

                meta_cols = [
                    "ticker",
                    "threshold",
                    "expiry_date",
                    "market_id",
                    "event_id",
                    "event_endDate",
                    "market_question",
                    "market_slug",
                    "event_title",
                    "yes_token_id",
                    "no_token_id",
                    "condition_id",
                    "resolution_time_utc",
                    "expiry_date_utc",
                ]
                meta_cols = [c for c in meta_cols if c in markets_week.columns]
                market_meta = markets_week[meta_cols].copy()
                snapshot = prn_snapshot.merge(
                    market_meta,
                    on=["ticker", "threshold", "expiry_date"],
                    how="left",
                    validate="m:1",
                )
                snapshot = snapshot[snapshot["market_id"].notna()].copy()
                snapshot["market_id"] = snapshot["market_id"].astype(str)
                if snapshot.empty:
                    print("[Markets] pRN snapshot rows did not map to weekly markets; skipping snapshot_daily.", flush=True)
                    snapshot_ready = False
                    snapshot_time_utc = None
                else:
                    snapshot["snapshot_time_utc"] = snapshot_time_utc
                    snapshot["snapshot_time_local"] = snapshot_time_utc.astimezone(ZoneInfo(cfg.tz_name)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    snapshot["snapshot_session"] = "daily_close"
                    snapshot["week_monday"] = week_monday.isoformat()
                    snapshot["week_friday"] = week_friday.isoformat()
                    snapshot["week_sunday"] = week_sunday.isoformat()
                    snapshot["K"] = snapshot["threshold"]

                    expiry_ts = None
                    if "resolution_time_utc" in snapshot.columns:
                        expiry_ts = pd.to_datetime(snapshot["resolution_time_utc"], utc=True, errors="coerce")
                    if expiry_ts is None or expiry_ts.isna().all():
                        if "expiry_date_utc" in snapshot.columns:
                            expiry_ts = pd.to_datetime(snapshot["expiry_date_utc"], utc=True, errors="coerce")
                    if expiry_ts is None:
                        expiry_ts = pd.to_datetime(snapshot["event_endDate"], utc=True, errors="coerce")
                    snapshot["expiry_ts_utc"] = expiry_ts
                    snapshot["T_days"] = (
                        snapshot["expiry_ts_utc"] - pd.to_datetime(snapshot["snapshot_time_utc"], utc=True, errors="coerce")
                    ).dt.total_seconds() / (60 * 60 * 24)

                    if "S_asof_close" in snapshot.columns:
                        snapshot["S"] = pd.to_numeric(snapshot["S_asof_close"], errors="coerce")
                    elif "S" not in snapshot.columns:
                        snapshot["S"] = np.nan

                    if "pRN" not in snapshot.columns and "qRN" in snapshot.columns:
                        snapshot["pRN"] = 1.0 - pd.to_numeric(snapshot["qRN"], errors="coerce")
                    if "pRN_raw" not in snapshot.columns:
                        if "qRN_raw" in snapshot.columns:
                            snapshot["pRN_raw"] = 1.0 - pd.to_numeric(snapshot["qRN_raw"], errors="coerce")
                        else:
                            snapshot["pRN_raw"] = snapshot.get("pRN")

                    snapshot["prn_asof_utc"] = pd.to_datetime(snapshot["asof_time"], utc=True, errors="coerce")

                    if not price_hist.empty:
                        eligible = price_hist[price_hist["timestamp_utc"] <= snapshot_time_utc].copy()
                        if not eligible.empty:
                            eligible = eligible.sort_values("timestamp_utc")
                            last_trades = eligible.groupby(["market_id", "token_role"], as_index=False).tail(1)
                            yes_last = last_trades[last_trades["token_role"] == "yes"][["market_id", "price"]].rename(
                                columns={"price": "pPM_buy"}
                            )
                            no_last = last_trades[last_trades["token_role"] == "no"][["market_id", "price"]].rename(
                                columns={"price": "qPM_buy"}
                            )
                            snapshot = snapshot.merge(yes_last, on="market_id", how="left")
                            snapshot = snapshot.merge(no_last, on="market_id", how="left")

                    if "pPM_buy" not in snapshot.columns:
                        snapshot["pPM_buy"] = np.nan
                    if "qPM_buy" not in snapshot.columns:
                        snapshot["qPM_buy"] = np.nan
                    snapshot["pPM_buy"] = pd.to_numeric(snapshot["pPM_buy"], errors="coerce")
                    snapshot["qPM_buy"] = pd.to_numeric(snapshot["qPM_buy"], errors="coerce")

                    snapshot["pPM_mid"] = snapshot["pPM_buy"]
                    snapshot["qPM_mid"] = snapshot["qPM_buy"]
                    snapshot["yes_spread"] = np.nan
                    snapshot["no_spread"] = np.nan
                    snapshot["pm_ok"] = snapshot["pPM_buy"].notna() | snapshot["qPM_buy"].notna()
                    snapshot["pm_reason"] = np.where(snapshot["pm_ok"], "ok", "no_clob_trade")
                    snapshot["schema_version"] = SCHEMA_VERSION_SNAPSHOT_DAILY
                    snapshot["run_id"] = run_id

                    snapshot = enrich_snapshot_features(snapshot, cfg)

                    snapshot["snapshot_time_utc"] = pd.to_datetime(snapshot["snapshot_time_utc"], utc=True, errors="coerce")
                    snapshot["prn_asof_utc"] = pd.to_datetime(snapshot["prn_asof_utc"], utc=True, errors="coerce")
                    if (snapshot["prn_asof_utc"] > snapshot["snapshot_time_utc"]).any():
                        raise RuntimeError("Leak check failed: prn_asof_utc > snapshot_time_utc")

                    snapshot["snapshot_time_utc"] = snapshot["snapshot_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    snapshot["prn_asof_utc"] = snapshot["prn_asof_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                    snapshot_path = run_dir / "snapshot_daily.csv"
                    snapshot_rows_appended = append_snapshot_daily(snapshot, snapshot_path)
                    last_snapshot_date = snapshot_date.isoformat()

    # Append decision_features for latest close
    if snapshot_ready and last_snapshot_date and not args.dry_run:
        build_features_path = REPO_ROOT / "src" / "scripts" / "02-polymarket-build-features-v1.0.py"
        features_cmd = [
            sys.executable,
            str(build_features_path),
            "--dim-market",
            str(run_dir / "dim_market_weekly.csv"),
            "--bars-dir",
            str(BARS_HISTORY_DIR),
            "--out-dir",
            str(run_dir),
            "--decision-freq",
            "1d",
            "--start-date",
            last_snapshot_date,
            "--end-date",
            last_snapshot_date,
            "--append",
            "--skip-subgraph-labels",
            "--prn-asof-tz",
            cfg.prn_asof_tz,
            "--prn-asof-close-time",
            cfg.prn_asof_close_time,
        ]
        if prn_path is not None:
            features_cmd.extend(["--prn-dataset", str(prn_path)])

        progress("features", 0, 1)
        result = subprocess.run(features_cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout, flush=True)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr, flush=True)
            raise RuntimeError("build-features append failed")
        progress("features", 1, 1)

    # Update refresh index
    refresh_markets = dict(index_markets) if isinstance(index_markets, dict) else {}
    if not price_hist.empty:
        for (market_id, token_role), grp in price_hist.groupby(["market_id", "token_role"]):
            latest = pd.to_datetime(grp["timestamp_utc"], utc=True, errors="coerce").max()
            if pd.isna(latest):
                continue
            entry = refresh_markets.get(market_id)
            if isinstance(entry, str):
                entry = {"yes": entry}
            if not isinstance(entry, dict):
                entry = {}
            entry[str(token_role)] = latest.strftime("%Y-%m-%dT%H:%M:%SZ")
            refresh_markets[market_id] = entry

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
        "despike_adjusted": despike_adjusted,
        "prn_dataset": str(prn_path) if prn_path else None,
        "prn_missing": prn_missing,
        "last_snapshot_date": last_snapshot_date,
        "snapshot_rows_appended": snapshot_rows_appended,
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
