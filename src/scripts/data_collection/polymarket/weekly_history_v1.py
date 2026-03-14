#!/usr/bin/env python3
"""
polymarket-weekly-history.py

Backfill weekly Polymarket events ("TICKER finishes week of DATE above $Z")
from the Gamma API, fetch CLOB price history, and build hourly/daily bars.
Optionally ingest subgraph trades if configured.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from support.script_paths import REPO_ROOT, SCRIPTS_ROOT, SRC_ROOT, prepend_sys_path

prepend_sys_path(REPO_ROOT)
prepend_sys_path(SRC_ROOT)
prepend_sys_path(SCRIPTS_ROOT)

from polymarket.weekly_history_io import (
    append_df_to_csv_with_schema,
    build_bars_from_prices,
    build_bars_from_trades,
    clean_price_history,
    fetch_price_history,
)
from polymarket.master_bars import (
    MASTER_BAR_COLUMNS,
    build_master_bar_rows,
    master_bar_path,
    upsert_master_bars,
)

# Endpoints
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
GAMMA_EVENT_SLUG = "https://gamma-api.polymarket.com/events/slug"
GAMMA_MARKET_SLUG = "https://gamma-api.polymarket.com/markets/slug"
CLOB_PRICE_HISTORY = "https://clob.polymarket.com/prices-history"

SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION_MARKETS = "pm_weekly_markets_v1.0"
SCHEMA_VERSION_PRICES = "pm_weekly_prices_v1.0"
SCHEMA_VERSION_BARS = "pm_bars_history_v1.0"
HISTORY_RESUME_STATE_FILENAME = ".history_resume_state.json"
LEGACY_SUBGRAPH_YES_TRADES_FILENAME = "subgraph_yes_trades.csv"
SUBGRAPH_INFO_FILENAME = "subgraph_info.json"
TOKEN_ROLES: Tuple[str, ...] = ("yes", "no")

DEFAULT_TICKERS_WEEKLY = [
    "NVDA",
    "TSLA",
    "GOOGL",
    "OPEN",
    "PLTR",
    "AAPL",
    "AMZN",
    "NFLX",
    "META",
    "MSFT",
]

DEFAULT_OUT_DIR = REPO_ROOT / "src" / "data" / "raw" / "polymarket" / "weekly_history"
DEFAULT_DIM_MARKET_PATH = REPO_ROOT / "src" / "data" / "models" / "polymarket" / "dim_market_weekly.csv"

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

COMPANY_NAME_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "open": "OPEN",
    "palantir": "PLTR",
}

@dataclass(frozen=True)
class Config:
    request_timeout_s: int = 30
    sleep_between_requests_s: float = 0.08
    gamma_page_size: int = 1000
    gamma_max_pages: int = 2000
    clob_fidelity_min: int = 60
    clob_max_range_days: int = 15
    bars_freqs: Tuple[str, ...] = ("1h", "1d")
    include_subgraph: bool = False
    max_subgraph_entities: int = 1_000_000
    despike_enabled: bool = False
    despike_jump: float = 0.25
    despike_revert: float = 0.1
    clob_price_history_url: str = CLOB_PRICE_HISTORY


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, default=str))
    temp_path.replace(path)


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _history_state_path(run_dir: Path) -> Path:
    return run_dir / HISTORY_RESUME_STATE_FILENAME


def _new_history_state(
    *,
    run_id: str,
    tickers: List[str],
    start_date: Optional[date],
    end_date: Optional[date],
    include_subgraph: bool,
    bars_freqs: Tuple[str, ...],
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "run_id": run_id,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": now,
        "updated_at_utc": now,
        "status": "running",
        "phase": "history",
        "tickers": list(tickers),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "include_subgraph": bool(include_subgraph),
        "bars_freqs": list(bars_freqs),
        "markets_total": 0,
        "completed_market_ids": [],
        "failed_market_ids": [],
        "subgraph_completed": False,
        "bars_completed": False,
        "manifest_written": False,
    }


def _load_history_state(run_dir: Path) -> Dict[str, Any]:
    return _safe_read_json(_history_state_path(run_dir))


def _write_history_state(run_dir: Path, payload: Dict[str, Any]) -> None:
    state = dict(payload)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(_history_state_path(run_dir), state)


def _completed_market_ids_from_state(state: Dict[str, Any]) -> set[str]:
    values = state.get("completed_market_ids")
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if value is not None and str(value).strip()}


def _failed_market_ids_from_state(state: Dict[str, Any]) -> set[str]:
    values = state.get("failed_market_ids")
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if value is not None and str(value).strip()}


def _purge_market_rows(path: Path, market_id: Any) -> int:
    if not path.exists():
        return 0
    try:
        df = pd.read_csv(path)
    except Exception:
        return 0
    if "market_id" not in df.columns:
        return 0
    market_id_text = str(market_id)
    before = len(df)
    filtered = df[df["market_id"].astype(str) != market_id_text].copy()
    removed = before - len(filtered)
    if removed <= 0:
        return 0
    temp_path = path.with_suffix(path.suffix + ".tmp")
    filtered.to_csv(temp_path, index=False)
    temp_path.replace(path)
    return removed


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        return sum(1 for _ in handle)


def _load_yes_price_history(prices_path: Path) -> pd.DataFrame:
    if not prices_path.exists():
        return pd.DataFrame(columns=["timestamp_utc", "price", "market_id"])
    try:
        prices = pd.read_csv(prices_path)
    except Exception:
        return pd.DataFrame(columns=["timestamp_utc", "price", "market_id"])
    required = {"timestamp_utc", "price", "market_id"}
    if not required.issubset(prices.columns):
        return pd.DataFrame(columns=["timestamp_utc", "price", "market_id"])
    if "token_role" in prices.columns:
        prices = prices[prices["token_role"].astype(str).str.lower() == "yes"].copy()
    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True, errors="coerce")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices["market_id"] = prices["market_id"].astype(str)
    prices = prices.dropna(subset=["timestamp_utc", "price", "market_id"])
    return prices[["timestamp_utc", "price", "market_id"]].sort_values(
        ["market_id", "timestamp_utc"],
        kind="mergesort",
    )


def _run_raw_dir(run_dir: Path) -> Path:
    return run_dir / "raw"


def _run_raw_side_dir(run_dir: Path, token_role: str) -> Path:
    return _run_raw_dir(run_dir) / token_role


def _run_analysis_side_bars_dir(run_dir: Path, token_role: str) -> Path:
    return run_dir / "analysis" / "bars_history" / token_role


def _side_price_history_path(run_dir: Path, token_role: str) -> Path:
    return _run_raw_side_dir(run_dir, token_role) / "price_history.csv"


def _side_subgraph_trades_path(run_dir: Path, token_role: str) -> Path:
    return _run_raw_side_dir(run_dir, token_role) / "subgraph_trades.csv"


def _side_fact_trade_dir(fact_trade_root: Path, token_role: str) -> Path:
    return fact_trade_root / token_role / "fact_trade"


def _display_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _write_empty_csv(path: Path, columns: Iterable[str]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(columns=list(columns)).to_csv(path, index=False)


def _ensure_master_bar_file(path: Path) -> None:
    if path.exists():
        return
    _write_empty_csv(path, MASTER_BAR_COLUMNS)


def _summarize_csv_artifact(path: Path, *, root: Path) -> Dict[str, Any]:
    return {
        "path": _display_path(path, root=root),
        "rows": _count_csv_rows(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _load_price_history_for_role(prices_path: Path, *, token_role: str) -> pd.DataFrame:
    if not prices_path.exists():
        return pd.DataFrame(columns=["timestamp_utc", "price", "market_id"])
    try:
        prices = pd.read_csv(prices_path)
    except Exception:
        return pd.DataFrame(columns=["timestamp_utc", "price", "market_id"])
    required = {"timestamp_utc", "price", "market_id"}
    if not required.issubset(prices.columns):
        return pd.DataFrame(columns=["timestamp_utc", "price", "market_id"])
    if "token_role" in prices.columns:
        prices = prices[prices["token_role"].astype(str).str.lower() == token_role].copy()
    prices["timestamp_utc"] = pd.to_datetime(prices["timestamp_utc"], utc=True, errors="coerce")
    prices["price"] = pd.to_numeric(prices["price"], errors="coerce")
    prices["market_id"] = prices["market_id"].astype(str)
    prices = prices.dropna(subset=["timestamp_utc", "price", "market_id"])
    return prices[["timestamp_utc", "price", "market_id"]].sort_values(
        ["market_id", "timestamp_utc"],
        kind="mergesort",
    )


def _subgraph_info_path(run_dir: Path) -> Path:
    return run_dir / SUBGRAPH_INFO_FILENAME


def _legacy_subgraph_yes_trades_path(run_dir: Path) -> Path:
    return run_dir / LEGACY_SUBGRAPH_YES_TRADES_FILENAME


def _write_subgraph_cache(run_dir: Path, info: Dict[str, Any], side_trades: Dict[str, pd.DataFrame]) -> None:
    info_path = _subgraph_info_path(run_dir)
    for token_role in TOKEN_ROLES:
        side_path = _side_subgraph_trades_path(run_dir, token_role)
        trades = side_trades.get(token_role, pd.DataFrame())
        if trades.empty:
            _write_empty_csv(side_path, ["market_id", "timestamp_utc", "price", "size"])
            continue
        cached = trades.copy()
        if "timestamp_utc" in cached.columns:
            cached["timestamp_utc"] = pd.to_datetime(cached["timestamp_utc"], utc=True, errors="coerce").dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
        ensure_dir(side_path.parent)
        cached.to_csv(side_path, index=False)
    _atomic_write_json(info_path, info if isinstance(info, dict) else {})


def _load_subgraph_cache(run_dir: Path) -> tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    info = _safe_read_json(_subgraph_info_path(run_dir))
    trades_by_side: Dict[str, pd.DataFrame] = {}
    for token_role in TOKEN_ROLES:
        side_path = _side_subgraph_trades_path(run_dir, token_role)
        if not side_path.exists() and token_role == "yes":
            side_path = _legacy_subgraph_yes_trades_path(run_dir)
        if not side_path.exists():
            trades_by_side[token_role] = pd.DataFrame()
            continue
        try:
            trades = pd.read_csv(side_path)
        except Exception:
            trades_by_side[token_role] = pd.DataFrame()
            continue
        if "timestamp_utc" in trades.columns:
            trades["timestamp_utc"] = pd.to_datetime(trades["timestamp_utc"], utc=True, errors="coerce")
        trades_by_side[token_role] = trades
    return info, trades_by_side


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def normalize_list_field(x: Any) -> Optional[List[Any]]:
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None


def _split_list_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[\n,]+", value)
    return [p.strip() for p in parts if p and p.strip()]


def _extract_slug(value: str) -> Optional[str]:
    if not value:
        return None
    raw = value.strip()
    if not raw or raw.startswith("#"):
        return None
    for pattern in (
        r"/event/([^/?#]+)",
        r"/market/([^/?#]+)",
        r"/events/slug/([^/?#]+)",
        r"/markets/slug/([^/?#]+)",
    ):
        match = re.search(pattern, raw)
        if match:
            return match.group(1).strip()
    if "/" not in raw and " " not in raw:
        return raw
    return None


def _load_event_sources(urls_file: Optional[str], urls_arg: Optional[str]) -> List[str]:
    sources: List[str] = []
    if urls_arg:
        sources.extend(_split_list_arg(urls_arg))
    if urls_file:
        path = Path(urls_file)
        if not path.exists():
            raise FileNotFoundError(f"event-urls-file not found: {path}")
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            for col in ("event_url", "url", "slug", "event_slug", "market_url", "market_slug"):
                if col in df.columns:
                    sources.extend(df[col].dropna().astype(str).tolist())
                    break
            else:
                for col in df.columns:
                    sources.extend(df[col].dropna().astype(str).tolist())
        else:
            sources.extend([line.strip() for line in path.read_text().splitlines() if line.strip()])
    cleaned: List[str] = []
    for item in sources:
        if not item:
            continue
        cleaned.append(item.strip())
    seen: Dict[str, None] = {}
    for item in cleaned:
        if item in seen:
            continue
        seen[item] = None
    return list(seen.keys())


def parse_iso_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts.date()


def parse_date_arg(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}' (expected YYYY-MM-DD).") from exc


def date_to_utc_start(d: date) -> datetime:
    return datetime.combine(d, dt_time(0, 0), tzinfo=timezone.utc)


def date_to_utc_end(d: date) -> datetime:
    return datetime.combine(d, dt_time(23, 59, 59), tzinfo=timezone.utc)


def finish_week_bounds(anchor_date: date) -> Tuple[date, date, date]:
    monday = anchor_date - timedelta(days=anchor_date.weekday())
    friday = monday + timedelta(days=4)
    sunday = monday + timedelta(days=6)
    return monday, friday, sunday


# ----------------------------
# Parsing helpers
# ----------------------------

def _parse_slug_prefix(slug: str) -> Optional[str]:
    if not isinstance(slug, str) or not slug:
        return None
    m = re.match(r"^([a-z0-9]+)(?:-close)?-above", slug)
    if m:
        return m.group(1)
    return None


def _slugify_ticker(ticker: str) -> str:
    return ticker.strip().lower().replace(".", "").replace("/", "-").replace(" ", "")


def _parse_threshold(question: str) -> Optional[float]:
    if not isinstance(question, str):
        return None
    m = re.search(r"\$(\d+(?:\.\d+)?)", question)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    m = re.search(r"\babove\s+(\d+(?:\.\d+)?)\b", question, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _parse_date_from_slug(slug: str) -> Optional[date]:
    if not isinstance(slug, str) or not slug:
        return None
    m = re.search(
        r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
        r"september|sep|sept|october|oct|november|nov|december|dec)-(\d{1,2})-(\d{4})",
        slug,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    month = MONTHS.get(m.group(1).lower())
    day = int(m.group(2))
    year = int(m.group(3))
    if not month:
        return None
    try:
        return date(year, month, day)
    except Exception:
        return None


def _parse_date_from_question(question: str) -> Optional[date]:
    if not isinstance(question, str) or not question:
        return None

    m = re.search(
        r"(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
        r"september|sep|sept|october|oct|november|nov|december|dec)\s+(\d{1,2})(?:st|nd|rd|th)?(?:,)?\s+(\d{4})",
        question,
        flags=re.IGNORECASE,
    )
    if m:
        month = MONTHS.get(m.group(1).lower())
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except Exception:
                pass

    m = re.search(
        r"week\s+(?:of|ending)\s+(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|"
        r"september|sep|sept|october|oct|november|nov|december|dec)\s+(\d{1,2})(?:-\d{1,2})?(?:,)?\s+(\d{4})",
        question,
        flags=re.IGNORECASE,
    )
    if m:
        month = MONTHS.get(m.group(1).lower())
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day)
            except Exception:
                pass

    return None


def _infer_ticker(question: str, slug: str, allowlist: Optional[List[str]]) -> Tuple[Optional[str], str]:
    allowset = {t.upper() for t in allowlist} if allowlist else set()
    slug_prefix = _parse_slug_prefix(slug)

    if slug_prefix and allowlist:
        slug_map = {_slugify_ticker(t): t.upper() for t in allowlist}
        if slug_prefix in slug_map:
            return slug_map[slug_prefix], "slug"

    if allowlist and question:
        q = question.upper()
        for t in allowlist:
            if re.search(rf"\b{re.escape(t.upper())}\b", q):
                return t.upper(), "question"

    if allowlist and question:
        q_lower = question.lower()
        for company_name, ticker in COMPANY_NAME_MAP.items():
            if ticker.upper() in allowset and re.search(rf"\b{re.escape(company_name)}\b", q_lower):
                return ticker.upper(), "company_name"

    if slug_prefix:
        return slug_prefix.upper(), "slug_fallback"

    if question:
        tokens = re.findall(r"\b[A-Z]{1,6}\b", question)
        if tokens:
            return tokens[0].upper(), "question_fallback"

    return None, "none"


def _is_weekly_question(question: str, slug: str) -> bool:
    if not question and not slug:
        return False
    q = (question or "").lower()
    if "week" in q and ("finish" in q or "finishes" in q or "ending" in q or "close" in q):
        if "above" in q or "over" in q or "$" in q:
            return True
    # Fallback to slug pattern if question is sparse
    if slug and "-above-on-" in slug:
        return True
    return False


# ----------------------------
# Gamma fetch
# ----------------------------

def fetch_gamma_events(
    session: requests.Session,
    cfg: Config,
    params: Dict[str, Any],
) -> Iterable[List[Dict[str, Any]]]:
    offset = 0
    limit = int(params.get("limit", cfg.gamma_page_size))
    for page_idx in range(cfg.gamma_max_pages):
        page_params = dict(params)
        page_params.update({"limit": limit, "offset": offset})
        resp = session.get(GAMMA_EVENTS, params=page_params, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise ValueError("Gamma /events returned non-list payload.")
        yield data
        if len(data) < limit:
            break
        offset += limit
        time.sleep(cfg.sleep_between_requests_s)


def fetch_event_by_slug(
    session: requests.Session,
    cfg: Config,
    slug: str,
) -> Optional[Dict[str, Any]]:
    resp = session.get(f"{GAMMA_EVENT_SLUG}/{slug}", timeout=cfg.request_timeout_s)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Gamma /events/slug returned non-dict payload.")
    return data


def fetch_market_by_slug(
    session: requests.Session,
    cfg: Config,
    slug: str,
) -> Optional[Dict[str, Any]]:
    resp = session.get(f"{GAMMA_MARKET_SLUG}/{slug}", timeout=cfg.request_timeout_s)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("Gamma /markets/slug returned non-dict payload.")
    return data


def _event_from_market(market: Dict[str, Any]) -> Dict[str, Any]:
    event_payload: Dict[str, Any] = {}
    events = market.get("events")
    if isinstance(events, list) and events:
        if isinstance(events[0], dict):
            event_payload = dict(events[0])
    event_payload["markets"] = [market]
    return event_payload


def fetch_events_by_sources(
    session: requests.Session,
    cfg: Config,
    sources: List[str],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for raw in sources:
        slug = _extract_slug(raw)
        if not slug:
            print(f"[Weekly History] Skipping unrecognized event source: {raw}")
            continue

        event = fetch_event_by_slug(session, cfg, slug)
        if event is not None:
            events.append(event)
            continue

        market = fetch_market_by_slug(session, cfg, slug)
        if market is not None:
            events.append(_event_from_market(market))
            continue

        print(f"[Weekly History] Slug not found via Gamma: {slug}")
        time.sleep(cfg.sleep_between_requests_s)
    return events


def _market_end_date(market: Dict[str, Any], event: Dict[str, Any]) -> Optional[date]:
    for key in ("endDateIso", "endDate", "endDateTime", "end_time"):
        dt = parse_iso_date(market.get(key))
        if dt:
            return dt
    for key in ("endDateIso", "endDate", "endDateTime", "end_time"):
        dt = parse_iso_date(event.get(key))
        if dt:
            return dt
    return None


def extract_weekly_markets(
    events_pages: Iterable[List[Dict[str, Any]]],
    allowlist: List[str],
    start_date: Optional[date],
    end_date: Optional[date],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    events_seen: Dict[str, Dict[str, Any]] = {}
    allowset = {t.upper() for t in allowlist}

    for events in events_pages:
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("id") or "").strip() or None
            event_slug = str(event.get("slug") or "").strip() or None
            event_title = str(event.get("title") or event.get("question") or "").strip() or None
            event_end = _market_end_date({}, event)

            if event_id and event_id not in events_seen:
                events_seen[event_id] = {
                    "event_id": event_id,
                    "event_slug": event_slug,
                    "event_title": event_title,
                    "event_endDate": event_end.isoformat() if event_end else None,
                }

            markets = event.get("markets") or []
            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, dict):
                    continue

                question = str(market.get("question") or event.get("question") or "").strip()
                slug = str(market.get("slug") or "").strip()
                if not _is_weekly_question(question, slug):
                    continue

                ticker, ticker_source = _infer_ticker(question, slug, allowlist)
                if ticker is None or ticker.upper() not in allowset:
                    continue

                threshold = _parse_threshold(question)
                if threshold is None:
                    continue

                end_date_local = _market_end_date(market, event)
                if end_date_local is None:
                    end_date_local = _parse_date_from_question(question) or _parse_date_from_slug(slug)
                if end_date_local is None:
                    continue

                if start_date and end_date_local < start_date:
                    continue
                if end_date and end_date_local > end_date:
                    continue

                week_monday, week_friday, week_sunday = finish_week_bounds(end_date_local)

                token_ids = normalize_list_field(
                    market.get("clobTokenIds") or market.get("outcomeTokenIds")
                )
                token_ids = token_ids if isinstance(token_ids, list) else []
                yes_token = str(token_ids[0]) if len(token_ids) >= 1 and token_ids[0] else None
                no_token = str(token_ids[1]) if len(token_ids) >= 2 and token_ids[1] else None

                rows.append(
                    {
                        "event_id": event_id,
                        "event_slug": event_slug,
                        "event_title": event_title,
                        "event_endDate": end_date_local.isoformat(),
                        "market_id": str(market.get("id") or "").strip() or None,
                        "condition_id": str(market.get("conditionId") or market.get("condition_id") or "").strip() or None,
                        "market_slug": slug or None,
                        "market_question": question or None,
                        "ticker": ticker,
                        "ticker_source": ticker_source,
                        "threshold": float(threshold),
                        "week_monday": week_monday.isoformat(),
                        "week_friday": week_friday.isoformat(),
                        "week_sunday": week_sunday.isoformat(),
                        "expiry_date_utc": date_to_utc_start(end_date_local).isoformat().replace("+00:00", "Z"),
                        "resolution_time_utc": date_to_utc_end(end_date_local).isoformat().replace("+00:00", "Z"),
                        "yes_token_id": yes_token,
                        "no_token_id": no_token,
                        "enable_order_book": market.get("enableOrderBook"),
                        "active": market.get("active"),
                        "closed": market.get("closed"),
                        "schema_version": SCHEMA_VERSION_MARKETS,
                    }
                )

    markets_df = pd.DataFrame(rows)
    events_df = pd.DataFrame(events_seen.values())
    if not markets_df.empty:
        markets_df["threshold"] = pd.to_numeric(markets_df["threshold"], errors="coerce")
    return markets_df, events_df


# ----------------------------
# Price history + bars
# ----------------------------



# ----------------------------
# Subgraph ingest (optional)
# ----------------------------

def _normalize_trades(entities: List[dict]) -> pd.DataFrame:
    if not entities:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "block_number",
                "timestamp_utc",
                "market_id",
                "outcome_token_id",
                "outcome",
                "price",
                "size",
                "side",
                "tx_hash",
                "schema_version",
            ]
        )

    df = pd.DataFrame(entities)
    df = df.rename(
        columns={
            "id": "trade_id",
            "blockNumber": "block_number",
            "timestamp": "timestamp_raw",
            "marketId": "market_id",
            "outcomeTokenId": "outcome_token_id",
            "transactionHash": "tx_hash",
        }
    )

    df["block_number"] = pd.to_numeric(df.get("block_number"), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["size"] = pd.to_numeric(df.get("size"), errors="coerce")

    ts = pd.to_datetime(df["timestamp_raw"], unit="s", utc=True, errors="coerce")
    df["timestamp_utc"] = ts

    df["side"] = df.get("side")
    df["side"] = df["side"].astype(str).str.lower()

    df["schema_version"] = "pm_fact_trade_v1.0"

    keep = [
        "trade_id",
        "block_number",
        "timestamp_utc",
        "market_id",
        "outcome_token_id",
        "price",
        "size",
        "side",
        "tx_hash",
        "schema_version",
    ]
    return df[keep]


def _write_trade_partitions(df: pd.DataFrame, out_dir: Path) -> int:
    if df.empty:
        return 0

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df["trade_date"] = df["timestamp_utc"].dt.strftime("%Y-%m-%d")
    df["timestamp_utc"] = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    cols = [
        "trade_id",
        "block_number",
        "timestamp_utc",
        "market_id",
        "outcome_token_id",
        "price",
        "size",
        "side",
        "tx_hash",
        "schema_version",
    ]

    count = 0
    for trade_date, part in df.groupby("trade_date"):
        path = out_dir / f"date={trade_date}" / "trades.csv"
        part = part.reindex(columns=cols)
        if path.exists():
            try:
                existing = pd.read_csv(path)
            except Exception:
                existing = pd.DataFrame(columns=cols)
            existing = existing.reindex(columns=cols)
            merged = pd.concat([existing, part], ignore_index=True, sort=False)
            if "trade_id" in merged.columns:
                merged["trade_id"] = merged["trade_id"].astype(str)
                merged = merged.drop_duplicates(subset=["trade_id"], keep="last")
            merged = merged.sort_values(["timestamp_utc", "trade_id"], kind="mergesort")
            temp_path = path.with_suffix(path.suffix + ".tmp")
            ensure_dir(path.parent)
            merged.to_csv(temp_path, index=False)
            temp_path.replace(path)
        else:
            append_df_to_csv_with_schema(part, path)
        count += 1

    return count


def maybe_ingest_subgraph_trades(
    market_ids: List[str],
    since_ts: Optional[int],
    cfg: Config,
    fact_trade_root: Path,
    *,
    yes_token_ids_by_market: Dict[str, str],
    no_token_ids_by_market: Dict[str, str],
) -> tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    result: Dict[str, Any] = {"ok": False}
    try:
        from polymarket.subgraph_client import SubgraphClient
        from polymarket.graphql_queries import get_query
    except Exception as exc:
        result["error"] = f"subgraph import failed: {exc}"
        return result, {token_role: pd.DataFrame() for token_role in TOKEN_ROLES}

    try:
        client = SubgraphClient()
    except Exception as exc:
        result["error"] = f"subgraph not configured: {exc}"
        return result, {token_role: pd.DataFrame() for token_role in TOKEN_ROLES}

    query_name = "tradesByMarket"
    try:
        sq = get_query(query_name)
    except Exception as exc:
        result["error"] = f"subgraph query unavailable: {exc}"
        return result, {token_role: pd.DataFrame() for token_role in TOKEN_ROLES}

    variables: Dict[str, Any] = {}
    if since_ts is not None:
        variables["since"] = int(since_ts)
    if market_ids:
        variables["marketIds"] = market_ids

    try:
        pull = client.pull(sq, variable_overrides=variables or None)
    except Exception as exc:
        result["error"] = f"subgraph pull failed: {exc}"
        return result, {token_role: pd.DataFrame() for token_role in TOKEN_ROLES}

    if pull.total_entities > cfg.max_subgraph_entities:
        result["error"] = (
            f"subgraph pull too large ({pull.total_entities} entities > {cfg.max_subgraph_entities}); skipped filtering"
        )
        return result, {token_role: pd.DataFrame() for token_role in TOKEN_ROLES}

    try:
        entities = client.entities_from_run(pull.run_dir)
    except Exception as exc:
        result["error"] = f"subgraph load failed: {exc}"
        return result, {token_role: pd.DataFrame() for token_role in TOKEN_ROLES}

    if market_ids:
        market_set = set(market_ids)
        entities = [e for e in entities if str(e.get("marketId")) in market_set]

    df = _normalize_trades(entities)
    df["market_id"] = df["market_id"].astype(str)
    df["outcome_token_id"] = df["outcome_token_id"].astype(str)
    side_trades: Dict[str, pd.DataFrame] = {}
    side_partitions: Dict[str, int] = {}
    side_entities: Dict[str, int] = {}
    token_maps = {"yes": yes_token_ids_by_market, "no": no_token_ids_by_market}
    for token_role in TOKEN_ROLES:
        token_map = token_maps[token_role]
        trades = df[
            df["market_id"].map(lambda market_id: token_map.get(str(market_id)))
            == df["outcome_token_id"]
        ].copy()
        side_trades[token_role] = trades
        side_entities[token_role] = len(trades)
        ensure_dir(_side_fact_trade_dir(fact_trade_root, token_role))
        side_partitions[token_role] = _write_trade_partitions(
            trades,
            _side_fact_trade_dir(fact_trade_root, token_role),
        )

    result.update(
        {
            "ok": True,
            "run_id": pull.run_id,
            "run_dir": str(pull.run_dir),
            "total_entities": len(df),
            "yes_entities": side_entities.get("yes", 0),
            "no_entities": side_entities.get("no", 0),
            "partitions": side_partitions,
        }
    )
    return result, side_trades


# ----------------------------
# Dim market
# ----------------------------

def build_dim_market(markets: pd.DataFrame) -> pd.DataFrame:
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


def write_dim_market(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=False)
            return csv_path
    df.to_csv(path, index=False)
    return path


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill weekly Polymarket events and price history.")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-id", type=str, default=None, help="Optional custom run directory name.")
    parser.add_argument("--bars-dir", type=str, default=None)
    parser.add_argument("--dim-market-out", type=str, default=str(DEFAULT_DIM_MARKET_PATH))
    parser.add_argument(
        "--fact-trade-dir",
        type=str,
        default=None,
        help="Optional root directory for side-split subgraph trade artifacts (defaults to run_dir/raw).",
    )
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--tickers-csv", type=str, default=None, help="CSV with a 'ticker' column")
    parser.add_argument(
        "--event-urls",
        type=str,
        default=None,
        help="Comma- or newline-separated Polymarket event/market URLs or slugs.",
    )
    parser.add_argument(
        "--event-urls-file",
        type=str,
        default=None,
        help="Text/CSV file with event or market URLs/slugs (one per line or a 'url'/'slug' column).",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD (UTC)")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD (UTC)")
    parser.add_argument("--fidelity-min", type=int, default=Config().clob_fidelity_min, help="CLOB history fidelity (minutes)")
    parser.add_argument("--despike", action="store_true", help="Remove single-point price spikes that immediately revert")
    parser.add_argument("--despike-jump", type=float, default=Config().despike_jump, help="Min jump size to treat as spike")
    parser.add_argument("--despike-revert", type=float, default=Config().despike_revert, help="Max revert distance to treat as spike")
    parser.add_argument("--bars-freqs", type=str, default="1d,1h", help="Comma-separated bar freqs (e.g. 1d,1h)")
    parser.add_argument("--include-subgraph", action="store_true", help="Attempt subgraph trade ingest if configured")
    parser.add_argument("--max-subgraph-entities", type=int, default=Config().max_subgraph_entities)
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run in the same run directory.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_tickers_arg(tickers_csv: Optional[str], tickers_list: Optional[str]) -> List[str]:
    if tickers_list:
        values = [t.strip().upper() for t in tickers_list.split(",") if t.strip()]
        if not values:
            raise ValueError("Provided --tickers is empty after parsing.")
        return values
    if tickers_csv:
        path = Path(tickers_csv)
        if not path.exists():
            raise FileNotFoundError(f"tickers-csv not found: {path}")
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            raise ValueError(f"tickers-csv missing 'ticker' column. Found: {list(df.columns)}")
        tickers = df["ticker"].dropna().astype(str).str.strip().str.upper().tolist()
        tickers = [t for t in tickers if t]
        if not tickers:
            raise ValueError("No tickers found in --tickers-csv.")
        return tickers
    return DEFAULT_TICKERS_WEEKLY.copy()


def main() -> None:
    args = parse_args()

    tickers = _load_tickers_arg(args.tickers_csv, args.tickers)
    event_sources = _load_event_sources(args.event_urls_file, args.event_urls)
    slug_requested = bool(args.event_urls_file or args.event_urls)
    start_date = parse_date_arg(args.start_date)
    end_date = parse_date_arg(args.end_date)
    if start_date and end_date and start_date > end_date:
        raise ValueError("start-date must be <= end-date.")

    cfg = Config(
        clob_fidelity_min=int(args.fidelity_min),
        bars_freqs=tuple([f.strip() for f in args.bars_freqs.split(",") if f.strip()]),
        include_subgraph=bool(args.include_subgraph),
        max_subgraph_entities=int(args.max_subgraph_entities),
        despike_enabled=bool(args.despike),
        despike_jump=float(args.despike_jump),
        despike_revert=float(args.despike_revert),
    )

    out_dir = Path(args.out_dir)
    dim_market_out = Path(args.dim_market_out)
    requested_run_id = (args.run_id or "").strip()
    if requested_run_id:
        if requested_run_id in {".", ".."} or "/" in requested_run_id or "\\" in requested_run_id:
            raise ValueError("run-id must be a single directory name (no path separators).")
        run_id = requested_run_id
    else:
        run_id = datetime.now(timezone.utc).strftime("weekly-history-%Y%m%dT%H%M%SZ")
    run_dir = out_dir / "runs" / run_id
    bars_dir = Path(args.bars_dir) if args.bars_dir else run_dir / "bars_history"
    raw_dir = _run_raw_dir(run_dir)
    analysis_dir = run_dir / "analysis"
    fact_trade_root = Path(args.fact_trade_dir) if args.fact_trade_dir else raw_dir
    markets_path = run_dir / "weekly_markets.csv"
    events_path = run_dir / "weekly_events.csv"
    prices_path = run_dir / "price_history.csv"
    resume_mode = bool(args.resume and not args.dry_run)

    if not args.dry_run:
        if run_dir.exists() and not resume_mode:
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        ensure_dir(run_dir)
        ensure_dir(bars_dir)
        ensure_dir(raw_dir)
        ensure_dir(analysis_dir)
        for token_role in TOKEN_ROLES:
            ensure_dir(_run_raw_side_dir(run_dir, token_role))
            ensure_dir(_run_analysis_side_bars_dir(run_dir, token_role))
            side_prices_path = _side_price_history_path(run_dir, token_role)
            if not side_prices_path.exists():
                _write_empty_csv(
                    side_prices_path,
                    [
                        "timestamp_utc",
                        "price",
                        "token_id",
                        "schema_version",
                        "market_id",
                        "ticker",
                        "threshold",
                        "token_role",
                        "fidelity_min",
                    ],
                )

    session = make_session()

    params: Dict[str, Any] = {
        "closed": "true",
        "limit": cfg.gamma_page_size,
        "order": "endDate",
        "ascending": "true",
    }
    if start_date:
        params["end_date_min"] = date_to_utc_start(start_date).isoformat().replace("+00:00", "Z")
    if end_date:
        params["end_date_max"] = date_to_utc_end(end_date).isoformat().replace("+00:00", "Z")

    print(f"[Weekly History] tickers={','.join(tickers)}", flush=True)
    print(f"[Weekly History] start_date={start_date} end_date={end_date}", flush=True)
    print(f"[Weekly History] run_id={run_id}", flush=True)
    print(f"[Weekly History] script_version={SCRIPT_VERSION}", flush=True)
    if resume_mode:
        print(f"[Weekly History] resume=true run_dir={run_dir}", flush=True)

    if slug_requested and not event_sources:
        print("[Weekly History] No event sources parsed from provided URLs/slugs.")
        return

    if resume_mode and markets_path.exists():
        markets_df = pd.read_csv(markets_path)
        events_df = pd.read_csv(events_path) if events_path.exists() else pd.DataFrame()
        print("[Weekly History] resume=using existing weekly_markets.csv", flush=True)
    else:
        if event_sources:
            print(f"[Weekly History] event_sources={len(event_sources)} (slug-based)")
            events = fetch_events_by_sources(session, cfg, event_sources)
            if not events:
                print("[Weekly History] No events resolved from event sources.")
                return
            pages = [events]
        else:
            print("[Weekly History] event_sources=discovery")
            pages = fetch_gamma_events(session, cfg, params)
        markets_df, events_df = extract_weekly_markets(pages, tickers, start_date, end_date)

    if markets_df.empty:
        print("[Weekly History] No weekly markets found.")
        return

    if "market_id" in markets_df.columns:
        markets_df = markets_df[markets_df["market_id"].notna()].copy()
    if "yes_token_id" in markets_df.columns:
        markets_df = markets_df[markets_df["yes_token_id"].notna()].copy()

    if markets_df.empty:
        print("[Weekly History] No weekly markets with CLOB tokens found.")
        return

    if "market_id" in markets_df.columns:
        markets_df = markets_df.drop_duplicates(subset=["market_id"], keep="first")

    markets_df = markets_df.sort_values(["ticker", "event_endDate", "threshold"], kind="mergesort")
    markets_df = markets_df.reset_index(drop=True)

    if not args.dry_run:
        markets_df.to_csv(markets_path, index=False)
        events_df.to_csv(events_path, index=False)
        dim_market = build_dim_market(markets_df)
        dim_path = write_dim_market(dim_market, dim_market_out)
        print(f"[Weekly History] dim_market={dim_path}", flush=True)

    history_state: Dict[str, Any] = {}
    completed_market_ids: set[str] = set()
    failed_market_ids: set[str] = set()
    if not args.dry_run:
        history_state = (
            _load_history_state(run_dir)
            if resume_mode
            else _new_history_state(
                run_id=run_id,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                include_subgraph=cfg.include_subgraph,
                bars_freqs=cfg.bars_freqs,
            )
        )
        if not history_state:
            history_state = _new_history_state(
                run_id=run_id,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                include_subgraph=cfg.include_subgraph,
                bars_freqs=cfg.bars_freqs,
            )
        history_state["run_id"] = run_id
        history_state["status"] = "running"
        history_state["phase"] = "history"
        history_state["tickers"] = list(tickers)
        history_state["start_date"] = start_date.isoformat() if start_date else None
        history_state["end_date"] = end_date.isoformat() if end_date else None
        history_state["include_subgraph"] = bool(cfg.include_subgraph)
        history_state["bars_freqs"] = list(cfg.bars_freqs)
        history_state["markets_total"] = int(len(markets_df))
        completed_market_ids = _completed_market_ids_from_state(history_state)
        failed_market_ids = _failed_market_ids_from_state(history_state)
        _write_history_state(run_dir, history_state)
        if resume_mode and completed_market_ids:
            print(
                "[Weekly History] resume=skipping "
                f"{len(completed_market_ids)}/{len(markets_df)} completed markets",
                flush=True,
            )

    market_metadata = markets_df[
        [
            "market_id",
            "event_id",
            "event_slug",
            "market_slug",
            "ticker",
            "threshold",
            "week_friday",
            "expiry_date_utc",
            "yes_token_id",
        ]
    ].copy()
    market_metadata["market_id"] = market_metadata["market_id"].astype(str)

    # Fetch price history
    despike_adjusted = 0
    master_bar_updates: Dict[str, Dict[str, Any]] = {}

    start_dt = date_to_utc_start(start_date) if start_date else None
    end_dt = date_to_utc_end(end_date) if end_date else None

    markets_total = len(markets_df)
    def _safe_job_id(value: Any) -> str:
        text = "NA" if value is None else str(value)
        return "".join(
            ch if ch.isalnum() or ch in ("_", "-", ".", ":") else "_"
            for ch in text
        )

    for idx, row in markets_df.iterrows():
        market_id = row.get("market_id")
        yes_token = row.get("yes_token_id")
        no_token = row.get("no_token_id")
        ticker = row.get("ticker")
        threshold = row.get("threshold")
        market_id_text = str(market_id)

        job_id = _safe_job_id(f"{ticker}:{threshold}:{market_id}")
        if market_id_text in completed_market_ids:
            continue

        print(
            f"[Weekly History] Market start {idx + 1}/{markets_total} "
            f"job_id={job_id} ticker={ticker} threshold={threshold} market_id={market_id}",
            flush=True,
        )
        print(
            f"[Weekly History] Processing market {idx + 1}/{markets_total}: "
            f"{ticker} @ ${threshold} (market_id={market_id})",
            flush=True,
        )
        market_failed = False
        if not args.dry_run:
            purge_targets = [prices_path] + [
                _side_price_history_path(run_dir, token_role) for token_role in TOKEN_ROLES
            ]
            removed_rows = 0
            for target in purge_targets:
                removed_rows += _purge_market_rows(target, market_id_text)
            if removed_rows > 0:
                print(
                    f"[Weekly History] resume=purged {removed_rows} partial rows for market_id={market_id_text}",
                    flush=True,
                )

        for token_role, token_id in [("yes", yes_token), ("no", no_token)]:
            if not token_id:
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
                if history.empty:
                    continue

                history = history.copy()
                history["market_id"] = market_id
                history["ticker"] = ticker
                history["threshold"] = threshold
                history["token_role"] = token_role
                history["fidelity_min"] = cfg.clob_fidelity_min
                history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True, errors="coerce")
                history, adjusted = clean_price_history(
                    history,
                    cfg.despike_enabled,
                    cfg.despike_jump,
                    cfg.despike_revert,
                )
                despike_adjusted += adjusted

                if not args.dry_run:
                    history_out = history.copy()
                    history_out["timestamp_utc"] = history_out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                    append_df_to_csv_with_schema(history_out, prices_path)
                    append_df_to_csv_with_schema(history_out, _side_price_history_path(run_dir, token_role))

            except Exception as exc:
                market_failed = True
                print(f"[Weekly History] Failed processing {ticker} @ ${threshold} token={token_id} ({token_role}): {exc}")
                import traceback
                print(f"[Weekly History] Traceback: {traceback.format_exc()}")
                continue

            time.sleep(cfg.sleep_between_requests_s)

        print(
            f"[Weekly History] Market complete {idx + 1}/{markets_total} "
            f"job_id={job_id} status={'failed' if market_failed else 'ok'}",
            flush=True,
        )
        if not args.dry_run:
            completed_market_ids.add(market_id_text)
            if market_failed:
                failed_market_ids.add(market_id_text)
            history_state["completed_market_ids"] = sorted(completed_market_ids)
            history_state["failed_market_ids"] = sorted(failed_market_ids)
            _write_history_state(run_dir, history_state)

    subgraph_info: Dict[str, Any] = {}
    subgraph_trades_by_side: Dict[str, pd.DataFrame] = {
        token_role: pd.DataFrame() for token_role in TOKEN_ROLES
    }
    if cfg.include_subgraph and not args.dry_run:
        history_state["phase"] = "subgraph"
        _write_history_state(run_dir, history_state)
        market_ids = [m for m in markets_df["market_id"].dropna().astype(str).unique().tolist() if m]
        since_ts = int(start_dt.timestamp()) if start_dt else None
        yes_token_ids_by_market = {
            str(row["market_id"]): str(row["yes_token_id"])
            for _, row in markets_df[["market_id", "yes_token_id"]].dropna().iterrows()
        }
        no_token_ids_by_market = {
            str(row["market_id"]): str(row["no_token_id"])
            for _, row in markets_df[["market_id", "no_token_id"]].dropna().iterrows()
        }
        if resume_mode and history_state.get("subgraph_completed") and (
            _side_subgraph_trades_path(run_dir, "yes").exists()
            or _legacy_subgraph_yes_trades_path(run_dir).exists()
        ):
            subgraph_info, subgraph_trades_by_side = _load_subgraph_cache(run_dir)
            print("[Weekly History] resume=using cached subgraph trades", flush=True)
        else:
            subgraph_info, subgraph_trades_by_side = maybe_ingest_subgraph_trades(
                market_ids,
                since_ts,
                cfg,
                fact_trade_root,
                yes_token_ids_by_market=yes_token_ids_by_market,
                no_token_ids_by_market=no_token_ids_by_market,
            )
            _write_subgraph_cache(run_dir, subgraph_info, subgraph_trades_by_side)
            history_state["subgraph_completed"] = True
            _write_history_state(run_dir, history_state)
        if subgraph_info.get("ok"):
            print(
                "[Weekly History] subgraph trades "
                f"run_id={subgraph_info.get('run_id')} "
                f"yes_entities={subgraph_info.get('yes_entities')} "
                f"no_entities={subgraph_info.get('no_entities')}",
                flush=True,
            )
        else:
            print(f"[Weekly History] subgraph skipped: {subgraph_info.get('error')}", flush=True)

    if not args.dry_run:
        history_state["phase"] = "bars"
        _write_history_state(run_dir, history_state)
        analysis_side_updates: Dict[str, Dict[str, Dict[str, Any]]] = {
            token_role: {} for token_role in TOKEN_ROLES
        }
        for freq in cfg.bars_freqs:
            for token_role in TOKEN_ROLES:
                side_prices_path = _side_price_history_path(run_dir, token_role)
                side_clob_source = _load_price_history_for_role(side_prices_path, token_role=token_role)
                if side_clob_source.empty and prices_path.exists():
                    side_clob_source = _load_price_history_for_role(prices_path, token_role=token_role)

                fallback_bars = pd.DataFrame(columns=["timestamp_utc", "market_id"])
                if not side_clob_source.empty:
                    fallback_bars = build_bars_from_prices(
                        side_clob_source,
                        freq,
                        schema_version=SCHEMA_VERSION_BARS,
                    )
                    fallback_bars = build_master_bar_rows(
                        fallback_bars,
                        market_metadata,
                        bar_source="clob_fallback",
                        written_by_run_id=run_id,
                        schema_version=SCHEMA_VERSION_BARS,
                    )

                merged_bars = fallback_bars
                side_trades = subgraph_trades_by_side.get(token_role, pd.DataFrame())
                if not side_trades.empty:
                    trade_bars = build_bars_from_trades(
                        side_trades,
                        freq,
                        schema_version=SCHEMA_VERSION_BARS,
                    )
                    trade_bars = build_master_bar_rows(
                        trade_bars,
                        market_metadata,
                        bar_source="subgraph",
                        written_by_run_id=run_id,
                        schema_version=SCHEMA_VERSION_BARS,
                    )
                    merged_bars = pd.concat([fallback_bars, trade_bars], ignore_index=True)

                analysis_bars_dir = _run_analysis_side_bars_dir(run_dir, token_role)
                analysis_path = master_bar_path(analysis_bars_dir, freq)
                if merged_bars.empty:
                    _ensure_master_bar_file(analysis_path)
                    analysis_side_updates[token_role][freq] = {
                        "name": analysis_path.name,
                        "path": _display_path(analysis_path, root=run_dir),
                        "frequency": freq,
                        "rows_written": 0,
                        "rows_total": 0,
                    }
                else:
                    analysis_meta = upsert_master_bars(merged_bars, analysis_bars_dir, freq)
                    analysis_meta["path"] = _display_path(Path(str(analysis_meta["path"])), root=run_dir)
                    analysis_side_updates[token_role][freq] = analysis_meta
                if token_role == "yes":
                    master_bar_updates[freq] = upsert_master_bars(merged_bars, bars_dir, freq)
        history_state["bars_completed"] = True
        _write_history_state(run_dir, history_state)

    if args.dry_run:
        print("[Weekly History] dry-run complete (no files written).")
        return

    price_rows = _count_csv_rows(prices_path)
    raw_side_outputs = {
        token_role: {
            "price_history": _summarize_csv_artifact(
                _side_price_history_path(run_dir, token_role),
                root=run_dir,
            )
        }
        for token_role in TOKEN_ROLES
    }
    analysis_side_outputs = {
        token_role: {
            "bars_dir": _display_path(_run_analysis_side_bars_dir(run_dir, token_role), root=run_dir),
            "bars": {
                freq: {
                    **analysis_side_updates.get(token_role, {}).get(freq, {}),
                    "bar_source": (
                        "subgraph"
                        if not subgraph_trades_by_side.get(token_role, pd.DataFrame()).empty
                        else "clob_fallback"
                    ),
                }
                for freq in cfg.bars_freqs
            },
        }
        for token_role in TOKEN_ROLES
    }
    subgraph_side_outputs = {"enabled": bool(cfg.include_subgraph)}
    if cfg.include_subgraph:
        for token_role in TOKEN_ROLES:
            subgraph_side_outputs[token_role] = {
                "trades": _summarize_csv_artifact(
                    _side_subgraph_trades_path(run_dir, token_role),
                    root=run_dir,
                ),
                "fact_trade_dir": _display_path(_side_fact_trade_dir(fact_trade_root, token_role), root=run_dir),
                "partitions": int((subgraph_info.get("partitions") or {}).get(token_role, 0)),
                "bar_source": (
                    "subgraph"
                    if not subgraph_trades_by_side.get(token_role, pd.DataFrame()).empty
                    else "clob_fallback"
                ),
            }
    manifest = {
        "run_id": run_id,
        "script_version": SCRIPT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "markets": len(markets_df),
        "price_rows": price_rows,
        "despike": {
            "enabled": cfg.despike_enabled,
            "jump": cfg.despike_jump,
            "revert": cfg.despike_revert,
            "adjusted_points": despike_adjusted,
        },
        "bars_dir": str(bars_dir),
        "bar_storage_version": "master_csv_v1",
        "fact_trade_dir": str(fact_trade_root),
        "raw_side_outputs": raw_side_outputs,
        "analysis_side_outputs": analysis_side_outputs,
        "subgraph_side_outputs": subgraph_side_outputs,
        "master_bars": master_bar_updates,
        "dim_market": str(dim_market_out),
        "subgraph": subgraph_info,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    history_state["status"] = "completed"
    history_state["phase"] = "complete"
    history_state["manifest_written"] = True
    _write_history_state(run_dir, history_state)

    print("[Weekly History] complete", flush=True)
    print(f"[Weekly History] run_dir={run_dir}", flush=True)
    print(f"[Weekly History] price_rows={price_rows}", flush=True)
    print(f"[Weekly History] master_bars={json.dumps(master_bar_updates)}", flush=True)
    print(f"run_id={run_id}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)
