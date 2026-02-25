#!/usr/bin/env python3
"""
02-polymarket-weekly-history-v1.0.py

Backfill weekly Polymarket events ("TICKER finishes week of DATE above $Z")
from the Gamma API, fetch CLOB price history, and build hourly/daily bars.
Optionally ingest subgraph trades if configured.
"""

from __future__ import annotations

import argparse
import json
import os
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

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# Endpoints
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
GAMMA_EVENT_SLUG = "https://gamma-api.polymarket.com/events/slug"
GAMMA_MARKET_SLUG = "https://gamma-api.polymarket.com/markets/slug"
CLOB_PRICE_HISTORY = "https://clob.polymarket.com/prices-history"

SCRIPT_VERSION = "1.0.0"
SCHEMA_VERSION_MARKETS = "pm_weekly_markets_v1.0"
SCHEMA_VERSION_PRICES = "pm_weekly_prices_v1.0"
SCHEMA_VERSION_BARS = "pm_bars_history_v1.0"

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
DEFAULT_BARS_DIR = REPO_ROOT / "src" / "data" / "analysis" / "polymarket" / "bars_history"
DEFAULT_DIM_MARKET_PATH = REPO_ROOT / "src" / "data" / "models" / "polymarket" / "dim_market_weekly.csv"
DEFAULT_FACT_TRADE_DIR = REPO_ROOT / "src" / "data" / "raw" / "polymarket" / "weekly_history" / "fact_trade"

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

FREQ_ALIASES = {
    "60m": "1h",
    "1h": "1h",
    "1d": "1D",
    "1D": "1D",
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


# ----------------------------
# Utilities
# ----------------------------

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

def fetch_price_history(
    session: requests.Session,
    token_id: str,
    cfg: Config,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
) -> pd.DataFrame:
    def _payload_to_df(payload: dict) -> pd.DataFrame:
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

    base_params: Dict[str, Any] = {
        "market": token_id,
        "fidelity": cfg.clob_fidelity_min,
    }

    if not (start_dt or end_dt):
        params = dict(base_params)
        params.update({"interval": "max"})
        resp = session.get(CLOB_PRICE_HISTORY, params=params, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        return _payload_to_df(resp.json())

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
        frame = _payload_to_df(resp.json())
        if not frame.empty:
            frames.append(frame)
        if cfg.sleep_between_requests_s > 0:
            time.sleep(cfg.sleep_between_requests_s)
        cur_start = cur_end + 1

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("timestamp_utc").reset_index(drop=True)
    if start_dt:
        out = out[out["timestamp_utc"] >= start_dt]
    if end_dt:
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


def _build_bars_from_prices(df: pd.DataFrame, freq: str) -> pd.DataFrame:
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
    ohlc["schema_version"] = SCHEMA_VERSION_BARS
    return ohlc


def _write_bars(bars: pd.DataFrame, bars_dir: Path, freq: str) -> int:
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
        append_df_to_csv_with_schema(part, path)
        count += 1

    return count


def maybe_ingest_subgraph_trades(
    market_ids: List[str],
    since_ts: Optional[int],
    cfg: Config,
    out_dir: Path,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": False}
    try:
        from polymarket.subgraph_client import SubgraphClient
        from polymarket.graphql_queries import get_query
    except Exception as exc:
        result["error"] = f"subgraph import failed: {exc}"
        return result

    try:
        client = SubgraphClient()
    except Exception as exc:
        result["error"] = f"subgraph not configured: {exc}"
        return result

    query_name = "tradesByMarket"
    try:
        sq = get_query(query_name)
    except Exception as exc:
        result["error"] = f"subgraph query unavailable: {exc}"
        return result

    variables: Dict[str, Any] = {}
    if since_ts is not None:
        variables["since"] = int(since_ts)
    if market_ids:
        variables["marketIds"] = market_ids

    try:
        pull = client.pull(sq, variable_overrides=variables or None)
    except Exception as exc:
        result["error"] = f"subgraph pull failed: {exc}"
        return result

    if pull.total_entities > cfg.max_subgraph_entities:
        result["error"] = (
            f"subgraph pull too large ({pull.total_entities} entities > {cfg.max_subgraph_entities}); skipped filtering"
        )
        return result

    try:
        entities = client.entities_from_run(pull.run_dir)
    except Exception as exc:
        result["error"] = f"subgraph load failed: {exc}"
        return result

    if market_ids:
        market_set = set(market_ids)
        entities = [e for e in entities if str(e.get("marketId")) in market_set]

    df = _normalize_trades(entities)
    ensure_dir(out_dir)
    partitions = _write_trade_partitions(df, out_dir)

    result.update(
        {
            "ok": True,
            "run_id": pull.run_id,
            "run_dir": str(pull.run_dir),
            "total_entities": len(df),
            "partitions": partitions,
        }
    )
    return result


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
    parser.add_argument("--bars-dir", type=str, default=str(DEFAULT_BARS_DIR))
    parser.add_argument("--dim-market-out", type=str, default=str(DEFAULT_DIM_MARKET_PATH))
    parser.add_argument("--fact-trade-dir", type=str, default=str(DEFAULT_FACT_TRADE_DIR))
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
    parser.add_argument("--bars-freqs", type=str, default="1h,1d", help="Comma-separated bar freqs (e.g. 1h,1d)")
    parser.add_argument("--include-subgraph", action="store_true", help="Attempt subgraph trade ingest if configured")
    parser.add_argument("--max-subgraph-entities", type=int, default=Config().max_subgraph_entities)
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
    bars_dir = Path(args.bars_dir)
    dim_market_out = Path(args.dim_market_out)
    fact_trade_dir = Path(args.fact_trade_dir)
    run_id = datetime.now(timezone.utc).strftime("weekly-history-%Y%m%dT%H%M%SZ")
    run_dir = out_dir / "runs" / run_id

    if not args.dry_run:
        ensure_dir(run_dir)
        ensure_dir(bars_dir)

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

    if slug_requested and not event_sources:
        print("[Weekly History] No event sources parsed from provided URLs/slugs.")
        return

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
        markets_path = run_dir / "weekly_markets.csv"
        events_path = run_dir / "weekly_events.csv"
        markets_df.to_csv(markets_path, index=False)
        events_df.to_csv(events_path, index=False)
        dim_market = build_dim_market(markets_df)
        dim_path = write_dim_market(dim_market, dim_market_out)
        print(f"[Weekly History] dim_market={dim_path}", flush=True)

    # Fetch price history
    prices_path = run_dir / "price_history.csv"
    price_rows = 0
    despike_adjusted = 0
    bar_partitions = 0

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

        job_id = _safe_job_id(f"{ticker}:{threshold}:{market_id}")
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

        for token_role, token_id in [("yes", yes_token), ("no", no_token)]:
            if not token_id:
                continue
            try:
                history = fetch_price_history(session, str(token_id), cfg, start_dt, end_dt)
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
                    price_rows += len(history_out)

                    if token_role == "yes":
                        bars_in = history[["timestamp_utc", "price"]].copy()
                        bars_in["market_id"] = market_id
                        for freq in cfg.bars_freqs:
                            bars = _build_bars_from_prices(bars_in, freq)
                            bar_partitions += _write_bars(bars, bars_dir, freq)

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

    subgraph_info: Dict[str, Any] = {}
    if cfg.include_subgraph and not args.dry_run:
        market_ids = [m for m in markets_df["market_id"].dropna().astype(str).unique().tolist() if m]
        since_ts = int(start_dt.timestamp()) if start_dt else None
        subgraph_info = maybe_ingest_subgraph_trades(market_ids, since_ts, cfg, fact_trade_dir)
        if subgraph_info.get("ok"):
            print(f"[Weekly History] subgraph trades run_id={subgraph_info.get('run_id')}")
        else:
            print(f"[Weekly History] subgraph skipped: {subgraph_info.get('error')}")

    if args.dry_run:
        print("[Weekly History] dry-run complete (no files written).")
        return

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
        "fact_trade_dir": str(fact_trade_dir),
        "bar_partitions": bar_partitions,
        "dim_market": str(dim_market_out),
        "subgraph": subgraph_info,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("[Weekly History] complete", flush=True)
    print(f"[Weekly History] run_dir={run_dir}", flush=True)
    print(f"[Weekly History] price_rows={price_rows}", flush=True)
    print(f"[Weekly History] bar_partitions={bar_partitions}", flush=True)
    print(f"run_id={run_id}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)
