#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, time as dt_time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from scipy.stats import norm
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration.calibrate_common import EPS, _logit


# -----------------------------
# Defaults + Endpoints
# -----------------------------

DEFAULT_TICKERS_WEEKLY = ["AAPL", "GOOGL", "MSFT", "META", "AMZN", "PLTR", "NVDA", "TSLA", "NFLX", "OPEN"]
DEFAULT_TICKERS_1DTE = ["TSLA", "GOOGL", "NVDA", "MSFT", "AAPL", "AMZN"]
GAMMA_PUBLIC_SEARCH = "https://gamma-api.polymarket.com/public-search"
GAMMA_EVENT_BY_SLUG = "https://gamma-api.polymarket.com/events/slug/{}"
CLOB_PRICES = "https://clob.polymarket.com/prices"  # POST [{"token_id": "...", "side": "BUY"}]

SCRIPT_VER = "1.5.1"
SCHEMA_VERSION = "pPM_polymarket_snapshot_v1.0.1"
DEFAULT_EXCHANGE_CALENDAR = "XNYS"
DEFAULT_OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "raw", "polymarket", "snapshots")
)


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class Config:
    tz_name: str = "Europe/Paris"
    risk_free_rate: float = 0.03
    sleep_between_slugs_s: float = 0.10
    request_timeout_s: int = 30

    # yfinance pRN controls
    call_smooth_window: int = 5
    slope_bracket_width: int = 2
    max_deltaK_frac_spot: float = 0.30
    min_calls_for_bidask: int = 10
    rel_spread_cap: float = 2.0

    apply_monotone_pav: bool = True

    # rv20 fallback controls
    rv20_fallback: float = 0.30  # Default fallback when all rv20 methods fail
    rv20_min_window: int = 2  # Minimum trading days for short-window fallback
    rv20_history_days: int = 400  # Days of history to fetch for rv20
    rv20_max_retries: int = 3  # Max retries for yfinance fetch


# -----------------------------
# Small utilities
# -----------------------------

def _local_date(dt_like, tz_name: str) -> Optional[date]:
    dt = pd.to_datetime(dt_like, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.tz_convert(ZoneInfo(tz_name)).date()


def load_exchange_calendar(calendar_name: str) -> Tuple[Optional[str], Any]:
    try:
        import exchange_calendars as xcals
        return "exchange_calendars", xcals.get_calendar(calendar_name)
    except Exception:
        pass
    try:
        import pandas_market_calendars as pmc
        return "pandas_market_calendars", pmc.get_calendar(calendar_name)
    except Exception:
        return None, None


def _calendar_sessions(
    calendar_kind: Optional[str],
    calendar: Any,
    start_date: date,
    end_date: date,
) -> List[date]:
    if calendar is None or calendar_kind is None:
        return []
    try:
        if calendar_kind == "exchange_calendars":
            sessions = calendar.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
            return [ts.date() for ts in sessions]
        if calendar_kind == "pandas_market_calendars":
            sessions = calendar.valid_days(start_date=start_date, end_date=end_date)
            return [ts.date() for ts in sessions]
    except Exception:
        return []
    return []


def _is_trading_day(
    day: date,
    calendar_kind: Optional[str],
    calendar: Any,
) -> bool:
    if calendar is None or calendar_kind is None:
        return day.weekday() < 5
    try:
        if calendar_kind == "exchange_calendars":
            return bool(calendar.is_session(pd.Timestamp(day)))
        if calendar_kind == "pandas_market_calendars":
            sessions = calendar.valid_days(start_date=day, end_date=day)
            return len(sessions) > 0
    except Exception:
        return day.weekday() < 5
    return day.weekday() < 5


def next_trading_day(
    today_local: date,
    calendar_kind: Optional[str],
    calendar: Any,
) -> Optional[date]:
    sessions = _calendar_sessions(calendar_kind, calendar, today_local + timedelta(days=1), today_local + timedelta(days=10))
    if sessions:
        return sessions[0]
    # fallback: next weekday
    d = today_local
    for _ in range(10):
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            return d
    return None


def resolve_target_date(
    today_local: date,
    contract_type: str,
    contract_1dte: str,
    calendar_kind: Optional[str],
    calendar: Any,
) -> date:
    contract_type = (contract_type or "weekly").strip().lower()
    if contract_type == "1dte":
        mode = (contract_1dte or "close_tomorrow").strip().lower()
        if mode not in {"close_today", "close_tomorrow"}:
            raise ValueError("contract_1dte must be close_today or close_tomorrow.")
        delta = 0 if mode == "close_today" else 1
        return today_local + timedelta(days=delta)

    # weekly (default): keep calendar Friday for slug compatibility
    friday = finish_week_friday(today_local)
    if calendar is not None and not _is_trading_day(friday, calendar_kind, calendar):
        print(
            f"[Calendar] {friday.isoformat()} is not a trading day for {calendar_kind}; "
            f"using calendar Friday for weekly contracts (override with --target-date if needed)."
        )
    return friday


def build_1dte_slug(ticker: str, target_date: date) -> str:
    # Daily (1DTE) contracts include "close" in the slug.
    prefix = ticker_to_slug_prefix(ticker)
    month = target_date.strftime("%B").lower()
    return f"{prefix}-close-above-on-{month}-{target_date.day}-{target_date.year}"


def build_market_slug(ticker: str, target_date: date, contract_type: str) -> str:
    if (contract_type or "weekly").strip().lower() == "1dte":
        return build_1dte_slug(ticker, target_date)
    return build_weekly_slug(ticker, target_date)


def classify_session(
    asof_utc: datetime,
    tz_name: str,
    calendar_kind: Optional[str],
    calendar: Any,
) -> str:
    tz = ZoneInfo(tz_name)
    asof_ts = pd.to_datetime(asof_utc, utc=True, errors="coerce")
    if pd.isna(asof_ts):
        return "PRE"

    local_dt = asof_ts.tz_convert(tz)
    local_date = local_dt.date()

    open_utc = None
    close_utc = None

    if calendar is not None and calendar_kind is not None:
        try:
            if calendar_kind == "exchange_calendars":
                session = pd.Timestamp(local_date)
                if calendar.is_session(session):
                    open_ts, close_ts = calendar.open_and_close_for_session(session)
                    open_utc = pd.to_datetime(open_ts, utc=True)
                    close_utc = pd.to_datetime(close_ts, utc=True)
            elif calendar_kind == "pandas_market_calendars":
                sched = calendar.schedule(start_date=local_date, end_date=local_date)
                if not sched.empty:
                    open_ts = sched.iloc[0]["market_open"]
                    close_ts = sched.iloc[0]["market_close"]
                    open_utc = pd.to_datetime(open_ts, utc=True)
                    close_utc = pd.to_datetime(close_ts, utc=True)
        except Exception:
            open_utc = None
            close_utc = None

    if open_utc is None or close_utc is None:
        open_local = datetime.combine(local_date, dt_time(9, 30), tzinfo=tz)
        close_local = datetime.combine(local_date, dt_time(16, 0), tzinfo=tz)
        open_utc = open_local.astimezone(timezone.utc)
        close_utc = close_local.astimezone(timezone.utc)

    if asof_ts < open_utc:
        return "PRE"
    if asof_ts <= close_utc:
        return "REG"
    return "POST"


def market_close_utc_for_date(
    session_date: date,
    tz_name: str,
    calendar_kind: Optional[str],
    calendar: Any,
) -> datetime:
    tz = ZoneInfo(tz_name)
    if calendar is not None and calendar_kind is not None:
        try:
            if calendar_kind == "exchange_calendars":
                session = pd.Timestamp(session_date)
                if calendar.is_session(session):
                    _, close_ts = calendar.open_and_close_for_session(session)
                    return pd.to_datetime(close_ts, utc=True).to_pydatetime()
            elif calendar_kind == "pandas_market_calendars":
                sched = calendar.schedule(start_date=session_date, end_date=session_date)
                if not sched.empty:
                    close_ts = sched.iloc[0]["market_close"]
                    return pd.to_datetime(close_ts, utc=True).to_pydatetime()
        except Exception:
            pass

    close_local = datetime.combine(session_date, dt_time(16, 0), tzinfo=tz)
    return close_local.astimezone(timezone.utc)


def previous_trading_day(
    target_date: date,
    calendar_kind: Optional[str],
    calendar: Any,
) -> date:
    sessions = _calendar_sessions(
        calendar_kind,
        calendar,
        target_date - timedelta(days=10),
        target_date - timedelta(days=1),
    )
    if sessions:
        return sessions[-1]
    return target_date - timedelta(days=1)


def enforce_1dte_contract_exists(
    *,
    asof_utc: datetime,
    target_date: date,
    tz_name: str,
    calendar_kind: Optional[str],
    calendar: Any,
) -> None:
    creation_day = previous_trading_day(target_date, calendar_kind, calendar)
    close_utc = market_close_utc_for_date(creation_day, tz_name, calendar_kind, calendar)
    if asof_utc < close_utc:
        raise RuntimeError(
            "The requested 1DTE contract does not exist yet.\n"
            "1DTE contracts are created at market close the day before expiration."
        )

def fetch_event_by_slug(session: requests.Session, slug: str, cfg: Config) -> Optional[dict]:
    resp = session.get(GAMMA_EVENT_BY_SLUG.format(slug), timeout=cfg.request_timeout_s)
    resp.raise_for_status()
    ev = resp.json()
    return ev if isinstance(ev, dict) else None


def event_end_local(ev: Optional[dict], tz_name: str) -> Optional[date]:
    if not isinstance(ev, dict):
        return None
    for key in ("endDate", "endDateIso", "end_time"):
        local = _local_date(ev.get(key), tz_name)
        if local is not None:
            return local
    return None


def ensure_events_exist(
    tickers: List[str],
    target_date: date,
    contract_type: str,
    *,
    cfg: Config,
    slug_overrides: Dict[str, str],
    session: requests.Session,
) -> Optional[Tuple[str, str, str]]:
    for ticker in tickers:
        slug = slug_overrides.get(ticker) or build_market_slug(ticker, target_date, contract_type)
        try:
            ev = fetch_event_by_slug(session, slug, cfg)
        except requests.HTTPError as exc:
            resp = exc.response
            if resp is not None and resp.status_code == 404:
                return ticker, slug, "missing"
            raise
        if ev is None:
            return ticker, slug, "missing"
        end_local = event_end_local(ev, cfg.tz_name)
        if end_local != target_date:
            return ticker, slug, "date_mismatch"
    return None

def discover_finishweek_event_slug(
    *,
    ticker: str,
    expected_friday: date,
    cfg: Config,
    session: requests.Session,
) -> Optional[str]:
    """
    Uses Gamma public-search to find the event slug that:
      - ends on expected_friday (local date),
      - and matches the "above-on-<month>-<day>-<year>" pattern.
    """
    month = expected_friday.strftime("%B").lower()
    day = expected_friday.day
    year = expected_friday.year

    # Good query for Gamma search
    q = f"{ticker} above on {month} {day} {year}"

    try:
        r = session.get(
            GAMMA_PUBLIC_SEARCH,
            params={
                "q": q,
                "limit_per_type": 25,
                "events_status": "active",
                "keep_closed_markets": 0,
            },
            timeout=cfg.request_timeout_s,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    events = data.get("events") or []
    if not isinstance(events, list):
        return None

    # Desired slug suffix
    want_suffix = f"-above-on-{month}-{day}-{year}"

    # Prefer events whose endDate matches Friday AND slug matches suffix AND ticker matches
    candidates = []
    for ev in events:
        if not isinstance(ev, dict):
            continue

        slug = str(ev.get("slug") or "").strip()
        if not slug or want_suffix not in slug:
            continue

        # endDate check (event-level)
        ev_end = _local_date(ev.get("endDate"), cfg.tz_name)
        if ev_end is not None and ev_end != expected_friday:
            continue

        # ticker field sometimes present in search results
        ev_ticker = str(ev.get("ticker") or "").strip().upper()
        if ev_ticker and ev_ticker != ticker.upper():
            continue

        score = ev.get("score")
        score_f = None
        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None

        candidates.append((score_f, slug))

    if not candidates:
        return None

    # pick highest score if available, else first
    candidates.sort(key=lambda x: (x[0] is None, -(x[0] or 0.0)))
    return candidates[0][1]

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def finish_week_friday(today_local: date) -> date:
    # Next Friday relative to today (Mon->same week Fri, Sat/Sun->next week Fri)
    weekday = today_local.weekday()  # Mon=0 ... Sun=6
    delta = (4 - weekday) % 7        # Friday=4
    return today_local + timedelta(days=delta)

def trading_week_bounds(today_local: date) -> Tuple[date, date, date]:
    friday = finish_week_friday(today_local)
    monday = friday - timedelta(days=4)
    sunday = monday + timedelta(days=6)
    return monday, friday, sunday


def trading_week_bounds_for_date(anchor_date: date) -> Tuple[date, date, date]:
    monday = anchor_date - timedelta(days=anchor_date.weekday())
    friday = monday + timedelta(days=4)
    sunday = monday + timedelta(days=6)
    return monday, friday, sunday

def parse_tickers_arg(
    tickers_csv: Optional[str],
    tickers_list: Optional[str],
    default_tickers: Optional[List[str]] = None,
) -> List[str]:
    if tickers_list:
        tickers = [t.strip().upper() for t in tickers_list.split(",") if t.strip()]
        if not tickers:
            raise ValueError("Provided --tickers is empty after parsing.")
        return tickers

    if tickers_csv:
        df = pd.read_csv(tickers_csv)
        if "ticker" not in df.columns:
            raise ValueError(f"--tickers-csv must contain a 'ticker' column. Found: {list(df.columns)}")
        tickers = df["ticker"].dropna().astype(str).str.strip().str.upper().tolist()
        tickers = [t for t in tickers if t]
        if not tickers:
            raise ValueError("No tickers found in --tickers-csv.")
        return tickers

    return list(default_tickers or DEFAULT_TICKERS_WEEKLY)


def allowed_tickers_for_contract(contract_type: str) -> List[str]:
    ct = (contract_type or "weekly").strip().lower()
    if ct == "1dte":
        return DEFAULT_TICKERS_1DTE.copy()
    return DEFAULT_TICKERS_WEEKLY.copy()


def validate_ticker_universe(tickers: List[str], contract_type: str) -> None:
    allowed = set(allowed_tickers_for_contract(contract_type))
    disallowed = sorted({t for t in tickers if t not in allowed})
    if disallowed:
        allowed_sorted = ", ".join(sorted(allowed))
        raise ValueError(
            f"Tickers not allowed for contract_type={contract_type}: {', '.join(disallowed)}. "
            f"Allowed tickers: {allowed_sorted}."
        )


def load_slug_overrides(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Slug overrides file not found: {path}")

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("Slug overrides JSON must be an object/dict of ticker->slug.")
        return {str(k).upper(): str(v) for k, v in data.items()}

    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        if not {"ticker", "slug"}.issubset(df.columns):
            raise ValueError(f"Slug overrides CSV must contain columns ticker, slug. Found: {list(df.columns)}")
        out: Dict[str, str] = {}
        for _, r in df.iterrows():
            t = str(r["ticker"]).strip().upper()
            s = str(r["slug"]).strip()
            if t and s:
                out[t] = s
        return out

    raise ValueError("Slug overrides file must be .json or .csv")


def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def normalize_list_field(x):
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


def _as_bool(x) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    if isinstance(x, (int, float)):
        if x == 1:
            return True
        if x == 0:
            return False
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    return None


def round7(x) -> Optional[float]:
    if x is None:
        return None
    y = round(float(x), 7)
    return 0.0 if y == 0.0 else y


def make_session(cfg: Config) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def append_df_to_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    df.to_csv(path, mode="a", header=not file_exists, index=False)


def append_df_to_csv_with_schema(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    if not file_exists:
        df.to_csv(path, index=False)
        return

    existing_cols = list(pd.read_csv(path, nrows=0).columns)
    new_cols = [c for c in df.columns if c not in existing_cols]
    if not new_cols:
        df = df.reindex(columns=existing_cols)
        df.to_csv(path, mode="a", header=False, index=False)
        return

    # Extend existing file with new columns appended at the end
    existing = pd.read_csv(path)
    for col in new_cols:
        existing[col] = np.nan
    new_order = existing_cols + new_cols
    temp_path = path + ".tmp"
    existing = existing.reindex(columns=new_order)
    existing.to_csv(temp_path, index=False)
    df = df.reindex(columns=new_order)
    df.to_csv(temp_path, mode="a", header=False, index=False)
    os.replace(temp_path, path)


def unique_run_id(runs_dir: str, contract_type: str) -> str:
    prefix = (contract_type or "weekly").strip().lower()
    run_id = f"{prefix}-" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_path = os.path.join(runs_dir, run_id)
    while os.path.exists(run_path):
        time.sleep(1.0)
        run_id = f"{prefix}-" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        run_path = os.path.join(runs_dir, run_id)
    return run_id


def get_spot_asof_yf(
    ticker: str,
    asof_utc: datetime,
    tz_name: str,
) -> Tuple[Optional[float], str, Optional[str]]:
    """
    Fetch spot price as of a given UTC timestamp.

    Returns:
        Tuple of (price, source, actual_timestamp_iso) where actual_timestamp_iso
        is the ISO-formatted UTC timestamp of the actual price bar returned.
    """
    asof_ts = pd.to_datetime(asof_utc, utc=True, errors="coerce")
    if pd.isna(asof_ts):
        return None, "none", None

    # 1) Try intraday 1m bars (regular session only - no prepost for accurate spot)
    try:
        intraday = yf.download(
            ticker,
            period="5d",
            interval="1m",
            prepost=False,  # Regular session only for accurate spot price
            progress=False,
        )
        if intraday is not None and not intraday.empty and "Close" in intraday.columns:
            intraday = intraday.copy()
            idx = intraday.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            else:
                idx = idx.tz_convert("UTC")
            intraday.index = idx
            intraday = intraday.sort_index()
            intraday = intraday.loc[intraday.index <= asof_ts]
            if not intraday.empty:
                close_col = intraday["Close"]
                if isinstance(close_col, pd.DataFrame):
                    close_series = close_col.iloc[:, 0]
                else:
                    close_series = close_col
                price_val = close_series.iloc[-1]
                if isinstance(price_val, pd.Series):
                    price_val = price_val.iloc[0]
                price = float(price_val)
                actual_ts = intraday.index[-1]
                if np.isfinite(price):
                    return price, "intraday_1m", actual_ts.isoformat()
    except Exception:
        pass

    # 2) Fallback: previous close (strictly before local market date)
    try:
        daily = yf.download(
            ticker,
            period="5d",
            interval="1d",
            progress=False,
        )
        if daily is not None and not daily.empty and "Close" in daily.columns:
            daily = daily.sort_index()
            asof_local_date = asof_ts.tz_convert(ZoneInfo(tz_name)).date()
            idx = daily.index
            if idx.tz is None:
                local_dates = idx.date
            else:
                local_dates = idx.tz_convert(ZoneInfo(tz_name)).date
            mask = np.array([d < asof_local_date for d in local_dates], dtype=bool)
            if mask.any():
                filtered = daily.loc[mask]
                price = float(filtered["Close"].iloc[-1])
                actual_ts = filtered.index[-1]
                if idx.tz is None:
                    actual_ts = pd.Timestamp(actual_ts).tz_localize("UTC")
                else:
                    actual_ts = pd.Timestamp(actual_ts).tz_convert("UTC")
                if np.isfinite(price):
                    return price, "prev_close", actual_ts.isoformat()
    except Exception:
        pass

    return None, "none", None


# -----------------------------
# Slug construction (like v1.0.0)
# -----------------------------

def ticker_to_slug_prefix(ticker: str) -> str:
    t = ticker.strip().lower()
    return t.replace(".", "").replace("/", "-").replace(" ", "")


def build_weekly_slug(ticker: str, week_friday: date) -> str:
    prefix = ticker_to_slug_prefix(ticker)
    month = week_friday.strftime("%B").lower()
    return f"{prefix}-above-on-{month}-{week_friday.day}-{week_friday.year}"


def extract_strike_K_from_question(question: str) -> Optional[float]:
    if not isinstance(question, str):
        return None
    m = re.search(r"\$(\d+(?:\.\d+)?)", question)
    return float(m.group(1)) if m else None


def market_tradeable(m: dict) -> tuple[bool, str]:
    token_ids = normalize_list_field(m.get("clobTokenIds"))
    if not isinstance(token_ids, list) or len(token_ids) < 2 or not token_ids[0] or not token_ids[1]:
        return False, "missing_clob_tokens"
    if _as_bool(m.get("enableOrderBook")) is False:
        return False, "orderbook_disabled"
    if _as_bool(m.get("closed")) is True:
        return False, "closed"
    if _as_bool(m.get("archived")) is True:
        return False, "archived"
    if _as_bool(m.get("active")) is False:
        return False, "inactive"
    if _as_bool(m.get("isResolved")) is True:
        return False, "resolved"
    if _as_bool(m.get("acceptingOrders")) is False:
        return False, "not_accepting_orders"
    return True, "ok"


# -----------------------------
# CLOB pricing (ONLY)
# -----------------------------

def get_prices_bulk(
    token_ids: List[str],
    cfg: Config,
    side: str,
    session: requests.Session,
) -> Dict[str, Optional[float]]:
    side = side.upper().strip()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side must be BUY or SELL")

    out: Dict[str, Optional[float]] = {}

    seen = set()
    uniq: List[str] = []
    for tid in token_ids:
        if tid and tid not in seen:
            seen.add(tid)
            uniq.append(tid)

    CHUNK = 500
    for i in range(0, len(uniq), CHUNK):
        chunk = uniq[i:i + CHUNK]
        payload = [{"token_id": tid, "side": side} for tid in chunk]
        try:
            r = session.post(
                CLOB_PRICES,
                json=payload,
                timeout=cfg.request_timeout_s,
                headers={"Content-Type": "application/json"},
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                for tid in chunk:
                    val = data.get(str(tid))
                    out[tid] = safe_float(val.get(side)) if isinstance(val, dict) else None
            else:
                for tid in chunk:
                    out[tid] = None
        except Exception:
            for tid in chunk:
                out[tid] = None

    return out


# -----------------------------
# Polymarket snapshot (Gamma meta + CLOB pricing)
# -----------------------------

def fetch_polymarket_snapshot(
    tickers: List[str],
    week_monday: date,
    week_friday: date,
    week_sunday: date,
    target_date: date,
    contract_type: str,
    snapshot_time_utc: datetime,
    snapshot_time_local: str,
    snapshot_session: str,
    *,
    cfg: Config,
    slug_overrides: Dict[str, str],
    session: requests.Session,
) -> pd.DataFrame:
    """
    Fetch Polymarket markets ending on target_date in cfg.tz_name,
    for tickers' 'above-on-<month>-<day>-<year>' contracts.

    Behavior:
    - Try deterministic slug (override or build_market_slug) and keep only markets that end on target_date.
    - Filter per-market by market endDate (fallback event endDate) to enforce target_date ending.
    - Fetch CLOB BUY/SELL prices in bulk for tokens (no Gamma fallback here, by design).
    """
    rows: List[dict] = []
    errors: List[dict] = []
    all_token_ids: List[str] = []

    tz = ZoneInfo(cfg.tz_name)

    def _parse_dt_utc(x) -> Optional[pd.Timestamp]:
        dt = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt

    def _local_date_from_dt(dt_utc: Optional[pd.Timestamp]) -> Optional[date]:
        if dt_utc is None:
            return None
        try:
            return dt_utc.tz_convert(tz).date()
        except Exception:
            return None

    for t in tickers:
        slug_initial = slug_overrides.get(t) or build_market_slug(t, target_date, contract_type)
        slug = slug_initial

        try:
            ev = fetch_event_by_slug(session, slug, cfg)
            end_local = event_end_local(ev, cfg.tz_name)
            if end_local is None or end_local != target_date:
                errors.append({
                    "ticker": t,
                    "slug": slug,
                    "stage": "event_end_mismatch",
                    "error": f"event_end_local={end_local} expected={target_date.isoformat()}",
                })
                continue

            event_id = ev.get("id")
            event_title = ev.get("title") or ev.get("question")
            updated_at = ev.get("updatedAt")
            event_end_raw = ev.get("endDate") or ev.get("endDateIso") or ev.get("end_time")

            markets = ev.get("markets", [])
            if not isinstance(markets, list) or not markets:
                errors.append({"ticker": t, "slug": slug, "stage": "no_markets", "error": "no_markets"})
                continue

            for m in markets:
                ok, reason = market_tradeable(m)
                if not ok:
                    continue

                # HARD FILTER at market level:
                # If market has its own endDate, use it; else use event endDate.
                m_end_raw = m.get("endDate") or event_end_raw
                m_end_dt_utc = _parse_dt_utc(m_end_raw)
                m_end_local = _local_date_from_dt(m_end_dt_utc)
                if m_end_local is not None and m_end_local != target_date:
                    continue

                question = m.get("question") or m.get("title")
                K = extract_strike_K_from_question(question)

                token_ids = normalize_list_field(m.get("clobTokenIds"))
                yes_token_id = token_ids[0] if isinstance(token_ids, list) and len(token_ids) >= 2 else None
                no_token_id = token_ids[1] if isinstance(token_ids, list) and len(token_ids) >= 2 else None

                rows.append(
                    {
                        "snapshot_time_utc": snapshot_time_utc,
                        "snapshot_time_local": snapshot_time_local,
                        "snapshot_session": snapshot_session,
                        "week_monday": week_monday.isoformat(),
                        "week_friday": week_friday.isoformat(),
                        "week_sunday": week_sunday.isoformat(),
                        "ticker": t,
                        "slug": slug,
                        "event_id": event_id,
                        "event_title": event_title,
                        "event_updatedAt": updated_at,
                        "event_endDate": m_end_raw,  # store market end if present, else event end
                        "expiry_ts_utc": m_end_dt_utc,
                        "market_id": m.get("id"),
                        "condition_id": m.get("conditionId") or m.get("condition_id"),
                        "market_question": question,
                        "K": K,
                        "yes_token_id": str(yes_token_id) if yes_token_id else None,
                        "no_token_id": str(no_token_id) if no_token_id else None,
                        "pm_ok_meta": True,
                        "pm_reason_meta": reason,
                        "schema_version": SCHEMA_VERSION,
                    }
                )

                if yes_token_id:
                    all_token_ids.append(str(yes_token_id))
                if no_token_id:
                    all_token_ids.append(str(no_token_id))

        except Exception as e:
            errors.append({"ticker": t, "slug": slug, "stage": "fetch", "error": str(e)})

        # keep your pacing
        try:
            time.sleep(cfg.sleep_between_slugs_s)
        except Exception:
            pass

    pm = pd.DataFrame(rows)

    if pm.empty:
        print("[Polymarket] discovery returned 0 rows.")
        if errors:
            print(pd.DataFrame(errors).head(20))
        return pd.DataFrame(columns=[
            "snapshot_time_utc","snapshot_time_local","snapshot_session",
            "week_monday","week_friday","week_sunday","ticker","slug",
            "event_id","event_title","event_updatedAt","event_endDate","expiry_ts_utc","market_id","condition_id",
            "market_question","K","yes_token_id","no_token_id",
            "pPM_buy","qPM_buy","pPM_mid","qPM_mid","yes_spread","no_spread",
            "pm_ok","pm_reason","pm_ok_meta","pm_reason_meta","T_days","schema_version"
        ])

    pm["snapshot_time_utc"] = pd.to_datetime(pm["snapshot_time_utc"], utc=True, errors="coerce")
    pm["event_endDate"] = pd.to_datetime(pm["event_endDate"], utc=True, errors="coerce")
    pm["expiry_ts_utc"] = pd.to_datetime(pm.get("expiry_ts_utc"), utc=True, errors="coerce")
    pm["expiry_ts_utc"] = pm["expiry_ts_utc"].fillna(pm["event_endDate"])
    pm["K"] = pd.to_numeric(pm["K"], errors="coerce")
    pm["schema_version"] = SCHEMA_VERSION

    # Prices (CLOB only)
    buy = get_prices_bulk(all_token_ids, cfg, side="BUY", session=session)
    sell = get_prices_bulk(all_token_ids, cfg, side="SELL", session=session)

    pm["yes_token_id"] = pm["yes_token_id"].astype("string")
    pm["no_token_id"] = pm["no_token_id"].astype("string")

    pm["yes_buy"] = pm["yes_token_id"].map(buy)
    pm["yes_sell"] = pm["yes_token_id"].map(sell)
    pm["no_buy"] = pm["no_token_id"].map(buy)
    pm["no_sell"] = pm["no_token_id"].map(sell)

    for c in ["yes_buy","yes_sell","no_buy","no_sell"]:
        pm[c] = pd.to_numeric(pm[c], errors="coerce")

    yes_bid = pm[["yes_buy","yes_sell"]].min(axis=1)
    yes_ask = pm[["yes_buy","yes_sell"]].max(axis=1)
    no_bid  = pm[["no_buy","no_sell"]].min(axis=1)
    no_ask  = pm[["no_buy","no_sell"]].max(axis=1)

    pm["pPM_buy"] = yes_ask
    pm["qPM_buy"] = no_ask
    pm["pPM_mid"] = np.where(yes_bid.notna() & yes_ask.notna(), 0.5*(yes_bid+yes_ask), np.nan)
    pm["qPM_mid"] = np.where(no_bid.notna() & no_ask.notna(), 0.5*(no_bid+no_ask), np.nan)
    pm["yes_spread"] = yes_ask - yes_bid
    pm["no_spread"]  = no_ask - no_bid

    pm["pm_ok"] = pm["pPM_buy"].notna() | pm["qPM_buy"].notna()
    pm["pm_reason"] = np.where(pm["pm_ok"], "ok", "no_clob_price")

    # IMPORTANT: do NOT normalize expiry_ts_utc; use actual UTC timestamp
    pm["T_days"] = (pm["expiry_ts_utc"] - pm["snapshot_time_utc"]).dt.total_seconds() / (60*60*24)

    pm.drop(columns=["yes_buy","yes_sell","no_buy","no_sell"], inplace=True, errors="ignore")

    n_total = len(pm)
    n_ok = int(pm["pm_ok"].sum())
    print(f"[Polymarket] rows={n_total} ok_any={n_ok}")
    if errors:
        print(f"[Polymarket] errors={len(errors)} (showing up to 20)")
        print(pd.DataFrame(errors).head(20))

    return pm


# -----------------------------
# Historical data enrichment (rv20, dividends, forward price)
# -----------------------------

@dataclass
class RV20Result:
    """Result of rv20 computation with metadata about source."""
    rv20: float
    source: str  # "realized_20d", "realized_{N}d_scaled", "implied_atm", "fallback"
    window: int  # Number of days used (0 for implied/fallback)
    is_missing: bool  # True if fallback was used


def _extract_close_series(hist: pd.DataFrame, ticker_str: str) -> Optional[pd.Series]:
    """
    Extract close price series from yfinance DataFrame, handling MultiIndex and various column names.
    """
    if hist is None or hist.empty:
        return None

    close = None

    # Try standard Close column
    if "Close" in hist.columns:
        close = hist["Close"]
    # Try Adj Close as fallback
    elif "Adj Close" in hist.columns:
        close = hist["Adj Close"]
    # Handle MultiIndex columns (yfinance sometimes returns these)
    elif isinstance(hist.columns, pd.MultiIndex):
        # Try first level
        if "Close" in hist.columns.get_level_values(0):
            close = hist["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        elif "Adj Close" in hist.columns.get_level_values(0):
            close = hist["Adj Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        # Try last level
        elif "Close" in hist.columns.get_level_values(-1):
            try:
                close = hist.xs("Close", axis=1, level=-1)
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            except Exception:
                pass
        elif "Adj Close" in hist.columns.get_level_values(-1):
            try:
                close = hist.xs("Adj Close", axis=1, level=-1)
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
            except Exception:
                pass

    # Ensure we have a Series
    if close is not None:
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        if isinstance(close, pd.Series):
            close = close.dropna().sort_index()
            if len(close) > 0:
                return close
    return None


def _fetch_yf_history_with_retry(
    ticker_str: str,
    start_date: date,
    end_date: date,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Fetch yfinance history with retries and fallback to Ticker.history().
    Uses explicit start/end dates instead of period for time-safety.
    """
    hist = None
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    for attempt in range(max_retries):
        try:
            hist = yf.download(
                ticker_str,
                start=start_str,
                end=end_str,
                interval="1d",
                progress=False,
            )
            if hist is not None and not hist.empty:
                return hist
        except Exception:
            pass

        # Short backoff between retries
        if attempt < max_retries - 1:
            time.sleep(0.5 * (attempt + 1))

    # Fallback to Ticker.history()
    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker_str)
            hist = stock.history(start=start_str, end=end_str, interval="1d")
            if hist is not None and not hist.empty:
                return hist
        except Exception:
            pass

        if attempt < max_retries - 1:
            time.sleep(0.5 * (attempt + 1))

    return None


def _compute_rv_from_closes(
    closes: np.ndarray,
    target_window: int = 20,
) -> Tuple[Optional[float], int]:
    """
    Compute realized volatility from close prices.

    Returns (annualized_vol, window_used).
    If window < target_window, scales the vol estimate: rv_target ≈ rv_N * sqrt(target/N)
    """
    n = len(closes)
    if n < 2:
        return None, 0

    log_returns = np.diff(np.log(closes))
    if len(log_returns) < 1:
        return None, 0

    daily_vol = float(np.std(log_returns, ddof=1))
    if not np.isfinite(daily_vol) or daily_vol <= 0:
        return None, 0

    window_used = n  # Number of close prices used
    annualized_vol = daily_vol * math.sqrt(252)

    # Scale if we have fewer than target window
    if window_used < target_window:
        # Scale factor: estimate what 20-day vol would be
        # This is an approximation assuming vol scales with sqrt(time)
        scale = math.sqrt(target_window / max(window_used, 1))
        annualized_vol = annualized_vol * scale

    return float(np.clip(annualized_vol, 0.0, 10.0)), window_used


def compute_rv20_robust(
    ticker_str: str,
    asof_utc: datetime,
    tz_name: str,
    cfg: Config,
    atm_sigma: Optional[float] = None,
) -> RV20Result:
    """
    Compute 20-day realized volatility (annualized) with multiple fallback strategies.

    Fallback order:
    1. Full 20-day realized vol from yfinance
    2. Short-window realized vol (2-19 days), scaled to 20-day equivalent
    3. ATM implied vol from option chain (if provided)
    4. Config fallback value (rv20_fallback)

    Always returns a valid RV20Result with non-NaN rv20.
    """
    asof_local = _local_date(asof_utc, tz_name)
    if asof_local is None:
        # Can't determine local date - use fallback
        if atm_sigma is not None and np.isfinite(atm_sigma) and atm_sigma > 0:
            return RV20Result(
                rv20=float(np.clip(atm_sigma, 0.0, 10.0)),
                source="implied_atm",
                window=0,
                is_missing=False,
            )
        return RV20Result(
            rv20=cfg.rv20_fallback,
            source="fallback",
            window=0,
            is_missing=True,
        )

    # Try yfinance fetch
    try:
        # Fetch enough history: 400 days to cover 20 trading days plus holidays/weekends
        start_date = asof_local - timedelta(days=cfg.rv20_history_days)
        end_date = asof_local + timedelta(days=1)  # Include asof_local

        hist = _fetch_yf_history_with_retry(
            ticker_str,
            start_date,
            end_date,
            max_retries=cfg.rv20_max_retries,
        )

        close = _extract_close_series(hist, ticker_str)

        if close is not None and len(close) >= cfg.rv20_min_window:
            # Filter to bars at or before asof_local (time-safe)
            idx = close.index
            if idx.tz is None:
                local_dates = idx.date
            else:
                local_dates = idx.tz_convert(ZoneInfo(tz_name)).date
            mask = np.array([d <= asof_local for d in local_dates], dtype=bool)

            if mask.any():
                filtered_close = close.loc[mask]

                # Try full 20-day window first
                if len(filtered_close) >= 20:
                    closes = filtered_close.iloc[-20:].to_numpy(dtype=float)
                    rv, window = _compute_rv_from_closes(closes, target_window=20)
                    if rv is not None:
                        return RV20Result(
                            rv20=rv,
                            source="realized_20d",
                            window=20,
                            is_missing=False,
                        )

                # Fallback: short-window realized vol (scale to 20-day equivalent)
                if len(filtered_close) >= cfg.rv20_min_window:
                    closes = filtered_close.iloc[-len(filtered_close):].to_numpy(dtype=float)
                    rv, window = _compute_rv_from_closes(closes, target_window=20)
                    if rv is not None:
                        return RV20Result(
                            rv20=rv,
                            source=f"realized_{window}d_scaled",
                            window=window,
                            is_missing=False,
                        )

    except Exception as e:
        # Log but continue to fallbacks
        print(f"[rv20] yfinance fetch failed for {ticker_str}: {e}")

    # Fallback: ATM implied vol if provided
    if atm_sigma is not None and np.isfinite(atm_sigma) and atm_sigma > 0:
        return RV20Result(
            rv20=float(np.clip(atm_sigma, 0.0, 10.0)),
            source="implied_atm",
            window=0,
            is_missing=False,
        )

    # Final fallback: config default
    return RV20Result(
        rv20=cfg.rv20_fallback,
        source="fallback",
        window=0,
        is_missing=True,
    )


def compute_rv20(ticker_str: str, asof_utc: datetime, tz_name: str) -> Optional[float]:
    """
    Legacy wrapper for backward compatibility.
    Compute 20-day realized volatility (annualized) from historical close data.

    Returns None if insufficient data available (original behavior).
    Use compute_rv20_robust() for guaranteed non-None result.
    """
    cfg = Config()  # Use default config
    result = compute_rv20_robust(ticker_str, asof_utc, tz_name, cfg)
    # Return None for fallback to maintain legacy behavior
    if result.is_missing:
        return None
    return result.rv20


def fetch_dividend_yield(ticker_str: str, asof_utc: datetime, tz_name: str) -> Optional[float]:
    """
    Fetch trailing 12-month dividend yield from yfinance.

    Returns annualized dividend yield, or None if not available.
    """
    try:
        asof_local = _local_date(asof_utc, tz_name)
        if asof_local is None:
            return None

        stock = yf.Ticker(ticker_str)

        # Try to get dividend data
        div_yield = None

        if hasattr(stock, "dividends") and not stock.dividends.empty:
            divs = stock.dividends.copy()
            idx = divs.index
            if idx.tz is None:
                div_dates = idx.date
            else:
                div_dates = idx.tz_convert(ZoneInfo(tz_name)).date
            start = asof_local - timedelta(days=365)
            mask = np.array([(d >= start) and (d <= asof_local) for d in div_dates], dtype=bool)
            if mask.any():
                annual_div = float(divs.loc[mask].sum())
                if annual_div >= 0:
                    hist = yf.download(
                        ticker_str,
                        period="400d",
                        interval="1d",
                        progress=False,
                    )
                    if not hist.empty and "Close" in hist.columns:
                        idx = hist.index
                        if idx.tz is None:
                            local_dates = idx.date
                        else:
                            local_dates = idx.tz_convert(ZoneInfo(tz_name)).date
                        price_mask = np.array([d <= asof_local for d in local_dates], dtype=bool)
                        if price_mask.any():
                            price = float(hist.loc[price_mask, "Close"].iloc[-1])
                            if price > 0:
                                div_yield = annual_div / price

        if div_yield is not None and np.isfinite(div_yield) and 0.0 <= div_yield <= 0.2:
            return float(div_yield)

        # Fallback: use info only if available (may be forward-looking)
        if hasattr(stock, "info") and isinstance(stock.info, dict):
            for key in ["trailingAnnualDividendYield", "dividendYield", "annualDividendYield"]:
                if key in stock.info:
                    val = stock.info[key]
                    if val is not None:
                        div_yield = float(val)
                        if np.isfinite(div_yield) and 0.0 <= div_yield <= 1.0:
                            return div_yield

        return None
    except Exception:
        return None


def count_splits_in_range(
    ticker_str: str,
    start_date: date,
    end_date: date,
    tz_name: str,
) -> Optional[int]:
    try:
        if start_date is None or end_date is None or start_date > end_date:
            return None
        stock = yf.Ticker(ticker_str)
        splits = getattr(stock, "splits", None)
        if splits is None or getattr(splits, "empty", True):
            return 0
        idx = splits.index
        if idx.tz is None:
            split_dates = idx.date
        else:
            split_dates = idx.tz_convert(ZoneInfo(tz_name)).date
        mask = np.array([(d >= start_date) and (d <= end_date) for d in split_dates], dtype=bool)
        return int(mask.sum())
    except Exception:
        return None


def infer_asof_fallback_days(spot_source: Optional[str]) -> Optional[int]:
    if not spot_source:
        return None
    src = spot_source.strip().lower()
    if src in ("intraday_1m", "intraday_prepost_1m"):
        return 0
    if src == "prev_close":
        return 1
    return None


def compute_band_counts(
    strikes: np.ndarray,
    spot: float,
    *,
    min_inside: int = 10,
    start_abs_logm: float = 0.06,
    cap_abs_logm: float = 0.10,
    step: float = 0.01,
) -> Tuple[float, float]:
    k = np.asarray(strikes, dtype=float)
    k = k[np.isfinite(k) & (k > 0)]
    if k.size == 0 or not np.isfinite(spot) or spot <= 0:
        return np.nan, np.nan
    k_min = float(np.min(k))
    k_max = float(np.max(k))

    used = float(start_abs_logm)
    while used <= cap_abs_logm + 1e-12:
        logm = np.log(k / float(spot))
        k_band = k[np.abs(logm) <= used]
        k_inside = k_band[(k_band > k_min) & (k_band < k_max)]
        if int(k_inside.size) >= int(min_inside):
            return float(k_band.size), float(k_inside.size)
        used += step

    logm = np.log(k / float(spot))
    k_band = k[np.abs(logm) <= cap_abs_logm]
    k_inside = k_band[(k_band > k_min) & (k_band < k_max)]
    return float(k_band.size), float(k_inside.size)


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


# Cache for historical computations (avoid redundant API calls)
_rv20_cache: Dict[str, RV20Result] = {}
_div_yield_cache: Dict[str, Optional[float]] = {}
_split_count_cache: Dict[Tuple[str, date, date], Optional[int]] = {}


# -----------------------------
# pRN estimation (yfinance)
# -----------------------------

def compute_d2(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    return (math.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def risk_neutral_prob_bs_tail(S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    d2 = compute_d2(S, K, T, r, sigma)
    return float(norm.cdf(d2)) if d2 is not None else None


def _monotone_decreasing_calls(c: np.ndarray) -> np.ndarray:
    c_rev = c[::-1]
    c_rev_mon = np.maximum.accumulate(c_rev)
    return c_rev_mon[::-1]


def _enforce_pRN_monotone_decreasing_by_K(gr: pd.DataFrame) -> pd.DataFrame:
    # pRN(K) must be non-increasing in K
    out = gr.copy()
    out["K"] = pd.to_numeric(out["K"], errors="coerce")
    out["pRN_raw"] = pd.to_numeric(out["pRN_raw"], errors="coerce")
    out = out.dropna(subset=["K"]).sort_values("K", ascending=True)

    if out.empty:
        out["pRN"] = np.nan
        out["rn_monotone_adjusted"] = False
        return out

    p = out["pRN_raw"].clip(0.0, 1.0).interpolate(limit_direction="both").ffill().bfill()
    p_mon = np.maximum.accumulate(p.to_numpy()[::-1])[::-1]
    out["pRN"] = np.clip(p_mon, 0.0, 1.0)
    out["rn_monotone_adjusted"] = (out["pRN_raw"].round(7) != out["pRN"].round(7)).fillna(False)
    out["pRN"] = out["pRN"].round(7)
    out["pRN_raw"] = out["pRN_raw"].round(7)
    return out


def compute_pRN_snapshot(
    pm: pd.DataFrame,
    cfg: Config,
    *,
    asof_utc: datetime,
    tz_name: str,
    allow_nonlive: bool,
) -> pd.DataFrame:
    if pm is None or pm.empty:
        return pd.DataFrame(columns=[
            "ticker","event_endDate","K","S","pRN_raw","pRN","rn_method","rn_quote_source",
            "n_calls_used","calls_k_min","calls_k_max","deltaK","rn_monotone_adjusted",
            "spot_source","spot_asof_utc","rn_asof_utc",
            "rv20","rv20_source","rv20_window","is_missing_rv20","dividend_yield","forward_price",
            "rel_spread_median","n_chain_raw","n_chain_used","n_band_raw","n_band_inside",
            "dropped_intrinsic","asof_fallback_days","expiry_fallback_days",
            "split_events_in_preload_range","spot_scale_used",
        ])

    asof_ts = pd.to_datetime(asof_utc, utc=True, errors="coerce")
    if pd.isna(asof_ts):
        raise ValueError("asof_utc could not be parsed.")
    asof_local_date = _local_date(asof_ts, tz_name)

    nonlive_block = False
    now_utc = datetime.now(timezone.utc)
    if not allow_nonlive and asof_ts.date() != now_utc.date():
        nonlive_block = True
        print(
            f"[yfinance] Non-live snapshot detected (asof={asof_ts.date()} now={now_utc.date()}). "
            "Option chain disabled; set pRN to NaN. Use --allow-nonlive to override."
        )

    df = pm.copy()
    df["event_endDate"] = pd.to_datetime(df["event_endDate"], utc=True, errors="coerce")
    df["expiry_ts_utc"] = pd.to_datetime(df.get("expiry_ts_utc"), utc=True, errors="coerce")
    df["expiry_ts_utc"] = df["expiry_ts_utc"].fillna(df["event_endDate"])
    df["expiry_date"] = df["expiry_ts_utc"].dt.tz_convert(ZoneInfo(tz_name)).dt.date
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["T_days"] = pd.to_numeric(df["T_days"], errors="coerce")
    df = df[df["expiry_date"].notna() & df["K"].notna() & (df["K"] > 0) & df["T_days"].notna() & (df["T_days"] > 0)].copy()

    rows: List[dict] = []
    spot_cache: Dict[str, Tuple[Optional[float], str, Optional[str]]] = {}
    asof_iso = asof_ts.isoformat()

    grouped = df.groupby(["ticker", "expiry_date"], dropna=True)
    print(f"[yfinance] groups={len(grouped)} (ticker, expiry)")

    for (ticker, expiry_date), g in grouped:
        try:
            if ticker not in spot_cache:
                spot_cache[ticker] = get_spot_asof_yf(str(ticker), asof_ts.to_pydatetime(), tz_name)
            S_spot, spot_source, spot_actual_ts = spot_cache[ticker]
            # Use actual price timestamp, not requested timestamp
            spot_asof = spot_actual_ts if S_spot is not None else None
            rn_asof = asof_iso
            asof_fallback_days = infer_asof_fallback_days(spot_source)
            expiry_fallback_days = 0

            split_count = None
            if asof_local_date is not None and isinstance(expiry_date, date):
                split_key = (str(ticker), asof_local_date, expiry_date)
                if split_key not in _split_count_cache:
                    _split_count_cache[split_key] = count_splits_in_range(
                        str(ticker),
                        asof_local_date,
                        expiry_date,
                        tz_name,
                    )
                split_count = _split_count_cache.get(split_key)

            spot_scale_used = "split_adj" if (split_count is not None and split_count > 0) else "raw"
            rel_spread_median = np.nan
            n_chain_raw = np.nan
            n_chain_used = np.nan
            n_band_raw = np.nan
            n_band_inside = np.nan
            dropped_intrinsic = np.nan

            # Compute historical features once per ticker (with caching)
            # Use compute_rv20_robust which guarantees a non-None result
            if ticker not in _rv20_cache:
                _rv20_cache[ticker] = compute_rv20_robust(
                    str(ticker), asof_ts.to_pydatetime(), tz_name, cfg, atm_sigma=None
                )
            rv20_result = _rv20_cache[ticker]

            if ticker not in _div_yield_cache:
                _div_yield_cache[ticker] = fetch_dividend_yield(str(ticker), asof_ts.to_pydatetime(), tz_name)
            div_yield_ticker = _div_yield_cache[ticker]
            if div_yield_ticker is None or not np.isfinite(div_yield_ticker):
                div_yield_ticker = 0.0
                _div_yield_cache[ticker] = div_yield_ticker

            common_fields = {
                "spot_source": spot_source,
                "spot_asof_utc": spot_asof,
                "rn_asof_utc": rn_asof,
                "rv20": rv20_result.rv20,
                "rv20_source": rv20_result.source,
                "rv20_window": rv20_result.window,
                "is_missing_rv20": rv20_result.is_missing,
                "dividend_yield": div_yield_ticker,
                "forward_price": np.nan,
                "rel_spread_median": rel_spread_median,
                "n_chain_raw": n_chain_raw,
                "n_chain_used": n_chain_used,
                "n_band_raw": n_band_raw,
                "n_band_inside": n_band_inside,
                "dropped_intrinsic": dropped_intrinsic,
                "asof_fallback_days": asof_fallback_days,
                "expiry_fallback_days": expiry_fallback_days,
                "split_events_in_preload_range": split_count if split_count is not None else np.nan,
                "spot_scale_used": spot_scale_used,
            }

            if nonlive_block:
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_date,
                        "K": round7(float(r.K)),
                        "S": round7(S_spot) if S_spot is not None else np.nan,
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "disabled_nonlive",
                        "rn_quote_source": None,
                        "n_calls_used": 0,
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                        **common_fields,
                    })
                continue

            if S_spot is None or not np.isfinite(S_spot):
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_date,
                        "K": round7(float(r.K)),
                        "S": np.nan,
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "no_spot_asof",
                        "rn_quote_source": None,
                        "n_calls_used": 0,
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                        **common_fields,
                    })
                continue

            S = float(S_spot)

            stock = yf.Ticker(str(ticker))
            expiry_str = pd.Timestamp(expiry_date).strftime("%Y-%m-%d")

            opts = getattr(stock, "options", [])
            if expiry_str not in opts:
                # still return rows, but missing pRN
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_date,
                        "K": round7(float(r.K)),
                        "S": round7(S),
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "no_chain",
                        "rn_quote_source": None,
                        "n_calls_used": 0,
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                        **common_fields,
                    })
                continue

            chain = stock.option_chain(expiry_str)
            calls_raw = chain.calls.copy()
            if calls_raw.empty:
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_date,
                        "K": round7(float(r.K)),
                        "S": round7(S),
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "empty_calls",
                        "rn_quote_source": None,
                        "n_calls_used": 0,
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                        **common_fields,
                    })
                continue

            # ATM sigma for BS fallback
            st = pd.to_numeric(calls_raw.get("strike"), errors="coerce")
            iv = pd.to_numeric(calls_raw.get("impliedVolatility"), errors="coerce")
            atm_sigma = float("nan")
            if st.notna().any() and iv.notna().any():
                atm_idx = (st - S).abs().idxmin()
                atm_sigma = float(iv.loc[atm_idx]) if pd.notna(iv.loc[atm_idx]) else float(np.nanmedian(iv))

            keep_cols = [c for c in ["strike", "bid", "ask", "lastPrice"] if c in calls_raw.columns]
            calls = calls_raw[keep_cols].copy()
            calls["strike"] = pd.to_numeric(calls["strike"], errors="coerce")
            calls["bid"] = pd.to_numeric(calls.get("bid"), errors="coerce")
            calls["ask"] = pd.to_numeric(calls.get("ask"), errors="coerce")
            calls["lastPrice"] = pd.to_numeric(calls.get("lastPrice"), errors="coerce")

            calls["mid_ba"] = np.where(
                calls["bid"].notna() & calls["ask"].notna()
                & (calls["bid"] >= 0) & (calls["ask"] > 0) & (calls["ask"] >= calls["bid"]),
                0.5 * (calls["bid"] + calls["ask"]),
                np.nan,
            )
            calls["spread"] = calls["ask"] - calls["bid"]
            calls["rel_spread"] = calls["spread"] / calls["mid_ba"]

            calls_ba = calls.dropna(subset=["strike", "mid_ba"]).copy()
            calls_ba = calls_ba[calls_ba["mid_ba"] > 0].copy()
            calls_ba = calls_ba[(calls_ba["rel_spread"].isna()) | (calls_ba["rel_spread"] <= cfg.rel_spread_cap)].copy()
            n_chain_raw = int(len(calls_raw))
            if "rel_spread" in calls_ba.columns and not calls_ba["rel_spread"].empty:
                rel_spread_median = float(np.nanmedian(calls_ba["rel_spread"].to_numpy(dtype=float)))
            else:
                rel_spread_median = np.nan
            common_fields.update({
                "rel_spread_median": rel_spread_median,
                "n_chain_raw": n_chain_raw,
            })

            use_lastprice = False
            if len(calls_ba) < cfg.min_calls_for_bidask:
                calls_lp = calls.dropna(subset=["strike", "lastPrice"]).copy()
                calls_lp = calls_lp[calls_lp["lastPrice"] > 0].copy()
                calls_lp["mid"] = calls_lp["lastPrice"]
                calls_use = calls_lp
                use_lastprice = True
            else:
                calls_ba["mid"] = calls_ba["mid_ba"]
                calls_use = calls_ba

            calls_use = calls_use.dropna(subset=["strike", "mid"]).copy()
            calls_use = calls_use.sort_values("strike").drop_duplicates("strike", keep="last")
            n_chain_used = int(len(calls_use))
            common_fields.update({
                "n_chain_used": n_chain_used,
            })

            if len(calls_use) < 3:
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_date,
                        "K": round7(float(r.K)),
                        "S": round7(S),
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "too_few_calls",
                        "rn_quote_source": "lastPrice" if use_lastprice else "bidask_mid",
                        "n_calls_used": int(len(calls_use)),
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                        **common_fields,
                    })
                continue

            # reference T
            T_years_ref = float(np.nanmedian((g["T_days"].to_numpy(dtype=float) / 365.25)))
            T_years_ref = max(T_years_ref, 1e-6)

            k_arr = calls_use["strike"].to_numpy(dtype=float)
            c_arr = calls_use["mid"].to_numpy(dtype=float)

            # no-arb guard (drop extreme below intrinsic, clip remaining)
            discK = k_arr * np.exp(-cfg.risk_free_rate * T_years_ref)
            intrinsic_lb = np.maximum(S - discK, 0.0)
            too_far_below = c_arr < 0.80 * intrinsic_lb
            dropped_intrinsic = int(np.sum(too_far_below))
            common_fields.update({
                "dropped_intrinsic": dropped_intrinsic,
            })
            calls_use = calls_use.loc[~too_far_below].copy()

            if len(calls_use) < 3:
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_date,
                        "K": round7(float(r.K)),
                        "S": round7(S),
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "arb_drop_all",
                        "rn_quote_source": "lastPrice" if use_lastprice else "bidask_mid",
                        "n_calls_used": int(len(calls_use)),
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                        **common_fields,
                    })
                continue

            k_arr = calls_use["strike"].to_numpy(dtype=float)
            c_arr = calls_use["mid"].to_numpy(dtype=float)
            n_band_raw, n_band_inside = compute_band_counts(k_arr, S)
            common_fields.update({
                "n_band_raw": n_band_raw,
                "n_band_inside": n_band_inside,
            })
            discK = k_arr * np.exp(-cfg.risk_free_rate * T_years_ref)
            intrinsic_lb = np.maximum(S - discK, 0.0)
            c_arr = np.maximum(c_arr, intrinsic_lb)
            c_arr = np.minimum(c_arr, 1.5 * S)

            # enforce call monotone decreasing in K
            c_mon = _monotone_decreasing_calls(c_arr)
            if cfg.call_smooth_window >= 3:
                c_mon = pd.Series(c_mon).rolling(cfg.call_smooth_window, center=True, min_periods=1).median().to_numpy()

            k_min, k_max = float(np.min(k_arr)), float(np.max(k_arr))
            n_calls_used = int(len(k_arr))

            for r in g.itertuples(index=False):
                K = float(r.K)
                T_years = float(r.T_days) / 365.25
                T_years = max(T_years, 1e-6)

                if K <= k_min:
                    prn = 1.0
                    rn_method = "extrap_lowK"
                    dK = None
                elif K >= k_max:
                    prn = 0.0
                    rn_method = "extrap_highK"
                    dK = None
                else:
                    i = int(np.searchsorted(k_arr, K))
                    left = max(i - cfg.slope_bracket_width, 0)
                    right = min(i + 1, len(k_arr) - 1)

                    K1, K2 = float(k_arr[left]), float(k_arr[right])
                    C1, C2 = float(c_mon[left]), float(c_mon[right])
                    dK = (K2 - K1) if K2 > K1 else None

                    prn = None
                    rn_method = "call_slope"
                    if dK is not None and dK > 0:
                        slope = (C2 - C1) / dK
                        if np.isfinite(slope) and slope <= 0:
                            prn_raw = -math.exp(cfg.risk_free_rate * T_years) * slope
                            if np.isfinite(prn_raw):
                                prn = float(np.clip(prn_raw, 0.0, 1.0))
                        else:
                            rn_method = "bad_slope"
                    else:
                        rn_method = "no_bracket"

                    max_deltaK = cfg.max_deltaK_frac_spot * S
                    if (prn is None) or (dK is not None and dK > max_deltaK):
                        if np.isfinite(atm_sigma) and atm_sigma > 0:
                            prn_fb = risk_neutral_prob_bs_tail(S, K, T_years, cfg.risk_free_rate, float(atm_sigma))
                            if prn_fb is not None and np.isfinite(prn_fb):
                                prn = float(np.clip(prn_fb, 0.0, 1.0))
                                rn_method = "bs_atm_iv_fallback"
                        else:
                            rn_method = "missing_no_iv"

                    if prn is None:
                        prn = np.nan

                # Compute forward price if we have dividend yield
                forward_price = None
                if div_yield_ticker is not None:
                    forward_price = compute_forward_price(S, cfg.risk_free_rate, div_yield_ticker, T_years)

                rows.append({
                    "ticker": ticker,
                    "event_endDate": expiry_date,
                    "K": round7(K),
                    "S": round7(S),
                    "pRN_raw": round7(prn) if np.isfinite(prn) else np.nan,
                    "pRN": np.nan,  # filled after monotone enforcement
                    "rn_method": rn_method,
                    "rn_quote_source": "lastPrice" if use_lastprice else "bidask_mid",
                    "n_calls_used": n_calls_used,
                    "calls_k_min": round7(k_min),
                    "calls_k_max": round7(k_max),
                    "deltaK": round7(dK) if dK is not None else np.nan,
                    "rn_monotone_adjusted": False,  # updated later
                    "forward_price": round7(forward_price) if forward_price is not None else np.nan,
                    **common_fields,
                })

        except Exception as e:
            print(f"[yfinance] ❌ {ticker} {expiry_date}: {e}")

    rn = pd.DataFrame(rows)
    if rn.empty:
        return rn

    # enforce monotone pRN per (ticker, expiry)
    rn["event_endDate"] = pd.to_datetime(rn["event_endDate"], utc=True, errors="coerce").dt.normalize()
    rn["K"] = pd.to_numeric(rn["K"], errors="coerce")
    rn["pRN_raw"] = pd.to_numeric(rn["pRN_raw"], errors="coerce")

    out_parts = []
    for (t, ex), gr in rn.groupby(["ticker", "event_endDate"], dropna=True):
        if cfg.apply_monotone_pav:
            out_parts.append(_enforce_pRN_monotone_decreasing_by_K(gr))
        else:
            tmp = gr.copy()
            tmp["pRN"] = tmp["pRN_raw"]
            tmp["rn_monotone_adjusted"] = False
            out_parts.append(tmp)

    rn2 = pd.concat(out_parts, ignore_index=True) if out_parts else rn
    keep = [
        "ticker","event_endDate","K","S","pRN_raw","pRN",
        "rn_method","rn_quote_source","rn_monotone_adjusted",
        "n_calls_used","calls_k_min","calls_k_max","deltaK",
        "spot_source","spot_asof_utc","rn_asof_utc",
        "rv20","rv20_source","rv20_window","is_missing_rv20","dividend_yield","forward_price",  # Historical features computed from yfinance
        "rel_spread_median","n_chain_raw","n_chain_used","n_band_raw","n_band_inside",
        "dropped_intrinsic","asof_fallback_days","expiry_fallback_days",
        "split_events_in_preload_range","spot_scale_used",
    ]
    keep = [c for c in keep if c in rn2.columns]
    return rn2[keep].copy()


# -----------------------------
# Merge + simple edge columns (vs pRN)
# -----------------------------

def merge_pm_rn(pm: pd.DataFrame, rn: pd.DataFrame) -> pd.DataFrame:
    if pm is None or pm.empty:
        return pm.copy() if pm is not None else pd.DataFrame()
    if rn is None or rn.empty:
        out = pm.copy()
        out["S"] = np.nan
        out["pRN_raw"] = np.nan
        out["pRN"] = np.nan
        out["rn_method"] = None
        out["rn_quote_source"] = None
        out["rn_monotone_adjusted"] = False
        out["spot_source"] = None
        out["spot_asof_utc"] = None
        out["rn_asof_utc"] = None
        return out

    left = pm.copy()
    right = rn.copy()

    left["event_endDate"] = pd.to_datetime(left["event_endDate"], utc=True, errors="coerce").dt.normalize()
    right["event_endDate"] = pd.to_datetime(right["event_endDate"], utc=True, errors="coerce").dt.normalize()

    left["K_round"] = pd.to_numeric(left["K"], errors="coerce").round(4)
    right["K_round"] = pd.to_numeric(right["K"], errors="coerce").round(4)

    merged = left.merge(
        right.drop(columns=["K"], errors="ignore"),
        on=["ticker", "event_endDate", "K_round"],
        how="left",
        validate="m:1",
    )
    merged.drop(columns=["K_round"], inplace=True, errors="ignore")
    return merged


def add_qrn_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["pRN"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out["qRN"] = 1.0 - out["pRN"]
    out["qRN"] = pd.to_numeric(out["qRN"], errors="coerce").round(7)
    return out


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "snapshot_time_utc",
        "week_monday","week_friday","week_sunday",
        "ticker","slug",
        "event_title","market_question",
        "event_id","market_id","condition_id",
        "event_endDate",
        "K","T_days",
        "pPM_buy","pPM_mid","yes_spread",
        "qPM_buy","qPM_mid","no_spread",
        "pm_ok","pm_reason",
        "yes_token_id","no_token_id",
        "S","pRN","qRN","pRN_raw",
        "rn_method","rn_quote_source","rn_monotone_adjusted",
        "n_calls_used","calls_k_min","calls_k_max","deltaK",
    ]
    extras = [c for c in df.columns if c not in cols]
    final_cols = cols + extras
    return df.reindex(columns=final_cols).copy()



SNAPSHOT_STANDARD_FEATURE_COLUMNS = [
    "expiry_date",
    "expiry_close_date_used",
    "asof_date",
    "S_asof_close",
    "S_asof_close_adj",
    "S_expiry_close",
    "S_expiry_close_adj",
    "log_m",
    "abs_log_m",
    "log_m_fwd",
    "abs_log_m_fwd",
    "log_T_days",
    "T_years",
    "sqrt_T_years",
    "rv20",
    "rv20_source",
    "rv20_window",
    "is_missing_rv20",
    "rv20_sqrtT",
    "log_m_over_volT",
    "abs_log_m_over_volT",
    "log_m_fwd_over_volT",
    "abs_log_m_fwd_over_volT",
    "prn_raw_gap",
    "x_prn_x_tdays",
    "x_prn_x_rv20",
    "x_prn_x_logm",
    "x_m",
    "x_abs_m",
    "forward_price",
    "dividend_yield",
    "r",
    "spot_scale_used",
    "rel_spread_median",
    "n_chain_raw",
    "n_chain_used",
    "n_band_raw",
    "n_band_inside",
    "dropped_intrinsic",
    "asof_fallback_days",
    "expiry_fallback_days",
    "split_events_in_preload_range",
]


def enrich_snapshot_features(
    df: pd.DataFrame,
    cfg: Config,
    manifest: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Enrich snapshot with all guaranteed baseline columns and derived features.

    This function GUARANTEES that all columns from pipeline_schema_contract.json
    baseline schema will exist (even if NaN). This ensures schema compatibility
    with any downstream calibration model.
    """
    out = df.copy()

    # Step 1: Populate baseline columns
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

    # Ensure baseline diagnostic columns exist
    baseline_defaults = {
        "rv20": cfg.rv20_fallback,  # Fallback value when rv20 is truly missing
        "rv20_source": "fallback",
        "rv20_window": 0,
        "is_missing_rv20": True,
        "dividend_yield": 0.0,  # Default to 0 (no dividends) for forward price computation
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

    # Step 2: Compute derived features
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

    # Ensure pRN_raw exists for gap calculation
    if "pRN_raw" not in out.columns or pd.isna(out["pRN_raw"]).all():
        out["pRN_raw"] = out["pRN"]
    out["prn_raw_gap"] = prn - pd.to_numeric(out["pRN_raw"], errors="coerce")

    # Interaction features
    out["x_prn_x_tdays"] = out["x_logit_prn"] * out["T_days"]
    out["x_prn_x_rv20"] = out["x_logit_prn"] * pd.to_numeric(out.get("rv20", np.nan), errors="coerce")
    out["x_prn_x_logm"] = out["x_logit_prn"] * out["log_m"]

    # Volatility-scaled features
    rv20 = pd.to_numeric(out.get("rv20", np.nan), errors="coerce")
    vol_denom = rv20 * out["sqrt_T_years"]
    vol_denom = vol_denom.replace(0, np.nan)
    out["rv20_sqrtT"] = rv20 * out["sqrt_T_years"]
    out["log_m_over_volT"] = out["log_m"] / vol_denom
    out["abs_log_m_over_volT"] = out["log_m"].abs() / vol_denom

    # Forward price features
    # Ensure dividend_yield has a fallback (0.0 = no dividends)
    if "dividend_yield" not in out.columns:
        out["dividend_yield"] = 0.0
    out["dividend_yield"] = pd.to_numeric(out["dividend_yield"], errors="coerce").fillna(0.0)

    if "forward_price" not in out.columns:
        out["forward_price"] = np.nan
    F = pd.to_numeric(out.get("forward_price", np.nan), errors="coerce")

    # Compute forward_price for any rows where it's missing but S is available
    S_vals = pd.to_numeric(out.get("S", np.nan), errors="coerce")
    r_vals = pd.to_numeric(out.get("r", cfg.risk_free_rate), errors="coerce").fillna(cfg.risk_free_rate)
    q_vals = out["dividend_yield"]  # Already filled with 0.0 fallback

    # Compute forward price: F = S * exp((r - q) * T)
    forward_computed = S_vals * np.exp((r_vals - q_vals) * out["T_years"])
    forward_computed = forward_computed.where(np.isfinite(forward_computed) & (forward_computed > 0), np.nan)

    # Fill in missing forward_price values with computed values
    F = F.where(F.notna() & np.isfinite(F), forward_computed)
    out["forward_price"] = F
    out["log_m_fwd"] = np.log(np.clip(K, 1e-12, None) / np.clip(F, 1e-12, None))
    out["abs_log_m_fwd"] = out["log_m_fwd"].abs()
    out["log_m_fwd_over_volT"] = out["log_m_fwd"] / vol_denom
    out["abs_log_m_fwd_over_volT"] = out["abs_log_m_fwd"] / vol_denom

    moneyness = out["log_m_fwd"].where(np.isfinite(out["log_m_fwd"]), out["log_m"])
    out["x_m"] = out["x_logit_prn"] * moneyness
    out["x_abs_m"] = out["x_logit_prn"] * moneyness.abs()

    # Step 3: Add any columns from model manifest (for forward compatibility)
    if manifest is not None:
        for col in manifest.get("required_columns", []):
            if col not in out.columns:
                out[col] = np.nan

    return out

# -----------------------------
# Naming helpers (updated convention)
# Following format: polymarket-snapshot-YYYY-MM-DD-HH.csv
# where HH is 24-hour zero-padded hour
# -----------------------------

def fname_pm_snapshot_new_convention() -> str:
    """Generate snapshot filename: polymarket-snapshot-YYYY-MM-DD-HH.csv"""
    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")
    hour_str = now_utc.strftime("%H")
    return f"polymarket-snapshot-{date_str}-{hour_str}.csv"


def fname_dataset_history_by_contract(contract_type: str) -> str:
    """Historic dataset filename per contract type."""
    ct = (contract_type or "weekly").strip().lower()
    suffix = "weekly" if ct == "weekly" else "1dte"
    return f"polymarket-snapshot-history-{suffix}.csv"


# =============================================================================
# NaN Diagnostics
# =============================================================================

def diagnose_nans(df: pd.DataFrame, label: str = "Dataset") -> None:
    """
    Analyze and print per-column NaN rates and top reasons.

    Column sources (yfinance-backed):
    - snapshot_time_utc: Timestamp of snapshot run
    - ticker, K, T_days: Polymarket extraction (always present)
    - S, pRN, pRN_raw: From yfinance option chain (may be NaN if no chain)
    - rv20: Computed from yfinance historical close (may be NaN if insufficient data)
    - dividend_yield: From yfinance info or dividends (may be NaN if not available)
    - forward_price: Computed from S, r, dividend_yield (NaN if S or div_yield NaN)
    - Spread columns: From Polymarket CLOB prices (may be NaN if no CLOB data)
    - pPM_*, qPM_*: Polymarket bid/ask (may be NaN if CLOB unavailable)
    """
    if df is None or df.empty:
        print(f"[NaN Diagnostics] {label}: DataFrame is empty")
        return

    print(f"\n[NaN Diagnostics] {label} ({len(df)} rows)")
    print("=" * 80)

    nan_summary = []
    for col in df.columns:
        n_nan = int(df[col].isna().sum())
        rate = 100.0 * n_nan / len(df) if len(df) > 0 else 0.0
        nan_summary.append({
            "column": col,
            "n_nan": n_nan,
            "nan_rate_%": rate,
            "dtype": str(df[col].dtype),
        })

    # Sort by NaN rate descending
    nan_summary.sort(key=lambda x: x["nan_rate_%"], reverse=True)

    # Print summary table
    print(f"{'Column':<35} {'N_NaN':>8} {'Rate%':>8} {'Dtype':<15}")
    print("-" * 80)
    for row in nan_summary:
        if row["nan_rate_%"] > 0:  # Only show columns with NaNs
            print(f"{row['column']:<35} {row['n_nan']:>8} {row['nan_rate_%']:>7.1f}% {row['dtype']:<15}")

    # Diagnose top reasons for NaNs
    print("\n[NaN Root Causes]")
    print("-" * 80)

    nan_reasons = {
        "S": "No option chain available (yfinance has no expiry)",
        "pRN": "No option chain or insufficient calls (need ≥3 strikes)",
        "pRN_raw": "Call slope method failed or insufficient data",
        # rv20 no longer has NaN - uses fallback chain (realized->scaled->implied->sentinel)
        # dividend_yield defaults to 0.0 (no dividends) when not available
        # forward_price is computed when S is available (uses dividend_yield=0.0 if missing)
        "pPM_buy": "No CLOB BUY price available for yes token",
        "qPM_buy": "No CLOB BUY price available for no token",
        "pPM_mid": "Both bid and ask needed to compute midpoint; CLOB not available",
        "qPM_mid": "Both bid and ask needed to compute midpoint; CLOB not available",
        "yes_spread": "Bid/ask missing from CLOB prices",
        "no_spread": "Bid/ask missing from CLOB prices",
    }

    for col, reason in nan_reasons.items():
        if col in df.columns:
            rate = 100.0 * df[col].isna().sum() / len(df) if len(df) > 0 else 0.0
            if rate > 0:
                print(f"  {col:<30} {rate:>6.1f}% NaN  ← {reason}")

    print("\n[Column Sources (yfinance or derived)]")
    print("-" * 80)
    sources = {
        "Polymarket Metadata": ["snapshot_time_utc", "ticker", "slug", "event_id", "event_title",
                               "market_id", "condition_id", "market_question", "K", "week_monday",
                               "week_friday", "event_endDate", "yes_token_id", "no_token_id"],
        "yfinance Option Chain": ["S (spot price)", "pRN (risk-neutral prob)", "pRN_raw"],
        "yfinance Historical": [
            "rv20 (20-day realized vol - guaranteed non-NaN via fallback chain)",
            "rv20_source (realized_20d, realized_Nd_scaled, implied_atm, fallback)",
            "rv20_window (N days used, 0 for implied/fallback)",
            "is_missing_rv20 (True if fallback used)",
            "dividend_yield (from info or dividends, defaults to 0.0)"
        ],
        "Computed from yfinance": [
            "forward_price (S * exp((r-q)*T) - guaranteed when S available, uses q=0 if missing)",
            "T_days (expiry_ts_utc - snapshot_time)"
        ],
        "Polymarket CLOB Prices": ["pPM_buy (yes ask)", "qPM_buy (no ask)", "pPM_mid", "qPM_mid", "yes_spread", "no_spread"],
        "Features/Derived": ["All log_*, sqrt_*, ratio columns derived from above"],
    }
    for category, cols_list in sources.items():
        print(f"  {category}")
        for col_desc in cols_list:
            print(f"    - {col_desc}")


def print_timing_summary(timings: Dict[str, float]) -> None:
    """Print execution timing summary."""
    print("\n[Execution Timing]")
    print("=" * 80)
    print(f"{'Step':<40} {'Duration (s)':>15} {'%':>8}")
    print("-" * 80)
    total = sum(timings.values())
    for step, dur in timings.items():
        pct = 100.0 * dur / total if total > 0 else 0.0
        print(f"{step:<40} {dur:>15.2f} {pct:>7.1f}%")
    print("-" * 80)
    print(f"{'TOTAL':<40} {total:>15.2f} {'':>8}")
    print("=" * 80)


def validate_rows(
    df: pd.DataFrame,
    *,
    snapshot_time_utc: datetime,
    contract_type: str,
    contract_1dte: str,
    target_date: date,
    tz_name: str,
    calendar_kind: Optional[str],
    calendar: Any,
) -> None:
    if df is None or df.empty:
        print("[Validate] DataFrame is empty; skipping checks.")
        return

    errors: List[str] = []

    if {"spot_source", "spot_asof_utc", "snapshot_time_utc"}.issubset(df.columns):
        mask = df["spot_source"].fillna("none") != "none"
        if mask.any():
            spot_asof = pd.to_datetime(df.loc[mask, "spot_asof_utc"], utc=True, errors="coerce")
            snap = pd.to_datetime(df.loc[mask, "snapshot_time_utc"], utc=True, errors="coerce")
            if spot_asof.isna().any() or snap.isna().any() or not (spot_asof == snap).all():
                errors.append("spot_asof_utc does not match snapshot_time_utc for spot-bearing rows.")

    if "T_days" in df.columns:
        t_days = pd.to_numeric(df["T_days"], errors="coerce")
        if (t_days < 0).any():
            errors.append("T_days contains negative values.")

    if "expiry_ts_utc" in df.columns:
        expiry = pd.to_datetime(df["expiry_ts_utc"], utc=True, errors="coerce")
        if expiry.isna().any():
            errors.append("expiry_ts_utc is missing or unparsable in some rows.")

    if errors:
        raise ValueError("[Validate] " + " | ".join(errors))

    if contract_type == "1dte":
        snapshot_local_date = pd.to_datetime(snapshot_time_utc, utc=True, errors="coerce")
        if pd.isna(snapshot_local_date):
            return
        snapshot_local_date = snapshot_local_date.tz_convert(ZoneInfo(tz_name)).date()
        delta = 0 if contract_1dte == "close_today" else 1
        expected = snapshot_local_date + timedelta(days=delta)
        if expected is not None and target_date != expected:
            print(
                f"[Validate] WARNING: 1DTE target_date={target_date.isoformat()} "
                f"does not match the expected {contract_1dte.replace('_', ' ')} date "
                f"({expected.isoformat()}) for snapshot local date {snapshot_local_date.isoformat()}."
            )


def enforce_prn_complete(
    df: pd.DataFrame,
    *,
    asof_utc: datetime,
    contract_type: str,
    contract_1dte: str,
    calendar_kind: Optional[str],
    calendar: Any,
) -> None:
    if df is None or df.empty:
        raise RuntimeError("Snapshot contains no rows; pRN guard triggered.")
    if "pRN" not in df.columns:
        raise RuntimeError("Snapshot missing pRN column; pRN guard triggered.")
    missing_mask = pd.to_numeric(df["pRN"], errors="coerce").isna()
    if not missing_mask.any():
        return

    total = len(df)
    missing = int(missing_mask.sum())
    session = classify_session(asof_utc, "America/New_York", calendar_kind, calendar)

    lines = [
        f"[pRN Guard] Missing pRN for {missing}/{total} rows. Snapshot will not be written.",
    ]
    if session == "PRE":
        lines.append(
            "Snapshot time is before US market open. yfinance often does not publish same-day option chains premarket."
        )
        if contract_type == "1dte" and contract_1dte == "close_today":
            lines.append(
                "For 1DTE close_today, retry after 09:30 America/New_York or switch to close_tomorrow."
            )
    else:
        lines.append(
            "If this happens during regular hours, yfinance may be rate-limited or missing the expiry; retry shortly."
        )

    raise RuntimeError("\n".join(lines))

# -----------------------------
# Main
# -----------------------------

def main() -> None:
    """
    Main pipeline with timing instrumentation and NaN diagnostics.

    Process:
    1. Fetch Polymarket snapshot (markets ending on target_date)
    2. Compute pRN from yfinance option chains
    3. Merge and enrich with derived features
    4. Write snapshot with new naming convention: polymarket-snapshot-YYYY-MM-DD-HH.csv
    5. Append to historic dataset (per-contract history CSV)
    6. Print NaN diagnostics and timing summary
    """
    parser = argparse.ArgumentParser(description="Polymarket snapshot + pRN with NaN diagnostics (v1.5.1).")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--tickers-csv", type=str, default=None, help="CSV with a 'ticker' column")
    parser.add_argument("--slug-overrides", type=str, default=None, help="Optional .json/.csv mapping ticker->slug")
    parser.add_argument("--feature-manifest", type=str, default=None, help="Optional feature_manifest.json produced by the calibrator to ensure required columns are present.")
    parser.add_argument("--contract-type", type=str, default="weekly", choices=["weekly", "1dte"], help="Contract type to fetch (weekly or 1dte).")
    parser.add_argument("--contract-1dte", type=str, default="close_tomorrow", choices=["close_today", "close_tomorrow"], help="For 1dte: contract expiring at today's or tomorrow's market close.")
    parser.add_argument("--target-date", type=str, default=None, help="Override target date (YYYY-MM-DD) for contract resolution.")
    parser.add_argument("--exchange-calendar", type=str, default=DEFAULT_EXCHANGE_CALENDAR, help="Exchange calendar name (default: XNYS).")

    parser.add_argument("--risk-free-rate", type=float, default=Config().risk_free_rate)
    parser.add_argument("--tz", type=str, default=Config().tz_name)
    parser.add_argument("--keep-nonexec", action="store_true", help="Keep pm_ok=False rows (default keeps them anyway)")
    parser.add_argument("--allow-nonlive", action="store_true", help="Allow non-live snapshots to compute pRN from yfinance option chains.")
    parser.add_argument("--dry-run", action="store_true", help="Run fetch + compute + validation without writing any files.")
    args = parser.parse_args()

    cfg = Config(
        tz_name=args.tz,
        risk_free_rate=float(args.risk_free_rate),
    )
    feature_manifest: Optional[Dict[str, Any]] = None
    if args.feature_manifest:
        manifest_path = Path(args.feature_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Feature manifest not found: {manifest_path}")
        with manifest_path.open() as fh:
            feature_manifest = json.load(fh)
        if not isinstance(feature_manifest, dict):
            raise ValueError("Feature manifest must be a JSON object.")
        print(f"[Snapshot] Loaded feature manifest with {len(feature_manifest.get('required_columns', []))} columns")

    # Timing instrumentation
    timings: Dict[str, float] = {}
    overall_start = time.time()

    snapshot_time_utc = datetime.now(timezone.utc)
    asof_utc = snapshot_time_utc

    calendar_kind, calendar = load_exchange_calendar(args.exchange_calendar)
    if calendar is None:
        print(f"[Calendar] '{args.exchange_calendar}' unavailable; using fixed 09:30–16:00 local fallback.")

    tz = ZoneInfo(cfg.tz_name)
    snapshot_time_local_dt = snapshot_time_utc.astimezone(tz)
    snapshot_time_local = snapshot_time_local_dt.isoformat()
    snapshot_session = classify_session(asof_utc, cfg.tz_name, calendar_kind, calendar)

    runs_dir = os.path.join(args.out_dir, "runs", args.contract_type)
    history_dir = os.path.join(args.out_dir, "history")
    if not args.dry_run:
        ensure_dir(args.out_dir)
        ensure_dir(runs_dir)
        ensure_dir(history_dir)

    today_local = snapshot_time_local_dt.date()
    if today_local.weekday() >= 5:
        print(
            "[Weekend Warning] This snapshot was taken during the weekend. "
            "No Polymarket contract will be fetched unless it explicitly matches the requested expiration."
        )
    if args.target_date:
        try:
            target_date = date.fromisoformat(args.target_date)
        except ValueError as exc:
            raise ValueError(f"--target-date must be YYYY-MM-DD; got {args.target_date}") from exc
    else:
        target_date = resolve_target_date(today_local, args.contract_type, args.contract_1dte, calendar_kind, calendar)

    week_monday, week_friday, week_sunday = trading_week_bounds_for_date(target_date)

    if args.contract_type == "weekly" and target_date.weekday() != 4:
        print(f"[Contract] WARNING: weekly target_date={target_date.isoformat()} is not a Friday.")
    if args.contract_type == "1dte":
        enforce_1dte_contract_exists(
            asof_utc=asof_utc,
            target_date=target_date,
            tz_name=cfg.tz_name,
            calendar_kind=calendar_kind,
            calendar=calendar,
        )

    default_tickers = allowed_tickers_for_contract(args.contract_type)
    tickers = parse_tickers_arg(args.tickers_csv, args.tickers, default_tickers=default_tickers)
    validate_ticker_universe(tickers, args.contract_type)
    slug_overrides = load_slug_overrides(args.slug_overrides)

    session = make_session(cfg)

    missing = ensure_events_exist(
        tickers=tickers,
        target_date=target_date,
        contract_type=args.contract_type,
        cfg=cfg,
        slug_overrides=slug_overrides,
        session=session,
    )
    if missing is not None:
        ticker_missing, slug_missing, reason_missing = missing
        reason_text = "not found" if reason_missing == "missing" else "ends on the wrong day"
        if args.contract_type == "1dte":
            raise RuntimeError(
                "The requested 1DTE contract does not exist yet.\n"
                "1DTE contracts are created at market close the day before expiration."
            )
        print(
            f"[Polymarket] contracts for target date {target_date.isoformat()} "
            f"(ticker={ticker_missing}, slug={slug_missing}) haven't been created yet ({reason_text}). Run snapshot later."
        )
        return

    run_id = unique_run_id(runs_dir, args.contract_type) if not args.dry_run else "dry_run"
    run_dir = os.path.join(runs_dir, run_id) if not args.dry_run else None
    if run_dir:
        ensure_dir(run_dir)

    print(f"[Week] {week_monday.isoformat()} → {week_sunday.isoformat()} (week Fri {week_friday.isoformat()}) tz={cfg.tz_name}")
    if args.contract_type == "1dte":
        print(
            f"[Target] contract_type={args.contract_type} mode={args.contract_1dte} "
            f"target_date={target_date.isoformat()} calendar={args.exchange_calendar}"
        )
    else:
        print(f"[Target] contract_type={args.contract_type} target_date={target_date.isoformat()} calendar={args.exchange_calendar}")
    print(f"[Snapshot] asof_utc={snapshot_time_utc.isoformat()} session={snapshot_session}")
    print(f"[Tickers] n={len(tickers)}  {', '.join(tickers)}")
    print(f"[Run] run_id={run_id}")
    print(f"[Config] r={cfg.risk_free_rate:.4f}")
    print(f"[Script Version] {SCRIPT_VER} (schema={SCHEMA_VERSION})")

    # =========================================================================
    # 1) Fetch Polymarket snapshot
    # =========================================================================
    t0 = time.time()
    pm = fetch_polymarket_snapshot(
        tickers=tickers,
        week_monday=week_monday,
        week_friday=week_friday,
        week_sunday=week_sunday,
        target_date=target_date,
        contract_type=args.contract_type,
        snapshot_time_utc=snapshot_time_utc,
        snapshot_time_local=snapshot_time_local,
        snapshot_session=snapshot_session,
        cfg=cfg,
        slug_overrides=slug_overrides,
        session=session,
    )
    timings["Fetch Polymarket"] = time.time() - t0
    diagnose_nans(pm, "After Polymarket Fetch")

    # =========================================================================
    # 2) Compute pRN from yfinance
    # =========================================================================
    t0 = time.time()
    rn = compute_pRN_snapshot(
        pm,
        cfg,
        asof_utc=asof_utc,
        tz_name=cfg.tz_name,
        allow_nonlive=args.allow_nonlive,
    )
    timings["Compute pRN (yfinance)"] = time.time() - t0
    diagnose_nans(rn, "After pRN Computation")

    # =========================================================================
    # 3) Merge Polymarket + pRN, add derived features
    # =========================================================================
    t0 = time.time()
    merged = merge_pm_rn(pm, rn)
    merged = add_qrn_column(merged)
    enforce_prn_complete(
        merged,
        asof_utc=asof_utc,
        contract_type=args.contract_type,
        contract_1dte=args.contract_1dte,
        calendar_kind=calendar_kind,
        calendar=calendar,
    )
    final_df = select_final_columns(merged)

    sort_cols = [c for c in ["ticker", "event_endDate", "K"] if c in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    final_df = enrich_snapshot_features(final_df, cfg, feature_manifest)
    timings["Merge & Enrich Features"] = time.time() - t0
    diagnose_nans(final_df, "After Feature Enrichment")

    validate_rows(
        final_df,
        snapshot_time_utc=snapshot_time_utc,
        contract_type=args.contract_type,
        contract_1dte=args.contract_1dte,
        target_date=target_date,
        tz_name=cfg.tz_name,
        calendar_kind=calendar_kind,
        calendar=calendar,
    )

    if args.dry_run:
        timings["TOTAL"] = time.time() - overall_start
        print_timing_summary(timings)
        print("\n[Dry Run] Validation passed. No files were written.")
        return

    # =========================================================================
    # 4) Write 3 CSVs to runs directory
    # =========================================================================
    t0 = time.time()
    pm_path = os.path.join(run_dir, "polymarket.csv")
    rn_path = os.path.join(run_dir, "rn.csv")
    final_path = os.path.join(run_dir, "final.csv")

    pm.to_csv(pm_path, index=False)
    rn.to_csv(rn_path, index=False)
    final_df.to_csv(final_path, index=False)

    print(f"\n[Write Run CSVs] {run_dir}")
    print(f"  - polymarket.csv (rows={len(pm)})")
    print(f"  - rn.csv (rows={len(rn)})")
    print(f"  - final.csv (rows={len(final_df)})")
    timings["Write Run CSVs"] = time.time() - t0

    # =========================================================================
    # 5) Append to historic dataset
    # =========================================================================
    t0 = time.time()
    hist_filename = fname_dataset_history_by_contract(args.contract_type)
    hist_path = os.path.join(history_dir, hist_filename)

    hist = final_df.copy()
    hist["run_id"] = run_id
    hist["run_time_utc"] = datetime.now(timezone.utc).isoformat()
    hist["run_contract_type"] = args.contract_type

    append_df_to_csv_with_schema(hist, hist_path)
    print(f"[Append Historic] {hist_path} (appended {len(hist)} rows)")
    timings["Append Historic"] = time.time() - t0

    # =========================================================================
    # 6) Final diagnostics and timing summary
    # =========================================================================
    timings["TOTAL"] = time.time() - overall_start
    print_timing_summary(timings)

    print("\n[Summary]")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"  - polymarket.csv, rn.csv, final.csv")
    print(f"Historic appended: {hist_path}")
    print("\nColumn sources:")
    print("  Polymarket: ticker, K, event_id, market_id, yes_token_id, no_token_id")
    print("  yfinance option chain: S (spot), pRN (risk-neutral prob), pRN_raw")
    print("  yfinance history: rv20 (guaranteed non-NaN via fallback chain), rv20_source, rv20_window, is_missing_rv20")
    print("  yfinance info: dividend_yield (annualized)")
    print("  Derived: forward_price, T_days, all log_* and ratio features")
    print("  CLOB prices: pPM_buy, pPM_mid, qPM_buy, qPM_mid (may be NaN if unavailable)")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {exc}")
        sys.exit(1)
