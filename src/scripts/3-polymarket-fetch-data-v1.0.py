#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from scipy.stats import norm
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo


# -----------------------------
# Defaults + Endpoints
# -----------------------------

DEFAULT_TICKERS = ["NVDA", "AAPL", "GOOGL", "MSFT", "META", "AMZN", "TSLA", "PLTR", "OPEN", "NFLX"]
GAMMA_PUBLIC_SEARCH = "https://gamma-api.polymarket.com/public-search"
GAMMA_EVENT_BY_SLUG = "https://gamma-api.polymarket.com/events/slug/{}"
CLOB_PRICES = "https://clob.polymarket.com/prices"  # POST [{"token_id": "...", "side": "BUY"}]

SCRIPT_VER = "1.4.0"


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


# -----------------------------
# Small utilities
# -----------------------------

def _local_date(dt_like, tz_name: str) -> Optional[date]:
    dt = pd.to_datetime(dt_like, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.tz_convert(ZoneInfo(tz_name)).date()

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

def parse_tickers_arg(tickers_csv: Optional[str], tickers_list: Optional[str]) -> List[str]:
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

    return DEFAULT_TICKERS.copy()


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
    *,
    cfg: Config,
    slug_overrides: Dict[str, str],
    session: requests.Session,
) -> pd.DataFrame:
    """
    Fetch Polymarket "finish week" markets only (end on week_friday in cfg.tz_name),
    for tickers' weekly 'above-on-<month>-<day>-<year>' contracts.

    Behavior:
    - Try deterministic slug (override or build_weekly_slug).
    - If fetched event does NOT end on the expected Friday (local date), attempt discovery
      via Gamma public-search to find the correct finish-week slug and refetch.
    - Filter per-market by market endDate (fallback event endDate) to enforce Friday-ending.
    - Fetch CLOB BUY/SELL prices in bulk for tokens (no Gamma fallback here, by design).
    """
    GAMMA_PUBLIC_SEARCH = "https://gamma-api.polymarket.com/public-search"

    snapshot_time_utc = datetime.now(timezone.utc)

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

    def _slug_suffix_for_friday(friday: date) -> str:
        month = friday.strftime("%B").lower()
        return f"-above-on-{month}-{friday.day}-{friday.year}"

    def _looks_like_finish_week(title: str) -> bool:
        s = (title or "").lower()
        return ("finish week" in s) or ("finish the week" in s) or ("finish week of" in s)

    def discover_finishweek_slug(ticker: str) -> Optional[str]:
        """
        Search Gamma for the correct finish-week event slug.
        Hard constraints:
          - slug contains the Friday suffix "-above-on-<month>-<day>-<year>"
          - event endDate (local) == week_friday (when available)
        Soft preferences:
          - title suggests 'finish week'
          - higher Gamma 'score'
        """
        want_suffix = _slug_suffix_for_friday(week_friday)

        queries = [
            f"{ticker} finish week of {week_monday.strftime('%B')} {week_monday.day} {week_monday.year}",
            f"{ticker} above on {week_friday.strftime('%B')} {week_friday.day} {week_friday.year}",
        ]

        candidates = []
        for q in queries:
            try:
                r = session.get(
                    GAMMA_PUBLIC_SEARCH,
                    params={
                        "q": q,
                        "limit_per_type": 50,
                        "events_status": "active",
                        "keep_closed_markets": 0,
                    },
                    timeout=cfg.request_timeout_s,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                errors.append({"ticker": ticker, "stage": "public_search", "q": q, "error": str(e)})
                continue

            events = data.get("events") or []
            if not isinstance(events, list):
                continue

            for ev in events:
                if not isinstance(ev, dict):
                    continue
                slug = str(ev.get("slug") or "").strip()
                if not slug or (want_suffix not in slug):
                    continue

                title = ev.get("title") or ev.get("question") or ""
                end_dt_utc = _parse_dt_utc(ev.get("endDate") or ev.get("endDateIso") or ev.get("end_time"))
                end_local = _local_date_from_dt(end_dt_utc)

                # If end date exists and mismatches, reject
                if end_local is not None and end_local != week_friday:
                    continue

                score = ev.get("score")
                try:
                    score_f = float(score) if score is not None else 0.0
                except Exception:
                    score_f = 0.0

                pref = 1 if _looks_like_finish_week(str(title)) else 0
                candidates.append((pref, score_f, slug))

            if candidates:
                break  # stop after first query that yields candidates

        if not candidates:
            return None

        # Prefer finish-week title match, then highest score
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    def fetch_event_by_slug(slug: str) -> Optional[dict]:
        resp = session.get(GAMMA_EVENT_BY_SLUG.format(slug), timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        ev = resp.json()
        return ev if isinstance(ev, dict) else None

    def event_end_local_matches_friday(ev: dict) -> Optional[bool]:
        end_raw = ev.get("endDate") or ev.get("endDateIso") or ev.get("end_time")
        end_dt_utc = _parse_dt_utc(end_raw)
        end_local = _local_date_from_dt(end_dt_utc)
        if end_local is None:
            return None
        return end_local == week_friday

    for t in tickers:
        slug_initial = slug_overrides.get(t) or build_weekly_slug(t, week_friday)
        slug = slug_initial

        try:
            ev = fetch_event_by_slug(slug)

            # Hard check: if the event clearly ends NOT on Friday, attempt discovery & refetch
            match = event_end_local_matches_friday(ev)
            if match is False:
                discovered = discover_finishweek_slug(t)
                if discovered and discovered != slug:
                    slug = discovered
                    ev = fetch_event_by_slug(slug)
                    match2 = event_end_local_matches_friday(ev)
                    if match2 is False:
                        # Still wrong -> skip (better than polluting with daily event)
                        errors.append({
                            "ticker": t,
                            "slug": slug,
                            "stage": "refetch_mismatch",
                            "error": f"event_end_local != week_friday ({week_friday.isoformat()})",
                        })
                        continue
                else:
                    errors.append({
                        "ticker": t,
                        "slug": slug_initial,
                        "stage": "slug_mismatch_no_discovery",
                        "error": f"event_end_local != week_friday ({week_friday.isoformat()}) and no better slug found",
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
                if m_end_local is not None and m_end_local != week_friday:
                    continue

                question = m.get("question") or m.get("title")
                K = extract_strike_K_from_question(question)

                token_ids = normalize_list_field(m.get("clobTokenIds"))
                yes_token_id = token_ids[0] if isinstance(token_ids, list) and len(token_ids) >= 2 else None
                no_token_id = token_ids[1] if isinstance(token_ids, list) and len(token_ids) >= 2 else None

                rows.append(
                    {
                        "snapshot_time_utc": snapshot_time_utc,
                        "week_monday": week_monday.isoformat(),
                        "week_friday": week_friday.isoformat(),
                        "week_sunday": week_sunday.isoformat(),
                        "ticker": t,
                        "slug": slug,
                        "event_id": event_id,
                        "event_title": event_title,
                        "event_updatedAt": updated_at,
                        "event_endDate": m_end_raw,  # store market end if present, else event end
                        "market_id": m.get("id"),
                        "condition_id": m.get("conditionId") or m.get("condition_id"),
                        "market_question": question,
                        "K": K,
                        "yes_token_id": str(yes_token_id) if yes_token_id else None,
                        "no_token_id": str(no_token_id) if no_token_id else None,
                        "pm_ok_meta": True,
                        "pm_reason_meta": reason,
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
            "snapshot_time_utc","week_monday","week_friday","week_sunday","ticker","slug",
            "event_id","event_title","event_updatedAt","event_endDate","market_id","condition_id",
            "market_question","K","yes_token_id","no_token_id",
            "pPM_buy","qPM_buy","pPM_mid","qPM_mid","yes_spread","no_spread",
            "pm_ok","pm_reason","pm_ok_meta","pm_reason_meta","T_days"
        ])

    pm["snapshot_time_utc"] = pd.to_datetime(pm["snapshot_time_utc"], utc=True, errors="coerce")
    pm["event_endDate"] = pd.to_datetime(pm["event_endDate"], utc=True, errors="coerce")
    pm["K"] = pd.to_numeric(pm["K"], errors="coerce")

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

    # IMPORTANT: do NOT normalize event_endDate; keep actual UTC timestamp
    pm["T_days"] = (pm["event_endDate"] - pm["snapshot_time_utc"]).dt.total_seconds() / (60*60*24)

    pm.drop(columns=["yes_buy","yes_sell","no_buy","no_sell"], inplace=True, errors="ignore")

    n_total = len(pm)
    n_ok = int(pm["pm_ok"].sum())
    print(f"[Polymarket] rows={n_total} ok_any={n_ok}")
    if errors:
        print(f"[Polymarket] errors={len(errors)} (showing up to 20)")
        print(pd.DataFrame(errors).head(20))

    return pm


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


def compute_pRN_snapshot(pm: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if pm is None or pm.empty:
        return pd.DataFrame(columns=[
            "ticker","event_endDate","K","S","pRN_raw","pRN","rn_method","rn_quote_source",
            "n_calls_used","calls_k_min","calls_k_max","deltaK","rn_monotone_adjusted"
        ])

    df = pm.copy()
    df["event_endDate"] = pd.to_datetime(df["event_endDate"], utc=True, errors="coerce").dt.normalize()
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    df["T_days"] = pd.to_numeric(df["T_days"], errors="coerce")
    df = df[df["event_endDate"].notna() & df["K"].notna() & (df["K"] > 0) & df["T_days"].notna() & (df["T_days"] > 0)].copy()

    rows: List[dict] = []

    grouped = df.groupby(["ticker", "event_endDate"], dropna=True)
    print(f"[yfinance] groups={len(grouped)} (ticker, expiry)")

    for (ticker, expiry_dt), g in grouped:
        try:
            stock = yf.Ticker(str(ticker))
            expiry_str = pd.Timestamp(expiry_dt).strftime("%Y-%m-%d")

            opts = getattr(stock, "options", [])
            if expiry_str not in opts:
                # still return rows, but missing pRN
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_dt,
                        "K": round7(float(r.K)),
                        "S": np.nan,
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "no_chain",
                        "rn_quote_source": None,
                        "n_calls_used": 0,
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                    })
                continue

            chain = stock.option_chain(expiry_str)
            calls_raw = chain.calls.copy()
            if calls_raw.empty:
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_dt,
                        "K": round7(float(r.K)),
                        "S": np.nan,
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "empty_calls",
                        "rn_quote_source": None,
                        "n_calls_used": 0,
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                    })
                continue

            hist = stock.history(period="1d")
            if hist.empty or "Close" not in hist.columns:
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_dt,
                        "K": round7(float(r.K)),
                        "S": np.nan,
                        "pRN_raw": np.nan,
                        "pRN": np.nan,
                        "rn_method": "no_spot",
                        "rn_quote_source": None,
                        "n_calls_used": 0,
                        "calls_k_min": np.nan,
                        "calls_k_max": np.nan,
                        "deltaK": np.nan,
                        "rn_monotone_adjusted": False,
                    })
                continue

            S = float(hist["Close"].iloc[-1])

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

            if len(calls_use) < 3:
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_dt,
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
            calls_use = calls_use.loc[~too_far_below].copy()

            if len(calls_use) < 3:
                for r in g.itertuples(index=False):
                    rows.append({
                        "ticker": ticker,
                        "event_endDate": expiry_dt,
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
                    })
                continue

            k_arr = calls_use["strike"].to_numpy(dtype=float)
            c_arr = calls_use["mid"].to_numpy(dtype=float)
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

                rows.append({
                    "ticker": ticker,
                    "event_endDate": expiry_dt,
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
                })

        except Exception as e:
            print(f"[yfinance] ❌ {ticker} {expiry_dt}: {e}")

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


def add_edge_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["pPM_buy","qPM_buy","pRN"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["pPM_exec"] = out["pPM_buy"]
    out["qPM_exec"] = out["qPM_buy"]
    out["qRN"] = 1.0 - out["pRN"]

    out["edgeYES_rn"] = out["pRN"] - out["pPM_exec"]
    out["edgeNO_rn"] = (1.0 - out["pRN"]) - out["qPM_exec"]
    out["delta_P_rn"] = out[["edgeYES_rn", "edgeNO_rn"]].max(axis=1, skipna=True)

    for c in ["qRN","edgeYES_rn","edgeNO_rn","delta_P_rn"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(7)

    return out


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "snapshot_time_utc",
        "week_monday","week_friday","week_sunday",
        "ticker","slug",
        "event_endDate",
        "event_id","market_id","condition_id",
        "yes_token_id","no_token_id",
        "event_title","market_question",
        "K","T_days",
        "pPM_buy","pPM_mid","yes_spread",
        "qPM_buy","qPM_mid","no_spread",
        "pm_ok","pm_reason",
        "S","pRN","qRN","pRN_raw","rn_method","rn_quote_source","rn_monotone_adjusted",
        "n_calls_used","calls_k_min","calls_k_max","deltaK",
        "edgeYES_rn","edgeNO_rn","delta_P_rn",
    ]
    cols_present = [c for c in cols if c in df.columns]
    return df[cols_present].copy()


# -----------------------------
# Naming helpers (your simplified convention)
# -----------------------------

def fname_pm_snapshot(day: str) -> str:
    return f"01__pPM__snapshot__{day}__v{SCRIPT_VER}.csv"


def fname_rn_snapshot(day: str, r: float) -> str:
    return f"01__pRN__snapshot__{day}__v{SCRIPT_VER}.csv"


def fname_dataset_snapshot(day: str, r: float) -> str:
    return f"01__pPM_dataset__snapshot__{day}__v{SCRIPT_VER}.csv"


def fname_dataset_history(r: float) -> str:
    return f"01__pPM_dataset__history__rolling__v{SCRIPT_VER}.csv"


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly Polymarket snapshot + pRN (trimmed v1.4.0).")
    parser.add_argument("--out-dir", type=str, default="./data")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated tickers")
    parser.add_argument("--tickers-csv", type=str, default=None, help="CSV with a 'ticker' column")
    parser.add_argument("--slug-overrides", type=str, default=None, help="Optional .json/.csv mapping ticker->slug")

    parser.add_argument("--risk-free-rate", type=float, default=Config().risk_free_rate)
    parser.add_argument("--tz", type=str, default=Config().tz_name)
    parser.add_argument("--keep-nonexec", action="store_true", help="Keep pm_ok=False rows (default keeps them anyway)")
    args = parser.parse_args()

    cfg = Config(
        tz_name=args.tz,
        risk_free_rate=float(args.risk_free_rate),
    )

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "runs"))
    ensure_dir(os.path.join(args.out_dir, "history"))

    tz = ZoneInfo(cfg.tz_name)
    today_local = datetime.now(tz).date()
    week_monday, week_friday, week_sunday = trading_week_bounds(today_local)

    assert week_friday.weekday() == 4, f"week_friday must be Friday, got {week_friday} (weekday={week_friday.weekday()})"

    tickers = parse_tickers_arg(args.tickers_csv, args.tickers)
    slug_overrides = load_slug_overrides(args.slug_overrides)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out_dir, "runs", run_id)
    ensure_dir(run_dir)

    day_tag = today_local.isoformat()

    print(f"[Week] {week_monday.isoformat()} → {week_sunday.isoformat()} (closes {week_friday.isoformat()}) tz={cfg.tz_name}")
    print(f"[Tickers] n={len(tickers)}  {', '.join(tickers)}")
    print(f"[Run] run_id={run_id}")
    print(f"[Config] r={cfg.risk_free_rate:.4f}")

    session = make_session(cfg)

    # 1) Polymarket snapshot
    pm = fetch_polymarket_snapshot(
        tickers=tickers,
        week_monday=week_monday,
        week_friday=week_friday,
        week_sunday=week_sunday,
        cfg=cfg,
        slug_overrides=slug_overrides,
        session=session,
    )

    pm_file = fname_pm_snapshot(day_tag)
    pm_top = os.path.join(args.out_dir, pm_file)
    pm_run = os.path.join(run_dir, pm_file)
    pm.to_csv(pm_top, index=False)
    pm.to_csv(pm_run, index=False)
    print(f"[Write] {pm_top} (rows={len(pm)})")
    print(f"[Write] {pm_run} (rows={len(pm)})")

    # 2) pRN snapshot
    rn = compute_pRN_snapshot(pm, cfg)

    rn_file = fname_rn_snapshot(day_tag, cfg.risk_free_rate)
    rn_top = os.path.join(args.out_dir, rn_file)
    rn_run = os.path.join(run_dir, rn_file)
    rn.to_csv(rn_top, index=False)
    rn.to_csv(rn_run, index=False)
    print(f"[Write] {rn_top} (rows={len(rn)})")
    print(f"[Write] {rn_run} (rows={len(rn)})")

    # 3) Merge + add simple RN edges
    merged = merge_pm_rn(pm, rn)
    merged = add_edge_columns(merged)
    final_df = select_final_columns(merged)

    sort_cols = [c for c in ["ticker", "event_endDate", "K"] if c in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    ds_file = fname_dataset_snapshot(day_tag, cfg.risk_free_rate)
    ds_top = os.path.join(args.out_dir, ds_file)
    ds_run = os.path.join(run_dir, ds_file)
    final_df.to_csv(ds_top, index=False)
    final_df.to_csv(ds_run, index=False)
    print(f"[Write] {ds_top} (rows={len(final_df)})")
    print(f"[Write] {ds_run} (rows={len(final_df)})")

    # 4) Append rolling history (merged only)
    hist_path = os.path.join(args.out_dir, "history", fname_dataset_history(cfg.risk_free_rate))
    hist = final_df.copy()
    hist["run_id"] = run_id
    hist["run_time_utc"] = datetime.now(timezone.utc).isoformat()
    append_df_to_csv(hist, hist_path)
    print(f"[History] appended -> {hist_path}")


if __name__ == "__main__":
    main()