"""On-demand pRN computation via Theta Terminal.

Fetches option chains from Theta, computes risk-neutral probabilities using
Breeden-Litzenberger with relaxed thresholds (suitable for backtest overlay,
not model training), and returns results in the same format as prn_overlay.py.

Disk-cached per (ticker, expiry, asof_date) to avoid redundant Theta calls.
"""

from __future__ import annotations

import json
import logging
import math
import socket
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from app.models.bars import PrnOverlayResponse, PrnPoint, PrnStrikeSeries

log = logging.getLogger("prn_on_demand")

BASE_DIR = Path(__file__).resolve().parents[5]
CACHE_DIR = BASE_DIR / "src" / "data" / "cache" / "prn_theta"
DEFAULT_THETA_URL = "http://127.0.0.1:25503/v3"

# DTE values we serve (weekly options: Mon=4, Tue=3, Wed=2, Thu=1 for Fri expiry)
ALLOWED_DTES = {1, 2, 3, 4}

# Concurrency cap for Theta calls
_theta_semaphore = threading.Semaphore(3)

# ---------------------------------------------------------------------------
# Relaxed config for overlay quality (vs. training quality)
# ---------------------------------------------------------------------------
RISK_FREE_RATE = 0.03
MIN_STRIKES_FOR_CURVE = 5       # pipeline uses 10
MAX_ABS_LOGM = 0.10             # pipeline uses 0.06
MAX_ABS_LOGM_CAP = 0.15         # pipeline uses 0.10
BAND_WIDEN_STEP = 0.01
PREFER_BIDASK_MIN = 5           # pipeline uses 10
REL_SPREAD_MAX = 2.0
INTRINSIC_TOL = 0.98
INSANE_PRICE_MULT = 1.5
TIMEOUT_S = 30
CACHE_TTL_RECENT_HOURS = 6


# ---------------------------------------------------------------------------
# Helpers (extracted from 01-option-chain-build-historic-dataset-v1.0.py)
# ---------------------------------------------------------------------------

def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _enforce_call_monotone_decreasing(prices: np.ndarray) -> np.ndarray:
    c = np.asarray(prices, dtype=float)
    c_rev = c[::-1]
    c_rev_mon = np.maximum.accumulate(c_rev)
    return c_rev_mon[::-1]


def _pava_isotonic_increasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if w is None:
        w_arr = np.ones(n, dtype=float)
    else:
        w_arr = np.asarray(w, dtype=float)
        w_arr = np.where(np.isfinite(w_arr) & (w_arr > 0), w_arr, 1.0)

    v: List[float] = []
    wt: List[float] = []
    ln: List[int] = []

    for yi, wi in zip(y, w_arr):
        v.append(float(yi))
        wt.append(float(wi))
        ln.append(1)
        while len(v) >= 2 and v[-2] > v[-1]:
            v1, w1, l1 = v.pop(), wt.pop(), ln.pop()
            v0, w0, l0 = v.pop(), wt.pop(), ln.pop()
            new_w = w0 + w1
            new_v = (v0 * w0 + v1 * w1) / new_w
            v.append(new_v)
            wt.append(new_w)
            ln.append(l0 + l1)

    out = np.empty(n, dtype=float)
    idx = 0
    for val, length in zip(v, ln):
        out[idx : idx + length] = val
        idx += length
    return out


def _isotonic_decreasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    return -_pava_isotonic_increasing(-np.asarray(y, dtype=float), w=w)


# ---------------------------------------------------------------------------
# Theta client (minimal, mirrors pipeline ThetaClient)
# ---------------------------------------------------------------------------

def _theta_is_available(theta_url: str = DEFAULT_THETA_URL) -> bool:
    parsed = urlparse(theta_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return True
    except OSError:
        return False


def _theta_get_json(session: requests.Session, theta_url: str, path: str, params: dict) -> list:
    url = f"{theta_url.rstrip('/')}/{path.lstrip('/')}"
    r = session.get(url, params=params, timeout=TIMEOUT_S)
    if r.status_code in (404, 472):
        return []
    r.raise_for_status()
    payload = r.json()
    if isinstance(payload, dict):
        resp = payload.get("response", []) or []
        return resp if isinstance(resp, list) else []
    return payload if isinstance(payload, list) else []


def _fetch_call_chain(
    session: requests.Session,
    theta_url: str,
    symbol: str,
    asof: date,
    expiration: date,
) -> pd.DataFrame:
    """Fetch call option chain from Theta Terminal for a single asof+expiry."""
    params = {
        "symbol": symbol,
        "expiration": _yyyymmdd(expiration),
        "right": "call",
        "start_date": _yyyymmdd(asof),
        "end_date": _yyyymmdd(asof),
        "format": "json",
    }
    with _theta_semaphore:
        resp = _theta_get_json(session, theta_url, "option/history/eod", params)

    rows = []
    for item in resp:
        contract = item.get("contract", {}) or {}
        data_list = item.get("data", []) or []
        for bar in data_list:
            rows.append({
                "strike": contract.get("strike"),
                "bid": bar.get("bid"),
                "ask": bar.get("ask"),
                "close": bar.get("close"),
                "volume": bar.get("volume"),
                "count": bar.get("count"),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Call curve building (relaxed thresholds)
# ---------------------------------------------------------------------------

def _build_call_curve(
    chain: pd.DataFrame,
    spot: float,
    T_years: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build clean (strike, call_mid) arrays from raw chain. Returns None on failure."""
    if chain is None or chain.empty:
        return None, None

    df = chain.copy()
    for c in ["strike", "bid", "ask", "close", "volume", "count"]:
        if c not in df.columns:
            df[c] = np.nan

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Bid-ask midpoint
    df["mid_ba"] = np.where(
        df["bid"].notna() & df["ask"].notna()
        & (df["bid"] >= 0) & (df["ask"] > 0)
        & (df["ask"] >= df["bid"]),
        0.5 * (df["bid"] + df["ask"]),
        np.nan,
    )
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid_ba"]

    # Quote selection: prefer bid-ask, fall back to close
    if df["mid_ba"].notna().sum() >= PREFER_BIDASK_MIN:
        use = df.dropna(subset=["strike", "mid_ba"]).copy()
        use = use[use["mid_ba"] > 0].copy()
        use["mid"] = use["mid_ba"]
        use = use[(use["rel_spread"].isna()) | (use["rel_spread"] <= REL_SPREAD_MAX)].copy()
    else:
        use = df.dropna(subset=["strike", "close"]).copy()
        use = use[use["close"] > 0].copy()
        use["mid"] = use["close"]

    use = use.dropna(subset=["strike", "mid"]).copy()
    use = use[(use["strike"] > 0) & (use["mid"] > 0)].copy()
    if use.empty:
        return None, None

    # Intrinsic floor
    T_ref = max(float(T_years), 1e-8)
    discK = use["strike"].astype(float) * np.exp(-RISK_FREE_RATE * T_ref)
    fwd_disc = float(spot) * np.exp(0.0)  # q=0 simplification for overlay
    intrinsic = np.maximum(fwd_disc - discK, 0.0)
    use = use[use["mid"].astype(float) >= INTRINSIC_TOL * intrinsic].copy()

    # Insane price filter
    use = use[use["mid"].astype(float) <= INSANE_PRICE_MULT * float(spot)].copy()

    use = use.sort_values("strike").drop_duplicates("strike", keep="last")

    if len(use) < MIN_STRIKES_FOR_CURVE:
        return None, None

    k_arr = use["strike"].to_numpy(dtype=float)
    c_arr = use["mid"].to_numpy(dtype=float)

    # Monotone enforce + rolling median smoothing
    c_mon = _enforce_call_monotone_decreasing(c_arr)
    c_mon = pd.Series(c_mon).rolling(5, center=True, min_periods=1).median().to_numpy(dtype=float)

    return k_arr, c_mon


# ---------------------------------------------------------------------------
# pRN computation (Breeden-Litzenberger)
# ---------------------------------------------------------------------------

def _compute_prn(
    k_arr: np.ndarray,
    c_arr: np.ndarray,
    K_targets: np.ndarray,
    T_years: float,
) -> Optional[np.ndarray]:
    """Compute pRN for target strikes. Returns None on failure."""
    k = np.asarray(k_arr, dtype=float)
    c = np.asarray(c_arr, dtype=float)
    Kt = np.asarray(K_targets, dtype=float)

    if k.size < 3 or c.size != k.size or Kt.size == 0:
        return None

    dk0 = np.diff(k)
    dc0 = np.diff(c)
    ok = np.isfinite(dk0) & (dk0 > 0) & np.isfinite(dc0)
    if int(ok.sum()) < 2:
        return None

    dk = dk0[ok]
    dc = dc0[ok]
    k_left = k[:-1][ok]
    k_right = k[1:][ok]
    k_mid = 0.5 * (k_left + k_right)

    slope = dc / dk
    p_int = -np.exp(RISK_FREE_RATE * float(T_years)) * slope
    p_int = np.clip(p_int, 0.0, 1.0)

    # Isotonic decreasing on intervals
    p_iso = _isotonic_decreasing(p_int)
    p_iso = np.clip(p_iso, 0.0, 1.0)

    # Interpolate to targets
    p_t = np.interp(Kt, k_mid, p_iso)

    # Target-level isotonic safeguard
    order = np.argsort(Kt)
    p_sorted = p_t[order]
    p_sorted_iso = _isotonic_decreasing(p_sorted)
    p_sorted_iso = np.clip(p_sorted_iso, 0.0, 1.0)

    out = np.empty_like(p_t)
    out[order] = p_sorted_iso
    return out


# ---------------------------------------------------------------------------
# Band selection (adaptive, relaxed)
# ---------------------------------------------------------------------------

def _pick_band_strikes(
    all_strikes: np.ndarray,
    spot: float,
) -> np.ndarray:
    """Select strikes within adaptive moneyness band around spot."""
    k = np.asarray(all_strikes, dtype=float)
    k = k[np.isfinite(k) & (k > 0)]
    if k.size == 0 or not np.isfinite(spot) or spot <= 0:
        return np.array([], dtype=float)

    used = MAX_ABS_LOGM
    while used <= MAX_ABS_LOGM_CAP + 1e-12:
        logm = np.log(k / float(spot))
        band = k[np.abs(logm) <= used]
        if band.size >= MIN_STRIKES_FOR_CURVE:
            return np.sort(np.unique(band))
        used += BAND_WIDEN_STEP

    # Return whatever we have at cap
    logm = np.log(k / float(spot))
    band = k[np.abs(logm) <= MAX_ABS_LOGM_CAP]
    return np.sort(np.unique(band))


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_path(ticker: str, expiry: date, asof: date) -> Path:
    return (
        CACHE_DIR / ticker.upper()
        / f"exp_{_yyyymmdd(expiry)}"
        / f"asof_{_yyyymmdd(asof)}.json"
    )


def _cache_read(ticker: str, expiry: date, asof: date) -> Optional[dict]:
    p = _cache_path(ticker, expiry, asof)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # Check TTL for recent dates
        cached_at = data.get("cached_at")
        if cached_at:
            age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(cached_at)).total_seconds() / 3600
            days_old = (date.today() - asof).days
            if days_old <= 1 and age_hours > CACHE_TTL_RECENT_HOURS:
                return None  # stale for recent dates
        return data
    except Exception:
        return None


def _cache_write(ticker: str, expiry: date, asof: date, prn_rows: list) -> None:
    p = _cache_path(ticker, expiry, asof)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticker": ticker,
        "expiry": _yyyymmdd(expiry),
        "asof": _yyyymmdd(asof),
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "rows": prn_rows,
    }
    p.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def _trading_dates_in_range(start: date, end: date) -> List[date]:
    """Generate weekday dates in [start, end]."""
    dates = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            dates.append(d)
        d += timedelta(days=1)
    return dates


def _friday_of_week(d: date) -> date:
    """Return the Friday of the week containing d."""
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _dte_for_asof(asof: date, expiry: date) -> int:
    """Compute DTE (calendar days), clamped to positive."""
    return max((expiry - asof).days, 0)


def _asof_date_to_eod_ms(d: date) -> int:
    """US-market-close UTC ms (21:00 UTC) — pRN uses EOD chain, must not plot before close."""
    dt = datetime(d.year, d.month, d.day, 21, 0, 0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Fetch spot price from Theta (stock EOD)
# ---------------------------------------------------------------------------

def _fetch_spot(
    session: requests.Session,
    theta_url: str,
    symbol: str,
    asof: date,
) -> Optional[float]:
    """Fetch EOD stock close for symbol on asof date (with 3-day lookback)."""
    start = asof - timedelta(days=5)
    params = {
        "symbol": symbol,
        "start_date": _yyyymmdd(start),
        "end_date": _yyyymmdd(asof),
        "format": "json",
    }
    with _theta_semaphore:
        data = _theta_get_json(session, theta_url, "stock/history/eod", params)
    if not data:
        return None

    df = pd.DataFrame(data)
    if df.empty or "close" not in df.columns:
        return None
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    if df.empty:
        return None

    # Return most recent close
    return float(df["close"].iloc[-1])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_prn_on_demand(
    ticker: str,
    time_min: Optional[str] = None,
    time_max: Optional[str] = None,
    dte_list: Optional[List[int]] = None,
    target_strikes: Optional[List[float]] = None,
    theta_url: str = DEFAULT_THETA_URL,
) -> PrnOverlayResponse:
    """Compute pRN on-demand via Theta Terminal for a ticker + date range.

    For each trading day in the range, determines the Friday expiry for that week,
    computes DTE, and if the DTE is in the requested set, fetches the call chain
    from Theta and computes pRN.

    If target_strikes is provided, pRN is interpolated to those exact strikes
    (matching Polymarket strikes).  Otherwise, uses moneyness band selection.

    Returns PrnOverlayResponse in the same format as prn_overlay.get_prn_overlay().
    """
    if dte_list is None:
        dte_list = [4, 3, 2, 1]
    allowed = set(dte_list) & ALLOWED_DTES

    # Parse date bounds
    date_start = date.today() - timedelta(days=7)
    date_end = date.today()
    if time_min:
        date_start = datetime.fromisoformat(time_min.replace("Z", "+00:00")).date()
    if time_max:
        date_end = datetime.fromisoformat(time_max.replace("Z", "+00:00")).date()

    # Safety: cap range to 10 days
    if (date_end - date_start).days > 10:
        date_end = date_start + timedelta(days=10)

    # Check Theta availability
    if not _theta_is_available(theta_url):
        log.warning("Theta Terminal not available at %s", theta_url)
        return PrnOverlayResponse(
            ticker=ticker,
            strikes=[],
            metadata={"error": "theta_unavailable", "source": "theta_on_demand"},
        )

    trading_days = _trading_dates_in_range(date_start, date_end)
    if not trading_days:
        return PrnOverlayResponse(
            ticker=ticker, strikes=[],
            metadata={"error": "no_trading_days", "source": "theta_on_demand"},
        )

    session = requests.Session()

    # Group trading days by (expiry_friday, dte) and process
    # strike -> list of PrnPoint
    groups: Dict[float, List[PrnPoint]] = {}
    theta_calls = 0
    cache_hits = 0
    chains_empty = 0
    prn_failures = 0

    for asof in trading_days:
        expiry = _friday_of_week(asof)
        dte = _dte_for_asof(asof, expiry)
        if dte not in allowed or dte < 1:
            continue

        # Check cache first
        cached = _cache_read(ticker, expiry, asof)
        if cached is not None:
            cache_hits += 1
            for row in cached.get("rows", []):
                strike = row.get("strike")
                prn_val = row.get("pRN")
                if strike is not None and prn_val is not None:
                    groups.setdefault(round(float(strike), 2), []).append(PrnPoint(
                        asof_date=asof.isoformat(),
                        asof_date_ms=_asof_date_to_eod_ms(asof),
                        dte=dte,
                        pRN=float(prn_val),
                    ))
            continue

        # Fetch spot price
        spot = _fetch_spot(session, theta_url, ticker, asof)
        if spot is None or not math.isfinite(spot) or spot <= 0:
            log.debug("No spot for %s on %s", ticker, asof)
            _cache_write(ticker, expiry, asof, [])
            continue

        # Fetch call chain
        try:
            theta_calls += 1
            chain = _fetch_call_chain(session, theta_url, ticker, asof, expiry)
        except Exception as e:
            log.warning("Theta call failed for %s %s->%s: %s", ticker, asof, expiry, e)
            continue

        if chain.empty:
            chains_empty += 1
            _cache_write(ticker, expiry, asof, [])
            continue

        T_years = max(dte / 365.0, 1e-8)

        # Build call curve from full chain
        k_arr, c_arr = _build_call_curve(chain, spot, T_years)
        if k_arr is None or c_arr is None:
            prn_failures += 1
            _cache_write(ticker, expiry, asof, [])
            continue

        # Determine target strikes for pRN interpolation:
        # - If caller provided target_strikes (Polymarket strikes), use those
        #   that fall within the chain's strike range (interpolation, not extrapolation)
        # - Otherwise, use adaptive moneyness band selection
        if target_strikes:
            k_min, k_max = float(k_arr[0]), float(k_arr[-1])
            k_targets = np.array(
                sorted(set(round(s, 2) for s in target_strikes
                           if k_min <= s <= k_max and math.isfinite(s))),
                dtype=float,
            )
        else:
            k_targets = _pick_band_strikes(k_arr, spot)

        if k_targets.size < 1:
            prn_failures += 1
            _cache_write(ticker, expiry, asof, [])
            continue

        # Compute pRN at target strikes
        prn_arr = _compute_prn(k_arr, c_arr, k_targets, T_years)
        if prn_arr is None:
            prn_failures += 1
            _cache_write(ticker, expiry, asof, [])
            continue

        # Build cache rows and group results
        cache_rows = []
        for k_val, p_val in zip(k_targets, prn_arr):
            if not math.isfinite(p_val):
                continue
            strike_norm = round(float(k_val), 2)
            cache_rows.append({"strike": strike_norm, "pRN": round(float(p_val), 6)})
            groups.setdefault(strike_norm, []).append(PrnPoint(
                asof_date=asof.isoformat(),
                asof_date_ms=_asof_date_to_eod_ms(asof),
                dte=dte,
                pRN=float(p_val),
            ))

        _cache_write(ticker, expiry, asof, cache_rows)

    session.close()

    # Build sorted output, deduplicating by (asof_date, dte) per strike
    strikes_list: List[PrnStrikeSeries] = []
    for strike in sorted(groups.keys()):
        seen: set = set()
        deduped: List[PrnPoint] = []
        for p in sorted(groups[strike], key=lambda p: p.asof_date_ms):
            key = (p.asof_date, p.dte)
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        label = str(int(strike)) if strike == int(strike) else f"{strike:.2f}"
        strikes_list.append(PrnStrikeSeries(
            strike=strike,
            strike_label=label,
            points=deduped,
        ))

    log.info(
        "prn_on_demand: ticker=%s range=%s..%s theta_calls=%d cache_hits=%d "
        "empty=%d failures=%d strikes=%d",
        ticker, date_start, date_end, theta_calls, cache_hits,
        chains_empty, prn_failures, len(strikes_list),
    )

    return PrnOverlayResponse(
        ticker=ticker,
        strikes=strikes_list,
        metadata={
            "source": "theta_on_demand",
            "theta_calls": theta_calls,
            "cache_hits": cache_hits,
            "chains_empty": chains_empty,
            "prn_failures": prn_failures,
            "strikes_count": len(strikes_list),
            "date_range": f"{date_start}..{date_end}",
        },
    )
