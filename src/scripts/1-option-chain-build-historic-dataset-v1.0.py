#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import os
import sys
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None


PM10_TICKERS = ["AAPL", "GOOGL", "MSFT", "META", "AMZN", "PLTR", "NVDA", "NFLX", "OPEN", "TSLA"]


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    theta_base_url: str = "http://127.0.0.1:25503/v3"
    timeout_s: int = 30
    risk_free_rate: float = 0.03

    # Option fetch
    option_strike_range: int = 60
    retry_full_chain_if_band_thin: bool = True
    try_saturday_expiry_fallback: bool = True

    # Missing close handling
    max_forward_days_for_asof: int = 3
    max_backward_days_for_expiry_close: int = 3

    # Band selection
    max_abs_logm: float = 0.06
    max_abs_logm_cap: float = 0.10
    band_widen_step: float = 0.01
    adaptive_band: bool = True
    max_band_strikes: int = 0  # 0 = keep all

    # Minimum strikes thresholds
    min_strikes_for_curve: int = 10
    min_strikes_in_prn_band: int = 7

    # Curve cleaning
    rel_spread_max_per_strike: float = 2.0
    intrinsic_tol: float = 0.98
    insane_price_multiple: float = 1.5

    # Quote preference + liquidity
    prefer_bidask: bool = True
    min_trade_count: int = 0
    min_volume: int = 0

    # Hard filters
    min_chain_used_hard: int = 0
    max_rel_spread_median_hard: float = 1e9
    hard_drop_close_fallback: bool = False

    # Training band
    min_prn_train: float = 0.10
    max_prn_train: float = 0.90

    # Stock close source
    stock_source: str = "yfinance"
    stock_preload_buffer_days: int = 7

    # Split adjustment
    apply_split_adjustment: bool = True
    split_source: str = "yfinance"

    # Dividends / forward
    dividend_source: str = "yfinance"
    dividend_lookback_days: int = 365
    dividend_yield_default: float = 0.0
    use_forward_moneyness: bool = True

    # Weighting controls
    add_group_weights: bool = True
    add_ticker_weights: bool = True
    use_soft_quality_weight: bool = True

    # Cache
    use_cache: bool = True

    # Volatility proxy
    rv_lookback_days: int = 20

    # Optional sanity checks
    sanity_report: bool = False
    sanity_drop: bool = False
    sanity_abs_logm_max: float = 0.40
    sanity_k_over_s_min: float = 0.25
    sanity_k_over_s_max: float = 4.0


# ----------------------------
# Date helpers
# ----------------------------

def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def iso_week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def iso_week_friday(d: date) -> date:
    return iso_week_monday(d) + timedelta(days=4)


def mondays_in_range(start: date, end: date) -> List[date]:
    return [d.date() for d in pd.date_range(start=start, end=end, freq="W-MON")]


def asof_days_mon_to_thu(week_monday: date) -> List[date]:
    return [week_monday + timedelta(days=i) for i in range(0, 4)]


# ----------------------------
# Theta client
# ----------------------------

class ThetaClient:
    def __init__(self, base_url: str, timeout_s: int = 30, *, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.verbose = bool(verbose)

    def _get_json(self, session: requests.Session, path: str, params: dict) -> list:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            r = session.get(url, params=params, timeout=self.timeout_s)
            if r.status_code in (404, 472):
                return []
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict):
                resp = payload.get("response", []) or []
                return resp if isinstance(resp, list) else []
            return payload if isinstance(payload, list) else []
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError("Cannot connect to Theta Terminal (is it running on 25503?).") from e
        except Exception as e:
            if self.verbose:
                print(f"[THETA] ⚠️ GET failed {url} params={params} err={e}")
            return []

    def stock_eod_range(self, session: requests.Session, symbol: str, start: date, end: date) -> pd.DataFrame:
        data = self._get_json(
            session,
            "stock/history/eod",
            {"symbol": symbol, "start_date": yyyymmdd(start), "end_date": yyyymmdd(end), "format": "json"},
        )
        return pd.DataFrame(data)

    def option_eod_chain(
        self,
        session: requests.Session,
        symbol: str,
        asof: date,
        expiration: date,
        *,
        right: str = "call",
        strike_range: Optional[int] = None,
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "expiration": yyyymmdd(expiration),
            "right": right,
            "start_date": yyyymmdd(asof),
            "end_date": yyyymmdd(asof),
            "format": "json",
        }
        if strike_range is not None:
            params["strike_range"] = int(strike_range)

        resp = self._get_json(session, "option/history/eod", params)

        rows = []
        for item in resp:
            contract = item.get("contract", {}) or {}
            data_list = item.get("data", []) or []
            for bar in data_list:
                rows.append(
                    {
                        "symbol": contract.get("symbol", symbol),
                        "strike": contract.get("strike"),
                        "right": contract.get("right"),
                        "expiration": contract.get("expiration"),
                        "bid": bar.get("bid"),
                        "ask": bar.get("ask"),
                        "close": bar.get("close"),
                        "volume": bar.get("volume"),
                        "count": bar.get("count"),
                        "created": bar.get("created"),
                        "last_trade": bar.get("last_trade"),
                    }
                )
        return pd.DataFrame(rows)


# ----------------------------
# Thread-local session
# ----------------------------

_thread_local = threading.local()

def get_thread_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


# ----------------------------
# Stock close preload + split handling
# ----------------------------

def _yf_download_closes(tickers: List[str], start0: date, end0: date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. Install with: pip install yfinance")
    tlist = " ".join(sorted(set(tickers)))
    df = yf.download(
        tickers=tlist,
        start=start0.isoformat(),
        end=(end0 + timedelta(days=1)).isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned empty dataframe.")
    return df


def _yf_download_splits(tickers: List[str], start0: date, end0: date) -> Dict[str, pd.Series]:
    if yf is None:
        raise RuntimeError("yfinance not installed. Install with: pip install yfinance")
    out: Dict[str, pd.Series] = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            s = tk.splits
            if s is None or len(s) == 0:
                out[t] = pd.Series(dtype=float)
                continue
            s = s[(s.index.date >= start0) & (s.index.date <= end0)]
            out[t] = s.astype(float)
        except Exception:
            out[t] = pd.Series(dtype=float)
    return out


def _build_split_adjustment_factor(dates: List[date], splits: pd.Series) -> Dict[date, float]:
    if splits is None or len(splits) == 0:
        return {d: 1.0 for d in dates}
    split_events = sorted(
        [(pd.Timestamp(idx).date(), float(val)) for idx, val in splits.items() if np.isfinite(val) and val > 0]
    )
    if not split_events:
        return {d: 1.0 for d in dates}

    dates_sorted = sorted(dates)
    factors: Dict[date, float] = {}
    for d in dates_sorted:
        f = 1.0
        for ds, r in split_events:
            if d < ds:
                f *= r
        factors[d] = f
    return factors


def infer_close_is_already_split_adjusted(close_map: Dict[date, float], splits: pd.Series) -> Optional[bool]:
    if splits is None or len(splits) == 0 or not close_map:
        return None

    events = sorted([(pd.Timestamp(idx).date(), float(val)) for idx, val in splits.items() if np.isfinite(val) and val > 0])
    if not events:
        return None

    dates = sorted(close_map.keys())
    for ds, r in events:
        pre = [d for d in dates if d < ds]
        post = [d for d in dates if d >= ds]
        if not pre or not post:
            continue
        d_pre = pre[-1]
        d_post = post[0]
        c_pre = float(close_map[d_pre])
        c_post = float(close_map[d_post])
        if not (np.isfinite(c_pre) and np.isfinite(c_post) and c_pre > 0 and c_post > 0):
            continue

        ratio = c_pre / c_post
        if abs(ratio - r) / r <= 0.25:
            return False
        if abs(ratio - 1.0) <= 0.25:
            return True
    return None


def preload_stock_closes(
    *,
    theta: ThetaClient,
    tickers: List[str],
    start: date,
    end: date,
    cfg: Config,
    stock_source: str,
) -> Tuple[Dict[str, Dict[date, float]], Dict[str, Dict[date, float]], Dict[str, int]]:
    stock_source = (stock_source or "yfinance").strip().lower()
    if stock_source not in {"yfinance", "theta", "auto"}:
        raise ValueError("stock_source must be one of: yfinance, theta, auto")

    start0 = start - timedelta(days=int(cfg.stock_preload_buffer_days))
    end0 = end + timedelta(days=int(cfg.stock_preload_buffer_days))

    raw_out: Dict[str, Dict[date, float]] = {t: {} for t in tickers}
    adj_out: Dict[str, Dict[date, float]] = {t: {} for t in tickers}
    split_counts: Dict[str, int] = {t: 0 for t in tickers}

    def _fill_from_yf() -> None:
        nonlocal raw_out, adj_out, split_counts
        print(f"[STOCK] Preloading yfinance Close for {start0}..{end0} (tickers={len(tickers)}) ...")
        df = _yf_download_closes(tickers, start0, end0)

        if isinstance(df.columns, pd.MultiIndex):
            for t in tickers:
                if t not in df.columns.levels[0]:
                    continue
                sub = df[t].copy()
                if "Close" not in sub.columns:
                    continue
                s = pd.to_numeric(sub["Close"], errors="coerce").dropna()
                for ts, v in s.items():
                    raw_out[t][pd.Timestamp(ts).date()] = float(v)
        else:
            if "Close" in df.columns and tickers:
                t = tickers[0]
                s = pd.to_numeric(df["Close"], errors="coerce").dropna()
                for ts, v in s.items():
                    raw_out[t][pd.Timestamp(ts).date()] = float(v)

        if cfg.apply_split_adjustment and cfg.split_source == "yfinance":
            print("[STOCK] Fetching split events via yfinance ...")
            splits_by_t = _yf_download_splits(tickers, start0, end0)

            for t in tickers:
                splits = splits_by_t.get(t, pd.Series(dtype=float))
                split_counts[t] = int(len(splits)) if splits is not None else 0
                if not raw_out.get(t):
                    continue

                dates = list(raw_out[t].keys())
                f_map = _build_split_adjustment_factor(dates, splits)
                already_adj = infer_close_is_already_split_adjusted(raw_out[t], splits)

                if already_adj is True:
                    adj_out[t] = dict(raw_out[t])
                    raw_out[t] = {d: float(raw_out[t][d] * float(f_map.get(d, 1.0))) for d in dates}
                else:
                    adj_out[t] = {
                        d: float(raw_out[t][d] / float(f_map.get(d, 1.0))) if float(f_map.get(d, 1.0)) > 0 else float(raw_out[t][d])
                        for d in dates
                    }
        else:
            for t in tickers:
                adj_out[t] = dict(raw_out[t])

    def _theta_fetch(sym: str) -> Dict[date, float]:
        session = requests.Session()
        df = theta.stock_eod_range(session, sym, start0, end0)
        m: Dict[date, float] = {}
        if df is None or df.empty:
            return m
        dcol = None
        for c in ["date", "time", "t", "timestamp"]:
            if c in df.columns:
                dcol = c
                break
        if dcol is None or "close" not in df.columns:
            return m
        ds = pd.to_datetime(df[dcol], errors="coerce")
        cs = pd.to_numeric(df["close"], errors="coerce")
        for dts, c in zip(ds, cs):
            if pd.isna(dts) or not np.isfinite(c):
                continue
            m[dts.date()] = float(c)
        return m

    if stock_source in {"yfinance", "auto"}:
        try:
            _fill_from_yf()
        except Exception as e:
            if stock_source == "yfinance":
                raise
            print(f"[STOCK] ⚠️ yfinance preload failed, falling back to Theta: {e}")

    if stock_source in {"theta", "auto"}:
        need = [t for t in tickers if not raw_out.get(t)]
        if need:
            print(f"[STOCK] Preloading closes via Theta fallback for {len(need)} tickers...")
        for t in need:
            try:
                raw_out[t] = _theta_fetch(t)
                adj_out[t] = dict(raw_out[t])
                split_counts[t] = 0
            except Exception:
                raw_out[t] = {}
                adj_out[t] = {}
                split_counts[t] = 0

    still_missing = [t for t in tickers if not raw_out.get(t)]
    if still_missing:
        print(f"[STOCK] ⚠️ No close data from yfinance/Theta for: {still_missing}")

    return raw_out, adj_out, split_counts


# ----------------------------
# Dividend preload (time-safe)
# ----------------------------

@dataclass
class DividendHistory:
    dates: List[date]
    cumsum: np.ndarray
    available: bool = True

    def sum_in_range(self, start: date, end: date) -> float:
        if not self.available or not self.dates:
            return 0.0
        l = bisect.bisect_left(self.dates, start)
        r = bisect.bisect_right(self.dates, end)
        if r <= l:
            return 0.0
        s = float(self.cumsum[r - 1])
        if l > 0:
            s -= float(self.cumsum[l - 1])
        return s


def _build_dividend_history_from_series(s: pd.Series) -> DividendHistory:
    if s is None or len(s) == 0:
        return DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=True)
    s = s.groupby(s.index.date).sum()
    if s is None or len(s) == 0:
        return DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=True)
    dates = sorted(list(s.index))
    amounts = [float(s.loc[d]) for d in dates]
    cumsum = np.cumsum(amounts, dtype=float)
    return DividendHistory(dates=dates, cumsum=cumsum, available=True)


def _yf_download_dividends(
    tickers: List[str],
    start0: date,
    end0: date,
    lookback_days: int,
) -> Dict[str, DividendHistory]:
    if yf is None:
        raise RuntimeError("yfinance not installed. Install with: pip install yfinance")
    out: Dict[str, DividendHistory] = {}
    fetch_start = start0 - timedelta(days=int(lookback_days))
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            s = tk.dividends
            if s is None or len(s) == 0:
                out[t] = DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=True)
                continue
            s = s[(s.index.date >= fetch_start) & (s.index.date <= end0)]
            out[t] = _build_dividend_history_from_series(s)
        except Exception:
            out[t] = DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=False)
    return out


def preload_dividend_histories(
    *,
    tickers: List[str],
    start: date,
    end: date,
    cfg: Config,
) -> Dict[str, DividendHistory]:
    if cfg.dividend_source != "yfinance":
        return {t: DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=False) for t in tickers}
    print(f"[DIV] Preloading dividends via yfinance for {start}..{end} (lookback_days={cfg.dividend_lookback_days}) ...")
    return _yf_download_dividends(tickers, start, end, cfg.dividend_lookback_days)


# ----------------------------
# Small utilities
# ----------------------------

def get_close_with_fallback_map(
    close_map: Dict[date, float],
    d0: date,
    *,
    direction: str,
    max_days: int,
) -> Tuple[Optional[float], Optional[date], int]:
    step = 1 if direction == "forward" else -1
    for k in range(0, max_days + 1):
        d = d0 + timedelta(days=step * k)
        if d in close_map:
            return close_map[d], d, k
    return None, None, max_days + 1


def realized_vol_proxy(close_map: Dict[date, float], asof_used: date, lookback: int) -> float:
    if lookback <= 2:
        return np.nan
    dates = sorted([d for d in close_map.keys() if d <= asof_used])
    if len(dates) < lookback + 1:
        return np.nan
    sel = dates[-(lookback + 1):]
    px = np.array([close_map[d] for d in sel], dtype=float)
    if not np.all(np.isfinite(px)) or np.any(px <= 0):
        return np.nan
    rets = np.diff(np.log(px))
    if rets.size < 2:
        return np.nan
    return float(np.sqrt(252.0) * np.nanstd(rets, ddof=1))


def fix_negative_zero(x: float, eps: float = 5e-13) -> float:
    if x is None or not np.isfinite(x):
        return x
    return 0.0 if abs(float(x)) < eps else float(x)


# ----------------------------
# Curve cleaning + isotonic
# ----------------------------

def enforce_call_monotone_decreasing(prices: np.ndarray) -> np.ndarray:
    c = np.asarray(prices, dtype=float)
    c_rev = c[::-1]
    c_rev_mon = np.maximum.accumulate(c_rev)
    return c_rev_mon[::-1]


def pava_isotonic_increasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    v: List[float] = []
    wt: List[float] = []
    ln: List[int] = []

    for yi, wi in zip(y, w):
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


def isotonic_decreasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    return -pava_isotonic_increasing(-np.asarray(y, dtype=float), w=w)


def build_call_curve_from_eod(
    chain: pd.DataFrame,
    *,
    spot: float,
    T_years: float,
    r: float,
    q: float = 0.0,
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], dict]:
    diag = {
        "n_raw": int(len(chain)) if chain is not None else 0,
        "n_used": 0,
        "quote_source": None,
        "rel_spread_median": None,
        "dropped_liquidity": 0,
        "dropped_intrinsic": 0,
        "dropped_insane": 0,
    }
    if chain is None or chain.empty:
        return None, None, diag

    df = chain.copy()
    for c in ["strike", "bid", "ask", "close", "volume", "count"]:
        if c not in df.columns:
            df[c] = np.nan

    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["bid"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ask"] = pd.to_numeric(df["ask"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df["count"] = pd.to_numeric(df["count"], errors="coerce")

    before = len(df)
    if cfg.min_trade_count > 0:
        df = df[(df["count"].fillna(0) >= cfg.min_trade_count)].copy()
    if cfg.min_volume > 0:
        df = df[(df["volume"].fillna(0) >= cfg.min_volume)].copy()
    diag["dropped_liquidity"] = int(before - len(df))

    df["mid_ba"] = np.where(
        df["bid"].notna()
        & df["ask"].notna()
        & (df["bid"] >= 0)
        & (df["ask"] > 0)
        & (df["ask"] >= df["bid"]),
        0.5 * (df["bid"] + df["ask"]),
        np.nan,
    )
    df["spread"] = df["ask"] - df["bid"]
    df["rel_spread"] = df["spread"] / df["mid_ba"]

    if cfg.prefer_bidask and df["mid_ba"].notna().sum() >= 10:
        diag["quote_source"] = "bidask_mid"
        use = df.dropna(subset=["strike", "mid_ba"]).copy()
        use = use[use["mid_ba"] > 0].copy()
        use["mid"] = use["mid_ba"]
        # drop ultra-wide quotes per strike (if rel_spread computable)
        use = use[(use["rel_spread"].isna()) | (use["rel_spread"] <= cfg.rel_spread_max_per_strike)].copy()
    else:
        diag["quote_source"] = "close_fallback"
        use = df.dropna(subset=["strike", "close"]).copy()
        use = use[use["close"] > 0].copy()
        use["mid"] = use["close"]

    use = use.dropna(subset=["strike", "mid"]).copy()
    use = use[(use["strike"] > 0) & (use["mid"] > 0)].copy()
    if use.empty:
        return None, None, diag

    # Intrinsic bound (discounted strike)
    T_ref = max(float(T_years), 1e-8)
    discK = use["strike"].astype(float) * np.exp(-float(r) * T_ref)
    fwd_disc = float(spot) * np.exp(-float(q) * T_ref)
    intrinsic = np.maximum(fwd_disc - discK, 0.0)

    before = len(use)
    use = use[use["mid"].astype(float) >= float(cfg.intrinsic_tol) * intrinsic].copy()
    diag["dropped_intrinsic"] = int(before - len(use))

    before = len(use)
    use = use[use["mid"].astype(float) <= float(cfg.insane_price_multiple) * float(spot)].copy()
    diag["dropped_insane"] = int(before - len(use))

    use = use.sort_values("strike").drop_duplicates("strike", keep="last")
    diag["n_used"] = int(len(use))

    if "rel_spread" in use.columns:
        m = pd.to_numeric(use["rel_spread"], errors="coerce").median()
        diag["rel_spread_median"] = float(m) if np.isfinite(m) else None

    if len(use) < 5:
        return None, None, diag

    k_arr = use["strike"].to_numpy(dtype=float)
    c_arr = use["mid"].to_numpy(dtype=float)

    # Light cleaning: enforce monotone decreasing, then a small rolling median to de-noise
    c_mon = enforce_call_monotone_decreasing(c_arr)
    c_mon = pd.Series(c_mon).rolling(5, center=True, min_periods=1).median().to_numpy(dtype=float)

    return k_arr, c_mon, diag


def compute_prn_from_call_curve(
    k_arr: np.ndarray,
    c_arr: np.ndarray,
    *,
    K_targets: np.ndarray,
    T_years: float,
    r: float,
) -> Tuple[np.ndarray, dict]:
    """
    Breeden-Litzenberger-ish discrete slope:
      pRN(K_mid) ≈ -exp(rT) * dC/dK  (clipped to [0,1])

    We compute on strike midpoints, then interpolate to K_targets.
    We also compute a "raw" (no isotonic) interpolation to targets for audit.
    Then apply isotonic decreasing enforcement (pRN must be non-increasing in K).
    """
    diag = {
        "monotone_adjusted_intervals": False,
        "monotone_adjusted_targets": False,
        "p_targets_raw": None,
    }

    k = np.asarray(k_arr, dtype=float)
    c = np.asarray(c_arr, dtype=float)
    Kt = np.asarray(K_targets, dtype=float)

    if k.size < 3 or c.size != k.size or Kt.size == 0:
        return np.full_like(Kt, np.nan, dtype=float), diag

    dk0 = np.diff(k)
    dc0 = np.diff(c)
    ok = np.isfinite(dk0) & (dk0 > 0) & np.isfinite(dc0)
    if int(ok.sum()) < 2:
        return np.full_like(Kt, np.nan, dtype=float), diag

    dk = dk0[ok]
    dc = dc0[ok]
    k_left = k[:-1][ok]
    k_right = k[1:][ok]
    k_mid = 0.5 * (k_left + k_right)

    slope = dc / dk
    p_int = -np.exp(float(r) * float(T_years)) * slope
    p_int = np.clip(p_int, 0.0, 1.0)

    # Raw interpolation to targets (audit)
    p_t_raw = np.interp(Kt, k_mid, p_int)
    diag["p_targets_raw"] = p_t_raw.copy()

    # Isotonic decreasing on interval-based p(K_mid)
    p_iso = isotonic_decreasing(p_int)
    if not np.allclose(p_iso, p_int, atol=1e-12, rtol=0):
        diag["monotone_adjusted_intervals"] = True
    p_iso = np.clip(p_iso, 0.0, 1.0)

    # Interpolate isotonic curve to targets
    p_t = np.interp(Kt, k_mid, p_iso)

    # Target-level isotonic safeguard (ensures monotone even if Kt has weird ordering)
    order = np.argsort(Kt)
    p_sorted = p_t[order]
    p_sorted_iso = isotonic_decreasing(p_sorted)
    if not np.allclose(p_sorted_iso, p_sorted, atol=1e-12, rtol=0):
        diag["monotone_adjusted_targets"] = True
    p_sorted_iso = np.clip(p_sorted_iso, 0.0, 1.0)

    out = np.empty_like(p_t)
    out[order] = p_sorted_iso
    return out, diag


# ----------------------------
# Band selection (adaptive)
# ----------------------------

def _band_strikes_with_abslogm(strikes: np.ndarray, spot: float, abslogm: float, *, cap: int = 0) -> np.ndarray:
    k = np.asarray(strikes, dtype=float)
    k = k[np.isfinite(k) & (k > 0)]
    if k.size == 0 or not np.isfinite(spot) or spot <= 0:
        return np.array([], dtype=float)
    logm = np.log(k / float(spot))
    keep = np.abs(logm) <= float(abslogm)
    k_band = np.sort(np.unique(k[keep]))
    if cap and k_band.size > cap:
        idx = np.argsort(np.abs(np.log(k_band / float(spot))))
        k_band = np.sort(k_band[idx[:cap]])
    return k_band


def pick_band_strikes(
    strikes: np.ndarray,
    spot: float,
    *,
    cfg: Config,
    k_min: float,
    k_max: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    start = float(cfg.max_abs_logm)
    cap = float(cfg.max_abs_logm_cap)
    step = float(cfg.band_widen_step)

    if not cfg.adaptive_band:
        k_band = _band_strikes_with_abslogm(strikes, spot, start, cap=int(cfg.max_band_strikes))
        k_inside = k_band[(k_band > k_min) & (k_band < k_max)]
        return k_band, k_inside, start

    used = start
    while used <= cap + 1e-12:
        k_band = _band_strikes_with_abslogm(strikes, spot, used, cap=int(cfg.max_band_strikes))
        k_inside = k_band[(k_band > k_min) & (k_band < k_max)]
        if int(k_inside.size) >= int(cfg.min_strikes_for_curve):
            return k_band, k_inside, used
        used += step

    used = cap
    k_band = _band_strikes_with_abslogm(strikes, spot, used, cap=int(cfg.max_band_strikes))
    k_inside = k_band[(k_band > k_min) & (k_band < k_max)]
    return k_band, k_inside, used


def strike_spacing_stats(k: np.ndarray) -> Tuple[float, float]:
    k = np.asarray(k, dtype=float)
    if k.size < 3:
        return np.nan, np.nan
    dk = np.diff(np.sort(k))
    dk = dk[np.isfinite(dk) & (dk > 0)]
    if dk.size == 0:
        return np.nan, np.nan
    return float(np.median(dk)), float(np.min(dk))


# ----------------------------
# Soft quality weighting
# ----------------------------

def compute_quality_weight(
    *,
    quote_source: str,
    rel_spread_median: Optional[float],
    prn_adj_intervals: bool,
    prn_adj_targets: bool,
) -> float:
    w = 1.0
    if quote_source == "close_fallback":
        w *= 0.25
    if rel_spread_median is not None and np.isfinite(rel_spread_median):
        w *= 1.0 / (1.0 + float(rel_spread_median))
    if prn_adj_intervals:
        w *= 0.85
    if prn_adj_targets:
        w *= 0.90
    return float(np.clip(w, 0.05, 1.0))


# ----------------------------
# Spot-scale scoring (raw vs split_adj)
# ----------------------------

def _score_spot_scale(*, k_arr: np.ndarray, spot: float, cfg: Config) -> Tuple[float, int, float]:
    if k_arr is None or len(k_arr) < 3 or not np.isfinite(spot) or spot <= 0:
        return -1e18, 0, float(cfg.max_abs_logm)
    k_min = float(np.min(k_arr))
    k_max = float(np.max(k_arr))
    _, k_inside, used = pick_band_strikes(k_arr, float(spot), cfg=cfg, k_min=k_min, k_max=k_max)
    n_inside = int(k_inside.size)
    bonus = 1e6 if n_inside >= int(cfg.min_strikes_for_curve) else 0.0
    return float(bonus + n_inside), n_inside, float(used)


# ----------------------------
# Per-job processing
# ----------------------------

def process_one(
    *,
    theta: ThetaClient,
    cfg: Config,
    ticker: str,
    asof_target: date,     # snapshot day (Mon/Tue/Wed/Thu)
    week_monday: date,
    week_friday: date,
    raw_closes_by_ticker: Dict[str, Dict[date, float]],
    adj_closes_by_ticker: Dict[str, Dict[date, float]],
    split_event_counts: Dict[str, int],
    dividend_histories: Dict[str, DividendHistory],
    option_chain_cache: Dict[Tuple[str, date, date, Optional[int]], pd.DataFrame],
    cache_lock: threading.Lock,
) -> Tuple[List[dict], Optional[dict]]:

    raw_map = raw_closes_by_ticker.get(ticker, {})
    adj_map = adj_closes_by_ticker.get(ticker, {})
    split_n = int(split_event_counts.get(ticker, 0))

    # As-of close (forward fallback from asof_target)
    S0_raw, asof_raw, asof_fwd = get_close_with_fallback_map(
        raw_map, asof_target, direction="forward", max_days=cfg.max_forward_days_for_asof
    )
    S0_adj, asof_adj, _ = get_close_with_fallback_map(
        adj_map, asof_target, direction="forward", max_days=cfg.max_forward_days_for_asof
    )
    if S0_raw is None or asof_raw is None or S0_adj is None or asof_adj is None:
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "missing_S0",
            "detail": f"forward<={cfg.max_forward_days_for_asof}",
        }

    # Expiry close (backward fallback from week_friday for outcome label)
    ST_raw, exp_raw, exp_bwd = get_close_with_fallback_map(
        raw_map, week_friday, direction="backward", max_days=cfg.max_backward_days_for_expiry_close
    )
    ST_adj, exp_adj, _ = get_close_with_fallback_map(
        adj_map, week_friday, direction="backward", max_days=cfg.max_backward_days_for_expiry_close
    )
    if ST_raw is None or exp_raw is None or ST_adj is None or exp_adj is None:
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "missing_ST",
            "detail": f"backward<={cfg.max_backward_days_for_expiry_close}",
        }

    # Keep consistent dates (prefer adjusted date choices)
    asof_used = asof_adj
    expiry_close_used = exp_adj

    # Horizon: always to Friday "event" date (week_friday), not to expiry_close_used fallback.
    # But the option chain itself is for "expiration_used" (Fri or Sat fallback), see below.
    T_days = int((week_friday - asof_used).days)
    if T_days <= 0:
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "bad_T_days",
            "detail": str(T_days),
        }
    T_years = float(T_days) / 365.25

    rv20_raw = realized_vol_proxy(raw_map, asof_used, cfg.rv_lookback_days)
    rv20_adj = realized_vol_proxy(adj_map, asof_used, cfg.rv_lookback_days)

    # Dividends (time-safe trailing window up to asof_used)
    div_sum = np.nan
    div_source_used = "none"
    if cfg.dividend_source == "yfinance":
        hist = dividend_histories.get(ticker)
        if hist is not None and hist.available:
            start_lb = asof_used - timedelta(days=int(cfg.dividend_lookback_days))
            div_sum = float(hist.sum_in_range(start_lb, asof_used))
            div_source_used = "yfinance"
        else:
            div_source_used = "default"
    else:
        div_source_used = "default" if float(cfg.dividend_yield_default) != 0.0 else "none"

    div_sum_annual = np.nan
    if np.isfinite(div_sum) and int(cfg.dividend_lookback_days) > 0:
        div_sum_annual = float(div_sum) * (365.25 / float(cfg.dividend_lookback_days))

    def _div_yield(spot: float) -> float:
        if not np.isfinite(spot) or spot <= 0:
            return float(cfg.dividend_yield_default)
        if not np.isfinite(div_sum_annual):
            return float(cfg.dividend_yield_default)
        return float(div_sum_annual / float(spot))

    div_yield_raw = _div_yield(float(S0_raw))
    div_yield_adj = _div_yield(float(S0_adj))

    session = get_thread_session()

    # --- option expiration (chain fetch) ---
    expiration_requested = week_friday
    expiration_used = week_friday
    expiry_convention = "FRI"

    def _fetch_chain_for_exp(exp: date, strike_range: Optional[int]) -> pd.DataFrame:
        key = (ticker, asof_used, exp, strike_range)
        if not cfg.use_cache:
            return theta.option_eod_chain(session, ticker, asof=asof_used, expiration=exp, right="call", strike_range=strike_range)
        with cache_lock:
            cached = option_chain_cache.get(key)
        if cached is not None:
            return cached
        ch = theta.option_eod_chain(session, ticker, asof=asof_used, expiration=exp, right="call", strike_range=strike_range)
        with cache_lock:
            option_chain_cache[key] = ch
        return ch

    def fetch_chain(strike_range: Optional[int]) -> pd.DataFrame:
        nonlocal expiration_used, expiry_convention
        ch = _fetch_chain_for_exp(week_friday, strike_range)
        if ch is not None and not ch.empty:
            expiration_used = week_friday
            expiry_convention = "FRI"
            return ch
        if cfg.try_saturday_expiry_fallback:
            sat = week_friday + timedelta(days=1)
            ch2 = _fetch_chain_for_exp(sat, strike_range)
            if ch2 is not None and not ch2.empty:
                expiration_used = sat
                expiry_convention = "SAT_FALLBACK"
                return ch2
        return ch

    strike_range_first = None if (cfg.option_strike_range <= 0) else int(cfg.option_strike_range)
    chain = fetch_chain(strike_range_first)

    if chain is None or chain.empty:
        if cfg.retry_full_chain_if_band_thin:
            chain = fetch_chain(None)
        if chain is None or chain.empty:
            return [], {
                "ticker": ticker,
                "week_monday": week_monday.isoformat(),
                "week_friday": week_friday.isoformat(),
                "asof_target": asof_target.isoformat(),
                "drop_reason": "empty_option_chain",
                "detail": f"asof={asof_used.isoformat()} exp_req={expiration_requested.isoformat()} exp_used={expiration_used.isoformat()} conv={expiry_convention}",
            }

    # ---- Build curve twice and choose best spot scale ----
    curve_candidates = []
    q_adj = float(div_yield_adj)
    q_raw = float(div_yield_raw)

    def _forward_price(spot: float, q: float) -> float:
        if not np.isfinite(spot) or spot <= 0:
            return np.nan
        return float(spot) * float(np.exp((float(cfg.risk_free_rate) - float(q)) * float(T_years)))

    for label, spot, q in [
        ("split_adj", float(S0_adj), q_adj),
        ("raw", float(S0_raw), q_raw),
    ]:
        fwd = _forward_price(float(spot), float(q))
        spot_ref = fwd if (cfg.use_forward_moneyness and np.isfinite(fwd) and fwd > 0) else float(spot)
        k_arr, c_arr, diag_curve = build_call_curve_from_eod(
            chain, spot=float(spot), T_years=float(T_years), r=float(cfg.risk_free_rate), q=float(q), cfg=cfg
        )
        if k_arr is None or c_arr is None:
            continue
        score, n_inside, used_abslogm_hint = _score_spot_scale(k_arr=k_arr, spot=float(spot_ref), cfg=cfg)
        n_used = int(diag_curve.get("n_used") or 0)
        curve_candidates.append(
            (score, n_used, label, float(spot), float(q), float(fwd), float(spot_ref), float(used_abslogm_hint), k_arr, c_arr, diag_curve)
        )

    if not curve_candidates:
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "cannot_build_curve",
            "detail": f"asof={asof_used.isoformat()} exp_used={expiration_used.isoformat()}",
        }

    curve_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_score, best_n_used, spot_scale_used, spot_used, q_used, fwd_used, spot_ref_used, _, k_arr, c_arr, diag_curve = curve_candidates[0]
    score_raw = next((x[0] for x in curve_candidates if x[2] == "raw"), -1e18)
    score_adj = next((x[0] for x in curve_candidates if x[2] == "split_adj"), -1e18)

    # Hard filters
    if cfg.hard_drop_close_fallback and diag_curve.get("quote_source") != "bidask_mid":
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "hard_drop_close_fallback",
            "detail": asof_used.isoformat(),
        }

    if cfg.min_chain_used_hard > 0 and int(diag_curve.get("n_used") or 0) < cfg.min_chain_used_hard:
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "hard_min_chain_used",
            "detail": str(diag_curve.get("n_used")),
        }

    rsm = diag_curve.get("rel_spread_median")
    if (rsm is not None) and np.isfinite(rsm) and float(rsm) > cfg.max_rel_spread_median_hard:
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "hard_max_rel_spread_median",
            "detail": str(rsm),
        }

    # Adaptive band
    k_min = float(np.min(k_arr))
    k_max = float(np.max(k_arr))
    spot_ref_used = fwd_used if (cfg.use_forward_moneyness and np.isfinite(fwd_used) and fwd_used > 0) else float(spot_used)
    moneyness_ref = "forward" if (cfg.use_forward_moneyness and np.isfinite(fwd_used) and fwd_used > 0) else "spot"
    k_band, k_band_inside, used_abslogm = pick_band_strikes(
        k_arr, float(spot_ref_used), cfg=cfg, k_min=k_min, k_max=k_max
    )
    n_band_raw = int(k_band.size)
    n_band_inside = int(k_band_inside.size)

    if n_band_inside < cfg.min_strikes_for_curve:
        # optionally retry full chain if band thin and first pass strike_range limited
        if cfg.retry_full_chain_if_band_thin and (strike_range_first is not None):
            chain_full = fetch_chain(None)
            if chain_full is not None and not chain_full.empty:
                k2, c2, d2 = build_call_curve_from_eod(
                    chain_full, spot=float(spot_used), T_years=float(T_years), r=float(cfg.risk_free_rate), q=float(q_used), cfg=cfg
                )
                if k2 is not None and c2 is not None:
                    kmin2, kmax2 = float(np.min(k2)), float(np.max(k2))
                    kb2, ki2, used2 = pick_band_strikes(k2, float(spot_ref_used), cfg=cfg, k_min=kmin2, k_max=kmax2)
                    if int(ki2.size) > n_band_inside:
                        k_arr, c_arr, diag_curve = k2, c2, d2
                        k_min, k_max = kmin2, kmax2
                        k_band, k_band_inside, used_abslogm = kb2, ki2, used2
                        n_band_raw = int(k_band.size)
                        n_band_inside = int(k_band_inside.size)

    if n_band_inside < cfg.min_strikes_for_curve:
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "thin_band_inside",
            "detail": f"inside={n_band_inside} raw={n_band_raw} spot_scale={spot_scale_used} used_abslogm={used_abslogm:.4f} moneyness_ref={moneyness_ref} spot_ref={spot_ref_used:.6f} spot={spot_used:.6f} splits_preload={split_n}",
        }

    # pRN on strikes in band
    pRN, diag_prn = compute_prn_from_call_curve(
        k_arr, c_arr, K_targets=k_band_inside, T_years=float(T_years), r=float(cfg.risk_free_rate)
    )
    if pRN is None or len(pRN) != len(k_band_inside):
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "bad_prn_compute",
            "detail": asof_used.isoformat(),
        }

    pRN_raw_targets = diag_prn.get("p_targets_raw", None)

    # Choose prices/outcomes consistent with chosen spot scale
    if spot_scale_used == "raw":
        S0_used = float(S0_raw)
        ST_used = float(ST_raw)
        rv20_used = rv20_raw
        div_yield_used = float(div_yield_raw)
    else:
        S0_used = float(S0_adj)
        ST_used = float(ST_adj)
        rv20_used = rv20_adj
        div_yield_used = float(div_yield_adj)

    forward_used = float(fwd_used) if (np.isfinite(fwd_used) and fwd_used > 0) else _forward_price(S0_used, div_yield_used)

    # Rows + apply pRN band
    tmp_rows: List[dict] = []
    for i, (K, p) in enumerate(zip(k_band_inside, pRN)):
        if not np.isfinite(p):
            continue
        p = fix_negative_zero(float(np.round(float(p), 7)))
        if not (cfg.min_prn_train <= p <= cfg.max_prn_train):
            continue

        p_raw = np.nan
        if pRN_raw_targets is not None and i < len(pRN_raw_targets):
            pr = float(pRN_raw_targets[i])
            p_raw = float(np.round(pr, 7)) if np.isfinite(pr) else np.nan

        tmp_rows.append(
            {
                # identity
                "ticker": ticker,
                "week_monday": week_monday.isoformat(),
                "week_friday": week_friday.isoformat(),
                "asof_target": asof_target.isoformat(),
                "asof_date": asof_used.isoformat(),
                "expiry_close_date_used": expiry_close_used.isoformat(),

                # option chain expiries (requested Fri, maybe Sat fallback)
                "option_expiration_requested": expiration_requested.isoformat(),
                "option_expiration_used": expiration_used.isoformat(),
                "expiry_convention": expiry_convention,

                # horizons (to Friday)
                "T_days": int(T_days),
                "T_years": float(np.round(float(T_years), 9)),
                "r": float(cfg.risk_free_rate),

                # raw/adj audit
                "S_asof_close_raw": float(np.round(float(S0_raw), 7)),
                "S_expiry_close_raw": float(np.round(float(ST_raw), 7)),
                "S_asof_close_adj": float(np.round(float(S0_adj), 7)),
                "S_expiry_close_adj": float(np.round(float(ST_adj), 7)),
                "split_events_in_preload_range": split_n,
                "split_adjustment_applied": bool(cfg.apply_split_adjustment),

                # dividends / forward (time-safe trailing proxy)
                "dividend_source_used": div_source_used,
                "dividend_lookback_days": int(cfg.dividend_lookback_days),
                "dividend_sum_lookback": float(np.round(div_sum, 8)) if np.isfinite(div_sum) else np.nan,
                "dividend_yield_raw": float(np.round(div_yield_raw, 8)) if np.isfinite(div_yield_raw) else np.nan,
                "dividend_yield_adj": float(np.round(div_yield_adj, 8)) if np.isfinite(div_yield_adj) else np.nan,

                # chosen scale
                "spot_scale_used": spot_scale_used,
                "spot_scale_score_raw": float(score_raw),
                "spot_scale_score_adj": float(score_adj),

                # used (consistent with chosen scale)
                "S_asof_close": float(np.round(float(S0_used), 7)),
                "S_expiry_close": float(np.round(float(ST_used), 7)),
                "dividend_yield": float(np.round(div_yield_used, 8)) if np.isfinite(div_yield_used) else np.nan,
                "forward_price": float(np.round(forward_used, 7)) if np.isfinite(forward_used) else np.nan,

                "asof_fallback_days": int(asof_fwd),
                "expiry_fallback_days": int(exp_bwd),

                # strike + moneyness
                "K": float(np.round(float(K), 7)),
                "log_m": float(np.round(np.log(float(K) / float(S0_used)), 9)),
                "abs_log_m": float(np.round(abs(np.log(float(K) / float(S0_used))), 9)),
                "log_m_fwd": float(np.round(np.log(float(K) / float(forward_used)), 9)) if np.isfinite(forward_used) and forward_used > 0 else np.nan,
                "abs_log_m_fwd": float(np.round(abs(np.log(float(K) / float(forward_used))), 9)) if np.isfinite(forward_used) and forward_used > 0 else np.nan,

                # vol proxy
                "rv20": float(np.round(rv20_used, 8)) if np.isfinite(rv20_used) else np.nan,

                # pRN (+ audit raw targets)
                "pRN": p,
                "qRN": float(np.round(1.0 - p, 7)),
                "pRN_raw": p_raw,
                "qRN_raw": float(np.round(1.0 - p_raw, 7)) if np.isfinite(p_raw) else np.nan,

                # realized outcome label (close used at/near Friday)
                "outcome_ST_gt_K": 1 if float(ST_used) > float(K) else 0,

                # band diagnostics
                "max_abs_logm_start": float(cfg.max_abs_logm),
                "max_abs_logm_cap": float(cfg.max_abs_logm_cap),
                "used_max_abs_logm": float(np.round(used_abslogm, 6)),
                "n_band_raw": int(n_band_raw),
                "n_band_inside": int(n_band_inside),
                "moneyness_ref": moneyness_ref,
                "moneyness_ref_price": float(np.round(float(spot_ref_used), 7)) if np.isfinite(spot_ref_used) else np.nan,
                "calls_k_min": float(np.round(k_min, 7)),
                "calls_k_max": float(np.round(k_max, 7)),

                # quote/curve diagnostics
                "theta_quote_source": diag_curve.get("quote_source"),
                "n_chain_raw": diag_curve.get("n_raw"),
                "n_chain_used": diag_curve.get("n_used"),
                "rel_spread_median": diag_curve.get("rel_spread_median"),
                "dropped_liquidity": diag_curve.get("dropped_liquidity"),
                "dropped_intrinsic": diag_curve.get("dropped_intrinsic"),
                "dropped_insane": diag_curve.get("dropped_insane"),
                "prn_monotone_adj_intervals": bool(diag_prn.get("monotone_adjusted_intervals")),
                "prn_monotone_adj_targets": bool(diag_prn.get("monotone_adjusted_targets")),
            }
        )

    if len(tmp_rows) < int(cfg.min_strikes_in_prn_band):
        return [], {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "too_few_in_prn_band",
            "detail": f"kept={len(tmp_rows)} need={cfg.min_strikes_in_prn_band} inside={n_band_inside} used_abslogm={used_abslogm:.4f} spot_scale={spot_scale_used} moneyness_ref={moneyness_ref}",
        }

    # Group id (per ticker + snapshot day + week)
    group_id = f"{ticker}|{asof_used.isoformat()}|{week_friday.isoformat()}"
    med_dk, min_dk = strike_spacing_stats(np.array([r["K"] for r in tmp_rows], dtype=float))

    for rr in tmp_rows:
        rr["group_id"] = group_id
        rr["median_dK"] = float(np.round(med_dk, 6)) if np.isfinite(med_dk) else np.nan
        rr["min_dK"] = float(np.round(min_dk, 6)) if np.isfinite(min_dk) else np.nan

        if cfg.use_soft_quality_weight:
            rr["quality_weight"] = compute_quality_weight(
                quote_source=str(rr.get("theta_quote_source") or ""),
                rel_spread_median=rr.get("rel_spread_median"),
                prn_adj_intervals=bool(rr.get("prn_monotone_adj_intervals")),
                prn_adj_targets=bool(rr.get("prn_monotone_adj_targets")),
            )
        else:
            rr["quality_weight"] = 1.0

    return tmp_rows, None


# ----------------------------
# Optional sanity report/drop (group-level)
# ----------------------------

def _sanity_report_and_optional_drop(out_df: pd.DataFrame, drops: List[dict], cfg: Config) -> pd.DataFrame:
    if out_df is None or out_df.empty:
        return out_df
    if "group_id" not in out_df.columns:
        return out_df
    needed = {"K", "S_asof_close", "abs_log_m", "ticker", "week_monday", "week_friday", "asof_date"}
    if not needed.issubset(set(out_df.columns)):
        return out_df

    tmp = out_df.copy()
    tmp["K"] = pd.to_numeric(tmp["K"], errors="coerce")
    tmp["S_asof_close"] = pd.to_numeric(tmp["S_asof_close"], errors="coerce")
    tmp["abs_log_m"] = pd.to_numeric(tmp["abs_log_m"], errors="coerce")
    tmp["K_over_S"] = tmp["K"] / tmp["S_asof_close"]
    tmp = tmp.replace([np.inf, -np.inf], np.nan)

    g = tmp.groupby("group_id").agg(
        ticker=("ticker", "first"),
        week_monday=("week_monday", "first"),
        week_friday=("week_friday", "first"),
        asof_date=("asof_date", "first"),
        n=("K", "count"),
        med_abs_log_m=("abs_log_m", "median"),
        med_K_over_S=("K_over_S", "median"),
        min_K_over_S=("K_over_S", "min"),
        max_K_over_S=("K_over_S", "max"),
    ).reset_index()
    g["med_K_over_S_dist1"] = (g["med_K_over_S"] - 1.0).abs()

    if cfg.sanity_report:
        worst_abs = g.sort_values("med_abs_log_m", ascending=False).head(5)
        worst_ks = g.sort_values("med_K_over_S_dist1", ascending=False).head(5)
        print("\n[SANITY] Worst 5 groups by median abs_log_m:")
        for _, row in worst_abs.iterrows():
            print(
                f"  {row['group_id']} | {row['ticker']} | asof={row['asof_date']} | {row['week_monday']}→{row['week_friday']} | "
                f"n={int(row['n'])} | med_abs_log_m={row['med_abs_log_m']:.4f} | "
                f"med_K/S={row['med_K_over_S']:.4f} | K/S=[{row['min_K_over_S']:.4f},{row['max_K_over_S']:.4f}]"
            )
        print("\n[SANITY] Worst 5 groups by |median(K/S)-1|:")
        for _, row in worst_ks.iterrows():
            print(
                f"  {row['group_id']} | {row['ticker']} | asof={row['asof_date']} | {row['week_monday']}→{row['week_friday']} | "
                f"n={int(row['n'])} | med_K/S={row['med_K_over_S']:.4f} (dist={row['med_K_over_S_dist1']:.4f}) | "
                f"med_abs_log_m={row['med_abs_log_m']:.4f}"
            )
        print("")

    if not cfg.sanity_drop:
        return out_df

    bad_groups = set(
        g[(~np.isfinite(g["med_abs_log_m"])) | (g["med_abs_log_m"] > float(cfg.sanity_abs_logm_max))]["group_id"].tolist()
    ) | set(
        g[
            (~np.isfinite(g["med_K_over_S"]))
            | (g["med_K_over_S"] < float(cfg.sanity_k_over_s_min))
            | (g["med_K_over_S"] > float(cfg.sanity_k_over_s_max))
        ]["group_id"].tolist()
    )

    if not bad_groups:
        return out_df

    bad_mask = out_df["group_id"].isin(bad_groups)
    print(f"[SANITY] Dropping obviously broken groups: groups={len(bad_groups)} rows={int(bad_mask.sum())}")

    for gid in sorted(bad_groups):
        row = g[g["group_id"] == gid].head(1)
        if len(row) == 1:
            r = row.iloc[0]
            drops.append(
                {
                    "ticker": str(r["ticker"]),
                    "week_monday": str(r["week_monday"]),
                    "week_friday": str(r["week_friday"]),
                    "asof_date": str(r["asof_date"]),
                    "drop_reason": "sanity_bad_strike_or_scale",
                    "detail": f"group_id={gid} med_abs_log_m={float(r['med_abs_log_m']):.4f} med_K_over_S={float(r['med_K_over_S']):.4f}",
                }
            )

    return out_df.loc[~bad_mask].reset_index(drop=True)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--out-dir", type=str, default="./data/raw/option-chain")
    ap.add_argument("--out-name", type=str, default="pRN__history__mon_thu__PM10__v1.6.0.csv")

    ap.add_argument("--tickers", type=str, default=",".join(PM10_TICKERS))
    ap.add_argument("--start", type=str, required=True, help="YYYY-MM-DD (range used to generate Mondays)")
    ap.add_argument("--end", type=str, required=True, help="YYYY-MM-DD (range used to generate Mondays)")

    ap.add_argument("--theta-base-url", type=str, default=Config().theta_base_url)
    ap.add_argument("--stock-source", type=str, default=Config().stock_source, choices=["yfinance", "theta", "auto"])
    ap.add_argument("--timeout-s", type=int, default=Config().timeout_s)
    ap.add_argument("--r", type=float, default=Config().risk_free_rate)

    # Band + thresholds
    ap.add_argument("--max-abs-logm", type=float, default=Config().max_abs_logm)
    ap.add_argument("--max-abs-logm-cap", type=float, default=Config().max_abs_logm_cap)
    ap.add_argument("--band-widen-step", type=float, default=Config().band_widen_step)
    ap.add_argument("--no-adaptive-band", action="store_true")
    ap.add_argument("--max-band-strikes", type=int, default=Config().max_band_strikes)

    ap.add_argument("--min-band-strikes", type=int, default=Config().min_strikes_for_curve)
    ap.add_argument("--min-band-prn-strikes", type=int, default=Config().min_strikes_in_prn_band)

    # Option chain / expiry
    ap.add_argument("--strike-range", type=int, default=Config().option_strike_range)
    ap.add_argument("--no-retry-full-chain", action="store_true")
    ap.add_argument("--no-sat-expiry-fallback", action="store_true")
    ap.add_argument("--threads", type=int, default=6)

    # Quote/liquidity
    ap.add_argument("--prefer-bidask", action=argparse.BooleanOptionalAction, default=Config().prefer_bidask)
    ap.add_argument("--min-trade-count", type=int, default=0)
    ap.add_argument("--min-volume", type=int, default=0)

    # Hard filters
    ap.add_argument("--min-chain-used-hard", type=int, default=0)
    ap.add_argument("--max-rel-spread-median-hard", type=float, default=1e9)
    ap.add_argument("--hard-drop-close-fallback", action="store_true")

    # Training band
    ap.add_argument("--min-prn-train", type=float, default=0.10)
    ap.add_argument("--max-prn-train", type=float, default=0.90)

    # Split adjustment
    ap.add_argument("--no-split-adjust", action="store_true")

    # Dividends / forward
    ap.add_argument("--dividend-source", type=str, default=Config().dividend_source, choices=["yfinance", "none"])
    ap.add_argument("--dividend-lookback-days", type=int, default=Config().dividend_lookback_days)
    ap.add_argument("--dividend-yield-default", type=float, default=Config().dividend_yield_default)
    ap.add_argument("--no-forward-moneyness", action="store_true")

    # Weights
    ap.add_argument("--no-group-weights", action="store_true")
    ap.add_argument("--no-ticker-weights", action="store_true")
    ap.add_argument("--no-soft-quality-weight", action="store_true")

    # Vol proxy
    ap.add_argument("--rv-lookback-days", type=int, default=20)

    # Cache
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=Config().use_cache)

    # Drops
    ap.add_argument("--write-drops", action="store_true")
    ap.add_argument("--drops-name", type=str, default="pRN__history__mon_thu__drops__v1.6.0.csv")

    # Sanity
    ap.add_argument("--sanity-report", action="store_true")
    ap.add_argument("--sanity-drop", action="store_true")
    ap.add_argument("--sanity-abs-logm-max", type=float, default=Config().sanity_abs_logm_max)
    ap.add_argument("--sanity-k-over-s-min", type=float, default=Config().sanity_k_over_s_min)
    ap.add_argument("--sanity-k-over-s-max", type=float, default=Config().sanity_k_over_s_max)

    ap.add_argument("--verbose-skips", action="store_true")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided.")

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        raise SystemExit("--end must be >= --start")

    cfg = Config(
        theta_base_url=str(args.theta_base_url),
        timeout_s=int(args.timeout_s),
        risk_free_rate=float(args.r),

        option_strike_range=int(args.strike_range),
        retry_full_chain_if_band_thin=(not bool(args.no_retry_full_chain)),
        try_saturday_expiry_fallback=(not bool(args.no_sat_expiry_fallback)),

        max_abs_logm=float(args.max_abs_logm),
        max_abs_logm_cap=float(args.max_abs_logm_cap),
        band_widen_step=float(args.band_widen_step),
        adaptive_band=(not bool(args.no_adaptive_band)),
        max_band_strikes=int(args.max_band_strikes),

        min_strikes_for_curve=int(args.min_band_strikes),
        min_strikes_in_prn_band=int(args.min_band_prn_strikes),

        prefer_bidask=bool(args.prefer_bidask),
        min_trade_count=int(args.min_trade_count),
        min_volume=int(args.min_volume),

        min_chain_used_hard=int(args.min_chain_used_hard),
        max_rel_spread_median_hard=float(args.max_rel_spread_median_hard),
        hard_drop_close_fallback=bool(args.hard_drop_close_fallback),

        min_prn_train=float(args.min_prn_train),
        max_prn_train=float(args.max_prn_train),

        apply_split_adjustment=(not bool(args.no_split_adjust)),

        dividend_source=str(args.dividend_source),
        dividend_lookback_days=int(args.dividend_lookback_days),
        dividend_yield_default=float(args.dividend_yield_default),
        use_forward_moneyness=(not bool(args.no_forward_moneyness)),

        add_group_weights=(not bool(args.no_group_weights)),
        add_ticker_weights=(not bool(args.no_ticker_weights)),
        use_soft_quality_weight=(not bool(args.no_soft_quality_weight)),

        rv_lookback_days=int(args.rv_lookback_days),

        use_cache=bool(args.cache),

        stock_source=str(args.stock_source),

        sanity_report=bool(args.sanity_report),
        sanity_drop=bool(args.sanity_drop),
        sanity_abs_logm_max=float(args.sanity_abs_logm_max),
        sanity_k_over_s_min=float(args.sanity_k_over_s_min),
        sanity_k_over_s_max=float(args.sanity_k_over_s_max),
    )

    # basic parameter guards
    if not (0.0 < cfg.min_prn_train < cfg.max_prn_train < 1.0):
        raise SystemExit("Require 0 < --min-prn-train < --max-prn-train < 1")
    if cfg.max_abs_logm_cap < cfg.max_abs_logm:
        raise SystemExit("--max-abs-logm-cap must be >= --max-abs-logm")
    if cfg.band_widen_step <= 0:
        raise SystemExit("--band-widen-step must be > 0")
    if cfg.min_strikes_for_curve < 3:
        print("[WARN] min_strikes_for_curve < 3 is likely too low.")
    if cfg.min_strikes_in_prn_band < 1:
        raise SystemExit("--min-band-prn-strikes must be >= 1")
    if cfg.dividend_lookback_days <= 0:
        raise SystemExit("--dividend-lookback-days must be > 0")
    if cfg.dividend_source == "yfinance" and yf is None:
        raise SystemExit("dividend_source=yfinance requires yfinance installed.")

    theta = ThetaClient(cfg.theta_base_url, timeout_s=cfg.timeout_s, verbose=bool(args.verbose_skips))

    def _clean_args_for_run_dir(argv: List[str]) -> List[str]:
        cleaned: List[str] = []
        skip_next = False
        for a in argv:
            if skip_next:
                skip_next = False
                continue
            if a in {"--out-dir", "--out-name"}:
                skip_next = True
                continue
            if a.startswith("--out-dir=") or a.startswith("--out-name="):
                continue
            cleaned.append(a)
        return cleaned

    def _slugify_args(argv: List[str], max_len: int = 140) -> str:
        raw = " ".join(argv).strip()
        if not raw:
            return ""
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._=-")
        out = []
        prev_us = False
        for ch in raw:
            if ch in allowed:
                out.append(ch)
                prev_us = False
            else:
                if not prev_us:
                    out.append("_")
                    prev_us = True
        slug = "".join(out).strip("_")
        if len(slug) <= max_len:
            return slug
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
        keep = max_len - len("__sha1=") - len(digest)
        return f"{slug[:keep]}__sha1={digest}"

    run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    args_for_slug = _clean_args_for_run_dir(sys.argv[1:])
    args_slug = _slugify_args(args_for_slug)
    run_dir_name = f"start={start.isoformat()}__end={end.isoformat()}"
    if args_slug:
        run_dir_name = f"{run_dir_name}__args={args_slug}"
    run_dir_name = f"{run_dir_name}__run={run_ts}"

    out_base_dir = args.out_dir
    os.makedirs(out_base_dir, exist_ok=True)
    run_dir = os.path.join(out_base_dir, run_dir_name)
    os.makedirs(run_dir, exist_ok=False)

    out_path = os.path.join(run_dir, args.out_name)
    drops_path = os.path.join(run_dir, args.drops_name)

    mondays = mondays_in_range(start, end)
    print(f"[PLAN] Weeks={len(mondays)} (Mondays) range={start}..{end} tickers={len(tickers)} snapshots/day=4 (Mon-Thu)")
    print(f"[UNIVERSE] {tickers}")
    print(f"[OUT] base={out_base_dir} run_dir={run_dir}")
    print(f"[CFG] pRN train band: [{cfg.min_prn_train}, {cfg.max_prn_train}]")
    print(f"[CFG] band start={cfg.max_abs_logm} cap={cfg.max_abs_logm_cap} step={cfg.band_widen_step} adaptive={cfg.adaptive_band}")
    print(f"[CFG] min strikes: curve={cfg.min_strikes_for_curve} after_pRN={cfg.min_strikes_in_prn_band}")
    print(f"[CFG] expiry fallback Fri->Sat: {cfg.try_saturday_expiry_fallback}")
    print(f"[CFG] split_adjustment: {cfg.apply_split_adjustment}")
    print(
        f"[CFG] dividend_source={cfg.dividend_source} lookback_days={cfg.dividend_lookback_days} "
        f"default_yield={cfg.dividend_yield_default} use_forward_moneyness={cfg.use_forward_moneyness}"
    )
    print(f"[CFG] prefer_bidask: {cfg.prefer_bidask}")
    print(f"[CFG] threads={args.threads} cache={cfg.use_cache} stock_source={cfg.stock_source}")
    if cfg.sanity_report or cfg.sanity_drop:
        print(
            f"[CFG] sanity_report={cfg.sanity_report} sanity_drop={cfg.sanity_drop} "
            f"abs_logm_max={cfg.sanity_abs_logm_max} K/S in [{cfg.sanity_k_over_s_min},{cfg.sanity_k_over_s_max}]"
        )

    # Preload stock closes for the whole range + a small cushion (because we also use Thu and exp-close fallback)
    preload_start = start
    preload_end = end + timedelta(days=4)
    print(f"[STOCK] Preloading closes for {preload_start}..{preload_end} ...")
    raw_closes_by_ticker, adj_closes_by_ticker, split_counts = preload_stock_closes(
        theta=theta,
        tickers=tickers,
        start=preload_start,
        end=preload_end,
        cfg=cfg,
        stock_source=args.stock_source,
    )

    missing = [t for t in tickers if len(raw_closes_by_ticker.get(t, {})) == 0]
    if missing:
        print(f"[STOCK] ⚠️ No close data for: {missing}")

    dividend_histories = preload_dividend_histories(
        tickers=tickers,
        start=preload_start,
        end=preload_end,
        cfg=cfg,
    )

    option_chain_cache: Dict[Tuple[str, date, date, Optional[int]], pd.DataFrame] = {}
    cache_lock = threading.Lock()

    rows: List[dict] = []
    drops: List[dict] = []

    # Jobs = for each week, snapshot days Mon/Tue/Wed/Thu, for each ticker (expiry always that week's Friday)
    def jobs():
        for mon in mondays:
            fri = iso_week_friday(mon)
            for asof_target in asof_days_mon_to_thu(mon):
                for t in tickers:
                    yield (t, mon, fri, asof_target)

    total_jobs = len(mondays) * 4 * len(tickers)
    done = 0
    kept_groups = 0

    with ThreadPoolExecutor(max_workers=max(1, int(args.threads))) as ex:
        futures = {
            ex.submit(
                process_one,
                theta=theta,
                cfg=cfg,
                ticker=t,
                asof_target=asof_target,
                week_monday=mon,
                week_friday=fri,
                raw_closes_by_ticker=raw_closes_by_ticker,
                adj_closes_by_ticker=adj_closes_by_ticker,
                split_event_counts=split_counts,
                dividend_histories=dividend_histories,
                option_chain_cache=option_chain_cache,
                cache_lock=cache_lock,
            ): (t, mon, asof_target)
            for (t, mon, fri, asof_target) in jobs()
        }

        for fut in as_completed(futures):
            t, mon, asof_target = futures[fut]
            done += 1
            try:
                rws, drop_log = fut.result()
            except Exception as e:
                rws, drop_log = [], {
                    "ticker": t,
                    "week_monday": mon.isoformat(),
                    "week_friday": iso_week_friday(mon).isoformat(),
                    "asof_target": asof_target.isoformat(),
                    "drop_reason": "exception",
                    "detail": str(e),
                }
                if args.verbose_skips:
                    print(f"[ERR] {t} week={mon} asof_target={asof_target}: {e}")

            if rws:
                kept_groups += 1
                rows.extend(rws)
            if drop_log is not None:
                drops.append(drop_log)

            if done % 100 == 0 or done == total_jobs:
                print(
                    f"[PROGRESS] {done}/{total_jobs} jobs | groups_kept={kept_groups} | rows={len(rows)} | last={t} week={mon} asof_target={asof_target}"
                )

    if not rows:
        print("[RESULT] No rows produced.")
        if args.write_drops and drops:
            pd.DataFrame(drops).to_csv(drops_path, index=False)
            print(f"[WRITE] drops: {drops_path}")
        return

    out_df = pd.DataFrame(rows)

    # parse + sort
    out_df["asof_date"] = pd.to_datetime(out_df["asof_date"], errors="coerce")
    out_df = out_df.dropna(subset=["asof_date"]).copy()
    out_df = out_df.sort_values(["asof_date", "ticker", "K"]).reset_index(drop=True)

    # optional sanity (report and/or drop)
    out_df = _sanity_report_and_optional_drop(out_df, drops, cfg)

    # Group weights: each group sums to ~1 across its rows
    if cfg.add_group_weights and "group_id" in out_df.columns:
        gsize = out_df.groupby("group_id")["K"].transform("count").astype(float)
        out_df["group_size"] = gsize
        out_df["group_weight"] = (1.0 / gsize).astype(float)
    else:
        out_df["group_size"] = np.nan
        out_df["group_weight"] = 1.0

    # Ticker weights: each ticker sums to ~1 across its groups
    if cfg.add_ticker_weights and "group_id" in out_df.columns:
        gcount = out_df.groupby("ticker")["group_id"].nunique()
        out_df["ticker_group_count"] = out_df["ticker"].map(gcount).astype(float)
        out_df["ticker_weight"] = (1.0 / out_df["ticker_group_count"]).astype(float)
    else:
        out_df["ticker_group_count"] = np.nan
        out_df["ticker_weight"] = 1.0

    # Final sample weight
    qw = pd.to_numeric(out_df.get("quality_weight", 1.0), errors="coerce").fillna(1.0).clip(0.01, 1.0)
    out_df["quality_weight"] = qw
    out_df["sample_weight_final"] = (out_df["group_weight"] * out_df["ticker_weight"] * out_df["quality_weight"]).astype(float)
    out_df["sample_weight_final"] = (
        pd.to_numeric(out_df["sample_weight_final"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Summary
    weeks = out_df["asof_date"].dt.to_period("W-MON").nunique()
    groups = out_df["group_id"].nunique() if "group_id" in out_df.columns else np.nan
    print(f"[SUMMARY] rows={len(out_df)} tickers={out_df['ticker'].nunique()} weeks={weeks} groups={groups}")
    if "group_id" in out_df.columns:
        print("[SUMMARY] per-ticker groups:")
        print(out_df.groupby("ticker")["group_id"].nunique().sort_values(ascending=False).to_string())

    out_df.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path}")

    if args.write_drops:
        drops_df = pd.DataFrame(drops) if drops else pd.DataFrame(columns=["ticker", "week_monday", "week_friday", "asof_target", "drop_reason", "detail"])
        drops_df.to_csv(drops_path, index=False)
        print(f"[WRITE] drops: {drops_path}")


if __name__ == "__main__":
    main()
