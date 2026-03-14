from __future__ import annotations

import bisect
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None

PM10_TICKERS = ["AAPL", "GOOGL", "MSFT", "META", "AMZN", "PLTR", "NVDA", "NFLX", "OPEN", "TSLA"]
RN_METHOD = "breeden_litzenberger_call_curve"
DEFAULT_THREADS = 6
DEFAULT_PRN_VERSION = "v1"
DEFAULT_PRN_ASOF_TZ = "America/New_York"
DEFAULT_PRN_ASOF_CLOSE_TIME = "16:00"
QUALITY_REL_SPREAD_WARN = 0.25
QUALITY_CHAIN_USED_WARN = 12
POLYMARKET_EXACT_SCHEMA_VERSION = "pm_run_local_prn_exact_v1.0"
POLYMARKET_EXACT_COLUMNS = [
    "row_id",
    "asof_ts",
    "asof_time",
    "option_type",
    "ticker",
    "market_id",
    "event_id",
    "week_monday",
    "week_friday",
    "asof_target",
    "asof_date",
    "snapshot_date",
    "expiry_close_date_used",
    "option_expiration_requested",
    "option_expiration_used",
    "expiry_convention",
    "expiry_date",
    "event_endDate",
    "T_days",
    "T_years",
    "r",
    "K",
    "threshold",
    "pRN",
    "qRN",
    "pRN_raw",
    "qRN_raw",
    "rv20",
    "log_m",
    "abs_log_m",
    "log_m_fwd",
    "abs_log_m_fwd",
    "S_asof_close",
    "forward_price",
    "dividend_yield",
    "rn_method",
    "theta_quote_source",
    "prn_config_hash",
    "prn_version",
    "coverage_status",
    "drop_reason",
    "schema_version",
]


@dataclass
class Config:
    theta_base_url: str = "http://127.0.0.1:25503/v3"
    timeout_s: int = 30
    risk_free_rate: float = 0.03
    option_strike_range: int = 60
    retry_full_chain_if_band_thin: bool = True
    try_saturday_expiry_fallback: bool = True
    max_forward_days_for_asof: int = 3
    max_backward_days_for_expiry_close: int = 3
    max_abs_logm: float = 0.06
    max_abs_logm_cap: float = 0.10
    band_widen_step: float = 0.01
    adaptive_band: bool = True
    max_band_strikes: int = 0
    min_strikes_for_curve: int = 10
    min_strikes_in_prn_band: int = 7
    rel_spread_max_per_strike: float = 2.0
    intrinsic_tol: float = 0.98
    insane_price_multiple: float = 1.5
    prefer_bidask: bool = True
    min_trade_count: int = 0
    min_volume: int = 0
    min_chain_used_hard: int = 0
    max_rel_spread_median_hard: float = 1e9
    hard_drop_close_fallback: bool = False
    min_prn_train: float = 0.10
    max_prn_train: float = 0.90
    stock_source: str = "yfinance"
    stock_preload_buffer_days: int = 7
    apply_split_adjustment: bool = True
    split_source: str = "yfinance"
    dividend_source: str = "yfinance"
    dividend_lookback_days: int = 365
    dividend_yield_default: float = 0.0
    use_forward_moneyness: bool = True
    ticker_reweight_mode: str = "none"
    ticker_reweight_alpha_min: float = 0.5
    ticker_reweight_alpha_max: float = 2.0
    trade_focus_beta: float = 1.0
    trade_focus_tickers: Tuple[str, ...] = tuple(PM10_TICKERS)
    use_cache: bool = True
    rv_lookback_days: int = 20
    sanity_report: bool = False
    sanity_drop: bool = False
    sanity_abs_logm_max: float = 0.40
    sanity_k_over_s_min: float = 0.25
    sanity_k_over_s_max: float = 4.0


@dataclass
class DividendHistory:
    dates: List[date]
    cumsum: np.ndarray
    available: bool = True

    def sum_in_range(self, start: date, end: date) -> float:
        if not self.available or not self.dates:
            return 0.0
        left = bisect.bisect_left(self.dates, start)
        right = bisect.bisect_right(self.dates, end)
        if right <= left:
            return 0.0
        total = float(self.cumsum[right - 1])
        if left > 0:
            total -= float(self.cumsum[left - 1])
        return total


@dataclass
class PolymarketExactPrnBuildResult:
    rows: pd.DataFrame
    drop_logs: List[dict]
    required_market_snapshots: int
    ok_market_snapshots: int
    missing_market_snapshots: int
    coverage_counts: Dict[str, int]
    drop_reason_counts: Dict[str, int]
    prn_version: str
    prn_config_hash: str


def yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def iso_ts(d: date) -> str:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def compute_row_id(
    *,
    asof_ts: str,
    ticker: str,
    expiry_date_used: str,
    strike: float,
    option_type: str,
) -> str:
    payload = f"{asof_ts}|{ticker}|{expiry_date_used}|{strike:.7f}|{option_type}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def compute_prn_config_hash(cfg: Config) -> str:
    payload = {
        "risk_free_rate": cfg.risk_free_rate,
        "option_strike_range": cfg.option_strike_range,
        "retry_full_chain_if_band_thin": cfg.retry_full_chain_if_band_thin,
        "try_saturday_expiry_fallback": cfg.try_saturday_expiry_fallback,
        "max_abs_logm": cfg.max_abs_logm,
        "max_abs_logm_cap": cfg.max_abs_logm_cap,
        "band_widen_step": cfg.band_widen_step,
        "adaptive_band": cfg.adaptive_band,
        "max_band_strikes": cfg.max_band_strikes,
        "min_strikes_for_curve": cfg.min_strikes_for_curve,
        "min_strikes_in_prn_band": cfg.min_strikes_in_prn_band,
        "rel_spread_max_per_strike": cfg.rel_spread_max_per_strike,
        "intrinsic_tol": cfg.intrinsic_tol,
        "insane_price_multiple": cfg.insane_price_multiple,
        "prefer_bidask": cfg.prefer_bidask,
        "min_trade_count": cfg.min_trade_count,
        "min_volume": cfg.min_volume,
        "use_forward_moneyness": cfg.use_forward_moneyness,
        "dividend_source": cfg.dividend_source,
        "dividend_lookback_days": cfg.dividend_lookback_days,
        "dividend_yield_default": cfg.dividend_yield_default,
        "stock_source": cfg.stock_source,
        "apply_split_adjustment": cfg.apply_split_adjustment,
    }
    raw = "|".join(f"{key}={payload[key]}" for key in sorted(payload))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def iso_week_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def iso_week_friday(d: date) -> date:
    return iso_week_monday(d) + timedelta(days=4)


_WEEKDAY_MAP = {
    "mon": 0,
    "monday": 0,
    "tue": 1,
    "tues": 1,
    "tuesday": 1,
    "wed": 2,
    "weds": 2,
    "wednesday": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "thursday": 3,
    "fri": 4,
    "friday": 4,
    "sat": 5,
    "saturday": 5,
    "sun": 6,
    "sunday": 6,
}


def parse_weekdays(raw: str, *, label: str) -> List[int]:
    if raw is None:
        return []
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not parts:
        return []
    out: List[int] = []
    seen = set()
    for part in parts:
        if part.isdigit():
            weekday = int(part)
        else:
            key = part[:3]
            if key not in _WEEKDAY_MAP:
                raise SystemExit(f"{label} must use weekdays like mon,tue,... or 0..6 (0=Mon). Got: {part}")
            weekday = _WEEKDAY_MAP[key]
        if weekday < 0 or weekday > 6:
            raise SystemExit(f"{label} weekday out of range 0..6: {weekday}")
        if weekday in seen:
            continue
        seen.add(weekday)
        out.append(weekday)
    return out


def parse_dte_list(raw: str) -> List[int]:
    if raw is None:
        return []
    stripped = raw.strip()
    if not stripped:
        return []
    out: List[int] = []
    for part in stripped.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_raw, end_raw = item.split("-", 1)
            start_val = int(start_raw.strip())
            end_val = int(end_raw.strip())
            if end_val < start_val:
                raise SystemExit(f"dte range must be ascending: {item}")
            out.extend(list(range(start_val, end_val + 1)))
        else:
            out.append(int(item))
    deduped: List[int] = []
    seen = set()
    for value in out:
        if value < 0:
            raise SystemExit(f"dte values must be >= 0; got {value}")
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def resolve_dte_list(
    *,
    dte_list_raw: str,
    dte_min: Optional[int],
    dte_max: Optional[int],
    dte_step: int,
) -> List[int]:
    parsed = parse_dte_list(dte_list_raw)
    if parsed:
        return parsed
    if dte_min is None and dte_max is None:
        return []
    if dte_step <= 0:
        raise SystemExit("--dte-step must be >= 1")
    min_val = int(dte_min) if dte_min is not None else 0
    max_val = int(dte_max) if dte_max is not None else min_val
    if max_val < min_val:
        raise SystemExit("--dte-max must be >= --dte-min")
    return list(range(min_val, max_val + 1, int(dte_step)))


def dates_in_range_by_weekday(start: date, end: date, weekdays: List[int]) -> List[date]:
    if not weekdays:
        return []
    out: List[date] = []
    weekday_set = set(weekdays)
    cursor = start
    while cursor <= end:
        if cursor.weekday() in weekday_set:
            out.append(cursor)
        cursor += timedelta(days=1)
    return out


def build_schedule_entries(
    *,
    start: date,
    end: date,
    schedule_mode: str,
    expiry_weekdays: List[int],
    asof_weekdays: List[int],
    dte_list: List[int],
) -> Tuple[List[Tuple[date, date, date]], Dict[str, int]]:
    entries: List[Tuple[date, date, date]] = []
    stats = {"skipped_after_expiry": 0, "skipped_out_of_range": 0}

    if schedule_mode == "weekly":
        mondays = [d.date() for d in pd.date_range(start=start, end=end, freq="W-MON")]
        for monday in mondays:
            friday = monday + timedelta(days=4)
            if friday < start or friday > end:
                stats["skipped_out_of_range"] += 1
                continue
            if dte_list:
                for dte in dte_list:
                    asof_target = friday - timedelta(days=int(dte))
                    if asof_target < start or asof_target > end:
                        stats["skipped_out_of_range"] += 1
                        continue
                    if asof_target > friday:
                        stats["skipped_after_expiry"] += 1
                        continue
                    entries.append((monday, friday, asof_target))
            else:
                for weekday in asof_weekdays:
                    asof_target = monday + timedelta(days=int(weekday))
                    if asof_target < start or asof_target > end:
                        stats["skipped_out_of_range"] += 1
                        continue
                    if asof_target > friday:
                        stats["skipped_after_expiry"] += 1
                        continue
                    entries.append((monday, friday, asof_target))
    elif schedule_mode == "expiry_range":
        expiries = dates_in_range_by_weekday(start, end, expiry_weekdays)
        for expiry in expiries:
            week_monday = iso_week_monday(expiry)
            if dte_list:
                for dte in dte_list:
                    asof_target = expiry - timedelta(days=int(dte))
                    if asof_target > expiry:
                        stats["skipped_after_expiry"] += 1
                        continue
                    entries.append((week_monday, expiry, asof_target))
            else:
                asof_candidates = dates_in_range_by_weekday(week_monday, expiry, asof_weekdays)
                for asof_target in asof_candidates:
                    if asof_target > expiry:
                        stats["skipped_after_expiry"] += 1
                        continue
                    entries.append((week_monday, expiry, asof_target))
    else:
        raise SystemExit(f"--schedule-mode must be one of: weekly, expiry_range (got {schedule_mode})")

    return entries, stats


class ThetaClient:
    def __init__(self, base_url: str, timeout_s: int = 30, *, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.verbose = bool(verbose)

    def _get_json(self, session: requests.Session, path: str, params: dict) -> list:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = session.get(url, params=params, timeout=self.timeout_s)
            if response.status_code in (404, 472):
                return []
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                resp = payload.get("response", []) or []
                return resp if isinstance(resp, list) else []
            return payload if isinstance(payload, list) else []
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError("Cannot connect to Theta Terminal (is it running on 25503?).") from exc
        except Exception as exc:
            if self.verbose:
                print(f"[THETA] GET failed {url} params={params} err={exc}")
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


_thread_local = threading.local()


def get_thread_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def _yf_download_closes(tickers: List[str], start0: date, end0: date) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. Install with: pip install yfinance")
    ticker_list = " ".join(sorted(set(tickers)))
    df = yf.download(
        tickers=ticker_list,
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
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            splits = ticker_obj.splits
            if splits is None or len(splits) == 0:
                out[ticker] = pd.Series(dtype=float)
                continue
            splits = splits[(splits.index.date >= start0) & (splits.index.date <= end0)]
            out[ticker] = splits.astype(float)
        except Exception:
            out[ticker] = pd.Series(dtype=float)
    return out


def _build_split_adjustment_factor(dates: List[date], splits: pd.Series) -> Dict[date, float]:
    if splits is None or len(splits) == 0:
        return {d: 1.0 for d in dates}
    split_events = sorted(
        [(pd.Timestamp(idx).date(), float(val)) for idx, val in splits.items() if np.isfinite(val) and val > 0]
    )
    if not split_events:
        return {d: 1.0 for d in dates}

    factors: Dict[date, float] = {}
    for current_date in sorted(dates):
        factor = 1.0
        for split_date, ratio in split_events:
            if current_date < split_date:
                factor *= ratio
        factors[current_date] = factor
    return factors


def infer_close_is_already_split_adjusted(close_map: Dict[date, float], splits: pd.Series) -> Optional[bool]:
    if splits is None or len(splits) == 0 or not close_map:
        return None

    events = sorted([(pd.Timestamp(idx).date(), float(val)) for idx, val in splits.items() if np.isfinite(val) and val > 0])
    if not events:
        return None

    dates = sorted(close_map.keys())
    for split_date, ratio in events:
        before = [d for d in dates if d < split_date]
        after = [d for d in dates if d >= split_date]
        if not before or not after:
            continue
        prev_date = before[-1]
        next_date = after[0]
        prev_close = float(close_map[prev_date])
        next_close = float(close_map[next_date])
        if not (np.isfinite(prev_close) and np.isfinite(next_close) and prev_close > 0 and next_close > 0):
            continue
        observed = prev_close / next_close
        if abs(observed - ratio) / ratio <= 0.25:
            return False
        if abs(observed - 1.0) <= 0.25:
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

    raw_out: Dict[str, Dict[date, float]] = {ticker: {} for ticker in tickers}
    adj_out: Dict[str, Dict[date, float]] = {ticker: {} for ticker in tickers}
    split_counts: Dict[str, int] = {ticker: 0 for ticker in tickers}

    def _fill_from_yfinance() -> None:
        nonlocal raw_out, adj_out, split_counts
        df = _yf_download_closes(tickers, start0, end0)
        if isinstance(df.columns, pd.MultiIndex):
            for ticker in tickers:
                if ticker not in df.columns.levels[0]:
                    continue
                sub = df[ticker].copy()
                if "Close" not in sub.columns:
                    continue
                close_series = pd.to_numeric(sub["Close"], errors="coerce").dropna()
                for ts, value in close_series.items():
                    raw_out[ticker][pd.Timestamp(ts).date()] = float(value)
        else:
            if "Close" in df.columns and tickers:
                ticker = tickers[0]
                close_series = pd.to_numeric(df["Close"], errors="coerce").dropna()
                for ts, value in close_series.items():
                    raw_out[ticker][pd.Timestamp(ts).date()] = float(value)

        if cfg.apply_split_adjustment and cfg.split_source == "yfinance":
            splits_by_ticker = _yf_download_splits(tickers, start0, end0)
            for ticker in tickers:
                splits = splits_by_ticker.get(ticker, pd.Series(dtype=float))
                split_counts[ticker] = int(len(splits)) if splits is not None else 0
                if not raw_out.get(ticker):
                    continue

                dates = list(raw_out[ticker].keys())
                factor_map = _build_split_adjustment_factor(dates, splits)
                already_adjusted = infer_close_is_already_split_adjusted(raw_out[ticker], splits)

                if already_adjusted is True:
                    adj_out[ticker] = dict(raw_out[ticker])
                    raw_out[ticker] = {
                        d: float(raw_out[ticker][d] * float(factor_map.get(d, 1.0))) for d in dates
                    }
                else:
                    adj_out[ticker] = {
                        d: float(raw_out[ticker][d] / float(factor_map.get(d, 1.0)))
                        if float(factor_map.get(d, 1.0)) > 0
                        else float(raw_out[ticker][d])
                        for d in dates
                    }
        else:
            for ticker in tickers:
                adj_out[ticker] = dict(raw_out[ticker])

    def _theta_fetch(symbol: str) -> Dict[date, float]:
        session = requests.Session()
        df = theta.stock_eod_range(session, symbol, start0, end0)
        mapping: Dict[date, float] = {}
        if df is None or df.empty:
            return mapping
        date_col = None
        for col in ["date", "time", "t", "timestamp"]:
            if col in df.columns:
                date_col = col
                break
        if date_col is None or "close" not in df.columns:
            return mapping
        dates = pd.to_datetime(df[date_col], errors="coerce")
        closes = pd.to_numeric(df["close"], errors="coerce")
        for dt_value, close_value in zip(dates, closes):
            if pd.isna(dt_value) or not np.isfinite(close_value):
                continue
            mapping[dt_value.date()] = float(close_value)
        return mapping

    if stock_source in {"yfinance", "auto"}:
        try:
            _fill_from_yfinance()
        except Exception as exc:
            if stock_source == "yfinance":
                raise
            print(f"[STOCK] yfinance preload failed, falling back to Theta: {exc}")

    if stock_source in {"theta", "auto"}:
        missing = [ticker for ticker in tickers if not raw_out.get(ticker)]
        for ticker in missing:
            try:
                raw_out[ticker] = _theta_fetch(ticker)
                adj_out[ticker] = dict(raw_out[ticker])
                split_counts[ticker] = 0
            except Exception:
                raw_out[ticker] = {}
                adj_out[ticker] = {}
                split_counts[ticker] = 0

    return raw_out, adj_out, split_counts


def _build_dividend_history_from_series(series: pd.Series) -> DividendHistory:
    if series is None or len(series) == 0:
        return DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=True)
    grouped = series.groupby(series.index.date).sum()
    if grouped is None or len(grouped) == 0:
        return DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=True)
    dates = sorted(list(grouped.index))
    amounts = [float(grouped.loc[d]) for d in dates]
    return DividendHistory(dates=dates, cumsum=np.cumsum(amounts, dtype=float), available=True)


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
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            dividends = ticker_obj.dividends
            if dividends is None or len(dividends) == 0:
                out[ticker] = DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=True)
                continue
            dividends = dividends[(dividends.index.date >= fetch_start) & (dividends.index.date <= end0)]
            out[ticker] = _build_dividend_history_from_series(dividends)
        except Exception:
            out[ticker] = DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=False)
    return out


def preload_dividend_histories(
    *,
    tickers: List[str],
    start: date,
    end: date,
    cfg: Config,
) -> Dict[str, DividendHistory]:
    if cfg.dividend_source != "yfinance":
        return {
            ticker: DividendHistory(dates=[], cumsum=np.array([], dtype=float), available=False)
            for ticker in tickers
        }
    return _yf_download_dividends(tickers, start, end, cfg.dividend_lookback_days)


def get_close_with_fallback_map(
    close_map: Dict[date, float],
    target_date: date,
    *,
    direction: str,
    max_days: int,
) -> Tuple[Optional[float], Optional[date], int]:
    step = 1 if direction == "forward" else -1
    for offset in range(0, max_days + 1):
        candidate = target_date + timedelta(days=step * offset)
        if candidate not in close_map:
            continue
        value = close_map.get(candidate)
        try:
            value = float(value)
        except Exception:
            continue
        if not (np.isfinite(value) and value > 0):
            continue
        return value, candidate, offset
    return None, None, max_days + 1


def realized_vol_proxy(close_map: Dict[date, float], asof_used: date, lookback: int) -> float:
    if lookback <= 2:
        return np.nan
    dates = sorted([current for current in close_map.keys() if current <= asof_used])
    if len(dates) < lookback + 1:
        return np.nan
    selected = dates[-(lookback + 1):]
    prices = np.array([close_map[current] for current in selected], dtype=float)
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        return np.nan
    rets = np.diff(np.log(prices))
    if rets.size < 2:
        return np.nan
    return float(np.sqrt(252.0) * np.nanstd(rets, ddof=1))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def progress_update_interval(total_jobs: int) -> int:
    total = max(0, int(total_jobs))
    return 1 if total <= 10 else 10


LIVE_GROUP_CHECK_FIELDS = (
    ("flag_asof_close_fallback", "asof_close_fallback"),
    ("flag_expiry_close_fallback", "expiry_close_fallback"),
    ("flag_expiry_saturday_fallback", "expiry_saturday_fallback"),
    ("flag_quote_close_fallback", "quote_close_fallback"),
    ("flag_low_chain_used", "low_chain_used"),
    ("flag_wide_rel_spread", "wide_rel_spread"),
)


def build_live_telemetry(tickers: List[str], planned_jobs_per_ticker: int) -> Dict[str, object]:
    return {
        "phase": "planning",
        "drop_reasons": {},
        "group_checks": {metric_key: 0 for _, metric_key in LIVE_GROUP_CHECK_FIELDS},
        "tickers": [
            {
                "ticker": ticker,
                "completed_jobs": 0,
                "planned_jobs": int(planned_jobs_per_ticker),
                "kept_groups": 0,
                "rows": 0,
                "issue_count_sum": 0.0,
                "flagged_rows": 0,
                "fallback_rows": 0,
                "wide_spread_rows": 0,
                "clean_rows": 0,
                "watch_rows": 0,
                "noisy_rows": 0,
                "drop_reasons": {},
            }
            for ticker in tickers
        ],
    }


def _get_live_ticker_entry(telemetry: Dict[str, object], ticker: str) -> Dict[str, object]:
    items = telemetry.setdefault("tickers", [])
    if not isinstance(items, list):
        items = []
        telemetry["tickers"] = items
    for item in items:
        if isinstance(item, dict) and str(item.get("ticker")) == ticker:
            return item
    entry = {
        "ticker": ticker,
        "completed_jobs": 0,
        "planned_jobs": 0,
        "kept_groups": 0,
        "rows": 0,
        "issue_count_sum": 0.0,
        "flagged_rows": 0,
        "fallback_rows": 0,
        "wide_spread_rows": 0,
        "clean_rows": 0,
        "watch_rows": 0,
        "noisy_rows": 0,
        "drop_reasons": {},
    }
    items.append(entry)
    return entry


def _increment_reason_count(bucket: Dict[str, object], reason: Optional[str]) -> None:
    if not reason:
        return
    bucket[reason] = int(bucket.get(reason) or 0) + 1


def _live_issue_count(row: Dict[str, object]) -> float:
    value = row.get("quality_issue_count")
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    return numeric if np.isfinite(numeric) else 0.0


def _live_quality_bucket(row: Dict[str, object], issue_count: float) -> str:
    bucket = str(row.get("quality_bucket") or "").strip().lower()
    if bucket in {"clean", "watch", "noisy"}:
        return bucket
    if issue_count <= 0:
        return "clean"
    if issue_count <= 2:
        return "watch"
    return "noisy"


def update_live_telemetry_for_job(
    telemetry: Dict[str, object],
    ticker: str,
    rows: List[dict],
    drop_log: Optional[dict],
) -> None:
    entry = _get_live_ticker_entry(telemetry, ticker)
    entry["completed_jobs"] = int(entry.get("completed_jobs") or 0) + 1

    if rows:
        entry["kept_groups"] = int(entry.get("kept_groups") or 0) + 1
        entry["rows"] = int(entry.get("rows") or 0) + len(rows)
        first_row = rows[0]
        group_checks = telemetry.setdefault("group_checks", {})
        if not isinstance(group_checks, dict):
            group_checks = {}
            telemetry["group_checks"] = group_checks
        for flag_key, metric_key in LIVE_GROUP_CHECK_FIELDS:
            if bool(first_row.get(flag_key, False)):
                group_checks[metric_key] = int(group_checks.get(metric_key) or 0) + 1
        issue_count_sum = 0.0
        flagged_rows = 0
        fallback_rows = 0
        wide_spread_rows = 0
        clean_rows = 0
        watch_rows = 0
        noisy_rows = 0
        for row in rows:
            issue_count = _live_issue_count(row)
            issue_count_sum += issue_count
            if issue_count > 0:
                flagged_rows += 1
            if bool(row.get("flag_asof_close_fallback", False)) or bool(
                row.get("flag_expiry_close_fallback", False)
            ):
                fallback_rows += 1
            if bool(row.get("flag_wide_rel_spread", False)):
                wide_spread_rows += 1
            quality_bucket = _live_quality_bucket(row, issue_count)
            if quality_bucket == "clean":
                clean_rows += 1
            elif quality_bucket == "watch":
                watch_rows += 1
            else:
                noisy_rows += 1
        entry["issue_count_sum"] = float(entry.get("issue_count_sum") or 0.0) + issue_count_sum
        entry["flagged_rows"] = int(entry.get("flagged_rows") or 0) + flagged_rows
        entry["fallback_rows"] = int(entry.get("fallback_rows") or 0) + fallback_rows
        entry["wide_spread_rows"] = int(entry.get("wide_spread_rows") or 0) + wide_spread_rows
        entry["clean_rows"] = int(entry.get("clean_rows") or 0) + clean_rows
        entry["watch_rows"] = int(entry.get("watch_rows") or 0) + watch_rows
        entry["noisy_rows"] = int(entry.get("noisy_rows") or 0) + noisy_rows

    reason = None
    if isinstance(drop_log, dict):
        raw_reason = drop_log.get("drop_reason")
        if raw_reason is not None:
            reason = str(raw_reason).strip() or None
    if reason:
        drop_reasons = telemetry.setdefault("drop_reasons", {})
        if not isinstance(drop_reasons, dict):
            drop_reasons = {}
            telemetry["drop_reasons"] = drop_reasons
        entry_drop_reasons = entry.setdefault("drop_reasons", {})
        if not isinstance(entry_drop_reasons, dict):
            entry_drop_reasons = {}
            entry["drop_reasons"] = entry_drop_reasons
        _increment_reason_count(drop_reasons, reason)
        _increment_reason_count(entry_drop_reasons, reason)


def emit_live_telemetry(telemetry: Dict[str, object]) -> None:
    print(f"[LIVE] {json.dumps(telemetry, sort_keys=True)}", flush=True)


def fix_negative_zero(x: float, eps: float = 5e-13) -> float:
    if x is None or not np.isfinite(x):
        return x
    return 0.0 if abs(float(x)) < eps else float(x)


def enforce_call_monotone_decreasing(prices: np.ndarray) -> np.ndarray:
    curve = np.asarray(prices, dtype=float)
    reverse_curve = curve[::-1]
    monotone_reverse = np.maximum.accumulate(reverse_curve)
    return monotone_reverse[::-1]


def pava_isotonic_increasing(y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    values: List[float] = []
    weights: List[float] = []
    lengths: List[int] = []

    for y_i, w_i in zip(y, w):
        values.append(float(y_i))
        weights.append(float(w_i))
        lengths.append(1)
        while len(values) >= 2 and values[-2] > values[-1]:
            last_value, last_weight, last_length = values.pop(), weights.pop(), lengths.pop()
            prev_value, prev_weight, prev_length = values.pop(), weights.pop(), lengths.pop()
            merged_weight = prev_weight + last_weight
            merged_value = (prev_value * prev_weight + last_value * last_weight) / merged_weight
            values.append(merged_value)
            weights.append(merged_weight)
            lengths.append(prev_length + last_length)

    out = np.empty(n, dtype=float)
    idx = 0
    for value, length in zip(values, lengths):
        out[idx : idx + length] = value
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
    for col in ["strike", "bid", "ask", "close", "volume", "count"]:
        if col not in df.columns:
            df[col] = np.nan

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

    T_ref = max(float(T_years), 1e-8)
    discounted_strike = use["strike"].astype(float) * np.exp(-float(r) * T_ref)
    forward_discounted = float(spot) * np.exp(-float(q) * T_ref)
    intrinsic = np.maximum(forward_discounted - discounted_strike, 0.0)

    before = len(use)
    use = use[use["mid"].astype(float) >= float(cfg.intrinsic_tol) * intrinsic].copy()
    diag["dropped_intrinsic"] = int(before - len(use))

    before = len(use)
    use = use[use["mid"].astype(float) <= float(cfg.insane_price_multiple) * float(spot)].copy()
    diag["dropped_insane"] = int(before - len(use))

    use = use.sort_values("strike").drop_duplicates("strike", keep="last")
    diag["n_used"] = int(len(use))

    if "rel_spread" in use.columns:
        median = pd.to_numeric(use["rel_spread"], errors="coerce").median()
        diag["rel_spread_median"] = float(median) if np.isfinite(median) else None

    if len(use) < 5:
        return None, None, diag

    strikes = use["strike"].to_numpy(dtype=float)
    prices = use["mid"].to_numpy(dtype=float)
    monotone = enforce_call_monotone_decreasing(prices)
    monotone = pd.Series(monotone).rolling(5, center=True, min_periods=1).median().to_numpy(dtype=float)
    return strikes, monotone, diag


def compute_prn_from_call_curve(
    k_arr: np.ndarray,
    c_arr: np.ndarray,
    *,
    K_targets: np.ndarray,
    T_years: float,
    r: float,
) -> Tuple[np.ndarray, dict]:
    diag = {
        "monotone_adjusted_intervals": False,
        "monotone_adjusted_targets": False,
        "p_targets_raw": None,
    }

    k = np.asarray(k_arr, dtype=float)
    c = np.asarray(c_arr, dtype=float)
    targets = np.asarray(K_targets, dtype=float)
    if k.size < 3 or c.size != k.size or targets.size == 0:
        return np.full_like(targets, np.nan, dtype=float), diag

    delta_k = np.diff(k)
    delta_c = np.diff(c)
    valid = np.isfinite(delta_k) & (delta_k > 0) & np.isfinite(delta_c)
    if int(valid.sum()) < 2:
        return np.full_like(targets, np.nan, dtype=float), diag

    delta_k = delta_k[valid]
    delta_c = delta_c[valid]
    left = k[:-1][valid]
    right = k[1:][valid]
    midpoints = 0.5 * (left + right)

    slope = delta_c / delta_k
    p_interval = -np.exp(float(r) * float(T_years)) * slope
    p_interval = np.clip(p_interval, 0.0, 1.0)

    p_targets_raw = np.interp(targets, midpoints, p_interval)
    diag["p_targets_raw"] = p_targets_raw.copy()

    p_iso = isotonic_decreasing(p_interval)
    if not np.allclose(p_iso, p_interval, atol=1e-12, rtol=0):
        diag["monotone_adjusted_intervals"] = True
    p_iso = np.clip(p_iso, 0.0, 1.0)

    p_targets = np.interp(targets, midpoints, p_iso)
    order = np.argsort(targets)
    sorted_targets = p_targets[order]
    sorted_targets_iso = isotonic_decreasing(sorted_targets)
    if not np.allclose(sorted_targets_iso, sorted_targets, atol=1e-12, rtol=0):
        diag["monotone_adjusted_targets"] = True
    sorted_targets_iso = np.clip(sorted_targets_iso, 0.0, 1.0)

    out = np.empty_like(p_targets)
    out[order] = sorted_targets_iso
    return out, diag


def _band_strikes_with_abslogm(strikes: np.ndarray, spot: float, abslogm: float, *, cap: int = 0) -> np.ndarray:
    strikes = np.asarray(strikes, dtype=float)
    strikes = strikes[np.isfinite(strikes) & (strikes > 0)]
    if strikes.size == 0 or not np.isfinite(spot) or spot <= 0:
        return np.array([], dtype=float)
    log_m = np.log(strikes / float(spot))
    keep = np.abs(log_m) <= float(abslogm)
    band = np.sort(np.unique(strikes[keep]))
    if cap and band.size > cap:
        idx = np.argsort(np.abs(np.log(band / float(spot))))
        band = np.sort(band[idx[:cap]])
    return band


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
        band = _band_strikes_with_abslogm(strikes, spot, start, cap=int(cfg.max_band_strikes))
        inside = band[(band > k_min) & (band < k_max)]
        return band, inside, start

    used = start
    while used <= cap + 1e-12:
        band = _band_strikes_with_abslogm(strikes, spot, used, cap=int(cfg.max_band_strikes))
        inside = band[(band > k_min) & (band < k_max)]
        if int(inside.size) >= int(cfg.min_strikes_for_curve):
            return band, inside, used
        used += step

    used = cap
    band = _band_strikes_with_abslogm(strikes, spot, used, cap=int(cfg.max_band_strikes))
    inside = band[(band > k_min) & (band < k_max)]
    return band, inside, used


def strike_spacing_stats(strikes: np.ndarray) -> Tuple[float, float]:
    strikes = np.asarray(strikes, dtype=float)
    if strikes.size < 3:
        return np.nan, np.nan
    delta = np.diff(np.sort(strikes))
    delta = delta[np.isfinite(delta) & (delta > 0)]
    if delta.size == 0:
        return np.nan, np.nan
    return float(np.median(delta)), float(np.min(delta))


def _score_spot_scale(*, k_arr: np.ndarray, spot: float, cfg: Config) -> Tuple[float, int, float]:
    if k_arr is None or len(k_arr) < 3 or not np.isfinite(spot) or spot <= 0:
        return -1e18, 0, float(cfg.max_abs_logm)
    k_min = float(np.min(k_arr))
    k_max = float(np.max(k_arr))
    _, inside, used = pick_band_strikes(k_arr, float(spot), cfg=cfg, k_min=k_min, k_max=k_max)
    n_inside = int(inside.size)
    bonus = 1e6 if n_inside >= int(cfg.min_strikes_for_curve) else 0.0
    return float(bonus + n_inside), n_inside, float(used)


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
    return dt_time(16, 0)


def _asof_time_for_date(
    value: date,
    *,
    tz_name: str,
    close_time: str,
) -> pd.Timestamp:
    close_t = _parse_close_time(close_time)
    naive = pd.Timestamp(datetime.combine(value, close_t))
    localized = naive.tz_localize(tz_name, nonexistent="shift_forward", ambiguous="NaT")
    return localized.tz_convert("UTC")


def _base_target_row(
    *,
    ticker: str,
    target: dict,
    week_monday: date,
    week_friday: date,
    asof_target: date,
    asof_date_used: date,
    asof_time: pd.Timestamp,
    prn_version: str,
    prn_config_hash: str,
    drop_reason: Optional[str],
    coverage_status: str,
) -> dict:
    strike = float(target["K"])
    return {
        "row_id": compute_row_id(
            asof_ts=iso_ts(asof_date_used),
            ticker=ticker,
            expiry_date_used=week_friday.isoformat(),
            strike=strike,
            option_type="call",
        ),
        "asof_ts": iso_ts(asof_date_used),
        "asof_time": asof_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "option_type": "call",
        "ticker": ticker,
        "market_id": str(target.get("market_id") or ""),
        "event_id": str(target.get("event_id") or ""),
        "week_monday": week_monday.isoformat(),
        "week_friday": week_friday.isoformat(),
        "asof_target": asof_target.isoformat(),
        "asof_date": asof_date_used.isoformat(),
        "snapshot_date": asof_date_used.isoformat(),
        "expiry_close_date_used": week_friday.isoformat(),
        "option_expiration_requested": week_friday.isoformat(),
        "option_expiration_used": week_friday.isoformat(),
        "expiry_convention": "FRI",
        "expiry_date": week_friday.isoformat(),
        "event_endDate": target.get("event_endDate") or week_friday.isoformat(),
        "T_days": (week_friday - asof_date_used).days,
        "T_years": max((week_friday - asof_date_used).days, 0) / 365.25,
        "r": np.nan,
        "K": strike,
        "threshold": strike,
        "pRN": np.nan,
        "qRN": np.nan,
        "pRN_raw": np.nan,
        "qRN_raw": np.nan,
        "rv20": np.nan,
        "log_m": np.nan,
        "abs_log_m": np.nan,
        "log_m_fwd": np.nan,
        "abs_log_m_fwd": np.nan,
        "S_asof_close": np.nan,
        "forward_price": np.nan,
        "dividend_yield": np.nan,
        "rn_method": RN_METHOD,
        "theta_quote_source": np.nan,
        "prn_config_hash": prn_config_hash,
        "prn_version": prn_version,
        "coverage_status": coverage_status,
        "drop_reason": drop_reason,
        "schema_version": POLYMARKET_EXACT_SCHEMA_VERSION,
    }


def _failure_rows_for_targets(
    *,
    ticker: str,
    targets: List[dict],
    week_monday: date,
    week_friday: date,
    asof_target: date,
    asof_date_used: Optional[date],
    prn_version: str,
    prn_config_hash: str,
    tz_name: str,
    close_time: str,
    drop_reason: str,
) -> List[dict]:
    effective_asof = asof_date_used or asof_target
    asof_time = _asof_time_for_date(effective_asof, tz_name=tz_name, close_time=close_time)
    rows = []
    for target in targets:
        row = _base_target_row(
            ticker=ticker,
            target=target,
            week_monday=week_monday,
            week_friday=week_friday,
            asof_target=asof_target,
            asof_date_used=effective_asof,
            asof_time=asof_time,
            prn_version=prn_version,
            prn_config_hash=prn_config_hash,
            drop_reason=drop_reason,
            coverage_status="missing",
        )
        rows.append(row)
    return rows


def process_one(
    *,
    theta: ThetaClient,
    cfg: Config,
    ticker: str,
    asof_target: date,
    week_monday: date,
    week_friday: date,
    raw_closes_by_ticker: Dict[str, Dict[date, float]],
    adj_closes_by_ticker: Dict[str, Dict[date, float]],
    split_event_counts: Dict[str, int],
    dividend_histories: Dict[str, DividendHistory],
    option_chain_cache: Dict[Tuple[str, date, date, Optional[int]], pd.DataFrame],
    cache_lock: threading.Lock,
    targets: Optional[List[dict]] = None,
    prn_version: str = DEFAULT_PRN_VERSION,
    prn_config_hash: str = "",
    prn_asof_tz: str = DEFAULT_PRN_ASOF_TZ,
    prn_asof_close_time: str = DEFAULT_PRN_ASOF_CLOSE_TIME,
) -> Tuple[List[dict], Optional[dict]]:
    raw_map = raw_closes_by_ticker.get(ticker, {})
    adj_map = adj_closes_by_ticker.get(ticker, {})
    split_n = int(split_event_counts.get(ticker, 0))

    S0_raw, asof_raw, asof_fwd = get_close_with_fallback_map(
        raw_map, asof_target, direction="forward", max_days=cfg.max_forward_days_for_asof
    )
    S0_adj, asof_adj, _ = get_close_with_fallback_map(
        adj_map, asof_target, direction="forward", max_days=cfg.max_forward_days_for_asof
    )
    if S0_raw is None or asof_raw is None or S0_adj is None or asof_adj is None:
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "missing_S0",
            "detail": f"forward<={cfg.max_forward_days_for_asof}",
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=None,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="missing_S0",
                ),
                drop_log,
            )
        return [], drop_log

    ST_raw, exp_raw, exp_bwd = get_close_with_fallback_map(
        raw_map, week_friday, direction="backward", max_days=cfg.max_backward_days_for_expiry_close
    )
    ST_adj, exp_adj, _ = get_close_with_fallback_map(
        adj_map, week_friday, direction="backward", max_days=cfg.max_backward_days_for_expiry_close
    )
    if ST_raw is None or exp_raw is None or ST_adj is None or exp_adj is None:
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "missing_ST",
            "detail": f"backward<={cfg.max_backward_days_for_expiry_close}",
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=asof_adj,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="missing_ST",
                ),
                drop_log,
            )
        return [], drop_log

    asof_used = asof_adj
    expiry_close_used = exp_adj
    asof_time = _asof_time_for_date(asof_used, tz_name=prn_asof_tz, close_time=prn_asof_close_time)

    T_days = int((week_friday - asof_used).days)
    if T_days <= 0:
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "bad_T_days",
            "detail": str(T_days),
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=asof_used,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="bad_T_days",
                ),
                drop_log,
            )
        return [], drop_log
    T_years = float(T_days) / 365.25

    rv5_raw = realized_vol_proxy(raw_map, asof_used, 5)
    rv10_raw = realized_vol_proxy(raw_map, asof_used, 10)
    rv20_raw = realized_vol_proxy(raw_map, asof_used, cfg.rv_lookback_days)
    rv5_adj = realized_vol_proxy(adj_map, asof_used, 5)
    rv10_adj = realized_vol_proxy(adj_map, asof_used, 10)
    rv20_adj = realized_vol_proxy(adj_map, asof_used, cfg.rv_lookback_days)

    div_sum = np.nan
    if cfg.dividend_source == "yfinance":
        hist = dividend_histories.get(ticker)
        if hist is not None and hist.available:
            start_lb = asof_used - timedelta(days=int(cfg.dividend_lookback_days))
            div_sum = float(hist.sum_in_range(start_lb, asof_used))

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

    expiration_requested = week_friday
    expiration_used = week_friday
    expiry_convention = "FRI"

    def _fetch_chain_for_exp(expiration: date, strike_range: Optional[int]) -> pd.DataFrame:
        key = (ticker, asof_used, expiration, strike_range)
        if not cfg.use_cache:
            return theta.option_eod_chain(
                session, ticker, asof=asof_used, expiration=expiration, right="call", strike_range=strike_range
            )
        with cache_lock:
            cached = option_chain_cache.get(key)
        if cached is not None:
            return cached
        chain = theta.option_eod_chain(
            session, ticker, asof=asof_used, expiration=expiration, right="call", strike_range=strike_range
        )
        with cache_lock:
            option_chain_cache[key] = chain
        return chain

    def _fetch_chain(strike_range: Optional[int]) -> pd.DataFrame:
        nonlocal expiration_used, expiry_convention
        chain = _fetch_chain_for_exp(week_friday, strike_range)
        if chain is not None and not chain.empty:
            expiration_used = week_friday
            expiry_convention = "FRI"
            return chain
        if cfg.try_saturday_expiry_fallback:
            saturday = week_friday + timedelta(days=1)
            chain_sat = _fetch_chain_for_exp(saturday, strike_range)
            if chain_sat is not None and not chain_sat.empty:
                expiration_used = saturday
                expiry_convention = "SAT_FALLBACK"
                return chain_sat
        return chain

    strike_range_first = None if (cfg.option_strike_range <= 0) else int(cfg.option_strike_range)
    chain = _fetch_chain(strike_range_first)
    if chain is None or chain.empty:
        if cfg.retry_full_chain_if_band_thin:
            chain = _fetch_chain(None)
        if chain is None or chain.empty:
            detail = (
                f"asof={asof_used.isoformat()} exp_req={expiration_requested.isoformat()} "
                f"exp_used={expiration_used.isoformat()} conv={expiry_convention}"
            )
            drop_log = {
                "ticker": ticker,
                "week_monday": week_monday.isoformat(),
                "week_friday": week_friday.isoformat(),
                "asof_target": asof_target.isoformat(),
                "drop_reason": "empty_option_chain",
                "detail": detail,
            }
            if targets:
                return (
                    _failure_rows_for_targets(
                        ticker=ticker,
                        targets=targets,
                        week_monday=week_monday,
                        week_friday=week_friday,
                        asof_target=asof_target,
                        asof_date_used=asof_used,
                        prn_version=prn_version,
                        prn_config_hash=prn_config_hash,
                        tz_name=prn_asof_tz,
                        close_time=prn_asof_close_time,
                        drop_reason="empty_option_chain",
                    ),
                    drop_log,
                )
            return [], drop_log

    curve_candidates = []

    def _forward_price(spot: float, q: float) -> float:
        if not np.isfinite(spot) or spot <= 0:
            return np.nan
        return float(spot) * float(np.exp((float(cfg.risk_free_rate) - float(q)) * float(T_years)))

    for label, spot, q in [("split_adj", float(S0_adj), div_yield_adj), ("raw", float(S0_raw), div_yield_raw)]:
        forward = _forward_price(float(spot), float(q))
        spot_ref = forward if (cfg.use_forward_moneyness and np.isfinite(forward) and forward > 0) else float(spot)
        k_arr, c_arr, diag_curve = build_call_curve_from_eod(
            chain, spot=float(spot), T_years=float(T_years), r=float(cfg.risk_free_rate), q=float(q), cfg=cfg
        )
        if k_arr is None or c_arr is None:
            continue
        score, _, used_abslogm_hint = _score_spot_scale(k_arr=k_arr, spot=float(spot_ref), cfg=cfg)
        n_used = int(diag_curve.get("n_used") or 0)
        curve_candidates.append(
            (
                score,
                n_used,
                label,
                float(spot),
                float(q),
                float(forward),
                float(spot_ref),
                float(used_abslogm_hint),
                k_arr,
                c_arr,
                diag_curve,
            )
        )

    if not curve_candidates:
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "cannot_build_curve",
            "detail": f"asof={asof_used.isoformat()} exp_used={expiration_used.isoformat()}",
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=asof_used,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="cannot_build_curve",
                ),
                drop_log,
            )
        return [], drop_log

    curve_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, spot_scale_used, _, _, forward_used, _, _, k_arr, c_arr, diag_curve = curve_candidates[0]

    if cfg.hard_drop_close_fallback and diag_curve.get("quote_source") != "bidask_mid":
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "hard_drop_close_fallback",
            "detail": asof_used.isoformat(),
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=asof_used,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="hard_drop_close_fallback",
                ),
                drop_log,
            )
        return [], drop_log

    if cfg.min_chain_used_hard > 0 and int(diag_curve.get("n_used") or 0) < cfg.min_chain_used_hard:
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "hard_min_chain_used",
            "detail": str(diag_curve.get("n_used")),
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=asof_used,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="hard_min_chain_used",
                ),
                drop_log,
            )
        return [], drop_log

    rel_spread_median = diag_curve.get("rel_spread_median")
    if (rel_spread_median is not None) and np.isfinite(rel_spread_median) and float(rel_spread_median) > cfg.max_rel_spread_median_hard:
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "hard_max_rel_spread_median",
            "detail": str(rel_spread_median),
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=asof_used,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="hard_max_rel_spread_median",
                ),
                drop_log,
            )
        return [], drop_log

    k_min = float(np.min(k_arr))
    k_max = float(np.max(k_arr))
    spot_used = float(S0_adj) if spot_scale_used == "split_adj" else float(S0_raw)
    dividend_yield_used = float(div_yield_adj) if spot_scale_used == "split_adj" else float(div_yield_raw)
    forward_price_used = float(forward_used) if np.isfinite(forward_used) and forward_used > 0 else _forward_price(spot_used, dividend_yield_used)
    spot_ref_used = forward_price_used if (cfg.use_forward_moneyness and np.isfinite(forward_price_used) and forward_price_used > 0) else float(spot_used)
    moneyness_ref = "forward" if (cfg.use_forward_moneyness and np.isfinite(forward_price_used) and forward_price_used > 0) else "spot"
    k_band, k_band_inside, used_abslogm = pick_band_strikes(
        k_arr, float(spot_ref_used), cfg=cfg, k_min=k_min, k_max=k_max
    )
    n_band_raw = int(k_band.size)
    n_band_inside = int(k_band_inside.size)

    if n_band_inside < cfg.min_strikes_for_curve and cfg.retry_full_chain_if_band_thin and (strike_range_first is not None):
        chain_full = _fetch_chain(None)
        if chain_full is not None and not chain_full.empty:
            k2, c2, diag2 = build_call_curve_from_eod(
                chain_full,
                spot=float(spot_used),
                T_years=float(T_years),
                r=float(cfg.risk_free_rate),
                q=float(dividend_yield_used),
                cfg=cfg,
            )
            if k2 is not None and c2 is not None:
                kmin2 = float(np.min(k2))
                kmax2 = float(np.max(k2))
                band2, inside2, used2 = pick_band_strikes(k2, float(spot_ref_used), cfg=cfg, k_min=kmin2, k_max=kmax2)
                if int(inside2.size) > n_band_inside:
                    k_arr, c_arr, diag_curve = k2, c2, diag2
                    k_min, k_max = kmin2, kmax2
                    k_band, k_band_inside, used_abslogm = band2, inside2, used2
                    n_band_raw = int(k_band.size)
                    n_band_inside = int(k_band_inside.size)

    if n_band_inside < cfg.min_strikes_for_curve:
        detail = (
            f"inside={n_band_inside} raw={n_band_raw} spot_scale={spot_scale_used} "
            f"used_abslogm={used_abslogm:.4f} moneyness_ref={moneyness_ref}"
        )
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "thin_band_inside",
            "detail": detail,
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=asof_used,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="thin_band_inside",
                ),
                drop_log,
            )
        return [], drop_log

    if targets:
        target_strikes = np.array([float(target["K"]) for target in targets], dtype=float)
    else:
        target_strikes = np.array(k_band_inside, dtype=float)

    pRN_values, diag_prn = compute_prn_from_call_curve(
        k_arr, c_arr, K_targets=target_strikes, T_years=float(T_years), r=float(cfg.risk_free_rate)
    )
    if pRN_values is None or len(pRN_values) != len(target_strikes):
        drop_log = {
            "ticker": ticker,
            "week_monday": week_monday.isoformat(),
            "week_friday": week_friday.isoformat(),
            "asof_target": asof_target.isoformat(),
            "drop_reason": "bad_prn_compute",
            "detail": asof_used.isoformat(),
        }
        if targets:
            return (
                _failure_rows_for_targets(
                    ticker=ticker,
                    targets=targets,
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=asof_used,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="bad_prn_compute",
                ),
                drop_log,
            )
        return [], drop_log

    raw_targets = diag_prn.get("p_targets_raw", None)
    midpoint_support = 0.5 * (k_arr[:-1] + k_arr[1:]) if len(k_arr) >= 2 else np.array([], dtype=float)
    support_min = float(np.min(midpoint_support)) if midpoint_support.size else np.nan
    support_max = float(np.max(midpoint_support)) if midpoint_support.size else np.nan

    rv5_used = rv5_adj if spot_scale_used == "split_adj" else rv5_raw
    rv10_used = rv10_adj if spot_scale_used == "split_adj" else rv10_raw
    rv20_used = rv20_adj if spot_scale_used == "split_adj" else rv20_raw
    ST_used = float(ST_adj) if spot_scale_used == "split_adj" else float(ST_raw)

    median_dK, min_dK = strike_spacing_stats(k_arr)

    rows: List[dict] = []
    iter_targets: Iterable[dict]
    if targets:
        iter_targets = targets
    else:
        iter_targets = [
            {"K": float(strike), "market_id": "", "event_id": "", "event_endDate": week_friday.isoformat()}
            for strike in target_strikes
        ]

    for idx, target in enumerate(iter_targets):
        strike = float(target["K"])
        row = _base_target_row(
            ticker=ticker,
            target=target,
            week_monday=week_monday,
            week_friday=week_friday,
            asof_target=asof_target,
            asof_date_used=asof_used,
            asof_time=asof_time,
            prn_version=prn_version,
            prn_config_hash=prn_config_hash,
            drop_reason=None,
            coverage_status="ok",
        )
        row["option_expiration_requested"] = expiration_requested.isoformat()
        row["option_expiration_used"] = expiration_used.isoformat()
        row["expiry_convention"] = expiry_convention
        row["expiry_date"] = expiration_used.isoformat()
        row["expiry_close_date_used"] = expiry_close_used.isoformat()
        row["T_days"] = T_days
        row["T_years"] = T_years
        row["r"] = float(cfg.risk_free_rate)
        row["theta_quote_source"] = diag_curve.get("quote_source")
        row["S_asof_close"] = spot_used
        row["forward_price"] = forward_price_used
        row["dividend_yield"] = dividend_yield_used
        row["rv20"] = rv20_used
        row["pRN"] = float(np.round(pRN_values[idx], 7)) if np.isfinite(pRN_values[idx]) else np.nan
        row["qRN"] = float(np.round(1.0 - row["pRN"], 7)) if np.isfinite(row["pRN"]) else np.nan
        p_raw = float(raw_targets[idx]) if raw_targets is not None and idx < len(raw_targets) else np.nan
        row["pRN_raw"] = p_raw if np.isfinite(p_raw) else np.nan
        row["qRN_raw"] = float(np.round(1.0 - p_raw, 7)) if np.isfinite(p_raw) else np.nan
        row["log_m"] = float(np.log(strike / spot_used)) if np.isfinite(spot_used) and spot_used > 0 else np.nan
        row["abs_log_m"] = abs(float(row["log_m"])) if np.isfinite(row["log_m"]) else np.nan
        row["log_m_fwd"] = (
            float(np.log(strike / forward_price_used))
            if np.isfinite(forward_price_used) and forward_price_used > 0
            else np.nan
        )
        row["abs_log_m_fwd"] = abs(float(row["log_m_fwd"])) if np.isfinite(row["log_m_fwd"]) else np.nan
        row["flag_asof_close_fallback"] = bool(asof_fwd > 0)
        row["flag_expiry_close_fallback"] = bool(exp_bwd > 0)
        row["flag_expiry_saturday_fallback"] = bool(expiry_convention == "SAT_FALLBACK")
        row["flag_quote_close_fallback"] = bool(diag_curve.get("quote_source") == "close_fallback")
        row["flag_low_chain_used"] = bool(int(diag_curve.get("n_used") or 0) < QUALITY_CHAIN_USED_WARN)
        row["flag_wide_rel_spread"] = bool(
            diag_curve.get("rel_spread_median") is not None
            and np.isfinite(diag_curve.get("rel_spread_median"))
            and float(diag_curve.get("rel_spread_median")) > QUALITY_REL_SPREAD_WARN
        )
        row["spot_scale_used"] = spot_scale_used
        row["S_expiry_close"] = ST_used
        row["split_events_in_preload_range"] = split_n
        row["n_chain_used"] = int(diag_curve.get("n_used") or 0)
        row["n_chain_raw"] = int(diag_curve.get("n_raw") or 0)
        row["rel_spread_median"] = diag_curve.get("rel_spread_median")
        row["median_dK"] = median_dK
        row["min_dK"] = min_dK
        row["used_max_abs_logm"] = used_abslogm
        row["n_band_raw"] = n_band_raw
        row["n_band_inside"] = n_band_inside
        row["moneyness_ref"] = moneyness_ref
        row["moneyness_ref_price"] = spot_ref_used
        row["calls_k_min"] = k_min
        row["calls_k_max"] = k_max

        if np.isfinite(support_min) and np.isfinite(support_max) and not (support_min <= strike <= support_max):
            row["coverage_status"] = "missing"
            row["drop_reason"] = "target_outside_curve_support"
            row["pRN"] = np.nan
            row["qRN"] = np.nan
            row["pRN_raw"] = np.nan
            row["qRN_raw"] = np.nan
        rows.append(row)

    return rows, None


def _normalize_polymarket_targets(weekly_markets: pd.DataFrame) -> pd.DataFrame:
    required = {"market_id", "ticker", "threshold", "week_friday"}
    missing = required - set(weekly_markets.columns)
    if missing:
        raise ValueError(f"weekly_markets is missing required columns: {sorted(missing)}")

    df = weekly_markets.copy()
    df["market_id"] = df["market_id"].astype(str).str.strip()
    df["event_id"] = df.get("event_id", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce").round(6)
    df["week_friday"] = pd.to_datetime(df["week_friday"], errors="coerce").dt.date
    if "week_monday" in df.columns:
        df["week_monday"] = pd.to_datetime(df["week_monday"], errors="coerce").dt.date
    else:
        df["week_monday"] = df["week_friday"].apply(lambda value: value - timedelta(days=4) if isinstance(value, date) else None)
    if "event_endDate" in df.columns:
        df["event_endDate"] = pd.to_datetime(df["event_endDate"], errors="coerce").dt.date
        df["event_endDate"] = df["event_endDate"].apply(lambda value: value.isoformat() if isinstance(value, date) else None)
    else:
        df["event_endDate"] = df["week_friday"].apply(lambda value: value.isoformat() if isinstance(value, date) else None)

    df = df.dropna(subset=["market_id", "ticker", "threshold", "week_friday", "week_monday"])
    df = df[df["market_id"] != ""].copy()
    df = df.sort_values(["ticker", "week_friday", "threshold", "market_id"]).reset_index(drop=True)
    return df


def _polymarket_job_targets(df: pd.DataFrame) -> Dict[Tuple[str, date, date, date], List[dict]]:
    jobs: Dict[Tuple[str, date, date, date], List[dict]] = {}
    for row in df.itertuples(index=False):
        week_monday = row.week_monday
        week_friday = row.week_friday
        if not isinstance(week_monday, date) or not isinstance(week_friday, date):
            continue
        asof_targets = [week_monday + timedelta(days=offset) for offset in range(0, 4)]
        for asof_target in asof_targets:
            target = {
                "market_id": str(row.market_id),
                "event_id": str(getattr(row, "event_id", "") or ""),
                "event_endDate": getattr(row, "event_endDate", None) or week_friday.isoformat(),
                "K": float(row.threshold),
            }
            key = (str(row.ticker), week_monday, week_friday, asof_target)
            jobs.setdefault(key, []).append(target)
    return jobs


def build_polymarket_exact_prn(
    weekly_markets: pd.DataFrame,
    *,
    cfg: Optional[Config] = None,
    threads: int = DEFAULT_THREADS,
    prn_version: str = DEFAULT_PRN_VERSION,
    prn_asof_tz: str = DEFAULT_PRN_ASOF_TZ,
    prn_asof_close_time: str = DEFAULT_PRN_ASOF_CLOSE_TIME,
    verbose_skips: bool = False,
) -> PolymarketExactPrnBuildResult:
    cfg = cfg or Config()
    normalized = _normalize_polymarket_targets(weekly_markets)
    jobs_by_key = _polymarket_job_targets(normalized)
    tickers = sorted(normalized["ticker"].dropna().unique().tolist())
    prn_config_hash = compute_prn_config_hash(cfg)

    if not jobs_by_key:
        return PolymarketExactPrnBuildResult(
            rows=pd.DataFrame(columns=POLYMARKET_EXACT_COLUMNS),
            drop_logs=[],
            required_market_snapshots=0,
            ok_market_snapshots=0,
            missing_market_snapshots=0,
            coverage_counts={},
            drop_reason_counts={},
            prn_version=prn_version,
            prn_config_hash=prn_config_hash,
        )

    schedule_entries = list(jobs_by_key.keys())
    min_asof = min(asof_target for _, _, _, asof_target in schedule_entries)
    max_asof = max(asof_target for _, _, _, asof_target in schedule_entries)
    min_expiry = min(week_friday for _, _, week_friday, _ in schedule_entries)
    max_expiry = max(week_friday for _, _, week_friday, _ in schedule_entries)

    theta = ThetaClient(cfg.theta_base_url, timeout_s=cfg.timeout_s, verbose=verbose_skips)
    back_pad = max(0, int(cfg.max_backward_days_for_expiry_close))
    fwd_pad = max(4, int(cfg.max_forward_days_for_asof))
    preload_start = min(min_asof, min_expiry - timedelta(days=back_pad))
    preload_end = max(max_expiry, max_asof + timedelta(days=fwd_pad))

    raw_closes_by_ticker, adj_closes_by_ticker, split_counts = preload_stock_closes(
        theta=theta,
        tickers=tickers,
        start=preload_start,
        end=preload_end,
        cfg=cfg,
        stock_source=cfg.stock_source,
    )
    dividend_histories = preload_dividend_histories(
        tickers=tickers,
        start=preload_start,
        end=preload_end,
        cfg=cfg,
    )

    option_chain_cache: Dict[Tuple[str, date, date, Optional[int]], pd.DataFrame] = {}
    cache_lock = threading.Lock()

    rows: List[dict] = []
    drop_logs: List[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as executor:
        futures = {
            executor.submit(
                process_one,
                theta=theta,
                cfg=cfg,
                ticker=ticker,
                asof_target=asof_target,
                week_monday=week_monday,
                week_friday=week_friday,
                raw_closes_by_ticker=raw_closes_by_ticker,
                adj_closes_by_ticker=adj_closes_by_ticker,
                split_event_counts=split_counts,
                dividend_histories=dividend_histories,
                option_chain_cache=option_chain_cache,
                cache_lock=cache_lock,
                targets=targets,
                prn_version=prn_version,
                prn_config_hash=prn_config_hash,
                prn_asof_tz=prn_asof_tz,
                prn_asof_close_time=prn_asof_close_time,
            ): (ticker, week_monday, week_friday, asof_target)
            for (ticker, week_monday, week_friday, asof_target), targets in jobs_by_key.items()
        }

        for future in as_completed(futures):
            ticker, week_monday, week_friday, asof_target = futures[future]
            try:
                result_rows, drop_log = future.result()
            except Exception as exc:
                result_rows = _failure_rows_for_targets(
                    ticker=ticker,
                    targets=jobs_by_key[(ticker, week_monday, week_friday, asof_target)],
                    week_monday=week_monday,
                    week_friday=week_friday,
                    asof_target=asof_target,
                    asof_date_used=None,
                    prn_version=prn_version,
                    prn_config_hash=prn_config_hash,
                    tz_name=prn_asof_tz,
                    close_time=prn_asof_close_time,
                    drop_reason="exception",
                )
                drop_log = {
                    "ticker": ticker,
                    "week_monday": week_monday.isoformat(),
                    "week_friday": week_friday.isoformat(),
                    "asof_target": asof_target.isoformat(),
                    "drop_reason": "exception",
                    "detail": str(exc),
                }
                if verbose_skips:
                    print(f"[ERR] {ticker} week={week_monday} asof_target={asof_target}: {exc}")

            rows.extend(result_rows)
            if drop_log is not None:
                drop_logs.append(drop_log)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        out_df = pd.DataFrame(columns=POLYMARKET_EXACT_COLUMNS)
    else:
        for column in POLYMARKET_EXACT_COLUMNS:
            if column not in out_df.columns:
                out_df[column] = np.nan
        out_df = out_df.reindex(columns=POLYMARKET_EXACT_COLUMNS + [c for c in out_df.columns if c not in POLYMARKET_EXACT_COLUMNS])
        out_df = out_df.sort_values(["ticker", "week_friday", "asof_target", "threshold", "market_id"]).reset_index(drop=True)

    coverage_counts = (
        out_df.get("coverage_status", pd.Series(dtype=str)).fillna("missing").value_counts().to_dict()
        if not out_df.empty
        else {}
    )
    drop_reason_counts = (
        out_df.get("drop_reason", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .value_counts()
        .to_dict()
        if not out_df.empty
        else {}
    )
    ok_rows = int((out_df.get("coverage_status", pd.Series(dtype=str)) == "ok").sum()) if not out_df.empty else 0
    required_market_snapshots = len(rows)
    missing_rows = max(required_market_snapshots - ok_rows, 0)

    return PolymarketExactPrnBuildResult(
        rows=out_df,
        drop_logs=drop_logs,
        required_market_snapshots=required_market_snapshots,
        ok_market_snapshots=ok_rows,
        missing_market_snapshots=missing_rows,
        coverage_counts={str(k): int(v) for k, v in coverage_counts.items()},
        drop_reason_counts={str(k): int(v) for k, v in drop_reason_counts.items()},
        prn_version=prn_version,
        prn_config_hash=prn_config_hash,
    )
