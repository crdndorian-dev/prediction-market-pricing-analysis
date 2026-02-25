from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from app.models.markets import (
    MarketsJobStatus,
    MarketsProgress,
    MarketsRefreshRequest,
    MarketsRefreshResult,
    MarketsSeriesByTickerResponse,
    MarketsSeriesPoint,
    MarketsSeriesResponse,
    MarketsSummaryItem,
    MarketsSummaryResponse,
)

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "07-polymarket-markets-refresh-v1.0.py"
WEEKLY_HISTORY_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket" / "weekly_history"
RUNS_DIR = WEEKLY_HISTORY_DIR / "runs"
LATEST_POINTER_PATH = WEEKLY_HISTORY_DIR / "latest.json"

_SUMMARY_CACHE: Dict[str, Tuple[float, MarketsSummaryResponse]] = {}
_SERIES_CACHE: Dict[str, Tuple[float, MarketsSeriesResponse]] = {}
_SERIES_BY_TICKER_CACHE: Dict[str, Tuple[float, MarketsSeriesByTickerResponse]] = {}
_CACHE_TTL_SECONDS = 600

_PROGRESS_RE = re.compile(r"\[Markets\] PROGRESS stage=(?P<stage>\w+) current=(?P<current>\d+) total=(?P<total>\d+)")
_RUN_ID_RE = re.compile(r"run_id=([0-9A-Za-z_-]+)")
_WEEK_RE = re.compile(r"week_friday=([0-9]{4}-[0-9]{2}-[0-9]{2})")


# -----------------------------
# Utilities
# -----------------------------


def _read_latest_run_id() -> Optional[str]:
    if not LATEST_POINTER_PATH.exists():
        return None
    try:
        payload = json.loads(LATEST_POINTER_PATH.read_text())
    except Exception:
        return None
    return payload.get("run_id") if isinstance(payload, dict) else None


def _fallback_latest_run_dir() -> Path:
    run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No weekly history runs found")
    return run_dirs[0]


def _resolve_run_dir(run_id: Optional[str]) -> Path:
    if run_id:
        run_dir = RUNS_DIR / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_id}")
        return run_dir

    latest_id = _read_latest_run_id()
    if latest_id:
        candidate = RUNS_DIR / latest_id
        if candidate.exists():
            return candidate
    return _fallback_latest_run_dir()


def _parse_iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}' (expected YYYY-MM-DD).") from exc


def _resolve_week_friday(week_friday: Optional[str], tz_name: str = "America/New_York") -> date:
    parsed = _parse_iso_date(week_friday)
    if parsed:
        return parsed
    now_local = datetime.now(timezone.utc).astimezone(ZoneInfo(tz_name))
    weekday = now_local.weekday()
    if weekday >= 5:
        return now_local.date() - timedelta(days=weekday - 4)
    return now_local.date() + timedelta(days=(4 - weekday))


def _week_bounds(week_friday: date) -> Tuple[str, str, str]:
    monday = week_friday - timedelta(days=4)
    sunday = monday + timedelta(days=6)
    return monday.isoformat(), week_friday.isoformat(), sunday.isoformat()


def _cache_get(cache: Dict[str, Tuple[float, object]], key: str):
    if key in cache:
        ts, value = cache[key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return value
        del cache[key]
    return None


def _cache_put(cache: Dict[str, Tuple[float, object]], key: str, value: object):
    cache[key] = (time.time(), value)


# -----------------------------
# Job manager
# -----------------------------


def _build_command(payload: MarketsRefreshRequest) -> List[str]:
    cmd = [sys.executable, str(SCRIPT_PATH)]
    if payload.run_id:
        cmd.extend(["--run-id", payload.run_id])
    if payload.week_friday:
        cmd.extend(["--week-friday", payload.week_friday])
    if payload.tickers:
        cmd.extend(["--tickers", ",".join(payload.tickers)])
    if payload.force_refresh:
        cmd.append("--force-refresh")
    return cmd


class MarketsJob:
    def __init__(self, job_id: str, payload: MarketsRefreshRequest) -> None:
        self.job_id = job_id
        self.payload = payload
        self.status = "queued"
        self.progress: Optional[MarketsProgress] = None
        self.result: Optional[MarketsRefreshResult] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_lines: List[str] = []
        self._stderr_lines: List[str] = []
        self._run_id: Optional[str] = None
        self._week_friday: Optional[str] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def to_status(self) -> MarketsJobStatus:
        return MarketsJobStatus(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    def _update_progress(self, line: str) -> None:
        match = _PROGRESS_RE.search(line)
        if not match:
            return
        self.progress = MarketsProgress(
            stage=match.group("stage"),
            current=int(match.group("current")),
            total=int(match.group("total")),
        )

    def _parse_metadata(self, line: str) -> None:
        match = _RUN_ID_RE.search(line)
        if match:
            self._run_id = match.group(1)
        week_match = _WEEK_RE.search(line)
        if week_match:
            self._week_friday = week_match.group(1)

    def _snapshot_result(self, ok: bool, cmd: List[str], start_ts: float, run_dir: Optional[Path]) -> None:
        self.result = MarketsRefreshResult(
            ok=ok,
            run_id=self._run_id,
            week_friday=self._week_friday,
            run_dir=str(run_dir.relative_to(BASE_DIR)) if run_dir else None,
            stdout="".join(self._stdout_lines),
            stderr="".join(self._stderr_lines),
            duration_s=round(time.time() - start_ts, 3),
            command=cmd,
        )

    def _run(self) -> None:
        self.started_at = datetime.utcnow()
        self.status = "running"

        cmd = _build_command(self.payload)
        start_ts = time.time()
        run_dir: Optional[Path] = None

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._process = proc

        def read_stdout():
            nonlocal run_dir
            if proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    if line:
                        self._stdout_lines.append(line)
                        self._update_progress(line)
                        self._parse_metadata(line)
                        if self._run_id:
                            candidate = RUNS_DIR / self._run_id
                            if candidate.exists():
                                run_dir = candidate
                        self._snapshot_result(False, cmd, start_ts, run_dir)
                proc.stdout.close()

        def read_stderr():
            if proc.stderr:
                for line in iter(proc.stderr.readline, ""):
                    if line:
                        self._stderr_lines.append(line)
                        self._snapshot_result(False, cmd, start_ts, run_dir)
                proc.stderr.close()

        stdout_thread = threading.Thread(target=read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        proc.wait()
        stdout_thread.join(timeout=1)
        stderr_thread.join(timeout=1)

        if run_dir is None and self._run_id:
            candidate = RUNS_DIR / self._run_id
            run_dir = candidate if candidate.exists() else None

        ok = proc.returncode == 0
        self.status = "finished" if ok else "failed"
        if not ok:
            self.error = "Markets refresh failed."
        self._snapshot_result(ok, cmd, start_ts, run_dir)
        self.finished_at = datetime.utcnow()


class MarketsJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, MarketsJob] = {}

    def start_job(self, payload: MarketsRefreshRequest) -> str:
        job_id = uuid4().hex
        job = MarketsJob(job_id, payload)
        self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> MarketsJobStatus:
        return self._get_job(job_id).to_status()

    def list_jobs(self) -> List[MarketsJobStatus]:
        return [job.to_status() for job in self._jobs.values()]

    def _get_job(self, job_id: str) -> MarketsJob:
        if job_id not in self._jobs:
            raise KeyError(job_id)
        return self._jobs[job_id]


MARKETS_JOB_MANAGER = MarketsJobManager()


def start_markets_job(payload: MarketsRefreshRequest) -> str:
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Markets refresh script not found: {SCRIPT_PATH}")
    # Bust all series/summary caches so the new run's data is read fresh.
    _clear_series_caches()
    return MARKETS_JOB_MANAGER.start_job(payload)


def get_markets_job(job_id: str) -> MarketsJobStatus:
    return MARKETS_JOB_MANAGER.get_status(job_id)


# -----------------------------
# Summary + series
# -----------------------------


def get_markets_summary(
    week_friday: Optional[str] = None,
    run_id: Optional[str] = None,
) -> MarketsSummaryResponse:
    week_date = _resolve_week_friday(week_friday)
    monday, friday, sunday = _week_bounds(week_date)
    run_dir = _resolve_run_dir(run_id)
    trading_universe_tickers: List[str] = []
    weekly_markets_path = run_dir / "weekly_markets.csv"
    if weekly_markets_path.exists():
        try:
            weekly_df = pd.read_csv(weekly_markets_path, usecols=["ticker", "week_friday"])
        except ValueError:
            weekly_df = pd.read_csv(weekly_markets_path)
        if "ticker" in weekly_df.columns:
            weekly_df["ticker"] = weekly_df["ticker"].astype(str).str.upper()
            if "week_friday" in weekly_df.columns:
                filtered = weekly_df[weekly_df["week_friday"] == friday]
                if not filtered.empty:
                    weekly_df = filtered
            trading_universe_tickers = sorted(
                set(weekly_df["ticker"].dropna().tolist())
            )
    prn_path = run_dir / "markets_prn_hourly.csv"
    if not prn_path.exists():
        last_refresh = None
        refresh_path = run_dir / "markets_refresh.json"
        if refresh_path.exists():
            try:
                payload = json.loads(refresh_path.read_text())
                last_refresh = payload.get("created_at_utc") or payload.get("updated_at_utc")
            except Exception:
                last_refresh = None
        return MarketsSummaryResponse(
            run_id=run_dir.name,
            week_friday=friday,
            week_monday=monday,
            week_sunday=sunday,
            last_refresh_utc=last_refresh,
            trading_universe_tickers=trading_universe_tickers,
            markets=[],
        )

    cache_key = f"summary|{run_dir.name}|{friday}"
    cached = _cache_get(_SUMMARY_CACHE, cache_key)
    if cached:
        return cached

    df = pd.read_csv(prn_path)
    if "week_friday" in df.columns:
        df = df[df["week_friday"] == friday]
    if df.empty:
        summary = MarketsSummaryResponse(
            run_id=run_dir.name,
            week_friday=friday,
            week_monday=monday,
            week_sunday=sunday,
            last_refresh_utc=None,
            trading_universe_tickers=trading_universe_tickers,
            markets=[],
        )
        _cache_put(_SUMMARY_CACHE, cache_key, summary)
        return summary

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

    items: List[MarketsSummaryItem] = []
    for (ticker, threshold), group in df.groupby(["ticker", "threshold"], dropna=True):
        last_ts = group["timestamp_utc"].max()
        items.append(
            MarketsSummaryItem(
                ticker=str(ticker),
                threshold=float(threshold),
                market_id=str(group["market_id"].iloc[0]) if "market_id" in group.columns else None,
                event_id=str(group["event_id"].iloc[0]) if "event_id" in group.columns else None,
                event_endDate=str(group["event_endDate"].iloc[0]) if "event_endDate" in group.columns else None,
                points=len(group),
                last_timestamp_utc=last_ts.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(last_ts) else None,
                has_polymarket=group["polymarket_buy"].notna().any() if "polymarket_buy" in group.columns else False,
                has_prn=group["pRN"].notna().any() if "pRN" in group.columns else False,
            )
        )

    last_refresh = None
    refresh_path = run_dir / "markets_refresh.json"
    if refresh_path.exists():
        try:
            payload = json.loads(refresh_path.read_text())
            last_refresh = payload.get("created_at_utc") or payload.get("updated_at_utc")
        except Exception:
            last_refresh = None

    summary = MarketsSummaryResponse(
        run_id=run_dir.name,
        week_friday=friday,
        week_monday=monday,
        week_sunday=sunday,
        last_refresh_utc=last_refresh,
        trading_universe_tickers=trading_universe_tickers,
        markets=sorted(items, key=lambda item: (item.ticker, item.threshold)),
    )
    _cache_put(_SUMMARY_CACHE, cache_key, summary)
    return summary


def _clear_series_caches() -> None:
    """Bust all series caches (called when a new refresh job starts)."""
    _SERIES_CACHE.clear()
    _SERIES_BY_TICKER_CACHE.clear()
    _SUMMARY_CACHE.clear()


def _build_series_point(row: pd.Series, col_flags: Dict[str, bool]) -> MarketsSeriesPoint:
    """
    Build a MarketsSeriesPoint from a DataFrame row.

    Bid/ask resolution (no data leakage — values are read as-of each timestamp):
      - polymarket_ask  → 'polymarket_ask' column if present, else fallback to 'polymarket_buy'
      - polymarket_bid  → 'polymarket_bid' column if present, else None
      - polymarket_buy  → kept verbatim for backward compat
    """
    def _f(col: str) -> Optional[float]:
        if col_flags.get(col) and pd.notna(row[col]):
            return float(row[col])
        return None

    buy = _f("polymarket_buy")
    bid = _f("polymarket_bid")
    # ask: use dedicated column if available, otherwise fall back to buy price
    ask = _f("polymarket_ask") if col_flags.get("polymarket_ask") else buy

    return MarketsSeriesPoint(
        timestamp_utc=row["timestamp_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        polymarket_buy=buy,
        polymarket_bid=bid,
        polymarket_ask=ask,
        pRN=_f("pRN"),
        spot=_f("spot"),
    )


def _col_flags(df: pd.DataFrame) -> Dict[str, bool]:
    cols = {"polymarket_buy", "polymarket_bid", "polymarket_ask", "pRN", "spot"}
    return {c: c in df.columns for c in cols}


def _load_series_rows(
    prn_path: Path,
    ticker: str,
    threshold: float,
    week_friday: str,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for chunk in pd.read_csv(prn_path, chunksize=100_000):
        if "ticker" not in chunk.columns or "threshold" not in chunk.columns:
            continue
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper()
        chunk = chunk[chunk["ticker"] == ticker]
        if chunk.empty:
            continue
        chunk["threshold"] = pd.to_numeric(chunk["threshold"], errors="coerce")
        chunk = chunk[np.isclose(chunk["threshold"], threshold)]
        if chunk.empty:
            continue
        if "week_friday" in chunk.columns:
            chunk = chunk[chunk["week_friday"] == week_friday]
        if chunk.empty:
            continue
        rows.append(chunk)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])
    df = df.sort_values("timestamp_utc")
    return df


def get_markets_series(
    ticker: str,
    threshold: float,
    week_friday: Optional[str] = None,
    run_id: Optional[str] = None,
) -> MarketsSeriesResponse:
    run_dir = _resolve_run_dir(run_id)
    prn_path = run_dir / "markets_prn_hourly.csv"
    if not prn_path.exists():
        raise FileNotFoundError("markets_prn_hourly.csv not found")

    week_date = _resolve_week_friday(week_friday)
    week_key = week_date.isoformat()
    ticker = ticker.strip().upper()
    threshold = float(threshold)

    cache_key = f"series|{run_dir.name}|{week_key}|{ticker}|{threshold}"
    cached = _cache_get(_SERIES_CACHE, cache_key)
    if cached:
        return cached

    df = _load_series_rows(prn_path, ticker, threshold, week_key)
    if df.empty:
        raise FileNotFoundError("No series rows for the requested ticker/threshold/week")

    flags = _col_flags(df)
    points = [_build_series_point(row, flags) for _, row in df.iterrows()]

    response = MarketsSeriesResponse(
        run_id=run_dir.name,
        ticker=ticker,
        threshold=threshold,
        week_friday=week_key,
        market_id=str(df["market_id"].iloc[0]) if "market_id" in df.columns else None,
        event_id=str(df["event_id"].iloc[0]) if "event_id" in df.columns else None,
        points=points,
    )
    _cache_put(_SERIES_CACHE, cache_key, response)
    return response


def get_markets_series_by_ticker(
    ticker: str,
    week_friday: Optional[str] = None,
    run_id: Optional[str] = None,
) -> MarketsSeriesByTickerResponse:
    run_dir = _resolve_run_dir(run_id)
    prn_path = run_dir / "markets_prn_hourly.csv"
    if not prn_path.exists():
        raise FileNotFoundError("markets_prn_hourly.csv not found")

    week_date = _resolve_week_friday(week_friday)
    week_key = week_date.isoformat()
    ticker = ticker.strip().upper()

    cache_key = f"series_by_ticker|{run_dir.name}|{week_key}|{ticker}"
    cached = _cache_get(_SERIES_BY_TICKER_CACHE, cache_key)
    if cached:
        return cached

    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(prn_path, chunksize=100_000):
        if "ticker" not in chunk.columns:
            continue
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper()
        chunk = chunk[chunk["ticker"] == ticker]
        if chunk.empty:
            continue
        if "week_friday" in chunk.columns:
            chunk = chunk[chunk["week_friday"] == week_key]
        if chunk.empty:
            continue
        frames.append(chunk)

    if not frames:
        raise FileNotFoundError("No series rows for the requested ticker/week")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp_utc"])

    flags = _col_flags(df)
    strikes: List[MarketsSeriesResponse] = []
    for threshold, group in df.groupby("threshold", dropna=True):
        group = group.sort_values("timestamp_utc")
        points = [_build_series_point(row, flags) for _, row in group.iterrows()]
        strikes.append(
            MarketsSeriesResponse(
                run_id=run_dir.name,
                ticker=ticker,
                threshold=float(threshold),
                week_friday=week_key,
                market_id=str(group["market_id"].iloc[0]) if "market_id" in group.columns else None,
                event_id=str(group["event_id"].iloc[0]) if "event_id" in group.columns else None,
                points=points,
            )
        )

    response = MarketsSeriesByTickerResponse(
        run_id=run_dir.name,
        ticker=ticker,
        week_friday=week_key,
        strikes=sorted(strikes, key=lambda item: item.threshold),
    )
    _cache_put(_SERIES_BY_TICKER_CACHE, cache_key, response)
    return response
