"""Service layer for bar history data loading, caching, and processing."""

import csv
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.models.bars import (
    BarDataPoint,
    BarsRequest,
    BarsResponse,
    ByStrikeRequest,
    ByStrikeResponse,
    StrikeSeries,
)


# Project root (same convention as market_map.py)
BASE_DIR = Path(__file__).resolve().parents[5]
WEEKLY_HISTORY_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket" / "weekly_history"
WEEKLY_HISTORY_RUNS_DIR = WEEKLY_HISTORY_DIR / "runs"
LATEST_POINTER_PATH = WEEKLY_HISTORY_DIR / "latest.json"

# In-memory cache: key -> (timestamp, data)
_BARS_CACHE: Dict[str, Tuple[float, BarsResponse]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes

# Trading weeks cache: key -> (timestamp, response dict)
_TRADING_WEEKS_CACHE: Dict[str, Tuple[float, dict]] = {}
_TRADING_WEEKS_TTL_SECONDS = 600  # 10 minutes


def _get_cache_key(request: BarsRequest) -> str:
    """Generate cache key from request parameters."""
    key_parts = [
        request.run_id or "",
        request.market_id or "",
        request.ticker or "",
        request.time_min or "",
        request.time_max or "",
        str(request.max_points),
        request.view_mode,
    ]
    key_str = "|".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()


def _get_from_cache(cache_key: str) -> Optional[BarsResponse]:
    """Get response from cache if still valid."""
    if cache_key in _BARS_CACHE:
        cached_time, cached_response = _BARS_CACHE[cache_key]
        age = datetime.now().timestamp() - cached_time
        if age < _CACHE_TTL_SECONDS:
            return cached_response
        else:
            del _BARS_CACHE[cache_key]
    return None


def _put_in_cache(cache_key: str, response: BarsResponse) -> None:
    """Put response in cache with current timestamp."""
    _BARS_CACHE[cache_key] = (datetime.now().timestamp(), response)


def _read_latest_run_id() -> Optional[str]:
    """Read run_id from latest.json pointer file, if it exists."""
    if not LATEST_POINTER_PATH.exists():
        return None
    try:
        data = json.loads(LATEST_POINTER_PATH.read_text())
        return data.get("run_id") if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _find_price_history_csv(run_id: Optional[str]) -> Path:
    """Find price_history.csv file for the given run_id or the active/latest run."""
    base_dir = WEEKLY_HISTORY_RUNS_DIR

    if not base_dir.exists():
        raise FileNotFoundError(f"Weekly history runs directory not found: {base_dir}")

    if run_id:
        run_dir = base_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_id}")
    else:
        # Try latest.json pointer first, then fall back to name-sorted latest
        latest_id = _read_latest_run_id()
        if latest_id:
            candidate = base_dir / latest_id
            if candidate.exists() and (candidate / "price_history.csv").exists():
                run_dir = candidate
            else:
                # Pointer is stale; fall back
                run_dir = _fallback_latest_run_dir(base_dir)
        else:
            run_dir = _fallback_latest_run_dir(base_dir)

    csv_path = run_dir / "price_history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"price_history.csv not found in {run_dir.name}")

    return csv_path


def _fallback_latest_run_dir(base_dir: Path) -> Path:
    """Find the most recent run directory by name (backward-compat fallback)."""
    run_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No pipeline runs found")
    return run_dirs[0]


def _parse_timestamp(ts_str: str) -> int:
    """Parse ISO timestamp to milliseconds since epoch."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except Exception:
        raise ValueError(f"Invalid timestamp format: {ts_str}")


def _load_and_filter_bars(
    csv_path: Path,
    market_id: Optional[str],
    ticker: Optional[str],
    time_min: Optional[str],
    time_max: Optional[str],
    view_mode: str,
) -> Tuple[List[BarDataPoint], dict]:
    """Load CSV and filter bars based on request parameters.

    Time-safety: In decision_time mode, only returns bars with timestamp <= time_max.
    """
    time_min_ms = _parse_timestamp(time_min) if time_min else None
    time_max_ms = _parse_timestamp(time_max) if time_max else None

    bars: List[BarDataPoint] = []
    metadata = {
        "csv_path": str(csv_path),
        "rows_scanned": 0,
        "rows_filtered": 0,
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            metadata["rows_scanned"] += 1

            # Filter by market_id if specified
            if market_id and row.get("market_id") != market_id:
                continue

            # Filter by ticker if specified
            if ticker and row.get("ticker") != ticker:
                continue

            # Parse timestamp
            ts_str = row.get("timestamp_utc", "")
            if not ts_str:
                continue

            try:
                ts_ms = _parse_timestamp(ts_str)
            except ValueError:
                continue

            # Filter by time range
            if time_min_ms and ts_ms < time_min_ms:
                continue

            # TIME-SAFETY: In decision_time mode, exclude all bars after time_max
            if view_mode == "decision_time" and time_max_ms and ts_ms > time_max_ms:
                continue

            # In full_history mode, time_max is just a display hint (not enforced)

            # Extract price (use 'price' column if available, else 'close')
            price_str = row.get("price") or row.get("close", "")
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                continue

            # Build bar data point
            bar = BarDataPoint(
                timestamp=ts_str,
                timestamp_ms=ts_ms,
                price=price,
                open=float(row["open"]) if row.get("open") else None,
                high=float(row["high"]) if row.get("high") else None,
                low=float(row["low"]) if row.get("low") else None,
                close=float(row["close"]) if row.get("close") else None,
                volume=float(row["volume"]) if row.get("volume") else None,
            )
            bars.append(bar)

    metadata["rows_filtered"] = len(bars)

    # Sort by timestamp to ensure chronological order
    bars.sort(key=lambda b: b.timestamp_ms)

    return bars, metadata


def _downsample_bars(bars: List[BarDataPoint], max_points: int) -> List[BarDataPoint]:
    """Downsample bars to max_points using time-bucketing.

    Strategy: Divide time range into max_points buckets, take last bar per bucket.
    This preserves the final value in each time window and is time-safe.
    """
    if len(bars) <= max_points:
        return bars

    if not bars:
        return bars

    # Create time buckets
    min_ts = bars[0].timestamp_ms
    max_ts = bars[-1].timestamp_ms
    time_range = max_ts - min_ts

    if time_range == 0:
        # All bars have same timestamp, just take first N
        return bars[:max_points]

    bucket_size = time_range / max_points
    downsampled: List[BarDataPoint] = []
    current_bucket = 0
    last_bar_in_bucket: Optional[BarDataPoint] = None

    for bar in bars:
        bucket_idx = int((bar.timestamp_ms - min_ts) / bucket_size)

        if bucket_idx > current_bucket:
            # Moving to next bucket, save last bar from previous bucket
            if last_bar_in_bucket:
                downsampled.append(last_bar_in_bucket)
            current_bucket = bucket_idx
            last_bar_in_bucket = bar
        else:
            # Still in same bucket, update last bar
            last_bar_in_bucket = bar

    # Add last bar
    if last_bar_in_bucket:
        downsampled.append(last_bar_in_bucket)

    return downsampled


def get_bars(request: BarsRequest) -> BarsResponse:
    """Get bar history data with caching and downsampling.

    Time-safety: In decision_time mode, only returns bars <= time_max.
    Performance: Uses in-memory caching and efficient downsampling.
    """
    # Check cache first
    cache_key = _get_cache_key(request)
    cached = _get_from_cache(cache_key)
    if cached:
        return cached

    # Find CSV file
    csv_path = _find_price_history_csv(request.run_id)
    run_id = csv_path.parent.name

    # Load and filter bars
    bars, metadata = _load_and_filter_bars(
        csv_path,
        request.market_id,
        request.ticker,
        request.time_min,
        request.time_max,
        request.view_mode,
    )

    total_points = len(bars)

    # Downsample if needed
    downsampled = len(bars) > request.max_points
    if downsampled:
        bars = _downsample_bars(bars, request.max_points)

    # Build response
    response = BarsResponse(
        run_id=run_id,
        market_id=request.market_id,
        ticker=request.ticker,
        view_mode=request.view_mode,
        time_min=request.time_min,
        time_max=request.time_max,
        total_points=total_points,
        returned_points=len(bars),
        downsampled=downsampled,
        bars=bars,
        metadata=metadata,
    )

    # Cache response
    _put_in_cache(cache_key, response)

    return response


def _load_event_slugs(csv_path: Path) -> Dict[str, str]:
    """Load market_id -> event_slug mapping from weekly_markets.csv in the same run dir."""
    weekly_markets_path = csv_path.parent / "weekly_markets.csv"
    mapping: Dict[str, str] = {}
    if not weekly_markets_path.exists():
        return mapping
    try:
        with open(weekly_markets_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get("market_id", "").strip()
                slug = row.get("event_slug", "").strip()
                if mid and slug:
                    mapping[mid] = slug
    except Exception:
        pass
    return mapping


def get_bars_by_strike(request: ByStrikeRequest) -> ByStrikeResponse:
    """Load bars from price_history.csv grouped by strike for a single ticker.

    Returns one StrikeSeries per unique threshold value, sorted ascending.
    """
    log = logging.getLogger("bars.by_strike")

    csv_path = _find_price_history_csv(request.run_id)
    run_id = csv_path.parent.name
    event_slug_map = _load_event_slugs(csv_path)

    time_min_ms = _parse_timestamp(request.time_min) if request.time_min else None
    time_max_ms = _parse_timestamp(request.time_max) if request.time_max else None

    # (strike, market_id) -> list of BarDataPoint
    groups: Dict[Tuple[float, Optional[str]], List[BarDataPoint]] = {}
    rows_scanned = 0
    rows_matched = 0
    nan_dropped = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_scanned += 1

            if row.get("ticker") != request.ticker:
                continue
            if row.get("token_role", "yes") != request.token_role:
                continue

            ts_str = row.get("timestamp_utc", "")
            if not ts_str:
                continue
            try:
                ts_ms = _parse_timestamp(ts_str)
            except ValueError:
                continue

            if time_min_ms and ts_ms < time_min_ms:
                continue
            if time_max_ms and ts_ms > time_max_ms:
                continue

            price_str = row.get("price") or row.get("close", "")
            try:
                price = float(price_str)
            except (ValueError, TypeError):
                nan_dropped += 1
                continue

            if not math.isfinite(price):
                nan_dropped += 1
                continue

            strike_str = row.get("threshold", "")
            try:
                strike = float(strike_str)
            except (ValueError, TypeError):
                nan_dropped += 1
                continue
            if not math.isfinite(strike):
                nan_dropped += 1
                continue

            bar = BarDataPoint(
                timestamp=ts_str,
                timestamp_ms=ts_ms,
                price=price,
            )
            market_id = row.get("market_id")
            groups.setdefault((strike, market_id), []).append(bar)
            rows_matched += 1

    # Build per-strike series, sorted by strike ascending
    strikes_list: List[StrikeSeries] = []
    for strike, market_id in sorted(groups.keys(), key=lambda k: (k[0], k[1] or "")):
        all_bars = groups[(strike, market_id)]
        # Sort bars by timestamp
        all_bars.sort(key=lambda x: x.timestamp_ms)
        total = len(all_bars)
        # Downsample per strike
        if total > request.max_points_per_strike:
            all_bars = _downsample_bars(all_bars, request.max_points_per_strike)
        strikes_list.append(StrikeSeries(
            strike=strike,
            strike_label=str(int(strike)) if strike == int(strike) else f"{strike:.2f}",
            market_id=market_id,
            event_slug=event_slug_map.get(market_id) if market_id else None,
            total_points=total,
            returned_points=len(all_bars),
            bars=all_bars,
        ))

    log.info(
        "by_strike: ticker=%s run=%s scanned=%d matched=%d nan_dropped=%d strikes=%d",
        request.ticker, run_id, rows_scanned, rows_matched, nan_dropped, len(strikes_list),
    )

    return ByStrikeResponse(
        run_id=run_id,
        ticker=request.ticker,
        token_role=request.token_role,
        time_min=request.time_min,
        time_max=request.time_max,
        view_mode=request.view_mode,
        strikes=strikes_list,
        metadata={
            "csv_path": str(csv_path),
            "rows_scanned": rows_scanned,
            "rows_matched": rows_matched,
            "nan_dropped": nan_dropped,
            "strikes_count": len(strikes_list),
        },
    )


def list_bar_runs() -> dict:
    """List available pipeline runs with metadata."""
    base_dir = WEEKLY_HISTORY_RUNS_DIR

    if not base_dir.exists():
        return {"runs": []}

    active_run_id = _read_latest_run_id()
    runs = []
    for run_dir in sorted(base_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue

        price_csv = run_dir / "price_history.csv"
        manifest_json = run_dir / "manifest.json"

        if not price_csv.exists():
            continue

        run_info = {
            "run_id": run_dir.name,
            "has_price_history": price_csv.exists(),
            "price_history_size": price_csv.stat().st_size if price_csv.exists() else 0,
            "has_manifest": manifest_json.exists(),
            "is_active": run_dir.name == active_run_id,
        }

        # Load manifest if available
        if manifest_json.exists():
            try:
                with open(manifest_json, "r") as f:
                    manifest = json.load(f)
                    run_info["manifest"] = manifest
            except Exception:
                pass

        runs.append(run_info)

    return {"runs": runs}


def list_trading_weeks(ticker: str, run_id: Optional[str]) -> dict:
    """List available trading weeks (Mon-Fri) for a ticker."""
    csv_path = _find_price_history_csv(run_id)
    resolved_run_id = csv_path.parent.name

    cache_key = f"{resolved_run_id}|{ticker}"
    cached = _TRADING_WEEKS_CACHE.get(cache_key)
    if cached:
        cached_time, cached_response = cached
        age = datetime.now().timestamp() - cached_time
        if age < _TRADING_WEEKS_TTL_SECONDS:
            return cached_response
        del _TRADING_WEEKS_CACHE[cache_key]

    weeks = set()
    rows_scanned = 0
    rows_matched = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_scanned += 1
            if row.get("ticker") != ticker:
                continue

            ts_str = row.get("timestamp_utc", "")
            if not ts_str:
                continue
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                continue

            week_start = (dt.date() - timedelta(days=dt.weekday()))
            weeks.add(week_start)
            rows_matched += 1

    weeks_sorted = sorted(weeks)
    payload = {
        "run_id": resolved_run_id,
        "ticker": ticker,
        "weeks": [
            {
                "start_date": week.isoformat(),
                "end_date": (week + timedelta(days=4)).isoformat(),
            }
            for week in weeks_sorted
        ],
        "metadata": {
            "csv_path": str(csv_path),
            "rows_scanned": rows_scanned,
            "rows_matched": rows_matched,
            "weeks_count": len(weeks_sorted),
        },
    }

    _TRADING_WEEKS_CACHE[cache_key] = (datetime.now().timestamp(), payload)
    return payload
