"""Service layer for bar history data loading, caching, and processing."""

import hashlib
import json
import logging
import math
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
from app.services.polymarket_quality import load_market_quality_map
from app.services.run_csv_files import (
    combined_run_csv_size,
    get_run_csv_paths,
    has_run_csv,
    iter_deduped_csv_rows,
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


def _resolve_run_dir(run_id: Optional[str]) -> Path:
    """Resolve a weekly-history run directory with accessible price history."""
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
            if candidate.exists() and has_run_csv(candidate, "price_history.csv"):
                run_dir = candidate
            else:
                # Pointer is stale; fall back
                run_dir = _fallback_latest_run_dir(base_dir)
        else:
            run_dir = _fallback_latest_run_dir(base_dir)

    if not has_run_csv(run_dir, "price_history.csv"):
        raise FileNotFoundError(f"price_history.csv not found in {run_dir.name}")

    return run_dir


def _fallback_latest_run_dir(base_dir: Path) -> Path:
    """Find the most recent run directory by name (backward-compat fallback)."""
    run_dirs = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and has_run_csv(d, "price_history.csv")],
        reverse=True,
    )
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
    csv_paths: List[Path],
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
        "csv_path": str(csv_paths[0]),
        "csv_paths": [str(path) for path in csv_paths],
        "rows_scanned": 0,
        "rows_filtered": 0,
    }

    for row in iter_deduped_csv_rows(csv_paths, "price_history.csv"):
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
    run_dir = _resolve_run_dir(request.run_id)
    csv_paths = get_run_csv_paths(run_dir, "price_history.csv")
    run_id = run_dir.name

    # Load and filter bars
    bars, metadata = _load_and_filter_bars(
        csv_paths,
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


def _load_event_slugs(run_dir: Path) -> Dict[str, str]:
    """Load market_id -> event_slug mapping from weekly_markets sources in the run dir."""
    weekly_markets_paths = get_run_csv_paths(run_dir, "weekly_markets.csv")
    mapping: Dict[str, str] = {}
    if not weekly_markets_paths:
        return mapping
    try:
        for row in iter_deduped_csv_rows(weekly_markets_paths, "weekly_markets.csv"):
            mid = row.get("market_id", "").strip()
            slug = row.get("event_slug", "").strip()
            if mid and slug and mid not in mapping:
                mapping[mid] = slug
    except Exception:
        pass
    return mapping


def _load_gamma_volumes(run_dir: Path) -> Dict[str, Optional[float]]:
    """Load market_id -> gamma_volume mapping from weekly_markets sources."""
    weekly_markets_paths = get_run_csv_paths(run_dir, "weekly_markets.csv")
    mapping: Dict[str, Optional[float]] = {}
    if not weekly_markets_paths:
        return mapping
    try:
        for row in iter_deduped_csv_rows(weekly_markets_paths, "weekly_markets.csv"):
            mid = row.get("market_id", "").strip()
            if not mid or mid in mapping:
                continue
            vol_str = row.get("gamma_volume", "")
            if vol_str:
                try:
                    vol = float(vol_str)
                    if math.isfinite(vol):
                        mapping[mid] = vol
                        continue
                except (ValueError, TypeError):
                    pass
            mapping[mid] = None
    except Exception:
        pass
    return mapping


# Quality tier thresholds
LOW_VOLUME_THRESHOLD = 5000.0   # USD — markets below this are flagged as low_volume
# "Suspect midprice" heuristic: detects empty-book pattern (flat ~0.50 with spikes)
MIDPRICE_CLUSTER_BAND = (0.40, 0.60)  # price band considered "near midprice"
SUSPECT_CLUSTER_RATIO = 0.25          # flag if >=25% of bars sit in the midprice band
SUSPECT_JUMP_THRESHOLD = 0.25         # AND at least one jump >= 0.25 in a single bar
# Relative point density: strikes with far fewer points than the median are likely illiquid
POINT_DENSITY_SUSPECT_RATIO = 0.20    # flag if total_points < 20% of median across strikes


def get_bars_by_strike(request: ByStrikeRequest) -> ByStrikeResponse:
    """Load bars from price_history.csv grouped by strike for a single ticker.

    Returns one StrikeSeries per unique threshold value, sorted ascending.
    """
    log = logging.getLogger("bars.by_strike")

    run_dir = _resolve_run_dir(request.run_id)
    csv_paths = get_run_csv_paths(run_dir, "price_history.csv")
    run_id = run_dir.name
    event_slug_map = _load_event_slugs(run_dir)
    gamma_volume_map = _load_gamma_volumes(run_dir)
    market_quality_map = load_market_quality_map(run_dir)

    time_min_ms = _parse_timestamp(request.time_min) if request.time_min else None
    time_max_ms = _parse_timestamp(request.time_max) if request.time_max else None

    # (strike, market_id) -> list of BarDataPoint
    groups: Dict[Tuple[float, Optional[str]], List[BarDataPoint]] = {}
    rows_scanned = 0
    rows_matched = 0
    nan_dropped = 0

    for row in iter_deduped_csv_rows(csv_paths, "price_history.csv"):
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

    # Build per-strike series, sorted by strike ascending.
    # Two-pass: first compute metrics, then classify quality using relative point density.
    strikes_list: List[StrikeSeries] = []
    total_stale_runs = 0
    max_stale_hours_all = 0.0
    total_gaps = 0

    # --- Pass 1: compute per-strike metrics ---
    strike_metrics: List[dict] = []
    sorted_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1] or ""))

    for strike, market_id in sorted_keys:
        all_bars = groups[(strike, market_id)]
        all_bars.sort(key=lambda x: x.timestamp_ms)
        total = len(all_bars)

        stale_runs = 0
        max_stale_ms = 0
        gap_count = 0
        gap_threshold_ms = 4 * 3600_000  # 4 hours
        mid_cluster_count = 0
        max_jump = 0.0

        for idx in range(len(all_bars)):
            p = all_bars[idx].price
            if MIDPRICE_CLUSTER_BAND[0] <= p <= MIDPRICE_CLUSTER_BAND[1]:
                mid_cluster_count += 1
            if idx > 0:
                dt = all_bars[idx].timestamp_ms - all_bars[idx - 1].timestamp_ms
                if dt > gap_threshold_ms:
                    gap_count += 1
                if all_bars[idx].price == all_bars[idx - 1].price:
                    stale_runs += 1
                    max_stale_ms = max(max_stale_ms, dt)
                jump = abs(all_bars[idx].price - all_bars[idx - 1].price)
                if jump > max_jump:
                    max_jump = jump

        total_stale_runs += stale_runs
        max_stale_hours_all = max(max_stale_hours_all, max_stale_ms / 3600_000)
        total_gaps += gap_count

        stale_ratio = stale_runs / max(1, total - 1) if total > 1 else 0.0
        mid_cluster_ratio = mid_cluster_count / max(1, total) if total > 0 else 0.0
        vol = gamma_volume_map.get(market_id) if market_id else None

        strike_metrics.append({
            "strike": strike,
            "market_id": market_id,
            "all_bars": all_bars,
            "total": total,
            "stale_ratio": stale_ratio,
            "mid_cluster_ratio": mid_cluster_ratio,
            "max_jump": max_jump,
            "vol": vol,
        })

    # --- Pass 2: classify quality using relative point density ---
    point_counts = sorted(m["total"] for m in strike_metrics)
    median_points = point_counts[len(point_counts) // 2] if point_counts else 1
    density_threshold = median_points * POINT_DENSITY_SUSPECT_RATIO

    for m in strike_metrics:
        vol = m["vol"]
        mid_cluster_ratio = m["mid_cluster_ratio"]
        max_jump = m["max_jump"]
        stale_ratio = m["stale_ratio"]
        total = m["total"]

        quality = "good"
        if vol is not None and vol < LOW_VOLUME_THRESHOLD:
            quality = "low_volume"
        elif stale_ratio >= 0.35:
            quality = "stale"
        elif mid_cluster_ratio >= SUSPECT_CLUSTER_RATIO and max_jump >= SUSPECT_JUMP_THRESHOLD:
            quality = "suspect"
        elif total < density_threshold and max_jump >= SUSPECT_JUMP_THRESHOLD:
            quality = "suspect"

        all_bars = m["all_bars"]
        if total > request.max_points_per_strike:
            all_bars = _downsample_bars(all_bars, request.max_points_per_strike)
        strikes_list.append(StrikeSeries(
            strike=m["strike"],
            strike_label=str(int(m["strike"])) if m["strike"] == int(m["strike"]) else f"{m['strike']:.2f}",
            market_id=m["market_id"],
            event_slug=event_slug_map.get(m["market_id"]) if m["market_id"] else None,
            total_points=total,
            returned_points=len(all_bars),
            bars=all_bars,
            gamma_volume=vol,
            stale_ratio=round(stale_ratio, 3),
            midprice_cluster_ratio=round(mid_cluster_ratio, 3),
            max_jump=round(max_jump, 4),
            quality=quality,
            market_quality=market_quality_map.get(str(m["market_id"])) if m["market_id"] else None,
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
            "csv_path": str(csv_paths[0]),
            "csv_paths": [str(path) for path in csv_paths],
            "rows_scanned": rows_scanned,
            "rows_matched": rows_matched,
            "nan_dropped": nan_dropped,
            "strikes_count": len(strikes_list),
            "stale_run_count": total_stale_runs,
            "max_stale_hours": round(max_stale_hours_all, 2),
            "gap_count": total_gaps,
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

        price_csv_paths = get_run_csv_paths(run_dir, "price_history.csv")
        manifest_json = run_dir / "manifest.json"

        if not price_csv_paths:
            continue

        run_info = {
            "run_id": run_dir.name,
            "has_price_history": True,
            "price_history_size": combined_run_csv_size(run_dir, "price_history.csv"),
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
    run_dir = _resolve_run_dir(run_id)
    csv_paths = get_run_csv_paths(run_dir, "price_history.csv")
    resolved_run_id = run_dir.name

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

    for row in iter_deduped_csv_rows(csv_paths, "price_history.csv"):
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
            "csv_path": str(csv_paths[0]),
            "csv_paths": [str(path) for path in csv_paths],
            "rows_scanned": rows_scanned,
            "rows_matched": rows_matched,
            "weeks_count": len(weeks_sorted),
        },
    }

    _TRADING_WEEKS_CACHE[cache_key] = (datetime.now().timestamp(), payload)
    return payload
