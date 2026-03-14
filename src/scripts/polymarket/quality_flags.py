from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from option_chain.quality_flags import compute_quality_issue_count

PM_NO_RECENT_TRADE_HOURS_WARN = 24.0
PM_STALE_RATIO_WARN = 0.35
PM_MAX_STALE_HOURS_WARN = 12.0
PM_MIDPRICE_CLUSTER_RATIO_WARN = 0.25
PM_SUSPECT_JUMP_WARN = 0.25
PM_POINT_DENSITY_SUSPECT_RATIO = 0.20
PM_LOW_VOLUME_THRESHOLD_USD = 5000.0
PM_EXTREME_OTM_ABS_LOGM_FWD_WARN = 0.10
PM_TAIL_PRN_WARN_LOW = 0.05
PM_TAIL_PRN_WARN_HIGH = 0.95
QUALITY_LIVE_PREFIX = "[QUALITY_LIVE]"

OPTION_SIDE_FLAG_COLUMNS: Tuple[str, ...] = (
    "flag_asof_close_fallback",
    "flag_expiry_close_fallback",
    "flag_expiry_saturday_fallback",
    "flag_quote_close_fallback",
    "flag_low_chain_used",
    "flag_wide_rel_spread",
)

COUNTED_MARKET_FLAG_COLUMNS: Tuple[str, ...] = (
    "flag_prn_missing",
    "flag_pm_no_trade_history",
    "flag_pm_no_recent_trade",
    "flag_pm_stale_prices",
    "flag_pm_suspect_orderbook",
    "flag_pm_sparse_points",
    "flag_pm_low_volume",
    "flag_market_inactive",
    "flag_missing_token_ids",
    "flag_extreme_otm",
)

ADVISORY_FLAG_COLUMNS: Tuple[str, ...] = (
    "flag_not_relevant",
    "flag_prn_outside_curve_support",
)

COUNTED_QUALITY_FLAG_COLUMNS: Tuple[str, ...] = OPTION_SIDE_FLAG_COLUMNS + COUNTED_MARKET_FLAG_COLUMNS
QUALITY_FLAG_COLUMNS: Tuple[str, ...] = COUNTED_QUALITY_FLAG_COLUMNS + ADVISORY_FLAG_COLUMNS

QUALITY_METRIC_COLUMNS: Tuple[str, ...] = (
    "snapshot_date_used",
    "snapshot_time_used",
    "snapshot_coverage_status",
    "snapshot_drop_reason",
    "snapshot_pRN",
    "snapshot_abs_log_m_fwd",
    "yes_points",
    "stale_ratio",
    "max_stale_hours",
    "midprice_cluster_ratio",
    "max_jump",
    "hours_since_last_yes_trade",
    "gamma_volume",
    "quality_issue_count",
    "quality_bucket",
)

QUALITY_OUTPUT_COLUMNS: Tuple[str, ...] = (
    "market_id",
    "event_id",
    "ticker",
    "threshold",
    "week_monday",
    "week_friday",
    "event_endDate",
    "yes_token_id",
    "no_token_id",
    *QUALITY_FLAG_COLUMNS,
    *QUALITY_METRIC_COLUMNS,
)

QUALITY_BUCKETS: Tuple[str, ...] = ("clean", "watch", "noisy")
VOLUME_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "gamma_volume",
    "volume_usd",
    "volume",
    "total_volume",
)


@dataclass
class MarketQualityBuildResult:
    rows: pd.DataFrame
    summary: Dict[str, Any]
    flag_columns: List[str]
    counted_flag_columns: List[str]


def quality_bucket_from_issue_count(value: object) -> str:
    try:
        count = float(value)
    except Exception:
        count = 0.0
    if count <= 0:
        return "clean"
    if count <= 2:
        return "watch"
    return "noisy"


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=bool)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _safe_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return float(parsed)


def _safe_int(value: object) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _coerce_asof_timestamp(value: object, *, tz_name: str, close_time: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.notna(ts):
        return ts

    parsed_date = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed_date):
        return pd.NaT
    tz = ZoneInfo(tz_name)
    close_parts = [int(part) for part in (close_time or "16:00").split(":")[:2]]
    close_hour = close_parts[0] if close_parts else 16
    close_minute = close_parts[1] if len(close_parts) > 1 else 0
    local_dt = datetime.combine(parsed_date.date(), dt_time(close_hour, close_minute), tzinfo=tz)
    return pd.Timestamp(local_dt.astimezone(ZoneInfo("UTC")))


def _load_yes_price_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "price_history.csv"
    if not path.exists():
        return pd.DataFrame(columns=["market_id", "timestamp_utc", "price"])

    frames: List[pd.DataFrame] = []
    usecols = ["market_id", "timestamp_utc", "price", "token_role"]
    for chunk in pd.read_csv(path, usecols=lambda col: col in usecols, chunksize=100_000):
        if "market_id" not in chunk.columns or "timestamp_utc" not in chunk.columns or "price" not in chunk.columns:
            continue
        if "token_role" in chunk.columns:
            token_role = chunk["token_role"].astype(str).str.lower()
            chunk = chunk[token_role == "yes"]
        if chunk.empty:
            continue
        chunk["market_id"] = chunk["market_id"].astype(str)
        chunk["timestamp_utc"] = pd.to_datetime(chunk["timestamp_utc"], utc=True, errors="coerce")
        chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
        chunk = chunk.dropna(subset=["market_id", "timestamp_utc", "price"])
        if chunk.empty:
            continue
        frames.append(chunk[["market_id", "timestamp_utc", "price"]])

    if not frames:
        return pd.DataFrame(columns=["market_id", "timestamp_utc", "price"])

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.sort_values(["market_id", "timestamp_utc"]).drop_duplicates(
        subset=["market_id", "timestamp_utc"],
        keep="last",
    )
    return out.reset_index(drop=True)


def _latest_scheduled_snapshot(week_monday: Optional[date], week_friday: Optional[date]) -> Optional[date]:
    if week_monday is None and week_friday is None:
        return None
    if week_monday is None and week_friday is not None:
        return week_friday
    start = week_monday
    end = week_friday or week_monday
    if start is None or end is None:
        return None
    latest = start
    for offset in range(4):
        current_date = start + timedelta(days=offset)
        if current_date <= end:
            latest = current_date
    return latest


def _resolve_volume_value(row: Mapping[str, Any]) -> Optional[float]:
    for column in VOLUME_COLUMN_CANDIDATES:
        value = _safe_float(row.get(column))
        if value is not None:
            return value
    return None


def add_prn_quality_columns(prn_rows: pd.DataFrame) -> pd.DataFrame:
    if prn_rows.empty:
        out = prn_rows.copy()
        out["prn_quality_issue_count"] = pd.Series(dtype=int)
        out["prn_quality_bucket"] = pd.Series(dtype=str)
        return out

    out = prn_rows.copy()
    bool_flags = pd.DataFrame(index=out.index)
    for column in OPTION_SIDE_FLAG_COLUMNS:
        if column in out.columns:
            bool_flags[column] = _normalize_bool_series(out[column])
        else:
            bool_flags[column] = False

    out["prn_quality_issue_count"] = bool_flags.astype(int).sum(axis=1)
    out["prn_quality_bucket"] = out["prn_quality_issue_count"].map(quality_bucket_from_issue_count)
    return out


def _anchor_prn_row(
    prn_group: pd.DataFrame,
    *,
    week_monday: Optional[date],
    week_friday: Optional[date],
    tz_name: str,
    close_time: str,
) -> Dict[str, Any]:
    if prn_group.empty:
        snapshot_date = _latest_scheduled_snapshot(week_monday, week_friday)
        return {
            "snapshot_date_used": snapshot_date.isoformat() if snapshot_date else None,
            "snapshot_time_used": _coerce_asof_timestamp(snapshot_date, tz_name=tz_name, close_time=close_time),
            "snapshot_coverage_status": "missing",
            "snapshot_drop_reason": "missing_prn_rows",
            "snapshot_pRN": np.nan,
            "snapshot_abs_log_m_fwd": np.nan,
            "flag_prn_missing": True,
            "flag_prn_outside_curve_support": False,
            **{column: False for column in OPTION_SIDE_FLAG_COLUMNS},
        }

    work = prn_group.copy()
    work["snapshot_date"] = pd.to_datetime(work.get("snapshot_date"), errors="coerce").dt.date
    work["asof_time"] = pd.to_datetime(work.get("asof_time"), utc=True, errors="coerce")
    work["coverage_status"] = work.get("coverage_status", pd.Series(index=work.index, dtype=object)).fillna("missing").astype(str)
    work["drop_reason"] = work.get("drop_reason", pd.Series(index=work.index, dtype=object))
    work = work.sort_values(["snapshot_date", "asof_time"], na_position="last")

    ok_rows = work[work["coverage_status"].str.lower() == "ok"]
    selected = ok_rows.iloc[-1] if not ok_rows.empty else work.iloc[-1]
    snapshot_date = selected.get("snapshot_date")
    snapshot_time = selected.get("asof_time")
    if pd.isna(snapshot_time):
        snapshot_time = _coerce_asof_timestamp(snapshot_date, tz_name=tz_name, close_time=close_time)

    payload = {
        "snapshot_date_used": snapshot_date.isoformat() if isinstance(snapshot_date, date) else None,
        "snapshot_time_used": snapshot_time,
        "snapshot_coverage_status": str(selected.get("coverage_status") or "missing"),
        "snapshot_drop_reason": (
            None if pd.isna(selected.get("drop_reason")) else str(selected.get("drop_reason"))
        ),
        "snapshot_pRN": _safe_float(selected.get("pRN")),
        "snapshot_abs_log_m_fwd": _safe_float(selected.get("abs_log_m_fwd")),
        "flag_prn_missing": ok_rows.empty,
        "flag_prn_outside_curve_support": str(selected.get("drop_reason") or "") == "target_outside_curve_support",
    }
    for column in OPTION_SIDE_FLAG_COLUMNS:
        payload[column] = bool(_normalize_bool_series(pd.Series([selected.get(column, False)])).iloc[0])
    return payload


def _compute_trade_metrics(history: pd.DataFrame, snapshot_time: pd.Timestamp) -> Dict[str, Any]:
    if history.empty or pd.isna(snapshot_time):
        return {
            "yes_points": 0,
            "stale_ratio": 0.0,
            "max_stale_hours": 0.0,
            "midprice_cluster_ratio": 0.0,
            "max_jump": 0.0,
            "hours_since_last_yes_trade": np.nan,
            "flag_pm_no_trade_history": True,
            "flag_pm_no_recent_trade": False,
            "flag_pm_stale_prices": False,
            "flag_pm_suspect_orderbook": False,
        }

    eligible = history[history["timestamp_utc"] <= snapshot_time].copy()
    if eligible.empty:
        return {
            "yes_points": 0,
            "stale_ratio": 0.0,
            "max_stale_hours": 0.0,
            "midprice_cluster_ratio": 0.0,
            "max_jump": 0.0,
            "hours_since_last_yes_trade": np.nan,
            "flag_pm_no_trade_history": True,
            "flag_pm_no_recent_trade": False,
            "flag_pm_stale_prices": False,
            "flag_pm_suspect_orderbook": False,
        }

    eligible = eligible.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
    prices = eligible["price"].to_numpy(dtype=float)
    times = eligible["timestamp_utc"].to_numpy(dtype="datetime64[ns]")

    stale_runs = 0
    max_stale_hours = 0.0
    max_jump = 0.0
    midprice_cluster_ratio = float(((prices >= 0.40) & (prices <= 0.60)).mean()) if prices.size else 0.0
    for idx in range(1, len(prices)):
        delta_hours = (times[idx] - times[idx - 1]) / np.timedelta64(1, "h")
        if prices[idx] == prices[idx - 1]:
            stale_runs += 1
            if np.isfinite(delta_hours):
                max_stale_hours = max(max_stale_hours, float(delta_hours))
        jump = abs(float(prices[idx]) - float(prices[idx - 1]))
        if np.isfinite(jump):
            max_jump = max(max_jump, float(jump))

    stale_ratio = stale_runs / max(1, len(prices) - 1) if len(prices) > 1 else 0.0
    last_trade = eligible["timestamp_utc"].max()
    hours_since_last = (
        (snapshot_time - last_trade) / pd.Timedelta(hours=1)
        if pd.notna(last_trade) and pd.notna(snapshot_time)
        else np.nan
    )
    hours_since_last = float(hours_since_last) if pd.notna(hours_since_last) else np.nan

    return {
        "yes_points": int(len(eligible)),
        "stale_ratio": float(stale_ratio),
        "max_stale_hours": float(max_stale_hours),
        "midprice_cluster_ratio": float(midprice_cluster_ratio),
        "max_jump": float(max_jump),
        "hours_since_last_yes_trade": hours_since_last,
        "flag_pm_no_trade_history": False,
        "flag_pm_no_recent_trade": bool(np.isfinite(hours_since_last) and hours_since_last > PM_NO_RECENT_TRADE_HOURS_WARN),
        "flag_pm_stale_prices": bool(
            stale_ratio >= PM_STALE_RATIO_WARN or max_stale_hours >= PM_MAX_STALE_HOURS_WARN
        ),
        "flag_pm_suspect_orderbook": bool(
            midprice_cluster_ratio >= PM_MIDPRICE_CLUSTER_RATIO_WARN and max_jump >= PM_SUSPECT_JUMP_WARN
        ),
    }


def _build_partial_telemetry(rows: pd.DataFrame, *, total_markets: int, completed_markets: int) -> Dict[str, Any]:
    if rows.empty:
        return {
            "phase": "quality",
            "total_markets": int(total_markets),
            "completed_markets": int(completed_markets),
            "flagged_markets": 0,
            "flagged_share": 0.0,
            "bucket_counts": {bucket: 0 for bucket in QUALITY_BUCKETS},
            "top_flags": [],
            "prn_coverage_counts": {},
            "top_problem_tickers": [],
        }

    if "quality_issue_count" in rows.columns:
        issue_count = pd.to_numeric(rows["quality_issue_count"], errors="coerce").fillna(0)
    else:
        issue_count = pd.Series(
            [
                compute_quality_issue_count({column: row.get(column) for column in COUNTED_QUALITY_FLAG_COLUMNS})
                for row in rows.to_dict("records")
            ],
            index=rows.index,
            dtype=float,
        )
    flagged_mask = issue_count.gt(0)
    flag_counts = []
    for column in COUNTED_QUALITY_FLAG_COLUMNS:
        if column not in rows.columns:
            continue
        count = int(_normalize_bool_series(rows[column]).sum())
        if count <= 0:
            continue
        flag_counts.append(
            {
                "name": column,
                "count": count,
                "share": round(count / max(len(rows), 1), 6),
            }
        )
    flag_counts.sort(key=lambda item: (-item["count"], item["name"]))

    ticker_rows = (
        rows.assign(__flagged=flagged_mask, __issue_count=issue_count)
        .groupby("ticker", dropna=False)
        .agg(
            market_count=("market_id", "count"),
            flagged_market_count=("__flagged", "sum"),
            avg_issue_count=("__issue_count", "mean"),
        )
        .reset_index()
    )
    if not ticker_rows.empty:
        ticker_rows["flagged_share"] = ticker_rows["flagged_market_count"] / ticker_rows["market_count"].replace(0, np.nan)
    top_problem_tickers = []
    for record in ticker_rows.sort_values(
        ["flagged_market_count", "avg_issue_count", "market_count", "ticker"],
        ascending=[False, False, False, True],
    ).head(5).to_dict("records"):
        top_problem_tickers.append(
            {
                "ticker": str(record.get("ticker") or ""),
                "market_count": int(record.get("market_count") or 0),
                "flagged_market_count": int(record.get("flagged_market_count") or 0),
                "flagged_share": round(float(record.get("flagged_share") or 0.0), 6),
                "avg_issue_count": round(float(record.get("avg_issue_count") or 0.0), 4),
            }
        )

    bucket_counts = {
        bucket: (
            int((rows["quality_bucket"].astype(str) == bucket).sum())
            if "quality_bucket" in rows.columns
            else int((issue_count.map(quality_bucket_from_issue_count) == bucket).sum())
        )
        for bucket in QUALITY_BUCKETS
    }
    prn_coverage_counts = (
        rows["snapshot_coverage_status"].fillna("missing").astype(str).value_counts().to_dict()
        if "snapshot_coverage_status" in rows.columns
        else {}
    )

    return {
        "phase": "quality",
        "total_markets": int(total_markets),
        "completed_markets": int(completed_markets),
        "flagged_markets": int(flagged_mask.sum()),
        "flagged_share": round(float(flagged_mask.mean()), 6),
        "bucket_counts": bucket_counts,
        "top_flags": flag_counts[:5],
        "prn_coverage_counts": {str(key): int(value) for key, value in prn_coverage_counts.items()},
        "top_problem_tickers": top_problem_tickers,
    }


def emit_quality_live_telemetry(payload: Mapping[str, Any]) -> None:
    print(f"{QUALITY_LIVE_PREFIX} {json.dumps(dict(payload), sort_keys=True)}", flush=True)


def _top_flag_summary(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for column in COUNTED_QUALITY_FLAG_COLUMNS:
        if column not in rows.columns:
            continue
        count = int(_normalize_bool_series(rows[column]).sum())
        if count <= 0:
            continue
        out.append(
            {
                "name": column,
                "count": count,
                "share": round(count / max(len(rows), 1), 6),
            }
        )
    out.sort(key=lambda item: (-item["count"], item["name"]))
    return out


def _problem_ticker_summary(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    if rows.empty:
        return []
    work = rows.copy()
    issue_count = pd.to_numeric(work["quality_issue_count"], errors="coerce").fillna(0.0)
    work["__issue_count"] = issue_count
    work["__flagged"] = issue_count.gt(0)
    grouped = (
        work.groupby("ticker", dropna=False)
        .agg(
            market_count=("market_id", "count"),
            flagged_market_count=("__flagged", "sum"),
            avg_issue_count=("__issue_count", "mean"),
            clean_share=("quality_bucket", lambda s: float((s.astype(str) == "clean").mean())),
            watch_share=("quality_bucket", lambda s: float((s.astype(str) == "watch").mean())),
            noisy_share=("quality_bucket", lambda s: float((s.astype(str) == "noisy").mean())),
        )
        .reset_index()
    )
    grouped["flagged_share"] = grouped["flagged_market_count"] / grouped["market_count"].replace(0, np.nan)
    grouped = grouped.sort_values(
        ["flagged_market_count", "avg_issue_count", "market_count", "ticker"],
        ascending=[False, False, False, True],
    )
    out: List[Dict[str, Any]] = []
    for row in grouped.to_dict("records"):
        out.append(
            {
                "ticker": str(row.get("ticker") or ""),
                "market_count": int(row.get("market_count") or 0),
                "flagged_market_count": int(row.get("flagged_market_count") or 0),
                "flagged_share": round(float(row.get("flagged_share") or 0.0), 6),
                "avg_issue_count": round(float(row.get("avg_issue_count") or 0.0), 4),
                "clean_share": round(float(row.get("clean_share") or 0.0), 6),
                "watch_share": round(float(row.get("watch_share") or 0.0), 6),
                "noisy_share": round(float(row.get("noisy_share") or 0.0), 6),
            }
        )
    return out


def _weekly_summary(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    if rows.empty:
        return []
    work = rows.copy()
    work["__issue_count"] = pd.to_numeric(work["quality_issue_count"], errors="coerce").fillna(0.0)
    work["__flagged"] = work["__issue_count"].gt(0)
    grouped = (
        work.groupby("week_friday", dropna=False)
        .agg(
            market_count=("market_id", "count"),
            flagged_market_count=("__flagged", "sum"),
            avg_issue_count=("__issue_count", "mean"),
        )
        .reset_index()
    )
    grouped["flagged_share"] = grouped["flagged_market_count"] / grouped["market_count"].replace(0, np.nan)
    grouped = grouped.sort_values("week_friday")
    out: List[Dict[str, Any]] = []
    for row in grouped.to_dict("records"):
        avg_issue_count = _safe_float(row.get("avg_issue_count"))
        out.append(
            {
                "week_friday": str(row.get("week_friday") or ""),
                "market_count": int(row.get("market_count") or 0),
                "flagged_market_count": int(row.get("flagged_market_count") or 0),
                "flagged_share": round(float(row.get("flagged_share") or 0.0), 6),
                "avg_issue_count": round(avg_issue_count, 4) if avg_issue_count is not None else None,
                "quality_bucket": quality_bucket_from_issue_count(avg_issue_count or 0.0),
            }
        )
    return out


def build_quality_summary(rows: pd.DataFrame) -> Dict[str, Any]:
    if rows.empty:
        return {
            "market_count": 0,
            "flagged_market_count": 0,
            "flagged_share": 0.0,
            "bucket_counts": {bucket: 0 for bucket in QUALITY_BUCKETS},
            "prn_coverage_counts": {},
            "top_flags": [],
            "top_problem_tickers": [],
            "snapshot_anchor": "latest_safe_snapshot",
            "quality_columns": list(QUALITY_METRIC_COLUMNS),
            "quality_flag_columns": list(QUALITY_FLAG_COLUMNS),
        }

    issue_count = pd.to_numeric(rows["quality_issue_count"], errors="coerce").fillna(0.0)
    flagged_market_count = int(issue_count.gt(0).sum())
    prn_coverage_counts = rows["snapshot_coverage_status"].fillna("missing").astype(str).value_counts().to_dict()
    return {
        "market_count": int(len(rows)),
        "flagged_market_count": flagged_market_count,
        "flagged_share": round(flagged_market_count / max(len(rows), 1), 6),
        "bucket_counts": {
            bucket: int((rows["quality_bucket"].astype(str) == bucket).sum())
            for bucket in QUALITY_BUCKETS
        },
        "prn_coverage_counts": {str(key): int(value) for key, value in prn_coverage_counts.items()},
        "top_flags": _top_flag_summary(rows)[:10],
        "top_problem_tickers": _problem_ticker_summary(rows)[:10],
        "snapshot_anchor": "latest_safe_snapshot",
        "quality_columns": list(QUALITY_METRIC_COLUMNS),
        "quality_flag_columns": list(QUALITY_FLAG_COLUMNS),
    }


def build_quality_audit_payload(rows: pd.DataFrame) -> Dict[str, Any]:
    summary = build_quality_summary(rows)
    if rows.empty:
        return {
            "summary": summary,
            "flag_distribution": [],
            "problem_markets": [],
            "weekly_summary": [],
            "available_quality_flags": list(QUALITY_FLAG_COLUMNS),
        }

    problem_markets = []
    work = rows.copy()
    work["quality_issue_count"] = pd.to_numeric(work["quality_issue_count"], errors="coerce").fillna(0.0)
    work = work.sort_values(
        ["quality_issue_count", "quality_bucket", "ticker", "threshold", "market_id"],
        ascending=[False, False, True, True, True],
    )
    for row in work.head(25).to_dict("records"):
        active_flags = [
            column
            for column in QUALITY_FLAG_COLUMNS
            if column in row and bool(row.get(column))
        ]
        problem_markets.append(
            {
                "market_id": str(row.get("market_id") or ""),
                "event_id": str(row.get("event_id") or "") or None,
                "ticker": str(row.get("ticker") or ""),
                "threshold": _safe_float(row.get("threshold")),
                "week_friday": str(row.get("week_friday") or ""),
                "quality_issue_count": round(float(row.get("quality_issue_count") or 0.0), 4),
                "quality_bucket": str(row.get("quality_bucket") or quality_bucket_from_issue_count(0)),
                "active_flags": active_flags,
                "snapshot_coverage_status": str(row.get("snapshot_coverage_status") or "missing"),
                "snapshot_drop_reason": (
                    None if pd.isna(row.get("snapshot_drop_reason")) else str(row.get("snapshot_drop_reason"))
                ),
                "hours_since_last_yes_trade": _safe_float(row.get("hours_since_last_yes_trade")),
                "stale_ratio": _safe_float(row.get("stale_ratio")),
                "max_stale_hours": _safe_float(row.get("max_stale_hours")),
                "midprice_cluster_ratio": _safe_float(row.get("midprice_cluster_ratio")),
                "max_jump": _safe_float(row.get("max_jump")),
                "gamma_volume": _safe_float(row.get("gamma_volume")),
            }
        )

    return {
        "summary": summary,
        "flag_distribution": _top_flag_summary(rows),
        "problem_tickers": _problem_ticker_summary(rows)[:10],
        "problem_markets": problem_markets,
        "weekly_summary": _weekly_summary(rows),
        "available_quality_flags": list(QUALITY_FLAG_COLUMNS),
    }


def build_market_quality(
    run_dir: Path,
    weekly_markets: pd.DataFrame,
    prn_rows: pd.DataFrame,
    *,
    tz_name: str,
    close_time: str,
    emit_live: bool = False,
) -> MarketQualityBuildResult:
    if weekly_markets.empty:
        return MarketQualityBuildResult(
            rows=pd.DataFrame(columns=QUALITY_OUTPUT_COLUMNS),
            summary=build_quality_summary(pd.DataFrame(columns=QUALITY_OUTPUT_COLUMNS)),
            flag_columns=list(QUALITY_FLAG_COLUMNS),
            counted_flag_columns=list(COUNTED_QUALITY_FLAG_COLUMNS),
        )

    markets = weekly_markets.copy()
    for column in ("week_monday", "week_friday"):
        markets[column] = pd.to_datetime(markets.get(column), errors="coerce").dt.date
    markets["market_id"] = markets["market_id"].astype(str)
    markets["ticker"] = markets["ticker"].astype(str).str.upper()
    markets["threshold"] = pd.to_numeric(markets["threshold"], errors="coerce").round(6)
    markets = markets.dropna(subset=["market_id", "ticker", "threshold", "week_friday"])
    markets = markets.sort_values(["ticker", "week_friday", "threshold", "market_id"])
    markets = markets.drop_duplicates(subset=["market_id", "week_friday"], keep="last")

    prn = add_prn_quality_columns(prn_rows)
    if not prn.empty:
        prn["market_id"] = prn.get("market_id", pd.Series(index=prn.index, dtype=object)).astype(str)
        prn["week_friday"] = pd.to_datetime(prn.get("week_friday"), errors="coerce").dt.date

    yes_history = _load_yes_price_history(run_dir)
    history_by_market: Dict[str, pd.DataFrame] = {
        str(market_id): group.sort_values("timestamp_utc")
        for market_id, group in yes_history.groupby("market_id")
    }

    rows: List[Dict[str, Any]] = []
    total_markets = int(len(markets))
    if emit_live:
        emit_quality_live_telemetry(_build_partial_telemetry(pd.DataFrame(), total_markets=total_markets, completed_markets=0))

    for idx, market_row in enumerate(markets.to_dict("records"), start=1):
        market_id = str(market_row.get("market_id") or "")
        week_friday = market_row.get("week_friday")
        prn_group = pd.DataFrame()
        if not prn.empty:
            prn_group = prn[
                (prn["market_id"] == market_id)
                & (prn["week_friday"] == week_friday)
            ].copy()
        anchor = _anchor_prn_row(
            prn_group,
            week_monday=market_row.get("week_monday"),
            week_friday=week_friday,
            tz_name=tz_name,
            close_time=close_time,
        )
        snapshot_time = pd.to_datetime(anchor.get("snapshot_time_used"), utc=True, errors="coerce")
        trade_metrics = _compute_trade_metrics(history_by_market.get(market_id, pd.DataFrame()), snapshot_time)
        quality_row: Dict[str, Any] = {
            "market_id": market_id,
            "event_id": market_row.get("event_id"),
            "ticker": market_row.get("ticker"),
            "threshold": market_row.get("threshold"),
            "week_monday": market_row.get("week_monday").isoformat() if isinstance(market_row.get("week_monday"), date) else None,
            "week_friday": week_friday.isoformat() if isinstance(week_friday, date) else None,
            "event_endDate": market_row.get("event_endDate"),
            "yes_token_id": market_row.get("yes_token_id"),
            "no_token_id": market_row.get("no_token_id"),
            **anchor,
            **trade_metrics,
        }
        quality_row["snapshot_time_used"] = (
            snapshot_time.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(snapshot_time) else None
        )
        quality_row["gamma_volume"] = _resolve_volume_value(market_row)
        quality_row["flag_pm_low_volume"] = bool(
            quality_row["gamma_volume"] is not None and quality_row["gamma_volume"] < PM_LOW_VOLUME_THRESHOLD_USD
        )
        active_value = market_row.get("active")
        active_known = not pd.isna(active_value) if active_value is not None else False
        quality_row["flag_market_inactive"] = bool(
            (active_known and not bool(_normalize_bool_series(pd.Series([active_value])).iloc[0]))
            or bool(_normalize_bool_series(pd.Series([market_row.get("closed")])).iloc[0])
        )
        yes_token_id = market_row.get("yes_token_id")
        no_token_id = market_row.get("no_token_id")
        quality_row["flag_missing_token_ids"] = bool(
            pd.isna(yes_token_id)
            or pd.isna(no_token_id)
            or not str(yes_token_id).strip()
            or not str(no_token_id).strip()
        )
        snapshot_prn = _safe_float(quality_row.get("snapshot_pRN"))
        snapshot_abs_log_m_fwd = _safe_float(quality_row.get("snapshot_abs_log_m_fwd"))
        quality_row["flag_extreme_otm"] = bool(
            str(quality_row.get("snapshot_coverage_status") or "").lower() == "ok"
            and (
                (snapshot_abs_log_m_fwd is not None and snapshot_abs_log_m_fwd >= PM_EXTREME_OTM_ABS_LOGM_FWD_WARN)
                or (snapshot_prn is not None and snapshot_prn <= PM_TAIL_PRN_WARN_LOW)
                or (snapshot_prn is not None and snapshot_prn >= PM_TAIL_PRN_WARN_HIGH)
            )
        )
        rows.append(quality_row)

        if emit_live and (idx == total_markets or idx % 50 == 0):
            partial_df = pd.DataFrame(rows)
            emit_quality_live_telemetry(
                _build_partial_telemetry(
                    partial_df,
                    total_markets=total_markets,
                    completed_markets=idx,
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=QUALITY_OUTPUT_COLUMNS)
    else:
        median_points_by_group = (
            out.groupby(["ticker", "week_friday"], dropna=False)["yes_points"].median().to_dict()
        )

        def _is_sparse(row: pd.Series) -> bool:
            yes_points = _safe_float(row.get("yes_points"))
            if yes_points is None or yes_points <= 0:
                return False
            key = (str(row.get("ticker") or ""), str(row.get("week_friday") or ""))
            median_points = _safe_float(median_points_by_group.get(key))
            if median_points is None or median_points <= 0:
                return False
            return yes_points < (median_points * PM_POINT_DENSITY_SUSPECT_RATIO)

        out["flag_pm_sparse_points"] = out.apply(_is_sparse, axis=1)
        out["flag_not_relevant"] = (
            _normalize_bool_series(out["flag_prn_missing"])
            | _normalize_bool_series(out["flag_pm_no_trade_history"])
            | _normalize_bool_series(out["flag_pm_no_recent_trade"])
            | _normalize_bool_series(out["flag_extreme_otm"])
            | _normalize_bool_series(out["flag_market_inactive"])
        )

        issue_counts = []
        for row in out.to_dict("records"):
            quality_flags = {column: row.get(column) for column in COUNTED_QUALITY_FLAG_COLUMNS}
            issue_counts.append(compute_quality_issue_count(quality_flags))
        out["quality_issue_count"] = issue_counts
        out["quality_bucket"] = out["quality_issue_count"].map(quality_bucket_from_issue_count)

        for column in QUALITY_FLAG_COLUMNS:
            if column not in out.columns:
                out[column] = False
        for column in QUALITY_METRIC_COLUMNS:
            if column not in out.columns:
                out[column] = np.nan
        out = out.reindex(columns=list(QUALITY_OUTPUT_COLUMNS) + [c for c in out.columns if c not in QUALITY_OUTPUT_COLUMNS])

    summary = build_quality_summary(out)
    if emit_live:
        final_payload = _build_partial_telemetry(out, total_markets=total_markets, completed_markets=total_markets)
        final_payload["phase"] = "complete"
        emit_quality_live_telemetry(final_payload)
    return MarketQualityBuildResult(
        rows=out,
        summary=summary,
        flag_columns=list(QUALITY_FLAG_COLUMNS),
        counted_flag_columns=list(COUNTED_QUALITY_FLAG_COLUMNS),
    )
