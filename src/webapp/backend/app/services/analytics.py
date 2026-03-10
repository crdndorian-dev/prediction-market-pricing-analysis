from __future__ import annotations

import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from app.models.polymarket_history import (
    PolymarketDailyAnalyticsBreakdownResponse,
    PolymarketDailyAnalyticsBreakdownRow,
    PolymarketDailyAnalyticsCoverage,
    PolymarketDailyAnalyticsDay,
    PolymarketDailyAnalyticsResponse,
    PolymarketDailyAnalyticsSnapshot,
    PolymarketDailyAnalyticsStructure,
    PolymarketDailyAnalyticsStructureBucket,
    PolymarketDailyAnalyticsSummary,
)
from app.services.run_csv_files import primary_run_csv_path


BASE_DIR = Path(__file__).resolve().parents[5]
WEEKLY_HISTORY_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket" / "weekly_history"
RUNS_DIR = WEEKLY_HISTORY_DIR / "runs"

RUN_ANALYTICS_PRIMARY_METRIC = "daily_notional_volume"
RUN_ANALYTICS_METRIC_MODE = "true_volume"
DEFAULT_CI_LEVEL = 90
DEFAULT_BASELINE_WINDOW_DAYS = 28
MIN_HISTORY_DAYS = 7
ALLOWED_TOKEN_ROLES = {"all", "yes", "no"}
REQUIRED_TRADE_COLUMNS = {
    "timestamp_utc",
    "trade_id",
    "market_id",
    "ticker",
    "notional",
    "size",
}
OPTIONAL_TRADE_COLUMNS = {
    "token_role",
    "threshold",
    "week_friday",
    "event_endDate",
}
WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
EVENT_BUCKET_ORDER = ["D0", "D1", "D2", "D3-D4", "D5+", "Post-event", "Unknown"]
LADDER_BUCKET_ORDER = ["lower", "middle", "upper", "solo", "unknown"]

_CLEAN_TRADES_CACHE: Dict[str, Dict[str, Any]] = {}
_DAILY_SUMMARY_CACHE: Dict[str, PolymarketDailyAnalyticsResponse] = {}
_BREAKDOWN_CACHE: Dict[str, PolymarketDailyAnalyticsBreakdownResponse] = {}


def _safe_json_load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _resolve_run_dir(run_id: str) -> Path:
    run_id = str(run_id or "").strip()
    if not run_id:
        raise ValueError("run_id is required.")
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return run_dir


def _parse_date_filter(name: str, value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be in YYYY-MM-DD format.") from exc


def _normalize_tickers(tickers: Optional[Sequence[str]]) -> List[str]:
    if not tickers:
        return []
    normalized: List[str] = []
    for ticker in tickers:
        cleaned = str(ticker or "").strip().upper()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _normalize_token_role(value: Optional[str]) -> Literal["all", "yes", "no"]:
    raw = str(value or "all").strip().lower()
    if raw not in ALLOWED_TOKEN_ROLES:
        allowed = ", ".join(sorted(ALLOWED_TOKEN_ROLES))
        raise ValueError(f"token_role must be one of: {allowed}.")
    return raw  # type: ignore[return-value]


def _normalize_ci_level(value: Optional[int]) -> int:
    if value is None:
        return DEFAULT_CI_LEVEL
    try:
        ci_level = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ci_level must be an integer percentage.") from exc
    if ci_level not in {90, 95, 99}:
        raise ValueError("ci_level must be one of 90, 95, or 99.")
    return ci_level


def _make_cache_key(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def parse_tickers_query_param(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return _normalize_tickers([part for part in str(value).split(",")])


def assess_run_volume_analytics_readiness(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Assess whether a weekly-history run can support true-volume analytics."""
    manifest = manifest if isinstance(manifest, dict) else {}
    pipeline_args = manifest.get("pipeline_args")
    pipeline_args = pipeline_args if isinstance(pipeline_args, dict) else {}
    trade_artifact = manifest.get("trade_artifact")
    trade_artifact = trade_artifact if isinstance(trade_artifact, dict) else {}
    subgraph = manifest.get("subgraph")
    subgraph = subgraph if isinstance(subgraph, dict) else {}

    include_subgraph = bool(pipeline_args.get("include_subgraph"))
    trade_entities = _coerce_non_negative_int(trade_artifact.get("rows"))
    trade_days = _coerce_non_negative_int(trade_artifact.get("trade_days"))
    trade_artifact_path = trade_artifact.get("path")
    has_trades = bool(trade_artifact_path and (trade_entities or 0) > 0 and (trade_days or 0) > 0)

    warning: Optional[str]
    if has_trades:
        warning = None
    elif not include_subgraph:
        warning = (
            "Run was built without subgraph trade ingestion. Rebuild with "
            "include_subgraph enabled to unlock true-volume analytics."
        )
    elif subgraph.get("error"):
        warning = (
            "Trade ingest did not produce usable trades for this run: "
            f"{str(subgraph.get('error')).strip()}"
        )
    elif _coerce_non_negative_int(subgraph.get("total_entities")):
        warning = (
            "Run has subgraph trade metadata but no run-scoped trades artifact. "
            "Rebuild the run under the Step 2 trade-artifact contract to unlock analytics."
        )
    else:
        warning = (
            "Run requested subgraph trade ingestion but no run-scoped "
            "trades artifact was recorded."
        )

    return {
        "analytics_ready": has_trades,
        "primary_metric": RUN_ANALYTICS_PRIMARY_METRIC,
        "requires_true_volume": True,
        "include_subgraph_requested": include_subgraph,
        "has_trades": has_trades,
        "trade_entities": trade_entities,
        "trade_days": trade_days,
        "trade_artifact_path": trade_artifact_path,
        "warning": warning,
    }


def _bucket_event_proximity(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    try:
        days = int(value)
    except (TypeError, ValueError):
        return "Unknown"
    if days < 0:
        return "Post-event"
    if days == 0:
        return "D0"
    if days == 1:
        return "D1"
    if days == 2:
        return "D2"
    if days <= 4:
        return "D3-D4"
    return "D5+"


def _assign_ladder_buckets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    meta = (
        df[["market_id", "ticker", "week_friday", "threshold"]]
        .drop_duplicates(subset=["market_id"], keep="first")
        .copy()
    )
    meta["ladder_bucket"] = "unknown"
    meta["threshold_num"] = pd.to_numeric(meta["threshold"], errors="coerce")

    for _, group in meta.groupby(["ticker", "week_friday"], dropna=False):
        valid = group.dropna(subset=["threshold_num"]).sort_values("threshold_num").copy()
        if valid.empty:
            continue
        n = len(valid)
        if n == 1:
            meta.loc[valid.index, "ladder_bucket"] = "solo"
            continue
        positions = np.linspace(0.0, 1.0, num=n)
        buckets = []
        for position in positions:
            if position <= 1 / 3:
                buckets.append("lower")
            elif position >= 2 / 3:
                buckets.append("upper")
            else:
                buckets.append("middle")
        meta.loc[valid.index, "ladder_bucket"] = buckets

    out = df.merge(meta[["market_id", "ladder_bucket"]], on="market_id", how="left")
    out["ladder_bucket"] = out["ladder_bucket"].fillna("unknown")
    return out


def _load_clean_trade_frame(run_id: str) -> Dict[str, Any]:
    run_dir = _resolve_run_dir(run_id)
    manifest = _safe_json_load(run_dir / "manifest.json")
    readiness = assess_run_volume_analytics_readiness(manifest)
    if not readiness["analytics_ready"]:
        raise ValueError(readiness.get("warning") or "This run is not ready for true-volume analytics.")

    trades_path = primary_run_csv_path(run_dir, "trades.csv")
    if trades_path is None:
        raise RuntimeError(
            "Run manifest marks analytics as ready, but the run-scoped trades.csv artifact is missing."
        )

    cache_key = _make_cache_key({
        "run_id": run_id,
        "trades_path": str(trades_path),
        "mtime_ns": trades_path.stat().st_mtime_ns,
        "size": trades_path.stat().st_size,
    })
    cached = _CLEAN_TRADES_CACHE.get(cache_key)
    if cached is not None:
        return {
            **cached,
            "df": cached["df"].copy(),
            "warnings": list(cached["warnings"]),
            "available_tickers": list(cached["available_tickers"]),
        }

    df = pd.read_csv(trades_path)
    missing_required = sorted(REQUIRED_TRADE_COLUMNS - set(df.columns))
    if missing_required:
        missing_cols = ", ".join(missing_required)
        raise ValueError(f"trades.csv is missing required analytics columns: {missing_cols}")

    warnings: List[str] = []
    total_trade_rows = int(len(df))

    for column in OPTIONAL_TRADE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
            warnings.append(
                f"trades.csv is missing optional column {column}; related analytics will be partial."
            )

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["notional"] = pd.to_numeric(df["notional"], errors="coerce")
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["trade_id"] = df["trade_id"].astype("string").fillna("").str.strip()
    df["market_id"] = df["market_id"].astype("string").replace({"": pd.NA})
    df["ticker"] = df["ticker"].astype("string").fillna("").str.strip().str.upper().replace("", pd.NA)
    df["token_role"] = (
        df["token_role"].astype("string").fillna("").str.strip().str.lower().replace("", pd.NA)
    )
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["week_friday"] = df["week_friday"].astype("string").fillna("").str.strip().replace("", pd.NA)
    df["event_endDate"] = pd.to_datetime(df["event_endDate"], utc=True, errors="coerce")

    invalid_mask = (
        df["timestamp_utc"].isna()
        | df["notional"].isna()
        | df["size"].isna()
        | df["market_id"].isna()
    )
    invalid_rows = int(invalid_mask.sum())
    if invalid_rows:
        warnings.append(
            f"Dropped {invalid_rows} trade row(s) with invalid timestamp, numeric volume, or market_id."
        )
    df = df.loc[~invalid_mask].copy()

    valid_trade_rows = int(len(df))
    if df.empty:
        raise ValueError("trades.csv does not contain any valid rows for analytics.")

    duplicate_trade_ids = int(df.loc[df["trade_id"] != "", "trade_id"].duplicated().sum())
    if duplicate_trade_ids:
        warnings.append(
            f"Detected {duplicate_trade_ids} duplicate trade_id row(s) in trades.csv. Volume metrics may be overstated."
        )

    df["trade_date"] = df["timestamp_utc"].dt.date
    df["weekday"] = df["timestamp_utc"].dt.day_name().str[:3]
    event_dates = df["event_endDate"].dt.date
    df["days_to_event"] = (event_dates - df["trade_date"]).apply(
        lambda delta: delta.days if pd.notna(delta) else pd.NA
    )
    df["event_proximity_bucket"] = df["days_to_event"].apply(_bucket_event_proximity)
    df = _assign_ladder_buckets(df)
    df["event_endDate"] = df["event_endDate"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    available_tickers = sorted([value for value in df["ticker"].dropna().unique().tolist() if value])
    payload = {
        "df": df,
        "warnings": warnings,
        "artifact_path": readiness.get("trade_artifact_path") or trades_path.name,
        "total_trade_rows": total_trade_rows,
        "valid_trade_rows": valid_trade_rows,
        "available_tickers": available_tickers,
        "source_cache_key": cache_key,
    }
    _CLEAN_TRADES_CACHE[cache_key] = payload
    return {
        **payload,
        "df": df.copy(),
        "warnings": list(warnings),
        "available_tickers": list(available_tickers),
        "source_cache_key": cache_key,
    }


def _filter_trade_frame(
    base: Dict[str, Any],
    *,
    date_min: Optional[date],
    date_max: Optional[date],
    tickers: Sequence[str],
    token_role: Literal["all", "yes", "no"],
) -> Dict[str, Any]:
    df = base["df"].copy()
    warnings = list(base["warnings"])

    if token_role != "all":
        if "token_role" not in df.columns:
            raise ValueError("token_role filtering requested, but trades.csv has no token_role column.")
        df = df.loc[df["token_role"] == token_role].copy()

    if tickers:
        missing_ticker_rows = int(df["ticker"].isna().sum())
        if missing_ticker_rows:
            warnings.append(
                f"Excluded {missing_ticker_rows} trade row(s) with missing ticker values from ticker-filtered analytics."
            )
        df = df.loc[df["ticker"].isin(tickers)].copy()

    if date_min:
        df = df.loc[df["trade_date"] >= date_min].copy()
    if date_max:
        df = df.loc[df["trade_date"] <= date_max].copy()

    return {
        **base,
        "df": df,
        "warnings": warnings,
        "filtered_trade_rows": int(len(df)),
    }


def _build_snapshot_from_row(row: Optional[pd.Series]) -> Optional[PolymarketDailyAnalyticsSnapshot]:
    if row is None:
        return None
    return PolymarketDailyAnalyticsSnapshot(
        date=row["date"].isoformat(),
        weekday=str(row["weekday"]),
        daily_notional_volume=float(row["daily_notional_volume"]),
        share_volume=float(row["share_volume"]),
        trade_count=int(row["trade_count"]),
        active_markets=int(row.get("active_markets", 0)),
        active_tickers=int(row.get("active_tickers", 0)),
        expected_notional_volume=None if pd.isna(row.get("expected_notional_volume")) else float(row["expected_notional_volume"]),
        band_lo=None if pd.isna(row.get("band_lo")) else float(row["band_lo"]),
        band_hi=None if pd.isna(row.get("band_hi")) else float(row["band_hi"]),
        realized_expected_ratio=None if pd.isna(row.get("realized_expected_ratio")) else float(row["realized_expected_ratio"]),
        day_over_day_delta=None if pd.isna(row.get("day_over_day_delta")) else float(row["day_over_day_delta"]),
        day_over_day_pct=None if pd.isna(row.get("day_over_day_pct")) else float(row["day_over_day_pct"]),
        noise_cv_28d=None if pd.isna(row.get("noise_cv_28d")) else float(row["noise_cv_28d"]),
        noise_mad_ratio_28d=None if pd.isna(row.get("noise_mad_ratio_28d")) else float(row["noise_mad_ratio_28d"]),
        unusually_active=bool(row.get("unusually_active", False)),
        unusually_inactive=bool(row.get("unusually_inactive", False)),
        flagged_outlier=bool(row.get("flagged_outlier", False)),
        baseline_source=str(row.get("baseline_source") or "rolling_weekday_median"),
    )


def _zero_filled_daily_aggregate(
    df: pd.DataFrame,
    *,
    date_min: Optional[date],
    date_max: Optional[date],
) -> tuple[pd.DataFrame, int, date, date]:
    grouped = (
        df.groupby("trade_date", dropna=False)
        .agg(
            daily_notional_volume=("notional", "sum"),
            share_volume=("size", "sum"),
            trade_count=("trade_id", "size"),
            active_markets=("market_id", "nunique"),
            active_tickers=("ticker", lambda values: values.dropna().nunique()),
        )
        .sort_index()
    )
    observed_trade_days = int(len(grouped))
    effective_date_min = date_min or grouped.index.min()
    effective_date_max = date_max or grouped.index.max()
    if effective_date_min > effective_date_max:
        raise ValueError("No trade dates remain inside the selected filter window.")

    calendar_index = pd.Index(
        [ts.date() for ts in pd.date_range(effective_date_min, effective_date_max, freq="D")],
        name="trade_date",
    )
    daily = grouped.reindex(calendar_index, fill_value=0.0).reset_index().rename(columns={"trade_date": "date"})
    daily["trade_count"] = daily["trade_count"].astype(int)
    daily["active_markets"] = daily["active_markets"].astype(int)
    daily["active_tickers"] = daily["active_tickers"].astype(int)
    daily["weekday"] = pd.to_datetime(daily["date"]).dt.day_name().str[:3]
    return daily, observed_trade_days, effective_date_min, effective_date_max


def _rolling_outlier_flags(log_values: pd.Series, window_days: int) -> List[bool]:
    flags: List[bool] = []
    for idx in range(len(log_values)):
        history = log_values.iloc[max(0, idx - window_days):idx].dropna()
        if len(history) < MIN_HISTORY_DAYS:
            flags.append(False)
            continue
        median = float(history.median())
        mad = float((history - median).abs().median())
        scale = 1.4826 * mad
        if scale <= 0:
            flags.append(False)
            continue
        z_score = abs(float(log_values.iloc[idx]) - median) / scale
        flags.append(z_score > 3.5)
    return flags


def _compute_baseline_metrics(
    daily: pd.DataFrame,
    *,
    ci_level: int,
    exclude_flagged: bool,
    window_days: int = DEFAULT_BASELINE_WINDOW_DAYS,
) -> pd.DataFrame:
    out = daily.copy().sort_values("date").reset_index(drop=True)
    out["log_volume"] = np.log1p(out["daily_notional_volume"])
    out["flagged_outlier"] = _rolling_outlier_flags(out["log_volume"], window_days)
    out["expected_notional_volume"] = np.nan
    out["band_lo"] = np.nan
    out["band_hi"] = np.nan
    out["realized_expected_ratio"] = np.nan
    out["noise_cv_28d"] = np.nan
    out["noise_mad_ratio_28d"] = np.nan
    out["baseline_source"] = "rolling_weekday_median"

    tail_q = (100 - ci_level) / 200.0
    lower_q = tail_q
    upper_q = 1.0 - tail_q

    for idx in range(len(out)):
        history = out.iloc[max(0, idx - window_days):idx].copy()
        if exclude_flagged:
            history = history.loc[~history["flagged_outlier"]].copy()
        if len(history) < MIN_HISTORY_DAYS:
            out.at[idx, "baseline_source"] = "insufficient_history"
            continue

        same_weekday = history.loc[history["weekday"] == out.at[idx, "weekday"]].copy()
        if len(same_weekday) >= max(4, MIN_HISTORY_DAYS // 2):
            baseline_history = same_weekday
            out.at[idx, "baseline_source"] = "rolling_weekday_median"
        else:
            baseline_history = history
            out.at[idx, "baseline_source"] = "rolling_median"

        history_log = baseline_history["log_volume"].dropna()
        if len(history_log) < MIN_HISTORY_DAYS:
            out.at[idx, "baseline_source"] = "insufficient_history"
            continue

        expected_log = float(history_log.median())
        residuals = history_log - expected_log
        lower_log = expected_log + float(residuals.quantile(lower_q))
        upper_log = expected_log + float(residuals.quantile(upper_q))

        expected_value = max(0.0, math.expm1(expected_log))
        lower_value = max(0.0, math.expm1(lower_log))
        upper_value = max(0.0, math.expm1(upper_log))
        out.at[idx, "expected_notional_volume"] = expected_value
        out.at[idx, "band_lo"] = lower_value
        out.at[idx, "band_hi"] = upper_value

        current_value = float(out.at[idx, "daily_notional_volume"])
        if expected_value > 0:
            out.at[idx, "realized_expected_ratio"] = current_value / expected_value

        history_raw = history["daily_notional_volume"].astype(float)
        if exclude_flagged:
            history_raw = history_raw.loc[history_raw.index]
        mean_value = float(history_raw.mean()) if len(history_raw) else 0.0
        if len(history_raw) >= MIN_HISTORY_DAYS and mean_value > 0:
            out.at[idx, "noise_cv_28d"] = float(history_raw.std(ddof=0) / mean_value)
        median_raw = float(history_raw.median()) if len(history_raw) else 0.0
        if len(history_raw) >= MIN_HISTORY_DAYS and median_raw > 0:
            mad_raw = float((history_raw - median_raw).abs().median())
            out.at[idx, "noise_mad_ratio_28d"] = mad_raw / median_raw

    out["day_over_day_delta"] = out["daily_notional_volume"].diff()
    prev_values = out["daily_notional_volume"].shift(1)
    out["day_over_day_pct"] = np.where(prev_values > 0, out["day_over_day_delta"] / prev_values, np.nan)
    out["unusually_active"] = (
        out["band_hi"].notna() & (out["daily_notional_volume"] > out["band_hi"])
    )
    out["unusually_inactive"] = (
        out["band_lo"].notna() & (out["daily_notional_volume"] < out["band_lo"])
    )
    return out


def _build_structure_buckets(
    grouped: pd.DataFrame,
    *,
    key_column: str,
    total_notional: float,
    order: Optional[List[str]] = None,
) -> List[PolymarketDailyAnalyticsStructureBucket]:
    if grouped.empty:
        return []
    buckets: List[PolymarketDailyAnalyticsStructureBucket] = []
    for row in grouped.itertuples(index=False):
        total_value = float(row.total_notional_volume)
        buckets.append(
            PolymarketDailyAnalyticsStructureBucket(
                key=str(getattr(row, key_column)),
                label=str(getattr(row, key_column)),
                observations=int(row.observations),
                mean_daily_notional_volume=float(row.mean_daily_notional_volume),
                median_daily_notional_volume=float(row.median_daily_notional_volume),
                total_notional_volume=total_value,
                share_total_notional_volume=(total_value / total_notional) if total_notional > 0 else 0.0,
            )
        )
    if order:
        sort_index = {value: idx for idx, value in enumerate(order)}
        buckets.sort(key=lambda bucket: (sort_index.get(bucket.key, len(sort_index)), bucket.key))
    else:
        buckets.sort(key=lambda bucket: bucket.label)
    return buckets


def _build_structural_profiles(
    trade_df: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    *,
    exclude_flagged: bool,
) -> PolymarketDailyAnalyticsStructure:
    day_frame = daily_metrics.copy()
    if exclude_flagged:
        day_frame = day_frame.loc[~day_frame["flagged_outlier"]].copy()
    total_day_notional = float(day_frame["daily_notional_volume"].sum())
    weekday_grouped = (
        day_frame.groupby("weekday", dropna=False)
        .agg(
            observations=("date", "size"),
            mean_daily_notional_volume=("daily_notional_volume", "mean"),
            median_daily_notional_volume=("daily_notional_volume", "median"),
            total_notional_volume=("daily_notional_volume", "sum"),
        )
        .reset_index()
    )

    row_frame = trade_df.copy()
    if exclude_flagged and not daily_metrics.empty:
        flagged_dates = set(daily_metrics.loc[daily_metrics["flagged_outlier"], "date"].tolist())
        row_frame = row_frame.loc[~row_frame["trade_date"].isin(flagged_dates)].copy()

    event_grouped = (
        row_frame.groupby(["trade_date", "event_proximity_bucket"], dropna=False)
        .agg(daily_notional_volume=("notional", "sum"))
        .reset_index()
        .groupby("event_proximity_bucket", dropna=False)
        .agg(
            observations=("trade_date", "nunique"),
            mean_daily_notional_volume=("daily_notional_volume", "mean"),
            median_daily_notional_volume=("daily_notional_volume", "median"),
            total_notional_volume=("daily_notional_volume", "sum"),
        )
        .reset_index()
        .rename(columns={"event_proximity_bucket": "key"})
    )
    ladder_grouped = (
        row_frame.groupby(["trade_date", "ladder_bucket"], dropna=False)
        .agg(daily_notional_volume=("notional", "sum"))
        .reset_index()
        .groupby("ladder_bucket", dropna=False)
        .agg(
            observations=("trade_date", "nunique"),
            mean_daily_notional_volume=("daily_notional_volume", "mean"),
            median_daily_notional_volume=("daily_notional_volume", "median"),
            total_notional_volume=("daily_notional_volume", "sum"),
        )
        .reset_index()
        .rename(columns={"ladder_bucket": "key"})
    )

    total_event_notional = float(event_grouped["total_notional_volume"].sum()) if not event_grouped.empty else 0.0
    total_ladder_notional = float(ladder_grouped["total_notional_volume"].sum()) if not ladder_grouped.empty else 0.0

    return PolymarketDailyAnalyticsStructure(
        weekday=_build_structure_buckets(
            weekday_grouped.rename(columns={"weekday": "key"}),
            key_column="key",
            total_notional=total_day_notional,
            order=WEEKDAY_ORDER,
        ),
        event_proximity=_build_structure_buckets(
            event_grouped,
            key_column="key",
            total_notional=total_event_notional,
            order=EVENT_BUCKET_ORDER,
        ),
        ladder_bucket=_build_structure_buckets(
            ladder_grouped,
            key_column="key",
            total_notional=total_ladder_notional,
            order=LADDER_BUCKET_ORDER,
        ),
    )


def _build_empty_response(
    run_id: str,
    *,
    artifact_path: Optional[str],
    available_tickers: Sequence[str],
    requested_date_min: Optional[str],
    requested_date_max: Optional[str],
    tickers: Sequence[str],
    token_role: Literal["all", "yes", "no"],
    ci_level: int,
    exclude_flagged: bool,
    total_trade_rows: int,
    valid_trade_rows: int,
    filtered_trade_rows: int,
    warnings: Sequence[str],
) -> PolymarketDailyAnalyticsResponse:
    return PolymarketDailyAnalyticsResponse(
        run_id=run_id,
        analytics_ready=True,
        primary_metric=RUN_ANALYTICS_PRIMARY_METRIC,
        metric_mode=RUN_ANALYTICS_METRIC_MODE,
        token_role=token_role,
        ci_level=ci_level,
        exclude_flagged=exclude_flagged,
        baseline_window_days=DEFAULT_BASELINE_WINDOW_DAYS,
        latest_trade_date=None,
        comparison_date=None,
        available_tickers=list(available_tickers),
        coverage=PolymarketDailyAnalyticsCoverage(
            artifact_path=artifact_path,
            total_trade_rows=total_trade_rows,
            valid_trade_rows=valid_trade_rows,
            filtered_trade_rows=filtered_trade_rows,
            observed_trade_days=0,
            filled_days=0,
            effective_date_min=requested_date_min,
            effective_date_max=requested_date_max,
            requested_date_min=requested_date_min,
            requested_date_max=requested_date_max,
            requested_tickers=list(tickers),
            requested_token_role=token_role,
        ),
        summary=PolymarketDailyAnalyticsSummary(),
        days=[],
        structure=PolymarketDailyAnalyticsStructure(),
        warnings=list(warnings),
    )


def get_run_daily_analytics_summary(
    run_id: str,
    *,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
    tickers: Optional[Sequence[str]] = None,
    token_role: Optional[str] = "all",
    ci_level: Optional[int] = DEFAULT_CI_LEVEL,
    exclude_flagged: bool = False,
) -> PolymarketDailyAnalyticsResponse:
    requested_date_min = _parse_date_filter("date_min", date_min)
    requested_date_max = _parse_date_filter("date_max", date_max)
    if requested_date_min and requested_date_max and requested_date_min > requested_date_max:
        raise ValueError("date_min must be less than or equal to date_max.")

    selected_tickers = _normalize_tickers(tickers)
    selected_token_role = _normalize_token_role(token_role)
    selected_ci_level = _normalize_ci_level(ci_level)

    base = _load_clean_trade_frame(run_id)
    cache_key = _make_cache_key({
        "run_id": run_id,
        "source": base.get("source_cache_key"),
        "date_min": requested_date_min.isoformat() if requested_date_min else None,
        "date_max": requested_date_max.isoformat() if requested_date_max else None,
        "tickers": selected_tickers,
        "token_role": selected_token_role,
        "ci_level": selected_ci_level,
        "exclude_flagged": bool(exclude_flagged),
    })
    cached = _DAILY_SUMMARY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    filtered = _filter_trade_frame(
        base,
        date_min=requested_date_min,
        date_max=requested_date_max,
        tickers=selected_tickers,
        token_role=selected_token_role,
    )
    df = filtered["df"]
    warnings = list(filtered["warnings"])

    if df.empty:
        warnings.append("No trade rows matched the selected analytics filters.")
        response = _build_empty_response(
            run_id,
            artifact_path=filtered["artifact_path"],
            available_tickers=filtered["available_tickers"],
            requested_date_min=requested_date_min.isoformat() if requested_date_min else None,
            requested_date_max=requested_date_max.isoformat() if requested_date_max else None,
            tickers=selected_tickers,
            token_role=selected_token_role,
            ci_level=selected_ci_level,
            exclude_flagged=bool(exclude_flagged),
            total_trade_rows=filtered["total_trade_rows"],
            valid_trade_rows=filtered["valid_trade_rows"],
            filtered_trade_rows=filtered["filtered_trade_rows"],
            warnings=warnings,
        )
        _DAILY_SUMMARY_CACHE[cache_key] = response
        return response

    daily, observed_trade_days, effective_date_min, effective_date_max = _zero_filled_daily_aggregate(
        df,
        date_min=requested_date_min,
        date_max=requested_date_max,
    )
    daily_metrics = _compute_baseline_metrics(
        daily,
        ci_level=selected_ci_level,
        exclude_flagged=bool(exclude_flagged),
    )

    trade_days = sorted(df["trade_date"].dropna().unique().tolist())
    latest_trade_date = max(trade_days)
    comparison_date = latest_trade_date - timedelta(days=1) if observed_trade_days > 1 else None
    latest_row = daily_metrics.loc[daily_metrics["date"] == latest_trade_date].iloc[0]
    comparison_row = None
    if comparison_date is not None:
        comparison_rows = daily_metrics.loc[daily_metrics["date"] == comparison_date]
        comparison_row = comparison_rows.iloc[0] if not comparison_rows.empty else None
    else:
        warnings.append("Only one observed trade day is available, so prior-day comparison is unavailable.")

    latest_snapshot = _build_snapshot_from_row(latest_row)
    comparison_snapshot = _build_snapshot_from_row(comparison_row) if comparison_row is not None else None

    delta_notional = None
    delta_notional_pct = None
    delta_share_volume = None
    delta_trade_count = None
    if latest_snapshot and comparison_snapshot:
        delta_notional = latest_snapshot.daily_notional_volume - comparison_snapshot.daily_notional_volume
        delta_share_volume = latest_snapshot.share_volume - comparison_snapshot.share_volume
        delta_trade_count = latest_snapshot.trade_count - comparison_snapshot.trade_count
        if comparison_snapshot.daily_notional_volume != 0:
            delta_notional_pct = delta_notional / comparison_snapshot.daily_notional_volume

    structure = _build_structural_profiles(
        df,
        daily_metrics,
        exclude_flagged=bool(exclude_flagged),
    )

    days = [
        PolymarketDailyAnalyticsDay(
            date=row.date.isoformat(),
            weekday=str(row.weekday),
            daily_notional_volume=float(row.daily_notional_volume),
            share_volume=float(row.share_volume),
            trade_count=int(row.trade_count),
            active_markets=int(row.active_markets),
            active_tickers=int(row.active_tickers),
            expected_notional_volume=None if pd.isna(row.expected_notional_volume) else float(row.expected_notional_volume),
            band_lo=None if pd.isna(row.band_lo) else float(row.band_lo),
            band_hi=None if pd.isna(row.band_hi) else float(row.band_hi),
            realized_expected_ratio=None if pd.isna(row.realized_expected_ratio) else float(row.realized_expected_ratio),
            day_over_day_delta=None if pd.isna(row.day_over_day_delta) else float(row.day_over_day_delta),
            day_over_day_pct=None if pd.isna(row.day_over_day_pct) else float(row.day_over_day_pct),
            noise_cv_28d=None if pd.isna(row.noise_cv_28d) else float(row.noise_cv_28d),
            noise_mad_ratio_28d=None if pd.isna(row.noise_mad_ratio_28d) else float(row.noise_mad_ratio_28d),
            unusually_active=bool(row.unusually_active),
            unusually_inactive=bool(row.unusually_inactive),
            flagged_outlier=bool(row.flagged_outlier),
            baseline_source=str(row.baseline_source),
        )
        for row in daily_metrics.itertuples(index=False)
    ]

    response = PolymarketDailyAnalyticsResponse(
        run_id=run_id,
        analytics_ready=True,
        primary_metric=RUN_ANALYTICS_PRIMARY_METRIC,
        metric_mode=RUN_ANALYTICS_METRIC_MODE,
        token_role=selected_token_role,
        ci_level=selected_ci_level,
        exclude_flagged=bool(exclude_flagged),
        baseline_window_days=DEFAULT_BASELINE_WINDOW_DAYS,
        latest_trade_date=latest_trade_date.isoformat(),
        comparison_date=comparison_date.isoformat() if comparison_date else None,
        available_tickers=filtered["available_tickers"],
        coverage=PolymarketDailyAnalyticsCoverage(
            artifact_path=filtered["artifact_path"],
            total_trade_rows=filtered["total_trade_rows"],
            valid_trade_rows=filtered["valid_trade_rows"],
            filtered_trade_rows=filtered["filtered_trade_rows"],
            observed_trade_days=observed_trade_days,
            filled_days=int(len(days)),
            effective_date_min=effective_date_min.isoformat(),
            effective_date_max=effective_date_max.isoformat(),
            requested_date_min=requested_date_min.isoformat() if requested_date_min else None,
            requested_date_max=requested_date_max.isoformat() if requested_date_max else None,
            requested_tickers=selected_tickers,
            requested_token_role=selected_token_role,
        ),
        summary=PolymarketDailyAnalyticsSummary(
            latest_day=latest_snapshot,
            comparison_day=comparison_snapshot,
            delta_notional=delta_notional,
            delta_notional_pct=delta_notional_pct,
            delta_share_volume=delta_share_volume,
            delta_trade_count=delta_trade_count,
            latest_noise_cv_28d=latest_snapshot.noise_cv_28d if latest_snapshot else None,
            latest_noise_mad_ratio_28d=latest_snapshot.noise_mad_ratio_28d if latest_snapshot else None,
            latest_unusually_active=bool(latest_snapshot.unusually_active) if latest_snapshot else False,
            latest_unusually_inactive=bool(latest_snapshot.unusually_inactive) if latest_snapshot else False,
            latest_flagged_outlier=bool(latest_snapshot.flagged_outlier) if latest_snapshot else False,
        ),
        days=days,
        structure=structure,
        warnings=warnings,
    )
    _DAILY_SUMMARY_CACHE[cache_key] = response
    return response


def _compute_group_day_row(
    series: pd.DataFrame,
    *,
    selected_date: date,
    ci_level: int,
    exclude_flagged: bool,
) -> tuple[Optional[pd.Series], int]:
    metrics = _compute_baseline_metrics(
        series,
        ci_level=ci_level,
        exclude_flagged=exclude_flagged,
    )
    row_df = metrics.loc[metrics["date"] == selected_date]
    history_days = int((metrics["trade_count"] > 0).sum()) if "trade_count" in metrics.columns else int((metrics["daily_notional_volume"] > 0).sum())
    if row_df.empty:
        return None, history_days
    return row_df.iloc[0], history_days


def _peer_market_baseline(
    market_daily: pd.DataFrame,
    *,
    ticker: Optional[str],
    ladder_bucket: Optional[str],
    event_proximity_bucket: Optional[str],
    selected_date: date,
    ci_level: int,
) -> tuple[Optional[float], Optional[float], Optional[float], str]:
    peer = market_daily.loc[market_daily["trade_date"] < selected_date].copy()
    if ticker:
        peer = peer.loc[peer["ticker"] == ticker].copy()
    source = "ticker_peer"
    if ladder_bucket and event_proximity_bucket:
        specific = peer.loc[
            (peer["ladder_bucket"] == ladder_bucket)
            & (peer["event_proximity_bucket"] == event_proximity_bucket)
        ].copy()
        if len(specific) >= MIN_HISTORY_DAYS:
            peer = specific
            source = "ticker_ladder_event_peer"

    if len(peer) < MIN_HISTORY_DAYS:
        return None, None, None, "insufficient_history"

    log_values = np.log1p(peer["daily_notional_volume"].astype(float))
    expected_log = float(log_values.median())
    residuals = log_values - expected_log
    tail_q = (100 - ci_level) / 200.0
    lo = max(0.0, math.expm1(expected_log + float(residuals.quantile(tail_q))))
    hi = max(0.0, math.expm1(expected_log + float(residuals.quantile(1.0 - tail_q))))
    expected = max(0.0, math.expm1(expected_log))
    return expected, lo, hi, source


def get_run_daily_analytics_breakdown(
    run_id: str,
    *,
    breakdown_date: str,
    group_by: Literal["ticker", "market"] = "ticker",
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
    tickers: Optional[Sequence[str]] = None,
    token_role: Optional[str] = "all",
    ci_level: Optional[int] = DEFAULT_CI_LEVEL,
    exclude_flagged: bool = False,
) -> PolymarketDailyAnalyticsBreakdownResponse:
    selected_date = _parse_date_filter("date", breakdown_date)
    if selected_date is None:
        raise ValueError("date is required in YYYY-MM-DD format.")
    if group_by not in {"ticker", "market"}:
        raise ValueError("group_by must be 'ticker' or 'market'.")

    requested_date_min = _parse_date_filter("date_min", date_min)
    requested_date_max = _parse_date_filter("date_max", date_max)
    if requested_date_min and requested_date_max and requested_date_min > requested_date_max:
        raise ValueError("date_min must be less than or equal to date_max.")

    selected_tickers = _normalize_tickers(tickers)
    selected_token_role = _normalize_token_role(token_role)
    selected_ci_level = _normalize_ci_level(ci_level)

    base = _load_clean_trade_frame(run_id)
    cache_key = _make_cache_key({
        "run_id": run_id,
        "source": base.get("source_cache_key"),
        "date": selected_date.isoformat(),
        "group_by": group_by,
        "date_min": requested_date_min.isoformat() if requested_date_min else None,
        "date_max": requested_date_max.isoformat() if requested_date_max else None,
        "tickers": selected_tickers,
        "token_role": selected_token_role,
        "ci_level": selected_ci_level,
        "exclude_flagged": bool(exclude_flagged),
    })
    cached = _BREAKDOWN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    filtered = _filter_trade_frame(
        base,
        date_min=requested_date_min,
        date_max=requested_date_max,
        tickers=selected_tickers,
        token_role=selected_token_role,
    )
    df = filtered["df"]
    warnings = list(filtered["warnings"])

    if df.empty:
        response = PolymarketDailyAnalyticsBreakdownResponse(
            run_id=run_id,
            date=selected_date.isoformat(),
            group_by=group_by,
            token_role=selected_token_role,
            ci_level=selected_ci_level,
            exclude_flagged=bool(exclude_flagged),
            available_tickers=filtered["available_tickers"],
            selected_tickers=selected_tickers,
            rows=[],
            warnings=["No trade rows matched the selected analytics filters.", *warnings],
        )
        _BREAKDOWN_CACHE[cache_key] = response
        return response

    effective_date_min = requested_date_min or min(df["trade_date"].tolist())
    effective_date_max = requested_date_max or max(df["trade_date"].tolist())
    if not (effective_date_min <= selected_date <= effective_date_max):
        raise ValueError("date must fall inside the effective analytics window.")

    calendar_index = pd.Index(
        [ts.date() for ts in pd.date_range(effective_date_min, effective_date_max, freq="D")],
        name="trade_date",
    )

    if group_by == "ticker":
        meta = df.loc[df["ticker"].notna(), ["ticker"]].drop_duplicates().copy()
        meta["key"] = meta["ticker"]
    else:
        meta = (
            df[["market_id", "ticker", "threshold", "week_friday", "event_endDate", "ladder_bucket"]]
            .drop_duplicates(subset=["market_id"], keep="first")
            .copy()
        )
        meta["key"] = meta["market_id"]
        meta["label"] = meta.apply(
            lambda row: (
                f"{row['ticker']} {int(row['threshold']) if pd.notna(row['threshold']) and float(row['threshold']).is_integer() else row['threshold']}"
                if pd.notna(row["ticker"]) and pd.notna(row["threshold"])
                else str(row["market_id"])
            ),
            axis=1,
        )

    if meta.empty:
        response = PolymarketDailyAnalyticsBreakdownResponse(
            run_id=run_id,
            date=selected_date.isoformat(),
            group_by=group_by,
            token_role=selected_token_role,
            ci_level=selected_ci_level,
            exclude_flagged=bool(exclude_flagged),
            available_tickers=filtered["available_tickers"],
            selected_tickers=selected_tickers,
            rows=[],
            warnings=["No groups matched the selected analytics filters.", *warnings],
        )
        _BREAKDOWN_CACHE[cache_key] = response
        return response

    group_col = "ticker" if group_by == "ticker" else "market_id"
    grouped = (
        df.groupby([group_col, "trade_date"], dropna=False)
        .agg(
            daily_notional_volume=("notional", "sum"),
            share_volume=("size", "sum"),
            trade_count=("trade_id", "size"),
        )
        .reset_index()
    )
    if group_by == "market":
        grouped = grouped.merge(
            df[["market_id", "trade_date", "ticker", "ladder_bucket", "event_proximity_bucket"]]
            .drop_duplicates(subset=["market_id", "trade_date"], keep="first"),
            on=["market_id", "trade_date"],
            how="left",
        )

    market_daily = grouped.copy() if group_by == "market" else pd.DataFrame()
    selected_total = float(
        grouped.loc[grouped["trade_date"] == selected_date, "daily_notional_volume"].sum()
    )

    rows: List[PolymarketDailyAnalyticsBreakdownRow] = []
    for meta_row in meta.itertuples(index=False):
        key = getattr(meta_row, "key")
        group_series = grouped.loc[grouped[group_col] == key, ["trade_date", "daily_notional_volume", "share_volume", "trade_count"]].copy()
        if group_series.empty:
            continue
        observed_days = int(len(group_series))
        group_series = (
            group_series.set_index("trade_date")
            .reindex(calendar_index, fill_value=0.0)
            .reset_index()
            .rename(columns={"trade_date": "date"})
        )
        group_series["trade_count"] = group_series["trade_count"].astype(int)
        group_series["share_volume"] = group_series["share_volume"].astype(float)
        group_series["weekday"] = pd.to_datetime(group_series["date"]).dt.day_name().str[:3]
        group_series["active_markets"] = 0
        group_series["active_tickers"] = 0

        selected_row, history_days = _compute_group_day_row(
            group_series,
            selected_date=selected_date,
            ci_level=selected_ci_level,
            exclude_flagged=bool(exclude_flagged),
        )
        if selected_row is None:
            continue

        expected = None if pd.isna(selected_row.expected_notional_volume) else float(selected_row.expected_notional_volume)
        band_lo = None if pd.isna(selected_row.band_lo) else float(selected_row.band_lo)
        band_hi = None if pd.isna(selected_row.band_hi) else float(selected_row.band_hi)
        ratio = None if pd.isna(selected_row.realized_expected_ratio) else float(selected_row.realized_expected_ratio)
        baseline_source = str(selected_row.baseline_source)

        event_bucket = None
        if group_by == "market":
            day_meta = grouped.loc[
                (grouped["market_id"] == key) & (grouped["trade_date"] == selected_date),
                ["event_proximity_bucket"],
            ]
            if not day_meta.empty and pd.notna(day_meta.iloc[0]["event_proximity_bucket"]):
                event_bucket = str(day_meta.iloc[0]["event_proximity_bucket"])
            elif pd.notna(getattr(meta_row, "event_endDate")):
                event_end = pd.to_datetime(getattr(meta_row, "event_endDate"), utc=True, errors="coerce")
                if pd.notna(event_end):
                    event_bucket = _bucket_event_proximity((event_end.date() - selected_date).days)

            if expected is None:
                peer_expected, peer_lo, peer_hi, peer_source = _peer_market_baseline(
                    market_daily,
                    ticker=getattr(meta_row, "ticker", None),
                    ladder_bucket=getattr(meta_row, "ladder_bucket", None),
                    event_proximity_bucket=event_bucket,
                    selected_date=selected_date,
                    ci_level=selected_ci_level,
                )
                expected = peer_expected
                band_lo = peer_lo
                band_hi = peer_hi
                baseline_source = peer_source
                actual_value = float(selected_row.daily_notional_volume)
                ratio = (actual_value / expected) if expected and expected > 0 else None

        actual_notional = float(selected_row.daily_notional_volume)
        rows.append(
            PolymarketDailyAnalyticsBreakdownRow(
                key=str(key),
                label=str(getattr(meta_row, "label", getattr(meta_row, "ticker", key))),
                ticker=None if group_by == "ticker" else (str(getattr(meta_row, "ticker")) if pd.notna(getattr(meta_row, "ticker", None)) else None),
                market_id=str(getattr(meta_row, "market_id")) if group_by == "market" else None,
                threshold=float(getattr(meta_row, "threshold")) if group_by == "market" and pd.notna(getattr(meta_row, "threshold", None)) else None,
                week_friday=str(getattr(meta_row, "week_friday")) if group_by == "market" and pd.notna(getattr(meta_row, "week_friday", None)) else None,
                event_endDate=str(getattr(meta_row, "event_endDate")) if group_by == "market" and pd.notna(getattr(meta_row, "event_endDate", None)) else None,
                ladder_bucket=str(getattr(meta_row, "ladder_bucket")) if group_by == "market" and pd.notna(getattr(meta_row, "ladder_bucket", None)) else None,
                event_proximity_bucket=event_bucket if group_by == "market" else None,
                daily_notional_volume=actual_notional,
                share_volume=float(selected_row.share_volume),
                trade_count=int(selected_row.trade_count),
                expected_notional_volume=expected,
                band_lo=band_lo,
                band_hi=band_hi,
                realized_expected_ratio=ratio,
                unusually_active=bool((band_hi is not None) and actual_notional > band_hi),
                unusually_inactive=bool((band_lo is not None) and actual_notional < band_lo),
                flagged_outlier=bool(selected_row.flagged_outlier),
                baseline_source=baseline_source,
                history_days=history_days or observed_days,
                volume_share_of_day=(actual_notional / selected_total) if selected_total > 0 else None,
            )
        )

    rows.sort(
        key=lambda row: (
            row.unusually_active,
            row.daily_notional_volume,
            row.realized_expected_ratio if row.realized_expected_ratio is not None else -1.0,
        ),
        reverse=True,
    )

    response = PolymarketDailyAnalyticsBreakdownResponse(
        run_id=run_id,
        date=selected_date.isoformat(),
        group_by=group_by,
        token_role=selected_token_role,
        ci_level=selected_ci_level,
        exclude_flagged=bool(exclude_flagged),
        available_tickers=filtered["available_tickers"],
        selected_tickers=selected_tickers,
        rows=rows,
        warnings=warnings,
    )
    _BREAKDOWN_CACHE[cache_key] = response
    return response
