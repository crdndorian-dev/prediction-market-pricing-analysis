#!/usr/bin/env python3
"""
09-polymarket-analysis-refresh-v1.0.py

Seed the PostgreSQL Polymarket analysis store from weekly-history run artifacts,
optionally backfill trade history from the Polymarket orderbook subgraph, and
materialize market/stock/global daily research tables plus monitoring outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional runtime dependency
    stats = None

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "src" / "webapp" / "backend"
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.core.analysis_db import (  # type: ignore
    ensure_analysis_schemas,
    get_analysis_engine,
    pm_mart_dim_stock,
    pm_mart_fact_global_day,
    pm_mart_fact_market_day,
    pm_mart_fact_market_lifecycle_day,
    pm_mart_fact_stock_day,
    pm_raw_decision_features_daily,
    pm_raw_fact_trade,
    pm_raw_markets_prn_hourly,
    pm_raw_price_history,
    pm_raw_run_manifest,
    pm_raw_weekly_events,
    pm_raw_weekly_markets,
    pm_research_break_event,
    pm_research_drift_monitor,
    pm_research_outlier_flag,
    pm_research_refresh_audit,
    pm_research_threshold_rule,
    pm_research_variable_profile,
    replace_table_rows,
    upsert_rows,
)
from app.settings import load_project_env  # type: ignore
from polymarket.path_utils import manifest_path_candidates  # type: ignore
from polymarket.subgraph_client import SubgraphClient  # type: ignore


RUNS_DIR = REPO_ROOT / "src" / "data" / "raw" / "polymarket" / "weekly_history" / "runs"
GLOBAL_FACT_TRADE_DIR = REPO_ROOT / "src" / "data" / "raw" / "polymarket" / "weekly_history" / "fact_trade"
SCRIPT_VERSION = "1.0.0"
MARKET_SUBTYPE = "weekly_finish_above"
NY_TZ = "America/New_York"
MIN_CONTEXT_ROWS = 60
MIN_CONTEXT_DATES = 20
MIN_TICKER_ROWS = 90
MIN_GLOBAL_DTR_ROWS = 250
MIN_GLOBAL_ROWS = 500
VOLUME_AUTHORITATIVE_THRESHOLD = 0.75
PRICE_EPS = 1e-4
ROLLING_WINDOWS = (30, 90, 180)

MARKET_DAY_THRESHOLD_VARIABLES = [
    "notional_volume",
    "contract_volume",
    "trade_count",
    "avg_trade_size",
    "rv_logit_intraday",
]

STOCK_DAY_THRESHOLD_VARIABLES = [
    "notional_volume",
    "active_market_count",
    "volume_hhi_within_stock",
]

DRIFT_VARIABLES = [
    ("global_day", "notional_volume"),
    ("global_day", "trade_count"),
    ("global_day", "active_market_count"),
    ("stock_day", "notional_volume"),
    ("stock_day", "trade_count"),
    ("stock_day", "volume_hhi_within_stock"),
]

BREAK_VARIABLES = [
    ("global_day", "notional_volume"),
    ("global_day", "trade_count"),
    ("global_day", "active_market_count"),
    ("global_day", "stock_volume_hhi"),
    ("market_day", "rv_logit_intraday"),
    ("stock_day", "notional_volume"),
]


@dataclass
class BackfillResult:
    ok: bool
    rows: int
    start_date: Optional[date]
    end_date: Optional[date]
    error: Optional[str] = None


def progress(stage: str, current: int, total: int, detail: str = "") -> None:
    suffix = f" detail={detail}" if detail else ""
    print(f"[Analysis] PROGRESS stage={stage} current={current} total={total}{suffix}", flush=True)


def load_project_env_into_os() -> None:
    for key, value in load_project_env().items():
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh PostgreSQL Polymarket analysis tables.")
    parser.add_argument("--run-id", action="append", dest="run_ids", default=None)
    parser.add_argument("--skip-trade-backfill", action="store_true")
    parser.add_argument("--force-full-rebuild", action="store_true")
    parser.add_argument("--notes-author", type=str, default=None)
    return parser.parse_args()


def slug_to_stock_name(text_value: Any, ticker: str) -> Optional[str]:
    if not text_value:
        return None
    raw = str(text_value)
    match = re.search(r"Will\s+(.+?)\s+\(" + re.escape(str(ticker)) + r"\)", raw)
    if match:
        return match.group(1).strip()
    return None


def normalize_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text_value = str(value).strip().lower()
    if text_value in {"true", "1", "yes"}:
        return True
    if text_value in {"false", "0", "no"}:
        return False
    return None


def normalize_timestamp_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def trade_date_ny_from_ts(series: pd.Series) -> pd.Series:
    ts = normalize_timestamp_series(series)
    return ts.dt.tz_convert(NY_TZ).dt.date


def clip_probability(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, PRICE_EPS, 1.0 - PRICE_EPS)


def safe_logit(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = clip_probability(values)
    return np.log(arr / (1.0 - arr))


def safe_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(result):
        return result
    return None


def safe_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.date()


def dtr_bucket(value: Any) -> str:
    d = safe_float(value)
    if d is None:
        return "unknown"
    if d <= 1:
        return "dtr_0_1"
    if d <= 3:
        return "dtr_2_3"
    if d <= 7:
        return "dtr_4_7"
    return "dtr_8_plus"


def json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        if np.isnan(value):
            return None
        return value.item()
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    return value


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            clean[key] = json_ready(value)
        records.append(clean)
    return records


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def require_run_artifacts(run_dir: Path) -> None:
    required_files = [
        run_dir / "manifest.json",
        run_dir / "weekly_markets.csv",
        run_dir / "price_history.csv",
    ]
    missing = [path.name for path in required_files if not path.exists()]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Run {run_dir.name} is missing required artifacts: {missing_text}")


def discover_run_dirs(run_ids: Optional[list[str]]) -> list[Path]:
    if run_ids:
        dirs = []
        for run_id in run_ids:
            run_dir = RUNS_DIR / run_id
            if not run_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {run_id}")
            dirs.append(run_dir)
        return dirs
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"Weekly history runs directory not found: {RUNS_DIR}")
    return sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()])


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def decision_features_csv_path(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / f"{run_dir.name}-decision-features.csv",
        run_dir / "decision_features.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_weekly_markets(df: pd.DataFrame, run_id: str, manifest: dict[str, Any], source_file: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["run_id"] = run_id
    df["stock_name"] = df.apply(
        lambda row: slug_to_stock_name(row.get("market_question") or row.get("event_title"), row.get("ticker")),
        axis=1,
    )
    for column in ("event_endDate", "week_monday", "week_friday", "week_sunday"):
        if column in df.columns:
            target = "event_end_date" if column == "event_endDate" else column
            df[target] = pd.to_datetime(df[column], errors="coerce").dt.date
    for column in ("expiry_date_utc", "resolution_time_utc"):
        if column in df.columns:
            df[column] = normalize_timestamp_series(df[column])
    df["enable_order_book"] = df["enable_order_book"].map(normalize_bool) if "enable_order_book" in df.columns else None
    df["active"] = df["active"].map(normalize_bool) if "active" in df.columns else None
    df["closed"] = df["closed"].map(normalize_bool) if "closed" in df.columns else None
    df["threshold"] = pd.to_numeric(df.get("threshold"), errors="coerce")
    df["gamma_yes_price"] = pd.to_numeric(df.get("gamma_yes_price"), errors="coerce")
    df["market_subtype"] = MARKET_SUBTYPE
    df["source_file"] = source_file
    df["source_run_created_at_utc"] = normalize_timestamp_series(pd.Series([manifest.get("created_at_utc")] * len(df)))
    keep = [
        "run_id",
        "market_id",
        "event_id",
        "event_slug",
        "event_title",
        "event_end_date",
        "condition_id",
        "market_slug",
        "market_question",
        "ticker",
        "stock_name",
        "ticker_source",
        "threshold",
        "week_monday",
        "week_friday",
        "week_sunday",
        "expiry_date_utc",
        "resolution_time_utc",
        "yes_token_id",
        "no_token_id",
        "enable_order_book",
        "active",
        "closed",
        "gamma_yes_price",
        "market_subtype",
        "source_file",
        "source_run_created_at_utc",
        "schema_version",
    ]
    return df.reindex(columns=keep)


def normalize_weekly_events(df: pd.DataFrame, run_id: str, manifest: dict[str, Any], source_file: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["run_id"] = run_id
    df["event_end_date"] = pd.to_datetime(df.get("event_endDate"), errors="coerce").dt.date
    ticker_map = {}
    if "event_title" in df.columns:
        for idx, row in df.iterrows():
            title = str(row.get("event_title") or "")
            match = re.search(r"\(([A-Z]+)\)", title)
            ticker_map[idx] = match.group(1) if match else None
    df["ticker"] = pd.Series(ticker_map)
    df["stock_name"] = df.apply(lambda row: slug_to_stock_name(row.get("event_title"), row.get("ticker")), axis=1)
    df["source_file"] = source_file
    df["source_run_created_at_utc"] = normalize_timestamp_series(pd.Series([manifest.get("created_at_utc")] * len(df)))
    keep = [
        "run_id",
        "event_id",
        "event_slug",
        "event_title",
        "event_end_date",
        "ticker",
        "stock_name",
        "source_file",
        "source_run_created_at_utc",
    ]
    return df.reindex(columns=keep)


def normalize_price_history(df: pd.DataFrame, run_id: str, manifest: dict[str, Any], source_file: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["run_id"] = run_id
    df["timestamp_utc"] = normalize_timestamp_series(df["timestamp_utc"])
    df["trade_date_ny"] = trade_date_ny_from_ts(df["timestamp_utc"])
    for column in ("threshold", "price", "price_raw", "fidelity_min"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["source_file"] = source_file
    df["source_run_created_at_utc"] = normalize_timestamp_series(pd.Series([manifest.get("created_at_utc")] * len(df)))
    keep = [
        "run_id",
        "timestamp_utc",
        "market_id",
        "token_id",
        "token_role",
        "ticker",
        "threshold",
        "price",
        "price_raw",
        "fidelity_min",
        "trade_date_ny",
        "source_file",
        "source_run_created_at_utc",
        "schema_version",
    ]
    return df.reindex(columns=keep)


def normalize_markets_prn_hourly(df: pd.DataFrame, run_id: str, manifest: dict[str, Any], source_file: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["run_id"] = run_id
    for column in ("timestamp_utc", "prn_asof_time", "spot_asof_utc", "rn_asof_utc"):
        if column in df.columns:
            df[column] = normalize_timestamp_series(df[column])
    for column in ("week_monday", "week_friday", "week_sunday", "expiry_date", "event_endDate", "snapshot_date"):
        if column in df.columns:
            target = "event_end_date" if column == "event_endDate" else column
            df[target] = pd.to_datetime(df[column], errors="coerce").dt.date
    df["trade_date_ny"] = trade_date_ny_from_ts(df["timestamp_utc"])
    numeric_cols = [
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
        "T_days",
        "S_asof_close",
        "forward_price",
        "dividend_yield",
        "spot",
        "polymarket_buy",
        "polymarket_mid",
        "polymarket_ask",
        "polymarket_bid",
    ]
    for column in numeric_cols:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["source_run_created_at_utc"] = normalize_timestamp_series(pd.Series([manifest.get("created_at_utc")] * len(df)))
    keep = [
        "run_id",
        "timestamp_utc",
        "market_id",
        "week_monday",
        "week_friday",
        "week_sunday",
        "ticker",
        "threshold",
        "expiry_date",
        "event_id",
        "event_end_date",
        "prn_asof_time",
        "snapshot_date",
        "pRN",
        "qRN",
        "pRN_raw",
        "qRN_raw",
        "rv20",
        "log_m",
        "abs_log_m",
        "log_m_fwd",
        "abs_log_m_fwd",
        "T_days",
        "S_asof_close",
        "forward_price",
        "dividend_yield",
        "spot",
        "rn_method",
        "spot_source",
        "spot_asof_utc",
        "rn_asof_utc",
        "polymarket_buy",
        "polymarket_mid",
        "polymarket_ask",
        "polymarket_bid",
        "trade_date_ny",
        "source_run_created_at_utc",
        "schema_version",
    ]
    return df.reindex(columns=keep)


def normalize_decision_features(df: pd.DataFrame, run_id: str, manifest: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["run_id"] = run_id
    df["timestamp_utc"] = normalize_timestamp_series(df["timestamp_utc"])
    df["trade_date_ny"] = trade_date_ny_from_ts(df["timestamp_utc"])
    for column in ("expiry_date_utc", "resolution_time_utc", "prn_asof_time"):
        if column in df.columns:
            df[column] = normalize_timestamp_series(df[column])
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce").dt.date
    df["leak_check_passed"] = df["leak_check_passed"].map(normalize_bool) if "leak_check_passed" in df.columns else None
    numeric_cols = [column for column in df.columns if column not in {"run_id", "timestamp_utc", "market_id", "trade_date_ny", "ticker", "condition_id", "schema_version"}]
    for column in numeric_cols:
        if column in {"snapshot_date", "expiry_date_utc", "resolution_time_utc", "prn_asof_time", "leak_check_passed"}:
            continue
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["source_run_created_at_utc"] = normalize_timestamp_series(pd.Series([manifest.get("created_at_utc")] * len(df)))
    keep = [
        "run_id",
        "timestamp_utc",
        "market_id",
        "trade_date_ny",
        "ticker",
        "threshold",
        "condition_id",
        "expiry_date_utc",
        "resolution_time_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "pm_last",
        "pm_mid",
        "pm_bid",
        "pm_ask",
        "pm_spread",
        "pm_liquidity_proxy",
        "pm_momentum_1h",
        "pm_volatility",
        "pm_momentum_5m",
        "pm_momentum_1d",
        "pm_time_to_resolution",
        "snapshot_date",
        "prn_asof_time",
        "pRN",
        "qRN",
        "pRN_raw",
        "qRN_raw",
        "rv20",
        "log_m",
        "abs_log_m",
        "log_m_fwd",
        "abs_log_m_fwd",
        "T_days",
        "S_asof_close",
        "forward_price",
        "dividend_yield",
        "label",
        "leak_check_passed",
        "source_run_created_at_utc",
        "schema_version",
    ]
    return df.reindex(columns=keep)


def normalize_local_trades(df: pd.DataFrame, source_run_id: str) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["timestamp_utc"] = normalize_timestamp_series(df["timestamp_utc"])
    df["trade_date_ny"] = trade_date_ny_from_ts(df["timestamp_utc"])
    for column in ("block_number", "price", "size"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["source_type"] = "local_csv"
    df["source_run_id"] = source_run_id
    keep = [
        "trade_id",
        "market_id",
        "outcome_token_id",
        "timestamp_utc",
        "trade_date_ny",
        "block_number",
        "price",
        "size",
        "side",
        "tx_hash",
        "source_type",
        "source_run_id",
        "schema_version",
    ]
    return df.reindex(columns=keep)


def load_trade_partitions(run_dir: Path, manifest: dict[str, Any], run_id: str) -> pd.DataFrame:
    paths: list[Path] = []
    fact_trade_dir = manifest.get("fact_trade_dir")
    candidate_dirs: list[Path] = []
    if GLOBAL_FACT_TRADE_DIR.exists():
        candidate_dirs.append(GLOBAL_FACT_TRADE_DIR)
    if fact_trade_dir:
        candidate_dirs.extend(manifest_path_candidates(str(fact_trade_dir), REPO_ROOT))
    for candidate in candidate_dirs:
        if candidate.exists():
            paths.extend(sorted(candidate.glob("date=*/trades.csv")))
    unique_paths: list[Path] = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    if not unique_paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in unique_paths]
    if not frames:
        return pd.DataFrame()
    return normalize_local_trades(pd.concat(frames, ignore_index=True), run_id)


def load_run_bundle(run_dir: Path) -> dict[str, Any]:
    require_run_artifacts(run_dir)
    run_id = run_dir.name
    manifest = load_manifest(run_dir / "manifest.json")
    bundle = {
        "run_id": run_id,
        "manifest": manifest,
        "weekly_markets": normalize_weekly_markets(
            read_csv_if_exists(run_dir / "weekly_markets.csv"),
            run_id,
            manifest,
            "weekly_markets.csv",
        ),
        "weekly_events": normalize_weekly_events(
            read_csv_if_exists(run_dir / "weekly_events.csv"),
            run_id,
            manifest,
            "weekly_events.csv",
        ),
        "price_history": normalize_price_history(
            read_csv_if_exists(run_dir / "price_history.csv"),
            run_id,
            manifest,
            "price_history.csv",
        ),
        "markets_prn_hourly": normalize_markets_prn_hourly(
            read_csv_if_exists(run_dir / "markets_prn_hourly.csv"),
            run_id,
            manifest,
            "markets_prn_hourly.csv",
        ),
        "decision_features": normalize_decision_features(
            read_csv_if_exists(decision_features_csv_path(run_dir)) if decision_features_csv_path(run_dir) else pd.DataFrame(),
            run_id,
            manifest,
        ),
        "local_trades": load_trade_partitions(run_dir, manifest, run_id),
    }
    return bundle


def canonicalize_latest(df: pd.DataFrame, key_columns: list[str], sort_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work = work.sort_values(sort_columns).drop_duplicates(subset=key_columns, keep="last")
    return work.reset_index(drop=True)


def backfill_subgraph_trades(
    market_meta: pd.DataFrame,
    start_date: Optional[date],
    end_date: Optional[date],
) -> tuple[BackfillResult, pd.DataFrame]:
    if market_meta.empty or start_date is None or end_date is None:
        return BackfillResult(ok=True, rows=0, start_date=start_date, end_date=end_date), pd.DataFrame()

    try:
        client = SubgraphClient()
    except Exception as exc:  # pragma: no cover - network/runtime config dependent
        return BackfillResult(ok=False, rows=0, start_date=start_date, end_date=end_date, error=str(exc)), pd.DataFrame()

    since_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    since_ts = int(since_dt.timestamp())
    market_ids = sorted(set(str(value) for value in market_meta["market_id"].dropna().astype(str).tolist()))
    meta_lookup = market_meta.set_index("market_id")[["ticker", "threshold"]].to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for idx in range(0, len(market_ids), 100):
        chunk = market_ids[idx : idx + 100]
        entities = client.fetch_all("tradesByMarket", {"since": since_ts, "marketIds": chunk})
        for entity in entities:
            timestamp = pd.to_datetime(entity.get("timestamp"), unit="s", utc=True, errors="coerce")
            if pd.isna(timestamp) or timestamp >= end_dt:
                continue
            market_id = str(entity.get("marketId"))
            meta = meta_lookup.get(market_id, {})
            rows.append(
                {
                    "trade_id": entity.get("id"),
                    "market_id": market_id,
                    "outcome_token_id": entity.get("outcomeTokenId"),
                    "timestamp_utc": timestamp,
                    "trade_date_ny": timestamp.tz_convert(NY_TZ).date(),
                    "block_number": pd.to_numeric(entity.get("blockNumber"), errors="coerce"),
                    "price": pd.to_numeric(entity.get("price"), errors="coerce"),
                    "size": pd.to_numeric(entity.get("size"), errors="coerce"),
                    "side": str(entity.get("side") or "").lower() or None,
                    "tx_hash": entity.get("transactionHash"),
                    "ticker": meta.get("ticker"),
                    "threshold": meta.get("threshold"),
                    "source_type": "subgraph",
                    "source_run_id": None,
                    "schema_version": "pm_fact_trade_v1.0",
                }
            )
        progress("trade_backfill", min(idx + len(chunk), len(market_ids)), len(market_ids), detail=f"chunk={idx // 100 + 1}")
    if not rows:
        return BackfillResult(ok=True, rows=0, start_date=start_date, end_date=end_date), pd.DataFrame()
    return BackfillResult(ok=True, rows=len(rows), start_date=start_date, end_date=end_date), pd.DataFrame(rows)


def market_calendar_rows(market_meta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in market_meta.iterrows():
        week_monday = safe_date(row.get("week_monday"))
        resolution_ts = row.get("resolution_time_utc")
        resolution_date = safe_date(resolution_ts) or safe_date(row.get("week_friday"))
        if week_monday is None or resolution_date is None:
            continue
        current = week_monday
        while current <= resolution_date:
            rows.append(
                {
                    "trade_date_ny": current,
                    "market_id": row["market_id"],
                    "ticker": row.get("ticker"),
                    "stock_name": row.get("stock_name"),
                    "condition_id": row.get("condition_id"),
                    "event_id": row.get("event_id"),
                    "yes_token_id": row.get("yes_token_id"),
                    "no_token_id": row.get("no_token_id"),
                    "market_subtype": MARKET_SUBTYPE,
                    "threshold": row.get("threshold"),
                    "week_monday": week_monday,
                    "week_friday": safe_date(row.get("week_friday")),
                    "resolution_time_utc": row.get("resolution_time_utc"),
                }
            )
            current += timedelta(days=1)
    return pd.DataFrame(rows)


def aggregate_price_daily(price_history: pd.DataFrame) -> pd.DataFrame:
    if price_history.empty:
        return pd.DataFrame()
    yes_prices = price_history[price_history["token_role"] == "yes"].copy()
    if yes_prices.empty:
        return pd.DataFrame()
    yes_prices = yes_prices.sort_values(["market_id", "timestamp_utc"])
    yes_prices["local_ts"] = yes_prices["timestamp_utc"].dt.tz_convert(NY_TZ)
    yes_prices["local_hour"] = yes_prices["local_ts"].dt.hour
    grouped = []
    for (market_id, trade_date_ny), part in yes_prices.groupby(["market_id", "trade_date_ny"]):
        prices = pd.to_numeric(part["price"], errors="coerce").dropna()
        if prices.empty:
            continue
        pre_close = part[part["local_hour"] <= 16]
        close_1600 = pd.to_numeric(pre_close["price"], errors="coerce").dropna()
        price_values = prices.to_numpy(dtype=float)
        logit_values = safe_logit(price_values)
        grouped.append(
            {
                "market_id": market_id,
                "trade_date_ny": trade_date_ny,
                "hours_observed": int(part["local_hour"].nunique()),
                "open_prob": float(price_values[0]),
                "high_prob": float(np.nanmax(price_values)),
                "low_prob": float(np.nanmin(price_values)),
                "close_prob": float(price_values[-1]),
                "close_prob_1600_et": float(close_1600.iloc[-1]) if not close_1600.empty else float(price_values[-1]),
                "range_prob": float(np.nanmax(price_values) - np.nanmin(price_values)),
                "rv_logit_intraday": float(np.sqrt(np.nansum(np.diff(logit_values) ** 2))) if len(logit_values) > 1 else 0.0,
            }
        )
    out = pd.DataFrame(grouped)
    if out.empty:
        return out
    out = out.sort_values(["market_id", "trade_date_ny"])
    out["prev_close_prob"] = out.groupby("market_id")["close_prob"].shift(1)
    out["gap_logit"] = safe_logit(out["open_prob"].fillna(0.5)) - safe_logit(out["prev_close_prob"].fillna(out["open_prob"]).fillna(0.5))
    out["distance_to_boundary"] = np.minimum(out["close_prob"], 1.0 - out["close_prob"])
    out["price_complete_flag"] = out["hours_observed"] >= 2
    out["has_price_data"] = True
    return out


def aggregate_trade_daily(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    work = trades.copy()
    work["price"] = pd.to_numeric(work["price"], errors="coerce")
    work["size"] = pd.to_numeric(work["size"], errors="coerce")
    work = work.dropna(subset=["timestamp_utc"])
    work["notional"] = work["price"] * work["size"]
    grouped = []
    for (market_id, trade_date_ny), part in work.groupby(["market_id", "trade_date_ny"]):
        size = pd.to_numeric(part["size"], errors="coerce").fillna(0.0)
        notional = pd.to_numeric(part["notional"], errors="coerce").fillna(0.0)
        buy_mask = part["side"].astype(str).str.lower() == "buy"
        sell_mask = part["side"].astype(str).str.lower() == "sell"
        grouped.append(
            {
                "market_id": market_id,
                "trade_date_ny": trade_date_ny,
                "trade_count": int(len(part)),
                "contract_volume": float(size.sum()),
                "notional_volume": float(notional.sum()),
                "buy_volume": float(size[buy_mask].sum()),
                "sell_volume": float(size[sell_mask].sum()),
            }
        )
    out = pd.DataFrame(grouped)
    if out.empty:
        return out
    out["avg_trade_size"] = np.where(out["trade_count"] > 0, out["contract_volume"] / out["trade_count"], np.nan)
    out["buy_sell_imbalance"] = np.where(
        (out["buy_volume"] + out["sell_volume"]) > 0,
        (out["buy_volume"] - out["sell_volume"]) / (out["buy_volume"] + out["sell_volume"]),
        np.nan,
    )
    out["has_trade_data"] = True
    return out


def materialize_market_day(
    market_meta: pd.DataFrame,
    prices_daily: pd.DataFrame,
    trades_daily: pd.DataFrame,
    refresh_id: str,
    trade_coverage_dates: set[date],
) -> pd.DataFrame:
    calendar = market_calendar_rows(market_meta)
    if calendar.empty:
        return pd.DataFrame()
    out = calendar.merge(prices_daily, on=["market_id", "trade_date_ny"], how="left")
    out = out.merge(trades_daily, on=["market_id", "trade_date_ny"], how="left")
    out["listed_flag"] = True
    out["observed_flag"] = out[["open_prob", "close_prob", "trade_count"]].notna().any(axis=1)
    out["volume_complete_flag"] = out["trade_date_ny"].isin(trade_coverage_dates)
    if "has_trade_data" in out.columns:
        out["has_trade_data"] = out["has_trade_data"].fillna(False).astype(bool)
    else:
        out["has_trade_data"] = False
    for column in ("trade_count", "contract_volume", "notional_volume", "buy_volume", "sell_volume"):
        out[column] = pd.to_numeric(out.get(column), errors="coerce")
        out[column] = np.where(out["volume_complete_flag"], out[column].fillna(0.0), out[column])
    out["avg_trade_size"] = np.where(out["trade_count"] > 0, out["contract_volume"] / out["trade_count"], np.nan)
    out["turnover_intensity"] = np.where(out["hours_observed"] > 0, out["notional_volume"] / out["hours_observed"], np.nan)
    out["traded_flag"] = out["trade_count"].fillna(0) > 0
    if "price_complete_flag" in out.columns:
        out["price_complete_flag"] = out["price_complete_flag"].fillna(False)
    else:
        out["price_complete_flag"] = False
    out["has_price_data"] = out["open_prob"].notna() | out["close_prob"].notna()
    resolution_ts = normalize_timestamp_series(out["resolution_time_utc"])
    trade_close_ts = pd.to_datetime(out["trade_date_ny"].astype(str) + " 16:00:00").dt.tz_localize(NY_TZ).dt.tz_convert("UTC")
    out["days_to_resolution"] = (resolution_ts - trade_close_ts).dt.total_seconds() / 86400.0
    out["dtr_bucket"] = out["days_to_resolution"].map(dtr_bucket)
    out["price_impact_proxy"] = np.where(
        pd.to_numeric(out["notional_volume"], errors="coerce").fillna(0.0) > 0,
        np.abs(pd.to_numeric(out["close_prob"], errors="coerce") - pd.to_numeric(out["open_prob"], errors="coerce")) / out["notional_volume"],
        np.nan,
    )
    observed_counts = out.groupby("ticker")["observed_flag"].transform("sum")
    out["sparse_flag"] = observed_counts < MIN_CONTEXT_DATES
    out["coverage_class"] = np.where(out["sparse_flag"], "sparse", np.where(out["volume_complete_flag"], "covered", "incomplete"))
    out["quality_flag_json"] = [
        {
            "volume_complete": bool(volume_complete),
            "price_complete": bool(price_complete),
            "sparse": bool(sparse),
            "has_price_data": bool(has_price),
            "has_trade_data": bool(has_trade),
        }
        for volume_complete, price_complete, sparse, has_price, has_trade in zip(
            out["volume_complete_flag"].fillna(False),
            out["price_complete_flag"].fillna(False),
            out["sparse_flag"].fillna(False),
            out["has_price_data"].fillna(False),
            out["has_trade_data"].fillna(False),
        )
    ]
    out["latest_refresh_id"] = refresh_id
    keep = [column.name for column in pm_mart_fact_market_day.columns if column.name in out.columns]
    return out.reindex(columns=keep)


def materialize_stock_day(market_day: pd.DataFrame, refresh_id: str) -> pd.DataFrame:
    if market_day.empty:
        return pd.DataFrame(columns=[column.name for column in pm_mart_fact_stock_day.columns])
    rows: list[dict[str, Any]] = []
    for (trade_date_ny, ticker), part in market_day.groupby(["trade_date_ny", "ticker"]):
        part = part.copy()
        listed_count = int(part["listed_flag"].fillna(False).sum())
        active_count = int(part["observed_flag"].fillna(False).sum())
        traded_count = int(part["traded_flag"].fillna(False).sum())
        notional = pd.to_numeric(part["notional_volume"], errors="coerce").fillna(0.0)
        contracts = pd.to_numeric(part["contract_volume"], errors="coerce").fillna(0.0)
        trades = pd.to_numeric(part["trade_count"], errors="coerce").fillna(0.0)
        observed_hours = pd.to_numeric(part["hours_observed"], errors="coerce").fillna(0.0)
        total_notional = float(notional.sum())
        shares = (notional / total_notional).replace([np.inf, -np.inf], np.nan).fillna(0.0) if total_notional > 0 else pd.Series(0.0, index=part.index)
        stock_name_values = part["stock_name"].dropna()
        rows.append(
            {
                "trade_date_ny": trade_date_ny,
                "ticker": ticker,
                "stock_name": stock_name_values.iloc[0] if not stock_name_values.empty else ticker,
                "active_market_count": active_count,
                "traded_market_count": traded_count,
                "listed_market_count": listed_count,
                "trade_count": float(trades.sum()),
                "contract_volume": float(contracts.sum()),
                "notional_volume": total_notional,
                "buy_volume": float(pd.to_numeric(part["buy_volume"], errors="coerce").fillna(0.0).sum()),
                "sell_volume": float(pd.to_numeric(part["sell_volume"], errors="coerce").fillna(0.0).sum()),
                "avg_trade_size": float(contracts.sum() / trades.sum()) if trades.sum() > 0 else np.nan,
                "turnover_intensity": float(total_notional / observed_hours.sum()) if observed_hours.sum() > 0 else np.nan,
                "volume_hhi_within_stock": float(np.square(shares).sum()) if total_notional > 0 else np.nan,
                "top_market_share": float(shares.max()) if total_notional > 0 else np.nan,
                "close_prob_dispersion": float(pd.to_numeric(part["close_prob"], errors="coerce").std()) if len(part) > 1 else 0.0,
                "median_days_to_resolution": float(pd.to_numeric(part["days_to_resolution"], errors="coerce").median()) if not part["days_to_resolution"].dropna().empty else np.nan,
                "rv_logit_intraday_mean": float(pd.to_numeric(part["rv_logit_intraday"], errors="coerce").mean()) if not part["rv_logit_intraday"].dropna().empty else np.nan,
                "missingness_rate": float(1.0 - part["has_price_data"].fillna(False).mean()) if listed_count else np.nan,
                "sparsity_ratio": float(1.0 - (active_count / listed_count)) if listed_count else np.nan,
                "volume_complete_flag": bool(part["volume_complete_flag"].fillna(False).all()),
                "coverage_class": "covered" if bool(part["volume_complete_flag"].fillna(False).all()) else "incomplete",
                "threshold_scope_used": None,
                "latest_refresh_id": refresh_id,
            }
        )
    return pd.DataFrame(rows).reindex(columns=[column.name for column in pm_mart_fact_stock_day.columns])


def materialize_global_day(stock_day: pd.DataFrame, refresh_id: str) -> pd.DataFrame:
    if stock_day.empty:
        return pd.DataFrame(columns=[column.name for column in pm_mart_fact_global_day.columns])
    rows: list[dict[str, Any]] = []
    for trade_date_ny, part in stock_day.groupby("trade_date_ny"):
        notional = pd.to_numeric(part["notional_volume"], errors="coerce").fillna(0.0)
        total_notional = float(notional.sum())
        shares = (notional / total_notional).replace([np.inf, -np.inf], np.nan).fillna(0.0) if total_notional > 0 else pd.Series(0.0, index=part.index)
        rows.append(
            {
                "trade_date_ny": trade_date_ny,
                "active_stock_count": int((part["active_market_count"].fillna(0) > 0).sum()),
                "active_market_count": int(pd.to_numeric(part["active_market_count"], errors="coerce").fillna(0).sum()),
                "traded_market_count": int(pd.to_numeric(part["traded_market_count"], errors="coerce").fillna(0).sum()),
                "trade_count": float(pd.to_numeric(part["trade_count"], errors="coerce").fillna(0.0).sum()),
                "contract_volume": float(pd.to_numeric(part["contract_volume"], errors="coerce").fillna(0.0).sum()),
                "notional_volume": total_notional,
                "stock_volume_hhi": float(np.square(shares).sum()) if total_notional > 0 else np.nan,
                "top_stock_share": float(shares.max()) if total_notional > 0 else np.nan,
                "cross_sectional_close_dispersion": float(pd.to_numeric(part["close_prob_dispersion"], errors="coerce").mean()) if not part["close_prob_dispersion"].dropna().empty else np.nan,
                "missingness_rate": float(pd.to_numeric(part["missingness_rate"], errors="coerce").mean()) if not part["missingness_rate"].dropna().empty else np.nan,
                "volume_complete_flag": bool(part["volume_complete_flag"].fillna(False).all()),
                "latest_refresh_id": refresh_id,
            }
        )
    return pd.DataFrame(rows).reindex(columns=[column.name for column in pm_mart_fact_global_day.columns])


def materialize_market_lifecycle_day(market_day: pd.DataFrame, refresh_id: str) -> pd.DataFrame:
    if market_day.empty:
        return pd.DataFrame(columns=[column.name for column in pm_mart_fact_market_lifecycle_day.columns])
    work = market_day.sort_values(["market_id", "trade_date_ny"]).copy()
    work["lifecycle_day_index"] = work.groupby("market_id").cumcount() + 1
    work["lifecycle_total_days"] = work.groupby("market_id")["trade_date_ny"].transform("count")
    work["lifecycle_progress"] = work["lifecycle_day_index"] / work["lifecycle_total_days"].replace(0, np.nan)
    work["latest_refresh_id"] = refresh_id
    keep = [column.name for column in pm_mart_fact_market_lifecycle_day.columns]
    return work.reindex(columns=keep)


def compute_hill_tail(values: Iterable[float]) -> Optional[float]:
    arr = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=float)
    if arr.size < 10:
        return None
    threshold = np.nanpercentile(arr, 95)
    tail = np.sort(arr[arr >= threshold])
    if tail.size < 5 or threshold <= 0:
        return None
    logs = np.log(tail / threshold)
    denom = np.nanmean(logs)
    if not np.isfinite(denom) or denom <= 0:
        return None
    return float(1.0 / denom)


def describe_numeric(values: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {}
    quantiles = np.nanpercentile(arr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    median = float(np.nanmedian(arr))
    mad = float(np.nanmedian(np.abs(arr - median)))
    summary = {
        "mean": float(np.nanmean(arr)),
        "median": median,
        "stddev": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min_value": float(np.nanmin(arr)),
        "max_value": float(np.nanmax(arr)),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p10": float(quantiles[2]),
        "p25": float(quantiles[3]),
        "p50": float(quantiles[4]),
        "p75": float(quantiles[5]),
        "p90": float(quantiles[6]),
        "p95": float(quantiles[7]),
        "p99": float(quantiles[8]),
        "iqr": float(quantiles[5] - quantiles[3]),
        "mad": mad,
        "hill_tail_index": compute_hill_tail(arr),
    }
    if stats is not None and arr.size > 2:
        try:
            summary["skewness"] = float(stats.skew(arr, bias=False, nan_policy="omit"))
            summary["kurtosis"] = float(stats.kurtosis(arr, fisher=True, bias=False, nan_policy="omit"))
        except Exception:
            summary["skewness"] = None
            summary["kurtosis"] = None
    else:
        summary["skewness"] = None
        summary["kurtosis"] = None
    return summary


def build_variable_profiles(
    market_day: pd.DataFrame,
    stock_day: pd.DataFrame,
    global_day: pd.DataFrame,
    refresh_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = [
        ("market_day", "ALL", market_day, ["notional_volume", "contract_volume", "trade_count", "rv_logit_intraday", "days_to_resolution"]),
        ("global_day", "ALL", global_day, ["notional_volume", "trade_count", "active_market_count", "stock_volume_hhi"]),
    ]
    for ticker, part in stock_day.groupby("ticker"):
        specs.append(("stock_day", ticker, part.copy(), ["notional_volume", "trade_count", "active_market_count", "volume_hhi_within_stock"]))
    for scope_level, scope_key, frame, variables in specs:
        if frame.empty or "trade_date_ny" not in frame.columns:
            continue
        frame = frame.sort_values("trade_date_ny")
        for variable in variables:
            if variable not in frame.columns:
                continue
            for window in ("full", "rolling_30d", "rolling_90d", "rolling_180d"):
                if window == "full":
                    part = frame.copy()
                else:
                    days = int(window.split("_")[1].replace("d", ""))
                    cutoff = frame["trade_date_ny"].max() - timedelta(days=days - 1)
                    part = frame[frame["trade_date_ny"] >= cutoff].copy()
                summary = describe_numeric(part[variable])
                if not summary:
                    continue
                rows.append(
                    {
                        "refresh_id": refresh_id,
                        "scope_level": scope_level,
                        "scope_key": scope_key,
                        "window_label": window,
                        "variable_name": variable,
                        "analysis_date": frame["trade_date_ny"].max(),
                        "sample_start_date": part["trade_date_ny"].min(),
                        "sample_end_date": part["trade_date_ny"].max(),
                        "observation_count": int(pd.to_numeric(part[variable], errors="coerce").dropna().shape[0]),
                        "date_count": int(part["trade_date_ny"].nunique()),
                        **summary,
                        "metrics_json": None,
                    }
                )
    return pd.DataFrame(rows).reindex(columns=[column.name for column in pm_research_variable_profile.columns if column.name != "profile_id"])


def choose_context_market_day(frame: pd.DataFrame, row: pd.Series, variable_name: str) -> tuple[pd.DataFrame, str, str]:
    current_date = row["trade_date_ny"]
    window_start = current_date - timedelta(days=180)
    base = frame[
        (frame["trade_date_ny"] < current_date)
        & (frame["trade_date_ny"] >= window_start)
        & frame["volume_complete_flag"].fillna(False)
    ].copy()
    base["metric_value"] = pd.to_numeric(base[variable_name], errors="coerce")
    base = base.dropna(subset=["metric_value"])
    candidates = [
        ("ticker_x_dtr", base[(base["ticker"] == row["ticker"]) & (base["dtr_bucket"] == row["dtr_bucket"])]),
        ("ticker", base[base["ticker"] == row["ticker"]]),
        ("global_x_dtr", base[base["dtr_bucket"] == row["dtr_bucket"]]),
        ("global", base),
    ]
    for scope, candidate in candidates:
        count = int(candidate["metric_value"].shape[0])
        dates = int(candidate["trade_date_ny"].nunique())
        if scope == "ticker_x_dtr" and count >= MIN_CONTEXT_ROWS and dates >= MIN_CONTEXT_DATES:
            return candidate, scope, "dense_ticker_dtr"
        if scope == "ticker" and count >= MIN_TICKER_ROWS:
            return candidate, scope, "dense_ticker"
        if scope == "global_x_dtr" and count >= MIN_GLOBAL_DTR_ROWS:
            return candidate, scope, "fallback_global_dtr"
        if scope == "global" and count >= MIN_GLOBAL_ROWS:
            return candidate, scope, "fallback_global"
    return pd.DataFrame(), "sparse", "sparse"


def choose_context_stock_day(frame: pd.DataFrame, row: pd.Series, variable_name: str) -> tuple[pd.DataFrame, str, str]:
    current_date = row["trade_date_ny"]
    window_start = current_date - timedelta(days=180)
    base = frame[(frame["trade_date_ny"] < current_date) & (frame["trade_date_ny"] >= window_start)].copy()
    base["metric_value"] = pd.to_numeric(base[variable_name], errors="coerce")
    base = base.dropna(subset=["metric_value"])
    ticker_candidate = base[base["ticker"] == row["ticker"]]
    if int(ticker_candidate["metric_value"].shape[0]) >= MIN_TICKER_ROWS:
        return ticker_candidate, "ticker", "dense_ticker"
    if int(base["metric_value"].shape[0]) >= MIN_GLOBAL_ROWS:
        return base, "global", "fallback_global"
    return pd.DataFrame(), "sparse", "sparse"


def compute_threshold_from_context(candidate: pd.DataFrame) -> dict[str, Any]:
    metric = np.log1p(pd.to_numeric(candidate["metric_value"], errors="coerce").dropna().to_numpy(dtype=float))
    if metric.size == 0:
        return {}
    q95, q99, q995 = np.nanpercentile(metric, [95, 99, 99.5])
    median = float(np.nanmedian(metric))
    mad = float(np.nanmedian(np.abs(metric - median)))
    sigma_robust = max(1.4826 * mad, 0.10)
    cap_robust = median + 5.0 * sigma_robust
    cap_quant = q995 if metric.size >= MIN_GLOBAL_DTR_ROWS else q99
    upper_cap = max(q95, min(cap_quant, cap_robust))
    return {
        "q95": float(q95),
        "q99": float(q99),
        "q995": float(q995),
        "median": median,
        "mad": mad,
        "sigma_robust": float(sigma_robust),
        "cap_robust": float(cap_robust),
        "cap_quant": float(cap_quant),
        "upper_cap": float(upper_cap),
        "sample_count": int(metric.size),
        "date_count": int(candidate["trade_date_ny"].nunique()),
    }


def build_market_day_thresholds_and_flags(market_day: pd.DataFrame, refresh_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    updated = market_day.copy()
    for variable in MARKET_DAY_THRESHOLD_VARIABLES:
        if variable not in updated.columns:
            continue
        clean_values: dict[Any, Any] = {}
        sparse_flags: dict[Any, bool] = {}
        for idx, row in updated.sort_values(["trade_date_ny", "ticker", "market_id"]).iterrows():
            raw_value = safe_float(row.get(variable))
            candidate, scope_used, coverage_class = choose_context_market_day(updated, row, variable)
            rule = compute_threshold_from_context(candidate)
            authoritative = bool(row.get("volume_complete_flag")) if variable in {"notional_volume", "contract_volume", "trade_count", "avg_trade_size"} else bool(row.get("has_price_data"))
            upper_cap = rule.get("upper_cap")
            clean_value = raw_value
            clipped = False
            if raw_value is not None and upper_cap is not None and authoritative:
                log_value = math.log1p(max(raw_value, 0.0))
                clipped = log_value > upper_cap
                clean_value = math.expm1(min(log_value, upper_cap))
            threshold_rows.append(
                {
                    "refresh_id": refresh_id,
                    "trade_date_ny": row["trade_date_ny"],
                    "variable_name": variable,
                    "scope_level": "market_day",
                    "scope_key": f"{row['ticker']}::{row['market_id']}",
                    "dtr_bucket": row.get("dtr_bucket"),
                    "window_days": 180,
                    "sample_count": rule.get("sample_count"),
                    "date_count": rule.get("date_count"),
                    "coverage_class": coverage_class,
                    "scope_used": scope_used,
                    "q95": rule.get("q95"),
                    "q99": rule.get("q99"),
                    "q995": rule.get("q995"),
                    "median": rule.get("median"),
                    "mad": rule.get("mad"),
                    "sigma_robust": rule.get("sigma_robust"),
                    "cap_robust": rule.get("cap_robust"),
                    "cap_quant": rule.get("cap_quant"),
                    "upper_cap": upper_cap,
                    "authoritative_flag": authoritative,
                    "sparse_flag": scope_used == "sparse",
                    "metrics_json": {
                        "ticker": row["ticker"],
                        "market_id": row["market_id"],
                    },
                }
            )
            outlier_rows.append(
                {
                    "refresh_id": refresh_id,
                    "trade_date_ny": row["trade_date_ny"],
                    "market_id": row["market_id"],
                    "ticker": row["ticker"],
                    "scope_level": "market_day",
                    "scope_key": f"{row['ticker']}::{row['market_id']}",
                    "dtr_bucket": row.get("dtr_bucket"),
                    "variable_name": variable,
                    "raw_value": raw_value,
                    "clean_value": clean_value,
                    "upper_cap": upper_cap,
                    "rule_scope_used": scope_used,
                    "clipped_flag": clipped,
                    "diagnostic_flags_json": {
                        "coverage_class": coverage_class,
                        "volume_complete_flag": bool(row.get("volume_complete_flag")),
                    },
                    "authoritative_flag": authoritative,
                }
            )
            clean_values[idx] = clean_value
            sparse_flags[idx] = scope_used == "sparse"
        updated[f"{variable}_clean"] = updated.index.map(clean_values.get)
        if variable == "notional_volume":
            updated["sparse_flag"] = updated.index.map(lambda key: sparse_flags.get(key, False))
            updated["coverage_class"] = np.where(updated["sparse_flag"], "sparse", updated["coverage_class"])
    return (
        updated,
        pd.DataFrame(threshold_rows).reindex(columns=[column.name for column in pm_research_threshold_rule.columns if column.name != "rule_id"]),
        pd.DataFrame(outlier_rows).reindex(columns=[column.name for column in pm_research_outlier_flag.columns if column.name != "flag_id"]),
    )


def build_stock_day_thresholds_and_flags(stock_day: pd.DataFrame, refresh_id: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    updated = stock_day.copy()
    scope_used_map: dict[Any, Optional[str]] = {}
    for variable in STOCK_DAY_THRESHOLD_VARIABLES:
        if variable not in updated.columns:
            continue
        clean_values: dict[Any, Any] = {}
        for idx, row in updated.sort_values(["trade_date_ny", "ticker"]).iterrows():
            raw_value = safe_float(row.get(variable))
            candidate, scope_used, coverage_class = choose_context_stock_day(updated, row, variable)
            rule = compute_threshold_from_context(candidate)
            authoritative = bool(row.get("volume_complete_flag")) if variable == "notional_volume" else True
            upper_cap = rule.get("upper_cap")
            clean_value = raw_value
            clipped = False
            if raw_value is not None and upper_cap is not None and authoritative:
                log_value = math.log1p(max(raw_value, 0.0))
                clipped = log_value > upper_cap
                clean_value = math.expm1(min(log_value, upper_cap))
            threshold_rows.append(
                {
                    "refresh_id": refresh_id,
                    "trade_date_ny": row["trade_date_ny"],
                    "variable_name": variable,
                    "scope_level": "stock_day",
                    "scope_key": row["ticker"],
                    "dtr_bucket": None,
                    "window_days": 180,
                    "sample_count": rule.get("sample_count"),
                    "date_count": rule.get("date_count"),
                    "coverage_class": coverage_class,
                    "scope_used": scope_used,
                    "q95": rule.get("q95"),
                    "q99": rule.get("q99"),
                    "q995": rule.get("q995"),
                    "median": rule.get("median"),
                    "mad": rule.get("mad"),
                    "sigma_robust": rule.get("sigma_robust"),
                    "cap_robust": rule.get("cap_robust"),
                    "cap_quant": rule.get("cap_quant"),
                    "upper_cap": upper_cap,
                    "authoritative_flag": authoritative,
                    "sparse_flag": scope_used == "sparse",
                    "metrics_json": {"ticker": row["ticker"]},
                }
            )
            outlier_rows.append(
                {
                    "refresh_id": refresh_id,
                    "trade_date_ny": row["trade_date_ny"],
                    "market_id": None,
                    "ticker": row["ticker"],
                    "scope_level": "stock_day",
                    "scope_key": row["ticker"],
                    "dtr_bucket": None,
                    "variable_name": variable,
                    "raw_value": raw_value,
                    "clean_value": clean_value,
                    "upper_cap": upper_cap,
                    "rule_scope_used": scope_used,
                    "clipped_flag": clipped,
                    "diagnostic_flags_json": {"coverage_class": coverage_class},
                    "authoritative_flag": authoritative,
                }
            )
            clean_values[idx] = clean_value
            if variable == "notional_volume":
                scope_used_map[idx] = scope_used
        updated[f"{variable}_clean"] = updated.index.map(clean_values.get)
    updated["threshold_scope_used"] = updated.index.map(scope_used_map.get)
    return (
        updated,
        pd.DataFrame(threshold_rows).reindex(columns=[column.name for column in pm_research_threshold_rule.columns if column.name != "rule_id"]),
        pd.DataFrame(outlier_rows).reindex(columns=[column.name for column in pm_research_outlier_flag.columns if column.name != "flag_id"]),
    )


def psi_score(baseline: np.ndarray, recent: np.ndarray, bins: int = 10) -> Optional[float]:
    if baseline.size < 10 or recent.size < 10:
        return None
    cut_points = np.unique(np.nanpercentile(baseline, np.linspace(0, 100, bins + 1)))
    if cut_points.size < 3:
        return None
    expected, _ = np.histogram(baseline, bins=cut_points)
    actual, _ = np.histogram(recent, bins=cut_points)
    expected = expected / max(expected.sum(), 1)
    actual = actual / max(actual.sum(), 1)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def build_drift_monitor(
    market_day: pd.DataFrame,
    stock_day: pd.DataFrame,
    global_day: pd.DataFrame,
    refresh_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames = {
        "market_day": market_day,
        "stock_day": stock_day,
        "global_day": global_day,
    }
    for scope_level, variable_name in DRIFT_VARIABLES:
        frame = frames[scope_level]
        if variable_name not in frame.columns:
            continue
        grouped = [("ALL", frame)] if scope_level != "stock_day" else [(ticker, part.copy()) for ticker, part in frame.groupby("ticker")]
        for scope_key, part in grouped:
            part = part.sort_values("trade_date_ny")
            series = pd.to_numeric(part[variable_name], errors="coerce").dropna()
            if series.shape[0] < 20:
                continue
            recent_part = part.tail(30)
            baseline_part = part.iloc[max(0, len(part) - 210) : max(0, len(part) - 30)]
            baseline = pd.to_numeric(baseline_part[variable_name], errors="coerce").dropna().to_numpy(dtype=float)
            recent = pd.to_numeric(recent_part[variable_name], errors="coerce").dropna().to_numpy(dtype=float)
            if baseline.size < 10 or recent.size < 10:
                continue
            psi = psi_score(baseline, recent)
            ks_stat = None
            ks_pvalue = None
            if stats is not None:
                try:
                    ks = stats.ks_2samp(baseline, recent)
                    ks_stat = float(ks.statistic)
                    ks_pvalue = float(ks.pvalue)
                except Exception:
                    pass
            baseline_q25, baseline_q75 = np.nanpercentile(baseline, [25, 75])
            baseline_iqr = max(float(baseline_q75 - baseline_q25), 1e-6)
            recent_q10, recent_q50, recent_q90 = np.nanpercentile(recent, [10, 50, 90])
            baseline_q10, baseline_q50, baseline_q90 = np.nanpercentile(baseline, [10, 50, 90])
            median_shift = float(recent_q50 - baseline_q50)
            recent_iqr = float(np.nanpercentile(recent, 75) - np.nanpercentile(recent, 25))
            iqr_ratio = float(recent_iqr / baseline_iqr)
            drift_flag = bool((psi is not None and psi > 0.25) or (ks_pvalue is not None and ks_pvalue < 0.05 and abs(median_shift) > 0.25 * baseline_iqr))
            authoritative = True
            if scope_level in {"market_day", "stock_day"} and "volume_complete_flag" in part.columns:
                authoritative = bool(part["volume_complete_flag"].fillna(False).tail(30).all())
            rows.append(
                {
                    "refresh_id": refresh_id,
                    "scope_level": scope_level,
                    "scope_key": scope_key,
                    "variable_name": variable_name,
                    "window_label": "recent_30d_vs_prior_180d",
                    "baseline_start_date": baseline_part["trade_date_ny"].min(),
                    "baseline_end_date": baseline_part["trade_date_ny"].max(),
                    "recent_start_date": recent_part["trade_date_ny"].min(),
                    "recent_end_date": recent_part["trade_date_ny"].max(),
                    "baseline_count": int(baseline.size),
                    "recent_count": int(recent.size),
                    "psi": psi,
                    "ks_stat": ks_stat,
                    "ks_pvalue": ks_pvalue,
                    "median_shift": median_shift,
                    "iqr_ratio": iqr_ratio,
                    "quantile_drift_json": {
                        "baseline_q10": float(baseline_q10),
                        "baseline_q50": float(baseline_q50),
                        "baseline_q90": float(baseline_q90),
                        "recent_q10": float(recent_q10),
                        "recent_q50": float(recent_q50),
                        "recent_q90": float(recent_q90),
                    },
                    "drift_flag": drift_flag,
                    "authoritative_flag": authoritative,
                }
            )
    return pd.DataFrame(rows).reindex(columns=[column.name for column in pm_research_drift_monitor.columns if column.name != "drift_id"])


def optimal_partition_breaks(values: np.ndarray, min_segment_length: int = 20) -> list[int]:
    n = values.size
    if n < min_segment_length * 2:
        return []
    mean = np.nanmean(values)
    std = np.nanstd(values)
    scaled = (values - mean) / max(std, 1e-6)
    prefix = np.concatenate([[0.0], np.cumsum(scaled)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(scaled ** 2)])

    def segment_cost(start: int, end: int) -> float:
        length = end - start
        if length <= 0:
            return 0.0
        total = prefix[end] - prefix[start]
        total_sq = prefix_sq[end] - prefix_sq[start]
        return float(total_sq - (total ** 2) / length)

    penalty = 3.0 * math.log(max(n, 2))
    best = np.full(n + 1, np.inf)
    previous = np.full(n + 1, -1, dtype=int)
    best[0] = -penalty
    for end in range(min_segment_length, n + 1):
        for start in range(0, end - min_segment_length + 1):
            if start > 0 and start < min_segment_length:
                continue
            if end - start < min_segment_length:
                continue
            candidate = best[start] + segment_cost(start, end) + penalty
            if candidate < best[end]:
                best[end] = candidate
                previous[end] = start
    breaks: list[int] = []
    cursor = n
    while cursor > 0 and previous[cursor] >= 0:
        start = int(previous[cursor])
        if start == 0:
            break
        breaks.append(start)
        cursor = start
    return sorted(set(breaks))


def build_break_events(
    market_day: pd.DataFrame,
    stock_day: pd.DataFrame,
    global_day: pd.DataFrame,
    refresh_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    frames = {
        "market_day": market_day.groupby("trade_date_ny", as_index=False)[["rv_logit_intraday"]].mean().assign(scope_key="ALL"),
        "stock_day": stock_day,
        "global_day": global_day.assign(scope_key="ALL"),
    }
    for scope_level, variable_name in BREAK_VARIABLES:
        frame = frames[scope_level]
        if variable_name not in frame.columns:
            continue
        grouped = [("ALL", frame)] if scope_level != "stock_day" else [(ticker, part.copy()) for ticker, part in frame.groupby("ticker")]
        for scope_key, part in grouped:
            values = pd.to_numeric(part[variable_name], errors="coerce").dropna()
            if values.shape[0] < 40:
                continue
            ordered = part.loc[values.index].sort_values("trade_date_ny")
            series = values.loc[ordered.index].to_numpy(dtype=float)
            break_indices = optimal_partition_breaks(np.log1p(np.maximum(series, 0.0)) if variable_name != "rv_logit_intraday" else series)
            for idx in break_indices:
                if idx <= 0 or idx >= len(ordered):
                    continue
                before = series[max(0, idx - 20) : idx]
                after = series[idx : min(len(series), idx + 20)]
                rows.append(
                    {
                        "refresh_id": refresh_id,
                        "scope_level": scope_level,
                        "scope_key": scope_key,
                        "variable_name": variable_name,
                        "break_date": ordered.iloc[idx]["trade_date_ny"],
                        "method": "optimal_partition_mean_shift",
                        "segment_start_date": ordered.iloc[max(0, idx - 20)]["trade_date_ny"],
                        "segment_end_date": ordered.iloc[min(len(ordered) - 1, idx + 19)]["trade_date_ny"],
                        "before_mean": float(np.nanmean(before)) if before.size else None,
                        "after_mean": float(np.nanmean(after)) if after.size else None,
                        "penalty_value": float(3.0 * math.log(max(len(series), 2))),
                        "score": float(abs(np.nanmean(after) - np.nanmean(before))) if before.size and after.size else None,
                        "metrics_json": {"break_index": int(idx)},
                    }
                )
    return pd.DataFrame(rows).reindex(columns=[column.name for column in pm_research_break_event.columns if column.name != "break_id"])


def build_dim_stock(market_meta: pd.DataFrame, market_day: pd.DataFrame, refresh_id: str) -> pd.DataFrame:
    if market_meta.empty:
        return pd.DataFrame(columns=[column.name for column in pm_mart_dim_stock.columns])
    rows = []
    for ticker, part in market_meta.groupby("ticker"):
        stock_name = part["stock_name"].dropna().iloc[0] if not part["stock_name"].dropna().empty else ticker
        active_days = 0
        if not market_day.empty:
            active_days = int(market_day[market_day["ticker"] == ticker]["trade_date_ny"].nunique())
        rows.append(
            {
                "ticker": ticker,
                "stock_name": stock_name,
                "market_subtype": MARKET_SUBTYPE,
                "first_listed_date": part["week_monday"].min(),
                "last_listed_date": part["week_friday"].max(),
                "market_count": int(part["market_id"].nunique()),
                "active_trading_days": active_days,
                "latest_refresh_id": refresh_id,
            }
        )
    return pd.DataFrame(rows).reindex(columns=[column.name for column in pm_mart_dim_stock.columns])


def write_refresh_audit(
    conn,
    refresh_id: str,
    started_at: datetime,
    finished_at: datetime,
    ok: bool,
    volume_authoritative: bool,
    rows_loaded: dict[str, Any],
    coverage: dict[str, Any],
    params: dict[str, Any],
    warning_text: Optional[str] = None,
    error_text: Optional[str] = None,
) -> None:
    replace = [
        {
            "refresh_id": refresh_id,
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "ok": ok,
            "volume_authoritative": volume_authoritative,
            "rows_loaded_json": rows_loaded,
            "coverage_json": coverage,
            "params_json": params,
            "warning_text": warning_text,
            "error_text": error_text,
        }
    ]
    upsert_rows(conn, pm_research_refresh_audit, replace, conflict_columns=["refresh_id"])


def clear_computed_tables(conn) -> None:
    for table in [
        pm_mart_fact_market_lifecycle_day,
        pm_mart_fact_global_day,
        pm_mart_fact_stock_day,
        pm_mart_fact_market_day,
        pm_mart_dim_stock,
        pm_research_break_event,
        pm_research_drift_monitor,
        pm_research_outlier_flag,
        pm_research_threshold_rule,
        pm_research_variable_profile,
    ]:
        conn.execute(table.delete())


def clear_raw_tables(conn) -> None:
    for table in [
        pm_raw_decision_features_daily,
        pm_raw_markets_prn_hourly,
        pm_raw_fact_trade,
        pm_raw_price_history,
        pm_raw_weekly_events,
        pm_raw_weekly_markets,
        pm_raw_run_manifest,
    ]:
        conn.execute(table.delete())


def main() -> None:
    started_at = datetime.now(timezone.utc)
    load_project_env_into_os()
    args = parse_args()
    refresh_id = f"pm-analysis-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    print(f"[Analysis] refresh_id={refresh_id}", flush=True)

    run_dirs = discover_run_dirs(args.run_ids)
    progress("discover_runs", 0, len(run_dirs), detail="start")
    bundles: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        bundles.append(load_run_bundle(run_dir))
        progress("discover_runs", index, len(run_dirs), detail=run_dir.name)

    weekly_markets = pd.concat([bundle["weekly_markets"] for bundle in bundles if not bundle["weekly_markets"].empty], ignore_index=True) if bundles else pd.DataFrame()
    weekly_events = pd.concat([bundle["weekly_events"] for bundle in bundles if not bundle["weekly_events"].empty], ignore_index=True) if bundles else pd.DataFrame()
    price_history = pd.concat([bundle["price_history"] for bundle in bundles if not bundle["price_history"].empty], ignore_index=True) if bundles else pd.DataFrame()
    markets_prn_hourly = pd.concat([bundle["markets_prn_hourly"] for bundle in bundles if not bundle["markets_prn_hourly"].empty], ignore_index=True) if bundles else pd.DataFrame()
    decision_features = pd.concat([bundle["decision_features"] for bundle in bundles if not bundle["decision_features"].empty], ignore_index=True) if bundles else pd.DataFrame()
    local_trades = pd.concat([bundle["local_trades"] for bundle in bundles if not bundle["local_trades"].empty], ignore_index=True) if bundles else pd.DataFrame()

    manifest_rows = []
    for bundle in bundles:
        manifest = bundle["manifest"]
        manifest_rows.append(
            {
                "run_id": bundle["run_id"],
                "run_dir": str((RUNS_DIR / bundle["run_id"]).relative_to(REPO_ROOT)),
                "created_at_utc": normalize_timestamp_series(pd.Series([manifest.get("created_at_utc")])).iloc[0],
                "finished_at_utc": normalize_timestamp_series(pd.Series([manifest.get("finished_at_utc")])).iloc[0],
                "status": manifest.get("status"),
                "markets": manifest.get("markets"),
                "price_rows": manifest.get("price_rows"),
                "tickers_json": manifest.get("tickers"),
                "pipeline_args_json": manifest.get("pipeline_args"),
                "artifacts_json": manifest.get("artifacts"),
                "backfill_trade_ok": None,
                "backfill_trade_error": None,
                "backfill_trade_start_date": None,
                "backfill_trade_end_date": None,
                "backfill_trade_rows": None,
            }
        )
    run_manifest_df = pd.DataFrame(manifest_rows)

    market_meta = canonicalize_latest(weekly_markets, ["market_id"], ["source_run_created_at_utc", "run_id"])
    price_history_canonical = canonicalize_latest(price_history, ["market_id", "token_role", "token_id", "timestamp_utc"], ["source_run_created_at_utc", "run_id"])
    prn_canonical = canonicalize_latest(markets_prn_hourly, ["market_id", "timestamp_utc"], ["source_run_created_at_utc", "run_id"])
    decision_canonical = canonicalize_latest(decision_features, ["market_id", "timestamp_utc"], ["source_run_created_at_utc", "run_id"])
    local_trades = local_trades.drop_duplicates(subset=["trade_id"], keep="last") if not local_trades.empty else local_trades

    trade_backfill_warning = None
    backfill_rows_df = pd.DataFrame()
    backfill_start = market_meta["week_monday"].min() if not market_meta.empty else None
    backfill_end = market_meta["week_friday"].max() if not market_meta.empty else None
    if not args.skip_trade_backfill:
        try:
            backfill_result = backfill_subgraph_trades(market_meta, backfill_start, backfill_end)
            if isinstance(backfill_result, tuple):
                backfill_meta, backfill_rows_df = backfill_result
            else:
                backfill_meta = backfill_result
            if not backfill_meta.ok:
                trade_backfill_warning = backfill_meta.error
            for row in manifest_rows:
                row["backfill_trade_ok"] = backfill_meta.ok
                row["backfill_trade_error"] = backfill_meta.error
                row["backfill_trade_start_date"] = backfill_meta.start_date
                row["backfill_trade_end_date"] = backfill_meta.end_date
                row["backfill_trade_rows"] = backfill_meta.rows
        except Exception as exc:  # pragma: no cover - runtime/network dependent
            trade_backfill_warning = str(exc)
    else:
        trade_backfill_warning = "Trade backfill skipped by request."

    all_trades = pd.concat([local_trades, backfill_rows_df], ignore_index=True) if not backfill_rows_df.empty or not local_trades.empty else pd.DataFrame()
    if not all_trades.empty:
        all_trades = all_trades.drop_duplicates(subset=["trade_id"], keep="last")

    trade_coverage_dates: set[date] = set()
    if backfill_start and backfill_end and (trade_backfill_warning is None):
        current = backfill_start
        while current <= backfill_end:
            trade_coverage_dates.add(current)
            current += timedelta(days=1)
    else:
        if not all_trades.empty:
            trade_coverage_dates = set(pd.to_datetime(all_trades["trade_date_ny"], errors="coerce").dt.date.dropna().tolist())

    prices_daily = aggregate_price_daily(price_history_canonical)
    trades_daily = aggregate_trade_daily(all_trades)
    market_day = materialize_market_day(market_meta, prices_daily, trades_daily, refresh_id, trade_coverage_dates)
    stock_day = materialize_stock_day(market_day, refresh_id)
    global_day = materialize_global_day(stock_day, refresh_id)
    lifecycle_day = materialize_market_lifecycle_day(market_day, refresh_id)

    market_day, market_thresholds, market_outliers = build_market_day_thresholds_and_flags(market_day, refresh_id)
    stock_day, stock_thresholds, stock_outliers = build_stock_day_thresholds_and_flags(stock_day, refresh_id)
    dim_stock = build_dim_stock(market_meta, market_day, refresh_id)
    variable_profiles = build_variable_profiles(market_day, stock_day, global_day, refresh_id)
    drift_monitor = build_drift_monitor(market_day, stock_day, global_day, refresh_id)
    break_events = build_break_events(market_day, stock_day, global_day, refresh_id)
    threshold_rules = pd.concat([market_thresholds, stock_thresholds], ignore_index=True) if not market_thresholds.empty or not stock_thresholds.empty else pd.DataFrame(columns=[column.name for column in pm_research_threshold_rule.columns if column.name != "rule_id"])
    outlier_flags = pd.concat([market_outliers, stock_outliers], ignore_index=True) if not market_outliers.empty or not stock_outliers.empty else pd.DataFrame(columns=[column.name for column in pm_research_outlier_flag.columns if column.name != "flag_id"])

    volume_authoritative_ratio = float(market_day["volume_complete_flag"].fillna(False).mean()) if not market_day.empty else 0.0
    volume_authoritative = volume_authoritative_ratio >= VOLUME_AUTHORITATIVE_THRESHOLD

    engine = ensure_analysis_schemas(get_analysis_engine())
    rows_loaded: dict[str, int] = {}
    with engine.begin() as conn:
        if args.force_full_rebuild:
            clear_computed_tables(conn)
            clear_raw_tables(conn)
        progress("write_raw", 0, 7, detail="start")
        rows_loaded["pm_raw.weekly_markets"] = upsert_rows(conn, pm_raw_weekly_markets, df_to_records(weekly_markets))
        progress("write_raw", 1, 7, detail="weekly_markets")
        rows_loaded["pm_raw.weekly_events"] = upsert_rows(conn, pm_raw_weekly_events, df_to_records(weekly_events))
        progress("write_raw", 2, 7, detail="weekly_events")
        rows_loaded["pm_raw.price_history"] = upsert_rows(conn, pm_raw_price_history, df_to_records(price_history))
        progress("write_raw", 3, 7, detail="price_history")
        rows_loaded["pm_raw.fact_trade"] = upsert_rows(conn, pm_raw_fact_trade, df_to_records(all_trades))
        progress("write_raw", 4, 7, detail="fact_trade")
        rows_loaded["pm_raw.markets_prn_hourly"] = upsert_rows(conn, pm_raw_markets_prn_hourly, df_to_records(markets_prn_hourly))
        progress("write_raw", 5, 7, detail="markets_prn_hourly")
        rows_loaded["pm_raw.decision_features_daily"] = upsert_rows(conn, pm_raw_decision_features_daily, df_to_records(decision_features))
        rows_loaded["pm_raw.run_manifest"] = upsert_rows(conn, pm_raw_run_manifest, df_to_records(pd.DataFrame(manifest_rows)), conflict_columns=["run_id"])
        progress("write_raw", 7, 7, detail="run_manifest")

        clear_computed_tables(conn)
        progress("write_marts", 0, 7, detail="start")
        rows_loaded["pm_mart.dim_stock"] = replace_table_rows(conn, pm_mart_dim_stock, df_to_records(dim_stock))
        progress("write_marts", 1, 7, detail="dim_stock")
        rows_loaded["pm_mart.fact_market_day"] = replace_table_rows(conn, pm_mart_fact_market_day, df_to_records(market_day))
        progress("write_marts", 2, 7, detail="fact_market_day")
        rows_loaded["pm_mart.fact_stock_day"] = replace_table_rows(conn, pm_mart_fact_stock_day, df_to_records(stock_day))
        progress("write_marts", 3, 7, detail="fact_stock_day")
        rows_loaded["pm_mart.fact_global_day"] = replace_table_rows(conn, pm_mart_fact_global_day, df_to_records(global_day))
        progress("write_marts", 4, 7, detail="fact_global_day")
        rows_loaded["pm_mart.fact_market_lifecycle_day"] = replace_table_rows(conn, pm_mart_fact_market_lifecycle_day, df_to_records(lifecycle_day))
        progress("write_marts", 5, 7, detail="fact_market_lifecycle_day")
        rows_loaded["pm_research.variable_profile"] = replace_table_rows(conn, pm_research_variable_profile, df_to_records(variable_profiles))
        rows_loaded["pm_research.threshold_rule"] = replace_table_rows(conn, pm_research_threshold_rule, df_to_records(threshold_rules))
        rows_loaded["pm_research.outlier_flag"] = replace_table_rows(conn, pm_research_outlier_flag, df_to_records(outlier_flags))
        rows_loaded["pm_research.drift_monitor"] = replace_table_rows(conn, pm_research_drift_monitor, df_to_records(drift_monitor))
        rows_loaded["pm_research.break_event"] = replace_table_rows(conn, pm_research_break_event, df_to_records(break_events))
        progress("write_marts", 7, 7, detail="research_tables")

        finished_at = datetime.now(timezone.utc)
        coverage = {
            "volume_authoritative_ratio": volume_authoritative_ratio,
            "trade_coverage_dates": len(trade_coverage_dates),
            "market_day_count": int(len(market_day)),
            "stock_day_count": int(len(stock_day)),
            "global_day_count": int(len(global_day)),
        }
        write_refresh_audit(
            conn,
            refresh_id=refresh_id,
            started_at=started_at,
            finished_at=finished_at,
            ok=True,
            volume_authoritative=volume_authoritative,
            rows_loaded=rows_loaded,
            coverage=coverage,
            params={
                "run_ids": args.run_ids,
                "skip_trade_backfill": args.skip_trade_backfill,
                "force_full_rebuild": args.force_full_rebuild,
                "notes_author": args.notes_author,
                "script_version": SCRIPT_VERSION,
            },
            warning_text=trade_backfill_warning,
        )

    print(
        json.dumps(
            {
                "refresh_id": refresh_id,
                "rows_loaded": rows_loaded,
                "volume_authoritative": volume_authoritative,
                "volume_authoritative_ratio": volume_authoritative_ratio,
                "warning": trade_backfill_warning,
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
