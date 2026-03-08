from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import numpy as np
from sqlalchemy import text

from app.core.analysis_db import ensure_analysis_schemas, get_analysis_engine, list_analysis_tables, pm_research_research_note
from app.models.analysis import (
    AnalysisAlert,
    AnalysisCoverageSummary,
    AnalysisDriftResponse,
    AnalysisHeadlineMetric,
    AnalysisHistogramBin,
    AnalysisJobStatus,
    AnalysisOverviewResponse,
    AnalysisPriceBehaviorResponse,
    AnalysisProgress,
    AnalysisRefreshRequest,
    AnalysisRefreshResult,
    AnalysisSeriesPoint,
    AnalysisStructureResponse,
    AnalysisTableResponse,
    AnalysisTableRow,
    AnalysisThresholdRow,
    AnalysisVolumeResponse,
    ResearchNoteCreateRequest,
    ResearchNoteResponse,
    ResearchNotesResponse,
)
from app.settings import load_project_env
from app.services.process_runtime import spawn_managed_process


BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "09-polymarket-analysis-refresh-v1.0.py"

_PROGRESS_RE = re.compile(
    r"\[Analysis\] PROGRESS stage=(?P<stage>\w+) current=(?P<current>\d+) total=(?P<total>\d+)(?: detail=(?P<detail>.*))?"
)
_REFRESH_RE = re.compile(r"\[Analysis\] refresh_id=(?P<refresh_id>[0-9A-Za-z._:-]+)")

TABLE_OBJECTS = list_analysis_tables()
ALLOWED_TABLES = {
    "fact_market_day": TABLE_OBJECTS["pm_mart.fact_market_day"],
    "fact_stock_day": TABLE_OBJECTS["pm_mart.fact_stock_day"],
    "fact_global_day": TABLE_OBJECTS["pm_mart.fact_global_day"],
    "variable_profile": TABLE_OBJECTS["pm_research.variable_profile"],
    "threshold_rule": TABLE_OBJECTS["pm_research.threshold_rule"],
    "outlier_flag": TABLE_OBJECTS["pm_research.outlier_flag"],
    "drift_monitor": TABLE_OBJECTS["pm_research.drift_monitor"],
    "break_event": TABLE_OBJECTS["pm_research.break_event"],
    "refresh_audit": TABLE_OBJECTS["pm_research.refresh_audit"],
}

VOLUME_VARIABLE_SPECS: Dict[str, Dict[str, Any]] = {
    "notional_volume": {
        "series_global_column": "notional_volume",
        "series_stock_column": "notional_volume",
        "series_stock_value_2": "trade_count",
        "distribution_table": "pm_mart.fact_market_day",
        "distribution_value_column": "notional_volume",
        "distribution_clean_column": "notional_volume_clean",
        "threshold_scope_level": "market_day",
        "per_stock_column": "notional_volume",
    },
    "contract_volume": {
        "series_global_column": "contract_volume",
        "series_stock_column": "contract_volume",
        "series_stock_value_2": None,
        "distribution_table": "pm_mart.fact_market_day",
        "distribution_value_column": "contract_volume",
        "distribution_clean_column": "contract_volume_clean",
        "threshold_scope_level": "market_day",
        "per_stock_column": "contract_volume",
    },
    "trade_count": {
        "series_global_column": "trade_count",
        "series_stock_column": "trade_count",
        "series_stock_value_2": "notional_volume",
        "distribution_table": "pm_mart.fact_market_day",
        "distribution_value_column": "trade_count",
        "distribution_clean_column": "trade_count_clean",
        "threshold_scope_level": "market_day",
        "per_stock_column": "trade_count",
    },
    "avg_trade_size": {
        "series_global_expr": "CASE WHEN trade_count > 0 THEN contract_volume / trade_count ELSE NULL END",
        "series_stock_column": "avg_trade_size",
        "series_stock_value_2": "trade_count",
        "distribution_table": "pm_mart.fact_market_day",
        "distribution_value_column": "avg_trade_size",
        "distribution_clean_column": "avg_trade_size_clean",
        "threshold_scope_level": "market_day",
        "per_stock_column": "avg_trade_size",
    },
    "rv_logit_intraday": {
        "series_global_query": """
            SELECT trade_date_ny, AVG(rv_logit_intraday) AS value
            FROM pm_mart.fact_market_day
            GROUP BY trade_date_ny
            ORDER BY trade_date_ny DESC
            LIMIT :limit
        """,
        "series_ticker_query": """
            SELECT trade_date_ny, rv_logit_intraday_mean AS value
            FROM pm_mart.fact_stock_day
            WHERE ticker = :ticker
            ORDER BY trade_date_ny DESC
            LIMIT :limit
        """,
        "series_stock_column": "rv_logit_intraday_mean",
        "series_stock_value_2": None,
        "distribution_table": "pm_mart.fact_market_day",
        "distribution_value_column": "rv_logit_intraday",
        "distribution_clean_column": "rv_logit_intraday_clean",
        "threshold_scope_level": "market_day",
        "per_stock_column": "rv_logit_intraday_mean",
    },
    "active_market_count": {
        "series_global_column": "active_market_count",
        "series_stock_column": "active_market_count",
        "series_stock_value_2": "traded_market_count",
        "distribution_table": "pm_mart.fact_stock_day",
        "distribution_value_column": "active_market_count",
        "distribution_clean_column": "active_market_count_clean",
        "threshold_scope_level": "stock_day",
        "per_stock_column": "active_market_count",
    },
    "volume_hhi_within_stock": {
        "series_global_query": """
            SELECT trade_date_ny, AVG(volume_hhi_within_stock) AS value
            FROM pm_mart.fact_stock_day
            GROUP BY trade_date_ny
            ORDER BY trade_date_ny DESC
            LIMIT :limit
        """,
        "series_ticker_query": """
            SELECT trade_date_ny, volume_hhi_within_stock AS value
            FROM pm_mart.fact_stock_day
            WHERE ticker = :ticker
            ORDER BY trade_date_ny DESC
            LIMIT :limit
        """,
        "series_stock_column": "volume_hhi_within_stock",
        "series_stock_value_2": "top_market_share",
        "distribution_table": "pm_mart.fact_stock_day",
        "distribution_value_column": "volume_hhi_within_stock",
        "distribution_clean_column": "volume_hhi_within_stock_clean",
        "threshold_scope_level": "stock_day",
        "per_stock_column": "volume_hhi_within_stock",
    },
}


def _handle_response_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        item: Dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, datetime):
                item[key] = value.isoformat()
            elif hasattr(value, "isoformat") and not isinstance(value, (str, bytes)):
                try:
                    item[key] = value.isoformat()
                except Exception:
                    item[key] = value
            else:
                item[key] = value
        cleaned.append(item)
    return cleaned


def _fetch_all(sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    engine = get_analysis_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        return _handle_response_rows([dict(row) for row in result.mappings().all()])


def _fetch_one(sql: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    rows = _fetch_all(sql, params)
    return rows[0] if rows else None


def _load_env() -> Dict[str, str]:
    env = {**os.environ}
    env.update(load_project_env())
    existing = env.get("PYTHONPATH")
    root = str(BASE_DIR)
    env["PYTHONPATH"] = os.pathsep.join([existing, root]) if existing else root
    return env


def _normalize_ticker(ticker: Optional[str]) -> Optional[str]:
    if ticker is None:
        return None
    value = ticker.strip().upper()
    return value or None


def _require_volume_variable(variable_name: str) -> Dict[str, Any]:
    spec = VOLUME_VARIABLE_SPECS.get(variable_name)
    if spec is None:
        allowed = ", ".join(sorted(VOLUME_VARIABLE_SPECS))
        raise ValueError(f"Unsupported analysis variable: {variable_name}. Allowed values: {allowed}")
    return spec


def _latest_thresholds(
    *,
    variable_name: str,
    scope_level: str,
    ticker: Optional[str],
    limit: int = 100,
) -> List[AnalysisThresholdRow]:
    where_clauses = ["variable_name = :variable_name", "scope_level = :scope_level"]
    params: Dict[str, Any] = {
        "variable_name": variable_name,
        "scope_level": scope_level,
        "limit": limit,
    }
    if scope_level == "stock_day" and ticker:
        params["scope_key"] = ticker
        where_clauses.append("scope_key = :scope_key")
    sql = f"""
        SELECT trade_date_ny, variable_name, scope_used, coverage_class, upper_cap, q95, q99, q995, sample_count, sparse_flag, authoritative_flag, metrics_json
        FROM pm_research.threshold_rule
        WHERE {' AND '.join(where_clauses)}
        ORDER BY trade_date_ny DESC
        LIMIT :limit
    """
    rows = _fetch_all(sql, params)
    typed_rows: List[AnalysisThresholdRow] = []
    for row in rows:
        metrics = row.get("metrics_json") or {}
        typed_rows.append(
            AnalysisThresholdRow(
                trade_date_ny=str(row.get("trade_date_ny")),
                ticker=metrics.get("ticker"),
                variable_name=str(row.get("variable_name")),
                scope_used=str(row.get("scope_used")),
                coverage_class=row.get("coverage_class"),
                upper_cap=row.get("upper_cap"),
                q95=row.get("q95"),
                q99=row.get("q99"),
                q995=row.get("q995"),
                sample_count=row.get("sample_count"),
                sparse_flag=bool(row.get("sparse_flag")),
                authoritative_flag=bool(row.get("authoritative_flag")),
            )
        )
    return typed_rows


def _series_query_for_variable(spec: Dict[str, Any], *, ticker: Optional[str]) -> tuple[str, Dict[str, Any], Optional[str]]:
    params: Dict[str, Any] = {}
    if ticker and spec.get("series_ticker_query"):
        params["ticker"] = ticker
        return spec["series_ticker_query"], params, "stock_day"
    if ticker and spec.get("series_stock_column"):
        value_2_column = spec.get("series_stock_value_2")
        select_value_2 = f", {value_2_column} AS value_2" if value_2_column else ""
        params["ticker"] = ticker
        return (
            f"""
            SELECT trade_date_ny, {spec['series_stock_column']} AS value{select_value_2}
            FROM pm_mart.fact_stock_day
            WHERE ticker = :ticker
            ORDER BY trade_date_ny DESC
            LIMIT :limit
            """,
            params,
            "stock_day",
        )
    if spec.get("series_global_query"):
        return spec["series_global_query"], params, "global_day"
    if spec.get("series_global_expr"):
        return (
            f"""
            SELECT trade_date_ny, {spec['series_global_expr']} AS value
            FROM pm_mart.fact_global_day
            ORDER BY trade_date_ny DESC
            LIMIT :limit
            """,
            params,
            "global_day",
        )
    return (
        f"""
        SELECT trade_date_ny, {spec['series_global_column']} AS value
        FROM pm_mart.fact_global_day
        ORDER BY trade_date_ny DESC
        LIMIT :limit
        """,
        params,
        "global_day",
    )


def _distribution_query_for_variable(spec: Dict[str, Any], *, ticker: Optional[str]) -> tuple[str, Dict[str, Any]]:
    table_name = spec["distribution_table"]
    value_column = spec["distribution_value_column"]
    clean_column = spec["distribution_clean_column"]
    params: Dict[str, Any] = {}
    where_clauses = [f"{value_column} IS NOT NULL"]
    if ticker:
        params["ticker"] = ticker
        where_clauses.append("ticker = :ticker")
    return (
        f"""
        SELECT {value_column} AS value, COALESCE({clean_column}, {value_column}) AS clean_value
        FROM {table_name}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY trade_date_ny DESC
        LIMIT 5000
        """,
        params,
    )


def _table_order_sql(table) -> str:
    primary_keys = [column.name for column in table.primary_key.columns]
    if primary_keys:
        return ", ".join(f"{column} DESC" for column in primary_keys)
    first_column = next(iter(table.columns), None)
    return f"{first_column.name} DESC" if first_column is not None else "1 DESC"


def _analysis_table_query_parts(
    *,
    table_name: str,
    ticker: Optional[str],
    variable_name: Optional[str],
) -> tuple[Any, str, Dict[str, Any], str]:
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Unsupported table: {table_name}")
    table = ALLOWED_TABLES[table_name]
    columns = {column.name for column in table.columns}
    where_clauses = ["1=1"]
    params: Dict[str, Any] = {}
    if ticker is not None:
        if "ticker" not in columns:
            raise ValueError(f"Table {table_name} does not support ticker filtering.")
        params["ticker"] = ticker
        where_clauses.append("ticker = :ticker")
    if variable_name is not None:
        if "variable_name" not in columns:
            raise ValueError(f"Table {table_name} does not support variable filtering.")
        params["variable_name"] = variable_name
        where_clauses.append("variable_name = :variable_name")
    return table, " AND ".join(where_clauses), params, _table_order_sql(table)


def _histogram(values: List[float], bins: int = 20) -> List[AnalysisHistogramBin]:
    if not values:
        return []
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if array.size == 0:
        return []
    counts, edges = np.histogram(array, bins=min(bins, max(5, int(np.sqrt(array.size)))))
    return [
        AnalysisHistogramBin(start=float(edges[i]), end=float(edges[i + 1]), count=int(counts[i]))
        for i in range(len(counts))
    ]


def _latest_coverage() -> AnalysisCoverageSummary:
    refresh = _fetch_one(
        """
        SELECT refresh_id, finished_at_utc, created_at_utc, volume_authoritative, coverage_json
        FROM pm_research.refresh_audit
        ORDER BY created_at_utc DESC
        LIMIT 1
        """
    )
    counts = _fetch_one(
        """
        SELECT
          (SELECT COUNT(*) FROM pm_mart.fact_market_day) AS market_day_count,
          (SELECT COUNT(*) FROM pm_mart.fact_stock_day) AS stock_day_count,
          (SELECT COUNT(*) FROM pm_mart.fact_global_day) AS global_day_count,
          (SELECT COUNT(*) FROM pm_mart.dim_stock) AS stock_count,
          COALESCE((SELECT MAX(active_market_count) FROM pm_mart.fact_global_day), 0) AS active_market_count
        """
    ) or {}
    coverage_json = (refresh or {}).get("coverage_json") or {}
    return AnalysisCoverageSummary(
        latest_refresh_id=(refresh or {}).get("refresh_id"),
        latest_refresh_at=(refresh or {}).get("finished_at_utc") or (refresh or {}).get("created_at_utc"),
        volume_authoritative=bool((refresh or {}).get("volume_authoritative")),
        market_day_count=int(counts.get("market_day_count") or 0),
        stock_day_count=int(counts.get("stock_day_count") or 0),
        global_day_count=int(counts.get("global_day_count") or 0),
        stock_count=int(counts.get("stock_count") or 0),
        active_market_count=int(counts.get("active_market_count") or 0),
        trade_coverage_ratio=(coverage_json or {}).get("volume_authoritative_ratio"),
    )


def _series_from_rows(
    rows: List[Dict[str, Any]],
    *,
    x_key: str = "trade_date_ny",
    y_key: str = "value",
    y2_key: Optional[str] = None,
    y3_key: Optional[str] = None,
    label_key: Optional[str] = None,
) -> List[AnalysisSeriesPoint]:
    points: List[AnalysisSeriesPoint] = []
    for row in rows:
        points.append(
            AnalysisSeriesPoint(
                date=str(row.get(x_key)),
                value=row.get(y_key),
                value_2=row.get(y2_key) if y2_key else None,
                value_3=row.get(y3_key) if y3_key else None,
                label=str(row.get(label_key)) if label_key and row.get(label_key) is not None else None,
                metadata={key: value for key, value in row.items() if key not in {x_key, y_key, y2_key, y3_key, label_key}},
            )
        )
    return points


def build_analysis_overview() -> AnalysisOverviewResponse:
    coverage = _latest_coverage()
    latest_global = _fetch_one(
        """
        SELECT trade_date_ny, notional_volume, trade_count, active_market_count, top_stock_share
        FROM pm_mart.fact_global_day
        ORDER BY trade_date_ny DESC
        LIMIT 1
        """
    ) or {}
    previous_global = _fetch_one(
        """
        SELECT trade_date_ny, notional_volume, trade_count
        FROM pm_mart.fact_global_day
        ORDER BY trade_date_ny DESC
        OFFSET 1 LIMIT 1
        """
    ) or {}
    recent_rows = _fetch_all(
        """
        SELECT trade_date_ny, notional_volume AS value, trade_count AS value_2, active_market_count
        FROM pm_mart.fact_global_day
        ORDER BY trade_date_ny DESC
        LIMIT 30
        """
    )
    recent_rows.reverse()
    drift_alerts = _fetch_all(
        """
        SELECT scope_level, scope_key, variable_name, psi, ks_pvalue, median_shift
        FROM pm_research.drift_monitor
        WHERE drift_flag = TRUE
        ORDER BY created_at_utc DESC
        LIMIT 5
        """
    )
    alerts = [
        AnalysisAlert(
            level="warning",
            title=f"Drift: {row.get('variable_name')}",
            detail=f"{row.get('scope_level')} {row.get('scope_key')} PSI={row.get('psi')}",
        )
        for row in drift_alerts
    ]
    if not coverage.volume_authoritative:
        alerts.insert(
            0,
            AnalysisAlert(
                level="critical",
                title="Volume coverage is incomplete",
                detail="Volume-based research outputs are blocked or marked non-authoritative until trade backfill coverage improves.",
            ),
        )
    previous_notional = float(previous_global.get("notional_volume") or 0.0)
    latest_notional = float(latest_global.get("notional_volume") or 0.0)
    delta = None
    if previous_notional:
        delta = (latest_notional - previous_notional) / previous_notional
    tickers = _fetch_all(
        """
        SELECT ticker
        FROM pm_mart.dim_stock
        ORDER BY ticker ASC
        """
    )
    return AnalysisOverviewResponse(
        coverage=coverage,
        headline_metrics=[
            AnalysisHeadlineMetric(label="Latest NY date", value=latest_global.get("trade_date_ny")),
            AnalysisHeadlineMetric(label="Total notional volume", value=latest_notional, delta=delta),
            AnalysisHeadlineMetric(label="Daily trade count", value=latest_global.get("trade_count")),
            AnalysisHeadlineMetric(label="Active markets", value=latest_global.get("active_market_count")),
            AnalysisHeadlineMetric(label="Top stock share", value=latest_global.get("top_stock_share")),
        ],
        recent_trends=_series_from_rows(recent_rows, y_key="value", y2_key="value_2"),
        alerts=alerts,
        available_tickers=[str(row["ticker"]) for row in tickers if row.get("ticker")],
    )


def build_analysis_volume(
    *,
    ticker: Optional[str] = None,
    variable_name: str = "notional_volume",
    limit: int = 120,
) -> AnalysisVolumeResponse:
    ticker = _normalize_ticker(ticker)
    spec = _require_volume_variable(variable_name)
    coverage = _latest_coverage()
    total_sql, total_params, series_scope_level = _series_query_for_variable(spec, ticker=ticker)
    total_params["limit"] = limit
    total_series = _fetch_all(total_sql, total_params)
    total_series.reverse()
    per_stock_sql = f"""
        SELECT trade_date_ny, ticker AS label, {spec['per_stock_column']} AS value
        FROM pm_mart.fact_stock_day
        {{where_clause}}
        ORDER BY trade_date_ny DESC
        LIMIT :limit
    """.format(
        where_clause="WHERE ticker = :ticker" if ticker else "",
    )
    params: Dict[str, Any] = {"limit": limit}
    if ticker:
        params["ticker"] = ticker
    per_stock_rows = _fetch_all(per_stock_sql, params)
    per_stock_rows.reverse()
    distribution_sql, distribution_params = _distribution_query_for_variable(spec, ticker=ticker)
    distribution_rows = _fetch_all(distribution_sql, distribution_params)
    raw_values = [float(row["value"]) for row in distribution_rows if row.get("value") is not None]
    log_values = [float(np.log1p(value)) for value in raw_values if value >= 0]
    outlier_where = ["variable_name = :variable_name", "scope_level = :scope_level"]
    outlier_params: Dict[str, Any] = {
        "variable_name": variable_name,
        "scope_level": spec["threshold_scope_level"],
    }
    if ticker:
        outlier_params["ticker"] = ticker
        outlier_where.append("ticker = :ticker")
    outlier_rows = _fetch_all(
        f"""
        SELECT trade_date_ny, ticker, market_id, variable_name, raw_value, clean_value, upper_cap, rule_scope_used, clipped_flag
        FROM pm_research.outlier_flag
        WHERE {' AND '.join(outlier_where)}
        ORDER BY trade_date_ny DESC
        LIMIT 100
        """,
        outlier_params,
    )
    typed_thresholds = _latest_thresholds(
        variable_name=variable_name,
        scope_level=spec["threshold_scope_level"],
        ticker=ticker,
    )
    clipped_share = None
    if outlier_rows:
        clipped_share = float(np.mean([1.0 if row.get("clipped_flag") else 0.0 for row in outlier_rows]))
    return AnalysisVolumeResponse(
        coverage=coverage,
        selected_variable=variable_name,
        selected_scope_level=series_scope_level or spec["threshold_scope_level"],
        total_volume_series=_series_from_rows(total_series, y_key="value", y2_key="value_2"),
        per_stock_series=_series_from_rows(per_stock_rows, y_key="value", label_key="label"),
        hist_raw=_histogram(raw_values),
        hist_log=_histogram(log_values),
        outlier_diagnostics=outlier_rows,
        threshold_rows=typed_thresholds,
        clipped_share=clipped_share,
    )


def build_analysis_structure(limit: int = 120) -> AnalysisStructureResponse:
    coverage = _latest_coverage()
    concentration = _fetch_all(
        """
        SELECT trade_date_ny, stock_volume_hhi AS value, top_stock_share AS value_2
        FROM pm_mart.fact_global_day
        ORDER BY trade_date_ny DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )
    concentration.reverse()
    active_counts = _fetch_all(
        """
        SELECT trade_date_ny, active_market_count AS value, traded_market_count AS value_2
        FROM pm_mart.fact_global_day
        ORDER BY trade_date_ny DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )
    active_counts.reverse()
    participation = _fetch_all(
        """
        SELECT trade_date_ny, active_stock_count AS value
        FROM pm_mart.fact_global_day
        ORDER BY trade_date_ny DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )
    participation.reverse()
    lifecycle = _fetch_all(
        """
        SELECT ticker, AVG(lifecycle_progress) AS avg_lifecycle_progress, AVG(notional_volume) AS avg_notional_volume
        FROM pm_mart.fact_market_lifecycle_day
        GROUP BY ticker
        ORDER BY avg_notional_volume DESC NULLS LAST
        LIMIT 25
        """
    )
    return AnalysisStructureResponse(
        coverage=coverage,
        stock_concentration=_series_from_rows(concentration, y_key="value", y2_key="value_2"),
        active_market_counts=_series_from_rows(active_counts, y_key="value", y2_key="value_2"),
        stock_participation=_series_from_rows(participation),
        lifecycle_summary=lifecycle,
    )


def build_analysis_price_behavior(*, ticker: Optional[str] = None, limit: int = 120) -> AnalysisPriceBehaviorResponse:
    ticker = _normalize_ticker(ticker)
    coverage = _latest_coverage()
    where = "WHERE ticker = :ticker" if ticker else ""
    params: Dict[str, Any] = {"limit": limit}
    if ticker:
        params["ticker"] = ticker
    distribution_rows = _fetch_all(
        f"""
        SELECT close_prob
        FROM pm_mart.fact_market_day
        {where}
        AND close_prob IS NOT NULL
        ORDER BY trade_date_ny DESC
        LIMIT 5000
        """ if ticker else """
        SELECT close_prob
        FROM pm_mart.fact_market_day
        WHERE close_prob IS NOT NULL
        ORDER BY trade_date_ny DESC
        LIMIT 5000
        """,
        params if ticker else None,
    )
    close_values = [float(row["close_prob"]) for row in distribution_rows if row.get("close_prob") is not None]
    volatility = _fetch_all(
        f"""
        SELECT trade_date_ny, AVG(rv_logit_intraday) AS value, AVG(distance_to_boundary) AS value_2
        FROM pm_mart.fact_market_day
        {where}
        GROUP BY trade_date_ny
        ORDER BY trade_date_ny DESC
        LIMIT :limit
        """,
        params,
    )
    volatility.reverse()
    expiry_behavior = _fetch_all(
        f"""
        SELECT dtr_bucket AS label, AVG(notional_volume) AS value, AVG(rv_logit_intraday) AS value_2
        FROM pm_mart.fact_market_day
        {where}
        GROUP BY dtr_bucket
        ORDER BY dtr_bucket
        """,
        params if ticker else None,
    )
    convergence = _fetch_all(
        f"""
        SELECT ticker, AVG(distance_to_boundary) AS avg_distance_to_boundary, AVG(close_prob) AS avg_close_prob
        FROM pm_mart.fact_market_day
        {where}
        GROUP BY ticker
        ORDER BY avg_distance_to_boundary ASC NULLS LAST
        LIMIT 25
        """,
        params if ticker else None,
    )
    return AnalysisPriceBehaviorResponse(
        coverage=coverage,
        close_probability_distribution=_histogram(close_values),
        volatility_series=_series_from_rows(volatility, y_key="value", y2_key="value_2"),
        expiry_behavior=_series_from_rows(expiry_behavior, y_key="value", y2_key="value_2", label_key="label"),
        convergence_table=convergence,
    )


def build_analysis_drift(*, ticker: Optional[str] = None, variable_name: str = "notional_volume", limit: int = 120) -> AnalysisDriftResponse:
    ticker = _normalize_ticker(ticker)
    spec = _require_volume_variable(variable_name)
    coverage = _latest_coverage()
    rolling_sql, rolling_params, _ = _series_query_for_variable(spec, ticker=ticker)
    rolling_params["limit"] = limit
    rolling_rows = _fetch_all(rolling_sql, rolling_params)
    rolling_rows.reverse()
    quantiles = _fetch_all(
        """
        SELECT sample_end_date AS trade_date_ny, p10 AS value, p50 AS value_2, p90 AS value_3
        FROM (
          SELECT sample_end_date, p10, p50, p90
          FROM pm_research.variable_profile
          WHERE scope_level = CASE WHEN :ticker IS NULL THEN 'global_day' ELSE 'stock_day' END
            AND scope_key = CASE WHEN :ticker IS NULL THEN 'ALL' ELSE :ticker END
            AND variable_name = :variable_name
            AND window_label = 'rolling_90d'
          ORDER BY sample_end_date DESC
          LIMIT 90
        ) q
        ORDER BY trade_date_ny ASC
        """,
        {"ticker": ticker, "variable_name": variable_name},
    )
    drift_sql = """
        SELECT scope_level, scope_key, variable_name, psi, ks_pvalue, median_shift, iqr_ratio, drift_flag, recent_end_date
        FROM pm_research.drift_monitor
        WHERE variable_name = :variable_name
          AND (:ticker IS NULL OR scope_key = :ticker OR scope_key = 'ALL')
        ORDER BY created_at_utc DESC
        LIMIT 100
    """
    drift_rows = _fetch_all(drift_sql, {"ticker": ticker, "variable_name": variable_name})
    break_rows = _fetch_all(
        """
        SELECT scope_level, scope_key, variable_name, break_date, before_mean, after_mean, score
        FROM pm_research.break_event
        WHERE variable_name = :variable_name
          AND (:ticker IS NULL OR scope_key = :ticker OR scope_key = 'ALL')
        ORDER BY break_date DESC
        LIMIT 100
        """,
        {"ticker": ticker, "variable_name": variable_name},
    )
    alerts = [
        AnalysisAlert(
            level="warning" if row.get("drift_flag") else "info",
            title=f"{row.get('scope_key')} {row.get('variable_name')}",
            detail=f"PSI={row.get('psi')} median_shift={row.get('median_shift')}",
        )
        for row in drift_rows[:5]
    ]
    return AnalysisDriftResponse(
        coverage=coverage,
        selected_variable=variable_name,
        rolling_statistics=_series_from_rows(rolling_rows, y_key="value", y2_key="value_2"),
        rolling_quantiles=_series_from_rows(quantiles, y_key="value", y2_key="value_2", y3_key="value_3"),
        drift_rows=drift_rows,
        break_rows=break_rows,
        alerts=alerts,
    )


def query_analysis_table(
    *,
    table: str,
    page: int = 1,
    page_size: int = 50,
    ticker: Optional[str] = None,
    variable_name: Optional[str] = None,
) -> AnalysisTableResponse:
    ticker = _normalize_ticker(ticker)
    if variable_name is not None:
        _require_volume_variable(variable_name)
    page = max(page, 1)
    page_size = max(1, min(page_size, 500))
    table_obj, where_sql, params, order_sql = _analysis_table_query_parts(
        table_name=table,
        ticker=ticker,
        variable_name=variable_name,
    )
    full_table = f"{table_obj.schema}.{table_obj.name}"
    query_params: Dict[str, Any] = {**params, "limit": page_size, "offset": (page - 1) * page_size}
    total = _fetch_one(f"SELECT COUNT(*) AS total_rows FROM {full_table} WHERE {where_sql}", params) or {}
    rows = _fetch_all(
        f"SELECT * FROM {full_table} WHERE {where_sql} ORDER BY {order_sql} LIMIT :limit OFFSET :offset",
        query_params,
    )
    return AnalysisTableResponse(
        table=table,
        page=page,
        page_size=page_size,
        total_rows=int(total.get("total_rows") or 0),
        rows=[AnalysisTableRow(values=row) for row in rows],
    )


def list_research_notes() -> ResearchNotesResponse:
    rows = _fetch_all(
        """
        SELECT note_id, title, body, author, tags_json, pinned, created_at_utc, updated_at_utc
        FROM pm_research.research_note
        ORDER BY pinned DESC, updated_at_utc DESC
        """
    )
    notes = [
        ResearchNoteResponse(
            note_id=int(row["note_id"]),
            title=str(row["title"]),
            body=str(row["body"]),
            author=row.get("author"),
            tags=list(row.get("tags_json") or []),
            pinned=bool(row.get("pinned")),
            created_at_utc=str(row.get("created_at_utc")),
            updated_at_utc=str(row.get("updated_at_utc")),
        )
        for row in rows
    ]
    return ResearchNotesResponse(notes=notes)


def create_research_note(payload: ResearchNoteCreateRequest) -> ResearchNoteResponse:
    engine = ensure_analysis_schemas(get_analysis_engine())
    now = datetime.now(timezone.utc)
    record = {
        "title": payload.title.strip(),
        "body": payload.body.strip(),
        "author": payload.author.strip() if payload.author else None,
        "tags_json": [tag.strip() for tag in payload.tags if tag and tag.strip()],
        "pinned": bool(payload.pinned),
        "created_at_utc": now,
        "updated_at_utc": now,
    }
    with engine.begin() as conn:
        result = conn.execute(pm_research_research_note.insert().values(**record).returning(pm_research_research_note))
        row = dict(result.mappings().one())
    return ResearchNoteResponse(
        note_id=int(row["note_id"]),
        title=str(row["title"]),
        body=str(row["body"]),
        author=row.get("author"),
        tags=list(row.get("tags_json") or []),
        pinned=bool(row.get("pinned")),
        created_at_utc=str(row.get("created_at_utc")),
        updated_at_utc=str(row.get("updated_at_utc")),
    )


def export_analysis_table_rows(*, table: str, ticker: Optional[str] = None, variable_name: Optional[str] = None) -> List[Dict[str, Any]]:
    ticker = _normalize_ticker(ticker)
    if variable_name is not None:
        _require_volume_variable(variable_name)
    table_obj, where_sql, params, order_sql = _analysis_table_query_parts(
        table_name=table,
        ticker=ticker,
        variable_name=variable_name,
    )
    full_table = f"{table_obj.schema}.{table_obj.name}"
    return _fetch_all(
        f"SELECT * FROM {full_table} WHERE {where_sql} ORDER BY {order_sql}",
        params,
    )


class AnalysisJob:
    def __init__(self, job_id: str, payload: AnalysisRefreshRequest) -> None:
        self.job_id = job_id
        self.payload = payload
        self.status = "queued"
        self.progress: Optional[AnalysisProgress] = None
        self.result: Optional[AnalysisRefreshResult] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None
        self._stdout_lines: List[str] = []
        self._stderr_lines: List[str] = []
        self._refresh_id: Optional[str] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def to_status(self) -> AnalysisJobStatus:
        return AnalysisJobStatus(
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
        self.progress = AnalysisProgress(
            stage=match.group("stage"),
            current=int(match.group("current")),
            total=int(match.group("total")),
            detail=match.group("detail"),
        )

    def _parse_refresh_id(self, line: str) -> None:
        match = _REFRESH_RE.search(line)
        if match:
            self._refresh_id = match.group("refresh_id")

    def _build_command(self) -> List[str]:
        cmd = [sys.executable, str(SCRIPT_PATH)]
        if self.payload.run_ids:
            for run_id in self.payload.run_ids:
                cmd.extend(["--run-id", run_id])
        if self.payload.skip_trade_backfill:
            cmd.append("--skip-trade-backfill")
        if self.payload.force_full_rebuild:
            cmd.append("--force-full-rebuild")
        if self.payload.notes_author:
            cmd.extend(["--notes-author", self.payload.notes_author])
        return cmd

    def _run(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.status = "running"
        cmd = self._build_command()
        start = time.monotonic()
        try:
            handle = spawn_managed_process(
                cmd,
                job_id=self.job_id,
                service="analysis_refresh",
                env=_load_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            proc = handle.process
            if proc is None or proc.stdout is None or proc.stderr is None:
                raise RuntimeError("Failed to start analysis refresh process.")

            def read_stdout() -> None:
                for line in proc.stdout:
                    self._stdout_lines.append(line)
                    self._update_progress(line)
                    self._parse_refresh_id(line)

            def read_stderr() -> None:
                for line in proc.stderr:
                    self._stderr_lines.append(line)

            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            return_code = proc.wait()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)
            duration_s = round(time.monotonic() - start, 3)
            stdout = "".join(self._stdout_lines)
            stderr = "".join(self._stderr_lines)
            if return_code != 0:
                self.status = "failed"
                self.error = (stderr or stdout or "Analysis refresh failed.").strip()
                return
            self.result = AnalysisRefreshResult(
                ok=True,
                refresh_id=self._refresh_id,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration_s,
                command=cmd,
            )
            self.status = "finished"
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
        finally:
            self.finished_at = datetime.now(timezone.utc)


class AnalysisJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, AnalysisJob] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: AnalysisRefreshRequest) -> str:
        job_id = uuid4().hex
        job = AnalysisJob(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> AnalysisJobStatus:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job.to_status()

    def list_jobs(self) -> List[AnalysisJobStatus]:
        with self._lock:
            return [job.to_status() for job in self._jobs.values()]


ANALYSIS_JOB_MANAGER = AnalysisJobManager()


def start_analysis_job(payload: AnalysisRefreshRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    return ANALYSIS_JOB_MANAGER.start_job(payload)


def get_analysis_job(job_id: str) -> AnalysisJobStatus:
    return ANALYSIS_JOB_MANAGER.get_status(job_id)
