from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from app.models.polymarket_quality import (
    PolymarketMarketQuality,
    PolymarketQualityAuditResponse,
    PolymarketQualityBucketCounts,
    PolymarketQualityFlagSummary,
    PolymarketQualityMarketSample,
    PolymarketQualitySummary,
    PolymarketQualityTelemetry,
    PolymarketQualityTickerSummary,
    PolymarketQualityWeekSummary,
)
from app.services.run_csv_files import dedupe_merged_dataframe, get_run_csv_paths

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = BASE_DIR / "src" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from option_chain.exact_builder import DEFAULT_PRN_ASOF_CLOSE_TIME, DEFAULT_PRN_ASOF_TZ  # noqa: E402
from polymarket.quality_flags import (  # noqa: E402
    QUALITY_FLAG_COLUMNS,
    QUALITY_LIVE_PREFIX,
    build_market_quality,
    build_quality_audit_payload,
)

from app.services.polymarket_run_prn import find_run_local_prn_training_file

MARKET_QUALITY_FILENAME = "market_quality.csv"
MARKET_QUALITY_SUMMARY_FILENAME = "market_quality_summary.json"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _quality_summary_path(run_dir: Path) -> Path:
    return run_dir / MARKET_QUALITY_SUMMARY_FILENAME


def _quality_data_path(run_dir: Path) -> Path:
    return run_dir / MARKET_QUALITY_FILENAME


def parse_quality_telemetry_line(line: str) -> Optional[PolymarketQualityTelemetry]:
    raw = line.strip()
    if not raw.startswith(QUALITY_LIVE_PREFIX):
        return None
    payload_text = raw[len(QUALITY_LIVE_PREFIX):].strip()
    if not payload_text:
        return None
    try:
        payload = json.loads(payload_text)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    try:
        return PolymarketQualityTelemetry.model_validate(payload)
    except Exception:
        return None


def quality_summary_from_payload(payload: Optional[Dict[str, Any]]) -> Optional[PolymarketQualitySummary]:
    if not payload:
        return None
    try:
        return PolymarketQualitySummary.model_validate(payload)
    except Exception:
        return None


def _ensure_quality_artifacts(run_dir: Path) -> None:
    quality_path = _quality_data_path(run_dir)
    summary_path = _quality_summary_path(run_dir)
    if quality_path.exists() and summary_path.exists():
        return

    weekly_paths = get_run_csv_paths(run_dir, "weekly_markets.csv")
    training_path = find_run_local_prn_training_file(run_dir)
    if not weekly_paths or training_path is None or not training_path.exists():
        return

    weekly = dedupe_merged_dataframe(
        pd.concat([pd.read_csv(path) for path in weekly_paths], ignore_index=True, sort=False),
        "weekly_markets.csv",
    )
    prn_df = pd.read_csv(training_path)
    result = build_market_quality(
        run_dir,
        weekly,
        prn_df,
        tz_name=DEFAULT_PRN_ASOF_TZ,
        close_time=DEFAULT_PRN_ASOF_CLOSE_TIME,
        emit_live=False,
    )
    if not result.rows.empty:
        result.rows.to_csv(quality_path, index=False)
    else:
        pd.DataFrame(columns=[]).to_csv(quality_path, index=False)
    _write_json(summary_path, result.summary)

    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest:
        manifest["quality_summary"] = result.summary
        if "artifacts" in manifest and isinstance(manifest["artifacts"], dict):
            manifest["artifacts"][MARKET_QUALITY_FILENAME] = {"size_bytes": quality_path.stat().st_size}
            manifest["artifacts"][MARKET_QUALITY_SUMMARY_FILENAME] = {"size_bytes": summary_path.stat().st_size}
        _write_json(manifest_path, manifest)


def load_market_quality_df(run_dir: Path) -> pd.DataFrame:
    _ensure_quality_artifacts(run_dir)
    paths = get_run_csv_paths(run_dir, MARKET_QUALITY_FILENAME)
    if not paths:
        path = _quality_data_path(run_dir)
        if path.exists():
            paths = [path]
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return dedupe_merged_dataframe(pd.concat(frames, ignore_index=True, sort=False), MARKET_QUALITY_FILENAME)


def load_market_quality_summary(run_dir: Path) -> Optional[PolymarketQualitySummary]:
    _ensure_quality_artifacts(run_dir)
    payload = _read_json(_quality_summary_path(run_dir))
    if not payload:
        manifest = _read_json(run_dir / "manifest.json")
        payload = manifest.get("quality_summary") if isinstance(manifest.get("quality_summary"), dict) else {}
    return quality_summary_from_payload(payload)


def market_quality_from_row(row: Dict[str, Any]) -> PolymarketMarketQuality:
    active_flags = [column for column in QUALITY_FLAG_COLUMNS if bool(row.get(column))]
    return PolymarketMarketQuality(
        market_id=_safe_str(row.get("market_id")),
        ticker=_safe_str(row.get("ticker")),
        threshold=_safe_float(row.get("threshold")),
        week_friday=_safe_str(row.get("week_friday")),
        quality_issue_count=_safe_float(row.get("quality_issue_count")),
        quality_bucket=_safe_str(row.get("quality_bucket")),
        active_flags=active_flags,
        snapshot_date_used=_safe_str(row.get("snapshot_date_used")),
        snapshot_time_used=_safe_str(row.get("snapshot_time_used")),
        snapshot_coverage_status=_safe_str(row.get("snapshot_coverage_status")),
        snapshot_drop_reason=_safe_str(row.get("snapshot_drop_reason")),
        snapshot_pRN=_safe_float(row.get("snapshot_pRN")),
        snapshot_abs_log_m_fwd=_safe_float(row.get("snapshot_abs_log_m_fwd")),
        yes_points=_safe_int(row.get("yes_points")),
        stale_ratio=_safe_float(row.get("stale_ratio")),
        max_stale_hours=_safe_float(row.get("max_stale_hours")),
        midprice_cluster_ratio=_safe_float(row.get("midprice_cluster_ratio")),
        max_jump=_safe_float(row.get("max_jump")),
        hours_since_last_yes_trade=_safe_float(row.get("hours_since_last_yes_trade")),
        gamma_volume=_safe_float(row.get("gamma_volume")),
        flag_not_relevant=_safe_bool(row.get("flag_not_relevant")),
        flag_prn_missing=_safe_bool(row.get("flag_prn_missing")),
        flag_pm_no_trade_history=_safe_bool(row.get("flag_pm_no_trade_history")),
        flag_pm_no_recent_trade=_safe_bool(row.get("flag_pm_no_recent_trade")),
        flag_pm_stale_prices=_safe_bool(row.get("flag_pm_stale_prices")),
        flag_extreme_otm=_safe_bool(row.get("flag_extreme_otm")),
    )


def load_market_quality_map(run_dir: Path, *, key_column: str = "market_id") -> Dict[str, PolymarketMarketQuality]:
    df = load_market_quality_df(run_dir)
    if df.empty or key_column not in df.columns:
        return {}
    mapping: Dict[str, PolymarketMarketQuality] = {}
    for row in df.to_dict("records"):
        key = str(row.get(key_column) or "").strip()
        if not key:
            continue
        mapping[key] = market_quality_from_row(row)
    return mapping


def build_quality_audit_response(run_dir: Path) -> PolymarketQualityAuditResponse:
    df = load_market_quality_df(run_dir)
    if df.empty:
        return PolymarketQualityAuditResponse(
            run_id=run_dir.name,
            available=False,
            message="Quality audit unavailable for this legacy run.",
        )
    payload = build_quality_audit_payload(df)
    summary = quality_summary_from_payload(payload.get("summary")) or PolymarketQualitySummary()
    return PolymarketQualityAuditResponse(
        run_id=run_dir.name,
        summary=summary,
        flag_distribution=[
            PolymarketQualityFlagSummary.model_validate(item)
            for item in payload.get("flag_distribution", [])
        ],
        problem_tickers=[
            PolymarketQualityTickerSummary.model_validate(item)
            for item in payload.get("problem_tickers", [])
        ],
        problem_markets=[
            PolymarketQualityMarketSample.model_validate(item)
            for item in payload.get("problem_markets", [])
        ],
        weekly_summary=[
            PolymarketQualityWeekSummary.model_validate(item)
            for item in payload.get("weekly_summary", [])
        ],
        available_quality_flags=list(payload.get("available_quality_flags", [])),
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not pd.notna(parsed):
        return None
    return float(parsed)


def _safe_int(value: Any) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _safe_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    return text in {"1", "true", "t", "yes", "y"}


def _safe_str(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text
