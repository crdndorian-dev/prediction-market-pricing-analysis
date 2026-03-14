from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.services.run_csv_files import dedupe_merged_dataframe, get_run_csv_paths

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = BASE_DIR / "src" / "scripts"
WEEKLY_HISTORY_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket" / "weekly_history"
WEEKLY_HISTORY_RUNS_DIR = WEEKLY_HISTORY_DIR / "runs"
LATEST_POINTER_PATH = WEEKLY_HISTORY_DIR / "latest.json"

if str(SCRIPTS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPTS_DIR))

from option_chain.exact_builder import (  # noqa: E402
    DEFAULT_PRN_ASOF_CLOSE_TIME,
    DEFAULT_PRN_ASOF_TZ,
    DEFAULT_PRN_VERSION,
    DEFAULT_THREADS,
    Config as ExactPrnConfig,
    build_polymarket_exact_prn,
)
from polymarket.quality_flags import (  # noqa: E402
    add_prn_quality_columns,
    build_market_quality,
)
from polymarket.prn_loader import find_latest_prn_dataset  # noqa: E402

PRN_DATASET_DIRNAME = "prn_dataset"
RUN_LOCAL_PRN_META_NAME = "polymarket_run_prn_meta.json"
BUILD_META_NAME = "dataset_build_meta.json"
TRAINING_FILE_TEMPLATE = "training-{run_id}-prn.csv"
MARKET_QUALITY_FILENAME = "market_quality.csv"
MARKET_QUALITY_SUMMARY_FILENAME = "market_quality_summary.json"
REQUIRED_WEEKLY_MARKETS_COLUMNS = ("market_id", "event_id", "ticker", "threshold", "week_monday", "week_friday", "event_endDate")


@dataclass
class RunLocalPrnRefreshResult:
    run_dir: Path
    training_path: Path
    market_quality_path: Optional[Path] = None
    market_quality_summary_path: Optional[Path] = None
    quality_summary: Dict[str, Any] = field(default_factory=dict)
    source_training_path: Optional[Path] = None
    seeded_from_source: bool = False
    used_inferred_defaults: bool = False
    required_pairs: Set[Tuple[str, date]] = field(default_factory=set)
    existing_pairs_before: Set[Tuple[str, date]] = field(default_factory=set)
    missing_pairs_before: Set[Tuple[str, date]] = field(default_factory=set)
    missing_pairs_after: Set[Tuple[str, date]] = field(default_factory=set)
    existing_markets_pairs_before: Set[Tuple[str, date]] = field(default_factory=set)
    missing_markets_pairs_before: Set[Tuple[str, date]] = field(default_factory=set)
    required_week_fridays: List[date] = field(default_factory=list)
    affected_week_fridays: List[date] = field(default_factory=list)
    temp_run_dirs: List[Path] = field(default_factory=list)
    required_market_snapshots: int = 0
    ok_market_snapshots: int = 0
    missing_market_snapshots: int = 0
    coverage_counts: Dict[str, int] = field(default_factory=dict)
    drop_reason_counts: Dict[str, int] = field(default_factory=dict)
    prn_version: str = DEFAULT_PRN_VERSION
    prn_config_hash: str = ""


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path.resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _read_latest_run_id() -> Optional[str]:
    if not LATEST_POINTER_PATH.exists():
        return None
    try:
        payload = json.loads(LATEST_POINTER_PATH.read_text())
    except Exception:
        return None
    return payload.get("run_id") if isinstance(payload, dict) else None


def resolve_polymarket_run_dir(run_id: Optional[str]) -> Path:
    if run_id:
        run_dir = WEEKLY_HISTORY_RUNS_DIR / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_id}")
        return run_dir

    latest_id = _read_latest_run_id()
    if latest_id:
        candidate = WEEKLY_HISTORY_RUNS_DIR / latest_id
        if candidate.exists():
            return candidate

    run_dirs = sorted([d for d in WEEKLY_HISTORY_RUNS_DIR.iterdir() if d.is_dir()], reverse=True)
    if not run_dirs:
        raise FileNotFoundError("No weekly history runs found.")
    return run_dirs[0]


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_prn_dir(run_dir: Path) -> Path:
    return run_dir / PRN_DATASET_DIRNAME


def _run_prn_meta_path(run_dir: Path) -> Path:
    return _run_prn_dir(run_dir) / RUN_LOCAL_PRN_META_NAME


def _run_local_training_path(run_dir: Path) -> Path:
    return _run_prn_dir(run_dir) / TRAINING_FILE_TEMPLATE.format(run_id=run_dir.name)


def find_run_local_prn_training_file(run_dir: Path) -> Optional[Path]:
    prn_dir = _run_prn_dir(run_dir)
    if not prn_dir.exists():
        return None
    target = _run_local_training_path(run_dir)
    if target.exists():
        return target
    candidates = sorted(prn_dir.glob("training-*.csv"))
    return candidates[0] if candidates else None


def has_run_local_markets_artifact(run_dir: Path) -> bool:
    return bool(get_run_csv_paths(run_dir, "markets_prn_hourly.csv"))


def resolve_preferred_prn_dataset_path(
    run_dir: Path,
    explicit_prn_dataset: Optional[str | Path] = None,
) -> Optional[Path]:
    local_training = find_run_local_prn_training_file(run_dir)
    if local_training and local_training.exists():
        return local_training

    if explicit_prn_dataset:
        path = _resolve_project_path(explicit_prn_dataset)
        return path if path.exists() else None

    manifest = _load_manifest(run_dir)
    pipeline_args = manifest.get("pipeline_args")
    if isinstance(pipeline_args, dict):
        prn_value = pipeline_args.get("prn_dataset")
        if isinstance(prn_value, str) and prn_value.strip():
            path = _resolve_project_path(prn_value)
            if path.exists():
                return path

    latest = find_latest_prn_dataset()
    return latest if latest and latest.exists() else None


def _load_required_pairs(
    run_dir: Path,
    week_filter: Optional[Set[date]] = None,
) -> Tuple[Set[Tuple[str, date]], List[date], pd.DataFrame]:
    weekly_paths = get_run_csv_paths(run_dir, "weekly_markets.csv")
    if not weekly_paths:
        raise FileNotFoundError(f"weekly_markets.csv not found in {run_dir.name}")

    frames: List[pd.DataFrame] = []
    for path in weekly_paths:
        frame = pd.read_csv(path)
        frames.append(frame)

    df = dedupe_merged_dataframe(pd.concat(frames, ignore_index=True, sort=False), "weekly_markets.csv")
    missing_cols = [col for col in REQUIRED_WEEKLY_MARKETS_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"weekly_markets.csv is missing columns: {missing_cols}")

    df = df[list(REQUIRED_WEEKLY_MARKETS_COLUMNS)].copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["week_friday"] = pd.to_datetime(df["week_friday"], errors="coerce").dt.date
    df["week_monday"] = pd.to_datetime(df["week_monday"], errors="coerce").dt.date
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce").round(6)
    df = df.dropna(subset=["market_id", "ticker", "week_friday", "week_monday", "threshold"])
    if week_filter:
        df = df[df["week_friday"].isin(week_filter)]
    df = df.sort_values(["ticker", "week_friday", "threshold", "market_id"]).reset_index(drop=True)

    pairs = {
        (str(row.ticker).upper(), row.week_friday)
        for row in df.itertuples(index=False)
        if row.ticker and row.week_friday
    }
    weeks = sorted({week for _, week in pairs})
    return pairs, weeks, df


def _load_existing_pairs(
    training_path: Path,
    required_pairs: Set[Tuple[str, date]],
) -> Set[Tuple[str, date]]:
    if not training_path.exists() or not required_pairs:
        return set()

    required_tickers = {ticker for ticker, _ in required_pairs}
    required_weeks = {week for _, week in required_pairs}
    existing: Set[Tuple[str, date]] = set()

    for chunk in pd.read_csv(training_path, chunksize=100_000):
        if "ticker" not in chunk.columns or "week_friday" not in chunk.columns:
            continue
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper()
        chunk = chunk[chunk["ticker"].isin(required_tickers)]
        if chunk.empty:
            continue
        chunk["week_friday"] = pd.to_datetime(chunk["week_friday"], errors="coerce").dt.date
        chunk = chunk.dropna(subset=["week_friday"])
        chunk = chunk[chunk["week_friday"].isin(required_weeks)]
        if chunk.empty:
            continue
        if "coverage_status" in chunk.columns:
            chunk = chunk[chunk["coverage_status"].astype(str).str.lower() == "ok"]
        elif "pRN" in chunk.columns:
            chunk["pRN"] = pd.to_numeric(chunk["pRN"], errors="coerce")
            chunk = chunk[chunk["pRN"].notna()]
        if chunk.empty:
            continue
        existing.update(
            (str(row.ticker).upper(), row.week_friday)
            for row in chunk.itertuples(index=False)
        )
    return existing


def _load_existing_markets_pairs(
    run_dir: Path,
    required_pairs: Set[Tuple[str, date]],
) -> Set[Tuple[str, date]]:
    prn_paths = get_run_csv_paths(run_dir, "markets_prn_hourly.csv")
    if not prn_paths or not required_pairs:
        return set()

    required_tickers = {ticker for ticker, _ in required_pairs}
    required_weeks = {week for _, week in required_pairs}
    existing: Set[Tuple[str, date]] = set()

    for path in prn_paths:
        for chunk in pd.read_csv(path, chunksize=100_000):
            if "ticker" not in chunk.columns or "week_friday" not in chunk.columns:
                continue
            chunk["ticker"] = chunk["ticker"].astype(str).str.upper()
            chunk = chunk[chunk["ticker"].isin(required_tickers)]
            if chunk.empty:
                continue
            chunk["week_friday"] = pd.to_datetime(chunk["week_friday"], errors="coerce").dt.date
            chunk = chunk.dropna(subset=["week_friday"])
            chunk = chunk[chunk["week_friday"].isin(required_weeks)]
            if chunk.empty:
                continue
            if "pRN" in chunk.columns:
                chunk["pRN"] = pd.to_numeric(chunk["pRN"], errors="coerce")
                chunk = chunk[chunk["pRN"].notna()]
                if chunk.empty:
                    continue
            existing.update(
                (str(row.ticker).upper(), row.week_friday)
                for row in chunk.itertuples(index=False)
            )

    return existing


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _resolve_exact_builder_config(run_dir: Path) -> Tuple[ExactPrnConfig, Dict[str, Any]]:
    manifest = _load_manifest(run_dir)
    pipeline_args = manifest.get("pipeline_args") if isinstance(manifest.get("pipeline_args"), dict) else {}

    config = ExactPrnConfig()
    payload = {
        "source_mode": "theta_only",
        "tickers": pipeline_args.get("tickers"),
        "start": manifest.get("start_date") or pipeline_args.get("start_date"),
        "end": manifest.get("end_date") or pipeline_args.get("end_date"),
        "schedule_mode": "polymarket_exact_weekly",
        "expiry_weekdays": "fri",
        "asof_weekdays": "mon,tue,wed,thu",
        "threads": DEFAULT_THREADS,
        "prn_version": DEFAULT_PRN_VERSION,
        "prn_asof_tz": DEFAULT_PRN_ASOF_TZ,
        "prn_asof_close_time": DEFAULT_PRN_ASOF_CLOSE_TIME,
        "config": {
            "theta_base_url": config.theta_base_url,
            "risk_free_rate": config.risk_free_rate,
            "option_strike_range": config.option_strike_range,
            "retry_full_chain_if_band_thin": config.retry_full_chain_if_band_thin,
            "try_saturday_expiry_fallback": config.try_saturday_expiry_fallback,
            "max_abs_logm": config.max_abs_logm,
            "max_abs_logm_cap": config.max_abs_logm_cap,
            "band_widen_step": config.band_widen_step,
            "adaptive_band": config.adaptive_band,
            "max_band_strikes": config.max_band_strikes,
            "min_strikes_for_curve": config.min_strikes_for_curve,
            "min_strikes_in_prn_band": config.min_strikes_in_prn_band,
            "prefer_bidask": config.prefer_bidask,
            "stock_source": config.stock_source,
            "dividend_source": config.dividend_source,
            "dividend_lookback_days": config.dividend_lookback_days,
            "use_forward_moneyness": config.use_forward_moneyness,
            "use_cache": config.use_cache,
            "rv_lookback_days": config.rv_lookback_days,
        },
    }
    return config, payload


def _write_run_local_meta(
    run_dir: Path,
    *,
    builder_payload: Dict[str, Any],
    result: RunLocalPrnRefreshResult,
) -> None:
    prn_dir = _run_prn_dir(run_dir)
    prn_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "required_pairs": len(result.required_pairs),
        "existing_pairs_before": len(result.existing_pairs_before),
        "missing_pairs_before": len(result.missing_pairs_before),
        "missing_pairs_after": len(result.missing_pairs_after),
        "required_market_snapshots": result.required_market_snapshots,
        "ok_market_snapshots": result.ok_market_snapshots,
        "missing_market_snapshots": result.missing_market_snapshots,
        "coverage_counts": result.coverage_counts,
        "drop_reason_counts": result.drop_reason_counts,
        "quality_summary": result.quality_summary,
    }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_mode": "theta_only",
        "payload": builder_payload,
        "training_file": _run_local_training_path(run_dir).name,
        "market_quality_file": result.market_quality_path.name if result.market_quality_path else None,
        "market_quality_summary_file": (
            result.market_quality_summary_path.name if result.market_quality_summary_path else None
        ),
        "prn_version": result.prn_version,
        "prn_config_hash": result.prn_config_hash,
        "summary": summary,
    }
    _write_json(_run_prn_meta_path(run_dir), payload)

    build_meta_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [],
        "source_mode": "theta_only",
        "payload": builder_payload,
        "out_name": _run_local_training_path(run_dir).name,
        "drops_name": "",
        "market_quality_name": result.market_quality_path.name if result.market_quality_path else None,
        "prn_version": result.prn_version,
        "prn_config_hash": result.prn_config_hash,
        "summary": summary,
    }
    _write_json(_run_prn_dir(run_dir) / BUILD_META_NAME, build_meta_payload)


def _write_market_quality_artifacts(
    run_dir: Path,
    *,
    weekly_markets: pd.DataFrame,
    training_rows: pd.DataFrame,
) -> tuple[Path, Path, Dict[str, Any]]:
    quality_result = build_market_quality(
        run_dir,
        weekly_markets,
        training_rows,
        tz_name=DEFAULT_PRN_ASOF_TZ,
        close_time=DEFAULT_PRN_ASOF_CLOSE_TIME,
        emit_live=True,
    )
    quality_path = run_dir / MARKET_QUALITY_FILENAME
    summary_path = run_dir / MARKET_QUALITY_SUMMARY_FILENAME
    quality_result.rows.to_csv(quality_path, index=False)
    _write_json(summary_path, quality_result.summary)

    manifest_path = run_dir / "manifest.json"
    manifest = _load_manifest(run_dir)
    manifest["quality_summary"] = quality_result.summary
    manifest["artifacts"] = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    manifest["artifacts"][quality_path.name] = {"size_bytes": quality_path.stat().st_size}
    manifest["artifacts"][summary_path.name] = {"size_bytes": summary_path.stat().st_size}
    if manifest:
        _write_json(manifest_path, manifest)

    return quality_path, summary_path, quality_result.summary


def refresh_run_local_prn_dataset(
    run_dir: Path,
    *,
    week_fridays: Optional[Set[date]] = None,
    explicit_prn_dataset: Optional[str | Path] = None,
) -> RunLocalPrnRefreshResult:
    required_pairs, required_weeks, weekly_markets = _load_required_pairs(run_dir, week_filter=week_fridays)
    if not required_pairs or not required_weeks:
        raise ValueError(f"No weekly markets found to backfill for run {run_dir.name}.")

    local_training = _run_local_training_path(run_dir)
    existing_pairs_before = _load_existing_pairs(local_training, required_pairs)
    missing_pairs_before = required_pairs - existing_pairs_before
    existing_markets_pairs_before = _load_existing_markets_pairs(run_dir, required_pairs)
    missing_markets_pairs_before = required_pairs - existing_markets_pairs_before

    cfg, builder_payload = _resolve_exact_builder_config(run_dir)
    build_result = build_polymarket_exact_prn(
        weekly_markets,
        cfg=cfg,
        threads=DEFAULT_THREADS,
        prn_version=DEFAULT_PRN_VERSION,
        prn_asof_tz=DEFAULT_PRN_ASOF_TZ,
        prn_asof_close_time=DEFAULT_PRN_ASOF_CLOSE_TIME,
    )
    if build_result.rows.empty:
        raise RuntimeError(f"Exact run-local pRN build produced no rows for run {run_dir.name}.")
    if build_result.ok_market_snapshots <= 0:
        raise RuntimeError(
            f"Exact run-local pRN build produced zero covered market snapshots for run {run_dir.name}."
        )

    training_rows = add_prn_quality_columns(build_result.rows)
    local_training.parent.mkdir(parents=True, exist_ok=True)
    training_rows.to_csv(local_training, index=False)

    quality_path, quality_summary_path, quality_summary = _write_market_quality_artifacts(
        run_dir,
        weekly_markets=weekly_markets,
        training_rows=training_rows,
    )

    existing_pairs_after = _load_existing_pairs(local_training, required_pairs)
    missing_pairs_after = required_pairs - existing_pairs_after
    affected_week_fridays = sorted({week for _, week in required_pairs})

    result = RunLocalPrnRefreshResult(
        run_dir=run_dir,
        training_path=local_training,
        market_quality_path=quality_path,
        market_quality_summary_path=quality_summary_path,
        quality_summary=quality_summary,
        source_training_path=None,
        seeded_from_source=False,
        used_inferred_defaults=False,
        required_pairs=required_pairs,
        existing_pairs_before=existing_pairs_before,
        missing_pairs_before=missing_pairs_before,
        missing_pairs_after=missing_pairs_after,
        existing_markets_pairs_before=existing_markets_pairs_before,
        missing_markets_pairs_before=missing_markets_pairs_before,
        required_week_fridays=required_weeks,
        affected_week_fridays=affected_week_fridays,
        temp_run_dirs=[],
        required_market_snapshots=build_result.required_market_snapshots,
        ok_market_snapshots=build_result.ok_market_snapshots,
        missing_market_snapshots=build_result.missing_market_snapshots,
        coverage_counts=build_result.coverage_counts,
        drop_reason_counts=build_result.drop_reason_counts,
        prn_version=build_result.prn_version,
        prn_config_hash=build_result.prn_config_hash,
    )
    _write_run_local_meta(run_dir, builder_payload=builder_payload, result=result)
    return result
