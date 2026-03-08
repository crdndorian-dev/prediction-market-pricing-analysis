from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from app.models.datasets import DatasetRunRequest
from app.services.datasets import (
    BUILD_META_NAME,
    _build_dataset_command,
    _extract_run_dir_from_output,
    _find_training_file_path,
    _read_build_meta,
)
from app.services.run_csv_files import dedupe_merged_dataframe, get_run_csv_paths

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = BASE_DIR / "src" / "scripts"
BACKEND_DIR = BASE_DIR / "src" / "webapp" / "backend"
WEEKLY_HISTORY_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket" / "weekly_history"
WEEKLY_HISTORY_RUNS_DIR = WEEKLY_HISTORY_DIR / "runs"
LATEST_POINTER_PATH = WEEKLY_HISTORY_DIR / "latest.json"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from polymarket.prn_loader import find_latest_prn_dataset

PRN_DATASET_DIRNAME = "prn_dataset"
PRN_EXPIRY_COLUMNS = (
    "expiry_close_date_used",
    "option_expiration_used",
    "option_expiration_requested",
    "expiry_date",
)
RUN_LOCAL_PRN_META_NAME = "polymarket_run_prn_meta.json"
TRAINING_FILE_TEMPLATE = "training-{run_id}-prn.csv"
DEFAULT_PRN_VERSION = "v1"
DEFAULT_BUILDER_TIMEOUT_S = 30

_BUILDER_DEFAULT_HASH_CACHE: Optional[str] = None


@dataclass
class RunLocalPrnRefreshResult:
    run_dir: Path
    training_path: Path
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


def _model_dump(model: DatasetRunRequest) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(exclude_none=True)
    return model.dict(exclude_none=True)


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
    return _find_training_file_path(prn_dir)


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


def _read_uniform_prn_metadata(training_path: Path) -> Tuple[Optional[str], Optional[str]]:
    versions: Set[str] = set()
    hashes: Set[str] = set()

    for chunk in pd.read_csv(
        training_path,
        usecols=["prn_version", "prn_config_hash"],
        chunksize=100_000,
    ):
        versions.update(str(v).strip() for v in chunk["prn_version"].dropna().tolist() if str(v).strip())
        hashes.update(str(v).strip() for v in chunk["prn_config_hash"].dropna().tolist() if str(v).strip())
        if len(versions) > 1 or len(hashes) > 1:
            break

    version = next(iter(versions)) if len(versions) == 1 else None
    config_hash = next(iter(hashes)) if len(hashes) == 1 else None
    return version, config_hash


def _load_builder_default_prn_hash() -> str:
    global _BUILDER_DEFAULT_HASH_CACHE
    if _BUILDER_DEFAULT_HASH_CACHE is not None:
        return _BUILDER_DEFAULT_HASH_CACHE

    script_path = SCRIPTS_DIR / "01-option-chain-build-historic-dataset-v1.0.py"
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))

    spec = importlib.util.spec_from_file_location("option_chain_builder_default_hash", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load option-chain builder script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _BUILDER_DEFAULT_HASH_CACHE = str(module.compute_prn_config_hash(module.Config()))
    return _BUILDER_DEFAULT_HASH_CACHE


def _resolve_base_payload(
    training_path: Path,
    *,
    required_start: date,
    required_end: date,
    required_tickers: Set[str],
) -> Tuple[DatasetRunRequest, bool]:
    meta = _read_build_meta(training_path.parent)
    if meta and isinstance(meta.get("payload"), dict):
        payload = DatasetRunRequest(**meta["payload"])
        return payload, False

    version, config_hash = _read_uniform_prn_metadata(training_path)
    default_hash = _load_builder_default_prn_hash()
    if config_hash != default_hash:
        raise ValueError(
            "Source pRN dataset metadata is missing and its prn_config_hash does not match "
            "the builder defaults. Exact run-local backfill is blocked."
        )

    payload = DatasetRunRequest(
        tickers=",".join(sorted(required_tickers)) if required_tickers else None,
        start=required_start.isoformat(),
        end=required_end.isoformat(),
        prn_version=version or DEFAULT_PRN_VERSION,
        prn_config_hash=config_hash,
    )
    return payload, True


def _write_run_local_meta(
    run_dir: Path,
    *,
    source_training_path: Path,
    base_payload: DatasetRunRequest,
    used_inferred_defaults: bool,
) -> None:
    prn_dir = _run_prn_dir(run_dir)
    prn_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "payload": _model_dump(base_payload),
        "source_training_file": _display_path(source_training_path),
        "used_inferred_defaults": used_inferred_defaults,
        "training_file": _run_local_training_path(run_dir).name,
    }
    _run_prn_meta_path(run_dir).write_text(json.dumps(payload, indent=2, sort_keys=True))

    build_meta_payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [],
        "payload": _model_dump(base_payload),
        "out_name": _run_local_training_path(run_dir).name,
        "drops_name": "",
        "source_training_file": _display_path(source_training_path),
        "used_inferred_defaults": used_inferred_defaults,
    }
    (_run_prn_dir(run_dir) / BUILD_META_NAME).write_text(
        json.dumps(build_meta_payload, indent=2, sort_keys=True)
    )


def _load_required_pairs(
    run_dir: Path,
    week_filter: Optional[Set[date]] = None,
) -> Tuple[Set[Tuple[str, date]], List[date]]:
    weekly_paths = get_run_csv_paths(run_dir, "weekly_markets.csv")
    if not weekly_paths:
        raise FileNotFoundError(f"weekly_markets.csv not found in {run_dir.name}")

    frames: List[pd.DataFrame] = []
    for path in weekly_paths:
        try:
            frame = pd.read_csv(path, usecols=["ticker", "week_friday"])
        except ValueError:
            frame = pd.read_csv(path)
        frames.append(frame)

    df = dedupe_merged_dataframe(
        pd.concat(frames, ignore_index=True, sort=False),
        "weekly_markets.csv",
    )
    if "ticker" not in df.columns or "week_friday" not in df.columns:
        raise ValueError("weekly_markets.csv is missing ticker/week_friday columns.")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    week_series = pd.to_datetime(df["week_friday"], errors="coerce").dt.date
    df = df.assign(week_friday=week_series).dropna(subset=["ticker", "week_friday"])
    if week_filter:
        df = df[df["week_friday"].isin(week_filter)]

    pairs = {
        (str(row.ticker).upper(), row.week_friday)
        for row in df.itertuples(index=False)
        if row.ticker and row.week_friday
    }
    weeks = sorted({week for _, week in pairs})
    return pairs, weeks


def _resolve_expiry_column(path: Path) -> str:
    header = pd.read_csv(path, nrows=0)
    cols = set(header.columns.tolist())
    for candidate in PRN_EXPIRY_COLUMNS:
        if candidate in cols:
            return candidate
    raise KeyError(f"Training dataset missing expiry column: {path}")


def _load_existing_pairs(
    training_path: Path,
    required_pairs: Set[Tuple[str, date]],
) -> Set[Tuple[str, date]]:
    if not training_path.exists():
        return set()
    if not required_pairs:
        return set()

    expiry_col = _resolve_expiry_column(training_path)
    required_tickers = {ticker for ticker, _ in required_pairs}
    required_weeks = {week for _, week in required_pairs}
    existing: Set[Tuple[str, date]] = set()

    for chunk in pd.read_csv(
        training_path,
        usecols=["ticker", expiry_col],
        chunksize=100_000,
    ):
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper()
        chunk = chunk[chunk["ticker"].isin(required_tickers)]
        if chunk.empty:
            continue
        chunk["expiry_date"] = pd.to_datetime(chunk[expiry_col], errors="coerce").dt.date
        chunk = chunk.dropna(subset=["expiry_date"])
        chunk = chunk[chunk["expiry_date"].isin(required_weeks)]
        if chunk.empty:
            continue
        existing.update(
            (str(row.ticker).upper(), row.expiry_date)
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


def _stream_filter_source_training(
    source_training: Path,
    local_training: Path,
    required_pairs: Set[Tuple[str, date]],
) -> None:
    local_training.parent.mkdir(parents=True, exist_ok=True)
    expiry_col = _resolve_expiry_column(source_training)
    tmp_path = local_training.with_suffix(".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    wrote_any = False
    for chunk in pd.read_csv(source_training, chunksize=100_000):
        if "ticker" not in chunk.columns or expiry_col not in chunk.columns:
            continue
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper()
        chunk["__expiry_date__"] = pd.to_datetime(chunk[expiry_col], errors="coerce").dt.date
        mask = [
            (ticker, expiry_date) in required_pairs
            for ticker, expiry_date in zip(chunk["ticker"], chunk["__expiry_date__"])
        ]
        filtered = chunk.loc[mask].drop(columns=["__expiry_date__"])
        if filtered.empty:
            continue
        filtered.to_csv(tmp_path, mode="a", header=not wrote_any, index=False)
        wrote_any = True

    if not wrote_any:
        header = pd.read_csv(source_training, nrows=0)
        header.iloc[0:0].to_csv(tmp_path, index=False)

    os.replace(tmp_path, local_training)


def _compress_contiguous_fridays(weeks: Iterable[date]) -> List[Tuple[date, date]]:
    sorted_weeks = sorted(set(weeks))
    if not sorted_weeks:
        return []

    ranges: List[Tuple[date, date]] = []
    start = sorted_weeks[0]
    end = sorted_weeks[0]
    for current in sorted_weeks[1:]:
        if current == end + timedelta(days=7):
            end = current
            continue
        ranges.append((start, end))
        start = end = current
    ranges.append((start, end))
    return ranges


def _backfill_jobs(
    missing_pairs: Set[Tuple[str, date]],
) -> List[Tuple[List[str], date, date]]:
    by_ticker: Dict[str, List[date]] = {}
    for ticker, week in missing_pairs:
        by_ticker.setdefault(ticker, []).append(week)

    jobs: List[Tuple[List[str], date, date]] = []
    for ticker, weeks in sorted(by_ticker.items()):
        for start, end in _compress_contiguous_fridays(weeks):
            jobs.append(([ticker], start, end))
    return jobs


def _load_local_base_payload(
    run_dir: Path,
    training_path: Path,
    *,
    required_start: date,
    required_end: date,
    required_tickers: Set[str],
) -> Tuple[DatasetRunRequest, bool]:
    prn_dir = _run_prn_dir(run_dir)
    meta = _read_build_meta(prn_dir)
    if meta and isinstance(meta.get("payload"), dict):
        return DatasetRunRequest(**meta["payload"]), bool(meta.get("used_inferred_defaults"))

    run_meta_path = _run_prn_meta_path(run_dir)
    if run_meta_path.exists():
        payload = json.loads(run_meta_path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("payload"), dict):
            return DatasetRunRequest(**payload["payload"]), bool(payload.get("used_inferred_defaults"))

    source_path = resolve_preferred_prn_dataset_path(run_dir)
    if source_path is None:
        raise FileNotFoundError(f"No pRN dataset available for run {run_dir.name}.")
    base_payload, used_defaults = _resolve_base_payload(
        source_path,
        required_start=required_start,
        required_end=required_end,
        required_tickers=required_tickers,
    )
    _write_run_local_meta(
        run_dir,
        source_training_path=source_path,
        base_payload=base_payload,
        used_inferred_defaults=used_defaults,
    )
    return base_payload, used_defaults


def _merge_training_files(target_path: Path, incoming_path: Path) -> None:
    target_df = pd.read_csv(target_path)
    incoming_df = pd.read_csv(incoming_path)
    combined = pd.concat([target_df, incoming_df], ignore_index=True, sort=False)
    if "row_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["row_id"], keep="first")
    combined.to_csv(target_path, index=False)


def refresh_run_local_prn_dataset(
    run_dir: Path,
    *,
    week_fridays: Optional[Set[date]] = None,
    explicit_prn_dataset: Optional[str | Path] = None,
) -> RunLocalPrnRefreshResult:
    required_pairs, required_weeks = _load_required_pairs(run_dir, week_filter=week_fridays)
    if not required_pairs or not required_weeks:
        raise ValueError(f"No weekly markets found to backfill for run {run_dir.name}.")

    required_tickers = {ticker for ticker, _ in required_pairs}
    required_start = min(required_weeks)
    required_end = max(required_weeks)

    local_training = find_run_local_prn_training_file(run_dir)
    source_training: Optional[Path] = None
    seeded_from_source = False
    used_inferred_defaults = False

    if local_training is None or not local_training.exists():
        source_training = resolve_preferred_prn_dataset_path(run_dir, explicit_prn_dataset)
        if source_training is None:
            raise FileNotFoundError(f"No source pRN dataset found for run {run_dir.name}.")
        base_payload, used_inferred_defaults = _resolve_base_payload(
            source_training,
            required_start=required_start,
            required_end=required_end,
            required_tickers=required_tickers,
        )
        local_training = _run_local_training_path(run_dir)
        _stream_filter_source_training(source_training, local_training, required_pairs)
        _write_run_local_meta(
            run_dir,
            source_training_path=source_training,
            base_payload=base_payload,
            used_inferred_defaults=used_inferred_defaults,
        )
        seeded_from_source = True

    existing_pairs_before = _load_existing_pairs(local_training, required_pairs)
    missing_pairs_before = required_pairs - existing_pairs_before
    existing_markets_pairs_before = _load_existing_markets_pairs(run_dir, required_pairs)
    missing_markets_pairs_before = required_pairs - existing_markets_pairs_before
    affected_week_fridays = sorted({week for _, week in missing_pairs_before})
    temp_run_dirs: List[Path] = []

    if missing_pairs_before:
        prn_dir = _run_prn_dir(run_dir)
        backfill_out_dir = prn_dir / "_backfills"
        backfill_out_dir.mkdir(parents=True, exist_ok=True)
        base_payload, payload_used_defaults = _load_local_base_payload(
            run_dir,
            local_training,
            required_start=required_start,
            required_end=required_end,
            required_tickers=required_tickers,
        )
        used_inferred_defaults = used_inferred_defaults or payload_used_defaults

        for idx, (tickers, start_week, end_week) in enumerate(_backfill_jobs(missing_pairs_before), start=1):
            payload_data = _model_dump(base_payload)
            payload_data.update(
                {
                    "out_dir": str(backfill_out_dir.relative_to(BASE_DIR)),
                    "dataset_name": f"{run_dir.name}-prn-backfill-{int(time.time())}-{idx}",
                    "run_dir_name": f"{run_dir.name}-prn-backfill-{int(time.time())}-{idx}",
                    "tickers": ",".join(sorted(tickers)),
                    "start": start_week.isoformat(),
                    "end": end_week.isoformat(),
                    "schedule_mode": "expiry_range",
                    "expiry_weekdays": payload_data.get("expiry_weekdays") or "fri",
                    "write_snapshot": False,
                    "write_prn_view": True,
                    "write_train_view": True,
                    "write_legacy": False,
                    "write_drops": False,
                }
            )
            range_payload = DatasetRunRequest(**payload_data)
            cmd, _out_dir, _out_name, _drops_name = _build_dataset_command(range_payload)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            if result.returncode != 0:
                raise RuntimeError(
                    result.stderr or result.stdout or "Exact pRN backfill builder run failed."
                )
            backfill_run_dir = _extract_run_dir_from_output(result.stdout)
            if backfill_run_dir is None or not backfill_run_dir.exists():
                raise RuntimeError("Could not locate exact pRN backfill output directory.")
            temp_run_dirs.append(backfill_run_dir)
            backfill_training = _find_training_file_path(backfill_run_dir)
            if backfill_training is None:
                continue
            _merge_training_files(local_training, backfill_training)

        for tmp_dir in temp_run_dirs:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    existing_pairs_after = _load_existing_pairs(local_training, required_pairs)
    missing_pairs_after = required_pairs - existing_pairs_after

    return RunLocalPrnRefreshResult(
        run_dir=run_dir,
        training_path=local_training,
        source_training_path=source_training,
        seeded_from_source=seeded_from_source,
        used_inferred_defaults=used_inferred_defaults,
        required_pairs=required_pairs,
        existing_pairs_before=existing_pairs_before,
        missing_pairs_before=missing_pairs_before,
        missing_pairs_after=missing_pairs_after,
        existing_markets_pairs_before=existing_markets_pairs_before,
        missing_markets_pairs_before=missing_markets_pairs_before,
        required_week_fridays=required_weeks,
        affected_week_fridays=affected_week_fridays,
        temp_run_dirs=temp_run_dirs,
    )
