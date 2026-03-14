from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.models.polymarket_history import (
    PolymarketHistoryProgress,
    PolymarketHistoryJobStatus,
    PolymarketHistoryRunRequest,
    PolymarketHistoryRunResponse,
    PolymarketRunFeaturesRequest,
    PolymarketRunFeaturesResponse,
)
from app.models.polymarket_quality import PolymarketQualityAuditResponse, PolymarketQualitySummary
from app.services.process_runtime import (
    ManagedProcessHandle,
    clear_runtime_file,
    is_process_alive,
    managed_handle_from_runtime_payload,
    read_runtime_file,
    runtime_payload_for_handle,
    spawn_managed_process,
    terminate_managed_process,
    write_runtime_file,
)
from app.services.polymarket_quality import (
    build_quality_audit_response,
    load_market_quality_summary,
    parse_quality_telemetry_line,
)
from app.services.polymarket_run_prn import find_run_local_prn_training_file
from app.services.run_csv_files import PRESERVED_RUNTIME_CSVS
from app.services.script_entrypoints import (
    POLYMARKET_BUILD_FEATURES_SCRIPT,
    POLYMARKET_RUN_PRN_REFRESH_SCRIPT,
    POLYMARKET_WEEKLY_HISTORY_SCRIPT,
)

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = POLYMARKET_WEEKLY_HISTORY_SCRIPT.path
FEATURES_SCRIPT_PATH = POLYMARKET_BUILD_FEATURES_SCRIPT.path
RUN_LOCAL_PRN_REFRESH_SCRIPT_PATH = POLYMARKET_RUN_PRN_REFRESH_SCRIPT.path
DEFAULT_OUT_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket" / "weekly_history"
RUNS_DIR = DEFAULT_OUT_DIR / "runs"
DEFAULT_EVENT_URLS_FILE = BASE_DIR / "config" / "polymarket_event_urls.csv"
DIM_MARKET_WEEKLY_PATH = BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market_weekly.csv"
DEFAULT_BARS_DIR = BASE_DIR / "src" / "data" / "analysis" / "polymarket" / "bars_history"
ENV_FILE = BASE_DIR / ".env"
ENV_SAMPLE_FILE = BASE_DIR / "config" / "polymarket_subgraph.env.sample"
MAX_RUN_DIR_NAME_LEN = 140
LEGACY_DECISION_FEATURES_CSV = "decision_features.csv"
MASTER_BAR_FREQS = ("1h", "1d")
HISTORY_RESUME_STATE_FILENAME = ".history_resume_state.json"
POLYMARKET_RUNTIME_SERVICES = {"polymarket_history", "polymarket_run_prn", "polymarket_features"}
MAX_AUTOMATIC_RESUME_ATTEMPTS = 3

_HISTORY_COMPLETE_RE = re.compile(
    r"\[Weekly History\] Market complete (?P<current>\d+)/(?P<total>\d+)\s+job_id=(?P<job_id>[^\s]+)\s+status=(?P<status>\w+)"
)
_FEATURE_PROGRESS_RE = re.compile(
    r"\[features\] PROGRESS (?P<current>\d+)/(?P<total>\d+)\s+step=(?P<step>[^\s]+)"
)


class JobProgressTracker:
    def __init__(self) -> None:
        self._total: Optional[int] = None
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._failure_flag = False
        self._lock = threading.Lock()

    def set_total(self, total: int) -> None:
        if total <= 0:
            return
        with self._lock:
            if self._total is None or total > self._total:
                self._total = total

    def mark_completed(self, job_id: str, *, failed: bool = False) -> None:
        if not job_id:
            return
        with self._lock:
            if job_id in self._completed:
                return
            self._completed.add(job_id)
            if failed:
                self._failed.add(job_id)

    def mark_failed(self, job_id: Optional[str] = None) -> None:
        if job_id:
            self.mark_completed(job_id, failed=True)
            return
        with self._lock:
            self._failure_flag = True

    def seed(self, *, total: int, completed: int, failed: int = 0) -> None:
        if total <= 0:
            return
        with self._lock:
            self._total = total
            self._completed = {f"seed-{idx}" for idx in range(max(0, completed))}
            self._failed = {f"seed-failed-{idx}" for idx in range(max(0, min(failed, completed)))}
            self._failure_flag = failed > 0

    def snapshot(self) -> Optional[PolymarketHistoryProgress]:
        with self._lock:
            if not self._total:
                return None
            completed = min(len(self._completed), self._total)
            failed = max(len(self._failed), 1 if self._failure_flag else 0)
            failed = min(failed, completed)
            status = "running"
            if completed >= self._total:
                status = "failed" if failed > 0 else "completed"
            return PolymarketHistoryProgress(
                total=self._total,
                completed=completed,
                failed=failed,
                status=status,
            )


def _load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('\"')
        if key:
            env[key] = value
    return env


def _apply_subgraph_env(env: Dict[str, str]) -> None:
    if ENV_FILE.exists():
        env.update(_load_env_file(ENV_FILE))
        return
    if ENV_SAMPLE_FILE.exists():
        env.update(_load_env_file(ENV_SAMPLE_FILE))


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.absolute()
    try:
        path.relative_to(BASE_DIR)
    except ValueError as exc:
        raise ValueError("Path must be inside the project root.") from exc
    return path


def _normalize_tickers(tickers: Optional[List[str]]) -> Optional[List[str]]:
    if tickers is None:
        return None
    cleaned = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
    if not cleaned:
        raise ValueError("Tickers list is empty after cleaning.")
    return cleaned


def _parse_run_id(stdout: str) -> Optional[str]:
    for line in stdout.splitlines():
        if "run_id=" not in line:
            continue
        match = re.search(r"run_id=([0-9A-Za-z_-]+)", line)
        if match:
            return match.group(1)
    return None


def _default_polymarket_run_id() -> str:
    return datetime.now(timezone.utc).strftime("weekly-history-%Y%m%dT%H%M%SZ")


def _to_kebab_case(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    raw = re.sub(r"[\s_]+", "-", raw)
    raw = re.sub(r"[^a-zA-Z0-9-]", "", raw)
    raw = re.sub(r"-{2,}", "-", raw)
    return raw.strip("-").lower()


def _sanitize_run_dir_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw != Path(raw).name or raw in {".", ".."}:
        raise ValueError("run_dir_name must be a single directory name (no path separators).")
    kebab = _to_kebab_case(raw)
    if not kebab:
        raise ValueError("run_dir_name must contain at least one alphanumeric character.")
    if len(kebab) > MAX_RUN_DIR_NAME_LEN:
        raise ValueError(f"run_dir_name is too long (max {MAX_RUN_DIR_NAME_LEN} characters).")
    return kebab


def _list_run_file_names(run_dir: Path) -> List[str]:
    if not run_dir.exists():
        return []
    names: List[str] = []
    for item in sorted(run_dir.rglob("*")):
        if not item.is_file() or item.name.startswith("."):
            continue
        names.append(_artifact_display_path(item, root=run_dir))
    return names


def _default_run_bars_dir(run_dir: Path) -> Path:
    return run_dir / "bars_history"


def _resolve_run_bars_dir(
    run_dir: Path,
    manifest: Optional[Dict[str, Any]] = None,
    *,
    must_exist: bool = True,
) -> Path:
    manifest = manifest or {}
    raw_value = manifest.get("bars_dir")
    if raw_value:
        try:
            bars_dir = _resolve_project_path(str(raw_value))
        except ValueError:
            path = Path(str(raw_value)).expanduser()
            bars_dir = path if path.is_absolute() else run_dir / path
    else:
        bars_dir = _default_run_bars_dir(run_dir)
    if must_exist and not bars_dir.exists():
        raise FileNotFoundError(f"Run-local bars directory not found: {bars_dir}")
    return bars_dir


def _artifact_last_modified(path: Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _artifact_display_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _artifact_row_count(path: Path) -> Optional[int]:
    if path.suffix.lower() != ".csv":
        return None
    try:
        return _count_csv_rows(path)
    except Exception:
        return None


def _artifact_summary(path: Path, *, root: Path) -> Dict[str, Any]:
    return {
        "name": path.name,
        "path": _artifact_display_path(path, root=root),
        "size_bytes": path.stat().st_size,
        "row_count": _artifact_row_count(path),
        "last_modified": _artifact_last_modified(path),
    }


def _build_run_master_bar_artifacts(
    run_dir: Optional[Path],
    manifest: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if run_dir is None or not run_dir.exists():
        return []
    manifest = manifest or {}
    artifacts: List[Dict[str, Any]] = []
    try:
        bars_dir = _resolve_run_bars_dir(run_dir, manifest, must_exist=False)
    except Exception:
        bars_dir = _default_run_bars_dir(run_dir)
    master_bars = manifest.get("master_bars") if isinstance(manifest.get("master_bars"), dict) else {}
    for freq in MASTER_BAR_FREQS:
        path = bars_dir / f"{freq}.csv"
        meta = master_bars.get(freq) if isinstance(master_bars, dict) else None
        if isinstance(meta, dict) and meta.get("path"):
            try:
                candidate = Path(str(meta["path"]))
                if not candidate.is_absolute():
                    candidate = BASE_DIR / candidate
                path = candidate
            except Exception:
                path = bars_dir / f"{freq}.csv"
        if not path.exists() or not path.is_file():
            continue
        artifacts.append(
            {
                "name": path.name,
                "path": _artifact_display_path(path, root=run_dir),
                "frequency": freq,
                "size_bytes": path.stat().st_size,
                "row_count": _artifact_row_count(path),
                "last_modified": _artifact_last_modified(path),
            }
        )
    return artifacts


def _is_valid_run_relative_path(relative_path: str) -> bool:
    if not relative_path:
        return False
    normalized = relative_path.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return False
    return all(part not in {".", ".."} for part in parts)


def _resolve_run_artifact_path(run_dir: Path, relative_path: str) -> Path:
    if not _is_valid_run_relative_path(relative_path):
        raise ValueError("Invalid artifact path.")
    candidate = (run_dir / relative_path).resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ValueError("Artifact path must stay inside the run directory.") from exc
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"Artifact not found in run directory: {relative_path}")
    return candidate


def _resolve_latest_run_dir() -> Optional[Path]:
    latest_run_id = get_latest_run_id()
    if latest_run_id:
        latest_run_dir = RUNS_DIR / latest_run_id
        if latest_run_dir.exists() and latest_run_dir.is_dir():
            return latest_run_dir

    if not RUNS_DIR.exists():
        return None

    run_dirs = [item for item in RUNS_DIR.iterdir() if item.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda item: item.stat().st_mtime)


def _group_files_from_dir(
    directory: Path,
    *,
    root: Path,
    key: str,
    label: str,
    relative_path: str,
) -> Optional[Dict[str, Any]]:
    if not directory.exists() or not directory.is_dir():
        return None
    files = [
        _artifact_summary(path, root=root)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.name.startswith(".")
    ]
    if not files:
        return None
    return {
        "key": key,
        "label": label,
        "path": relative_path,
        "files": files,
    }


def _build_run_artifact_groups(
    run_dir: Optional[Path],
    manifest: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if run_dir is None or not run_dir.exists():
        return []
    manifest = manifest or {}
    groups: List[Dict[str, Any]] = []

    core_files = [
        _artifact_summary(path, root=run_dir)
        for path in sorted(run_dir.iterdir(), key=lambda item: item.name)
        if path.is_file() and not path.name.startswith(".")
    ]
    if core_files:
        groups.append(
            {
                "key": "core",
                "label": "Core run files",
                "path": ".",
                "files": core_files,
            }
        )

    prn_group = _group_files_from_dir(
        run_dir / "prn_dataset",
        root=run_dir,
        key="prn_dataset",
        label="pRN dataset",
        relative_path="prn_dataset",
    )
    if prn_group is not None:
        groups.append(prn_group)

    for token_role, label in (("yes", "YES raw history"), ("no", "NO raw history")):
        raw_group = _group_files_from_dir(
            run_dir / "raw" / token_role,
            root=run_dir,
            key=f"raw_{token_role}",
            label=label,
            relative_path=f"raw/{token_role}",
        )
        if raw_group is not None:
            groups.append(raw_group)

    bars_dir = _resolve_run_bars_dir(run_dir, manifest, must_exist=False)
    bars_group = _group_files_from_dir(
        bars_dir,
        root=run_dir,
        key="bars_history",
        label="Master bars",
        relative_path=_artifact_display_path(bars_dir, root=run_dir),
    )
    if bars_group is not None:
        groups.append(bars_group)

    for token_role, label in (("yes", "YES analysis bars"), ("no", "NO analysis bars")):
        analysis_group = _group_files_from_dir(
            run_dir / "analysis" / "bars_history" / token_role,
            root=run_dir,
            key=f"analysis_bars_{token_role}",
            label=label,
            relative_path=f"analysis/bars_history/{token_role}",
        )
        if analysis_group is not None:
            groups.append(analysis_group)

    quality_files: List[Path] = []
    for candidate in (
        run_dir / "market_quality.csv",
        run_dir / "market_quality_summary.json",
        run_dir / "feature_manifest.json",
    ):
        if candidate.exists() and candidate.is_file():
            quality_files.append(candidate)
    if quality_files:
        groups.append(
            {
                "key": "quality",
                "label": "Quality and manifest",
                "path": ".",
                "files": [_artifact_summary(path, root=run_dir) for path in quality_files],
            }
        )

    return groups


def _decision_features_prefixed_csv_name(run_dir: Path) -> str:
    return f"{run_dir.name}-decision-features.csv"


def _decision_features_csv_candidates(run_dir: Path) -> List[Path]:
    return [
        run_dir / _decision_features_prefixed_csv_name(run_dir),
        run_dir / LEGACY_DECISION_FEATURES_CSV,
    ]


def _find_existing_decision_features_csv(run_dir: Path) -> Optional[Path]:
    for candidate in _decision_features_csv_candidates(run_dir):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _write_event_urls_file(out_dir: Path, event_urls: List[str]) -> Path:
    tmp_dir = out_dir / "event_sources"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"event_urls_{uuid4().hex}.txt"
    path.write_text("\n".join(event_urls))
    return path


def _cleanup_transient_paths(paths: List[Path]) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            continue

        parent = path.parent
        try:
            if parent.name == "event_sources":
                parent.rmdir()
        except OSError:
            continue


def _list_existing_run_dirs(out_dir: Path) -> set[Path]:
    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        return set()

    existing: set[Path] = set()
    for item in runs_dir.iterdir():
        if not item.is_dir():
            continue
        try:
            existing.add(item.resolve())
        except OSError:
            continue
    return existing


def _resolve_run_dir_candidate(
    out_dir: Path,
    run_id: Optional[str],
    *,
    requested_run_dir_name: Optional[str] = None,
    preexisting_run_dirs: Optional[set[Path]] = None,
) -> Optional[Path]:
    runs_dir = out_dir / "runs"
    if not runs_dir.exists():
        return None

    if run_id:
        candidate = runs_dir / run_id
        if candidate.exists() and candidate.is_dir():
            return candidate

    if requested_run_dir_name:
        try:
            requested_name = _sanitize_run_dir_name(requested_run_dir_name)
        except ValueError:
            requested_name = None
        if requested_name:
            candidate = runs_dir / requested_name
            if candidate.exists() and candidate.is_dir():
                return candidate

    preexisting = preexisting_run_dirs or set()
    created_dirs: List[Path] = []
    for item in runs_dir.iterdir():
        if not item.is_dir():
            continue
        try:
            resolved = item.resolve()
        except OSError:
            continue
        if resolved in preexisting:
            continue
        created_dirs.append(item)

    if not created_dirs:
        return None

    def sort_key(path: Path) -> tuple[float, str]:
        try:
            return (path.stat().st_mtime, path.name)
        except OSError:
            return (0.0, path.name)

    created_dirs.sort(key=sort_key, reverse=True)
    return created_dirs[0]


def _delete_polymarket_run_dir(
    run_dir: Optional[Path],
    *,
    out_dir: Path,
    run_id: Optional[str] = None,
) -> bool:
    if run_dir is None:
        return False

    try:
        resolved_run_dir = run_dir.resolve()
        resolved_runs_root = (out_dir / "runs").resolve()
        resolved_run_dir.relative_to(resolved_runs_root)
    except Exception:
        return False

    try:
        if resolved_runs_root == RUNS_DIR.resolve():
            _clear_latest_pointer_if_matches(run_id or resolved_run_dir.name)
    except Exception:
        pass

    if not resolved_run_dir.exists():
        return False

    shutil.rmtree(resolved_run_dir, ignore_errors=True)
    return not resolved_run_dir.exists()


def _clear_deleted_run_artifacts(
    response: Optional[PolymarketHistoryRunResponse],
) -> None:
    if response is None:
        return
    response.files = []
    response.features_built = False
    response.features_path = None
    response.features_manifest_path = None
    response.master_bar_artifacts = []
    response.artifact_groups = []
    response.shared_artifacts = []
    response.quality_summary = None


def _delete_additional_created_run_dirs(
    *,
    out_dir: Path,
    preexisting_run_dirs: set[Path],
    primary_run_dir: Optional[Path],
) -> None:
    extra_run_dir = _resolve_run_dir_candidate(
        out_dir,
        None,
        requested_run_dir_name=None,
        preexisting_run_dirs=preexisting_run_dirs,
    )
    if extra_run_dir is None:
        return
    if primary_run_dir is not None:
        try:
            if extra_run_dir.resolve() == primary_run_dir.resolve():
                return
        except Exception:
            pass
    _delete_polymarket_run_dir(extra_run_dir, out_dir=out_dir, run_id=extra_run_dir.name)


def _history_state_path(run_dir: Path) -> Path:
    return run_dir / HISTORY_RESUME_STATE_FILENAME


def _load_history_resume_state(run_dir: Optional[Path]) -> Dict[str, Any]:
    if run_dir is None:
        return {}
    return _safe_json_load(_history_state_path(run_dir)) or {}


def _history_resume_completed_counts(run_dir: Optional[Path]) -> tuple[int, int, int]:
    state = _load_history_resume_state(run_dir)
    try:
        total = int(state.get("markets_total") or 0)
    except Exception:
        total = 0
    completed_raw = state.get("completed_market_ids")
    failed_raw = state.get("failed_market_ids")
    completed = len(completed_raw) if isinstance(completed_raw, list) else 0
    failed = len(failed_raw) if isinstance(failed_raw, list) else 0
    return total, completed, failed


def _history_manifest_exists(run_dir: Optional[Path]) -> bool:
    return bool(run_dir and (run_dir / "manifest.json").exists())


def _prn_phase_complete(run_dir: Optional[Path]) -> bool:
    if run_dir is None or not run_dir.exists():
        return False
    training_path = find_run_local_prn_training_file(run_dir)
    return bool(
        training_path
        and training_path.exists()
        and (run_dir / "market_quality.csv").exists()
        and (run_dir / "market_quality_summary.json").exists()
    )


def _features_phase_complete(run_dir: Optional[Path]) -> bool:
    if run_dir is None or not run_dir.exists():
        return False
    return (run_dir / "decision_features.parquet").exists() or _find_existing_decision_features_csv(run_dir) is not None


def _next_resume_phase(run_dir: Optional[Path], *, build_features: bool) -> str:
    if run_dir is None or not run_dir.exists():
        return "history"
    if not _history_manifest_exists(run_dir):
        return "history"
    if not _prn_phase_complete(run_dir):
        return "prn"
    if build_features and not _features_phase_complete(run_dir):
        return "features"
    return "finalizing"


def _runtime_payload_for_run_dir(run_dir: Optional[Path]) -> Dict[str, Any]:
    if run_dir is None or not run_dir.exists():
        return {}
    return read_runtime_file(run_dir) or {}


def _run_requires_features(
    manifest: Optional[Dict[str, Any]] = None,
    runtime_payload: Optional[Dict[str, Any]] = None,
) -> bool:
    manifest = manifest or {}
    runtime_payload = runtime_payload or {}

    pipeline_args = manifest.get("pipeline_args")
    if isinstance(pipeline_args, dict) and "build_features" in pipeline_args:
        return bool(pipeline_args.get("build_features"))

    if "build_features_requested" in manifest:
        return bool(manifest.get("build_features_requested"))

    request_payload = runtime_payload.get("request_payload")
    if isinstance(request_payload, dict) and "build_features" in request_payload:
        return bool(request_payload.get("build_features"))

    return False


def _run_features_pending(
    run_dir: Optional[Path],
    manifest: Optional[Dict[str, Any]] = None,
    runtime_payload: Optional[Dict[str, Any]] = None,
) -> bool:
    if run_dir is None or not run_dir.exists():
        return False
    manifest = manifest or {}
    runtime_payload = runtime_payload or {}
    if not _run_requires_features(manifest, runtime_payload):
        return False
    return _prn_phase_complete(run_dir) and not _features_phase_complete(run_dir)


def _run_pending_phase(
    run_dir: Optional[Path],
    manifest: Optional[Dict[str, Any]] = None,
    runtime_payload: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if run_dir is None or not run_dir.exists():
        return None
    manifest = manifest or {}
    runtime_payload = runtime_payload or {}
    if not _run_features_pending(run_dir, manifest, runtime_payload):
        return None
    return "features"


def _run_artifacts_accessible(
    run_dir: Optional[Path],
    manifest: Optional[Dict[str, Any]] = None,
    runtime_payload: Optional[Dict[str, Any]] = None,
) -> bool:
    if run_dir is None or not run_dir.exists():
        return False
    manifest = manifest or {}
    runtime_payload = runtime_payload or {}

    explicit = manifest.get("artifacts_accessible")
    if explicit is not None:
        return bool(explicit)

    if runtime_payload:
        next_phase = _next_resume_phase(
            run_dir,
            build_features=_run_requires_features(manifest, runtime_payload),
        )
        if next_phase != "finalizing":
            return False

    if _run_pending_phase(run_dir, manifest, runtime_payload) is not None:
        return False

    status = str(manifest.get("status") or "")
    if status in {"failed", "cancelled", "pending"}:
        return False

    return True


def _polymarket_runtime_payload(
    handle: ManagedProcessHandle,
    *,
    payload: PolymarketHistoryRunRequest,
    run_dir: Path,
    phase: str,
) -> Dict[str, Any]:
    runtime = runtime_payload_for_handle(handle)
    runtime["run_id"] = run_dir.name
    runtime["phase"] = phase
    runtime["request_payload"] = payload.model_dump(mode="json")
    runtime["resume_attempts"] = MAX_AUTOMATIC_RESUME_ATTEMPTS
    return runtime


def _write_polymarket_runtime_file(
    handle: ManagedProcessHandle,
    *,
    payload: PolymarketHistoryRunRequest,
    run_dir: Optional[Path],
    phase: str,
) -> None:
    if run_dir is None:
        return
    write_runtime_file(
        run_dir,
        _polymarket_runtime_payload(handle, payload=payload, run_dir=run_dir, phase=phase),
    )


def _iter_polymarket_runtime_dirs() -> List[tuple[Path, Dict[str, Any]]]:
    if not RUNS_DIR.exists():
        return []
    runtime_dirs: List[tuple[Path, Dict[str, Any]]] = []
    for entry in sorted(RUNS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        payload = read_runtime_file(entry)
        if not payload:
            continue
        service_name = str(payload.get("service") or "")
        if service_name not in POLYMARKET_RUNTIME_SERVICES:
            continue
        runtime_dirs.append((entry, payload))
    return runtime_dirs


def _find_polymarket_runtime_job(job_id: str) -> Optional[tuple[Path, Dict[str, Any]]]:
    for run_dir, payload in _iter_polymarket_runtime_dirs():
        runtime_job_id = payload.get("job_id")
        if runtime_job_id is None:
            continue
        if str(runtime_job_id) == job_id:
            return run_dir, payload
    return None


def _parse_runtime_started_at(payload: Dict[str, Any]) -> Optional[datetime]:
    started_at_raw = payload.get("started_at")
    if started_at_raw is None:
        return None
    try:
        return datetime.fromisoformat(str(started_at_raw))
    except Exception:
        return None


def _runtime_backed_polymarket_status(
    job_id: str,
    runtime_payload: Dict[str, Any],
    run_dir: Path,
    *,
    status: str = "running",
    error: Optional[str] = None,
    finished_at: Optional[datetime] = None,
) -> PolymarketHistoryJobStatus:
    total, completed, failed = _history_resume_completed_counts(run_dir)
    progress = None
    if total > 0:
        completed_clamped = min(completed, total)
        failed_clamped = min(failed, completed_clamped)
        progress_status: str = "running"
        if status != "running":
            progress_status = "failed" if failed_clamped > 0 else "completed"
        progress = PolymarketHistoryProgress(
            total=total,
            completed=completed_clamped,
            failed=failed_clamped,
            status=progress_status,  # type: ignore[arg-type]
        )
    manifest = _load_run_manifest(run_dir) if run_dir.exists() else {}
    artifacts_accessible = _run_artifacts_accessible(run_dir, manifest, runtime_payload)
    files = _list_run_file_names(run_dir) if run_dir.exists() and artifacts_accessible else []
    features_requested = _run_requires_features(manifest, runtime_payload)
    result = PolymarketHistoryRunResponse(
        ok=False,
        run_id=run_dir.name,
        out_dir=str(DEFAULT_OUT_DIR.relative_to(BASE_DIR)),
        run_dir=str(run_dir.relative_to(BASE_DIR)),
        files=files,
        stdout="",
        stderr="",
        duration_s=0.0,
        command=[str(item) for item in runtime_payload.get("command") or []],
        features_built=bool(manifest.get("features_built")),
        features_path=(
            str(_resolve_features_output_path(run_dir).relative_to(run_dir))
            if _resolve_features_output_path(run_dir) is not None
            else None
        ),
        features_manifest_path=(
            "feature_manifest.json" if (run_dir / "feature_manifest.json").exists() else None
        ),
        master_bar_artifacts=(
            _build_run_master_bar_artifacts(run_dir, manifest)
            if artifacts_accessible
            else []
        ),
        artifact_groups=(
            _build_run_artifact_groups(run_dir, manifest)
            if artifacts_accessible
            else []
        ),
        shared_artifacts=[],
        quality_summary=_load_quality_summary_for_run_dir(run_dir),
    )
    return PolymarketHistoryJobStatus(
        job_id=job_id,
        status=status,  # type: ignore[arg-type]
        phase=str(runtime_payload.get("phase") or _next_resume_phase(run_dir, build_features=features_requested)),  # type: ignore[arg-type]
        progress=progress,
        features_progress=None,
        telemetry=None,
        result=result,
        error=error,
        started_at=_parse_runtime_started_at(runtime_payload),
        finished_at=finished_at,
    )


def _build_history_command(
    payload: PolymarketHistoryRunRequest,
) -> tuple[List[str], Dict[str, str], Path, List[Path]]:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Weekly history script not found at {SCRIPT_PATH}")

    out_dir = _resolve_project_path(str(payload.out_dir)) if payload.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_paths: List[Path] = []

    try:
        cmd: List[str] = [sys.executable, str(SCRIPT_PATH), "--out-dir", str(out_dir)]
        sanitized_run_dir_name = _sanitize_run_dir_name(payload.run_dir_name) if payload.run_dir_name else None
        if not sanitized_run_dir_name:
            sanitized_run_dir_name = _default_polymarket_run_id()
        candidate_run_dir = out_dir / "runs" / sanitized_run_dir_name
        if candidate_run_dir.exists() and not payload.resume_existing:
            raise ValueError(f"Run directory already exists: {candidate_run_dir.relative_to(BASE_DIR)}")
        payload.run_dir_name = sanitized_run_dir_name
        cmd.extend(["--run-id", sanitized_run_dir_name])
        run_bars_dir = _default_run_bars_dir(candidate_run_dir)
        payload.bars_dir = str(run_bars_dir)
        cmd.extend(["--bars-dir", str(run_bars_dir)])
        if payload.resume_existing:
            cmd.append("--resume")

        tickers = _normalize_tickers(payload.tickers)
        if tickers:
            cmd.extend(["--tickers", ",".join(tickers)])

        if payload.tickers_csv:
            tickers_csv = _resolve_project_path(payload.tickers_csv)
            if not tickers_csv.exists():
                raise ValueError(f"tickers_csv not found: {tickers_csv}")
            cmd.extend(["--tickers-csv", str(tickers_csv)])

        event_urls_used = False
        if payload.event_urls:
            cleaned = [value.strip() for value in payload.event_urls if value and value.strip()]
            if cleaned:
                event_urls_file = _write_event_urls_file(out_dir, cleaned)
                temp_paths.append(event_urls_file)
                cmd.extend(["--event-urls-file", str(event_urls_file)])
                event_urls_used = True

        if payload.event_urls_file:
            event_urls_file = _resolve_project_path(payload.event_urls_file)
            if not event_urls_file.exists():
                raise ValueError(f"event_urls_file not found: {event_urls_file}")
            cmd.extend(["--event-urls-file", str(event_urls_file)])
            event_urls_used = True

        if not event_urls_used and DEFAULT_EVENT_URLS_FILE.exists():
            cmd.extend(["--event-urls-file", str(DEFAULT_EVENT_URLS_FILE)])

        if payload.start_date:
            cmd.extend(["--start-date", payload.start_date])
        if payload.end_date:
            cmd.extend(["--end-date", payload.end_date])

        if payload.fidelity_min is not None:
            cmd.extend(["--fidelity-min", str(payload.fidelity_min)])

        if payload.bars_freqs:
            cmd.extend(["--bars-freqs", payload.bars_freqs])

        if payload.dim_market_out:
            dim_path = _resolve_project_path(payload.dim_market_out)
            cmd.extend(["--dim-market-out", str(dim_path)])

        if payload.fact_trade_dir:
            fact_dir = _resolve_project_path(payload.fact_trade_dir)
            cmd.extend(["--fact-trade-dir", str(fact_dir)])

        if payload.include_subgraph:
            cmd.append("--include-subgraph")

        if payload.max_subgraph_entities is not None:
            cmd.extend(["--max-subgraph-entities", str(payload.max_subgraph_entities)])

        if payload.dry_run:
            cmd.append("--dry-run")

        env = {**os.environ}
        _apply_subgraph_env(env)
        existing = env.get("PYTHONPATH")
        root = str(BASE_DIR)
        if existing:
            env["PYTHONPATH"] = os.pathsep.join([existing, root])
        else:
            env["PYTHONPATH"] = root
        return cmd, env, out_dir, temp_paths
    except Exception:
        _cleanup_transient_paths(temp_paths)
        raise


# ---------------------------------------------------------------------------
# Run management helpers (Phase 0 + 1)
# ---------------------------------------------------------------------------


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    return _safe_json_load(run_dir / "manifest.json") or {}


def _load_quality_summary_for_run_dir(run_dir: Optional[Path]) -> Optional[PolymarketQualitySummary]:
    if run_dir is None or not run_dir.exists():
        return None
    try:
        summary = load_market_quality_summary(run_dir)
        if summary is not None:
            return summary
    except Exception:
        pass
    manifest = _load_run_manifest(run_dir)
    payload = manifest.get("quality_summary")
    if isinstance(payload, dict):
        try:
            return PolymarketQualitySummary.model_validate(payload)
        except Exception:
            return None
    return None


def _resolve_manifest_path(value: Optional[str], *, fallback: Path, label: str, must_exist: bool = True) -> Path:
    path = _resolve_project_path(value) if value else fallback
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_manifest_date(manifest: Dict[str, Any], key: str) -> Optional[str]:
    value = manifest.get(key)
    if value:
        return value
    pipeline_args = manifest.get("pipeline_args")
    if isinstance(pipeline_args, dict):
        value = pipeline_args.get(key)
    return value if value else None


def _expand_dim_market_candidates(path: Path) -> List[Path]:
    candidates = [path]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        candidates.append(path.with_suffix(".parquet"))
    elif suffix == ".parquet":
        candidates.append(path.with_suffix(".csv"))
    return candidates


def _find_dim_market_for_run(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    candidates: List[Path] = []

    raw_value = manifest.get("dim_market") or manifest.get("dim_market_out")
    if raw_value:
        try:
            resolved = _resolve_project_path(str(raw_value))
            candidates.extend(_expand_dim_market_candidates(resolved))
        except ValueError:
            path = Path(str(raw_value)).expanduser()
            candidates.extend(_expand_dim_market_candidates(path))

    candidates.extend(
        [
            run_dir / "dim_market_weekly.csv",
            run_dir / f"{run_dir.name}-dim-market-weekly.csv",
            run_dir / "dim_market.csv",
            run_dir / f"{run_dir.name}-dim-market.csv",
            run_dir / "dim_market.parquet",
            run_dir / f"{run_dir.name}-dim-market.parquet",
            run_dir / f"{run_dir.name}-dim-market-weekly.parquet",
        ]
    )

    candidates.extend(
        [
            DIM_MARKET_WEEKLY_PATH,
            DIM_MARKET_WEEKLY_PATH.with_suffix(".parquet"),
            BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market.parquet",
            BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market.csv",
        ]
    )

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    details = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"dim_market not found. Checked: {details}")


def _atomic_json_write(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically via temp + rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def _artifact_entry(file_path: Path) -> Dict[str, Any]:
    """Build an artifact inventory entry for a file."""
    entry: Dict[str, Any] = {"size_bytes": file_path.stat().st_size}
    if file_path.suffix.lower() == ".csv":
        try:
            with file_path.open(newline="") as handle:
                entry["rows"] = sum(1 for _ in handle) - 1  # subtract header
        except Exception:
            pass
    return entry


def _build_artifact_inventory(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan run dir and build artifact inventory."""
    inventory: Dict[str, Dict[str, Any]] = {}
    if not run_dir or not run_dir.exists():
        return inventory
    for item in sorted(run_dir.rglob("*")):
        if not item.is_file() or item.name == "manifest.json" or item.name.startswith("."):
            continue
        inventory[_artifact_display_path(item, root=run_dir)] = _artifact_entry(item)
    return inventory


def _enhance_manifest(
    run_dir: Path,
    *,
    status: str,
    duration_s: float,
    pipeline_args: Optional[Dict[str, Any]] = None,
    error_summary: Optional[str] = None,
    features_built: bool = False,
) -> None:
    """Enrich the existing manifest.json with run management metadata."""
    manifest_path = run_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}

    manifest["status"] = status
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["duration_s"] = duration_s
    manifest.setdefault("label", None)
    manifest.setdefault("pinned", False)

    if pipeline_args:
        manifest["pipeline_args"] = pipeline_args
    manifest["build_features_requested"] = _run_requires_features(
        manifest,
        {"request_payload": pipeline_args or {}},
    )
    manifest["features_built"] = features_built
    manifest["pending_phase"] = None
    manifest["artifacts_accessible"] = status == "success"
    if error_summary:
        manifest["error_summary"] = error_summary
    elif status == "success":
        manifest.pop("error_summary", None)

    manifest["artifacts"] = _build_artifact_inventory(run_dir)

    _atomic_json_write(manifest_path, manifest)


def _update_features_manifest(run_dir: Path, *, features_built: bool) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}
    manifest["build_features_requested"] = True
    manifest["features_built"] = features_built
    manifest["pending_phase"] = None if features_built else "features"
    manifest["artifacts_accessible"] = features_built
    manifest["status"] = "success" if features_built else "pending"
    if features_built:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest.pop("error_summary", None)
    manifest["artifacts"] = _build_artifact_inventory(run_dir)
    _atomic_json_write(manifest_path, manifest)


def _set_run_pending_state(
    run_dir: Path,
    *,
    build_features_requested: bool,
    pipeline_args: Optional[Dict[str, Any]] = None,
    error_summary: Optional[str] = None,
    pending_phase: str = "features",
    duration_s: Optional[float] = None,
    finished: bool = False,
) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}

    existing_pipeline_args = manifest.get("pipeline_args")
    merged_pipeline_args: Dict[str, Any] = {}
    if isinstance(existing_pipeline_args, dict):
        merged_pipeline_args.update(existing_pipeline_args)
    if pipeline_args:
        merged_pipeline_args.update(pipeline_args)
    if build_features_requested:
        merged_pipeline_args["build_features"] = True

    manifest["status"] = "pending"
    manifest["build_features_requested"] = build_features_requested
    if merged_pipeline_args:
        manifest["pipeline_args"] = merged_pipeline_args
    manifest["features_built"] = False
    manifest["pending_phase"] = pending_phase
    manifest["artifacts_accessible"] = False
    if duration_s is not None:
        manifest["duration_s"] = duration_s
    if finished:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    if error_summary:
        manifest["error_summary"] = error_summary
    else:
        manifest.pop("error_summary", None)
    manifest["artifacts"] = _build_artifact_inventory(run_dir)
    _atomic_json_write(manifest_path, manifest)


def _update_latest_pointer(run_id: str) -> None:
    """Write latest.json pointing to the given run, atomically."""
    latest_path = DEFAULT_OUT_DIR / "latest.json"
    payload = {
        "run_id": run_id,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_by": "pipeline",
    }
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(latest_path, payload)


def _copy_dim_market_to_run(run_dir: Path) -> None:
    """Copy dim_market_weekly.csv into the run dir for provenance."""
    if DIM_MARKET_WEEKLY_PATH.exists() and run_dir and run_dir.exists():
        dest = run_dir / "dim_market_weekly.csv"
        if not dest.exists():
            try:
                shutil.copy2(DIM_MARKET_WEEKLY_PATH, dest)
            except OSError:
                pass


def _pipeline_args_from_payload(payload: PolymarketHistoryRunRequest) -> Dict[str, Any]:
    """Extract pipeline args from the request payload for manifest storage."""
    args: Dict[str, Any] = {}
    if payload.tickers:
        args["tickers"] = payload.tickers
    if payload.start_date:
        args["start_date"] = payload.start_date
    if payload.end_date:
        args["end_date"] = payload.end_date
    if payload.fidelity_min is not None:
        args["fidelity_min"] = payload.fidelity_min
    if payload.bars_freqs:
        args["bars_freqs"] = payload.bars_freqs
    if payload.run_dir_name:
        sanitized_run_dir_name = _sanitize_run_dir_name(payload.run_dir_name)
        if sanitized_run_dir_name:
            args["run_dir_name"] = sanitized_run_dir_name
    args["include_subgraph"] = payload.include_subgraph
    args["build_features"] = payload.build_features
    args["skip_subgraph_labels"] = payload.skip_subgraph_labels
    return args


def get_latest_pointer() -> Optional[Dict[str, Any]]:
    """Read the latest.json pointer file."""
    return _safe_json_load(DEFAULT_OUT_DIR / "latest.json")


def get_latest_run_id() -> Optional[str]:
    """Return the run_id from latest.json, or None."""
    pointer = get_latest_pointer()
    if pointer:
        return pointer.get("run_id")
    return None


def _clear_latest_pointer_if_matches(run_id: str) -> None:
    latest_path = DEFAULT_OUT_DIR / "latest.json"
    pointer = _safe_json_load(latest_path)
    if not pointer or pointer.get("run_id") != run_id:
        return
    try:
        latest_path.unlink()
    except FileNotFoundError:
        return


def _ensure_run_features_csv(run_dir: Path) -> Optional[Path]:
    existing_csv = _find_existing_decision_features_csv(run_dir)
    if existing_csv:
        return existing_csv

    csv_path = run_dir / _decision_features_prefixed_csv_name(run_dir)
    parquet_path = run_dir / "decision_features.parquet"
    if not parquet_path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
    except Exception:
        return None
    return csv_path if csv_path.exists() else None


def _is_valid_run_filename(filename: str) -> bool:
    return bool(filename) and "/" not in filename and "\\" not in filename and ".." not in filename


def _resolve_run_file_path(run_dir: Path, filename: str) -> Path:
    """Resolve a safe file path under a run dir, with legacy/new decision_features.csv aliases."""
    if not _is_valid_run_filename(filename):
        raise ValueError("Invalid filename.")
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_dir.name}")

    prefixed_decision_name = _decision_features_prefixed_csv_name(run_dir)
    if filename in {LEGACY_DECISION_FEATURES_CSV, prefixed_decision_name}:
        _ensure_run_features_csv(run_dir)
        direct = run_dir / filename
        if direct.exists() and direct.is_file():
            return direct
        existing_decision_csv = _find_existing_decision_features_csv(run_dir)
        if existing_decision_csv and existing_decision_csv.exists():
            return existing_decision_csv

    file_path = run_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found in run directory: {filename}")
    return file_path


def _rename_run_csv_files(run_dir: Path) -> None:
    """Normalize top-level run CSV names to {run_dir}-{csv-type}.csv without failing the job."""
    if not run_dir.exists() or not run_dir.is_dir():
        return

    prefix = f"{run_dir.name}-"
    for item in sorted(run_dir.iterdir(), key=lambda path: path.name):
        if not item.is_file() or item.suffix.lower() != ".csv":
            continue
        if item.name in PRESERVED_RUNTIME_CSVS:
            continue
        if item.name.startswith(prefix):
            continue

        normalized_stem = _to_kebab_case(item.stem) or "csv"
        target_name = f"{prefix}{normalized_stem}.csv"
        if item.name == target_name:
            continue

        target_path = run_dir / target_name
        if target_path.exists():
            continue
        try:
            item.rename(target_path)
        except OSError:
            continue


def _resolve_features_output_path(run_dir: Path) -> Optional[Path]:
    features_csv = _find_existing_decision_features_csv(run_dir)
    if features_csv:
        return features_csv
    parquet_path = run_dir / "decision_features.parquet"
    if parquet_path.exists():
        return parquet_path
    return None


def _build_run_csv_files(run_dir: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    csv_entries: Dict[str, Dict[str, Any]] = {}

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for name, meta in artifacts.items():
            if not isinstance(name, str) or not name.lower().endswith(".csv"):
                continue
            if isinstance(meta, dict):
                size_bytes = meta.get("size_bytes")
                row_count = meta.get("rows")
            else:
                size_bytes = None
                row_count = None
            entry: Dict[str, Any] = {"name": name, "size_bytes": int(size_bytes) if isinstance(size_bytes, int) else 0}
            if isinstance(row_count, int):
                entry["row_count"] = max(0, row_count)
            csv_entries[name] = entry

    for item in sorted(run_dir.iterdir(), key=lambda path: path.name):
        if not item.is_file() or item.suffix.lower() != ".csv":
            continue
        existing = csv_entries.get(item.name)
        row_count = existing.get("row_count") if existing else None
        entry: Dict[str, Any] = {
            "name": item.name,
            "size_bytes": item.stat().st_size,
        }
        if isinstance(row_count, int):
            entry["row_count"] = row_count
        csv_entries[item.name] = entry

    parquet_path = run_dir / "decision_features.parquet"
    if parquet_path.exists() and _find_existing_decision_features_csv(run_dir) is None:
        synthetic_name = _decision_features_prefixed_csv_name(run_dir)
        csv_entries.setdefault(
            synthetic_name,
            {
                "name": synthetic_name,
                "size_bytes": 0,
            },
        )

    return [csv_entries[name] for name in sorted(csv_entries)]


def build_run_decision_features(
    run_id: str,
    payload: PolymarketRunFeaturesRequest,
) -> PolymarketRunFeaturesResponse:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")

    start = time.monotonic()
    existing_csv = _find_existing_decision_features_csv(run_dir)
    if existing_csv:
        features_manifest_path = run_dir / "feature_manifest.json"
        _update_features_manifest(run_dir, features_built=True)
        return PolymarketRunFeaturesResponse(
            ok=True,
            run_id=run_id,
            run_dir=str(run_dir.relative_to(BASE_DIR)),
            features_built=True,
            features_path=str(existing_csv.relative_to(run_dir)),
            features_manifest_path=(
                str(features_manifest_path.relative_to(run_dir))
                if features_manifest_path.exists()
                else None
            ),
            stdout="Decision features already present.",
            stderr="",
            duration_s=round(time.monotonic() - start, 3),
            command=[],
        )

    generated_csv = _ensure_run_features_csv(run_dir)
    if generated_csv:
        features_manifest_path = run_dir / "feature_manifest.json"
        _update_features_manifest(run_dir, features_built=True)
        return PolymarketRunFeaturesResponse(
            ok=True,
            run_id=run_id,
            run_dir=str(run_dir.relative_to(BASE_DIR)),
            features_built=True,
            features_path=str(generated_csv.relative_to(run_dir)),
            features_manifest_path=(
                str(features_manifest_path.relative_to(run_dir))
                if features_manifest_path.exists()
                else None
            ),
            stdout="Generated decision features CSV from existing parquet output.",
            stderr="",
            duration_s=round(time.monotonic() - start, 3),
            command=[],
        )

    manifest = _load_run_manifest(run_dir)
    bars_dir = _resolve_run_bars_dir(run_dir, manifest)
    if not bars_dir.is_dir():
        raise ValueError(f"bars_dir must be a directory: {bars_dir}")
    dim_market_path = _find_dim_market_for_run(run_dir, manifest)

    start_date = _resolve_manifest_date(manifest, "start_date")
    end_date = _resolve_manifest_date(manifest, "end_date")

    features_payload = PolymarketHistoryRunRequest(
        start_date=start_date,
        end_date=end_date,
        skip_subgraph_labels=payload.skip_subgraph_labels,
    )
    cmd, env = _build_features_command(
        features_payload,
        bars_dir,
        dim_market_path,
        run_dir,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    duration_s = round(time.monotonic() - start, 3)

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        _set_run_pending_state(
            run_dir,
            build_features_requested=True,
            pipeline_args=manifest.get("pipeline_args") if isinstance(manifest.get("pipeline_args"), dict) else None,
            error_summary=(stderr or "Decision features build failed.").strip(),
            pending_phase="features",
            duration_s=duration_s,
            finished=True,
        )
        return PolymarketRunFeaturesResponse(
            ok=False,
            run_id=run_id,
            run_dir=str(run_dir.relative_to(BASE_DIR)),
            features_built=False,
            features_path=None,
            features_manifest_path=None,
            stdout=stdout,
            stderr=stderr or "Decision features build failed.",
            duration_s=duration_s,
            command=cmd,
        )

    parquet_path = run_dir / "decision_features.parquet"
    csv_path = run_dir / LEGACY_DECISION_FEATURES_CSV
    if parquet_path.exists() and not csv_path.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
        except Exception:
            pass

    try:
        _rename_run_csv_files(run_dir)
    except Exception:
        pass

    features_path = _resolve_features_output_path(run_dir)
    features_manifest_path = run_dir / "feature_manifest.json"
    features_built = features_path is not None

    if features_built:
        try:
            _update_features_manifest(run_dir, features_built=True)
        except Exception:
            pass
    else:
        _set_run_pending_state(
            run_dir,
            build_features_requested=True,
            pipeline_args=manifest.get("pipeline_args") if isinstance(manifest.get("pipeline_args"), dict) else None,
            error_summary="Decision features build finished without writing an output file.",
            pending_phase="features",
            duration_s=duration_s,
            finished=True,
        )

    return PolymarketRunFeaturesResponse(
        ok=features_built,
        run_id=run_id,
        run_dir=str(run_dir.relative_to(BASE_DIR)),
        features_built=features_built,
        features_path=(
            str(features_path.relative_to(run_dir))
            if features_path is not None
            else None
        ),
        features_manifest_path=(
            str(features_manifest_path.relative_to(run_dir))
            if features_manifest_path.exists()
            else None
        ),
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
        command=cmd,
    )


def get_pipeline_run_file_path(run_id: str, filename: str) -> Path:
    """Resolve a file path under a run directory for safe download/open."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    manifest = _load_run_manifest(run_dir)
    runtime_payload = _runtime_payload_for_run_dir(run_dir)
    if not _run_artifacts_accessible(run_dir, manifest, runtime_payload):
        raise RuntimeError(
            f"Artifacts are unavailable while run '{run_id}' is pending decision features."
        )
    return _resolve_run_file_path(run_dir, filename)


def get_pipeline_run_artifact_file_path(run_id: str, relative_path: str) -> Path:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    manifest = _load_run_manifest(run_dir)
    runtime_payload = _runtime_payload_for_run_dir(run_dir)
    if not _run_artifacts_accessible(run_dir, manifest, runtime_payload):
        raise RuntimeError(
            f"Artifacts are unavailable while run '{run_id}' is pending decision features."
        )
    return _resolve_run_artifact_path(run_dir, relative_path)


def get_run_master_bar_file_path(run_id: str, freq: str) -> Path:
    if freq not in MASTER_BAR_FREQS:
        raise ValueError(f"Unsupported master bar frequency: {freq}")
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    manifest = _load_run_manifest(run_dir)
    runtime_payload = _runtime_payload_for_run_dir(run_dir)
    if not _run_artifacts_accessible(run_dir, manifest, runtime_payload):
        raise RuntimeError(
            f"Artifacts are unavailable while run '{run_id}' is pending decision features."
        )
    path = _resolve_run_bars_dir(run_dir, manifest, must_exist=False) / f"{freq}.csv"
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Run-local master bars file not found for run={run_id} freq={freq}")
    return path


def get_master_bar_file_path(freq: str) -> Path:
    run_dir = _resolve_latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError("No Polymarket run is available for master bar preview.")
    return get_run_master_bar_file_path(run_dir.name, freq)


def _read_csv_head_preview(
    path: Path,
    limit: int,
) -> tuple[List[str], List[Dict[str, Optional[str]]]]:
    rows: List[Dict[str, Optional[str]]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append({key: row.get(key) for key in headers})
    return headers, rows


def _read_csv_tail_preview(
    path: Path,
    limit: int,
) -> tuple[List[str], List[Dict[str, Optional[str]]], int]:
    buffer: deque[Dict[str, Optional[str]]] = deque(maxlen=limit)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        row_count = 0
        for row in reader:
            row_count += 1
            buffer.append({key: row.get(key) for key in headers})
    return headers, list(buffer), row_count


def _preview_csv_file(
    csv_path: Path,
    *,
    filename: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Preview is only supported for CSV files.")

    sanitized_limit = max(1, min(limit, 100))
    normalized_mode = mode.lower()
    if normalized_mode not in {"head", "tail"}:
        raise ValueError("mode must be 'head' or 'tail'")

    try:
        if normalized_mode == "tail":
            headers, rows, row_count = _read_csv_tail_preview(csv_path, sanitized_limit)
        else:
            headers, rows = _read_csv_head_preview(csv_path, sanitized_limit)
            row_count = None
    except Exception as exc:
        raise ValueError(f"Error reading CSV file: {exc}") from exc

    return {
        "filename": filename,
        "headers": headers,
        "rows": rows,
        "row_count": row_count,
        "mode": normalized_mode,
        "limit": sanitized_limit,
    }


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        return sum(1 for _ in handle)


def _master_bar_path(freq: str) -> Path:
    if freq not in MASTER_BAR_FREQS:
        raise ValueError(f"Unsupported master bar frequency: {freq}")
    run_dir = _resolve_latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError("No Polymarket run is available for master bars.")
    return get_run_master_bar_file_path(run_dir.name, freq)


def _shared_artifact_entry(path: Path, *, freq: str, row_count: Optional[int] = None) -> Dict[str, Any]:
    if row_count is None:
        try:
            row_count = _count_csv_rows(path)
        except Exception:
            row_count = None
    try:
        display_path = str(path.relative_to(BASE_DIR))
    except ValueError:
        display_path = str(path)
    return {
        "name": path.name,
        "path": display_path,
        "frequency": freq,
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def _build_shared_artifacts(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    return []


def get_run_artifact_csv_preview(
    run_id: str,
    relative_path: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    path = get_pipeline_run_artifact_file_path(run_id, relative_path)
    return _preview_csv_file(path, filename=relative_path, limit=limit, mode=mode)


def get_run_master_bar_csv_preview(
    run_id: str,
    freq: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    path = get_run_master_bar_file_path(run_id, freq)
    return _preview_csv_file(path, filename=f"{freq}.csv", limit=limit, mode=mode)


def get_master_bar_csv_preview(
    freq: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    path = get_master_bar_file_path(freq)
    return _preview_csv_file(path, filename=path.name, limit=limit, mode=mode)


# ---------------------------------------------------------------------------
# Run management: list / rename / set-active / delete
# ---------------------------------------------------------------------------


def list_pipeline_runs() -> List[Dict[str, Any]]:
    """List all pipeline runs with manifest data, newest first."""
    if not RUNS_DIR.exists():
        return []
    latest_run_id = get_latest_run_id()
    runs: List[Dict[str, Any]] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        manifest = _safe_json_load(run_dir / "manifest.json") or {}
        runtime_payload = _runtime_payload_for_run_dir(run_dir)
        runtime_status: Optional[PolymarketHistoryJobStatus] = None
        runtime_job_id = runtime_payload.get("job_id") if runtime_payload else None
        if runtime_job_id is not None:
            try:
                runtime_status = POLYMARKET_HISTORY_JOB_MANAGER.get_status(str(runtime_job_id))
            except Exception:
                runtime_status = None
            runtime_payload = _runtime_payload_for_run_dir(run_dir)

        features_requested = _run_requires_features(manifest, runtime_payload)
        features_pending = _run_features_pending(run_dir, manifest, runtime_payload)
        pending_phase = _run_pending_phase(run_dir, manifest, runtime_payload)

        status = str(manifest.get("status", "unknown"))
        if runtime_status and runtime_status.status in {"queued", "running"}:
            status = str(runtime_status.status)
        elif features_pending:
            status = "pending"
        elif status == "unknown" and _run_artifacts_accessible(run_dir, manifest, runtime_payload):
            status = "success"

        if status in {"failed", "cancelled"} and not features_pending:
            try:
                _clear_latest_pointer_if_matches(run_dir.name)
                shutil.rmtree(run_dir)
            except Exception:
                pass
            continue
        artifacts_accessible = _run_artifacts_accessible(run_dir, manifest, runtime_payload)
        size_bytes = sum(
            item.stat().st_size
            for item in run_dir.rglob("*")
            if item.is_file() and not item.name.startswith(".")
        )
        quality_summary = _load_quality_summary_for_run_dir(run_dir)
        features_built = bool(manifest.get("features_built")) or _features_phase_complete(run_dir)
        csv_files = _build_run_csv_files(run_dir, manifest) if artifacts_accessible else []
        master_bar_artifacts = (
            _build_run_master_bar_artifacts(run_dir, manifest) if artifacts_accessible else []
        )
        artifact_groups = _build_run_artifact_groups(run_dir, manifest) if artifacts_accessible else []
        runs.append({
            "run_id": run_dir.name,
            "run_dir": str(run_dir.relative_to(BASE_DIR)),
            "label": manifest.get("label"),
            "status": status,
            "created_at_utc": manifest.get("created_at_utc"),
            "finished_at_utc": manifest.get("finished_at_utc"),
            "duration_s": manifest.get("duration_s"),
            "tickers": manifest.get("tickers"),
            "start_date": manifest.get("start_date"),
            "end_date": manifest.get("end_date"),
            "markets": manifest.get("markets"),
            "price_rows": manifest.get("price_rows"),
            "features_built": features_built,
            "features_requested": features_requested,
            "pending_phase": pending_phase,
            "artifacts_accessible": artifacts_accessible,
            "pinned": manifest.get("pinned", False),
            "is_active": run_dir.name == latest_run_id,
            "artifact_count": len(manifest.get("artifacts", {})) if artifacts_accessible else 0,
            "csv_files": csv_files,
            "master_bar_artifacts": master_bar_artifacts,
            "artifact_groups": artifact_groups,
            "shared_artifacts": [],
            "size_bytes": size_bytes,
            "error_summary": manifest.get("error_summary"),
            "quality_summary": quality_summary.model_dump() if quality_summary is not None else None,
        })
    runs.sort(key=lambda r: r.get("created_at_utc") or "", reverse=True)
    return runs


def rename_pipeline_run(
    run_id: str,
    label: str,
    new_dir_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Rename a pipeline run's label and optionally its directory.

    When *new_dir_name* is provided the function performs a full rename:
    directory, manifest internals, prefixed CSV files, and latest.json pointer.
    When omitted, only the manifest label is updated (backward-compatible).
    """
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")

    effective_run_id = run_id
    effective_dir = run_dir

    if new_dir_name is not None:
        new_kebab = _to_kebab_case(new_dir_name)
        if not new_kebab:
            raise ValueError("Directory name must contain at least one alphanumeric character.")
        new_dir = RUNS_DIR / new_kebab
        if new_dir != run_dir:
            if new_dir.exists():
                raise ValueError(f"Target directory already exists: {new_kebab}")

            _rename_prefixed_run_files(run_dir, run_id, new_kebab)

            run_dir.rename(new_dir)
            effective_dir = new_dir
            effective_run_id = new_kebab

            latest_id = get_latest_run_id()
            if latest_id == run_id:
                _update_latest_pointer(new_kebab)

    manifest_path = effective_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}
    manifest["label"] = label.strip() if label else None
    if effective_run_id != run_id:
        manifest["run_id"] = effective_run_id
        pipeline_args = manifest.get("pipeline_args")
        if isinstance(pipeline_args, dict):
            pipeline_args["run_dir_name"] = effective_run_id
        run_local_bars_dir = _default_run_bars_dir(effective_dir)
        if run_local_bars_dir.exists():
            manifest["bars_dir"] = str(run_local_bars_dir)
        master_bars = manifest.get("master_bars")
        if isinstance(master_bars, dict):
            for freq, meta in master_bars.items():
                if not isinstance(meta, dict):
                    continue
                if freq in MASTER_BAR_FREQS and (run_local_bars_dir / f"{freq}.csv").exists():
                    meta["path"] = str(run_local_bars_dir / f"{freq}.csv")
        manifest["artifacts"] = _build_artifact_inventory(effective_dir)
    _atomic_json_write(manifest_path, manifest)

    return {
        "run_id": effective_run_id,
        "label": manifest["label"],
        "renamed_dir": effective_run_id != run_id,
        "run_dir": str(effective_dir.relative_to(BASE_DIR)),
    }


def _rename_prefixed_run_files(run_dir: Path, old_prefix: str, new_prefix: str) -> None:
    """Rename files inside *run_dir* whose name starts with *old_prefix*."""
    for item in sorted(run_dir.iterdir()):
        if not item.is_file():
            continue
        if item.name.startswith(old_prefix):
            new_name = new_prefix + item.name[len(old_prefix):]
            target = run_dir / new_name
            if not target.exists():
                item.rename(target)


def set_active_run(run_id: str) -> Dict[str, Any]:
    """Set a run as the active/default run via latest.json."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    _update_latest_pointer(run_id)
    return {"run_id": run_id, "active": True}


def delete_pipeline_run(run_id: str) -> Dict[str, Any]:
    """Delete a pipeline run directory. Prevents deleting the active run."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    latest_run_id = get_latest_run_id()
    if run_id == latest_run_id:
        raise ValueError(
            "Cannot delete the currently active run. "
            "Set a different run as active first."
        )
    shutil.rmtree(run_dir)
    return {"run_id": run_id, "deleted": True}


def toggle_pin_run(run_id: str) -> Dict[str, Any]:
    """Toggle the pinned state of a run."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    manifest_path = run_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}
    manifest["pinned"] = not manifest.get("pinned", False)
    _atomic_json_write(manifest_path, manifest)
    return {"run_id": run_id, "pinned": manifest["pinned"]}


def get_runs_storage_summary() -> Dict[str, Any]:
    """Get total storage used by all runs."""
    if not RUNS_DIR.exists():
        return {"total_runs": 0, "total_size_bytes": 0, "total_size_mb": 0}
    total = 0
    count = 0
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        count += 1
        for item in run_dir.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    return {
        "total_runs": count,
        "total_size_bytes": total,
        "total_size_mb": round(total / (1024 * 1024), 2),
    }


def _build_features_command(
    payload: PolymarketHistoryRunRequest,
    bars_dir: Path,
    dim_market_path: Path,
    out_dir: Path,
) -> tuple[List[str], Dict[str, str]]:
    if not FEATURES_SCRIPT_PATH.exists():
        raise RuntimeError(f"Features script not found: {FEATURES_SCRIPT_PATH}")

    cmd: List[str] = [sys.executable, str(FEATURES_SCRIPT_PATH)]
    cmd.extend(["--bars-dir", str(bars_dir)])
    cmd.extend(["--dim-market", str(dim_market_path)])
    cmd.extend(["--out-dir", str(out_dir)])

    prn_path: Optional[Path] = None
    local_prn = find_run_local_prn_training_file(out_dir)
    if local_prn and local_prn.exists():
        prn_path = local_prn
    elif payload.prn_dataset:
        prn_path = _resolve_project_path(payload.prn_dataset)

    if prn_path is not None and prn_path.exists():
        cmd.extend(["--prn-dataset", str(prn_path)])

    if payload.start_date:
        cmd.extend(["--start-date", payload.start_date])
    if payload.end_date:
        cmd.extend(["--end-date", payload.end_date])
    if payload.skip_subgraph_labels:
        cmd.append("--skip-subgraph-labels")

    env = {**os.environ}
    existing = env.get("PYTHONPATH")
    root = str(BASE_DIR)
    if existing:
        env["PYTHONPATH"] = os.pathsep.join([existing, root])
    else:
        env["PYTHONPATH"] = root

    return cmd, env


def _build_run_prn_refresh_command(
    payload: PolymarketHistoryRunRequest,
    run_dir: Path,
) -> tuple[List[str], Dict[str, str]]:
    if not RUN_LOCAL_PRN_REFRESH_SCRIPT_PATH.exists():
        raise RuntimeError(f"Run-local pRN refresh script not found: {RUN_LOCAL_PRN_REFRESH_SCRIPT_PATH}")

    cmd: List[str] = [sys.executable, str(RUN_LOCAL_PRN_REFRESH_SCRIPT_PATH), "--run-id", run_dir.name]

    env = {**os.environ}
    existing = env.get("PYTHONPATH")
    root = str(BASE_DIR)
    if existing:
        env["PYTHONPATH"] = os.pathsep.join([existing, root])
    else:
        env["PYTHONPATH"] = root
    return cmd, env


def _build_features(
    payload: PolymarketHistoryRunRequest,
    bars_dir: Path,
    dim_market_path: Path,
    out_dir: Path,
) -> tuple[Optional[Path], Optional[Path], str]:
    """Build decision features using script 8."""
    cmd, env = _build_features_command(payload, bars_dir, dim_market_path, out_dir)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    if result.returncode != 0:
        return None, None, result.stderr or result.stdout

    # Check which format was created
    parquet_path = out_dir / "decision_features.parquet"
    csv_path = out_dir / LEGACY_DECISION_FEATURES_CSV
    manifest_path = out_dir / "feature_manifest.json"

    # If parquet exists but CSV doesn't, create CSV from parquet for preview
    if parquet_path.exists() and not csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
        except Exception:
            pass  # If conversion fails, just use parquet

    features_path = csv_path if csv_path.exists() else parquet_path if parquet_path.exists() else None

    return (
        features_path,
        manifest_path if manifest_path.exists() else None,
        result.stdout,
    )


def run_polymarket_history(
    payload: PolymarketHistoryRunRequest,
) -> PolymarketHistoryRunResponse:
    cmd: List[str] = []
    out_dir = DEFAULT_OUT_DIR
    temp_paths: List[Path] = []
    preexisting_run_dirs: set[Path] = set()
    run_id: Optional[str] = None
    run_dir: Optional[Path] = None
    features_built = False
    features_path: Optional[Path] = None
    features_manifest_path: Optional[Path] = None

    try:
        cmd, env, out_dir, temp_paths = _build_history_command(payload)
        preexisting_run_dirs = _list_existing_run_dirs(out_dir)

        start = time.monotonic()
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        duration_s = round(time.monotonic() - start, 3)

        ok = result.returncode == 0
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        run_id = _parse_run_id(result.stdout)
        run_dir = _resolve_run_dir_candidate(
            out_dir,
            run_id,
            requested_run_dir_name=payload.run_dir_name,
            preexisting_run_dirs=preexisting_run_dirs,
        )
        files = _list_run_file_names(run_dir) if run_dir and run_dir.exists() else []
        manifest = _load_run_manifest(run_dir) if run_dir else {}

        if ok and run_dir and run_dir.exists():
            try:
                _copy_dim_market_to_run(run_dir)
            except Exception:
                pass
            try:
                prn_cmd, prn_env = _build_run_prn_refresh_command(payload, run_dir)
                prn_result = subprocess.run(prn_cmd, capture_output=True, text=True, check=False, env=prn_env)
                stdout = f"{stdout}\n{prn_result.stdout or ''}".strip()
                stderr = f"{stderr}\n{prn_result.stderr or ''}".strip()
                if prn_result.returncode != 0:
                    ok = False
                    if not (prn_result.stderr or "").strip():
                        stderr = f"{stderr}\n[pRN] Exact run-local pRN refresh failed.".strip()
            except Exception as exc:
                ok = False
                stderr = f"{stderr}\n[Run pRN] {exc}".strip()
            try:
                _rename_run_csv_files(run_dir)
            except Exception:
                pass
            files = _list_run_file_names(run_dir)
            manifest = _load_run_manifest(run_dir)

            if ok and payload.build_features and _prn_phase_complete(run_dir):
                try:
                    _set_run_pending_state(
                        run_dir,
                        build_features_requested=True,
                        pipeline_args=_pipeline_args_from_payload(payload),
                        pending_phase="features",
                    )
                except Exception:
                    pass
                try:
                    bars_dir = _resolve_run_bars_dir(run_dir, manifest, must_exist=False)
                    dim_market_path = _find_dim_market_for_run(run_dir, manifest)
                    features_path, features_manifest_path, features_stdout = _build_features(
                        payload,
                        bars_dir,
                        dim_market_path,
                        run_dir,
                    )
                    if features_stdout:
                        stdout = f"{stdout}\n{features_stdout}".strip()
                    features_built = features_path is not None
                    if features_built:
                        _update_features_manifest(run_dir, features_built=True)
                    else:
                        ok = False
                        _set_run_pending_state(
                            run_dir,
                            build_features_requested=True,
                            pipeline_args=_pipeline_args_from_payload(payload),
                            error_summary="Decision features are pending for this run.",
                            pending_phase="features",
                            duration_s=duration_s,
                            finished=True,
                        )
                except Exception as exc:
                    ok = False
                    stderr = f"{stderr}\n[Features] {exc}".strip()
                    try:
                        _set_run_pending_state(
                            run_dir,
                            build_features_requested=True,
                            pipeline_args=_pipeline_args_from_payload(payload),
                            error_summary=str(exc),
                            pending_phase="features",
                            duration_s=duration_s,
                            finished=True,
                        )
                    except Exception:
                        pass
                files = _list_run_file_names(run_dir)
                manifest = _load_run_manifest(run_dir)

        response = PolymarketHistoryRunResponse(
            ok=ok,
            run_id=run_id,
            out_dir=str(out_dir.relative_to(BASE_DIR)),
            run_dir=str(run_dir.relative_to(BASE_DIR)) if run_dir else None,
            files=files,
            stdout=stdout,
            stderr=stderr,
            duration_s=duration_s,
            command=cmd,
            features_built=features_built,
            features_path=(
                str(features_path.relative_to(run_dir))
                if features_path is not None and run_dir is not None
                else None
            ),
            features_manifest_path=(
                str(features_manifest_path.relative_to(run_dir))
                if features_manifest_path is not None and run_dir is not None
                else None
            ),
            master_bar_artifacts=(
                _build_run_master_bar_artifacts(run_dir, manifest)
                if _run_artifacts_accessible(run_dir, manifest, _runtime_payload_for_run_dir(run_dir))
                else []
            ),
            artifact_groups=(
                _build_run_artifact_groups(run_dir, manifest)
                if _run_artifacts_accessible(run_dir, manifest, _runtime_payload_for_run_dir(run_dir))
                else []
            ),
            shared_artifacts=[],
            quality_summary=_load_quality_summary_for_run_dir(run_dir),
        )

        preserve_pending_run = bool(
            run_dir
            and run_dir.exists()
            and payload.build_features
            and _run_features_pending(run_dir, manifest, _runtime_payload_for_run_dir(run_dir))
        )

        if not ok and run_dir and not preserve_pending_run:
            deleted = _delete_polymarket_run_dir(run_dir, out_dir=out_dir, run_id=run_id)
            if deleted:
                _clear_deleted_run_artifacts(response)

        return response
    except Exception:
        if run_dir:
            _delete_polymarket_run_dir(run_dir, out_dir=out_dir, run_id=run_id)
        raise
    finally:
        _cleanup_transient_paths(temp_paths)


class PolymarketHistoryJob:
    def __init__(self, job_id: str, payload: PolymarketHistoryRunRequest) -> None:
        self.job_id = job_id
        self.payload = payload.model_copy(deep=True)
        self.status = "queued"
        self.phase: Optional[str] = None
        self.result: Optional[PolymarketHistoryRunResponse] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._prn_process: Optional[subprocess.Popen[str]] = None
        self._features_process: Optional[subprocess.Popen[str]] = None
        self._process_handle: Optional[ManagedProcessHandle] = None
        self._prn_process_handle: Optional[ManagedProcessHandle] = None
        self._features_process_handle: Optional[ManagedProcessHandle] = None
        self._cancel_requested = False
        self._history_progress = JobProgressTracker()
        self._features_progress = JobProgressTracker()
        self._telemetry = None
        self._resume_attempts = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        if self.status in {"finished", "failed", "cancelled"}:
            return
        self._cancel_requested = True
        for proc, handle in (
            (self._process, self._process_handle),
            (self._prn_process, self._prn_process_handle),
            (self._features_process, self._features_process_handle),
        ):
            if proc and proc.poll() is None and handle is not None:
                terminate_managed_process(handle, term_timeout_s=5.0, kill_timeout_s=5.0)
                continue
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def to_status(self) -> PolymarketHistoryJobStatus:
        return PolymarketHistoryJobStatus(
            job_id=self.job_id,
            status=self.status,
            phase=self.phase,
            progress=self._history_progress.snapshot(),
            features_progress=self._features_progress.snapshot(),
            telemetry=self._telemetry,
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    def _seed_history_progress_from_run_dir(self, run_dir: Optional[Path]) -> None:
        total, completed, failed = _history_resume_completed_counts(run_dir)
        if total > 0:
            self._history_progress.seed(total=total, completed=completed, failed=failed)

    def _update_history_progress(self, line: str) -> None:
        match = _HISTORY_COMPLETE_RE.search(line)
        if not match:
            return
        total = int(match.group("total"))
        job_id = match.group("job_id")
        status = match.group("status").lower()
        failed = status not in {"ok", "success", "completed"}
        self._history_progress.set_total(total)
        self._history_progress.mark_completed(job_id, failed=failed)

    def _update_features_progress(self, line: str) -> None:
        match = _FEATURE_PROGRESS_RE.search(line)
        if not match:
            return
        total = int(match.group("total"))
        step_id = match.group("step")
        self._features_progress.set_total(total)
        self._features_progress.mark_completed(step_id)

    def _should_retry_phase(self, *, phase: str, run_dir: Optional[Path]) -> bool:
        if self._cancel_requested or run_dir is None or not run_dir.exists():
            return False
        if self._resume_attempts >= MAX_AUTOMATIC_RESUME_ATTEMPTS:
            return False
        if phase == "history":
            return not _history_manifest_exists(run_dir)
        if phase == "prn":
            return _history_manifest_exists(run_dir) and not _prn_phase_complete(run_dir)
        if phase == "features":
            return _prn_phase_complete(run_dir) and not _features_phase_complete(run_dir)
        return False

    def _run(self) -> None:
        if self.started_at is None:
            self.started_at = datetime.now(timezone.utc)
        if self._cancel_requested:
            self.status = "cancelled"
            self.error = "Weekly history run cancelled."
            self.finished_at = datetime.now(timezone.utc)
            return
        self.status = "running"
        self.phase = "history"
        out_dir = DEFAULT_OUT_DIR
        temp_paths: List[Path] = []
        preexisting_run_dirs: set[Path] = set()
        run_id: Optional[str] = None
        run_dir: Optional[Path] = None
        files: List[str] = []
        features_built = False
        features_path: Optional[Path] = None
        features_manifest_path: Optional[Path] = None
        try:
            if not self.payload.run_dir_name:
                self.payload.run_dir_name = _default_polymarket_run_id()
            self.payload.resume_existing = True
            cmd, env, out_dir, temp_paths = _build_history_command(self.payload)
            preexisting_run_dirs = _list_existing_run_dirs(out_dir)
            run_id = self.payload.run_dir_name
            run_dir = out_dir / "runs" / run_id if run_id else None
            start = time.monotonic()

            stdout_lines: List[str] = []
            stderr_lines: List[str] = []
            self._seed_history_progress_from_run_dir(run_dir)
            resume_phase = _next_resume_phase(run_dir, build_features=self.payload.build_features)

            def snapshot_result(ok: bool) -> None:
                manifest = _load_run_manifest(run_dir) if run_dir else {}
                self.result = PolymarketHistoryRunResponse(
                    ok=ok,
                    run_id=run_id,
                    out_dir=str(out_dir.relative_to(BASE_DIR)),
                    run_dir=str(run_dir.relative_to(BASE_DIR)) if run_dir else None,
                    files=list(files),
                    stdout=''.join(stdout_lines),
                    stderr=''.join(stderr_lines),
                    duration_s=round(time.monotonic() - start, 3),
                    command=cmd,
                    features_built=features_built,
                    features_path=(
                        str(features_path.relative_to(run_dir))
                        if features_path and run_dir
                        else None
                    ),
                    features_manifest_path=(
                        str(features_manifest_path.relative_to(run_dir))
                        if features_manifest_path and run_dir
                        else None
                    ),
                    master_bar_artifacts=_build_run_master_bar_artifacts(run_dir, manifest),
                    artifact_groups=_build_run_artifact_groups(run_dir, manifest),
                    shared_artifacts=[],
                    quality_summary=_load_quality_summary_for_run_dir(run_dir),
                )
            ok = False
            if resume_phase != "history" and run_dir and run_dir.exists():
                ok = True
                files = _list_run_file_names(run_dir)
                stdout_lines.append(
                    f"\n[Resume] Skipping history; resuming from {resume_phase} phase.\n"
                )
                snapshot_result(ok=False)
            else:
                while True:
                    self.phase = "history"
                    handle = spawn_managed_process(
                        cmd,
                        job_id=self.job_id,
                        service="polymarket_history",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        env=env,
                    )
                    proc = handle.process
                    if proc is None:
                        raise RuntimeError("Failed to start polymarket history process.")
                    self._process = proc
                    self._process_handle = handle
                    if run_dir is not None:
                        _write_polymarket_runtime_file(
                            handle,
                            payload=self.payload,
                            run_dir=run_dir,
                            phase="history",
                        )

                    def read_stdout() -> None:
                        nonlocal run_id
                        if proc.stdout:
                            for line in iter(proc.stdout.readline, ''):
                                if line:
                                    stdout_lines.append(line)
                                    self._update_history_progress(line)
                                    parsed = _parse_run_id(line) or _parse_run_id(''.join(stdout_lines))
                                    if parsed:
                                        run_id = parsed
                                    snapshot_result(ok=False)
                            proc.stdout.close()

                    def read_stderr() -> None:
                        if proc.stderr:
                            for line in iter(proc.stderr.readline, ''):
                                if line:
                                    stderr_lines.append(line)
                                    snapshot_result(ok=False)
                            proc.stderr.close()

                    stdout_thread = threading.Thread(target=read_stdout, daemon=True)
                    stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                    stdout_thread.start()
                    stderr_thread.start()

                    proc.wait()
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)

                    stdout = ''.join(stdout_lines)
                    run_id = _parse_run_id(stdout) or run_id
                    run_dir = _resolve_run_dir_candidate(
                        out_dir,
                        run_id,
                        requested_run_dir_name=self.payload.run_dir_name,
                        preexisting_run_dirs=preexisting_run_dirs,
                    )
                    self._seed_history_progress_from_run_dir(run_dir)
                    files = _list_run_file_names(run_dir) if run_dir and run_dir.exists() else []
                    ok = proc.returncode == 0 and not self._cancel_requested
                    self._process = None
                    self._process_handle = None
                    if ok or not self._should_retry_phase(phase="history", run_dir=run_dir):
                        break
                    self._resume_attempts += 1
                    stdout_lines.append(
                        "\n[Resume] History phase interrupted unexpectedly. "
                        f"Retrying from checkpoint (attempt {self._resume_attempts}/{MAX_AUTOMATIC_RESUME_ATTEMPTS}).\n"
                    )
                    snapshot_result(ok=False)

            if ok and run_dir and resume_phase in {"history", "prn"}:
                try:
                    _copy_dim_market_to_run(run_dir)
                except Exception:
                    pass
                self.phase = "prn"
                stdout_lines.append("\n[pRN] Refreshing exact run-local pRN...\n")
                snapshot_result(ok=False)

                if self._cancel_requested:
                    stdout_lines.append("[pRN] Cancel requested before run-local pRN refresh.\n")
                    ok = False
                else:
                    prn_cmd, prn_env = _build_run_prn_refresh_command(self.payload, run_dir)
                    while True:
                        prn_handle = spawn_managed_process(
                            prn_cmd,
                            job_id=self.job_id,
                            service="polymarket_run_prn",
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                            env=prn_env,
                        )
                        prn_proc = prn_handle.process
                        if prn_proc is None:
                            raise RuntimeError("Failed to start run-local pRN refresh process.")
                        self._prn_process = prn_proc
                        self._prn_process_handle = prn_handle
                        _write_polymarket_runtime_file(
                            prn_handle,
                            payload=self.payload,
                            run_dir=run_dir,
                            phase="prn",
                        )

                        def read_prn_stdout() -> None:
                            if prn_proc.stdout:
                                for line in iter(prn_proc.stdout.readline, ''):
                                    if line:
                                        stdout_lines.append(line)
                                        telemetry = parse_quality_telemetry_line(line)
                                        if telemetry is not None:
                                            self._telemetry = telemetry
                                            self.phase = "quality"
                                        snapshot_result(ok=False)
                                prn_proc.stdout.close()

                        def read_prn_stderr() -> None:
                            if prn_proc.stderr:
                                for line in iter(prn_proc.stderr.readline, ''):
                                    if line:
                                        stderr_lines.append(line)
                                        snapshot_result(ok=False)
                                prn_proc.stderr.close()

                        prn_stdout_thread = threading.Thread(target=read_prn_stdout, daemon=True)
                        prn_stderr_thread = threading.Thread(target=read_prn_stderr, daemon=True)
                        prn_stdout_thread.start()
                        prn_stderr_thread.start()

                        prn_proc.wait()
                        prn_stdout_thread.join(timeout=1)
                        prn_stderr_thread.join(timeout=1)

                        self._prn_process = None
                        self._prn_process_handle = None
                        if self._cancel_requested:
                            stdout_lines.append("\n[pRN] Run-local pRN refresh cancelled.\n")
                            ok = False
                            break
                        if prn_proc.returncode == 0:
                            stdout_lines.append("\n[pRN] Exact run-local pRN refresh complete.\n")
                            files = _list_run_file_names(run_dir)
                            if self._telemetry is not None:
                                self.phase = "quality"
                            ok = True
                            break
                        if not self._should_retry_phase(phase="prn", run_dir=run_dir):
                            stdout_lines.append("\n[pRN] Exact run-local pRN refresh failed.\n")
                            if not stderr_lines:
                                stderr_lines.append("[pRN] Exact run-local pRN refresh failed.\n")
                            ok = False
                            break
                        self._resume_attempts += 1
                        stdout_lines.append(
                            "\n[Resume] pRN phase interrupted unexpectedly. "
                            f"Retrying exact refresh (attempt {self._resume_attempts}/{MAX_AUTOMATIC_RESUME_ATTEMPTS}).\n"
                        )
                        snapshot_result(ok=False)

            if ok and self.payload.build_features and run_dir and _prn_phase_complete(run_dir):
                try:
                    _set_run_pending_state(
                        run_dir,
                        build_features_requested=True,
                        pipeline_args=_pipeline_args_from_payload(self.payload),
                        pending_phase="features",
                    )
                except Exception:
                    pass

            if ok and self.payload.build_features and run_dir and resume_phase in {"history", "prn", "features"}:
                self.phase = "features"
                stdout_lines.append("\n[Features] Building decision features...\n")
                snapshot_result(ok=False)

                if self._cancel_requested:
                    stdout_lines.append("[Features] Cancel requested before feature build.\n")
                else:
                    bars_dir_arg = (
                        _resolve_run_bars_dir(run_dir, _load_run_manifest(run_dir), must_exist=False)
                        if run_dir is not None
                        else Path(self.payload.bars_dir)
                    )
                    dim_market_arg = (
                        Path(self.payload.dim_market_out)
                        if self.payload.dim_market_out
                        else BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market_weekly.csv"
                    )
                    features_cmd, features_env = _build_features_command(
                        self.payload,
                        bars_dir_arg,
                        dim_market_arg,
                        run_dir,
                    )
                    while True:
                        features_handle = spawn_managed_process(
                            features_cmd,
                            job_id=self.job_id,
                            service="polymarket_features",
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=1,
                            env=features_env,
                        )
                        features_proc = features_handle.process
                        if features_proc is None:
                            raise RuntimeError("Failed to start feature build process.")
                        self._features_process = features_proc
                        self._features_process_handle = features_handle
                        _write_polymarket_runtime_file(
                            features_handle,
                            payload=self.payload,
                            run_dir=run_dir,
                            phase="features",
                        )

                        def read_features_stdout() -> None:
                            if features_proc.stdout:
                                for line in iter(features_proc.stdout.readline, ''):
                                    if line:
                                        stdout_lines.append(line)
                                        self._update_features_progress(line)
                                        snapshot_result(ok=False)
                                features_proc.stdout.close()

                        def read_features_stderr() -> None:
                            if features_proc.stderr:
                                for line in iter(features_proc.stderr.readline, ''):
                                    if line:
                                        stderr_lines.append(line)
                                        snapshot_result(ok=False)
                                features_proc.stderr.close()

                        features_stdout_thread = threading.Thread(
                            target=read_features_stdout,
                            daemon=True,
                        )
                        features_stderr_thread = threading.Thread(
                            target=read_features_stderr,
                            daemon=True,
                        )
                        features_stdout_thread.start()
                        features_stderr_thread.start()

                        features_proc.wait()
                        features_stdout_thread.join(timeout=1)
                        features_stderr_thread.join(timeout=1)
                        self._features_process = None
                        self._features_process_handle = None

                        if self._cancel_requested:
                            stdout_lines.append("\n[Features] Feature build cancelled.\n")
                            self._features_progress.mark_failed()
                            ok = False
                            break
                        if features_proc.returncode == 0:
                            parquet_path = run_dir / "decision_features.parquet"
                            csv_path = run_dir / LEGACY_DECISION_FEATURES_CSV
                            manifest_path = run_dir / "feature_manifest.json"

                            if parquet_path.exists() and not csv_path.exists():
                                try:
                                    import pandas as pd
                                    df = pd.read_parquet(parquet_path)
                                    df.to_csv(csv_path, index=False)
                                except Exception:
                                    pass

                            features_path = _resolve_features_output_path(run_dir)
                            features_manifest_path = (
                                manifest_path if manifest_path.exists() else None
                            )

                            if features_path:
                                features_built = True
                                stdout_lines.append(
                                    f"\n[Features] Built successfully: {features_path.name}\n"
                                )
                                files.append(features_path.name)
                                if features_manifest_path:
                                    files.append(features_manifest_path.name)
                            else:
                                stdout_lines.append(
                                    "\n[Features] Completed but output files not found.\n"
                                )
                            ok = True
                            break
                        self._features_progress.mark_failed()
                        if not self._should_retry_phase(phase="features", run_dir=run_dir):
                            stdout_lines.append("\n[Features] Failed to build features.\n")
                            ok = False
                            break
                        self._resume_attempts += 1
                        stdout_lines.append(
                            "\n[Resume] Features phase interrupted unexpectedly. "
                            f"Retrying feature build (attempt {self._resume_attempts}/{MAX_AUTOMATIC_RESUME_ATTEMPTS}).\n"
                        )
                        snapshot_result(ok=False)

            if run_dir and run_dir.exists():
                if ok and run_id:
                    try:
                        _copy_dim_market_to_run(run_dir)
                    except Exception:
                        pass
                try:
                    _rename_run_csv_files(run_dir)
                except Exception:
                    pass
                files = _list_run_file_names(run_dir)
                if features_manifest_path and not features_manifest_path.exists():
                    features_manifest_path = None
                features_path = _resolve_features_output_path(run_dir)

            duration_s = round(time.monotonic() - start, 3)
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)
            self.phase = "finalizing"
            preserve_pending_run = bool(
                run_dir
                and run_dir.exists()
                and not self._cancel_requested
                and self.payload.build_features
                and _run_features_pending(run_dir, _load_run_manifest(run_dir), {})
            )

            self.result = PolymarketHistoryRunResponse(
                ok=ok,
                run_id=run_id,
                out_dir=str(out_dir.relative_to(BASE_DIR)),
                run_dir=str(run_dir.relative_to(BASE_DIR)) if run_dir else None,
                files=files,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration_s,
                command=cmd,
                features_built=features_built,
                features_path=(
                    str(features_path.relative_to(run_dir))
                    if features_path and run_dir
                    else None
                ),
                features_manifest_path=(
                    str(features_manifest_path.relative_to(run_dir))
                    if features_manifest_path and run_dir
                    else None
                ),
                master_bar_artifacts=_build_run_master_bar_artifacts(
                    run_dir,
                    _load_run_manifest(run_dir) if run_dir else {},
                ),
                artifact_groups=_build_run_artifact_groups(
                    run_dir,
                    _load_run_manifest(run_dir) if run_dir else {},
                ),
                shared_artifacts=[],
                quality_summary=_load_quality_summary_for_run_dir(run_dir),
            )

            if self._cancel_requested:
                self.status = "cancelled"
                self.error = "Weekly history run cancelled."
            elif ok:
                self.status = "finished"
            else:
                self.status = "failed"
                self.error = (stderr or "").strip() or "Weekly history run failed."

            # --- Phase 0+1: post-run hooks (enhance manifest, latest pointer) ---
            if run_dir and run_dir.exists():
                try:
                    if preserve_pending_run:
                        pending_error = (
                            self.error
                            or (stderr or "").strip()
                            or "Decision features are pending for this run."
                        )
                        _set_run_pending_state(
                            run_dir,
                            build_features_requested=True,
                            pipeline_args=_pipeline_args_from_payload(self.payload),
                            error_summary=pending_error,
                            pending_phase="features",
                            duration_s=duration_s,
                            finished=True,
                        )
                    else:
                        final_status = "cancelled" if self._cancel_requested else ("success" if ok else "failed")
                        _enhance_manifest(
                            run_dir,
                            status=final_status,
                            duration_s=duration_s,
                            pipeline_args=_pipeline_args_from_payload(self.payload),
                            error_summary=self.error if not ok else None,
                            features_built=features_built,
                        )
                    if ok and run_id:
                        _update_latest_pointer(run_id)
                except Exception:
                    pass  # never let post-run hooks break the job status
                try:
                    clear_runtime_file(run_dir)
                except Exception:
                    pass

            # Cancelled and failed runs are auto-pruned so they never appear in history directories.
            if self.status in {"failed", "cancelled"} and run_dir and not preserve_pending_run:
                deleted = _delete_polymarket_run_dir(run_dir, out_dir=out_dir, run_id=run_id)
                if deleted:
                    _clear_deleted_run_artifacts(self.result)
                _delete_additional_created_run_dirs(
                    out_dir=out_dir,
                    preexisting_run_dirs=preexisting_run_dirs,
                    primary_run_dir=run_dir,
                )
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
            if run_dir:
                try:
                    clear_runtime_file(run_dir)
                except Exception:
                    pass
                preserve_pending_run = bool(
                    run_dir.exists()
                    and self.payload.build_features
                    and _run_features_pending(run_dir, _load_run_manifest(run_dir), {})
                )
                if preserve_pending_run:
                    try:
                        _set_run_pending_state(
                            run_dir,
                            build_features_requested=True,
                            pipeline_args=_pipeline_args_from_payload(self.payload),
                            error_summary=self.error,
                            pending_phase="features",
                            finished=True,
                        )
                    except Exception:
                        pass
                else:
                    deleted = _delete_polymarket_run_dir(run_dir, out_dir=out_dir, run_id=run_id)
                    if deleted:
                        _clear_deleted_run_artifacts(self.result)
                    _delete_additional_created_run_dirs(
                        out_dir=out_dir,
                        preexisting_run_dirs=preexisting_run_dirs,
                        primary_run_dir=run_dir,
                    )
        finally:
            _cleanup_transient_paths(temp_paths)
            self._process = None
            self._prn_process = None
            self._features_process = None
            self._process_handle = None
            self._prn_process_handle = None
            self._features_process_handle = None
            self.finished_at = datetime.now(timezone.utc)


class PolymarketHistoryJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, PolymarketHistoryJob] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: PolymarketHistoryRunRequest, *, job_id: Optional[str] = None) -> str:
        job_id = job_id or uuid4().hex
        job = PolymarketHistoryJob(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> PolymarketHistoryJobStatus:
        try:
            job = self._get_job(job_id)
        except KeyError:
            runtime = self._recover_or_runtime_status(job_id)
            if runtime is not None:
                return runtime
            raise
        return job.to_status()

    def list_jobs(self) -> List[PolymarketHistoryJobStatus]:
        with self._lock:
            statuses = [job.to_status() for job in self._jobs.values()]
        seen_job_ids = {status.job_id for status in statuses}
        for run_dir, payload in _iter_polymarket_runtime_dirs():
            runtime_job_id = payload.get("job_id")
            if runtime_job_id is None:
                continue
            runtime_job_id_str = str(runtime_job_id)
            if runtime_job_id_str in seen_job_ids:
                continue
            runtime_status = self._recover_or_runtime_status(runtime_job_id_str)
            if runtime_status is None:
                continue
            statuses.append(runtime_status)
            seen_job_ids.add(runtime_job_id_str)
        return statuses

    def cancel_job(self, job_id: str) -> PolymarketHistoryJobStatus:
        try:
            job = self._get_job(job_id)
            job.cancel()
            return job.to_status()
        except KeyError:
            runtime = _find_polymarket_runtime_job(job_id)
            if runtime is None:
                raise
            run_dir, payload = runtime
            handle = managed_handle_from_runtime_payload(run_dir, payload)
            if handle is not None and is_process_alive(handle.pid):
                result = terminate_managed_process(handle, term_timeout_s=5.0, kill_timeout_s=5.0)
                if not result.ok:
                    raise RuntimeError(
                        f"Failed to cancel polymarket history job '{job_id}' via runtime fallback: {result.reason}"
                    )
            clear_runtime_file(run_dir)
            _delete_polymarket_run_dir(run_dir, out_dir=DEFAULT_OUT_DIR, run_id=run_dir.name)
            return _runtime_backed_polymarket_status(
                job_id,
                payload,
                run_dir,
                status="cancelled",
                finished_at=datetime.now(timezone.utc),
                error="Weekly history run cancelled.",
            )

    def _get_job(self, job_id: str) -> PolymarketHistoryJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def _recover_or_runtime_status(self, job_id: str) -> Optional[PolymarketHistoryJobStatus]:
        runtime = _find_polymarket_runtime_job(job_id)
        if runtime is None:
            return None
        run_dir, payload = runtime
        handle = managed_handle_from_runtime_payload(run_dir, payload)
        if handle is not None and is_process_alive(handle.pid):
            return _runtime_backed_polymarket_status(job_id, payload, run_dir, status="running")
        request_payload = payload.get("request_payload")
        if not isinstance(request_payload, dict):
            clear_runtime_file(run_dir)
            return None
        try:
            request = PolymarketHistoryRunRequest.model_validate(request_payload)
        except Exception:
            clear_runtime_file(run_dir)
            return None
        request.run_dir_name = run_dir.name
        request.resume_existing = True
        next_phase = _next_resume_phase(run_dir, build_features=request.build_features)
        if next_phase == "finalizing":
            clear_runtime_file(run_dir)
            return _runtime_backed_polymarket_status(
                job_id,
                payload,
                run_dir,
                status="finished",
                finished_at=datetime.now(timezone.utc),
            )
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing is not None:
                return existing.to_status()
            job = PolymarketHistoryJob(job_id, request)
            job.started_at = _parse_runtime_started_at(payload)
            self._jobs[job_id] = job
        job.start()
        return job.to_status()


POLYMARKET_HISTORY_JOB_MANAGER = PolymarketHistoryJobManager()


def start_polymarket_history_job(payload: PolymarketHistoryRunRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    return POLYMARKET_HISTORY_JOB_MANAGER.start_job(payload)


def get_polymarket_history_job(job_id: str) -> PolymarketHistoryJobStatus:
    return POLYMARKET_HISTORY_JOB_MANAGER.get_status(job_id)


def cancel_polymarket_history_job(job_id: str) -> PolymarketHistoryJobStatus:
    return POLYMARKET_HISTORY_JOB_MANAGER.cancel_job(job_id)


def get_csv_preview(
    job_id: str,
    filename: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    """Read and preview a CSV file from a completed job run directory."""
    job_status = POLYMARKET_HISTORY_JOB_MANAGER.get_status(job_id)

    if not job_status.result or not job_status.result.run_dir:
        raise FileNotFoundError("Job has no run directory yet.")

    run_dir = BASE_DIR / job_status.result.run_dir
    csv_path = _resolve_run_file_path(run_dir, filename)

    return _preview_csv_file(csv_path, filename=filename, limit=limit, mode=mode)


def get_run_csv_preview(
    run_id: str,
    filename: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    """Read and preview a CSV file from a persisted run directory."""
    csv_path = get_pipeline_run_file_path(run_id, filename)
    return _preview_csv_file(csv_path, filename=filename, limit=limit, mode=mode)


def get_run_quality_audit(run_id: str) -> PolymarketQualityAuditResponse:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return build_quality_audit_response(run_dir)
