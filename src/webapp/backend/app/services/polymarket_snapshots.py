from __future__ import annotations

import csv
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
from typing import Dict, List, Optional, Set, Tuple
from uuid import uuid4

from app.models.polymarket import (
    PolymarketSnapshotFileSummary,
    PolymarketSnapshotHistoryResponse,
    PolymarketSnapshotListResponse,
    PolymarketSnapshotLatestResponse,
    PolymarketSnapshotPreviewResponse,
    PolymarketSnapshotJobStatus,
    PolymarketSnapshotRunRequest,
    PolymarketSnapshotRunResponse,
    PolymarketSnapshotRunSummary,
)

def _unique_dirs(paths: List[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path.absolute())
    return unique


BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "3-polymarket-fetch-data-v1.0.py"
POLYMKT_DATA_DIRS = _unique_dirs(
    [
        BASE_DIR / "src" / "data" / "raw" / "polymarket",
        BASE_DIR / "data" / "raw" / "polymarket",
    ],
)
OUTPUT_DIR = POLYMKT_DATA_DIRS[0]
HISTORY_FILE_NAMES = (
    "polymarket-snapshot-history-weekly.csv",
    "polymarket-snapshot-history-1dte.csv",
)
LEGACY_HISTORY_FILE_NAMES = (
    "polymarket-snapshot-history.csv",
    "pPM-dataset-history-rolling.csv",
)
RUN_CONTRACT_TYPES = ("weekly", "1dte")


def _polymarket_base_dirs(existing_only: bool = True) -> List[Path]:
    if existing_only:
        dirs = [path for path in POLYMKT_DATA_DIRS if path.exists()]
    else:
        dirs = POLYMKT_DATA_DIRS[:]
    if not dirs:
        dirs = [POLYMKT_DATA_DIRS[0]]
    return dirs


def _polymarket_history_dirs(existing_only: bool = True) -> List[Path]:
    dirs: List[Path] = []
    for base in _polymarket_base_dirs(existing_only=existing_only):
        history_dir = base / "history"
        if history_dir.exists() or not existing_only:
            dirs.append(history_dir)
    if not dirs:
        dirs = [(POLYMKT_DATA_DIRS[0] / "history")]
    return dirs


def _polymarket_runs_dirs(existing_only: bool = True) -> List[Path]:
    dirs: List[Path] = []
    for base in _polymarket_base_dirs(existing_only=existing_only):
        runs_dir = base / "runs"
        if runs_dir.exists() or not existing_only:
            dirs.append(runs_dir)
        for contract_type in RUN_CONTRACT_TYPES:
            ct_dir = runs_dir / contract_type
            if ct_dir.exists() or not existing_only:
                dirs.append(ct_dir)
    return _unique_dirs(dirs)

def _find_polymarket_base_for_path(path: Path) -> Path:
    for base in POLYMKT_DATA_DIRS:
        try:
            path.relative_to(base)
            return base
        except ValueError:
            continue
    raise ValueError("File must be under src/data/raw/polymarket or data/raw/polymarket.")


def _history_csv_paths() -> List[Path]:
    paths: List[Path] = []
    for history_dir in _polymarket_history_dirs(existing_only=False):
        for name in HISTORY_FILE_NAMES:
            paths.append(history_dir / name)
        for name in LEGACY_HISTORY_FILE_NAMES:
            paths.append(history_dir / name)
    return paths

_SNAPSHOT_PATTERN = re.compile(r"__snapshot__([0-9]{4}-[0-9]{2}-[0-9]{2})__v")


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


def _parse_run_time(run_id: str) -> Optional[str]:
    candidate = run_id
    for prefix in ("weekly-", "1dte-"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix):]
            break
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H-%M-%SZ"):
        try:
            dt = datetime.strptime(candidate, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return None


def _file_summary(path: Path, kind: Optional[str] = None) -> PolymarketSnapshotFileSummary:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return PolymarketSnapshotFileSummary(
        name=path.name,
        path=str(path.relative_to(BASE_DIR)),
        size_bytes=path.stat().st_size,
        last_modified=mtime,
        kind=kind,
    )


def _snapshot_kind(name: str) -> Optional[str]:
    lowered = name.lower()
    if "ppm_dataset__snapshot" in lowered:
        return "dataset"
    if "ppm__snapshot" in lowered:
        return "pPM"
    if "prn__snapshot" in lowered:
        return "pRN"
    return None


def get_latest_snapshot() -> PolymarketSnapshotLatestResponse:
    date_map: Dict[str, List[Path]] = {}
    for base_dir in _polymarket_base_dirs():
        if not base_dir.exists():
            continue
        for item in base_dir.iterdir():
            if not item.is_file():
                continue
            match = _SNAPSHOT_PATTERN.search(item.name)
            if not match:
                continue
            day = match.group(1)
            date_map.setdefault(day, []).append(item)

    if not date_map:
        history_file = _latest_history_file()
        return PolymarketSnapshotLatestResponse(
            date=None,
            files=[],
            history_file=history_file,
        )

    latest_date = max(date_map.keys())
    files = [
        _file_summary(path, _snapshot_kind(path.name)) for path in sorted(date_map[latest_date])
    ]
    files.sort(key=lambda entry: entry.name)
    history_file = _latest_history_file()
    return PolymarketSnapshotLatestResponse(
        date=latest_date,
        files=files,
        history_file=history_file,
    )


def _latest_history_file() -> Optional[PolymarketSnapshotFileSummary]:
    candidates: List[Path] = []
    for history_dir in _polymarket_history_dirs():
        if not history_dir.exists():
            continue
        candidates.extend([item for item in history_dir.iterdir() if item.is_file()])
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: item.stat().st_mtime)
    return _file_summary(latest, "history")


def list_history_files() -> PolymarketSnapshotHistoryResponse:
    summaries: List[PolymarketSnapshotFileSummary] = []
    for history_dir in _polymarket_history_dirs():
        if not history_dir.exists():
            continue
        for item in history_dir.iterdir():
            if not item.is_file():
                continue
            summaries.append(_file_summary(item, "history"))
    summaries.sort(key=lambda entry: entry.last_modified, reverse=True)
    return PolymarketSnapshotHistoryResponse(files=summaries)


def _remove_history_rows_for_run(run_id: str, contract_type: Optional[str] = None) -> None:
    for history_path in _history_csv_paths():
        if not history_path.exists():
            continue

        with history_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames
            if not headers:
                continue
            kept_rows = []
            removed = False
            for row in reader:
                if row.get("run_id") != run_id:
                    kept_rows.append(row)
                    continue
                if contract_type and "run_contract_type" in row:
                    if row.get("run_contract_type") != contract_type:
                        kept_rows.append(row)
                        continue
                removed = True
                continue

        if not removed:
            continue

        temp_path = history_path.with_suffix(history_path.suffix + ".tmp")
        with temp_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(kept_rows)
        temp_path.replace(history_path)


def _resolve_polymarket_file(path_value: str) -> Path:
    path = _resolve_project_path(path_value)
    _find_polymarket_base_for_path(path)
    return path


def get_polymarket_snapshot_file(path_value: str) -> Path:
    path = _resolve_polymarket_file(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")
    return path


def _find_run_dirs(run_id: str) -> List[Path]:
    matches: List[Path] = []
    for runs_dir in _polymarket_runs_dirs(existing_only=True):
        candidate = runs_dir / run_id
        if candidate.exists() and candidate.is_dir():
            matches.append(candidate)
    return matches


def _find_run_dir(run_id: str) -> Path:
    matches = _find_run_dirs(run_id)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Run not found: {run_id}")


def _read_csv_head(path: Path, limit: int) -> Tuple[List[str], List[Dict[str, Optional[str]]]]:
    rows: List[Dict[str, Optional[str]]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append({key: row.get(key) for key in headers})
    return headers, rows


def _read_csv_tail(
    path: Path, limit: int
) -> Tuple[List[str], List[Dict[str, Optional[str]]], int]:
    buffer: deque[Dict[str, Optional[str]]] = deque(maxlen=limit)
    row_count = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for row in reader:
            row_count += 1
            buffer.append({key: row.get(key) for key in headers})
    return headers, list(buffer), row_count


def preview_snapshot_file(
    path_value: str, limit: int = 20, mode: str = "head"
) -> PolymarketSnapshotPreviewResponse:
    path = _resolve_polymarket_file(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")

    if limit < 1:
        limit = 1
    mode = mode.lower()
    if mode not in {"head", "tail"}:
        raise ValueError("mode must be 'head' or 'tail'")

    headers: List[str]
    rows: List[Dict[str, Optional[str]]]
    row_count: Optional[int]

    if mode == "tail":
        headers, rows, row_count = _read_csv_tail(path, limit)
    else:
        headers, rows = _read_csv_head(path, limit)
        row_count = None

    return PolymarketSnapshotPreviewResponse(
        file=_file_summary(path, _snapshot_kind(path.name)),
        headers=headers,
        rows=rows,
        row_count=row_count,
        mode=mode,
        limit=limit,
    )


def list_polymarket_runs(limit: int = 20) -> PolymarketSnapshotListResponse:
    summaries: List[PolymarketSnapshotRunSummary] = []
    seen: Set[Path] = set()
    for runs_dir in _polymarket_runs_dirs():
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if run_dir.name in RUN_CONTRACT_TYPES and run_dir.parent.name == "runs":
                continue
            resolved_run = run_dir.resolve()
            if resolved_run in seen:
                continue
            seen.add(resolved_run)
            run_id = run_dir.name
            files = sorted([item.name for item in run_dir.iterdir() if item.is_file()])
            size_bytes = 0
            for item in run_dir.iterdir():
                if item.is_file():
                    size_bytes += item.stat().st_size
            mtime = run_dir.stat().st_mtime
            summaries.append(
                PolymarketSnapshotRunSummary(
                    run_id=run_id,
                    run_time_utc=_parse_run_time(run_id),
                    run_dir=str(run_dir.relative_to(BASE_DIR)),
                    files=files,
                    file_count=len(files),
                    size_bytes=size_bytes,
                    last_modified=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
                )
            )

    summaries.sort(
        key=lambda item: item.run_time_utc or item.last_modified or "",
        reverse=True,
    )
    if limit < 1:
        limit = 1
    base_dir_for_response = _polymarket_base_dirs()[0]
    return PolymarketSnapshotListResponse(
        out_dir=str(base_dir_for_response.relative_to(BASE_DIR)),
        runs=summaries[:limit],
    )


def delete_polymarket_run(run_id: str) -> None:
    run_paths = _find_run_dirs(run_id)
    if not run_paths:
        raise FileNotFoundError(f"Run not found: {run_id}")

    contract_types: Set[Optional[str]] = set()
    for run_path in run_paths:
        try:
            parts = run_path.parts
            if "runs" in parts:
                idx = parts.index("runs")
                if idx + 1 < len(parts) and parts[idx + 1] in RUN_CONTRACT_TYPES:
                    contract_types.add(parts[idx + 1])
                else:
                    contract_types.add(None)
            else:
                contract_types.add(None)
        except Exception:
            contract_types.add(None)

        shutil.rmtree(run_path)

    if contract_types:
        for contract_type in contract_types:
            _remove_history_rows_for_run(run_id, contract_type=contract_type)


def run_polymarket_snapshot(
    payload: PolymarketSnapshotRunRequest,
) -> PolymarketSnapshotRunResponse:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Snapshot script not found at {SCRIPT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [sys.executable, str(SCRIPT_PATH), "--out-dir", str(OUTPUT_DIR)]

    tickers = _normalize_tickers(payload.tickers)
    if tickers:
        cmd.extend(["--tickers", ",".join(tickers)])

    if payload.tickers_csv:
        tickers_csv = _resolve_project_path(payload.tickers_csv)
        if not tickers_csv.exists():
            raise ValueError(f"tickers_csv not found: {tickers_csv}")
        cmd.extend(["--tickers-csv", str(tickers_csv)])

    if payload.slug_overrides:
        slug_overrides = _resolve_project_path(payload.slug_overrides)
        if not slug_overrides.exists():
            raise ValueError(f"slug_overrides not found: {slug_overrides}")
        cmd.extend(["--slug-overrides", str(slug_overrides)])

    if payload.risk_free_rate is not None:
        cmd.extend(["--risk-free-rate", str(payload.risk_free_rate)])

    if payload.tz:
        cmd.extend(["--tz", payload.tz])

    if payload.contract_type:
        cmd.extend(["--contract-type", payload.contract_type])

    if payload.contract_1dte:
        cmd.extend(["--contract-1dte", payload.contract_1dte])

    if payload.target_date:
        cmd.extend(["--target-date", payload.target_date])

    if payload.exchange_calendar:
        cmd.extend(["--exchange-calendar", payload.exchange_calendar])

    if payload.allow_nonlive:
        cmd.append("--allow-nonlive")

    if payload.dry_run:
        cmd.append("--dry-run")

    if payload.keep_nonexec:
        cmd.append("--keep-nonexec")

    env = {**os.environ}
    existing = env.get("PYTHONPATH")
    root = str(BASE_DIR)
    if existing:
        env["PYTHONPATH"] = os.pathsep.join([existing, root])
    else:
        env["PYTHONPATH"] = root

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    duration_s = round(time.monotonic() - start, 3)

    run_id = _parse_run_id(result.stdout)
    run_dir: Optional[Path] = None
    if run_id:
        if payload.contract_type in RUN_CONTRACT_TYPES:
            candidate = OUTPUT_DIR / "runs" / payload.contract_type / run_id
            if candidate.exists():
                run_dir = candidate
        if run_dir is None:
            legacy = OUTPUT_DIR / "runs" / run_id
            if legacy.exists():
                run_dir = legacy
        if run_dir is None:
            matches = _find_run_dirs(run_id)
            if matches:
                run_dir = matches[0]
    files: List[str] = []
    if run_dir and run_dir.exists():
        files = sorted([item.name for item in run_dir.iterdir() if item.is_file()])

    return PolymarketSnapshotRunResponse(
        ok=result.returncode == 0,
        run_id=run_id,
        out_dir=str(OUTPUT_DIR.relative_to(BASE_DIR)),
        run_dir=str(run_dir.relative_to(BASE_DIR)) if run_dir else None,
        files=files,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=duration_s,
        command=cmd,
    )


class PolymarketSnapshotJob:
    def __init__(self, job_id: str, payload: PolymarketSnapshotRunRequest) -> None:
        self.job_id = job_id
        self.payload = payload
        self.status = "queued"
        self.result: Optional[PolymarketSnapshotRunResponse] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def to_status(self) -> PolymarketSnapshotJobStatus:
        return PolymarketSnapshotJobStatus(
            job_id=self.job_id,
            status=self.status,
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    def _run(self) -> None:
        self.started_at = datetime.utcnow()
        self.status = "running"
        try:
            self.result = run_polymarket_snapshot(self.payload)
            if self.result.ok:
                self.status = "finished"
            else:
                self.status = "failed"
                self.error = (self.result.stderr or "").strip() or "Snapshot run failed."
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
        finally:
            self.finished_at = datetime.utcnow()


class PolymarketSnapshotJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, PolymarketSnapshotJob] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: PolymarketSnapshotRunRequest) -> str:
        job_id = uuid4().hex
        job = PolymarketSnapshotJob(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> PolymarketSnapshotJobStatus:
        job = self._get_job(job_id)
        return job.to_status()

    def list_jobs(self) -> List[PolymarketSnapshotJobStatus]:
        with self._lock:
            return [job.to_status() for job in self._jobs.values()]

    def _get_job(self, job_id: str) -> PolymarketSnapshotJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


POLYMARKET_JOB_MANAGER = PolymarketSnapshotJobManager()


def start_polymarket_snapshot_job(payload: PolymarketSnapshotRunRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    return POLYMARKET_JOB_MANAGER.start_job(payload)


def get_polymarket_snapshot_job(job_id: str) -> PolymarketSnapshotJobStatus:
    return POLYMARKET_JOB_MANAGER.get_status(job_id)


def list_polymarket_snapshot_jobs() -> List[PolymarketSnapshotJobStatus]:
    return POLYMARKET_JOB_MANAGER.list_jobs()
