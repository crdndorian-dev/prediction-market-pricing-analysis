from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.models.market_map import (
    MarketMapDeleteResponse,
    MarketMapFileSummary,
    MarketMapJobStatus,
    MarketMapPreviewResponse,
    MarketMapRunRequest,
    MarketMapRunResponse,
)

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "02-polymarket-market-map-v1.0.py"
SUBGRAPH_RUNS_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket" / "subgraph" / "runs"
DEFAULT_OUT_PATH = BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market.parquet"
DEFAULT_OVERRIDES_PATH = BASE_DIR / "config" / "polymarket_market_overrides.csv"
ENV_FILE = BASE_DIR / ".env"
ENV_SAMPLE_FILE = BASE_DIR / "config" / "polymarket_subgraph.env.sample"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _iso_from_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _resolve_project_path(path_value: str, *, must_exist: bool = True) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.absolute()
    try:
        path.relative_to(BASE_DIR)
    except ValueError as exc:
        raise ValueError("Path must be inside the project root.") from exc
    if must_exist and not path.exists():
        raise ValueError(f"Path not found: {path}")
    return path


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
        value = value.strip().strip("'").strip('"')
        if key:
            env[key] = value
    return env


def _load_subgraph_env() -> Dict[str, str]:
    if ENV_FILE.exists():
        return _load_env_file(ENV_FILE)
    if ENV_SAMPLE_FILE.exists():
        return _load_env_file(ENV_SAMPLE_FILE)
    return {}


def _has_markets_run() -> bool:
    if not SUBGRAPH_RUNS_DIR.exists():
        return False
    for run_dir in SUBGRAPH_RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        manifest = _load_json(run_dir / "manifest.json")
        if manifest and manifest.get("query_name") == "markets":
            return True
    return False


def _find_dim_market_output(out_value: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if out_value:
        out_path = _resolve_project_path(out_value, must_exist=False)
        candidates.append(out_path)
        if out_path.suffix.lower() == ".parquet":
            candidates.append(out_path.with_suffix(".csv"))
        elif out_path.suffix.lower() == ".csv":
            candidates.append(out_path.with_suffix(".parquet"))
    else:
        candidates.append(DEFAULT_OUT_PATH)
        candidates.append(DEFAULT_OUT_PATH.with_suffix(".csv"))

    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _file_summary(path: Path) -> MarketMapFileSummary:
    return MarketMapFileSummary(
        name=path.name,
        path=str(path.relative_to(BASE_DIR)),
        size_bytes=path.stat().st_size,
        last_modified=_iso_from_mtime(path.stat().st_mtime),
    )


def _coerce_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        import pandas as pd  # type: ignore

        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, (datetime,)):
        return value.isoformat()
    return str(value)


def _read_table_head(path: Path, limit: int) -> tuple[List[str], List[Dict[str, Optional[str]]]]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required to preview dim_market.") from exc

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if limit < 1:
        limit = 1
    df = df.head(limit)
    headers = list(df.columns)
    rows: List[Dict[str, Optional[str]]] = []
    for _, row in df.iterrows():
        rows.append({key: _coerce_value(row.get(key)) for key in headers})
    return headers, rows


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open() as handle:
        next(handle, None)
        for _ in handle:
            count += 1
    return count


def _count_parquet_rows(path: Path) -> Optional[int]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore

        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _count_table_rows(path: Path) -> Optional[int]:
    if path.suffix.lower() == ".parquet":
        return _count_parquet_rows(path)
    return _count_csv_rows(path)


def _parse_stdout_value(stdout: str, prefix: str) -> Optional[str]:
    pattern = rf"{re.escape(prefix)}\s*=\s*(.+)"
    for line in stdout.splitlines():
        match = re.search(pattern, line)
        if match:
            return match.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Run dim_market (synchronous)
# ---------------------------------------------------------------------------

def run_market_map(payload: MarketMapRunRequest) -> MarketMapRunResponse:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Market map script not found at {SCRIPT_PATH}")

    run_dir = _resolve_project_path(payload.run_dir, must_exist=True) if payload.run_dir else None

    # Validate run_dir if provided - must have manifest.json
    if run_dir:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                f"Invalid run directory: {run_dir}. "
                f"Expected manifest.json at {manifest_path}. "
                f"Use the full path to a specific run directory (e.g., src/data/raw/polymarket/subgraph/runs/markets-...) "
                f"or leave empty to auto-fetch markets."
            )

    # Allow auto-fetch from the subgraph when no run_dir is provided.
    # The script will pull markets live if needed.

    overrides = (
        _resolve_project_path(payload.overrides, must_exist=True)
        if payload.overrides
        else DEFAULT_OVERRIDES_PATH
    )
    prn_dataset = (
        _resolve_project_path(payload.prn_dataset, must_exist=True)
        if payload.prn_dataset
        else None
    )
    out_path = (
        _resolve_project_path(payload.out, must_exist=False) if payload.out else None
    )

    cmd: List[str] = [sys.executable, str(SCRIPT_PATH)]
    if run_dir:
        cmd.extend(["--run-dir", str(run_dir)])
    if payload.run_id:
        cmd.extend(["--run-id", payload.run_id])
    if overrides:
        cmd.extend(["--overrides", str(overrides)])
    if payload.tickers:
        cmd.extend(["--tickers", payload.tickers])
    if prn_dataset:
        cmd.extend(["--prn-dataset", str(prn_dataset)])
    if out_path:
        cmd.extend(["--out", str(out_path)])
    cmd.append("--strict" if payload.strict else "--no-strict")

    env = {**os.environ}
    env.update(_load_subgraph_env())
    existing = env.get("PYTHONPATH")
    root = str(BASE_DIR)
    env["PYTHONPATH"] = os.pathsep.join([existing, root]) if existing else root

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    duration_s = round(time.monotonic() - start, 3)

    output_value = _parse_stdout_value(result.stdout, "[dim_market] output")
    source_run = _parse_stdout_value(result.stdout, "[dim_market] source_run")
    output_path = None
    if output_value:
        output_path = Path(output_value)
        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path
    else:
        output_path = _find_dim_market_output(payload.out)

    row_count = _count_table_rows(output_path) if output_path else None

    return MarketMapRunResponse(
        output_path=str(output_path.relative_to(BASE_DIR)) if output_path else None,
        row_count=row_count,
        source_run=source_run,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_market_map(limit: int = 20, path_value: Optional[str] = None) -> MarketMapPreviewResponse:
    path = _find_dim_market_output(path_value)
    if not path:
        raise ValueError("dim_market output not found.")
    sanitized_limit = max(1, min(limit, 100))
    headers, rows = _read_table_head(path, sanitized_limit)
    row_count = _count_table_rows(path)
    return MarketMapPreviewResponse(
        file=_file_summary(path),
        headers=headers,
        rows=rows,
        row_count=row_count,
        limit=sanitized_limit,
    )


# ---------------------------------------------------------------------------
# Delete output
# ---------------------------------------------------------------------------

def delete_market_map_output(path_value: Optional[str] = None) -> MarketMapDeleteResponse:
    deleted_paths: List[str] = []
    target = _find_dim_market_output(path_value)
    if not target:
        return MarketMapDeleteResponse(deleted=False, paths=[])

    # Also remove alternate extension if present
    candidates = [target]
    if target.suffix.lower() == ".parquet":
        candidates.append(target.with_suffix(".csv"))
    elif target.suffix.lower() == ".csv":
        candidates.append(target.with_suffix(".parquet"))

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            candidate.unlink()
            deleted_paths.append(str(candidate.relative_to(BASE_DIR)))

    return MarketMapDeleteResponse(deleted=bool(deleted_paths), paths=deleted_paths)


# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------

class MarketMapJob:
    def __init__(self, job_id: str, payload: MarketMapRunRequest) -> None:
        self.job_id = job_id
        self.payload = payload
        self.status = "queued"
        self.result: Optional[MarketMapRunResponse] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def to_status(self) -> MarketMapJobStatus:
        return MarketMapJobStatus(
            job_id=self.job_id,
            status=self.status,
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    def _run(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.status = "running"
        try:
            self.result = run_market_map(self.payload)
            if self.result.output_path:
                self.status = "finished"
            else:
                self.status = "failed"
                # Show stderr if available, otherwise show both stdout and a helpful message
                stderr = (self.result.stderr or "").strip()
                stdout = (self.result.stdout or "").strip()
                if stderr:
                    self.error = f"Market map failed:\n{stderr}"
                elif stdout:
                    self.error = f"dim_market output was not created. Check the output:\n{stdout[-500:]}"
                else:
                    self.error = "dim_market output was not created. No output captured."
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
        finally:
            self.finished_at = datetime.now(timezone.utc)


class MarketMapJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, MarketMapJob] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: MarketMapRunRequest) -> str:
        job_id = uuid4().hex
        job = MarketMapJob(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> MarketMapJobStatus:
        job = self._get_job(job_id)
        return job.to_status()

    def list_jobs(self) -> List[MarketMapJobStatus]:
        with self._lock:
            return [job.to_status() for job in self._jobs.values()]

    def _get_job(self, job_id: str) -> MarketMapJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


MARKET_MAP_JOB_MANAGER = MarketMapJobManager()


def start_market_map_job(payload: MarketMapRunRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    return MARKET_MAP_JOB_MANAGER.start_job(payload)


def get_market_map_job(job_id: str) -> MarketMapJobStatus:
    return MARKET_MAP_JOB_MANAGER.get_status(job_id)
