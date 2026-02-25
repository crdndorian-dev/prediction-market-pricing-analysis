from __future__ import annotations

import csv
import math
import subprocess
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import pandas as pd

from app.models.phat_edge import (
    PHATEdgeDistributionStats,
    PHATEdgeDeleteResponse,
    PHATEdgeFileSummary,
    PHATEdgeJobStatus,
    PHATEdgePreviewResponse,
    PHATEdgeRow,
    PHATEdgeRowsResponse,
    PHATEdgeRunRequest,
    PHATEdgeRunResponse,
    PHATEdgeRunSummary,
    PHATEdgeRunListResponse,
    PHATEdgeSummaryResponse,
)

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = (
    BASE_DIR / "src" / "scripts" / "06-compute-edge-v1.1.py"
)
OUTPUT_DIR = BASE_DIR / "src" / "data" / "analysis" / "phat-edge"


def _resolve_project_path(value: str, *, must_exist: bool = True) -> Path:
    path = Path(value)
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


def _default_output_path() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_DIR / f"phat-edge-{timestamp}.csv"


def _edge_file_summary(path: Path) -> PHATEdgeFileSummary:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return PHATEdgeFileSummary(
        name=path.name,
        path=str(path.relative_to(BASE_DIR)),
        size_bytes=path.stat().st_size,
        last_modified=mtime,
    )


def _read_csv_head(path: Path, limit: int) -> tuple[List[str], List[Dict[str, Optional[str]]]]:
    rows: List[Dict[str, Optional[str]]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append({key: row.get(key) for key in headers})
    return headers, rows


def _read_csv_tail(path: Path, limit: int) -> tuple[List[str], List[Dict[str, Optional[str]]], int]:
    buffer: deque[Dict[str, Optional[str]]] = deque(maxlen=limit)
    row_count = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for row in reader:
            row_count += 1
            buffer.append({key: row.get(key) for key in headers})
    return headers, list(buffer), row_count


def preview_phat_edge_file(
    path_value: str,
    *,
    limit: int = 20,
    mode: str = "head",
) -> PHATEdgePreviewResponse:
    path = _resolve_project_path(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")
    sanitized_limit = max(1, min(limit, 100))
    normalized_mode = mode.lower()
    if normalized_mode not in {"head", "tail"}:
        raise ValueError("mode must be 'head' or 'tail'")

    if normalized_mode == "tail":
        headers, rows, row_count = _read_csv_tail(path, sanitized_limit)
    else:
        headers, rows = _read_csv_head(path, sanitized_limit)
        row_count = None

    return PHATEdgePreviewResponse(
        file=_edge_file_summary(path),
        headers=headers,
        rows=rows,
        row_count=row_count,
        mode=normalized_mode,
        limit=sanitized_limit,
    )


def _iter_edge_runs() -> List[Path]:
    if not OUTPUT_DIR.exists():
        return []
    return [
        item
        for item in OUTPUT_DIR.iterdir()
        if item.is_file() and item.suffix.lower() == ".csv"
    ]


def list_phat_edge_runs(limit: int = 20) -> PHATEdgeRunListResponse:
    runs = sorted(
        _iter_edge_runs(),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if limit < 1:
        limit = 1
    return PHATEdgeRunListResponse(
        runs=[_edge_file_summary(path) for path in runs[:limit]],
    )


def summarize_phat_edge_file(path_value: str) -> PHATEdgeSummaryResponse:
    path = _resolve_project_path(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")
    p_hat_distribution, edge_distribution, top_edges = _collect_estimates(
        path,
        [],
    )
    return PHATEdgeSummaryResponse(
        file=_edge_file_summary(path),
        pHat_distribution=p_hat_distribution,
        edge_distribution=edge_distribution,
        top_edges=top_edges,
    )


def delete_phat_edge_run(path_value: str) -> PHATEdgeDeleteResponse:
    path = _resolve_project_path(path_value)
    try:
        path.relative_to(OUTPUT_DIR)
    except ValueError as exc:
        raise ValueError("Edge run must be inside src/data/analysis/phat-edge.") from exc
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")
    path.unlink()
    return PHATEdgeDeleteResponse(path=str(path.relative_to(BASE_DIR)), deleted=True)


def _format_bool_flag(value: Optional[bool], default: bool) -> str:
    effective = default if value is None else value
    return "true" if effective else "false"


def _parse_excludes(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [
        ticker.strip().upper()
        for ticker in value.split(",")
        if ticker.strip()
    ]


def _to_optional_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _build_distribution(
    df: pd.DataFrame, column: str
) -> Optional[PHATEdgeDistributionStats]:
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return PHATEdgeDistributionStats(
        count=int(series.count()),
        mean=float(series.mean()),
        min=float(series.min()),
        max=float(series.max()),
    )


def _build_top_edges(
    df: pd.DataFrame,
    exclude: List[str],
    limit: int = 10,
) -> List[PHATEdgeRow]:
    if "edge" not in df.columns:
        return []
    filtered = df.dropna(subset=["edge"]).copy()
    if filtered.empty:
        return []
    display_tickers = filtered["ticker"].astype("string").fillna("")
    filtered["__ticker_display"] = display_tickers
    filtered["__ticker_upper"] = display_tickers.str.upper()
    if exclude:
        blacklist = {ticker.upper() for ticker in exclude}
        filtered = filtered[~filtered["__ticker_upper"].isin(blacklist)]
    if filtered.empty:
        return []
    filtered = filtered.sort_values("edge", ascending=False).head(limit)
    results: List[PHATEdgeRow] = []
    for _, row in filtered.iterrows():
        edge_source_val = row.get("edge_source")
        edge_source = None
        if pd.notna(edge_source_val):
            edge_source = str(edge_source_val)
        spot_val = row.get("S")
        if spot_val is None or pd.isna(spot_val):
            spot_val = row.get("spot")
        if spot_val is None or pd.isna(spot_val):
            spot_val = row.get("spot_price")
        p_hat_val = _to_optional_float(row.get("pHAT"))
        q_hat_val = _to_optional_float(row.get("qHAT"))
        if q_hat_val is None and p_hat_val is not None:
            q_hat_val = 1.0 - p_hat_val
        results.append(
            PHATEdgeRow(
                ticker=str(row.get("__ticker_display") or ""),
                K=_to_optional_float(row.get("K")),
                spot=_to_optional_float(spot_val),
                pHAT=p_hat_val,
                qHAT=q_hat_val,
                edge=_to_optional_float(row.get("edge")),
                pPM_buy=_to_optional_float(row.get("pPM_buy")),
                qPM_buy=_to_optional_float(row.get("qPM_buy")),
                edge_source=edge_source,
                pRN=_to_optional_float(row.get("pRN")),
                qRN=_to_optional_float(row.get("qRN")),
            ),
        )
    return results


def _build_edge_rows(
    df: pd.DataFrame,
    exclude: List[str],
) -> List[PHATEdgeRow]:
    if "edge" not in df.columns:
        return []
    filtered = df.copy()
    if "ticker" in filtered.columns:
        display_tickers = filtered["ticker"].astype("string").fillna("")
        filtered["__ticker_display"] = display_tickers
        filtered["__ticker_upper"] = display_tickers.str.upper()
    else:
        filtered["__ticker_display"] = ""
        filtered["__ticker_upper"] = ""
    if exclude:
        blacklist = {ticker.upper() for ticker in exclude}
        filtered = filtered[~filtered["__ticker_upper"].isin(blacklist)]
    if filtered.empty:
        return []
    if "edge" in filtered.columns:
        filtered = filtered.sort_values("edge", ascending=False, na_position="last")
    results: List[PHATEdgeRow] = []
    for _, row in filtered.iterrows():
        edge_source_val = row.get("edge_source")
        edge_source = None
        if pd.notna(edge_source_val):
            edge_source = str(edge_source_val)
        spot_val = row.get("S")
        if spot_val is None or pd.isna(spot_val):
            spot_val = row.get("spot")
        if spot_val is None or pd.isna(spot_val):
            spot_val = row.get("spot_price")
        p_hat_val = _to_optional_float(row.get("pHAT"))
        q_hat_val = _to_optional_float(row.get("qHAT"))
        if q_hat_val is None and p_hat_val is not None:
            q_hat_val = 1.0 - p_hat_val
        results.append(
            PHATEdgeRow(
                ticker=str(row.get("__ticker_display") or ""),
                K=_to_optional_float(row.get("K")),
                spot=_to_optional_float(spot_val),
                pHAT=p_hat_val,
                qHAT=q_hat_val,
                edge=_to_optional_float(row.get("edge")),
                pPM_buy=_to_optional_float(row.get("pPM_buy")),
                qPM_buy=_to_optional_float(row.get("qPM_buy")),
                edge_source=edge_source,
                pRN=_to_optional_float(row.get("pRN")),
                qRN=_to_optional_float(row.get("qRN")),
            ),
        )
    return results


def _collect_estimates(
    path: Path, exclude: List[str]
) -> tuple[
    Optional[PHATEdgeDistributionStats],
    Optional[PHATEdgeDistributionStats],
    List[PHATEdgeRow],
]:
    try:
        df = pd.read_csv(path, low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None, None, []
    p_hat_distribution = _build_distribution(df, "pHAT")
    edge_distribution = _build_distribution(df, "edge")
    top_edges = _build_top_edges(df, exclude)
    return p_hat_distribution, edge_distribution, top_edges


def list_phat_edge_rows(path_value: str) -> PHATEdgeRowsResponse:
    path = _resolve_project_path(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")
    try:
        df = pd.read_csv(
            path,
            low_memory=False,
            usecols=lambda col: col
            in {
                "ticker",
                "K",
                "S",
                "spot",
                "spot_price",
                "pHAT",
                "qHAT",
                "edge",
                "pPM_buy",
                "qPM_buy",
                "edge_source",
                "pRN",
                "qRN",
            },
        )
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return PHATEdgeRowsResponse(
            file=_edge_file_summary(path),
            rows=[],
            row_count=0,
        )
    rows = _build_edge_rows(df, [])
    return PHATEdgeRowsResponse(
        file=_edge_file_summary(path),
        rows=rows,
        row_count=int(len(df)),
    )


def run_phat_edge(payload: PHATEdgeRunRequest) -> PHATEdgeRunResponse:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Inference script not found at {SCRIPT_PATH}")

    model_path = _resolve_project_path(payload.model_path)
    snapshot_path = _resolve_project_path(payload.snapshot_csv)

    output_path = (
        _resolve_project_path(payload.out_csv, must_exist=False)
        if payload.out_csv
        else _default_output_path()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_PATH),
        "--model-path",
        str(model_path),
        "--snapshot-csv",
        str(snapshot_path),
        "--out-csv",
        str(output_path),
        "--require-columns-strict",
        _format_bool_flag(payload.require_columns_strict, True),
        "--compute-edge",
        _format_bool_flag(payload.compute_edge, True),
        "--skip-edge-outside-prn-range",
        _format_bool_flag(payload.skip_edge_outside_prn_range, True),
    ]

    if payload.exclude_tickers:
        cmd.extend(["--exclude-tickers", payload.exclude_tickers])

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    duration_s = round(time.monotonic() - start, 3)

    summary = PHATEdgeRunSummary(
        model_path=str(model_path.relative_to(BASE_DIR)),
        snapshot_csv=str(snapshot_path.relative_to(BASE_DIR)),
        output_csv=str(output_path.relative_to(BASE_DIR)),
        duration_s=duration_s,
        ok=result.returncode == 0,
    )

    exclude_list = _parse_excludes(payload.exclude_tickers)
    p_hat_distribution, edge_distribution, top_edges = _collect_estimates(
        output_path,
        exclude_list,
    )

    return PHATEdgeRunResponse(
        ok=result.returncode == 0,
        command=cmd,
        stdout=result.stdout,
        stderr=result.stderr,
        run_summary=summary,
        pHat_distribution=p_hat_distribution,
        edge_distribution=edge_distribution,
        top_edges=top_edges,
    )


class PHATEdgeJob:
    def __init__(self, job_id: str, payload: PHATEdgeRunRequest) -> None:
        self.job_id = job_id
        self.payload = payload
        self.status = "queued"
        self.result: Optional[PHATEdgeRunResponse] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def to_status(self) -> PHATEdgeJobStatus:
        return PHATEdgeJobStatus(
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
            self.result = run_phat_edge(self.payload)
            if self.result.ok:
                self.status = "finished"
            else:
                self.status = "failed"
                self.error = (self.result.stderr or "").strip() or "Edge run failed."
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
        finally:
            self.finished_at = datetime.utcnow()


class PHATEdgeJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, PHATEdgeJob] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: PHATEdgeRunRequest) -> str:
        job_id = uuid4().hex
        job = PHATEdgeJob(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> PHATEdgeJobStatus:
        job = self._get_job(job_id)
        return job.to_status()

    def list_jobs(self) -> List[PHATEdgeJobStatus]:
        with self._lock:
            return [job.to_status() for job in self._jobs.values()]

    def _get_job(self, job_id: str) -> PHATEdgeJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


PHAT_EDGE_JOB_MANAGER = PHATEdgeJobManager()


def start_phat_edge_job(payload: PHATEdgeRunRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    return PHAT_EDGE_JOB_MANAGER.start_job(payload)


def get_phat_edge_job(job_id: str) -> PHATEdgeJobStatus:
    return PHAT_EDGE_JOB_MANAGER.get_status(job_id)


def list_phat_edge_jobs() -> List[PHATEdgeJobStatus]:
    return PHAT_EDGE_JOB_MANAGER.list_jobs()
