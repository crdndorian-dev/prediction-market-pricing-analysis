from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.models.polymarket import (
    PolymarketSnapshotFileSummary,
    PolymarketSnapshotHistoryResponse,
    PolymarketSnapshotListResponse,
    PolymarketSnapshotLatestResponse,
    PolymarketSnapshotPreviewResponse,
    PolymarketSnapshotRunRequest,
    PolymarketSnapshotRunResponse,
    PolymarketSnapshotRunSummary,
)

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "3-polymarket-fetch-data-v1.0.py"
OUTPUT_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket"
HISTORY_DIR = OUTPUT_DIR / "history"

_SNAPSHOT_PATTERN = re.compile(r"__snapshot__([0-9]{4}-[0-9]{2}-[0-9]{2})__v")


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.resolve()
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
    try:
        dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.isoformat()


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
    if not OUTPUT_DIR.exists():
        return PolymarketSnapshotLatestResponse(date=None, files=[], history_file=None)

    date_map: Dict[str, List[Path]] = {}
    for item in OUTPUT_DIR.iterdir():
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
        _file_summary(path, _snapshot_kind(path.name)) for path in date_map[latest_date]
    ]
    files.sort(key=lambda entry: entry.name)
    history_file = _latest_history_file()
    return PolymarketSnapshotLatestResponse(
        date=latest_date,
        files=files,
        history_file=history_file,
    )


def _latest_history_file() -> Optional[PolymarketSnapshotFileSummary]:
    if not HISTORY_DIR.exists():
        return None
    files = [item for item in HISTORY_DIR.iterdir() if item.is_file()]
    if not files:
        return None
    latest = max(files, key=lambda item: item.stat().st_mtime)
    return _file_summary(latest, "history")


def list_history_files() -> PolymarketSnapshotHistoryResponse:
    if not HISTORY_DIR.exists():
        return PolymarketSnapshotHistoryResponse(files=[])
    files = [item for item in HISTORY_DIR.iterdir() if item.is_file()]
    summaries = [_file_summary(item, "history") for item in files]
    summaries.sort(key=lambda entry: entry.last_modified, reverse=True)
    return PolymarketSnapshotHistoryResponse(files=summaries)


def _resolve_polymarket_file(path_value: str) -> Path:
    path = _resolve_project_path(path_value)
    try:
        path.relative_to(OUTPUT_DIR)
    except ValueError as exc:
        raise ValueError("File must be under src/data/polymarket.") from exc
    return path


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
    runs_dir = OUTPUT_DIR / "runs"
    if not runs_dir.exists():
        return PolymarketSnapshotListResponse(
            out_dir=str(OUTPUT_DIR.relative_to(BASE_DIR)),
            runs=[],
        )

    summaries: List[PolymarketSnapshotRunSummary] = []
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
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
    return PolymarketSnapshotListResponse(
        out_dir=str(OUTPUT_DIR.relative_to(BASE_DIR)),
        runs=summaries[:limit],
    )


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

    if payload.keep_nonexec:
        cmd.append("--keep-nonexec")

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    duration_s = round(time.monotonic() - start, 3)

    run_id = _parse_run_id(result.stdout)
    run_dir = OUTPUT_DIR / "runs" / run_id if run_id else None
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
