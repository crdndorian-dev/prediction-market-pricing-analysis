from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.polymarket import (
    PolymarketSnapshotHistoryResponse,
    PolymarketSnapshotJobStatus,
    PolymarketSnapshotLatestResponse,
    PolymarketSnapshotPreviewResponse,
    PolymarketSnapshotListResponse,
    PolymarketSnapshotDeleteResponse,
    PolymarketSnapshotRunRequest,
    PolymarketSnapshotRunResponse,
)
from app.services.polymarket_snapshots import (
    delete_polymarket_run,
    get_latest_snapshot,
    get_polymarket_snapshot_file,
    get_polymarket_snapshot_job,
    list_history_files,
    list_polymarket_runs,
    preview_snapshot_file,
    run_polymarket_snapshot,
    start_polymarket_snapshot_job,
)
from app.services.job_guard import ensure_no_active_jobs

router = APIRouter(prefix="/polymarket-snapshots", tags=["polymarket"])


@router.post("/run", response_model=PolymarketSnapshotRunResponse)
def run_snapshot(payload: PolymarketSnapshotRunRequest) -> PolymarketSnapshotRunResponse:
    try:
        ensure_no_active_jobs()
        return run_polymarket_snapshot(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status = 409 if "Another job is already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post("/jobs", response_model=PolymarketSnapshotJobStatus)
def start_snapshot_job(payload: PolymarketSnapshotRunRequest) -> PolymarketSnapshotJobStatus:
    try:
        job_id = start_polymarket_snapshot_job(payload)
        return get_polymarket_snapshot_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=PolymarketSnapshotJobStatus)
def get_snapshot_job(job_id: str) -> PolymarketSnapshotJobStatus:
    try:
        return get_polymarket_snapshot_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.get("/runs", response_model=PolymarketSnapshotListResponse)
def list_runs(limit: int = 20) -> PolymarketSnapshotListResponse:
    return list_polymarket_runs(limit=limit)


@router.get("/latest", response_model=PolymarketSnapshotLatestResponse)
def latest_snapshot() -> PolymarketSnapshotLatestResponse:
    return get_latest_snapshot()


@router.get("/history", response_model=PolymarketSnapshotHistoryResponse)
def history_files() -> PolymarketSnapshotHistoryResponse:
    return list_history_files()


@router.get("/preview", response_model=PolymarketSnapshotPreviewResponse)
def preview_snapshot(file: str, limit: int = 20, mode: str = "head") -> PolymarketSnapshotPreviewResponse:
    try:
        return preview_snapshot_file(file, limit=limit, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/file")
def download_snapshot_file(file: str) -> FileResponse:
    try:
        path = get_polymarket_snapshot_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(path, filename=path.name, media_type="text/csv")


@router.delete("/runs/{run_id}", response_model=PolymarketSnapshotDeleteResponse)
def delete_snapshot_run(run_id: str) -> PolymarketSnapshotDeleteResponse:
    try:
        delete_polymarket_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PolymarketSnapshotDeleteResponse(run_id=run_id, deleted=True)
