from fastapi import APIRouter, HTTPException

from app.models.polymarket import (
    PolymarketSnapshotHistoryResponse,
    PolymarketSnapshotLatestResponse,
    PolymarketSnapshotPreviewResponse,
    PolymarketSnapshotListResponse,
    PolymarketSnapshotRunRequest,
    PolymarketSnapshotRunResponse,
)
from app.services.polymarket_snapshots import (
    get_latest_snapshot,
    list_history_files,
    list_polymarket_runs,
    preview_snapshot_file,
    run_polymarket_snapshot,
)

router = APIRouter(prefix="/polymarket-snapshots", tags=["polymarket"])


@router.post("/run", response_model=PolymarketSnapshotRunResponse)
def run_snapshot(payload: PolymarketSnapshotRunRequest) -> PolymarketSnapshotRunResponse:
    try:
        return run_polymarket_snapshot(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
