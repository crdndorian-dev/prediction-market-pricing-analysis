from typing import Optional

from fastapi import APIRouter, HTTPException

from app.models.market_map import (
    MarketMapDeleteResponse,
    MarketMapJobStatus,
    MarketMapPreviewResponse,
    MarketMapRunRequest,
    MarketMapRunResponse,
)
from app.services.job_guard import ensure_no_active_jobs
from app.services.market_map import (
    delete_market_map_output,
    get_market_map_job,
    preview_market_map,
    run_market_map,
    start_market_map_job,
)

router = APIRouter(prefix="/market-map", tags=["market-map"])


@router.post("/run", response_model=MarketMapRunResponse)
def run(payload: MarketMapRunRequest) -> MarketMapRunResponse:
    try:
        ensure_no_active_jobs()
        return run_market_map(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status = 409 if "Another job is already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post("/jobs", response_model=MarketMapJobStatus)
def start_job(payload: MarketMapRunRequest) -> MarketMapJobStatus:
    try:
        job_id = start_market_map_job(payload)
        return get_market_map_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=MarketMapJobStatus)
def get_job(job_id: str) -> MarketMapJobStatus:
    try:
        return get_market_map_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.get("/preview", response_model=MarketMapPreviewResponse)
def preview(limit: int = 20, path: Optional[str] = None) -> MarketMapPreviewResponse:
    try:
        return preview_market_map(limit=limit, path_value=path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/output", response_model=MarketMapDeleteResponse)
def delete_output(path: Optional[str] = None) -> MarketMapDeleteResponse:
    try:
        return delete_market_map_output(path_value=path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
