from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.phat_edge import (
    PHATEdgeDeleteResponse,
    PHATEdgeJobStatus,
    PHATEdgePreviewResponse,
    PHATEdgeRowsResponse,
    PHATEdgeRunListResponse,
    PHATEdgeRunRequest,
    PHATEdgeRunResponse,
    PHATEdgeSummaryResponse,
)
from app.services.phat_edge import (
    run_phat_edge,
    start_phat_edge_job,
    get_phat_edge_job,
    delete_phat_edge_run,
    list_phat_edge_runs,
    preview_phat_edge_file,
    list_phat_edge_rows,
    summarize_phat_edge_file,
)
from app.services.job_guard import ensure_no_active_jobs

router = APIRouter(prefix="/phat-edge", tags=["pHAT edge"])


@router.post("/run", response_model=PHATEdgeRunResponse)
def run_phat_edge_route(payload: PHATEdgeRunRequest) -> PHATEdgeRunResponse:
    try:
        ensure_no_active_jobs()
        return run_phat_edge(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status = 409 if "Another job is already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post("/jobs", response_model=PHATEdgeJobStatus)
def start_phat_edge_job_route(payload: PHATEdgeRunRequest) -> PHATEdgeJobStatus:
    try:
        job_id = start_phat_edge_job(payload)
        return get_phat_edge_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=PHATEdgeJobStatus)
def get_phat_edge_job_route(job_id: str) -> PHATEdgeJobStatus:
    try:
        return get_phat_edge_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.get("/preview", response_model=PHATEdgePreviewResponse)
def preview_phat_edge_file_route(
    path: str,
    limit: int = 20,
    mode: str = "head",
) -> PHATEdgePreviewResponse:
    try:
        return preview_phat_edge_file(path, limit=limit, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs", response_model=PHATEdgeRunListResponse)
def list_phat_edge_runs_route(limit: int = 20) -> PHATEdgeRunListResponse:
    return list_phat_edge_runs(limit=limit)


@router.get("/summary", response_model=PHATEdgeSummaryResponse)
def summarize_phat_edge_file_route(path: str) -> PHATEdgeSummaryResponse:
    try:
        return summarize_phat_edge_file(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/rows", response_model=PHATEdgeRowsResponse)
def list_phat_edge_rows_route(path: str) -> PHATEdgeRowsResponse:
    try:
        return list_phat_edge_rows(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/runs", response_model=PHATEdgeDeleteResponse)
def delete_phat_edge_run_route(path: str) -> PHATEdgeDeleteResponse:
    try:
        return delete_phat_edge_run(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
