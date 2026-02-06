from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.models.datasets import (
    DatasetJobStatus,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetRunRenameRequest,
    DatasetRunRequest,
    DatasetRunResponse,
    DatasetRunSummary,
)
from app.services.datasets import (
    cancel_dataset_job,
    get_dataset_job,
    run_dataset,
    start_dataset_job,
    delete_dataset_run,
    list_dataset_runs,
    preview_dataset_file,
    get_dataset_file_path,
    rename_dataset_run,
)
from app.services.job_guard import ensure_no_active_jobs

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/run", response_model=DatasetRunResponse)
def run_dataset_route(payload: DatasetRunRequest) -> DatasetRunResponse:
    try:
        ensure_no_active_jobs()
        return run_dataset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status = 409 if "Another job is already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


def _handle_job_not_found(job_id: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.post("/jobs", response_model=DatasetJobStatus)
def start_dataset_job_route(payload: DatasetRunRequest) -> DatasetJobStatus:
    try:
        job_id = start_dataset_job(payload)
        return get_dataset_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status = 409 if "Another job is already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=DatasetJobStatus)
def get_dataset_job_route(job_id: str) -> DatasetJobStatus:
    try:
        return get_dataset_job(job_id)
    except KeyError:
        raise _handle_job_not_found(job_id)


@router.delete("/jobs/{job_id}", response_model=DatasetJobStatus)
def cancel_dataset_job_route(job_id: str) -> DatasetJobStatus:
    try:
        return cancel_dataset_job(job_id)
    except KeyError:
        raise _handle_job_not_found(job_id)


@router.get("/runs", response_model=DatasetListResponse)
def list_dataset_runs_route() -> DatasetListResponse:
    return list_dataset_runs()


@router.get("/runs/preview", response_model=DatasetPreviewResponse)
def preview_dataset_file_route(
    path: str = Query(...),
    limit: int = Query(20, ge=1),
    mode: Literal["head", "tail"] = Query("head"),
) -> DatasetPreviewResponse:
    try:
        return preview_dataset_file(path, limit=limit, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runs/file")
def dataset_file_route(path: str = Query(...)) -> FileResponse:
    try:
        file_path = get_dataset_file_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        file_path,
        media_type="text/csv",
        filename=file_path.name,
    )


@router.delete("/runs", response_model=DatasetRunSummary)
def delete_dataset_run_route(run_dir: str = Query(...)) -> DatasetRunSummary:
    try:
        return delete_dataset_run(run_dir)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {run_dir} not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/runs/rename", response_model=DatasetRunSummary)
def rename_dataset_run_route(payload: DatasetRunRenameRequest) -> DatasetRunSummary:
    try:
        return rename_dataset_run(payload.run_dir, payload.new_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Run {payload.run_dir} not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
