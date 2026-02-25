from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.models.polymarket_history import (
    PolymarketHistoryJobStatus,
    PolymarketHistoryRunRequest,
    PolymarketHistoryRunResponse,
)
from app.services.polymarket_history import (
    get_polymarket_history_job,
    run_polymarket_history,
    start_polymarket_history_job,
    cancel_polymarket_history_job,
    get_csv_preview,
    list_pipeline_runs,
    rename_pipeline_run,
    set_active_run,
    delete_pipeline_run,
    toggle_pin_run,
    get_runs_storage_summary,
    get_latest_pointer,
)
from app.services.job_guard import ensure_no_active_jobs

router = APIRouter(prefix="/polymarket-history", tags=["polymarket-history"])


@router.post("/run", response_model=PolymarketHistoryRunResponse)
def run_history(payload: PolymarketHistoryRunRequest) -> PolymarketHistoryRunResponse:
    try:
        ensure_no_active_jobs()
        return run_polymarket_history(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status = 409 if "Another job is already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post("/jobs", response_model=PolymarketHistoryJobStatus)
def start_history_job(payload: PolymarketHistoryRunRequest) -> PolymarketHistoryJobStatus:
    try:
        job_id = start_polymarket_history_job(payload)
        return get_polymarket_history_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=PolymarketHistoryJobStatus)
def get_history_job(job_id: str) -> PolymarketHistoryJobStatus:
    try:
        return get_polymarket_history_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.post("/jobs/{job_id}/cancel", response_model=PolymarketHistoryJobStatus)
def cancel_history_job(job_id: str) -> PolymarketHistoryJobStatus:
    try:
        return cancel_polymarket_history_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.get("/jobs/{job_id}/csv-preview/{filename}")
def get_csv_preview_endpoint(job_id: str, filename: str, limit: int = 100) -> Dict[str, Any]:
    try:
        return get_csv_preview(job_id, filename, limit)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Run management endpoints (Phase 2)
# ---------------------------------------------------------------------------


class RenameRunRequest(BaseModel):
    label: Optional[str] = Field(default=None, description="User-facing label for the run.")


class SetActiveRunRequest(BaseModel):
    run_id: str = Field(..., description="Run ID to set as active.")


@router.get("/runs")
def list_runs() -> Dict[str, Any]:
    """List all pipeline runs with manifest data, newest first."""
    runs = list_pipeline_runs()
    storage = get_runs_storage_summary()
    latest = get_latest_pointer()
    return {
        "runs": runs,
        "storage": storage,
        "latest": latest,
    }


@router.patch("/runs/{run_id}")
def rename_run(run_id: str, body: RenameRunRequest) -> Dict[str, Any]:
    """Update the user-facing label for a run (folder name is unchanged)."""
    try:
        return rename_pipeline_run(run_id, body.label or "")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.put("/runs/active")
def set_active(body: SetActiveRunRequest) -> Dict[str, Any]:
    """Set a run as the active/default run."""
    try:
        return set_active_run(body.run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/runs/{run_id}")
def delete_run(run_id: str) -> Dict[str, Any]:
    """Delete a pipeline run directory. Cannot delete the active run."""
    try:
        return delete_pipeline_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/runs/{run_id}/pin")
def pin_run(run_id: str) -> Dict[str, Any]:
    """Toggle the pinned state of a run."""
    try:
        return toggle_pin_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
