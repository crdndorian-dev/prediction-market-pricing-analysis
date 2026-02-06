from fastapi import APIRouter, HTTPException

from app.models.calibrate_models import (
    AutoModelRunRequest,
    CalibrateModelRunRequest,
    CalibrateModelRunResponse,
    CalibrationJobStatus,
    DatasetListResponse,
    ModelFileContentResponse,
    ModelFilesListResponse,
    ModelListResponse,
    ModelDetailResponse,
    ModelRunSummary,
    RegimePreviewRequest,
    RegimePreviewResponse,
    RenameModelRequest,
)
from app.services.calibrate_models import (
    delete_model,
    get_model_detail,
    get_model_file_content,
    list_datasets,
    list_model_files,
    list_models,
    preview_regime,
    start_auto_calibration_job,
    start_calibration_job,
    get_calibration_job,
    rename_model,
    run_auto_model_selection,
    run_calibration,
)
from app.services.job_guard import ensure_no_active_jobs

router = APIRouter(prefix="/calibrate-models", tags=["calibration"])


@router.get("/datasets", response_model=DatasetListResponse)
def list_dataset_files() -> DatasetListResponse:
    return list_datasets()


@router.get("/models", response_model=ModelListResponse)
def list_model_runs() -> ModelListResponse:
    return list_models()


@router.get("/models/{model_id}", response_model=ModelDetailResponse)
def get_model_detail_route(model_id: str) -> ModelDetailResponse:
    try:
        return get_model_detail(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/run", response_model=CalibrateModelRunResponse)
def run_calibration_route(
    payload: CalibrateModelRunRequest,
) -> CalibrateModelRunResponse:
    try:
        ensure_no_active_jobs()
        return run_calibration(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status = 409 if "Another job is already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post("/run-auto", response_model=CalibrateModelRunResponse)
def run_auto_model_selection_route(
    payload: AutoModelRunRequest,
) -> CalibrateModelRunResponse:
    try:
        ensure_no_active_jobs()
        return run_auto_model_selection(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        status = 409 if "Another job is already running" in str(exc) else 500
        raise HTTPException(status_code=status, detail=str(exc)) from exc


@router.post("/preview", response_model=RegimePreviewResponse)
def preview_regime_route(payload: RegimePreviewRequest) -> RegimePreviewResponse:
    try:
        return preview_regime(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/jobs", response_model=CalibrationJobStatus)
def start_calibration_job_route(payload: CalibrateModelRunRequest) -> CalibrationJobStatus:
    try:
        job_id = start_calibration_job(payload)
        return get_calibration_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/jobs-auto", response_model=CalibrationJobStatus)
def start_auto_calibration_job_route(payload: AutoModelRunRequest) -> CalibrationJobStatus:
    try:
        job_id = start_auto_calibration_job(payload)
        return get_calibration_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/jobs/{job_id}", response_model=CalibrationJobStatus)
def get_calibration_job_route(job_id: str) -> CalibrationJobStatus:
    try:
        return get_calibration_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.delete("/models/{model_id}", response_model=ModelRunSummary)
def delete_calibration_model(model_id: str) -> ModelRunSummary:
    try:
        return delete_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")


@router.patch("/models/{model_id}", response_model=ModelRunSummary)
def rename_calibration_model(model_id: str, payload: RenameModelRequest) -> ModelRunSummary:
    try:
        return rename_model(model_id, payload.new_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/models/{model_id}/files", response_model=ModelFilesListResponse)
def list_model_files_route(model_id: str) -> ModelFilesListResponse:
    try:
        return list_model_files(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")


@router.get("/models/{model_id}/files/{filename}", response_model=ModelFileContentResponse)
def get_model_file_content_route(model_id: str, filename: str) -> ModelFileContentResponse:
    try:
        return get_model_file_content(model_id, filename)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"File not found.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
