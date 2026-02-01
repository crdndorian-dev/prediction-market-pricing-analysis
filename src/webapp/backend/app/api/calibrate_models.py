from fastapi import APIRouter, HTTPException

from app.models.calibrate_models import (
    CalibrateModelRunRequest,
    CalibrateModelRunResponse,
    DatasetListResponse,
    ModelListResponse,
    ModelRunSummary,
)
from app.services.calibrate_models import (
    delete_model,
    list_datasets,
    list_models,
    run_calibration,
)

router = APIRouter(prefix="/calibrate-models", tags=["calibration"])


@router.get("/datasets", response_model=DatasetListResponse)
def list_dataset_files() -> DatasetListResponse:
    return list_datasets()


@router.get("/models", response_model=ModelListResponse)
def list_model_runs() -> ModelListResponse:
    return list_models()


@router.post("/run", response_model=CalibrateModelRunResponse)
def run_calibration_route(
    payload: CalibrateModelRunRequest,
) -> CalibrateModelRunResponse:
    try:
        return run_calibration(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/models/{model_id}", response_model=ModelRunSummary)
def delete_calibration_model(model_id: str) -> ModelRunSummary:
    try:
        return delete_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found.")
