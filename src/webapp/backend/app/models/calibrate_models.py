from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DatasetFileSummary(BaseModel):
    name: str
    path: str
    size_bytes: int
    last_modified: Optional[str]


class DatasetListResponse(BaseModel):
    base_dir: str
    datasets: List[DatasetFileSummary]


class ModelRunSummary(BaseModel):
    id: str
    path: str
    last_modified: Optional[str]
    has_metadata: bool
    has_metrics: bool


class ModelListResponse(BaseModel):
    base_dir: str
    models: List[ModelRunSummary]


class CalibrateModelRunRequest(BaseModel):
    csv: str = Field(..., description="Relative path to dataset CSV.")
    out_name: Optional[str] = Field(
        default=None,
        description="Folder name under src/data/models for outputs.",
    )
    target_col: Optional[str] = Field(default=None)
    week_col: Optional[str] = Field(default=None)
    ticker_col: Optional[str] = Field(default=None)
    weight_col: Optional[str] = Field(default=None)
    features: Optional[str] = Field(default=None)
    categorical_features: Optional[str] = Field(default=None)
    add_interactions: Optional[bool] = Field(default=None)
    calibrate: Optional[str] = Field(default=None)
    c_grid: Optional[str] = Field(default=None)
    train_decay_half_life_weeks: Optional[float] = Field(default=None)
    calib_frac_of_train: Optional[float] = Field(default=None)
    fit_weight_renorm: Optional[str] = Field(default=None)
    test_weeks: Optional[int] = Field(default=None)
    val_windows: Optional[int] = Field(default=None)
    val_window_weeks: Optional[int] = Field(default=None)
    n_bins: Optional[int] = Field(default=None)
    random_state: Optional[int] = Field(default=None)


class CalibrateModelRunResponse(BaseModel):
    ok: bool
    out_dir: str
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]
