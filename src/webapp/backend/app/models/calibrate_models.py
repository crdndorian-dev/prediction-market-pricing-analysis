from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

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


class SplitMetricSummary(BaseModel):
    split: str
    baseline_logloss: float
    model_logloss: float
    delta_model_minus_baseline: float
    baseline_brier: Optional[float] = None
    model_brier: Optional[float] = None
    delta_brier: Optional[float] = None
    baseline_ece: Optional[float] = None
    model_ece: Optional[float] = None
    delta_ece: Optional[float] = None
    baseline_ece_q: Optional[float] = None
    model_ece_q: Optional[float] = None
    delta_ece_q: Optional[float] = None
    delta_logloss_ci_lo: Optional[float] = None
    delta_logloss_ci_hi: Optional[float] = None
    delta_brier_ci_lo: Optional[float] = None
    delta_brier_ci_hi: Optional[float] = None
    delta_ece_ci_lo: Optional[float] = None
    delta_ece_ci_hi: Optional[float] = None
    delta_ece_q_ci_lo: Optional[float] = None
    delta_ece_q_ci_hi: Optional[float] = None
    bootstrap_n_groups: Optional[int] = None
    bootstrap_B: Optional[int] = None
    status: Literal["good", "unusable"]
    verdict: str


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
    foundation_tickers: Optional[str] = Field(default=None)
    foundation_weight: Optional[float] = Field(default=None)
    mode: Optional[str] = Field(default=None)
    ticker_intercepts: Optional[str] = Field(default=None)
    ticker_x_interactions: Optional[bool] = Field(default=None)
    ticker_min_support: Optional[int] = Field(default=None)
    ticker_min_support_interactions: Optional[int] = Field(default=None)
    train_tickers: Optional[str] = Field(default=None)
    tdays_allowed: Optional[str] = Field(default=None)
    asof_dow_allowed: Optional[str] = Field(default=None)
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
    eceq_bins: Optional[int] = Field(default=None)
    selection_objective: Optional[str] = Field(default=None)
    fallback_to_baseline_if_worse: Optional[bool] = Field(default=None)
    auto_drop_near_constant: Optional[bool] = Field(default=None)
    random_state: Optional[int] = Field(default=None)
    metrics_top_tickers: Optional[int] = Field(default=None)
    enable_x_abs_m: Optional[bool] = Field(default=None, description="Enable x_abs_m interaction feature")
    group_reweight: Optional[str] = Field(default=None, description="Group reweighting mode: none or chain")
    max_abs_logm: Optional[float] = Field(default=None, description="Maximum absolute log-moneyness filter")
    drop_prn_extremes: Optional[bool] = Field(default=None, description="Drop pRN extremes near 0 or 1")
    prn_eps: Optional[float] = Field(default=None, description="Epsilon for pRN extremes filter")
    bootstrap_ci: Optional[bool] = Field(default=None, description="Enable bootstrap CIs for delta metrics")
    bootstrap_b: Optional[int] = Field(default=None, description="Number of bootstrap resamples")
    bootstrap_seed: Optional[int] = Field(default=None, description="Bootstrap random seed")
    bootstrap_group: Optional[str] = Field(default=None, description="Bootstrap grouping: auto|ticker_day|day|iid")


class RenameModelRequest(BaseModel):
    new_name: str = Field(..., description="New name for the model folder.")


class AutoModelRunRequest(BaseModel):
    csv: str = Field(..., description="Relative path to dataset CSV.")
    run_name: Optional[str] = Field(
        default=None,
        description="Output folder name for best model under src/data/models.",
    )
    objective: Literal["logloss"] = Field(
        default="logloss",
        description="Objective the tuner optimizes (only logloss supported).",
    )
    max_trials: Optional[int] = Field(
        default=None,
        description="Maximum number of trials the tuner should run.",
    )
    seed: Optional[int] = Field(default=42, description="Seed propagated to each calibrator run.")
    parallel: Optional[int] = Field(
        default=1,
        description="How many calibrator trials run in parallel.",
    )
    baseline_args: Optional[str] = Field(
        default="",
        description="Extra CLI arguments always added to the calibrator.",
    )
    tdays_allowed: Optional[str] = Field(default=None)
    asof_dow_allowed: Optional[str] = Field(default=None)
    foundation_tickers: Optional[str] = Field(default=None)
    foundation_weight: Optional[float] = Field(default=None)
    bootstrap_ci: Optional[bool] = Field(default=None, description="Enable bootstrap CIs for delta metrics")
    bootstrap_b: Optional[int] = Field(default=None, description="Number of bootstrap resamples")
    bootstrap_seed: Optional[int] = Field(default=None, description="Bootstrap random seed")
    bootstrap_group: Optional[str] = Field(default=None, description="Bootstrap grouping: auto|ticker_day|day|iid")


class RegimePreviewRequest(BaseModel):
    csv: str = Field(..., description="Relative path to dataset CSV.")
    tdays_allowed: Optional[str] = Field(default=None)
    asof_dow_allowed: Optional[str] = Field(default=None)


class RegimePreviewResponse(BaseModel):
    rows_before: int
    rows_after: int
    tickers_after: int
    by_tdays: Dict[str, int]


class ModelDetailResponse(BaseModel):
    id: str
    path: str
    last_modified: Optional[str]
    has_metadata: bool
    has_metrics: bool
    files: List[str]
    features_used: Optional[List[str]] = None
    categorical_features_used: Optional[List[str]] = None
    metrics_summary: Optional[Dict[str, SplitMetricSummary]] = None
    model_equation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    feature_manifest: Optional[Dict[str, Any]] = None


class CalibrateModelRunResponse(BaseModel):
    ok: bool
    out_dir: str
    files: List[str]
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]
    metrics_summary: Optional[Dict[str, SplitMetricSummary]] = None
    auto_out_dir: Optional[str] = None
    features: Optional[List[str]] = None
    model_equation: Optional[str] = None


class CalibrationJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed"]
    mode: Literal["manual", "auto"]
    result: Optional[CalibrateModelRunResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class ModelFileSummary(BaseModel):
    name: str
    size_bytes: int
    is_viewable: bool


class ModelFilesListResponse(BaseModel):
    model_id: str
    files: List[ModelFileSummary]


class ModelFileContentResponse(BaseModel):
    model_id: str
    filename: str
    content: str
    content_type: Literal["json", "csv", "markdown", "text"]
    truncated: bool
