from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


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
    model_kind: Optional[Literal["calibrate", "mixed", "both"]] = Field(
        default=None,
        description="Run calibrator, mixed model, or both.",
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
    mixed_features: Optional[str] = Field(default=None, description="Decision features path for mixed model")
    mixed_out_dir: Optional[str] = Field(default=None, description="Output root directory for mixed models")
    mixed_run_id: Optional[str] = Field(default=None, description="Optional mixed model run id")
    mixed_model: Optional[Literal["residual", "blend"]] = Field(
        default=None,
        description="Mixed model type (residual or blend).",
    )
    mixed_pm_col: Optional[str] = Field(default=None, description="Polymarket probability column")
    mixed_prn_col: Optional[str] = Field(default=None, description="pRN probability column")
    mixed_label_col: Optional[str] = Field(default=None, description="Label column")
    mixed_features_cols: Optional[str] = Field(default=None, description="Comma-separated feature columns")
    mixed_train_frac: Optional[float] = Field(default=None, description="Train fraction for mixed model")
    mixed_walk_forward: Optional[bool] = Field(default=None, description="Enable mixed walk-forward")
    mixed_wf_train_days: Optional[int] = Field(default=None, description="Mixed WF train window (days)")
    mixed_wf_test_days: Optional[int] = Field(default=None, description="Mixed WF test window (days)")
    mixed_wf_step_days: Optional[int] = Field(default=None, description="Mixed WF step (days)")
    mixed_max_splits: Optional[int] = Field(default=None, description="Mixed max splits")
    mixed_embargo_days: Optional[int] = Field(default=None, description="Mixed embargo window (days)")
    mixed_min_time_to_resolution_days: Optional[float] = Field(
        default=None,
        description="Mixed minimum time to resolution (days)",
    )
    mixed_alpha: Optional[float] = Field(default=None, description="Mixed ridge alpha")
    two_stage_mode: Optional[bool] = Field(default=None, description="Enable two-stage Polymarket overlay")
    two_stage_prn_csv: Optional[str] = Field(default=None, description="Override pRN dataset for two-stage")
    two_stage_pm_csv: Optional[str] = Field(default=None, description="Polymarket dataset for two-stage")
    two_stage_label_col: Optional[str] = Field(default=None, description="Override label column for two-stage")

    @validator("two_stage_pm_csv")
    def validate_two_stage_pm_csv(cls, v, values):
        """Validate that two_stage_mode requires two_stage_pm_csv."""
        if values.get("two_stage_mode") and not v:
            raise ValueError(
                "two_stage_mode requires two_stage_pm_csv (Polymarket dataset path). "
                "Please select a Polymarket dataset for the two-stage overlay."
            )
        return v

    @validator("model_kind")
    def validate_model_kind_with_two_stage(cls, v, values):
        """Validate that two_stage_mode cannot be used with mixed-only model kind."""
        if values.get("two_stage_mode") and v == "mixed":
            raise ValueError(
                "two_stage_mode cannot be used with model_kind='mixed' alone. "
                "Use model_kind='calibrate' or 'both' instead."
            )
        return v


class RenameModelRequest(BaseModel):
    new_name: str = Field(..., description="New name for the model folder.")


class AutoModelRunRequest(BaseModel):
    csv: str = Field(..., description="Relative path to dataset CSV.")
    mode: Literal["option_only", "mixed"] = Field(
        default="option_only",
        description="Feature mode for auto calibration (options only vs mixed PM+options).",
    )
    pm_dataset_path: Optional[str] = Field(
        default=None,
        description="Polymarket dataset path (required for mixed auto runs).",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Output folder name for best model under src/data/models.",
    )
    model_kind: Optional[Literal["calibrate", "mixed", "both"]] = Field(
        default=None,
        description="Run calibrator, mixed model, or both.",
    )
    objective: Literal["logloss", "roll_val_logloss", "test_logloss"] = Field(
        default="logloss",
        description="Objective the tuner optimizes.",
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
    mixed_features: Optional[str] = Field(default=None, description="Decision features path for mixed model")
    mixed_out_dir: Optional[str] = Field(default=None, description="Output root directory for mixed models")
    mixed_run_id: Optional[str] = Field(default=None, description="Optional mixed model run id")
    mixed_model: Optional[Literal["residual", "blend"]] = Field(
        default=None,
        description="Mixed model type (residual or blend).",
    )
    mixed_pm_col: Optional[str] = Field(default=None, description="Polymarket probability column")
    mixed_prn_col: Optional[str] = Field(default=None, description="pRN probability column")
    mixed_label_col: Optional[str] = Field(default=None, description="Label column")
    mixed_features_cols: Optional[str] = Field(default=None, description="Comma-separated feature columns")
    mixed_train_frac: Optional[float] = Field(default=None, description="Train fraction for mixed model")
    mixed_walk_forward: Optional[bool] = Field(default=None, description="Enable mixed walk-forward")
    mixed_wf_train_days: Optional[int] = Field(default=None, description="Mixed WF train window (days)")
    mixed_wf_test_days: Optional[int] = Field(default=None, description="Mixed WF test window (days)")
    mixed_wf_step_days: Optional[int] = Field(default=None, description="Mixed WF step (days)")
    mixed_max_splits: Optional[int] = Field(default=None, description="Mixed max splits")
    mixed_embargo_days: Optional[int] = Field(default=None, description="Mixed embargo window (days)")
    mixed_min_time_to_resolution_days: Optional[float] = Field(
        default=None,
        description="Mixed minimum time to resolution (days)",
    )
    mixed_alpha: Optional[float] = Field(default=None, description="Mixed ridge alpha")

    @validator("pm_dataset_path", always=True)
    def validate_pm_dataset_path(cls, v, values):
        if values.get("mode") == "mixed" and not v:
            raise ValueError("mode='mixed' requires pm_dataset_path (Polymarket dataset path).")
        return v


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
    two_stage_metrics: Optional[List[Dict[str, Any]]] = None
    is_two_stage: bool = False
    stage1_equation: Optional[str] = None
    two_stage_equation: Optional[str] = None


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
    two_stage_metrics: Optional[List[Dict[str, Any]]] = None
    is_two_stage: bool = False
    stage1_equation: Optional[str] = None
    two_stage_equation: Optional[str] = None


class ProgressPayload(BaseModel):
    stage: str
    trials_total: int
    trials_done: int
    trials_failed: int
    best_score_so_far: Optional[float] = None
    last_error: Optional[str] = None


class CalibrationJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed"]
    mode: Literal["manual", "auto"]
    result: Optional[CalibrateModelRunResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: Optional[ProgressPayload] = None


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


# v2.0 Model Training
class TrainModelV2Request(BaseModel):
    training_mode: Literal["pretrain", "finetune", "joint", "two_stage"] = Field(
        default="two_stage",
        description="Training workflow mode.",
    )
    feature_sources: Literal["options", "pm", "both"] = Field(
        default="both",
        description="Which feature sets to use.",
    )
    prn_csv: str = Field(..., description="Path to pRN/options dataset CSV.")
    pm_csv: Optional[str] = Field(default=None, description="Path to Polymarket features dataset.")
    out_dir: str = Field(..., description="Output directory for model artifacts.")
    overlap_window: str = Field(default="90days", description="PM overlap window (e.g., '90days', '3months').")
    numeric_features: Optional[List[str]] = Field(default=None, description="Override numeric features list.")
    pm_features: Optional[List[str]] = Field(default=None, description="Override PM features list.")
    compute_edge: bool = Field(default=True, description="Compute and save edge predictions.")
    test_weeks: int = Field(default=20, description="Number of test weeks for validation.")
    random_state: int = Field(default=7, description="Random seed for reproducibility.")
    calibrate: Optional[str] = Field(default=None, description="Calibration mode (none, platt).")
    label_col: Optional[str] = Field(default=None, description="Override label column name.")


class EdgePrediction(BaseModel):
    ticker: str
    threshold: float
    expiry_date: str
    snapshot_date: str
    p_base: float
    p_pm: Optional[float] = None
    p_final: float
    edge: Optional[float] = None
    edge_lower: Optional[float] = None
    edge_upper: Optional[float] = None


class EdgePredictionsResponse(BaseModel):
    model_id: str
    count: int
    predictions: List[EdgePrediction]


class DatasetTickersResponse(BaseModel):
    dataset: str = Field(..., description="Dataset name/path")
    tickers: List[str] = Field(..., description="Unique tickers in dataset")
    count: int = Field(..., description="Number of unique tickers")


class FeatureStat(BaseModel):
    missing_pct: float = Field(..., description="Percentage of missing values")
    dtype: str = Field(..., description="Data type")
    nunique: int = Field(..., description="Number of unique values")


class RegimeInfo(BaseModel):
    tdays_mode: Optional[List[int]] = Field(None, description="Most common T_days values")
    is_weekly: Optional[bool] = Field(None, description="True if dataset is weekly regime")
    is_daily: Optional[bool] = Field(None, description="True if dataset is 1DTE/daily regime")


class DatasetFeaturesResponse(BaseModel):
    dataset: str = Field(..., description="Dataset name/path")
    available_columns: List[str] = Field(..., description="All column names in dataset")
    feature_stats: Dict[str, FeatureStat] = Field(..., description="Statistics per column")
    regime_info: RegimeInfo = Field(..., description="Regime detection info")
