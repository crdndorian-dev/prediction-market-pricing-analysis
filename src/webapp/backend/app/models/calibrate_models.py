from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator


class DatasetFileSummary(BaseModel):
    name: str
    path: str
    size_bytes: int
    last_modified: Optional[str]
    dataset_id: Optional[str] = None
    rows: Optional[int] = None
    date_col_used: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    week_count: Optional[int] = None
    ticker_count: Optional[int] = None
    ticker_sample: Optional[List[str]] = None
    available_weight_columns: Optional[List[str]] = None
    available_grouping_keys: Optional[List[str]] = None


class DatasetListResponse(BaseModel):
    base_dir: str
    datasets: List[DatasetFileSummary]


class ModelRunSummary(BaseModel):
    id: str
    path: str
    last_modified: Optional[str]
    has_metadata: bool
    has_metrics: bool
    dataset_id: Optional[str] = None
    dataset_path: Optional[str] = None
    train_date_start: Optional[str] = None
    train_date_end: Optional[str] = None
    tickers_summary: Optional[str] = None
    dow_regime: Optional[str] = None
    split_strategy: Optional[str] = None
    c_value: Optional[float] = None
    calibration_method: Optional[str] = None
    weighting_mode: Optional[str] = None
    is_two_stage: Optional[bool] = None
    run_type: Optional[Literal["manual", "auto"]] = None
    auto_status: Optional[str] = None
    selected_trial_id: Optional[int] = None
    has_selected_model: Optional[bool] = None


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


class SplitConfig(BaseModel):
    strategy: Literal["walk_forward", "single_holdout"] = "walk_forward"
    window_mode: Literal["rolling", "expanding"] = "rolling"
    train_window_weeks: Optional[int] = None
    validation_folds: Optional[int] = None
    validation_window_weeks: Optional[int] = None
    test_window_weeks: Optional[int] = None
    embargo_days: Optional[int] = None


class RegularizationConfig(BaseModel):
    c_grid: Optional[str] = None
    calibration_method: Optional[Literal["none", "platt"]] = None
    selection_objective: Optional[Literal["logloss", "brier", "ece_q"]] = None


class ModelStructureConfig(BaseModel):
    trading_universe_tickers: Optional[str] = None
    train_tickers: Optional[str] = None
    foundation_tickers: Optional[str] = None
    foundation_weight: Optional[float] = None
    ticker_intercepts: Optional[Literal["none", "all", "non_foundation"]] = None
    ticker_x_interactions: Optional[bool] = None
    ticker_min_support: Optional[int] = None
    ticker_min_support_interactions: Optional[int] = None


class WeightingConfig(BaseModel):
    base_weight_source: Optional[Literal["dataset_weight", "uniform"]] = None
    grouping_key: Optional[str] = None
    group_equalization: Optional[bool] = None
    renorm: Optional[Literal["mean1"]] = None
    trading_universe_upweight: Optional[float] = None
    ticker_balance_mode: Optional[Literal["none", "sqrt_inv_clipped"]] = None


class BootstrapConfig(BaseModel):
    bootstrap_ci: Optional[bool] = None
    bootstrap_group: Optional[str] = None
    bootstrap_b: Optional[int] = None
    bootstrap_seed: Optional[int] = None
    ci_level: Optional[Literal[90, 95, 99]] = None
    per_split_reporting: Optional[bool] = None
    per_fold_reporting: Optional[bool] = None
    allow_iid_bootstrap: Optional[bool] = False


class DiagnosticsConfig(BaseModel):
    split_timeline: Optional[bool] = None
    per_fold_delta_chart: Optional[bool] = None
    per_group_delta_distribution: Optional[bool] = None


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
    val_split_mode: Optional[str] = Field(
        default=None,
        description="Validation split mode for base calibrator: week_group or row_tail",
    )
    val_weeks: Optional[int] = Field(
        default=None,
        description="Optional explicit number of validation weeks when val_split_mode=week_group.",
    )
    n_bins: Optional[int] = Field(default=None)
    eceq_bins: Optional[int] = Field(default=None)
    selection_objective: Optional[str] = Field(default=None)
    strict_args: Optional[bool] = Field(
        default=None,
        description="Fail run when unsupported calibrator args are passed.",
    )
    fallback_to_baseline_if_worse: Optional[bool] = Field(default=None)
    auto_drop_near_constant: Optional[bool] = Field(default=None)
    random_state: Optional[int] = Field(default=None)
    metrics_top_tickers: Optional[int] = Field(default=None)
    enable_x_abs_m: Optional[bool] = Field(default=None, description="Enable x_abs_m interaction feature")
    group_reweight: Optional[str] = Field(
        default=None,
        description="Group reweighting mode: none or chain_snapshot",
    )
    max_abs_logm: Optional[float] = Field(default=None, description="Maximum absolute log-moneyness filter")
    drop_prn_extremes: Optional[bool] = Field(default=None, description="Drop pRN extremes near 0 or 1")
    prn_eps: Optional[float] = Field(default=None, description="Epsilon for pRN extremes filter")
    prn_below: Optional[float] = Field(default=None, description="Drop rows where pRN <= this threshold")
    prn_above: Optional[float] = Field(default=None, description="Drop rows where pRN >= this threshold")
    bootstrap_ci: Optional[bool] = Field(default=None, description="Enable bootstrap CIs for delta metrics")
    bootstrap_b: Optional[int] = Field(default=None, description="Number of bootstrap resamples")
    bootstrap_seed: Optional[int] = Field(default=None, description="Bootstrap random seed")
    bootstrap_group: Optional[str] = Field(default=None, description="Bootstrap grouping: auto|ticker_day|day|iid")
    allow_defaults: Optional[bool] = Field(default=False, description="Allow requested config to fall back to defaults.")
    allow_iid_bootstrap: Optional[bool] = Field(
        default=False,
        description="Allow IID bootstrap fallback when requested grouped bootstrap key is missing.",
    )
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
    run_mode: Optional[Literal["manual", "auto_search"]] = Field(default=None)
    random_seed: Optional[int] = Field(default=None)
    weight_col_strategy: Optional[Literal["auto", "weight_final", "sample_weight_final", "uniform"]] = Field(
        default=None
    )
    split: Optional[SplitConfig] = Field(default=None)
    regularization: Optional[RegularizationConfig] = Field(default=None)
    model_structure: Optional[ModelStructureConfig] = Field(default=None)
    weighting: Optional[WeightingConfig] = Field(default=None)
    bootstrap: Optional[BootstrapConfig] = Field(default=None)
    diagnostics: Optional[DiagnosticsConfig] = Field(default=None)

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


class AutoSearchConfig(BaseModel):
    feature_sets: Optional[List[List[str]]] = Field(default=None, description="List of feature sets to evaluate.")
    c_values: Optional[List[float]] = Field(default=None, description="Candidate C values for L2 logit.")
    calibration_methods: Optional[List[Literal["none", "platt"]]] = Field(
        default=None,
        description="Calibration methods to try.",
    )
    trading_universe_upweight: Optional[List[float]] = Field(
        default=None,
        description="Trading-universe upweight multipliers to test.",
    )
    foundation_weight: Optional[List[float]] = Field(
        default=None,
        description="Foundation weight multipliers to test.",
    )
    ticker_intercepts: Optional[List[Literal["off", "on"]]] = Field(
        default=None,
        description="Toggle ticker intercepts in search grid.",
    )
    allow_risky_features: bool = Field(
        default=False,
        description="Allow risky features (had_*, prn_raw_gap) in search grid.",
    )
    advanced_interactions: bool = Field(
        default=False,
        description="Include per-ticker interactions in the search grid.",
    )
    max_trials: Optional[int] = Field(default=None, description="Limit number of trials.")
    selection_rule: Optional[Literal["one_se", "epsilon"]] = Field(
        default="one_se",
        description="Selection rule applied to validation scores.",
    )
    epsilon: Optional[float] = Field(default=0.002, description="Epsilon margin for selection rule.")
    accept_delta_threshold: Optional[float] = Field(
        default=0.0,
        description="Maximum accepted central validation delta logloss (model - baseline).",
    )
    min_improve_fraction: Optional[float] = Field(
        default=0.5,
        description="Minimum fraction of folds that must improve to accept a candidate.",
    )
    max_worst_delta: Optional[float] = Field(
        default=0.01,
        description="Maximum allowed worst-fold validation delta logloss.",
    )
    outer_folds: Optional[int] = Field(default=None, description="Number of outer backtest folds.")
    outer_test_weeks: Optional[int] = Field(default=None, description="Test window size (weeks) per outer fold.")
    outer_gap_weeks: Optional[int] = Field(default=None, description="Gap window size (weeks) between train and test.")
    outer_selection_metric: Optional[
        Literal["median_delta_logloss", "worst_delta_logloss", "mean_delta_logloss"]
    ] = Field(default=None, description="Metric used to select models under outer CV.")
    outer_min_improve_fraction: Optional[float] = Field(
        default=None,
        description="Minimum fraction of outer folds that must improve.",
    )
    outer_max_worst_delta: Optional[float] = Field(
        default=None,
        description="Maximum allowed worst-fold delta logloss in outer CV.",
    )


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
    base_config: Optional[CalibrateModelRunRequest] = Field(
        default=None,
        description="Base calibration config (splits, weights, filters) for auto-search.",
    )
    search: Optional[AutoSearchConfig] = Field(
        default=None,
        description="Auto-search grid and selection configuration.",
    )

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


class WeightingPreviewRequest(BaseModel):
    csv: str = Field(..., description="Relative path to dataset CSV.")
    weight_col_strategy: Optional[Literal["auto", "weight_final", "sample_weight_final", "uniform"]] = Field(
        default="auto"
    )
    base_weight_source: Optional[Literal["dataset_weight", "uniform"]] = Field(default="dataset_weight")
    grouping_key: Optional[str] = Field(default="group_id")
    group_equalization: Optional[bool] = Field(default=True)
    trading_universe_tickers: Optional[str] = Field(default=None)
    trading_universe_upweight: Optional[float] = Field(default=1.0)
    ticker_balance_mode: Optional[Literal["none", "sqrt_inv_clipped"]] = Field(default="none")
    split_strategy: Optional[Literal["walk_forward", "single_holdout"]] = Field(default="walk_forward")
    test_window_weeks: Optional[int] = Field(default=20)
    validation_window_weeks: Optional[int] = Field(default=8)


class WeightingPreviewResponse(BaseModel):
    selected_weight_column: Optional[str] = None
    min_weight: float
    mean_weight: float
    max_weight: float
    group_sum_min: Optional[float] = None
    group_sum_mean: Optional[float] = None
    group_sum_max: Optional[float] = None
    split_group_counts: Dict[str, int] = Field(default_factory=dict)
    split_row_counts: Dict[str, int] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


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
    split_row_counts: Optional[Dict[str, int]] = None
    split_group_counts: Optional[Dict[str, int]] = None
    model_equation: Optional[str] = None
    model_equation_spec: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    feature_manifest: Optional[Dict[str, Any]] = None
    two_stage_metrics: Optional[List[Dict[str, Any]]] = None
    is_two_stage: bool = False
    stage1_equation: Optional[str] = None
    stage1_equation_spec: Optional[Dict[str, Any]] = None
    two_stage_equation: Optional[str] = None
    two_stage_equation_spec: Optional[Dict[str, Any]] = None
    combined_p_hat_equation: Optional[str] = None
    combined_p_hat_equation_spec: Optional[Dict[str, Any]] = None


class CalibrateModelRunResponse(BaseModel):
    ok: bool
    out_dir: str
    files: List[str]
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]
    metrics_summary: Optional[Dict[str, SplitMetricSummary]] = None
    split_row_counts: Optional[Dict[str, int]] = None
    split_group_counts: Optional[Dict[str, int]] = None
    auto_out_dir: Optional[str] = None
    features: Optional[List[str]] = None
    model_equation: Optional[str] = None
    model_equation_spec: Optional[Dict[str, Any]] = None
    two_stage_metrics: Optional[List[Dict[str, Any]]] = None
    is_two_stage: bool = False
    stage1_equation: Optional[str] = None
    stage1_equation_spec: Optional[Dict[str, Any]] = None
    two_stage_equation: Optional[str] = None
    two_stage_equation_spec: Optional[Dict[str, Any]] = None
    combined_p_hat_equation: Optional[str] = None
    combined_p_hat_equation_spec: Optional[Dict[str, Any]] = None
    artifact_manifest: Optional[List[Dict[str, str]]] = None
    diagnostics_available: Optional[Dict[str, bool]] = None
    warnings: Optional[List[str]] = None


class ProgressPayload(BaseModel):
    stage: str
    trials_total: int
    trials_done: int
    trials_failed: int
    best_score_so_far: Optional[float] = None
    last_error: Optional[str] = None
    phase: Optional[str] = None
    candidate_index: Optional[int] = None
    candidate_total: Optional[int] = None
    fold_index: Optional[int] = None
    fold_total: Optional[int] = None
    message: Optional[str] = None
    last_log_lines: Optional[List[str]] = None
    top_candidates: Optional[List[Dict[str, Any]]] = None


class CalibrationJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed", "cancelled"]
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
    relative_path: Optional[str] = None
    section: Optional[Literal["selected_model", "auto_search", "legacy_root"]] = None
    kind: Optional[Literal["file", "dir"]] = None


class ModelFilesListResponse(BaseModel):
    model_id: str
    files: List[ModelFileSummary]


class ModelFileContentResponse(BaseModel):
    model_id: str
    filename: str
    relative_path: Optional[str] = None
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
