from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DatasetRunRequest(BaseModel):
    out_dir: Optional[str] = Field(
        default=None,
        description="Output directory for the dataset (relative to project root).",
    )
    out_name: Optional[str] = Field(
        default=None,
        description="Output CSV filename for the dataset.",
    )
    run_dir_name: Optional[str] = Field(
        default=None,
        description="Optional run directory name override (basename only).",
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Kebab-case dataset name. Overrides out_name/run_dir_name/train_view_name with {purpose}-{name}.csv convention.",
    )
    training_dataset: Optional[str] = Field(
        default=None,
        description="Training dataset selection (legacy or train_view). Deprecated when dataset_name is set.",
    )
    schedule_mode: Optional[str] = Field(
        default=None,
        description="Snapshot schedule mode (weekly or expiry_range).",
    )
    expiry_weekdays: Optional[str] = Field(
        default=None,
        description="Comma-separated expiry weekdays.",
    )
    asof_weekdays: Optional[str] = Field(
        default=None,
        description="Comma-separated as-of weekdays.",
    )
    dte_list: Optional[str] = Field(
        default=None,
        description="Comma-separated DTE list (supports ranges).",
    )
    dte_min: Optional[int] = Field(
        default=None,
        description="Minimum DTE (inclusive).",
    )
    dte_max: Optional[int] = Field(
        default=None,
        description="Maximum DTE (inclusive).",
    )
    dte_step: Optional[int] = Field(
        default=None,
        description="DTE step for range.",
    )
    write_snapshot: Optional[bool] = Field(
        default=None,
        description="Write snapshot.csv output.",
    )
    write_prn_view: Optional[bool] = Field(
        default=None,
        description="Write prn_view.csv output.",
    )
    write_train_view: Optional[bool] = Field(
        default=None,
        description="Write train_view.csv output.",
    )
    write_legacy: Optional[bool] = Field(
        default=None,
        description="Write legacy CSV output (backward compatible).",
    )
    prn_version: Optional[str] = Field(
        default=None,
        description="pRN version tag.",
    )
    prn_config_hash: Optional[str] = Field(
        default=None,
        description="pRN config hash override.",
    )
    train_view_name: Optional[str] = Field(
        default=None,
        description="Filename for train_view output.",
    )
    drops_name: Optional[str] = Field(
        default=None,
        description="Output filename for dropped rows report.",
    )
    tickers: Optional[str] = Field(
        default=None,
        description="Comma-separated list of tickers.",
    )
    start: str = Field(..., description="Start date (YYYY-MM-DD).")
    end: str = Field(..., description="End date (YYYY-MM-DD).")

    theta_base_url: Optional[str] = Field(
        default=None,
        description="Theta Terminal base URL.",
    )
    stock_source: Optional[str] = Field(
        default=None,
        description="Stock source (yfinance/theta/auto).",
    )
    timeout_s: Optional[int] = Field(default=None, description="Request timeout in seconds.")
    r: Optional[float] = Field(default=None, description="Risk-free rate.")

    max_abs_logm: Optional[float] = Field(default=None)
    max_abs_logm_cap: Optional[float] = Field(default=None)
    band_widen_step: Optional[float] = Field(default=None)
    no_adaptive_band: Optional[bool] = Field(default=None)
    max_band_strikes: Optional[int] = Field(default=None)

    min_band_strikes: Optional[int] = Field(default=None)
    min_band_prn_strikes: Optional[int] = Field(default=None)

    strike_range: Optional[int] = Field(default=None)
    no_retry_full_chain: Optional[bool] = Field(default=None)
    no_sat_expiry_fallback: Optional[bool] = Field(default=None)
    threads: Optional[int] = Field(default=None)

    prefer_bidask: Optional[bool] = Field(default=None)
    min_trade_count: Optional[int] = Field(default=None)
    min_volume: Optional[int] = Field(default=None)

    min_chain_used_hard: Optional[int] = Field(default=None)
    max_rel_spread_median_hard: Optional[float] = Field(default=None)
    hard_drop_close_fallback: Optional[bool] = Field(default=None)

    min_prn_train: Optional[float] = Field(default=None)
    max_prn_train: Optional[float] = Field(default=None)

    no_split_adjust: Optional[bool] = Field(default=None)

    dividend_source: Optional[str] = Field(default=None)
    dividend_lookback_days: Optional[int] = Field(default=None)
    dividend_yield_default: Optional[float] = Field(default=None)
    no_forward_moneyness: Optional[bool] = Field(default=None)

    no_group_weights: Optional[bool] = Field(default=None)
    no_ticker_weights: Optional[bool] = Field(default=None)
    no_soft_quality_weight: Optional[bool] = Field(default=None)

    rv_lookback_days: Optional[int] = Field(default=None)

    ticker_reweight_mode: Optional[str] = Field(default=None)
    ticker_reweight_alpha_min: Optional[float] = Field(default=None)
    ticker_reweight_alpha_max: Optional[float] = Field(default=None)
    trade_focus_beta: Optional[float] = Field(default=None)
    trade_focus_tickers: Optional[str] = Field(default=None)

    cache: Optional[bool] = Field(default=None)

    write_drops: Optional[bool] = Field(default=None)

    sanity_report: Optional[bool] = Field(default=None)
    sanity_drop: Optional[bool] = Field(default=None)
    sanity_abs_logm_max: Optional[float] = Field(default=None)
    sanity_k_over_s_min: Optional[float] = Field(default=None)
    sanity_k_over_s_max: Optional[float] = Field(default=None)

    verbose_skips: Optional[bool] = Field(default=None)


class DatasetRunResponse(BaseModel):
    ok: bool
    out_dir: str
    out_name: str
    run_dir: Optional[str] = None
    output_file: Optional[str]
    drops_file: Optional[str]
    training_file: Optional[str] = None
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]


class DatasetBackfillRange(BaseModel):
    start: str = Field(..., description="Backfill start date (YYYY-MM-DD).")
    end: str = Field(..., description="Backfill end date (YYYY-MM-DD).")


class DatasetBackfillRequest(BaseModel):
    run_dir: str = Field(..., description="Option chain dataset run directory.")
    polymarket_run_id: str = Field(..., description="Polymarket weekly history run id.")
    allow_defaults: bool = Field(
        default=False,
        description="Allow backfill using default builder settings when config metadata is missing.",
    )


class DatasetBackfillResponse(BaseModel):
    ok: bool
    backfilled: bool
    run_dir: str
    training_file: Optional[str] = None
    used_defaults: bool = False
    required_start: Optional[str] = None
    required_end: Optional[str] = None
    existing_start: Optional[str] = None
    existing_end: Optional[str] = None
    backfill_ranges: List[DatasetBackfillRange] = Field(default_factory=list)
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    rows_added: Optional[int] = None
    message: str
    duration_s: float


class DatasetCleanupCriteria(BaseModel):
    quality_buckets: List[Literal["clean", "watch", "noisy"]] = Field(
        default_factory=lambda: ["noisy"],
    )
    min_quality_issue_count: Optional[int] = 3
    flag_columns: List[str] = Field(default_factory=list)
    flag_match_mode: Optional[Literal["any", "all"]] = "any"
    min_rel_spread_median: Optional[float] = None
    max_n_chain_used: Optional[float] = None


class DatasetCleanupRequest(BaseModel):
    run_dir: str = Field(..., description="Option chain dataset run directory.")
    criteria: DatasetCleanupCriteria = Field(
        default_factory=DatasetCleanupCriteria,
        description="Rules used to drop noisy rows from the selected training dataset.",
    )
    allow_defaults: bool = Field(
        default=False,
        description="Allow cleanup to use default weighting settings when build metadata is missing.",
    )


class DatasetCleanupPreviewResponse(BaseModel):
    run_dir: str
    training_file: str
    rows_before: int
    rows_to_drop: int
    rows_after: int
    drop_share: float
    used_defaults: bool = False
    would_drop_all: bool = False
    dropped_bucket_counts: Dict[str, int] = Field(default_factory=dict)
    matched_flag_counts: List["DatasetAuditFlagSummary"] = Field(default_factory=list)
    sample_rows: List["DatasetAuditRow"] = Field(default_factory=list)
    message: str


class DatasetCleanupResponse(DatasetCleanupPreviewResponse):
    ok: bool
    cleaned_run_dir: str
    cleaned_file: str


class DatasetFileSummary(BaseModel):
    name: str
    path: str
    size_bytes: int
    last_modified: Optional[str]


class DatasetRunSummary(BaseModel):
    id: str
    run_dir: str
    dataset_file: Optional[DatasetFileSummary]
    drops_file: Optional[DatasetFileSummary]
    training_file: Optional[DatasetFileSummary] = None
    files: List[DatasetFileSummary] = Field(default_factory=list)
    last_modified: Optional[str]
    status: Literal["ready", "creating"] = "ready"
    job_id: Optional[str] = None


class DatasetRunRenameRequest(BaseModel):
    run_dir: str = Field(..., description="Run directory to rename.")
    new_name: str = Field(
        ...,
        description="New run directory name (basename only).",
    )


class DatasetListResponse(BaseModel):
    base_dir: str
    runs: List[DatasetRunSummary]


class DatasetPreviewResponse(BaseModel):
    file: DatasetFileSummary
    headers: List[str]
    rows: List[Dict[str, Optional[str]]]
    row_count: Optional[int]
    mode: Literal["head", "tail"]
    limit: int


class DatasetAuditFlagSummary(BaseModel):
    name: str
    count: int
    share: float


class DatasetAuditDistribution(BaseModel):
    name: str
    min: Optional[float] = None
    p05: Optional[float] = None
    p50: Optional[float] = None
    p95: Optional[float] = None
    max: Optional[float] = None


class DatasetAuditTickerSummary(BaseModel):
    ticker: str
    row_count: int
    snapshot_count: Optional[int] = None
    avg_issue_count: Optional[float] = None
    flagged_share: Optional[float] = None
    fallback_share: Optional[float] = None
    wide_spread_share: Optional[float] = None
    clean_share: Optional[float] = None
    watch_share: Optional[float] = None
    noisy_share: Optional[float] = None


class DatasetAuditTimelinePoint(BaseModel):
    asof_date: str
    row_count: int
    snapshot_count: Optional[int] = None
    avg_issue_count: Optional[float] = None
    flagged_share: Optional[float] = None


class DatasetAuditRow(BaseModel):
    row_id: Optional[str] = None
    ticker: Optional[str] = None
    asof_date: Optional[str] = None
    expiry_date: Optional[str] = None
    K: Optional[float] = None
    pRN: Optional[float] = None
    quality_issue_count: Optional[float] = None
    rel_spread_median: Optional[float] = None
    n_chain_used: Optional[float] = None
    flags: List[str] = Field(default_factory=list)


class DatasetAuditHeatmapCell(BaseModel):
    ticker: str
    asof_date: str
    row_count: int
    flagged_share: Optional[float] = None
    avg_issue_count: Optional[float] = None
    quality_bucket: Optional[str] = None


class DatasetAuditRvBucketSummary(BaseModel):
    label: Literal["low", "mid", "high"]
    value_min: Optional[float] = None
    value_max: Optional[float] = None
    row_count: int
    row_share: Optional[float] = None
    avg_issue_count: Optional[float] = None
    flagged_share: Optional[float] = None


class DatasetAuditRvFeatureAudit(BaseModel):
    feature: str
    finite_row_count: int
    finite_row_share: Optional[float] = None
    buckets: List[DatasetAuditRvBucketSummary] = Field(default_factory=list)


class DatasetRowDetailResponse(BaseModel):
    file: DatasetFileSummary
    row_id: str
    row: Dict[str, Optional[str]]


class DatasetAuditResponse(BaseModel):
    file: DatasetFileSummary
    row_count: int
    column_count: int
    ticker_count: Optional[int] = None
    snapshot_count: Optional[int] = None
    group_count: Optional[int] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    expiry_start: Optional[str] = None
    expiry_end: Optional[str] = None
    available_rv_features: List[str] = Field(default_factory=list)
    available_quality_flags: List[str] = Field(default_factory=list)
    quality_flags: List[DatasetAuditFlagSummary] = Field(default_factory=list)
    numeric_distributions: List[DatasetAuditDistribution] = Field(default_factory=list)
    rv_feature_audit: List[DatasetAuditRvFeatureAudit] = Field(default_factory=list)
    top_problem_tickers: List[DatasetAuditTickerSummary] = Field(default_factory=list)
    timeline: List[DatasetAuditTimelinePoint] = Field(default_factory=list)
    heatmap_dates: List[str] = Field(default_factory=list)
    heatmap_cells: List[DatasetAuditHeatmapCell] = Field(default_factory=list)
    noisiest_rows: List[DatasetAuditRow] = Field(default_factory=list)


class DatasetJobGroupChecks(BaseModel):
    asof_close_fallback: int = 0
    expiry_close_fallback: int = 0
    expiry_saturday_fallback: int = 0
    quote_close_fallback: int = 0
    low_chain_used: int = 0
    wide_rel_spread: int = 0


class DatasetJobTickerTelemetry(BaseModel):
    ticker: str
    completed_jobs: int = 0
    planned_jobs: int = 0
    kept_groups: int = 0
    rows: int = 0
    issue_count_sum: float = 0
    flagged_rows: int = 0
    fallback_rows: int = 0
    wide_spread_rows: int = 0
    clean_rows: int = 0
    watch_rows: int = 0
    noisy_rows: int = 0
    drop_reasons: Dict[str, int] = Field(default_factory=dict)


class DatasetJobTelemetry(BaseModel):
    phase: Literal[
        "planning",
        "preloading_stock",
        "preloading_dividends",
        "building",
        "finalizing",
        "writing_outputs",
        "finished",
    ] = "planning"
    drop_reasons: Dict[str, int] = Field(default_factory=dict)
    group_checks: DatasetJobGroupChecks = Field(default_factory=DatasetJobGroupChecks)
    tickers: List[DatasetJobTickerTelemetry] = Field(default_factory=list)


class DatasetJobProgress(BaseModel):
    done: int
    total: int
    groups: int
    rows: int
    lastTicker: str
    lastWeek: str
    lastAsof: str


class DatasetJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed", "cancelled"]
    progress: Optional[DatasetJobProgress]
    telemetry: Optional[DatasetJobTelemetry] = None
    stdout: List[str]
    stderr: List[str]
    result: Optional[DatasetRunResponse]
    error: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


for _model in (DatasetCleanupPreviewResponse, DatasetCleanupResponse):
    _rebuild = getattr(_model, "model_rebuild", None)
    if callable(_rebuild):
        _rebuild()
    else:
        _model.update_forward_refs()
