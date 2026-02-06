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
    output_file: Optional[str]
    drops_file: Optional[str]
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]


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
    stdout: List[str]
    stderr: List[str]
    result: Optional[DatasetRunResponse]
    error: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
