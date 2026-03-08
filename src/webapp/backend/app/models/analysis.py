from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AnalysisRefreshRequest(BaseModel):
    run_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional weekly-history run ids to ingest. Defaults to all runs.",
    )
    skip_trade_backfill: bool = Field(
        default=False,
        description="Skip subgraph trade backfill and keep volume coverage as observed from local data only.",
    )
    force_full_rebuild: bool = Field(
        default=False,
        description="Clear and rebuild raw/mart/research tables before loading runs.",
    )
    notes_author: Optional[str] = Field(
        default=None,
        description="Optional author recorded in refresh audit metadata.",
    )


class AnalysisRefreshResult(BaseModel):
    ok: bool
    refresh_id: Optional[str]
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]


class AnalysisProgress(BaseModel):
    stage: Optional[str] = None
    current: int = 0
    total: int = 0
    detail: Optional[str] = None


class AnalysisJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed", "cancelled"]
    progress: Optional[AnalysisProgress] = None
    result: Optional[AnalysisRefreshResult] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class AnalysisCoverageSummary(BaseModel):
    latest_refresh_id: Optional[str] = None
    latest_refresh_at: Optional[str] = None
    volume_authoritative: bool = False
    market_day_count: int = 0
    stock_day_count: int = 0
    global_day_count: int = 0
    trade_coverage_ratio: Optional[float] = None
    stock_count: int = 0
    active_market_count: int = 0


class AnalysisHeadlineMetric(BaseModel):
    label: str
    value: Optional[float | int | str] = None
    delta: Optional[float] = None
    suffix: Optional[str] = None


class AnalysisSeriesPoint(BaseModel):
    date: str
    value: Optional[float] = None
    value_2: Optional[float] = None
    value_3: Optional[float] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisHistogramBin(BaseModel):
    start: float
    end: float
    count: int


class AnalysisThresholdRow(BaseModel):
    trade_date_ny: str
    ticker: Optional[str] = None
    variable_name: str
    scope_used: str
    coverage_class: Optional[str] = None
    upper_cap: Optional[float] = None
    q95: Optional[float] = None
    q99: Optional[float] = None
    q995: Optional[float] = None
    sample_count: Optional[int] = None
    sparse_flag: bool = False
    authoritative_flag: bool = False


class AnalysisAlert(BaseModel):
    level: Literal["info", "warning", "critical"]
    title: str
    detail: str


class AnalysisOverviewResponse(BaseModel):
    coverage: AnalysisCoverageSummary
    headline_metrics: List[AnalysisHeadlineMetric]
    recent_trends: List[AnalysisSeriesPoint]
    alerts: List[AnalysisAlert]
    available_tickers: List[str] = Field(default_factory=list)


class AnalysisVolumeResponse(BaseModel):
    coverage: AnalysisCoverageSummary
    selected_variable: str
    selected_scope_level: str
    total_volume_series: List[AnalysisSeriesPoint]
    per_stock_series: List[AnalysisSeriesPoint]
    hist_raw: List[AnalysisHistogramBin]
    hist_log: List[AnalysisHistogramBin]
    outlier_diagnostics: List[Dict[str, Any]]
    threshold_rows: List[AnalysisThresholdRow]
    clipped_share: Optional[float] = None


class AnalysisStructureResponse(BaseModel):
    coverage: AnalysisCoverageSummary
    stock_concentration: List[AnalysisSeriesPoint]
    active_market_counts: List[AnalysisSeriesPoint]
    stock_participation: List[AnalysisSeriesPoint]
    lifecycle_summary: List[Dict[str, Any]]


class AnalysisPriceBehaviorResponse(BaseModel):
    coverage: AnalysisCoverageSummary
    close_probability_distribution: List[AnalysisHistogramBin]
    volatility_series: List[AnalysisSeriesPoint]
    expiry_behavior: List[AnalysisSeriesPoint]
    convergence_table: List[Dict[str, Any]]


class AnalysisDriftResponse(BaseModel):
    coverage: AnalysisCoverageSummary
    selected_variable: str
    rolling_statistics: List[AnalysisSeriesPoint]
    rolling_quantiles: List[AnalysisSeriesPoint]
    drift_rows: List[Dict[str, Any]]
    break_rows: List[Dict[str, Any]]
    alerts: List[AnalysisAlert]


class AnalysisTableRow(BaseModel):
    values: Dict[str, Any]


class AnalysisTableResponse(BaseModel):
    table: str
    page: int
    page_size: int
    total_rows: int
    rows: List[AnalysisTableRow]


class ResearchNoteCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=240)
    body: str = Field(..., min_length=1)
    author: Optional[str] = Field(default=None, max_length=120)
    tags: List[str] = Field(default_factory=list)
    pinned: bool = False


class ResearchNoteResponse(BaseModel):
    note_id: int
    title: str
    body: str
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    pinned: bool = False
    created_at_utc: str
    updated_at_utc: str


class ResearchNotesResponse(BaseModel):
    notes: List[ResearchNoteResponse]
