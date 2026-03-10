from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PolymarketHistoryRunRequest(BaseModel):
    tickers: Optional[List[str]] = Field(
        default=None,
        description="Tickers to include (e.g. ['AAPL', 'NVDA']).",
    )
    tickers_csv: Optional[str] = Field(
        default=None,
        description="Relative path to CSV with a 'ticker' column.",
    )
    event_urls: Optional[List[str]] = Field(
        default=None,
        description="List of Polymarket event/market URLs or slugs to resolve via Gamma by-slug.",
    )
    event_urls_file: Optional[str] = Field(
        default=None,
        description="Relative path to text/CSV file with event URLs or slugs.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date YYYY-MM-DD (UTC).",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date YYYY-MM-DD (UTC).",
    )
    fidelity_min: Optional[int] = Field(
        default=None,
        description="CLOB price history fidelity in minutes.",
    )
    bars_freqs: Optional[str] = Field(
        default=None,
        description="Comma-separated bar frequencies (e.g. 1h,1d).",
    )
    out_dir: Optional[str] = Field(
        default=None,
        description="Output directory for weekly history runs.",
    )
    run_dir_name: Optional[str] = Field(
        default=None,
        description="Optional custom run directory name (sanitized to kebab-case).",
    )
    bars_dir: Optional[str] = Field(
        default=None,
        description="Bars output directory.",
    )
    dim_market_out: Optional[str] = Field(
        default=None,
        description="Output path for dim_market mapping.",
    )
    fact_trade_dir: Optional[str] = Field(
        default=None,
        description="Output directory for filtered subgraph trades.",
    )
    include_subgraph: bool = Field(
        default=False,
        description="Attempt subgraph trade ingest if configured.",
    )
    max_subgraph_entities: Optional[int] = Field(
        default=None,
        description="Safety cap on subgraph entities pulled.",
    )
    dry_run: bool = Field(
        default=False,
        description="Run without writing files.",
    )
    build_features: bool = Field(
        default=False,
        description="Build decision features after history completes.",
    )
    prn_dataset: Optional[str] = Field(
        default=None,
        description="Path to pRN dataset for feature building.",
    )
    skip_subgraph_labels: bool = Field(
        default=False,
        description="Skip fetching labels from the subgraph during feature build.",
    )


class PolymarketHistoryRunResponse(BaseModel):
    ok: bool
    run_id: Optional[str]
    out_dir: str
    run_dir: Optional[str]
    files: List[str]
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]
    features_built: bool = False
    features_path: Optional[str] = None
    features_manifest_path: Optional[str] = None


class PolymarketRunFeaturesRequest(BaseModel):
    prn_dataset: Optional[str] = Field(
        default=None,
        description="Path to pRN dataset CSV for decision feature building.",
    )
    skip_subgraph_labels: bool = Field(
        default=False,
        description="Skip fetching labels from the subgraph during feature build.",
    )


class PolymarketRunFeaturesResponse(BaseModel):
    ok: bool
    run_id: str
    run_dir: str
    features_built: bool = False
    features_path: Optional[str] = None
    features_manifest_path: Optional[str] = None
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]


class PolymarketHistoryProgress(BaseModel):
    total: int
    completed: int
    failed: int = 0
    status: Literal["running", "completed", "failed"]


class PolymarketHistoryJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed", "cancelled"]
    phase: Optional[Literal["history", "features", "finalizing"]] = None
    progress: Optional[PolymarketHistoryProgress] = None
    features_progress: Optional[PolymarketHistoryProgress] = None
    result: Optional[PolymarketHistoryRunResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class PolymarketDailyAnalyticsDay(BaseModel):
    date: str = Field(..., description="UTC calendar date (YYYY-MM-DD).")
    weekday: str = Field(..., description="Weekday label for the UTC date.")
    daily_notional_volume: float = Field(..., description="Observed notional trade volume for the day.")
    share_volume: float = Field(..., description="Observed share volume for the day.")
    trade_count: int = Field(..., description="Observed trade count for the day.")
    active_markets: int = Field(..., description="Distinct markets with trades on the day.")
    active_tickers: int = Field(..., description="Distinct tickers with trades on the day.")
    expected_notional_volume: Optional[float] = Field(
        default=None,
        description="Expected notional trade volume from the rolling empirical baseline.",
    )
    band_lo: Optional[float] = Field(default=None, description="Lower confidence band for the day.")
    band_hi: Optional[float] = Field(default=None, description="Upper confidence band for the day.")
    realized_expected_ratio: Optional[float] = Field(
        default=None,
        description="Observed volume divided by expected volume.",
    )
    day_over_day_delta: Optional[float] = Field(
        default=None,
        description="Observed notional delta versus the prior calendar day.",
    )
    day_over_day_pct: Optional[float] = Field(
        default=None,
        description="Observed notional delta percentage versus the prior calendar day.",
    )
    noise_cv_28d: Optional[float] = Field(
        default=None,
        description="Rolling 28-day coefficient of variation for observed volume.",
    )
    noise_mad_ratio_28d: Optional[float] = Field(
        default=None,
        description="Rolling 28-day MAD/median ratio for observed volume.",
    )
    unusually_active: bool = Field(
        default=False,
        description="Whether observed volume exceeded the upper confidence band.",
    )
    unusually_inactive: bool = Field(
        default=False,
        description="Whether observed volume fell below the lower confidence band.",
    )
    flagged_outlier: bool = Field(
        default=False,
        description="Robust outlier flag from a rolling median/MAD diagnostic.",
    )
    baseline_source: str = Field(
        default="rolling_weekday_median",
        description="Baseline source used for the day's expected volume.",
    )


class PolymarketDailyAnalyticsSnapshot(BaseModel):
    date: str = Field(..., description="UTC calendar date (YYYY-MM-DD).")
    weekday: str = Field(..., description="Weekday label for the UTC date.")
    daily_notional_volume: float = Field(..., description="Observed notional trade volume for the day.")
    share_volume: float = Field(..., description="Observed share volume for the day.")
    trade_count: int = Field(..., description="Observed trade count for the day.")
    active_markets: int = Field(..., description="Distinct markets with trades on the day.")
    active_tickers: int = Field(..., description="Distinct tickers with trades on the day.")
    expected_notional_volume: Optional[float] = None
    band_lo: Optional[float] = None
    band_hi: Optional[float] = None
    realized_expected_ratio: Optional[float] = None
    day_over_day_delta: Optional[float] = None
    day_over_day_pct: Optional[float] = None
    noise_cv_28d: Optional[float] = None
    noise_mad_ratio_28d: Optional[float] = None
    unusually_active: bool = False
    unusually_inactive: bool = False
    flagged_outlier: bool = False
    baseline_source: str = "rolling_weekday_median"


class PolymarketDailyAnalyticsSummary(BaseModel):
    latest_day: Optional[PolymarketDailyAnalyticsSnapshot] = None
    comparison_day: Optional[PolymarketDailyAnalyticsSnapshot] = None
    delta_notional: Optional[float] = Field(
        default=None,
        description="Latest-day notional minus comparison-day notional.",
    )
    delta_notional_pct: Optional[float] = Field(
        default=None,
        description="Latest-day notional delta divided by comparison-day notional.",
    )
    delta_share_volume: Optional[float] = Field(
        default=None,
        description="Latest-day share volume minus comparison-day share volume.",
    )
    delta_trade_count: Optional[int] = Field(
        default=None,
        description="Latest-day trade count minus comparison-day trade count.",
    )
    latest_noise_cv_28d: Optional[float] = Field(
        default=None,
        description="Latest-day rolling 28-day coefficient of variation.",
    )
    latest_noise_mad_ratio_28d: Optional[float] = Field(
        default=None,
        description="Latest-day rolling 28-day MAD/median ratio.",
    )
    latest_unusually_active: bool = False
    latest_unusually_inactive: bool = False
    latest_flagged_outlier: bool = False


class PolymarketDailyAnalyticsCoverage(BaseModel):
    artifact_path: Optional[str] = Field(default=None, description="Run-local trades artifact path.")
    total_trade_rows: int = Field(..., description="Rows loaded from the run-local trades artifact.")
    valid_trade_rows: int = Field(..., description="Rows remaining after schema and timestamp validation.")
    filtered_trade_rows: int = Field(..., description="Rows remaining after ticker/date filters.")
    observed_trade_days: int = Field(..., description="Observed trade days after filters and before zero-fill.")
    filled_days: int = Field(..., description="Calendar days returned after zero-fill.")
    effective_date_min: Optional[str] = Field(default=None, description="Effective minimum date returned.")
    effective_date_max: Optional[str] = Field(default=None, description="Effective maximum date returned.")
    requested_date_min: Optional[str] = Field(default=None, description="Requested minimum date filter.")
    requested_date_max: Optional[str] = Field(default=None, description="Requested maximum date filter.")
    requested_tickers: List[str] = Field(
        default_factory=list,
        description="Ticker filter applied to the artifact.",
    )
    requested_token_role: str = Field(default="all", description="Outcome token role filter applied.")


class PolymarketDailyAnalyticsStructureBucket(BaseModel):
    key: str
    label: str
    observations: int
    mean_daily_notional_volume: float
    median_daily_notional_volume: float
    total_notional_volume: float
    share_total_notional_volume: float


class PolymarketDailyAnalyticsStructure(BaseModel):
    weekday: List[PolymarketDailyAnalyticsStructureBucket] = Field(default_factory=list)
    event_proximity: List[PolymarketDailyAnalyticsStructureBucket] = Field(default_factory=list)
    ladder_bucket: List[PolymarketDailyAnalyticsStructureBucket] = Field(default_factory=list)


class PolymarketDailyAnalyticsResponse(BaseModel):
    run_id: str
    analytics_ready: bool
    primary_metric: str
    metric_mode: Literal["true_volume"]
    token_role: Literal["all", "yes", "no"] = "all"
    ci_level: int = 90
    exclude_flagged: bool = False
    baseline_window_days: int = 28
    latest_trade_date: Optional[str] = None
    comparison_date: Optional[str] = None
    available_tickers: List[str] = Field(default_factory=list)
    coverage: PolymarketDailyAnalyticsCoverage
    summary: PolymarketDailyAnalyticsSummary
    days: List[PolymarketDailyAnalyticsDay] = Field(default_factory=list)
    structure: PolymarketDailyAnalyticsStructure = Field(default_factory=PolymarketDailyAnalyticsStructure)
    warnings: List[str] = Field(default_factory=list)


class PolymarketDailyAnalyticsBreakdownRow(BaseModel):
    key: str
    label: str
    ticker: Optional[str] = None
    market_id: Optional[str] = None
    threshold: Optional[float] = None
    week_friday: Optional[str] = None
    event_endDate: Optional[str] = None
    ladder_bucket: Optional[str] = None
    event_proximity_bucket: Optional[str] = None
    daily_notional_volume: float
    share_volume: float
    trade_count: int
    expected_notional_volume: Optional[float] = None
    band_lo: Optional[float] = None
    band_hi: Optional[float] = None
    realized_expected_ratio: Optional[float] = None
    unusually_active: bool = False
    unusually_inactive: bool = False
    flagged_outlier: bool = False
    baseline_source: str = "rolling_weekday_median"
    history_days: int = 0
    volume_share_of_day: Optional[float] = None


class PolymarketDailyAnalyticsBreakdownResponse(BaseModel):
    run_id: str
    date: str
    group_by: Literal["ticker", "market"]
    token_role: Literal["all", "yes", "no"] = "all"
    ci_level: int = 90
    exclude_flagged: bool = False
    available_tickers: List[str] = Field(default_factory=list)
    selected_tickers: List[str] = Field(default_factory=list)
    rows: List[PolymarketDailyAnalyticsBreakdownRow] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
