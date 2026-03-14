from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PolymarketQualityBucketCounts(BaseModel):
    clean: int = 0
    watch: int = 0
    noisy: int = 0


class PolymarketQualityFlagSummary(BaseModel):
    name: str
    count: int
    share: Optional[float] = None


class PolymarketQualityTickerSummary(BaseModel):
    ticker: str
    market_count: int
    flagged_market_count: int
    flagged_share: Optional[float] = None
    avg_issue_count: Optional[float] = None
    clean_share: Optional[float] = None
    watch_share: Optional[float] = None
    noisy_share: Optional[float] = None


class PolymarketQualityMarketSample(BaseModel):
    market_id: str
    event_id: Optional[str] = None
    ticker: str
    threshold: Optional[float] = None
    week_friday: str
    quality_issue_count: Optional[float] = None
    quality_bucket: Optional[str] = None
    active_flags: List[str] = Field(default_factory=list)
    snapshot_coverage_status: Optional[str] = None
    snapshot_drop_reason: Optional[str] = None
    hours_since_last_yes_trade: Optional[float] = None
    stale_ratio: Optional[float] = None
    max_stale_hours: Optional[float] = None
    midprice_cluster_ratio: Optional[float] = None
    max_jump: Optional[float] = None
    gamma_volume: Optional[float] = None


class PolymarketQualityWeekSummary(BaseModel):
    week_friday: str
    market_count: int
    flagged_market_count: int
    flagged_share: Optional[float] = None
    avg_issue_count: Optional[float] = None
    quality_bucket: Optional[str] = None


class PolymarketQualitySummary(BaseModel):
    market_count: int = 0
    flagged_market_count: int = 0
    flagged_share: float = 0.0
    bucket_counts: PolymarketQualityBucketCounts = Field(default_factory=PolymarketQualityBucketCounts)
    prn_coverage_counts: Dict[str, int] = Field(default_factory=dict)
    top_flags: List[PolymarketQualityFlagSummary] = Field(default_factory=list)
    top_problem_tickers: List[PolymarketQualityTickerSummary] = Field(default_factory=list)
    snapshot_anchor: Optional[str] = None
    quality_columns: List[str] = Field(default_factory=list)
    quality_flag_columns: List[str] = Field(default_factory=list)


class PolymarketMarketQuality(BaseModel):
    market_id: Optional[str] = None
    ticker: Optional[str] = None
    threshold: Optional[float] = None
    week_friday: Optional[str] = None
    quality_issue_count: Optional[float] = None
    quality_bucket: Optional[str] = None
    active_flags: List[str] = Field(default_factory=list)
    snapshot_date_used: Optional[str] = None
    snapshot_time_used: Optional[str] = None
    snapshot_coverage_status: Optional[str] = None
    snapshot_drop_reason: Optional[str] = None
    snapshot_pRN: Optional[float] = None
    snapshot_abs_log_m_fwd: Optional[float] = None
    yes_points: Optional[int] = None
    stale_ratio: Optional[float] = None
    max_stale_hours: Optional[float] = None
    midprice_cluster_ratio: Optional[float] = None
    max_jump: Optional[float] = None
    hours_since_last_yes_trade: Optional[float] = None
    gamma_volume: Optional[float] = None
    flag_not_relevant: Optional[bool] = None
    flag_prn_missing: Optional[bool] = None
    flag_pm_no_trade_history: Optional[bool] = None
    flag_pm_no_recent_trade: Optional[bool] = None
    flag_pm_stale_prices: Optional[bool] = None
    flag_extreme_otm: Optional[bool] = None


class PolymarketQualityAuditResponse(BaseModel):
    run_id: str
    available: bool = True
    message: Optional[str] = None
    summary: PolymarketQualitySummary = Field(default_factory=PolymarketQualitySummary)
    flag_distribution: List[PolymarketQualityFlagSummary] = Field(default_factory=list)
    problem_tickers: List[PolymarketQualityTickerSummary] = Field(default_factory=list)
    problem_markets: List[PolymarketQualityMarketSample] = Field(default_factory=list)
    weekly_summary: List[PolymarketQualityWeekSummary] = Field(default_factory=list)
    available_quality_flags: List[str] = Field(default_factory=list)


class PolymarketQualityTelemetry(BaseModel):
    phase: Literal["quality", "complete"] = "quality"
    total_markets: int = 0
    completed_markets: int = 0
    flagged_markets: int = 0
    flagged_share: float = 0.0
    bucket_counts: PolymarketQualityBucketCounts = Field(default_factory=PolymarketQualityBucketCounts)
    top_flags: List[PolymarketQualityFlagSummary] = Field(default_factory=list)
    prn_coverage_counts: Dict[str, int] = Field(default_factory=dict)
    top_problem_tickers: List[PolymarketQualityTickerSummary] = Field(default_factory=list)
