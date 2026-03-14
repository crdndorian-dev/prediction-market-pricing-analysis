from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.models.polymarket_quality import PolymarketQualitySummary, PolymarketQualityTelemetry


class SharedArtifactSummary(BaseModel):
    name: str
    path: str
    frequency: Optional[str] = None
    size_bytes: int
    row_count: Optional[int] = None
    last_modified: Optional[str] = None


class RunArtifactSummary(BaseModel):
    name: str
    path: str
    size_bytes: int
    row_count: Optional[int] = None
    last_modified: Optional[str] = None


class RunArtifactGroupSummary(BaseModel):
    key: str
    label: str
    path: str
    files: List[RunArtifactSummary] = Field(default_factory=list)


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
        description="Deprecated compatibility field. Polymarket jobs now build and use a run-local exact Theta pRN file.",
    )
    skip_subgraph_labels: bool = Field(
        default=False,
        description="Skip fetching labels from the subgraph during feature build.",
    )
    resume_existing: bool = Field(
        default=False,
        description="Internal flag to allow resuming an interrupted run in an existing run directory.",
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
    master_bar_artifacts: List[SharedArtifactSummary] = Field(default_factory=list)
    artifact_groups: List[RunArtifactGroupSummary] = Field(default_factory=list)
    shared_artifacts: List[SharedArtifactSummary] = Field(default_factory=list)
    quality_summary: Optional[PolymarketQualitySummary] = None


class PolymarketRunFeaturesRequest(BaseModel):
    prn_dataset: Optional[str] = Field(
        default=None,
        description="Deprecated compatibility field. Decision features now default to the run-local exact Theta pRN file.",
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
    phase: Optional[Literal["history", "prn", "quality", "features", "finalizing"]] = None
    progress: Optional[PolymarketHistoryProgress] = None
    features_progress: Optional[PolymarketHistoryProgress] = None
    telemetry: Optional[PolymarketQualityTelemetry] = None
    result: Optional[PolymarketHistoryRunResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
