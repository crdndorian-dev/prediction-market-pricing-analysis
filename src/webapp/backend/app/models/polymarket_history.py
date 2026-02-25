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
