from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MarketMapRunRequest(BaseModel):
    run_dir: Optional[str] = Field(
        default=None,
        description="Use an existing subgraph run directory instead of fetching.",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional run_id when fetching markets from the subgraph.",
    )
    overrides: Optional[str] = Field(
        default=None,
        description="CSV overrides file for dim_market adjustments.",
    )
    tickers: Optional[str] = Field(
        default=None,
        description="Comma-separated ticker allowlist.",
    )
    prn_dataset: Optional[str] = Field(
        default=None,
        description="Path to pRN dataset CSV (to infer tickers).",
    )
    out: Optional[str] = Field(
        default=None,
        description="Output path for dim_market (parquet or csv).",
    )
    strict: bool = Field(
        default=True,
        description="Fail if any target market lacks ticker or threshold.",
    )


class MarketMapRunResponse(BaseModel):
    output_path: Optional[str] = None
    row_count: Optional[int] = None
    source_run: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    duration_s: Optional[float] = None


class MarketMapFileSummary(BaseModel):
    name: str
    path: str
    last_modified: Optional[str] = None
    size_bytes: Optional[int] = None


class MarketMapPreviewResponse(BaseModel):
    file: Optional[MarketMapFileSummary] = None
    headers: Optional[List[str]] = None
    rows: Optional[List[Dict[str, Optional[str]]]] = None
    row_count: Optional[int] = None
    limit: Optional[int] = None


class MarketMapDeleteResponse(BaseModel):
    deleted: bool
    paths: List[str] = Field(default_factory=list)


class MarketMapJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed"]
    result: Optional[MarketMapRunResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
