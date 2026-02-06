from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class PHATEdgeRunRequest(BaseModel):
    model_path: str = Field(
        ...,
        description="Project-relative path to the saved calibration artifact (model.joblib).",
    )
    snapshot_csv: str = Field(
        ...,
        description="Project-relative path to the Polymarket snapshot CSV to score.",
    )
    out_csv: Optional[str] = Field(
        default=None,
        description="Optional output path for the enriched CSV. Defaults to a timestamped file under src/data/analysis/phat-edge.",
    )
    exclude_tickers: Optional[str] = Field(
        default=None,
        description="Comma-separated tickers to omit from the edge summary.",
    )
    require_columns_strict: Optional[bool] = Field(
        default=True,
        description="Fail when required columns are missing (true) or allow NaNs (false).",
    )
    compute_edge: Optional[bool] = Field(
        default=True,
        description="Subtract buy price from pHAT when pricing data is available.",
    )
    skip_edge_outside_prn_range: Optional[bool] = Field(
        default=True,
        description="Skip edge when pRN is outside the model's training range (if available).",
    )


class PHATEdgeDistributionStats(BaseModel):
    count: int
    mean: float
    min: float
    max: float


class PHATEdgeRow(BaseModel):
    ticker: str
    K: Optional[float]
    spot: Optional[float] = None
    pHAT: Optional[float]
    qHAT: Optional[float] = None
    edge: Optional[float]
    pPM_buy: Optional[float]
    qPM_buy: Optional[float] = None
    edge_source: Optional[str] = None
    pRN: Optional[float] = None
    qRN: Optional[float] = None


class PHATEdgeRunSummary(BaseModel):
    model_path: str
    snapshot_csv: str
    output_csv: str
    duration_s: float
    ok: bool


class PHATEdgeRunResponse(BaseModel):
    ok: bool
    command: List[str]
    stdout: str
    stderr: str
    run_summary: PHATEdgeRunSummary
    pHat_distribution: Optional[PHATEdgeDistributionStats]
    edge_distribution: Optional[PHATEdgeDistributionStats]
    top_edges: List[PHATEdgeRow]


class PHATEdgeFileSummary(BaseModel):
    name: str
    path: str
    size_bytes: int
    last_modified: str


class PHATEdgePreviewResponse(BaseModel):
    file: PHATEdgeFileSummary
    headers: List[str]
    rows: List[dict]
    row_count: Optional[int] = None
    mode: Literal["head", "tail"]
    limit: int


class PHATEdgeRunListResponse(BaseModel):
    runs: List[PHATEdgeFileSummary]


class PHATEdgeSummaryResponse(BaseModel):
    file: PHATEdgeFileSummary
    pHat_distribution: Optional[PHATEdgeDistributionStats]
    edge_distribution: Optional[PHATEdgeDistributionStats]
    top_edges: List[PHATEdgeRow]


class PHATEdgeRowsResponse(BaseModel):
    file: PHATEdgeFileSummary
    rows: List[PHATEdgeRow]
    row_count: int


class PHATEdgeDeleteResponse(BaseModel):
    path: str
    deleted: bool


class PHATEdgeJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed"]
    result: Optional[PHATEdgeRunResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
