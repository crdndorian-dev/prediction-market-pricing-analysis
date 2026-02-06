from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class PolymarketSnapshotRunSummary(BaseModel):
    run_id: str
    run_time_utc: Optional[str]
    run_dir: str
    files: List[str]
    file_count: int
    size_bytes: int
    last_modified: Optional[str]


class PolymarketSnapshotListResponse(BaseModel):
    out_dir: str
    runs: List[PolymarketSnapshotRunSummary]


class PolymarketSnapshotDeleteResponse(BaseModel):
    run_id: str
    deleted: bool


class PolymarketSnapshotFileSummary(BaseModel):
    name: str
    path: str
    size_bytes: int
    last_modified: str
    kind: Optional[str] = None


class PolymarketSnapshotLatestResponse(BaseModel):
    date: Optional[str]
    files: List[PolymarketSnapshotFileSummary]
    history_file: Optional[PolymarketSnapshotFileSummary]


class PolymarketSnapshotHistoryResponse(BaseModel):
    files: List[PolymarketSnapshotFileSummary]


class PolymarketSnapshotPreviewResponse(BaseModel):
    file: PolymarketSnapshotFileSummary
    headers: List[str]
    rows: List[Dict[str, Optional[str]]]
    row_count: Optional[int]
    mode: str
    limit: int


class PolymarketSnapshotRunRequest(BaseModel):
    tickers: Optional[List[str]] = Field(
        default=None,
        description="Tickers to fetch (e.g. ['AAPL', 'NVDA']).",
    )
    tickers_csv: Optional[str] = Field(
        default=None,
        description="Relative path to CSV with a 'ticker' column.",
    )
    slug_overrides: Optional[str] = Field(
        default=None,
        description="Relative path to CSV/JSON with ticker -> slug overrides.",
    )
    risk_free_rate: Optional[float] = Field(
        default=None,
        description="Risk-free rate used for pRN calculation.",
    )
    tz: Optional[str] = Field(
        default=None,
        description="Timezone used to resolve the weekly market window.",
    )
    contract_type: Optional[Literal["weekly", "1dte"]] = Field(
        default=None,
        description="Contract type to fetch (weekly or 1dte).",
    )
    contract_1dte: Optional[Literal["close_today", "close_tomorrow"]] = Field(
        default=None,
        description="For 1dte: contract expiring at today's or tomorrow's market close.",
    )
    target_date: Optional[str] = Field(
        default=None,
        description="Override target date for contract resolution (YYYY-MM-DD).",
    )
    exchange_calendar: Optional[str] = Field(
        default=None,
        description="Exchange calendar name (e.g. XNYS).",
    )
    allow_nonlive: bool = Field(
        default=False,
        description="Allow non-live snapshots to compute pRN from yfinance option chains.",
    )
    dry_run: bool = Field(
        default=False,
        description="Run fetch + compute + validation without writing any files.",
    )
    keep_nonexec: bool = Field(
        default=False,
        description="Keep rows with pm_ok=False (same as script flag).",
    )


class PolymarketSnapshotRunResponse(BaseModel):
    ok: bool
    run_id: Optional[str]
    out_dir: str
    run_dir: Optional[str]
    files: List[str]
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]


class PolymarketSnapshotJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed"]
    result: Optional[PolymarketSnapshotRunResponse] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
