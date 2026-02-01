from __future__ import annotations

from typing import Dict, List, Optional

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
