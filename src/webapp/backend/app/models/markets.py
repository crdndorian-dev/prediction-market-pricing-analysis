from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MarketsRefreshRequest(BaseModel):
    week_friday: Optional[str] = Field(
        default=None,
        description="Target Friday (YYYY-MM-DD). Defaults to current week (prev Friday on weekends).",
    )
    tickers: Optional[List[str]] = Field(
        default=None,
        description="Tickers to refresh (e.g. ['AAPL', 'NVDA']).",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Weekly history run_id to update. Defaults to latest.json pointer.",
    )
    force_refresh: bool = Field(
        default=False,
        description="Ignore refresh index and refetch from week start.",
    )


class MarketsRefreshResult(BaseModel):
    ok: bool
    run_id: Optional[str]
    week_friday: Optional[str]
    run_dir: Optional[str]
    stdout: str
    stderr: str
    duration_s: float
    command: List[str]


class MarketsProgress(BaseModel):
    stage: Optional[str]
    current: int = 0
    total: int = 0


class MarketsJobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "finished", "failed"]
    progress: Optional[MarketsProgress] = None
    result: Optional[MarketsRefreshResult] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class MarketsSummaryItem(BaseModel):
    ticker: str
    threshold: float
    market_id: Optional[str]
    event_id: Optional[str]
    event_endDate: Optional[str]
    points: int
    last_timestamp_utc: Optional[str]
    has_polymarket: bool
    has_prn: bool


class MarketsSummaryResponse(BaseModel):
    run_id: Optional[str]
    week_friday: str
    week_monday: Optional[str]
    week_sunday: Optional[str]
    last_refresh_utc: Optional[str]
    trading_universe_tickers: List[str] = Field(
        default_factory=list,
        description="Tickers from the active weekly_markets.csv trading universe.",
    )
    markets: List[MarketsSummaryItem]


class MarketsSeriesPoint(BaseModel):
    timestamp_utc: str
    polymarket_buy: Optional[float] = None   # kept for backward compat
    polymarket_bid: Optional[float] = None   # best bid (sell YES)
    polymarket_ask: Optional[float] = None   # best ask (buy YES); falls back to polymarket_buy
    pRN: Optional[float] = None
    spot: Optional[float] = None


class MarketsSeriesResponse(BaseModel):
    run_id: Optional[str]
    ticker: str
    threshold: float
    week_friday: str
    market_id: Optional[str]
    event_id: Optional[str]
    points: List[MarketsSeriesPoint]
    metadata: Optional[dict] = None


class MarketsSeriesByTickerResponse(BaseModel):
    run_id: Optional[str]
    ticker: str
    week_friday: str
    strikes: List[MarketsSeriesResponse]
    metadata: Optional[dict] = None
