from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class BarsRequest(BaseModel):
    """Request parameters for fetching bar history data."""

    run_id: Optional[str] = Field(None, description="Pipeline run ID (e.g., weekly-history-20260210T162452Z)")
    market_id: Optional[str] = Field(None, description="Market ID to filter by")
    ticker: Optional[str] = Field(None, description="Ticker symbol to filter by")
    time_min: Optional[str] = Field(None, description="Minimum timestamp (ISO 8601)")
    time_max: Optional[str] = Field(None, description="Maximum timestamp (ISO 8601)")
    max_points: int = Field(1000, ge=10, le=10000, description="Maximum points to return (downsampling)")
    view_mode: Literal["decision_time", "full_history"] = Field(
        "decision_time",
        description="decision_time: only data <= time_max; full_history: all available data"
    )


class BarDataPoint(BaseModel):
    """Single bar data point."""

    timestamp: str = Field(..., description="ISO 8601 timestamp")
    timestamp_ms: int = Field(..., description="Unix timestamp in milliseconds")
    price: float = Field(..., description="Price (close or mid)")
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None


class BarsResponse(BaseModel):
    """Response containing bar history data."""

    run_id: Optional[str] = None
    market_id: Optional[str] = None
    ticker: Optional[str] = None
    view_mode: str
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    total_points: int = Field(..., description="Total points before downsampling")
    returned_points: int = Field(..., description="Points returned after downsampling")
    downsampled: bool = Field(..., description="Whether downsampling was applied")
    bars: List[BarDataPoint] = Field(..., description="Bar data points (sorted by timestamp)")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class BarsRunListResponse(BaseModel):
    """List of available pipeline runs."""

    runs: List[dict] = Field(..., description="Available pipeline runs with metadata")


# ---------------------------------------------------------------------------
# By-strike models (for BacktestsPage)
# ---------------------------------------------------------------------------

class ByStrikeRequest(BaseModel):
    """Request for bars grouped by strike price."""

    run_id: Optional[str] = Field(None, description="Pipeline run ID")
    ticker: str = Field(..., description="Ticker symbol (required)")
    time_min: Optional[str] = Field(None, description="Minimum timestamp (ISO 8601)")
    time_max: Optional[str] = Field(None, description="Maximum timestamp (ISO 8601)")
    token_role: str = Field("yes", description="Token role filter (yes/no)")
    max_points_per_strike: int = Field(500, ge=10, le=5000, description="Max points per strike")
    view_mode: Literal["decision_time", "full_history"] = Field("full_history")


class StrikeSeries(BaseModel):
    """One time-series for a single strike."""

    strike: float
    strike_label: str
    market_id: Optional[str] = None
    event_slug: Optional[str] = None
    total_points: int
    returned_points: int
    bars: List[BarDataPoint]


class ByStrikeResponse(BaseModel):
    """Bars grouped by strike price for a single ticker."""

    run_id: Optional[str] = None
    ticker: str
    token_role: str
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    view_mode: str
    strikes: List[StrikeSeries]
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# pRN overlay models (for BacktestsPage chart overlay)
# ---------------------------------------------------------------------------

class PrnPoint(BaseModel):
    """Single pRN observation for a strike on a given date."""

    asof_date: str = Field(..., description="Observation date (YYYY-MM-DD)")
    asof_date_ms: int = Field(..., description="Midday-UTC timestamp in ms (for chart x-axis)")
    dte: int = Field(..., description="Days to expiration (1-4)")
    pRN: float = Field(..., description="Risk-neutral probability [0, 1]")


class PrnStrikeSeries(BaseModel):
    """pRN time series for a single strike."""

    strike: float
    strike_label: str
    points: List[PrnPoint]


class PrnOverlayResponse(BaseModel):
    """pRN overlay data for all strikes of a ticker in a date range."""

    ticker: str
    dataset_path: Optional[str] = None
    strikes: List[PrnStrikeSeries]
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trading weeks (for BacktestsPage calendar)
# ---------------------------------------------------------------------------

class TradingWeek(BaseModel):
    """Single trading week (Monday -> Friday)."""

    start_date: str = Field(..., description="Week start date (YYYY-MM-DD, Monday)")
    end_date: str = Field(..., description="Week end date (YYYY-MM-DD, Friday)")


class TradingWeeksResponse(BaseModel):
    """Available trading weeks for a ticker/run."""

    run_id: Optional[str] = None
    ticker: str
    weeks: List[TradingWeek]
    metadata: dict = Field(default_factory=dict)
