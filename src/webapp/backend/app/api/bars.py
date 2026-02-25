"""API endpoints for bar history data."""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.models.bars import (
    BarsRequest,
    BarsResponse,
    BarsRunListResponse,
    ByStrikeRequest,
    ByStrikeResponse,
    PrnOverlayResponse,
    TradingWeeksResponse,
)
from app.services.bars import get_bars, get_bars_by_strike, list_bar_runs, list_trading_weeks
from app.services.prn_overlay import get_prn_overlay
from app.services.prn_on_demand import get_prn_on_demand


router = APIRouter(prefix="/bars", tags=["bars"])


@router.get("", response_model=BarsResponse)
def get_bar_history(
    run_id: Optional[str] = Query(None, description="Pipeline run ID"),
    market_id: Optional[str] = Query(None, description="Market ID to filter"),
    ticker: Optional[str] = Query(None, description="Ticker symbol to filter"),
    time_min: Optional[str] = Query(None, description="Min timestamp (ISO 8601)"),
    time_max: Optional[str] = Query(None, description="Max timestamp (ISO 8601)"),
    max_points: int = Query(1000, ge=10, le=10000, description="Max points to return"),
    view_mode: str = Query("decision_time", description="decision_time or full_history"),
) -> BarsResponse:
    """
    Get bar history data with time-safety and performance optimizations.

    Time-safety:
    - decision_time mode: Returns only bars with timestamp <= time_max (no future data)
    - full_history mode: Returns all available bars (use for post-trade analysis)

    Performance:
    - Results are cached for 5 minutes
    - Automatic downsampling to max_points using time-bucketing
    - Efficient CSV scanning with early filtering

    Example request:
    GET /bars?ticker=AAPL&time_min=2026-01-01T00:00:00Z&time_max=2026-01-10T00:00:00Z&max_points=500&view_mode=decision_time
    """
    try:
        request = BarsRequest(
            run_id=run_id,
            market_id=market_id,
            ticker=ticker,
            time_min=time_min,
            time_max=time_max,
            max_points=max_points,
            view_mode=view_mode,  # type: ignore
        )
        return get_bars(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}") from exc


@router.get("/by-strike", response_model=ByStrikeResponse)
def get_bars_by_strike_endpoint(
    ticker: str = Query(..., description="Ticker symbol (required)"),
    run_id: Optional[str] = Query(None, description="Pipeline run ID"),
    time_min: Optional[str] = Query(None, description="Min timestamp (ISO 8601)"),
    time_max: Optional[str] = Query(None, description="Max timestamp (ISO 8601)"),
    token_role: str = Query("yes", description="Token role (yes/no)"),
    max_points_per_strike: int = Query(500, ge=10, le=5000, description="Max points per strike"),
    view_mode: str = Query("full_history", description="decision_time or full_history"),
) -> ByStrikeResponse:
    """Get bars grouped by strike price for a single ticker and date range."""
    try:
        request = ByStrikeRequest(
            run_id=run_id,
            ticker=ticker,
            time_min=time_min,
            time_max=time_max,
            token_role=token_role,
            max_points_per_strike=max_points_per_strike,
            view_mode=view_mode,  # type: ignore
        )
        return get_bars_by_strike(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}") from exc


@router.get("/prn-overlay", response_model=PrnOverlayResponse)
def get_prn_overlay_endpoint(
    ticker: str = Query(..., description="Ticker symbol (required)"),
    time_min: Optional[str] = Query(None, description="Min timestamp (ISO 8601)"),
    time_max: Optional[str] = Query(None, description="Max timestamp (ISO 8601)"),
) -> PrnOverlayResponse:
    """Get pRN overlay data for charting alongside Polymarket price series."""
    try:
        return get_prn_overlay(ticker=ticker, time_min=time_min, time_max=time_max)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}") from exc


@router.get("/prn-overlay/theta", response_model=PrnOverlayResponse)
def get_prn_overlay_theta_endpoint(
    ticker: str = Query(..., description="Ticker symbol (required)"),
    time_min: Optional[str] = Query(None, description="Min timestamp (ISO 8601)"),
    time_max: Optional[str] = Query(None, description="Max timestamp (ISO 8601)"),
    dte_list: Optional[str] = Query(None, description="Comma-separated DTEs (e.g. '4,3,2,1')"),
    strikes: Optional[str] = Query(None, description="Comma-separated target strikes (Polymarket strikes)"),
) -> PrnOverlayResponse:
    """Compute pRN on-demand via Theta Terminal for backtest overlay.

    Uses relaxed quality thresholds to recover 1DTE and other thin-chain
    observations that the training pipeline drops.  Results are disk-cached.
    Falls back gracefully if Theta Terminal is unavailable.

    When ``strikes`` is provided, pRN is interpolated to those exact values
    (matching Polymarket strikes) instead of using moneyness band selection.
    """
    try:
        parsed_dtes = None
        if dte_list:
            parsed_dtes = [int(d.strip()) for d in dte_list.split(",") if d.strip()]
        parsed_strikes = None
        if strikes:
            parsed_strikes = [float(s.strip()) for s in strikes.split(",") if s.strip()]
        return get_prn_on_demand(
            ticker=ticker,
            time_min=time_min,
            time_max=time_max,
            dte_list=parsed_dtes,
            target_strikes=parsed_strikes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Theta pRN error: {str(exc)}") from exc


@router.get("/runs", response_model=BarsRunListResponse)
def list_runs() -> BarsRunListResponse:
    """List available pipeline runs with bar history data."""
    try:
        result = list_bar_runs()
        return BarsRunListResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list runs: {str(exc)}") from exc


@router.get("/trading-weeks", response_model=TradingWeeksResponse)
def list_trading_weeks_endpoint(
    ticker: str = Query(..., description="Ticker symbol (required)"),
    run_id: Optional[str] = Query(None, description="Pipeline run ID"),
) -> TradingWeeksResponse:
    """List trading weeks (Mon-Fri) that have bar data for a ticker."""
    try:
        result = list_trading_weeks(ticker, run_id)
        return TradingWeeksResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list trading weeks: {str(exc)}") from exc
