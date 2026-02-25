from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.models.markets import (
    MarketsJobStatus,
    MarketsRefreshRequest,
    MarketsSeriesByTickerResponse,
    MarketsSeriesResponse,
    MarketsSummaryResponse,
)
from app.services.job_guard import ensure_no_active_jobs
from app.services.markets import (
    get_markets_job,
    get_markets_series,
    get_markets_series_by_ticker,
    get_markets_summary,
    start_markets_job,
)

router = APIRouter(prefix="/markets", tags=["markets"])


@router.post("/refresh", response_model=MarketsJobStatus)
def refresh_markets(payload: MarketsRefreshRequest) -> MarketsJobStatus:
    try:
        ensure_no_active_jobs()
        job_id = start_markets_job(payload)
        return get_markets_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Markets refresh failed: {str(exc)}") from exc


@router.get("/jobs/{job_id}", response_model=MarketsJobStatus)
def get_job(job_id: str) -> MarketsJobStatus:
    try:
        return get_markets_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.get("/summary", response_model=MarketsSummaryResponse)
def summary(
    week_friday: Optional[str] = Query(None, description="Target Friday (YYYY-MM-DD)."),
    run_id: Optional[str] = Query(None, description="Run id for weekly history."),
) -> MarketsSummaryResponse:
    try:
        return get_markets_summary(week_friday=week_friday, run_id=run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Summary failed: {str(exc)}") from exc


@router.get("/series", response_model=MarketsSeriesResponse)
def series(
    ticker: str = Query(..., description="Ticker symbol"),
    threshold: float = Query(..., description="Strike threshold"),
    week_friday: Optional[str] = Query(None, description="Target Friday (YYYY-MM-DD)."),
    run_id: Optional[str] = Query(None, description="Run id for weekly history."),
) -> MarketsSeriesResponse:
    try:
        return get_markets_series(
            ticker=ticker,
            threshold=threshold,
            week_friday=week_friday,
            run_id=run_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Series failed: {str(exc)}") from exc


@router.get("/series/by-ticker", response_model=MarketsSeriesByTickerResponse)
def series_by_ticker(
    ticker: str = Query(..., description="Ticker symbol"),
    week_friday: Optional[str] = Query(None, description="Target Friday (YYYY-MM-DD)."),
    run_id: Optional[str] = Query(None, description="Run id for weekly history."),
) -> MarketsSeriesByTickerResponse:
    try:
        return get_markets_series_by_ticker(
            ticker=ticker,
            week_friday=week_friday,
            run_id=run_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Series failed: {str(exc)}") from exc
