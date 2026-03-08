from __future__ import annotations

import csv
import io
import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from app.models.analysis import (
    AnalysisDriftResponse,
    AnalysisJobStatus,
    AnalysisOverviewResponse,
    AnalysisPriceBehaviorResponse,
    AnalysisRefreshRequest,
    AnalysisStructureResponse,
    AnalysisTableResponse,
    AnalysisVolumeResponse,
    ResearchNoteCreateRequest,
    ResearchNoteResponse,
    ResearchNotesResponse,
)
from app.services.analysis import (
    build_analysis_drift,
    build_analysis_overview,
    build_analysis_price_behavior,
    build_analysis_structure,
    build_analysis_volume,
    create_research_note,
    export_analysis_table_rows,
    get_analysis_job,
    list_research_notes,
    query_analysis_table,
    start_analysis_job,
)


router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/refresh", response_model=AnalysisJobStatus)
def refresh_analysis(payload: AnalysisRefreshRequest) -> AnalysisJobStatus:
    try:
        job_id = start_analysis_job(payload)
        return get_analysis_job(job_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis refresh failed: {str(exc)}") from exc


@router.get("/jobs/{job_id}", response_model=AnalysisJobStatus)
def get_job(job_id: str) -> AnalysisJobStatus:
    try:
        return get_analysis_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")


@router.get("/overview", response_model=AnalysisOverviewResponse)
def overview() -> AnalysisOverviewResponse:
    try:
        return build_analysis_overview()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Overview query failed: {str(exc)}") from exc


@router.get("/volume", response_model=AnalysisVolumeResponse)
def volume(
    ticker: Optional[str] = Query(None),
    variable_name: str = Query("notional_volume"),
    limit: int = Query(120, ge=10, le=1000),
) -> AnalysisVolumeResponse:
    try:
        return build_analysis_volume(ticker=ticker, variable_name=variable_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Volume query failed: {str(exc)}") from exc


@router.get("/structure", response_model=AnalysisStructureResponse)
def structure(limit: int = Query(120, ge=10, le=1000)) -> AnalysisStructureResponse:
    try:
        return build_analysis_structure(limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Structure query failed: {str(exc)}") from exc


@router.get("/price-behavior", response_model=AnalysisPriceBehaviorResponse)
def price_behavior(
    ticker: Optional[str] = Query(None),
    limit: int = Query(120, ge=10, le=1000),
) -> AnalysisPriceBehaviorResponse:
    try:
        return build_analysis_price_behavior(ticker=ticker, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Price behavior query failed: {str(exc)}") from exc


@router.get("/drift", response_model=AnalysisDriftResponse)
def drift(
    ticker: Optional[str] = Query(None),
    variable_name: str = Query("notional_volume"),
    limit: int = Query(120, ge=10, le=1000),
) -> AnalysisDriftResponse:
    try:
        return build_analysis_drift(ticker=ticker, variable_name=variable_name, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift query failed: {str(exc)}") from exc


@router.get("/tables", response_model=AnalysisTableResponse)
def tables(
    table: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    ticker: Optional[str] = Query(None),
    variable_name: Optional[str] = Query(None),
) -> AnalysisTableResponse:
    try:
        return query_analysis_table(
            table=table,
            page=page,
            page_size=page_size,
            ticker=ticker,
            variable_name=variable_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Table query failed: {str(exc)}") from exc


@router.get("/tables/export", response_class=PlainTextResponse)
def export_table(
    table: str = Query(...),
    ticker: Optional[str] = Query(None),
    variable_name: Optional[str] = Query(None),
) -> PlainTextResponse:
    try:
        rows = export_analysis_table_rows(table=table, ticker=ticker, variable_name=variable_name)
        output = io.StringIO()
        writer = csv.writer(output)
        if rows:
            headers = list(rows[0].keys())
            writer.writerow(headers)
            for row in rows:
                writer.writerow([json.dumps(value) if isinstance(value, (dict, list)) else value for value in row.values()])
        return PlainTextResponse(content=output.getvalue(), media_type="text/csv")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CSV export failed: {str(exc)}") from exc


@router.get("/notes", response_model=ResearchNotesResponse)
def notes() -> ResearchNotesResponse:
    try:
        return list_research_notes()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Notes query failed: {str(exc)}") from exc


@router.post("/notes", response_model=ResearchNoteResponse)
def create_note(payload: ResearchNoteCreateRequest) -> ResearchNoteResponse:
    try:
        return create_research_note(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Note creation failed: {str(exc)}") from exc
