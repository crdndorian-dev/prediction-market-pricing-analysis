from fastapi import APIRouter

from app.services.dashboard import build_dashboard_payload

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("")
def get_dashboard():
    return build_dashboard_payload()
