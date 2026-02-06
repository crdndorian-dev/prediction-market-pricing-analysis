from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.calibrate_models import router as calibrate_models_router
from app.api.dashboard import router as dashboard_router
from app.api.datasets import router as datasets_router
from app.api.health import router as health_router
from app.api.polymarket_snapshots import router as polymarket_snapshots_router
from app.api.phat_edge import router as phat_edge_router

app = FastAPI(title="Polyedgetool Web App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(dashboard_router)
app.include_router(datasets_router)
app.include_router(calibrate_models_router)
app.include_router(polymarket_snapshots_router)
app.include_router(phat_edge_router)
