from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.calibrate_models import router as calibrate_models_router
from app.api.dashboard import router as dashboard_router
from app.api.datasets import router as datasets_router
from app.api.health import router as health_router
from app.api.polymarket_history import router as polymarket_history_router
from app.api.market_map import router as market_map_router
from app.api.bars import router as bars_router
from app.api.markets import router as markets_router

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
app.include_router(polymarket_history_router)
app.include_router(market_map_router)
app.include_router(bars_router)
app.include_router(markets_router)
