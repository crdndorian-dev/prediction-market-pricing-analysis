from __future__ import annotations

import os
from typing import Dict, List, Optional


RUNNING_STATUSES = {"queued", "running"}


def _max_active_jobs() -> Optional[int]:
    raw = os.environ.get("MAX_ACTIVE_JOBS")
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def list_active_jobs() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    from app.services.datasets import JOB_MANAGER as DATASET_JOB_MANAGER
    from app.services.calibrate_models import CALIBRATION_JOB_MANAGER
    from app.services.polymarket_history import POLYMARKET_HISTORY_JOB_MANAGER
    from app.services.markets import MARKETS_JOB_MANAGER
    try:
        from app.services.polymarket_subgraph import SUBGRAPH_JOB_MANAGER
    except ModuleNotFoundError:
        SUBGRAPH_JOB_MANAGER = None
    from app.services.market_map import MARKET_MAP_JOB_MANAGER
    try:
        from app.services.build_bars import BUILD_BARS_JOB_MANAGER
    except ModuleNotFoundError:
        BUILD_BARS_JOB_MANAGER = None
    try:
        from app.services.build_features import BUILD_FEATURES_JOB_MANAGER
    except ModuleNotFoundError:
        BUILD_FEATURES_JOB_MANAGER = None
    try:
        from app.services.train_model import TRAIN_MODEL_JOB_MANAGER
    except ModuleNotFoundError:
        TRAIN_MODEL_JOB_MANAGER = None
    try:
        from app.services.backtests import BACKTEST_JOB_MANAGER
    except ModuleNotFoundError:
        BACKTEST_JOB_MANAGER = None
    try:
        from app.services.signals import SIGNALS_JOB_MANAGER
    except ModuleNotFoundError:
        SIGNALS_JOB_MANAGER = None

    for status in DATASET_JOB_MANAGER.list_jobs():
        if status.status not in RUNNING_STATUSES:
            continue
        detail = "Option chain dataset build"
        if status.progress:
            detail = (
                f"{status.progress.done}/{status.progress.total} jobs"
            )
        items.append(
            {
                "jobId": status.job_id,
                "name": "Option chain dataset",
                "detail": detail,
                "state": status.status,
            }
        )

    for status in CALIBRATION_JOB_MANAGER.list_jobs():
        if status.status not in RUNNING_STATUSES:
            continue
        detail = "Auto tuner" if status.mode == "auto" else "Manual calibration"
        items.append(
            {
                "jobId": status.job_id,
                "name": "Calibrate models",
                "detail": detail,
                "state": status.status,
            }
        )

    for status in POLYMARKET_HISTORY_JOB_MANAGER.list_jobs():
        if status.status not in RUNNING_STATUSES:
            continue
        items.append(
            {
                "jobId": status.job_id,
                "name": "Polymarket weekly history",
                "detail": "Weekly backfill",
                "state": status.status,
            }
        )

    for status in MARKETS_JOB_MANAGER.list_jobs():
        if status.status not in RUNNING_STATUSES:
            continue
        detail = status.progress.stage if status.progress and status.progress.stage else "Markets refresh"
        items.append(
            {
                "jobId": status.job_id,
                "name": "Markets refresh",
                "detail": detail,
                "state": status.status,
            }
        )

    if SUBGRAPH_JOB_MANAGER is not None:
        for status in SUBGRAPH_JOB_MANAGER.list_jobs():
            if status.status not in RUNNING_STATUSES:
                continue
            items.append(
                {
                    "jobId": status.job_id,
                    "name": "Subgraph ingest",
                    "detail": "GraphQL pull",
                    "state": status.status,
                }
            )

    for status in MARKET_MAP_JOB_MANAGER.list_jobs():
        if status.status not in RUNNING_STATUSES:
            continue
        items.append(
            {
                "jobId": status.job_id,
                "name": "Market map",
                "detail": "dim_market build",
                "state": status.status,
            }
        )

    if BUILD_BARS_JOB_MANAGER is not None:
        for status in BUILD_BARS_JOB_MANAGER.list_jobs():
            if status.status not in RUNNING_STATUSES:
                continue
            items.append(
                {
                    "jobId": status.job_id,
                    "name": "Build bars",
                    "detail": "OHLC aggregation",
                    "state": status.status,
                }
            )

    if BUILD_FEATURES_JOB_MANAGER is not None:
        for status in BUILD_FEATURES_JOB_MANAGER.list_jobs():
            if status.status not in RUNNING_STATUSES:
                continue
            items.append(
                {
                    "jobId": status.job_id,
                    "name": "Build features",
                    "detail": "Decision features",
                    "state": status.status,
                }
            )

    if TRAIN_MODEL_JOB_MANAGER is not None:
        for status in TRAIN_MODEL_JOB_MANAGER.list_jobs():
            if status.status not in RUNNING_STATUSES:
                continue
            items.append(
                {
                    "jobId": status.job_id,
                    "name": "Train model",
                    "detail": "Mixed model training",
                    "state": status.status,
                }
            )

    if BACKTEST_JOB_MANAGER is not None:
        for status in BACKTEST_JOB_MANAGER.list_jobs():
            if status.status not in RUNNING_STATUSES:
                continue
            items.append(
                {
                    "jobId": status.job_id,
                    "name": "Backtest",
                    "detail": "Trade simulation",
                    "state": status.status,
                }
            )

    if SIGNALS_JOB_MANAGER is not None:
        for status in SIGNALS_JOB_MANAGER.list_jobs():
            if status.status not in RUNNING_STATUSES:
                continue
            items.append(
                {
                    "jobId": status.job_id,
                    "name": "Signals",
                    "detail": "Signal generation",
                    "state": status.status,
                }
            )

    return items


def ensure_no_active_jobs() -> None:
    max_jobs = _max_active_jobs()
    if max_jobs is None:
        return
    active = list_active_jobs()
    if len(active) < max_jobs:
        return
    current = active[0]
    raise RuntimeError(
        f"Maximum active jobs reached ({len(active)}/{max_jobs}). "
        f"Wait for {current['name']} to finish before starting a new one."
    )
