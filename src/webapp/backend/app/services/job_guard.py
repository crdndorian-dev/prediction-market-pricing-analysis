from __future__ import annotations

from typing import Dict, List


RUNNING_STATUSES = {"queued", "running"}


def list_active_jobs() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    from app.services.datasets import JOB_MANAGER as DATASET_JOB_MANAGER
    from app.services.calibrate_models import CALIBRATION_JOB_MANAGER
    from app.services.polymarket_snapshots import POLYMARKET_JOB_MANAGER
    from app.services.phat_edge import PHAT_EDGE_JOB_MANAGER

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
                "name": "Calibrate models",
                "detail": detail,
                "state": status.status,
            }
        )

    for status in POLYMARKET_JOB_MANAGER.list_jobs():
        if status.status not in RUNNING_STATUSES:
            continue
        items.append(
            {
                "name": "Polymarket snapshot",
                "detail": "Snapshot fetch",
                "state": status.status,
            }
        )

    for status in PHAT_EDGE_JOB_MANAGER.list_jobs():
        if status.status not in RUNNING_STATUSES:
            continue
        items.append(
            {
                "name": "Edge compute",
                "detail": "pHAT inference",
                "state": status.status,
            }
        )

    return items


def ensure_no_active_jobs() -> None:
    active = list_active_jobs()
    if not active:
        return
    current = active[0]
    raise RuntimeError(
        f"Another job is already running: {current['name']}. "
        "Wait for it to finish before starting a new one."
    )
