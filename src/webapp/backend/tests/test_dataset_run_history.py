from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "src" / "webapp" / "backend"
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.models.datasets import DatasetRunRequest
from app.services import datasets


def _configure_dataset_roots(monkeypatch, base_dir: Path) -> Path:
    dataset_root = base_dir / "src" / "data" / "raw" / "option-chain"
    dataset_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(datasets, "BASE_DIR", base_dir)
    monkeypatch.setattr(datasets, "DATASET_BASE_DIRS", [dataset_root])
    return dataset_root


def _write_run_artifacts(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = run_dir.name
    (run_dir / f"training-{dataset_name}.csv").write_text("col\n1\n")
    (run_dir / f"snapshot-{dataset_name}.csv").write_text("col\n1\n")


def test_list_dataset_runs_includes_predicted_creating_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_root = _configure_dataset_roots(monkeypatch, tmp_path)
    _write_run_artifacts(dataset_root / "existing-dataset")

    manager = datasets.DatasetJobManager()
    job = datasets.DatasetJob(
        "job-pending",
        DatasetRunRequest(
            out_dir="src/data/raw/option-chain",
            dataset_name="New Dataset",
            start="2025-01-01",
            end="2025-01-31",
        ),
    )
    job.status = "running"
    job.started_at = datetime(2026, 3, 12, 9, 0, tzinfo=timezone.utc)
    manager._jobs[job.job_id] = job
    monkeypatch.setattr(datasets, "JOB_MANAGER", manager)

    response = datasets.list_dataset_runs()

    creating = next(run for run in response.runs if run.job_id == "job-pending")
    ready = next(run for run in response.runs if run.id.endswith("/existing-dataset"))

    assert creating.run_dir == "src/data/raw/option-chain/new-dataset"
    assert creating.status == "creating"
    assert creating.files == []
    assert response.runs[0].id == creating.id
    assert ready.status == "ready"


def test_list_dataset_runs_tags_existing_directory_without_duplication(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_root = _configure_dataset_roots(monkeypatch, tmp_path)
    run_dir = dataset_root / "tagged-dataset"
    _write_run_artifacts(run_dir)

    manager = datasets.DatasetJobManager()
    job = datasets.DatasetJob(
        "job-existing",
        DatasetRunRequest(
            out_dir="src/data/raw/option-chain",
            dataset_name="Tagged Dataset",
            start="2025-01-01",
            end="2025-01-31",
        ),
    )
    job.status = "running"
    job.started_at = datetime(2026, 3, 12, 9, 30, tzinfo=timezone.utc)
    job.run_dir_path = run_dir
    manager._jobs[job.job_id] = job
    monkeypatch.setattr(datasets, "JOB_MANAGER", manager)

    response = datasets.list_dataset_runs()

    tagged_runs = [run for run in response.runs if run.id.endswith("/tagged-dataset")]

    assert len(tagged_runs) == 1
    assert tagged_runs[0].status == "creating"
    assert tagged_runs[0].job_id == "job-existing"
    assert tagged_runs[0].training_file is not None
