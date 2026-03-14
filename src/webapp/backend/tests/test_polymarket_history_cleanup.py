from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "src" / "webapp" / "backend"
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.models.polymarket_history import PolymarketHistoryRunRequest
from app.services import polymarket_history


def _configure_polymarket_roots(monkeypatch, base_dir: Path) -> Path:
    out_dir = base_dir / "src" / "data" / "raw" / "polymarket" / "weekly_history"
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(polymarket_history, "BASE_DIR", base_dir)
    monkeypatch.setattr(polymarket_history, "DEFAULT_OUT_DIR", out_dir)
    monkeypatch.setattr(polymarket_history, "RUNS_DIR", runs_dir)
    return out_dir


def _wait_for(predicate, *, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    assert predicate()


def _fake_history_command(out_dir: Path, script: str):
    def _builder(_payload: PolymarketHistoryRunRequest):
        return [sys.executable, "-c", script, str(out_dir)], {}, out_dir, []

    return _builder


def test_cancelled_polymarket_job_removes_known_run_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = _configure_polymarket_roots(monkeypatch, tmp_path)
    run_id = "cancelled-run"
    run_dir = out_dir / "runs" / run_id

    script = dedent(
        """
        import pathlib
        import sys
        import time

        out_dir = pathlib.Path(sys.argv[1])
        run_id = "cancelled-run"
        run_dir = out_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "weekly_markets.csv").write_text("market_id\\n1\\n")
        print(f"[Weekly History] run_id={run_id}", flush=True)
        time.sleep(30)
        """
    )
    monkeypatch.setattr(
        polymarket_history,
        "_build_history_command",
        _fake_history_command(out_dir, script),
    )

    job = polymarket_history.PolymarketHistoryJob(
        "job-cancel-known",
        PolymarketHistoryRunRequest(),
    )
    job.start()

    _wait_for(lambda: run_dir.exists())
    latest_path = out_dir / "latest.json"
    latest_path.write_text(json.dumps({"run_id": run_id}))

    job.cancel()
    assert job._thread is not None
    job._thread.join(timeout=10)

    assert job.status == "cancelled"
    assert not run_dir.exists()
    assert not latest_path.exists()
    assert job.result is not None
    assert job.result.files == []
    assert job.result.run_dir == "src/data/raw/polymarket/weekly_history/runs/cancelled-run"


def test_cancelled_polymarket_job_removes_new_run_dir_without_parsed_run_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = _configure_polymarket_roots(monkeypatch, tmp_path)
    orphan_run_dir = out_dir / "runs" / "orphan-run"

    script = dedent(
        """
        import pathlib
        import sys
        import time

        out_dir = pathlib.Path(sys.argv[1])
        run_dir = out_dir / "runs" / "orphan-run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "weekly_markets.csv").write_text("market_id\\n1\\n")
        time.sleep(30)
        """
    )
    monkeypatch.setattr(
        polymarket_history,
        "_build_history_command",
        _fake_history_command(out_dir, script),
    )

    job = polymarket_history.PolymarketHistoryJob(
        "job-cancel-orphan",
        PolymarketHistoryRunRequest(),
    )
    job.start()

    _wait_for(lambda: orphan_run_dir.exists())

    job.cancel()
    assert job._thread is not None
    job._thread.join(timeout=10)

    assert job.status == "cancelled"
    assert not orphan_run_dir.exists()
    assert job.result is not None
    assert job.result.files == []


def test_list_pipeline_runs_prunes_cancelled_run_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = _configure_polymarket_roots(monkeypatch, tmp_path)
    run_dir = out_dir / "runs" / "cancelled-history"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"status": "cancelled"}))
    (run_dir / "weekly_markets.csv").write_text("market_id\n1\n")
    (out_dir / "latest.json").write_text(json.dumps({"run_id": "cancelled-history"}))

    runs = polymarket_history.list_pipeline_runs()

    assert runs == []
    assert not run_dir.exists()
    assert not (out_dir / "latest.json").exists()


def test_run_polymarket_history_failed_process_prunes_run_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = _configure_polymarket_roots(monkeypatch, tmp_path)
    failed_run_dir = out_dir / "runs" / "failed-run"

    script = dedent(
        """
        import pathlib
        import sys

        out_dir = pathlib.Path(sys.argv[1])
        run_id = "failed-run"
        run_dir = out_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "weekly_markets.csv").write_text("market_id\\n1\\n")
        print(f"[Weekly History] run_id={run_id}", flush=True)
        raise SystemExit(3)
        """
    )
    monkeypatch.setattr(
        polymarket_history,
        "_build_history_command",
        _fake_history_command(out_dir, script),
    )

    response = polymarket_history.run_polymarket_history(PolymarketHistoryRunRequest())

    assert response.ok is False
    assert response.files == []
    assert response.run_dir == "src/data/raw/polymarket/weekly_history/runs/failed-run"
    assert not failed_run_dir.exists()
