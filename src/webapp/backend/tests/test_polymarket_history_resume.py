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
from app.services.process_runtime import write_runtime_file


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
    def _builder(payload: PolymarketHistoryRunRequest):
        run_id = payload.run_dir_name or "test-run"
        return [sys.executable, "-c", script, str(out_dir), run_id], {}, out_dir, []

    return _builder


def _fake_prn_command(_payload: PolymarketHistoryRunRequest, run_dir: Path):
    script = dedent(
        """
        import pathlib
        import sys

        run_dir = pathlib.Path(sys.argv[1])
        (run_dir / "prn_dataset").mkdir(parents=True, exist_ok=True)
        print("[pRN] noop", flush=True)
        """
    )
    return [sys.executable, "-c", script, str(run_dir)], {}


def test_polymarket_job_retries_history_phase_from_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = _configure_polymarket_roots(monkeypatch, tmp_path)
    run_id = "resume-history-run"

    script = dedent(
        """
        import json
        import pathlib
        import sys

        out_dir = pathlib.Path(sys.argv[1])
        run_id = sys.argv[2]
        run_dir = out_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        sentinel = run_dir / "attempt_once.flag"
        state_path = run_dir / ".history_resume_state.json"
        weekly_markets = run_dir / "weekly_markets.csv"
        price_history = run_dir / "price_history.csv"

        weekly_markets.write_text("market_id,event_id,ticker,threshold,week_monday,week_friday,event_endDate,yes_token_id\\n"
                                  "mkt-1,evt-1,NVDA,100,2026-01-05,2026-01-09,2026-01-09,tok-1\\n"
                                  "mkt-2,evt-2,NVDA,110,2026-01-05,2026-01-09,2026-01-09,tok-2\\n")
        price_history.write_text("timestamp_utc,price,market_id,token_role\\n2026-01-06T00:00:00Z,0.51,mkt-1,yes\\n")

        if not sentinel.exists():
            state = {
                "run_id": run_id,
                "status": "running",
                "phase": "history",
                "markets_total": 2,
                "completed_market_ids": ["mkt-1"],
                "failed_market_ids": [],
            }
            state_path.write_text(json.dumps(state))
            print(f"[Weekly History] run_id={run_id}", flush=True)
            sentinel.write_text("1")
            raise SystemExit(2)

        state = {
            "run_id": run_id,
            "status": "completed",
            "phase": "complete",
            "markets_total": 2,
            "completed_market_ids": ["mkt-1", "mkt-2"],
            "failed_market_ids": [],
        }
        state_path.write_text(json.dumps(state))
        manifest = {
            "run_id": run_id,
            "created_at_utc": "2026-03-13T00:00:00+00:00",
            "tickers": ["NVDA"],
            "start_date": "2026-01-05",
            "end_date": "2026-01-09",
            "markets": 2,
            "price_rows": 1,
            "master_bars": {},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        print(f"[Weekly History] run_id={run_id}", flush=True)
        print("[Weekly History] Market complete 2/2 job_id=NVDA:110:mkt-2 status=ok", flush=True)
        """
    )
    monkeypatch.setattr(
        polymarket_history,
        "_build_history_command",
        _fake_history_command(out_dir, script),
    )
    monkeypatch.setattr(
        polymarket_history,
        "_build_run_prn_refresh_command",
        _fake_prn_command,
    )

    job = polymarket_history.PolymarketHistoryJob(
        "job-resume-history",
        PolymarketHistoryRunRequest(run_dir_name=run_id),
    )
    job.start()
    assert job._thread is not None
    job._thread.join(timeout=10)

    assert job.status == "finished"
    assert job.result is not None
    assert job.result.ok is True
    assert job.result.run_id == run_id
    assert "Retrying from checkpoint" in job.result.stdout
    assert (out_dir / "runs" / run_id / "manifest.json").exists()
    assert not (out_dir / "runs" / run_id / ".job_runtime.json").exists()


def test_polymarket_job_copies_dim_market_before_prn_refresh(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = _configure_polymarket_roots(monkeypatch, tmp_path)
    run_id = "copy-dim-before-prn"
    dim_market_path = tmp_path / "src" / "data" / "models" / "polymarket" / "dim_market_weekly.csv"
    dim_market_path.parent.mkdir(parents=True, exist_ok=True)
    dim_market_path.write_text("market_id,ticker\nmkt-1,NVDA\n")
    monkeypatch.setattr(polymarket_history, "DIM_MARKET_WEEKLY_PATH", dim_market_path)

    history_script = dedent(
        """
        import json
        import pathlib
        import sys

        out_dir = pathlib.Path(sys.argv[1])
        run_id = sys.argv[2]
        run_dir = out_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ".history_resume_state.json").write_text(json.dumps({
            "run_id": run_id,
            "status": "completed",
            "phase": "complete",
            "markets_total": 1,
            "completed_market_ids": ["mkt-1"],
            "failed_market_ids": [],
        }))
        (run_dir / "weekly_markets.csv").write_text("market_id\\nmkt-1\\n")
        (run_dir / "price_history.csv").write_text(
            "timestamp_utc,price,market_id,token_role\\n2026-01-06T00:00:00Z,0.5,mkt-1,yes\\n"
        )
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id,
            "created_at_utc": "2026-03-13T00:00:00+00:00",
            "tickers": ["NVDA"],
            "start_date": "2026-01-05",
            "end_date": "2026-01-09",
            "markets": 1,
            "price_rows": 1,
            "master_bars": {},
        }))
        print(f"[Weekly History] run_id={run_id}", flush=True)
        """
    )

    def _fake_prn_command(_payload: PolymarketHistoryRunRequest, run_dir: Path):
        script = dedent(
            """
            import json
            import pathlib
            import sys

            run_dir = pathlib.Path(sys.argv[1])
            dim_market_path = run_dir / "dim_market_weekly.csv"
            if not dim_market_path.exists():
                raise SystemExit("dim_market_weekly.csv missing before pRN refresh")
            prn_dir = run_dir / "prn_dataset"
            prn_dir.mkdir(parents=True, exist_ok=True)
            (prn_dir / f"training-{run_dir.name}-prn.csv").write_text(
                "market_id,snapshot_date,asof_time,pRN,coverage_status\\n"
                "mkt-1,2026-01-06,2026-01-06T21:00:00Z,0.5,ok\\n"
            )
            (run_dir / "market_quality.csv").write_text("market_id,quality_issue_count\\nmkt-1,0\\n")
            (run_dir / "market_quality_summary.json").write_text(json.dumps({"market_count": 1}))
            (run_dir / "markets_prn_hourly.csv").write_text("timestamp_utc,market_id,pRN\\n")
            (run_dir / "snapshot_daily.csv").write_text("market_id\\nmkt-1\\n")
            print("[Run pRN] ok", flush=True)
            """
        )
        return [sys.executable, "-c", script, str(run_dir)], {}

    monkeypatch.setattr(
        polymarket_history,
        "_build_history_command",
        _fake_history_command(out_dir, history_script),
    )
    monkeypatch.setattr(polymarket_history, "_build_run_prn_refresh_command", _fake_prn_command)

    job = polymarket_history.PolymarketHistoryJob(
        "job-copy-dim-before-prn",
        PolymarketHistoryRunRequest(run_dir_name=run_id, build_features=False),
    )
    job.start()
    assert job._thread is not None
    job._thread.join(timeout=10)

    run_dir = out_dir / "runs" / run_id
    assert job.status == "finished"
    assert job.result is not None
    assert job.result.ok is True
    assert (run_dir / "dim_market_weekly.csv").exists()
    assert not (run_dir / "decision_features.parquet").exists()
    assert not (run_dir / "decision_features.csv").exists()
    assert not (run_dir / "feature_manifest.json").exists()


def test_polymarket_manager_recovers_stale_runtime_job_with_same_job_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = _configure_polymarket_roots(monkeypatch, tmp_path)
    run_id = "resume-after-restart"
    job_id = "job-runtime-recover"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".history_resume_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "running",
                "phase": "history",
                "markets_total": 2,
                "completed_market_ids": ["mkt-1"],
                "failed_market_ids": [],
            }
        )
    )

    runtime_payload = {
        "pid": 999999,
        "pgid": None,
        "job_id": job_id,
        "service": "polymarket_history",
        "started_at": "2026-03-13T10:00:00+00:00",
        "command": ["python", "weekly_history_v1.py"],
        "status": "running",
        "phase": "history",
        "request_payload": {
            "run_dir_name": run_id,
            "build_features": False,
            "include_subgraph": False,
            "resume_existing": True,
        },
    }
    write_runtime_file(run_dir, runtime_payload)

    script = dedent(
        """
        import json
        import pathlib
        import sys

        out_dir = pathlib.Path(sys.argv[1])
        run_id = sys.argv[2]
        run_dir = out_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / ".history_resume_state.json").write_text(json.dumps({
            "run_id": run_id,
            "status": "completed",
            "phase": "complete",
            "markets_total": 2,
            "completed_market_ids": ["mkt-1", "mkt-2"],
            "failed_market_ids": [],
        }))
        (run_dir / "weekly_markets.csv").write_text("market_id\\n1\\n")
        (run_dir / "price_history.csv").write_text("timestamp_utc,price,market_id,token_role\\n2026-01-06T00:00:00Z,0.5,mkt-1,yes\\n")
        (run_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id,
            "created_at_utc": "2026-03-13T10:00:00+00:00",
            "tickers": ["NVDA"],
            "start_date": "2026-01-05",
            "end_date": "2026-01-09",
            "markets": 2,
            "price_rows": 1,
            "master_bars": {},
        }))
        print(f"[Weekly History] run_id={run_id}", flush=True)
        print("[Weekly History] Market complete 2/2 job_id=NVDA:110:mkt-2 status=ok", flush=True)
        """
    )
    monkeypatch.setattr(
        polymarket_history,
        "_build_history_command",
        _fake_history_command(out_dir, script),
    )
    monkeypatch.setattr(
        polymarket_history,
        "_build_run_prn_refresh_command",
        _fake_prn_command,
    )

    manager = polymarket_history.PolymarketHistoryJobManager()

    initial_status = manager.get_status(job_id)
    assert initial_status.job_id == job_id
    assert initial_status.status in {"queued", "running"}

    _wait_for(lambda: manager.get_status(job_id).status == "finished")
    _wait_for(lambda: not (run_dir / ".job_runtime.json").exists())
    final_status = manager.get_status(job_id)
    assert final_status.status == "finished"
    assert final_status.result is not None
    assert final_status.result.run_id == run_id
    assert "job-runtime-recover" == final_status.job_id
    assert not (run_dir / ".job_runtime.json").exists()


def test_polymarket_manager_recovers_features_phase_after_restart(
    monkeypatch,
    tmp_path: Path,
) -> None:
    out_dir = _configure_polymarket_roots(monkeypatch, tmp_path)
    run_id = "resume-features-run"
    job_id = "job-runtime-features"
    run_dir = out_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / ".history_resume_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "phase": "complete",
                "markets_total": 2,
                "completed_market_ids": ["mkt-1", "mkt-2"],
                "failed_market_ids": [],
            }
        )
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at_utc": "2026-03-13T10:00:00+00:00",
                "tickers": ["NVDA"],
                "start_date": "2026-01-05",
                "end_date": "2026-01-09",
                "markets": 2,
                "price_rows": 4,
                "status": "pending",
                "pipeline_args": {
                    "build_features": True,
                    "run_dir_name": run_id,
                },
                "bars_dir": str(run_dir / "bars_history"),
                "master_bars": {},
                "features_built": False,
                "artifacts_accessible": False,
                "pending_phase": "features",
            }
        )
    )
    (run_dir / "weekly_markets.csv").write_text("market_id\nmkt-1\nmkt-2\n")
    (run_dir / "price_history.csv").write_text(
        "timestamp_utc,price,market_id,token_role\n2026-01-06T00:00:00Z,0.5,mkt-1,yes\n"
    )
    (run_dir / "market_quality.csv").write_text("market_id,quality_issue_count\nmkt-1,0\n")
    (run_dir / "market_quality_summary.json").write_text(
        json.dumps({"market_count": 2, "flagged_market_count": 0, "flagged_share": 0.0, "bucket_counts": {"clean": 2, "watch": 0, "noisy": 0}, "prn_coverage_counts": {"ok": 2}, "top_flags": [], "top_problem_tickers": []})
    )
    prn_dir = run_dir / "prn_dataset"
    prn_dir.mkdir(parents=True, exist_ok=True)
    (prn_dir / f"training-{run_id}-prn.csv").write_text(
        "market_id,snapshot_date,asof_time,pRN,coverage_status\nmkt-1,2026-01-06,2026-01-06T21:00:00Z,0.5,ok\n"
    )
    bars_dir = run_dir / "bars_history"
    bars_dir.mkdir(parents=True, exist_ok=True)
    (bars_dir / "1h.csv").write_text("timestamp_utc,market_id,close\n2026-01-06T21:00:00Z,mkt-1,0.5\n")
    (bars_dir / "1d.csv").write_text("timestamp_utc,market_id,close\n2026-01-06T00:00:00Z,mkt-1,0.5\n")
    (run_dir / "dim_market_weekly.csv").write_text("market_id\nmkt-1\n")

    runtime_payload = {
        "pid": 999999,
        "pgid": None,
        "job_id": job_id,
        "service": "polymarket_features",
        "started_at": "2026-03-13T10:00:00+00:00",
        "command": ["python", "weekly_history_v1.py"],
        "status": "running",
        "phase": "prn",
        "request_payload": {
            "run_dir_name": run_id,
            "build_features": True,
            "include_subgraph": False,
            "resume_existing": True,
            "skip_subgraph_labels": False,
        },
    }
    write_runtime_file(run_dir, runtime_payload)

    monkeypatch.setattr(
        polymarket_history,
        "_build_history_command",
        _fake_history_command(out_dir, "print('resume features')"),
    )

    features_script = dedent(
        """
        import json
        import pathlib
        import sys

        run_dir = pathlib.Path(sys.argv[1])
        (run_dir / "decision_features.csv").write_text("market_id,feature\\nmkt-1,1\\n")
        (run_dir / "feature_manifest.json").write_text(json.dumps({"features_built": True}))
        print("[features] PROGRESS 1/1 step=write_outputs", flush=True)
        """
    )

    def _fake_features_command(_payload, _bars_dir, _dim_market_path, out_dir):
        return [sys.executable, "-c", features_script, str(out_dir)], {}

    monkeypatch.setattr(polymarket_history, "_build_features_command", _fake_features_command)

    manager = polymarket_history.PolymarketHistoryJobManager()

    initial_status = manager.get_status(job_id)
    assert initial_status.job_id == job_id
    assert initial_status.status in {"queued", "running"}

    _wait_for(lambda: manager.get_status(job_id).status == "finished")
    _wait_for(lambda: not (run_dir / ".job_runtime.json").exists())
    final_status = manager.get_status(job_id)
    assert final_status.status == "finished"
    assert final_status.result is not None
    assert final_status.result.features_built is True
    assert final_status.result.features_path is not None
    assert (run_dir / final_status.result.features_path).exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "success"
    assert manifest["artifacts_accessible"] is True
    assert not (run_dir / ".job_runtime.json").exists()
