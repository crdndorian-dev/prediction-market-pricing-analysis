import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "webapp" / "backend"))

from app.models.markets import MarketsRefreshRequest
from app.services import markets


def test_start_markets_job_injects_resolved_run_id(monkeypatch, tmp_path):
    script_path = tmp_path / "fake-markets-refresh.py"
    script_path.write_text("#!/usr/bin/env python3\n")

    captured = {}

    def fake_start_job(payload):
        captured["payload"] = payload
        return "job-123"

    monkeypatch.setattr(markets, "SCRIPT_PATH", script_path)
    monkeypatch.setattr(markets, "_resolve_run_dir", lambda run_id: tmp_path / "runs" / "run-abc")
    monkeypatch.setattr(markets, "_clear_series_caches", lambda: None)
    monkeypatch.setattr(markets.MARKETS_JOB_MANAGER, "start_job", fake_start_job)

    original = MarketsRefreshRequest(week_friday="2026-02-27", force_refresh=True)
    job_id = markets.start_markets_job(original)

    assert job_id == "job-123"
    assert "payload" in captured
    assert captured["payload"].run_id == "run-abc"
    assert captured["payload"].week_friday == "2026-02-27"
    assert captured["payload"].force_refresh is True
    # Original request object should remain unchanged.
    assert original.run_id is None


def test_resolve_run_dir_falls_back_when_latest_pointer_is_stale(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / "weekly-history-20260212T014930Z").mkdir()

    latest_pointer = tmp_path / "latest.json"
    latest_pointer.write_text(json.dumps({"run_id": "weekly-history-20260225T163804Z"}))

    monkeypatch.setattr(markets, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(markets, "LATEST_POINTER_PATH", latest_pointer)

    resolved = markets._resolve_run_dir(None)
    assert resolved == runs_dir / "weekly-history-20260212T014930Z"


def test_resolve_run_dir_explicit_missing_run_raises(monkeypatch, tmp_path):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(markets, "RUNS_DIR", runs_dir)

    with pytest.raises(FileNotFoundError, match="does-not-exist"):
        markets._resolve_run_dir("does-not-exist")


def test_resolve_week_friday_uses_previous_friday_on_saturday(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
            return base if tz is None else base.astimezone(tz)

    monkeypatch.setattr(markets, "datetime", FixedDateTime)

    assert markets._resolve_week_friday(None) == date(2026, 3, 6)


def test_summarize_process_failure_prefers_stdout_fatal_message():
    message = markets._summarize_process_failure(
        ["[Markets] PROGRESS stage=prn current=13 total=13\n", "[FATAL] build failed\n"],
        [],
    )

    assert message == "build failed"
