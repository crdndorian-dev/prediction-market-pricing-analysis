from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "src" / "webapp" / "backend"
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"

for path in (BACKEND_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.models.polymarket_history import (
    PolymarketHistoryJobStatus,
    PolymarketHistoryRunRequest,
    PolymarketRunFeaturesRequest,
)
from app.services import polymarket_history as history_service
from data_collection.polymarket import weekly_history_v1
from feature_engineering.polymarket.build_features_v1 import _load_bars
from polymarket.master_bars import migrate_partitioned_bars_to_master, upsert_master_bars


def test_polymarket_history_job_status_accepts_prn_phase() -> None:
    status = PolymarketHistoryJobStatus(
        job_id="job-1",
        status="running",
        phase="prn",
    )

    assert status.phase == "prn"


def test_polymarket_history_job_status_accepts_quality_phase() -> None:
    status = PolymarketHistoryJobStatus(
        job_id="job-2",
        status="running",
        phase="quality",
    )

    assert status.phase == "quality"


def test_upsert_master_bars_prefers_newer_rows(tmp_path: Path) -> None:
    bars_dir = tmp_path / "bars_history"
    first = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-02T10:00:00Z",
                "market_id": "123",
                "close": 0.42,
                "written_by_run_id": "old-run",
                "bar_source": "clob_fallback",
                "schema_version": "pm_bars_history_v1.0",
            }
        ]
    )
    second = pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-02T10:00:00Z",
                "market_id": "123",
                "close": 0.55,
                "written_by_run_id": "new-run",
                "bar_source": "subgraph",
                "schema_version": "pm_bars_history_v1.0",
            }
        ]
    )

    upsert_master_bars(first, bars_dir, "1h")
    summary = upsert_master_bars(second, bars_dir, "1h")

    written = pd.read_csv(bars_dir / "1h.csv", dtype={"market_id": str})
    assert summary["rows_total"] == 1
    assert written.loc[0, "close"] == pytest.approx(0.55)
    assert written.loc[0, "bar_source"] == "subgraph"
    assert written.loc[0, "written_by_run_id"] == "new-run"


def test_migrate_partitioned_bars_to_master_replaces_tree(tmp_path: Path) -> None:
    part_dir = tmp_path / "bars_history" / "1h" / "market_id=321" / "date=2026-01-03"
    part_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-03T12:00:00Z",
                "market_id": "321",
                "open": 0.1,
                "high": 0.2,
                "low": 0.1,
                "close": 0.2,
                "volume": 5,
                "trade_count": 2,
                "schema_version": "pm_bars_history_v1.0",
            }
        ]
    ).to_csv(part_dir / "bars.csv", index=False)

    summary = migrate_partitioned_bars_to_master(tmp_path / "bars_history")

    assert (tmp_path / "bars_history" / "1h.csv").exists()
    assert not (tmp_path / "bars_history" / "1h").exists()
    assert summary["1h"]["rows_total"] == 1


def test_load_bars_reads_master_csv_with_filters(tmp_path: Path) -> None:
    bars_dir = tmp_path / "bars_history"
    bars_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2026-01-01T10:00:00Z",
                "market_id": "m1",
                "close": 0.11,
            },
            {
                "timestamp_utc": "2026-01-04T10:00:00Z",
                "market_id": "m2",
                "close": 0.22,
            },
        ]
    ).to_csv(bars_dir / "1h.csv", index=False)

    loaded = _load_bars(
        bars_dir,
        "1h",
        market_ids=["m1"],
        start_date="2026-01-01",
        end_date="2026-01-02",
    )

    assert list(loaded["market_id"]) == ["m1"]
    assert loaded.iloc[0]["close"] == pytest.approx(0.11)


def test_master_bar_service_helpers_expose_run_local_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path
    out_dir = base_dir / "src" / "data" / "raw" / "polymarket" / "weekly_history"
    runs_dir = out_dir / "runs"
    run_dir = runs_dir / "main-run"
    bars_dir = run_dir / "bars_history"
    bars_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "close": 0.3}]
    ).to_csv(bars_dir / "1h.csv", index=False)
    (out_dir / "latest.json").write_text('{"run_id": "main-run"}')
    manifest = {
        "bars_dir": str(bars_dir),
        "master_bars": {"1h": {"path": str(bars_dir / "1h.csv"), "rows_total": 1}},
    }

    monkeypatch.setattr(history_service, "BASE_DIR", base_dir)
    monkeypatch.setattr(history_service, "DEFAULT_OUT_DIR", out_dir)
    monkeypatch.setattr(history_service, "RUNS_DIR", runs_dir)

    artifacts = history_service._build_run_master_bar_artifacts(run_dir, manifest)
    groups = history_service._build_run_artifact_groups(run_dir, manifest)
    preview = history_service.get_run_master_bar_csv_preview("main-run", "1h", limit=5, mode="head")
    alias_preview = history_service.get_master_bar_csv_preview("1h", limit=5, mode="head")
    file_path = history_service.get_run_master_bar_file_path("main-run", "1h")

    assert artifacts[0]["frequency"] == "1h"
    assert artifacts[0]["row_count"] == 1
    assert artifacts[0]["path"] == "bars_history/1h.csv"
    assert any(group["key"] == "bars_history" for group in groups)
    assert preview["filename"] == "1h.csv"
    assert preview["rows"][0]["market_id"] == "1"
    assert alias_preview["rows"][0]["market_id"] == "1"
    assert file_path == bars_dir / "1h.csv"
    assert history_service._build_shared_artifacts({}) == []

    with pytest.raises(ValueError):
        history_service.get_master_bar_file_path("5m")


def test_build_history_command_uses_run_local_bars_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script_path = tmp_path / "weekly_history_v1.py"
    script_path.write_text("print('ok')\n")
    base_dir = tmp_path
    out_dir = base_dir / "src" / "data" / "raw" / "polymarket" / "weekly_history"

    monkeypatch.setattr(history_service, "BASE_DIR", base_dir)
    monkeypatch.setattr(history_service, "SCRIPT_PATH", script_path)
    monkeypatch.setattr(history_service, "DEFAULT_OUT_DIR", out_dir)

    payload = PolymarketHistoryRunRequest(run_dir_name="run-local-bars")
    cmd, _env, resolved_out_dir, _temp_paths = history_service._build_history_command(payload)

    bars_dir = out_dir / "runs" / "run-local-bars" / "bars_history"
    assert resolved_out_dir == out_dir
    assert payload.bars_dir == str(bars_dir)
    assert "--bars-dir" in cmd
    assert str(bars_dir) in cmd


def test_list_pipeline_runs_exposes_run_local_bar_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path
    out_dir = base_dir / "src" / "data" / "raw" / "polymarket" / "weekly_history"
    runs_dir = out_dir / "runs"
    run_dir = runs_dir / "dataset-a"
    bars_dir = run_dir / "bars_history"
    bars_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "close": 0.3}]
    ).to_csv(bars_dir / "1h.csv", index=False)
    (run_dir / "weekly_markets.csv").write_text("market_id\n1\n")
    (run_dir / "manifest.json").write_text(
        """
{
  "status": "success",
  "created_at_utc": "2026-01-03T00:00:00+00:00",
  "bars_dir": "%s",
  "master_bars": {"1h": {"path": "%s", "rows_total": 1}}
}
"""
        % (bars_dir, bars_dir / "1h.csv")
    )

    monkeypatch.setattr(history_service, "BASE_DIR", base_dir)
    monkeypatch.setattr(history_service, "DEFAULT_OUT_DIR", out_dir)
    monkeypatch.setattr(history_service, "RUNS_DIR", runs_dir)

    runs = history_service.list_pipeline_runs()

    assert len(runs) == 1
    run = runs[0]
    assert run["shared_artifacts"] == []
    assert run["master_bar_artifacts"][0]["path"] == "bars_history/1h.csv"
    assert any(group["key"] == "bars_history" for group in run["artifact_groups"])


def test_run_artifact_groups_include_side_split_history_and_analysis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path
    out_dir = base_dir / "src" / "data" / "raw" / "polymarket" / "weekly_history"
    runs_dir = out_dir / "runs"
    run_dir = runs_dir / "dataset-sides"
    legacy_bars_dir = run_dir / "bars_history"
    raw_yes_dir = run_dir / "raw" / "yes"
    raw_no_dir = run_dir / "raw" / "no"
    analysis_yes_dir = run_dir / "analysis" / "bars_history" / "yes"
    analysis_no_dir = run_dir / "analysis" / "bars_history" / "no"
    for directory in (legacy_bars_dir, raw_yes_dir, raw_no_dir, analysis_yes_dir, analysis_no_dir):
        directory.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "close": 0.3}]
    ).to_csv(legacy_bars_dir / "1h.csv", index=False)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "price": 0.31, "token_role": "yes"}]
    ).to_csv(raw_yes_dir / "price_history.csv", index=False)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "price": 0.69, "token_role": "no"}]
    ).to_csv(raw_no_dir / "price_history.csv", index=False)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "price": 0.32, "size": 5}]
    ).to_csv(raw_yes_dir / "subgraph_trades.csv", index=False)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "close": 0.3}]
    ).to_csv(analysis_yes_dir / "1h.csv", index=False)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "close": 0.7}]
    ).to_csv(analysis_no_dir / "1d.csv", index=False)
    (run_dir / "weekly_markets.csv").write_text("market_id\n1\n")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "created_at_utc": "2026-01-03T00:00:00+00:00",
                "bars_dir": str(legacy_bars_dir),
                "artifacts_accessible": True,
                "master_bars": {"1h": {"path": str(legacy_bars_dir / "1h.csv"), "rows_total": 1}},
            }
        )
    )

    monkeypatch.setattr(history_service, "BASE_DIR", base_dir)
    monkeypatch.setattr(history_service, "DEFAULT_OUT_DIR", out_dir)
    monkeypatch.setattr(history_service, "RUNS_DIR", runs_dir)

    groups = history_service._build_run_artifact_groups(run_dir, json.loads((run_dir / "manifest.json").read_text()))
    keys = {group["key"] for group in groups}
    preview = history_service.get_run_artifact_csv_preview(
        "dataset-sides",
        "raw/no/price_history.csv",
        limit=5,
        mode="head",
    )

    assert {"raw_yes", "raw_no", "analysis_bars_yes", "analysis_bars_no"} <= keys
    assert preview["rows"][0]["token_role"] == "no"
    assert any(group["path"] == "analysis/bars_history/yes" for group in groups)
    assert any(group["path"] == "analysis/bars_history/no" for group in groups)


def test_weekly_history_subgraph_partition_tracks_yes_and_no_sides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [
        {
            "id": "t-1",
            "blockNumber": "1",
            "timestamp": 1736200000,
            "marketId": "mkt-1",
            "outcomeTokenId": "yes-1",
            "price": "0.55",
            "size": "12",
            "side": "buy",
            "transactionHash": "0x1",
        },
        {
            "id": "t-2",
            "blockNumber": "2",
            "timestamp": 1736200600,
            "marketId": "mkt-1",
            "outcomeTokenId": "no-1",
            "price": "0.45",
            "size": "7",
            "side": "sell",
            "transactionHash": "0x2",
        },
    ]

    class _FakePull:
        total_entities = 2
        run_dir = tmp_path / "subgraph-run"
        run_id = "subgraph-run"

    class _FakeClient:
        def pull(self, _sq, variable_overrides=None):  # noqa: ANN001
            assert variable_overrides["marketIds"] == ["mkt-1"]
            return _FakePull()

        def entities_from_run(self, _run_dir):  # noqa: ANN001
            return entities

    import polymarket.graphql_queries as graphql_queries
    import polymarket.subgraph_client as subgraph_client

    monkeypatch.setattr(subgraph_client, "SubgraphClient", _FakeClient)
    monkeypatch.setattr(graphql_queries, "get_query", lambda _name: object())
    monkeypatch.setattr(weekly_history_v1, "_write_trade_partitions", lambda df, out_dir: len(df))

    info, side_trades = weekly_history_v1.maybe_ingest_subgraph_trades(
        ["mkt-1"],
        None,
        weekly_history_v1.Config(include_subgraph=True),
        tmp_path / "raw",
        yes_token_ids_by_market={"mkt-1": "yes-1"},
        no_token_ids_by_market={"mkt-1": "no-1"},
    )

    assert info["ok"] is True
    assert info["yes_entities"] == 1
    assert info["no_entities"] == 1
    assert info["partitions"] == {"yes": 1, "no": 1}
    assert set(side_trades) == {"yes", "no"}
    assert side_trades["yes"].iloc[0]["outcome_token_id"] == "yes-1"
    assert side_trades["no"].iloc[0]["outcome_token_id"] == "no-1"


def test_pending_feature_runs_hide_artifacts_until_manual_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_dir = tmp_path
    out_dir = base_dir / "src" / "data" / "raw" / "polymarket" / "weekly_history"
    runs_dir = out_dir / "runs"
    run_dir = runs_dir / "dataset-pending"
    bars_dir = run_dir / "bars_history"
    bars_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "close": 0.3}]
    ).to_csv(bars_dir / "1h.csv", index=False)
    pd.DataFrame(
        [{"timestamp_utc": "2026-01-02T00:00:00Z", "market_id": "1", "close": 0.3}]
    ).to_csv(bars_dir / "1d.csv", index=False)
    (run_dir / "weekly_markets.csv").write_text("market_id\n1\n")
    (run_dir / "price_history.csv").write_text(
        "timestamp_utc,price,market_id,token_role\n2026-01-02T00:00:00Z,0.3,1,yes\n"
    )
    (run_dir / "dim_market_weekly.csv").write_text("market_id\n1\n")
    prn_dir = run_dir / "prn_dataset"
    prn_dir.mkdir(parents=True, exist_ok=True)
    (prn_dir / "training-dataset-pending-prn.csv").write_text(
        "market_id,snapshot_date,asof_time,pRN,coverage_status\n1,2026-01-02,2026-01-02T21:00:00Z,0.5,ok\n"
    )
    (run_dir / "market_quality.csv").write_text("market_id,quality_issue_count\n1,0\n")
    (run_dir / "market_quality_summary.json").write_text(
        json.dumps(
            {
                "market_count": 1,
                "flagged_market_count": 0,
                "flagged_share": 0.0,
                "bucket_counts": {"clean": 1, "watch": 0, "noisy": 0},
                "prn_coverage_counts": {"ok": 1},
                "top_flags": [],
                "top_problem_tickers": [],
            }
        )
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "pending",
                "created_at_utc": "2026-01-03T00:00:00+00:00",
                "bars_dir": str(bars_dir),
                "master_bars": {
                    "1h": {"path": str(bars_dir / "1h.csv"), "rows_total": 1},
                    "1d": {"path": str(bars_dir / "1d.csv"), "rows_total": 1},
                },
                "pipeline_args": {"build_features": True},
                "build_features_requested": True,
                "features_built": False,
                "artifacts_accessible": False,
                "pending_phase": "features",
            }
        )
    )

    monkeypatch.setattr(history_service, "BASE_DIR", base_dir)
    monkeypatch.setattr(history_service, "DEFAULT_OUT_DIR", out_dir)
    monkeypatch.setattr(history_service, "RUNS_DIR", runs_dir)

    runs = history_service.list_pipeline_runs()

    assert len(runs) == 1
    run = runs[0]
    assert run["status"] == "pending"
    assert run["artifacts_accessible"] is False
    assert run["pending_phase"] == "features"
    assert run["master_bar_artifacts"] == []
    assert run["artifact_groups"] == []

    with pytest.raises(RuntimeError):
        history_service.get_run_master_bar_file_path("dataset-pending", "1h")

    features_script = (
        "import json, pathlib, sys;"
        "run_dir = pathlib.Path(sys.argv[1]);"
        "(run_dir / 'decision_features.csv').write_text('market_id,feature\\n1,1\\n');"
        "(run_dir / 'feature_manifest.json').write_text(json.dumps({'features_built': True}));"
        "print('[features] done', flush=True)"
    )

    def _fake_features_command(_payload, _bars_dir, _dim_market_path, out_dir):
        return [sys.executable, "-c", features_script, str(out_dir)], {}

    monkeypatch.setattr(history_service, "_build_features_command", _fake_features_command)

    response = history_service.build_run_decision_features(
        "dataset-pending",
        PolymarketRunFeaturesRequest(),
    )

    assert response.ok is True
    assert response.features_built is True

    runs = history_service.list_pipeline_runs()
    assert runs[0]["status"] == "success"
    assert runs[0]["artifacts_accessible"] is True
    assert runs[0]["master_bar_artifacts"][0]["path"] == "bars_history/1h.csv"
    assert history_service.get_run_master_bar_file_path("dataset-pending", "1h") == bars_dir / "1h.csv"
