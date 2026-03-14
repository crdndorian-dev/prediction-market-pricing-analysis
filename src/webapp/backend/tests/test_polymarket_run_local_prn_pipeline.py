from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "src" / "webapp" / "backend"
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.models.polymarket_history import PolymarketHistoryRunRequest
from app.services import polymarket_history, polymarket_quality, polymarket_run_prn
from feature_engineering.polymarket import build_features_v1
from option_chain import exact_builder
from orchestration.polymarket import markets_refresh_v1, run_prn_refresh_v1
from polymarket import quality_flags as polymarket_quality_flags


def test_exact_builder_returns_rows_for_polymarket_strike_absent_from_legacy_dataset(monkeypatch) -> None:
    legacy_training = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "threshold": 150.0,
                "week_friday": "2025-01-10",
            }
        ]
    )
    assert not legacy_training["threshold"].eq(153.5).any()

    weekly_markets = pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "event_id": "evt-1",
                "ticker": "AAPL",
                "threshold": 153.5,
                "week_monday": "2025-01-06",
                "week_friday": "2025-01-10",
                "event_endDate": "2025-01-10",
            }
        ]
    )

    def _fake_preload_stock_closes(**kwargs):
        return {}, {}, {}

    def _fake_preload_dividend_histories(**kwargs):
        return {}

    def _fake_process_one(**kwargs):
        asof_time = exact_builder._asof_time_for_date(
            kwargs["asof_target"],
            tz_name=kwargs["prn_asof_tz"],
            close_time=kwargs["prn_asof_close_time"],
        )
        rows = []
        for target in kwargs["targets"] or []:
            row = exact_builder._base_target_row(
                ticker=kwargs["ticker"],
                target=target,
                week_monday=kwargs["week_monday"],
                week_friday=kwargs["week_friday"],
                asof_target=kwargs["asof_target"],
                asof_date_used=kwargs["asof_target"],
                asof_time=asof_time,
                prn_version=kwargs["prn_version"],
                prn_config_hash=kwargs["prn_config_hash"],
                drop_reason=None,
                coverage_status="ok",
            )
            row.update(
                {
                    "pRN": 0.61,
                    "qRN": 0.39,
                    "pRN_raw": 0.61,
                    "qRN_raw": 0.39,
                    "rv20": 0.2,
                    "log_m": 0.01,
                    "abs_log_m": 0.01,
                    "log_m_fwd": 0.01,
                    "abs_log_m_fwd": 0.01,
                    "T_days": 4,
                    "S_asof_close": 152.0,
                    "forward_price": 152.5,
                    "dividend_yield": 0.0,
                    "theta_quote_source": "mock",
                }
            )
            rows.append(row)
        return rows, None

    monkeypatch.setattr(exact_builder, "preload_stock_closes", _fake_preload_stock_closes)
    monkeypatch.setattr(exact_builder, "preload_dividend_histories", _fake_preload_dividend_histories)
    monkeypatch.setattr(exact_builder, "process_one", _fake_process_one)

    result = exact_builder.build_polymarket_exact_prn(weekly_markets, threads=1)

    assert not result.rows.empty
    assert set(result.rows["threshold"].tolist()) == {153.5}
    assert set(result.rows["market_id"].tolist()) == {"mkt-1"}
    assert set(result.rows["coverage_status"].tolist()) == {"ok"}
    assert result.required_market_snapshots == 4
    assert result.ok_market_snapshots == 4


def test_attach_prn_features_daily_prefers_market_id_join() -> None:
    base = pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date": date(2025, 1, 10),
                "decision_date": date(2025, 1, 6),
                "timestamp_utc": pd.Timestamp("2025-01-06T21:00:01Z"),
            }
        ]
    )
    prn = pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "ticker": "MSFT",
                "threshold": 205.0,
                "expiry_date": date(2025, 1, 17),
                "snapshot_date": date(2025, 1, 6),
                "asof_time": pd.Timestamp("2025-01-06T21:00:00Z"),
                "pRN": 0.64,
                "coverage_status": "ok",
                "drop_reason": np.nan,
            }
        ]
    )

    merged = build_features_v1._attach_prn_features_daily(base, prn)

    assert len(merged) == 1
    assert merged.iloc[0]["market_id"] == "mkt-1"
    assert merged.iloc[0]["pRN"] == 0.64
    assert merged.iloc[0]["coverage_status"] == "ok"


def test_attach_prn_features_daily_falls_back_without_market_id() -> None:
    base = pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date": date(2025, 1, 10),
                "decision_date": date(2025, 1, 6),
                "timestamp_utc": pd.Timestamp("2025-01-06T21:00:01Z"),
            }
        ]
    )
    prn = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date": date(2025, 1, 10),
                "snapshot_date": date(2025, 1, 6),
                "asof_time": pd.Timestamp("2025-01-06T21:00:00Z"),
                "pRN": 0.58,
                "coverage_status": "ok",
            }
        ]
    )

    merged = build_features_v1._attach_prn_features_daily(base, prn)

    assert len(merged) == 1
    assert merged.iloc[0]["pRN"] == 0.58
    assert merged.iloc[0]["coverage_status"] == "ok"


def test_merge_prn_on_hourly_base_is_time_safe_and_market_id_keyed() -> None:
    hourly_base = pd.DataFrame(
        [
            {
                "timestamp_utc": "2025-01-06T20:00:00Z",
                "market_id": "mkt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date": date(2025, 1, 10),
            },
            {
                "timestamp_utc": "2025-01-06T21:00:00Z",
                "market_id": "mkt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date": date(2025, 1, 10),
            },
            {
                "timestamp_utc": "2025-01-06T22:00:00Z",
                "market_id": "mkt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date": date(2025, 1, 10),
            },
        ]
    )
    prn = pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "ticker": "MSFT",
                "threshold": 205.0,
                "expiry_date": date(2025, 1, 17),
                "asof_time": pd.Timestamp("2025-01-06T21:00:00Z"),
                "pRN": 0.7,
                "coverage_status": "ok",
            }
        ]
    )

    merged = markets_refresh_v1.merge_prn_on_hourly_base(hourly_base, prn)

    assert pd.isna(merged.iloc[0]["prn_asof_time"])
    assert pd.isna(merged.iloc[0]["pRN"])
    assert str(merged.iloc[1]["prn_asof_time"]) == "2025-01-06 21:00:00+00:00"
    assert merged.iloc[1]["pRN"] == 0.7
    assert merged.iloc[2]["pRN"] == 0.7


def test_refresh_run_local_prn_dataset_writes_theta_only_meta(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    weekly_markets = pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "event_id": "evt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "week_monday": "2025-01-06",
                "week_friday": "2025-01-10",
                "event_endDate": "2025-01-10",
            }
        ]
    )
    weekly_markets.to_csv(run_dir / "weekly_markets.csv", index=False)
    (run_dir / "manifest.json").write_text(json.dumps({"start_date": "2025-01-06", "end_date": "2025-01-10"}))

    fake_rows = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "week_friday": "2025-01-10",
                "coverage_status": "ok",
                "drop_reason": np.nan,
                "pRN": 0.6,
            },
            {
                "ticker": "AAPL",
                "week_friday": "2025-01-10",
                "coverage_status": "missing",
                "drop_reason": "empty_option_chain",
                "pRN": np.nan,
            },
        ]
    )

    def _fake_build_polymarket_exact_prn(*args, **kwargs):
        return exact_builder.PolymarketExactPrnBuildResult(
            rows=fake_rows,
            drop_logs=[{"drop_reason": "empty_option_chain"}],
            required_market_snapshots=2,
            ok_market_snapshots=1,
            missing_market_snapshots=1,
            coverage_counts={"ok": 1, "missing": 1},
            drop_reason_counts={"empty_option_chain": 1},
            prn_version="v1",
            prn_config_hash="cfg123",
        )

    monkeypatch.setattr(polymarket_run_prn, "build_polymarket_exact_prn", _fake_build_polymarket_exact_prn)

    result = polymarket_run_prn.refresh_run_local_prn_dataset(run_dir)

    training_path = run_dir / "prn_dataset" / f"training-{run_dir.name}-prn.csv"
    meta_path = run_dir / "prn_dataset" / "polymarket_run_prn_meta.json"
    build_meta_path = run_dir / "prn_dataset" / "dataset_build_meta.json"

    assert result.training_path == training_path
    assert training_path.exists()
    assert meta_path.exists()
    assert build_meta_path.exists()

    meta = json.loads(meta_path.read_text())
    build_meta = json.loads(build_meta_path.read_text())
    quality_path = run_dir / "market_quality.csv"
    quality_summary_path = run_dir / "market_quality_summary.json"

    assert meta["source_mode"] == "theta_only"
    assert meta["summary"]["required_market_snapshots"] == 2
    assert meta["summary"]["missing_market_snapshots"] == 1
    assert meta["summary"]["drop_reason_counts"] == {"empty_option_chain": 1}
    assert "quality_summary" in meta["summary"]
    assert build_meta["source_mode"] == "theta_only"
    assert build_meta["summary"]["coverage_counts"] == {"ok": 1, "missing": 1}
    assert quality_path.exists()
    assert quality_summary_path.exists()

    training = pd.read_csv(training_path)
    assert "prn_quality_issue_count" in training.columns
    assert "prn_quality_bucket" in training.columns

    quality = pd.read_csv(quality_path)
    assert "quality_issue_count" in quality.columns
    assert "quality_bucket" in quality.columns
    assert "flag_prn_missing" in quality.columns

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert "quality_summary" in manifest


def test_polymarket_service_commands_use_run_local_prn_without_selection(tmp_path: Path) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    prn_dir = run_dir / "prn_dataset"
    prn_dir.mkdir(parents=True, exist_ok=True)
    local_training = prn_dir / f"training-{run_dir.name}-prn.csv"
    local_training.write_text("ticker,K,asof_time,option_expiration_used,pRN\nAAPL,150,2025-01-06T21:00:00Z,2025-01-10,0.6\n")
    external_training = tmp_path / "external.csv"
    external_training.write_text("legacy\n1\n")

    payload = PolymarketHistoryRunRequest(
        prn_dataset=str(external_training),
        skip_subgraph_labels=True,
    )

    features_cmd, _ = polymarket_history._build_features_command(
        payload,
        bars_dir=tmp_path / "bars",
        dim_market_path=tmp_path / "dim_market.csv",
        out_dir=run_dir,
    )
    refresh_cmd, _ = polymarket_history._build_run_prn_refresh_command(payload, run_dir)

    assert "--prn-dataset" in features_cmd
    prn_flag_index = features_cmd.index("--prn-dataset")
    assert features_cmd[prn_flag_index + 1] == str(local_training)
    assert "--prn-dataset" not in refresh_cmd


def test_build_features_main_uses_run_local_prn_without_explicit_selection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    prn_dir = run_dir / "prn_dataset"
    prn_dir.mkdir(parents=True, exist_ok=True)

    training_path = prn_dir / f"training-{run_dir.name}-prn.csv"
    pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "event_id": "evt-1",
                "ticker": "AAPL",
                "K": 150.0,
                "threshold": 150.0,
                "snapshot_date": "2025-01-06",
                "asof_time": "2025-01-06T21:00:00Z",
                "option_expiration_used": "2025-01-10",
                "expiry_date": "2025-01-10",
                "week_friday": "2025-01-10",
                "coverage_status": "ok",
                "drop_reason": np.nan,
                "pRN": 0.6,
                "qRN": 0.4,
                "pRN_raw": 0.6,
                "qRN_raw": 0.4,
                "rv20": 0.2,
                "log_m": 0.01,
                "abs_log_m": 0.01,
                "log_m_fwd": 0.01,
                "abs_log_m_fwd": 0.01,
                "T_days": 4,
                "S_asof_close": 151.0,
                "forward_price": 151.3,
                "dividend_yield": 0.0,
                "theta_quote_source": "mock",
                "rn_method": "run_local_theta_exact",
            }
        ]
    ).to_csv(training_path, index=False)

    bars_dir = tmp_path / "bars_history"
    bars_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2025-01-06T20:00:00Z",
                "market_id": "mkt-1",
                "open": 0.55,
                "high": 0.55,
                "low": 0.55,
                "close": 0.55,
                "volume": 10.0,
                "trade_count": 1,
            }
        ]
    ).to_csv(bars_dir / "1d.csv", index=False)

    pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "event_id": "evt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "week_monday": "2025-01-06",
                "week_friday": "2025-01-10",
                "flag_prn_missing": False,
                "flag_asof_close_fallback": False,
                "flag_expiry_close_fallback": False,
                "flag_expiry_saturday_fallback": False,
                "flag_quote_close_fallback": False,
                "flag_low_chain_used": False,
                "flag_wide_rel_spread": False,
                "flag_pm_no_trade_history": False,
                "flag_pm_no_recent_trade": False,
                "flag_pm_stale_prices": True,
                "flag_pm_suspect_orderbook": False,
                "flag_pm_sparse_points": False,
                "flag_pm_low_volume": False,
                "flag_market_inactive": False,
                "flag_missing_token_ids": False,
                "flag_extreme_otm": False,
                "flag_not_relevant": False,
                "flag_prn_outside_curve_support": False,
                "snapshot_date_used": "2025-01-06",
                "snapshot_time_used": "2025-01-06T21:00:00Z",
                "snapshot_coverage_status": "ok",
                "snapshot_drop_reason": np.nan,
                "snapshot_pRN": 0.6,
                "snapshot_abs_log_m_fwd": 0.01,
                "yes_points": 8,
                "stale_ratio": 0.5,
                "max_stale_hours": 14.0,
                "midprice_cluster_ratio": 0.2,
                "max_jump": 0.1,
                "hours_since_last_yes_trade": 2.0,
                "gamma_volume": 12000.0,
                "quality_issue_count": 1,
                "quality_bucket": "watch",
            }
        ]
    ).to_csv(run_dir / "market_quality.csv", index=False)

    dim_market_path = run_dir / "dim_market_weekly.csv"
    pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "condition_id": "cond-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date_utc": "2025-01-10T00:00:00Z",
                "resolution_time_utc": "2025-01-10T21:00:00Z",
            }
        ]
    ).to_csv(dim_market_path, index=False)

    hourly_base = pd.DataFrame(
        [
            {
                "timestamp_utc": "2025-01-06T20:00:00Z",
                "market_id": "mkt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date": date(2025, 1, 10),
            },
            {
                "timestamp_utc": "2025-01-06T21:00:00Z",
                "market_id": "mkt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "expiry_date": date(2025, 1, 10),
            },
        ]
    )
    prn_loaded = pd.read_csv(training_path)
    prn_loaded["asof_time"] = pd.to_datetime(prn_loaded["asof_time"], utc=True)
    prn_loaded["expiry_date"] = pd.to_datetime(prn_loaded["expiry_date"]).dt.date
    merged_hourly = markets_refresh_v1.merge_prn_on_hourly_base(hourly_base, prn_loaded)
    merged_hourly.to_csv(run_dir / "markets_prn_hourly.csv", index=False)

    def _fake_to_parquet(self, path, index=False):  # noqa: ANN001
        Path(path).write_bytes(b"PAR1")
        self.to_csv(Path(path).with_suffix(".csv"), index=index)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "polymarket-build-features.py",
            "--dim-market",
            str(dim_market_path),
            "--bars-dir",
            str(bars_dir),
            "--out-dir",
            str(run_dir),
            "--decision-freq",
            "1d",
            "--start-date",
            "2025-01-06",
            "--end-date",
            "2025-01-06",
            "--skip-subgraph-labels",
        ],
    )

    build_features_v1.main()

    assert training_path.exists()
    assert (run_dir / "markets_prn_hourly.csv").exists()
    assert (run_dir / "decision_features.parquet").exists()
    assert (run_dir / "feature_manifest.json").exists()

    manifest = json.loads((run_dir / "feature_manifest.json").read_text())
    assert manifest["prn_source"] == "run_local_theta_exact"
    assert manifest["prn_dataset_path"] == str(training_path)
    assert "quality_columns" in manifest
    assert "quality_flag_columns" in manifest
    assert "flag_pm_stale_prices" in manifest["quality_flag_columns"]
    assert "flag_pm_stale_prices" not in manifest["features"]

    feature_rows = pd.read_csv(run_dir / "decision_features.csv")
    assert "quality_issue_count" in feature_rows.columns
    assert "flag_pm_stale_prices" in feature_rows.columns


def test_market_quality_builder_marks_missing_prn_and_composite_flags(tmp_path: Path) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp_utc": "2025-01-06T15:00:00Z",
                "price": 0.51,
                "market_id": "mkt-1",
                "token_role": "yes",
            }
        ]
    ).to_csv(run_dir / "price_history.csv", index=False)
    weekly = pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "event_id": "evt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "week_monday": "2025-01-06",
                "week_friday": "2025-01-10",
                "event_endDate": "2025-01-10",
                "active": True,
                "closed": False,
                "yes_token_id": "yes-1",
                "no_token_id": "no-1",
            }
        ]
    )
    prn_rows = pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "ticker": "AAPL",
                "week_friday": "2025-01-10",
                "snapshot_date": "2025-01-06",
                "asof_time": "2025-01-06T21:00:00Z",
                "coverage_status": "missing",
                "drop_reason": "target_outside_curve_support",
                "flag_asof_close_fallback": True,
                "flag_expiry_close_fallback": False,
                "flag_expiry_saturday_fallback": False,
                "flag_quote_close_fallback": False,
                "flag_low_chain_used": False,
                "flag_wide_rel_spread": False,
            }
        ]
    )

    result = polymarket_quality_flags.build_market_quality(
        run_dir,
        weekly,
        prn_rows,
        tz_name="America/New_York",
        close_time="16:00",
        emit_live=False,
    )

    assert len(result.rows) == 1
    row = result.rows.iloc[0]
    assert bool(row["flag_prn_missing"]) is True
    assert bool(row["flag_prn_outside_curve_support"]) is True
    assert bool(row["flag_not_relevant"]) is True
    assert row["quality_issue_count"] == 2
    assert row["quality_bucket"] == "watch"


def test_polymarket_quality_service_builds_audit_from_run_artifact(tmp_path: Path) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "market_id": "mkt-1",
                "event_id": "evt-1",
                "ticker": "AAPL",
                "threshold": 150.0,
                "week_monday": "2025-01-06",
                "week_friday": "2025-01-10",
                "flag_prn_missing": False,
                "flag_asof_close_fallback": False,
                "flag_expiry_close_fallback": False,
                "flag_expiry_saturday_fallback": False,
                "flag_quote_close_fallback": False,
                "flag_low_chain_used": False,
                "flag_wide_rel_spread": False,
                "flag_pm_no_trade_history": False,
                "flag_pm_no_recent_trade": False,
                "flag_pm_stale_prices": True,
                "flag_pm_suspect_orderbook": False,
                "flag_pm_sparse_points": False,
                "flag_pm_low_volume": False,
                "flag_market_inactive": False,
                "flag_missing_token_ids": False,
                "flag_extreme_otm": False,
                "flag_not_relevant": False,
                "flag_prn_outside_curve_support": False,
                "snapshot_date_used": "2025-01-06",
                "snapshot_time_used": "2025-01-06T21:00:00Z",
                "snapshot_coverage_status": "ok",
                "snapshot_drop_reason": np.nan,
                "snapshot_pRN": 0.6,
                "snapshot_abs_log_m_fwd": 0.01,
                "yes_points": 10,
                "stale_ratio": 0.5,
                "max_stale_hours": 13.0,
                "midprice_cluster_ratio": 0.3,
                "max_jump": 0.1,
                "hours_since_last_yes_trade": 1.0,
                "gamma_volume": 10000.0,
                "quality_issue_count": 1,
                "quality_bucket": "watch",
            }
        ]
    ).to_csv(run_dir / "market_quality.csv", index=False)
    (run_dir / "market_quality_summary.json").write_text(
        json.dumps(
            {
                "market_count": 1,
                "flagged_market_count": 1,
                "flagged_share": 1.0,
                "bucket_counts": {"clean": 0, "watch": 1, "noisy": 0},
                "prn_coverage_counts": {"ok": 1},
                "top_flags": [{"name": "flag_pm_stale_prices", "count": 1, "share": 1.0}],
                "top_problem_tickers": [
                    {
                        "ticker": "AAPL",
                        "market_count": 1,
                        "flagged_market_count": 1,
                        "flagged_share": 1.0,
                        "avg_issue_count": 1.0,
                        "clean_share": 0.0,
                        "watch_share": 1.0,
                        "noisy_share": 0.0,
                    }
                ],
                "snapshot_anchor": "latest_safe_snapshot",
                "quality_columns": ["quality_issue_count"],
                "quality_flag_columns": ["flag_pm_stale_prices"],
            }
        )
    )

    audit = polymarket_quality.build_quality_audit_response(run_dir)
    summary = polymarket_quality.load_market_quality_summary(run_dir)
    quality_map = polymarket_quality.load_market_quality_map(run_dir)

    assert audit.available is True
    assert audit.summary.flagged_market_count == 1
    assert audit.problem_markets[0].market_id == "mkt-1"
    assert summary is not None
    assert summary.flagged_share == 1.0
    assert quality_map["mkt-1"].flag_pm_stale_prices is True


def test_polymarket_quality_service_marks_legacy_run_without_artifact_unavailable(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    audit = polymarket_quality.build_quality_audit_response(run_dir)

    assert audit.available is False
    assert "legacy run" in (audit.message or "").lower()


def test_run_prn_refresh_rebuilds_markets_without_internal_feature_append(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    training_path = run_dir / "prn_dataset" / "training-poly-run-prn.csv"
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text("market_id\nmkt-1\n")

    refresh_result = SimpleNamespace(
        training_path=training_path,
        seeded_from_source=False,
        used_inferred_defaults=False,
        required_pairs={("AAPL", date(2025, 1, 10))},
        missing_pairs_before=set(),
        missing_pairs_after=set(),
        missing_markets_pairs_before={("AAPL", date(2025, 1, 10))},
        affected_week_fridays=[date(2025, 1, 10)],
    )
    captured_cmds: list[list[str]] = []

    monkeypatch.setattr(run_prn_refresh_v1, "parse_args", lambda: SimpleNamespace(
        run_id="poly-run",
        prn_dataset=None,
        week_fridays=None,
        no_refresh_markets=False,
    ))
    monkeypatch.setattr(run_prn_refresh_v1, "resolve_polymarket_run_dir", lambda _run_id: run_dir)
    monkeypatch.setattr(run_prn_refresh_v1, "refresh_run_local_prn_dataset", lambda *args, **kwargs: refresh_result)

    def _fake_subprocess_run(cmd, capture_output, text, check):
        captured_cmds.append(list(cmd))
        return SimpleNamespace(returncode=0, stdout="[Markets] ok\n", stderr="")

    monkeypatch.setattr(run_prn_refresh_v1.subprocess, "run", _fake_subprocess_run)

    run_prn_refresh_v1.main()

    assert len(captured_cmds) == 1
    assert "--replace-week" in captured_cmds[0]
    assert "--skip-build-features-append" in captured_cmds[0]


def test_run_prn_refresh_failure_includes_nested_output_tail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    training_path = run_dir / "prn_dataset" / "training-poly-run-prn.csv"
    training_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.write_text("market_id\nmkt-1\n")

    refresh_result = SimpleNamespace(
        training_path=training_path,
        seeded_from_source=False,
        used_inferred_defaults=False,
        required_pairs={("AAPL", date(2025, 1, 10))},
        missing_pairs_before=set(),
        missing_pairs_after=set(),
        missing_markets_pairs_before={("AAPL", date(2025, 1, 10))},
        affected_week_fridays=[date(2025, 1, 10)],
    )

    monkeypatch.setattr(run_prn_refresh_v1, "parse_args", lambda: SimpleNamespace(
        run_id="poly-run",
        prn_dataset=None,
        week_fridays=None,
        no_refresh_markets=False,
    ))
    monkeypatch.setattr(run_prn_refresh_v1, "resolve_polymarket_run_dir", lambda _run_id: run_dir)
    monkeypatch.setattr(run_prn_refresh_v1, "refresh_run_local_prn_dataset", lambda *args, **kwargs: refresh_result)
    monkeypatch.setattr(
        run_prn_refresh_v1.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=2,
            stdout="[Markets] stage one\n[Markets] stage two\n",
            stderr="[FATAL] bad refresh\n",
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_prn_refresh_v1.main()

    message = str(exc_info.value)
    assert "2025-01-10" in message
    assert "[Markets] stage two" in message
    assert "[FATAL] bad refresh" in message


def test_markets_refresh_skip_flag_avoids_internal_feature_append(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = markets_refresh_v1.MarketsConfig()
    called = {"value": False}

    def _fake_subprocess_run(*args, **kwargs):
        called["value"] = True
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(markets_refresh_v1.subprocess, "run", _fake_subprocess_run)

    appended = markets_refresh_v1._maybe_append_decision_features(
        run_dir=run_dir,
        cfg=cfg,
        last_snapshot_date="2025-01-10",
        skip_build_features_append=True,
        dry_run=False,
    )

    assert appended is False
    assert called["value"] is False


def test_markets_refresh_parse_args_accepts_replace_and_skip_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["polymarket-markets-refresh.py", "--replace-week", "--skip-build-features-append"],
    )

    args = markets_refresh_v1.parse_args()

    assert args.replace_week is True
    assert args.skip_build_features_append is True


def test_markets_refresh_runs_internal_feature_append_when_not_skipped(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "poly-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = markets_refresh_v1.MarketsConfig()
    called = {"value": False}

    def _fake_progress(stage, current, total):
        return None

    def _fake_subprocess_run(cmd, capture_output, text):
        called["value"] = True
        assert "--append" in cmd
        assert str(run_dir / "dim_market_weekly.csv") in cmd
        return SimpleNamespace(returncode=0, stdout="[features] ok\n", stderr="")

    monkeypatch.setattr(markets_refresh_v1, "progress", _fake_progress)
    monkeypatch.setattr(markets_refresh_v1.subprocess, "run", _fake_subprocess_run)

    appended = markets_refresh_v1._maybe_append_decision_features(
        run_dir=run_dir,
        cfg=cfg,
        last_snapshot_date="2025-01-10",
        skip_build_features_append=False,
        dry_run=False,
    )

    assert appended is True
    assert called["value"] is True
