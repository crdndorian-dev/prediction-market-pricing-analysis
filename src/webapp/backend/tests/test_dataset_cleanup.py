from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "src" / "webapp" / "backend"
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.models.datasets import DatasetCleanupCriteria, DatasetCleanupRequest
from app.services import datasets
from dataset_building.option_chain.clean_training_dataset_v1_0 import (
    CleanupCriteria,
    apply_cleanup,
    build_cleanup_preview,
)
from option_chain.quality_flags import compute_quality_issue_count, is_split_context_date


def _make_training_df() -> pd.DataFrame:
    rows = [
        {
            "row_id": "r1",
            "ticker": "AAPL",
            "asof_date": "2025-07-07",
            "snapshot_date": "2025-07-07",
            "snapshot_dow": "MON",
            "expiry_date": "2025-07-11",
            "week_id": "2025-07-11",
            "K": 190.0,
            "pRN": 0.41,
            "quality_issue_count": 0,
            "quality_bucket": "clean",
            "rel_spread_median": 0.02,
            "n_chain_used": 90,
            "flag_missing_rv": False,
            "flag_band_edge": False,
            "flag_prn_monotone_adjusted": False,
            "weight_final": 999.0,
            "weighting_version": "stale",
        },
        {
            "row_id": "r2",
            "ticker": "AAPL",
            "asof_date": "2025-07-08",
            "snapshot_date": "2025-07-08",
            "snapshot_dow": "TUE",
            "expiry_date": "2025-07-11",
            "week_id": "2025-07-11",
            "K": 195.0,
            "pRN": 0.44,
            "quality_issue_count": 1,
            "quality_bucket": "watch",
            "rel_spread_median": 0.05,
            "n_chain_used": 85,
            "flag_missing_rv": True,
            "flag_band_edge": False,
            "flag_prn_monotone_adjusted": False,
            "weight_final": 999.0,
            "weighting_version": "stale",
        },
        {
            "row_id": "r3",
            "ticker": "MSFT",
            "asof_date": "2025-07-08",
            "snapshot_date": "2025-07-08",
            "snapshot_dow": "TUE",
            "expiry_date": "2025-07-11",
            "week_id": "2025-07-11",
            "K": 495.0,
            "pRN": 0.37,
            "quality_issue_count": 3,
            "quality_bucket": "noisy",
            "rel_spread_median": 0.18,
            "n_chain_used": 40,
            "flag_missing_rv": True,
            "flag_band_edge": True,
            "flag_prn_monotone_adjusted": False,
            "weight_final": 999.0,
            "weighting_version": "stale",
        },
        {
            "row_id": "r4",
            "ticker": "MSFT",
            "asof_date": "2025-07-09",
            "snapshot_date": "2025-07-09",
            "snapshot_dow": "WED",
            "expiry_date": "2025-07-11",
            "week_id": "2025-07-11",
            "K": 500.0,
            "pRN": 0.35,
            "quality_issue_count": 4,
            "quality_bucket": "noisy",
            "rel_spread_median": 0.12,
            "n_chain_used": 35,
            "flag_missing_rv": False,
            "flag_band_edge": False,
            "flag_prn_monotone_adjusted": True,
            "weight_final": 999.0,
            "weighting_version": "stale",
        },
        {
            "row_id": "r5",
            "ticker": "NVDA",
            "asof_date": "2025-07-09",
            "snapshot_date": "2025-07-09",
            "snapshot_dow": "WED",
            "expiry_date": "2025-07-11",
            "week_id": "2025-07-11",
            "K": 148.0,
            "pRN": 0.51,
            "quality_issue_count": 2,
            "quality_bucket": "watch",
            "rel_spread_median": 0.25,
            "n_chain_used": 45,
            "flag_missing_rv": False,
            "flag_band_edge": True,
            "flag_prn_monotone_adjusted": False,
            "weight_final": 999.0,
            "weighting_version": "stale",
        },
        {
            "row_id": "r6",
            "ticker": "NVDA",
            "asof_date": "2025-07-10",
            "snapshot_date": "2025-07-10",
            "snapshot_dow": "THU",
            "expiry_date": "2025-07-11",
            "week_id": "2025-07-11",
            "K": 150.0,
            "pRN": 0.49,
            "quality_issue_count": 0,
            "quality_bucket": "clean",
            "rel_spread_median": 0.03,
            "n_chain_used": 100,
            "flag_missing_rv": False,
            "flag_band_edge": False,
            "flag_prn_monotone_adjusted": False,
            "weight_final": 999.0,
            "weighting_version": "stale",
        },
    ]
    return pd.DataFrame(rows)


def _write_run(tmp_path: Path, *, with_meta: bool = False) -> tuple[Path, Path]:
    run_dir = tmp_path / "dataset-run"
    run_dir.mkdir()
    training_path = run_dir / "training-dataset-run.csv"
    training_df = _make_training_df()
    training_df.to_csv(training_path, index=False)
    training_df.to_csv(run_dir / "legacy-dataset-run.csv", index=False)
    training_df.loc[:, ["row_id", "ticker", "asof_date", "expiry_date", "K"]].assign(
        snapshot_metric=[10, 20, 30, 40, 50, 60]
    ).to_csv(run_dir / "snapshot-dataset-run.csv", index=False)
    training_df.loc[:, ["row_id", "ticker", "pRN"]].assign(
        curve_quality=["a", "a", "b", "b", "c", "c"]
    ).to_csv(run_dir / "prn-view-dataset-run.csv", index=False)
    pd.DataFrame(
        [
            {"row_id": "r1", "note": "keep"},
            {"row_id": "r3", "note": "drop"},
            {"row_id": "r6", "note": "keep"},
        ]
    ).to_csv(run_dir / "custom-row-id.csv", index=False)
    pd.DataFrame(
        [
            {"reason": "quality", "count": 2},
            {"reason": "coverage", "count": 1},
        ]
    ).to_csv(run_dir / "drops-dataset-run.csv", index=False)
    (run_dir / "custom-summary.csv").write_text("col\nkept\n")
    (run_dir / "training-dataset-run-cleaned-4.csv").write_text("stale\n1\n")
    (run_dir / "custom-cleaned.csv").write_text("stale\n1\n")
    if with_meta:
        (run_dir / "training_selection.json").write_text(
            json.dumps(
                {
                    "dataset_name": "dataset-run",
                    "training_file": "training-dataset-run.csv",
                    "train_view_file": "training-dataset-run.csv",
                    "legacy_file": "legacy-dataset-run.csv",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                },
                indent=2,
            )
        )
        (run_dir / "dataset_build_meta.json").write_text(
            json.dumps(
                {
                    "created_at_utc": "2026-01-01T00:00:00+00:00",
                    "command": ["python", "build_historic_dataset_v1_0.py"],
                    "out_name": "legacy-dataset-run.csv",
                    "drops_name": "drops-dataset-run.csv",
                    "payload": {
                        "dataset_name": "dataset-run",
                        "run_dir_name": "dataset-run",
                        "out_name": "legacy-dataset-run.csv",
                        "train_view_name": "training-dataset-run.csv",
                        "drops_name": "drops-dataset-run.csv",
                    },
                },
                indent=2,
            )
        )
    return run_dir, training_path


def _default_weighting_params() -> dict[str, object]:
    return {
        "ticker_reweight_mode": "none",
        "ticker_reweight_alpha_min": 0.5,
        "ticker_reweight_alpha_max": 2.0,
        "trade_focus_beta": 1.0,
        "trade_focus_tickers": None,
    }


def test_cleanup_preview_default_drops_only_noisy_rows(tmp_path: Path) -> None:
    _, training_path = _write_run(tmp_path)

    preview = build_cleanup_preview(training_path, CleanupCriteria())

    assert preview.rows_before == 6
    assert preview.rows_to_drop == 2
    assert preview.rows_after == 4
    assert preview.would_drop_all is False
    assert preview.dropped_bucket_counts == {"clean": 0, "watch": 0, "noisy": 2}
    assert {row.row_id for row in preview.sample_rows} == {"r3", "r4"}


def test_cleanup_preview_flag_match_mode_any_vs_all(tmp_path: Path) -> None:
    _, training_path = _write_run(tmp_path)

    preview_any = build_cleanup_preview(
        training_path,
        CleanupCriteria(
            quality_buckets=[],
            min_quality_issue_count=None,
            flag_columns=["flag_missing_rv", "flag_band_edge"],
            flag_match_mode="any",
        ),
    )
    preview_all = build_cleanup_preview(
        training_path,
        CleanupCriteria(
            quality_buckets=[],
            min_quality_issue_count=None,
            flag_columns=["flag_missing_rv", "flag_band_edge"],
            flag_match_mode="all",
        ),
    )

    assert preview_any.rows_to_drop == 3
    assert preview_all.rows_to_drop == 1
    assert preview_all.sample_rows[0].row_id == "r3"


def test_cleanup_preview_numeric_threshold_groups_are_anded(tmp_path: Path) -> None:
    _, training_path = _write_run(tmp_path)

    preview = build_cleanup_preview(
        training_path,
        CleanupCriteria(
            quality_buckets=[],
            min_quality_issue_count=None,
            min_rel_spread_median=0.12,
            max_n_chain_used=40,
        ),
    )

    assert preview.rows_to_drop == 2
    assert {row.row_id for row in preview.sample_rows} == {"r3", "r4"}


def test_cleanup_apply_rejects_zero_match_and_drop_all(tmp_path: Path) -> None:
    run_dir, training_path = _write_run(tmp_path)
    weighting_params = _default_weighting_params()

    preview = build_cleanup_preview(
        training_path,
        CleanupCriteria(
            quality_buckets=[],
            min_quality_issue_count=None,
            min_rel_spread_median=2.0,
        ),
    )
    assert preview.rows_to_drop == 0
    with pytest.raises(ValueError, match="No rows match"):
        apply_cleanup(
            run_dir,
            training_path,
            CleanupCriteria(
                quality_buckets=[],
                min_quality_issue_count=None,
                min_rel_spread_median=2.0,
            ),
            weighting_params,
        )

    drop_all_preview = build_cleanup_preview(
        training_path,
        CleanupCriteria(
            quality_buckets=[],
            min_quality_issue_count=None,
        ),
    )
    assert drop_all_preview.would_drop_all is True
    with pytest.raises(ValueError, match="would drop all rows"):
        apply_cleanup(
            run_dir,
            training_path,
            CleanupCriteria(
                quality_buckets=[],
                min_quality_issue_count=None,
            ),
            weighting_params,
        )


def test_cleanup_apply_clones_run_and_filters_all_row_id_csvs(
    tmp_path: Path,
) -> None:
    run_dir, training_path = _write_run(tmp_path, with_meta=True)
    weighting_params = _default_weighting_params()
    original_bytes = training_path.read_bytes()
    original_summary = (run_dir / "custom-summary.csv").read_text()

    result = apply_cleanup(
        run_dir,
        training_path,
        CleanupCriteria(),
        weighting_params,
    )

    cleaned_run_dir = tmp_path / "dataset-run-cleaned"
    cleaned_path = cleaned_run_dir / "training-dataset-run-cleaned.csv"
    assert result.cleaned_run_dir == cleaned_run_dir
    assert result.cleaned_path == cleaned_path
    assert cleaned_run_dir.exists()
    assert cleaned_path.exists()
    assert training_path.read_bytes() == original_bytes
    assert not (run_dir / ".cleanup-history").exists()

    cleaned = pd.read_csv(cleaned_path)
    assert cleaned.shape[0] == 4
    assert set(cleaned["row_id"]) == {"r1", "r2", "r5", "r6"}
    assert set(cleaned["weighting_version"]) == {"v3"}
    assert not cleaned["weight_final"].eq(999.0).all()
    assert set(pd.read_csv(cleaned_run_dir / "legacy-dataset-run-cleaned.csv")["row_id"]) == {
        "r1",
        "r2",
        "r5",
        "r6",
    }
    assert set(pd.read_csv(cleaned_run_dir / "snapshot-dataset-run-cleaned.csv")["row_id"]) == {
        "r1",
        "r2",
        "r5",
        "r6",
    }
    assert set(pd.read_csv(cleaned_run_dir / "prn-view-dataset-run-cleaned.csv")["row_id"]) == {
        "r1",
        "r2",
        "r5",
        "r6",
    }
    assert set(pd.read_csv(cleaned_run_dir / "custom-row-id.csv")["row_id"]) == {"r1", "r6"}
    assert (cleaned_run_dir / "custom-summary.csv").read_text() == original_summary
    assert (cleaned_run_dir / "drops-dataset-run-cleaned.csv").exists()
    assert not (cleaned_run_dir / "training-dataset-run-cleaned-4.csv").exists()
    assert not (cleaned_run_dir / "custom-cleaned.csv").exists()

    second_result = apply_cleanup(
        run_dir,
        training_path,
        CleanupCriteria(),
        weighting_params,
    )
    assert second_result.cleaned_run_dir == tmp_path / "dataset-run-cleaned-1"
    assert second_result.cleaned_path == (
        tmp_path / "dataset-run-cleaned-1" / "training-dataset-run-cleaned-1.csv"
    )


def test_cleanup_apply_rewrites_metadata_for_cleaned_clone(tmp_path: Path) -> None:
    run_dir, training_path = _write_run(tmp_path, with_meta=True)
    weighting_params = _default_weighting_params()

    result = apply_cleanup(
        run_dir,
        training_path,
        CleanupCriteria(),
        weighting_params,
    )

    cleaned_run_dir = result.cleaned_run_dir
    training_meta = json.loads((cleaned_run_dir / "training_selection.json").read_text())
    assert training_meta["dataset_name"] == "dataset-run-cleaned"
    assert training_meta["training_file"] == "training-dataset-run-cleaned.csv"
    assert training_meta["train_view_file"] == "training-dataset-run-cleaned.csv"
    assert training_meta["legacy_file"] == "legacy-dataset-run-cleaned.csv"

    build_meta = json.loads((cleaned_run_dir / "dataset_build_meta.json").read_text())
    assert build_meta["out_name"] == "legacy-dataset-run-cleaned.csv"
    assert build_meta["drops_name"] == "drops-dataset-run-cleaned.csv"
    assert build_meta["payload"]["dataset_name"] == "dataset-run-cleaned"
    assert build_meta["payload"]["run_dir_name"] == "dataset-run-cleaned"
    assert build_meta["payload"]["out_name"] == "legacy-dataset-run-cleaned.csv"
    assert build_meta["payload"]["train_view_name"] == "training-dataset-run-cleaned.csv"
    assert build_meta["payload"]["drops_name"] == "drops-dataset-run-cleaned.csv"


def test_service_cleanup_requires_metadata_unless_defaults_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, training_path = _write_run(tmp_path)
    original_bytes = training_path.read_bytes()

    monkeypatch.setattr(datasets, "_resolve_dataset_directory", lambda _: run_dir)
    monkeypatch.setattr(datasets, "_resolve_project_path", lambda value: Path(value))
    monkeypatch.setattr(datasets, "DATASET_BASE_DIRS", [tmp_path])

    with pytest.raises(ValueError, match="build metadata is missing"):
        datasets.preview_dataset_cleanup(
            DatasetCleanupRequest(run_dir=str(run_dir), criteria=DatasetCleanupCriteria())
        )

    preview = datasets.preview_dataset_cleanup(
        DatasetCleanupRequest(
            run_dir=str(run_dir),
            criteria=DatasetCleanupCriteria(),
            allow_defaults=True,
        )
    )
    assert preview.used_defaults is True
    assert preview.rows_to_drop == 2

    response = datasets.apply_dataset_cleanup(
        DatasetCleanupRequest(
            run_dir=str(run_dir),
            criteria=DatasetCleanupCriteria(),
            allow_defaults=True,
        )
    )

    assert response.ok is True
    assert response.used_defaults is True
    assert response.training_file == str(training_path)
    assert response.cleaned_run_dir == str(tmp_path / "dataset-run-cleaned")
    assert response.cleaned_file == str(
        tmp_path / "dataset-run-cleaned" / "training-dataset-run-cleaned.csv"
    )
    assert response.rows_before == 6
    assert response.rows_after == 4
    assert Path(response.cleaned_run_dir).exists()
    assert Path(response.cleaned_file).exists()
    assert set(
        pd.read_csv(Path(response.cleaned_run_dir) / "snapshot-dataset-run-cleaned.csv")[
            "row_id"
        ]
    ) == {"r1", "r2", "r5", "r6"}
    assert training_path.read_bytes() == original_bytes
    assert datasets._find_training_file_path(run_dir) == training_path
    assert datasets._find_training_file_path(Path(response.cleaned_run_dir)) == Path(
        response.cleaned_file
    )
    assert not (run_dir / ".cleanup-history").exists()

    source = pd.read_csv(training_path)
    assert source.shape[0] == 6
    assert set(source["row_id"]) == {"r1", "r2", "r3", "r4", "r5", "r6"}

    cleaned = pd.read_csv(Path(response.cleaned_file))
    assert cleaned.shape[0] == 4
    assert set(cleaned["row_id"]) == {"r1", "r2", "r5", "r6"}
    assert set(cleaned["weighting_version"]) == {"v3"}
    assert not cleaned["weight_final"].eq(999.0).all()

    source_audit = datasets.audit_dataset_file(str(training_path))
    assert source_audit.row_count == 6

    cleaned_audit = datasets.audit_dataset_file(response.cleaned_file)
    assert cleaned_audit.row_count == 4
    assert all(row.row_id not in {"r3", "r4"} for row in cleaned_audit.noisiest_rows)


def test_service_cleanup_accepts_bucket_mode_without_flag_match_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _ = _write_run(tmp_path)

    monkeypatch.setattr(datasets, "_resolve_dataset_directory", lambda _: run_dir)
    monkeypatch.setattr(datasets, "_resolve_project_path", lambda value: Path(value))
    monkeypatch.setattr(datasets, "DATASET_BASE_DIRS", [tmp_path])

    preview = datasets.preview_dataset_cleanup(
        DatasetCleanupRequest(
            run_dir=str(run_dir),
            criteria=DatasetCleanupCriteria(
                quality_buckets=["watch"],
                min_quality_issue_count=None,
                flag_match_mode=None,
            ),
            allow_defaults=True,
        )
    )

    assert preview.used_defaults is True
    assert preview.rows_before == 6
    assert preview.rows_to_drop == 2
    assert preview.rows_after == 4
    assert preview.dropped_bucket_counts == {"clean": 0, "watch": 2, "noisy": 0}
    assert {row.row_id for row in preview.sample_rows} == {"r2", "r5"}


def test_quality_helpers_treat_split_context_as_informational() -> None:
    flags = {
        "flag_prn_monotone_adjusted": True,
        "flag_split_event_context": True,
        "flag_band_edge": True,
    }

    assert compute_quality_issue_count(flags) == 1
    assert is_split_context_date(pd.Timestamp("2025-07-08").date(), {pd.Timestamp("2025-07-08").date()})
    assert not is_split_context_date(
        pd.Timestamp("2025-07-07").date(),
        {pd.Timestamp("2025-07-08").date()},
    )


def test_cleanup_and_audit_fallback_ignore_informational_flags_without_saved_quality_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_path = tmp_path / "training-fallback.csv"
    pd.DataFrame(
        [
            {
                "row_id": "a",
                "ticker": "AAPL",
                "asof_date": "2025-07-07",
                "expiry_date": "2025-07-11",
                "K": 190.0,
                "pRN": 0.41,
                "rel_spread_median": 0.02,
                "n_chain_used": 90,
                "flag_prn_monotone_adjusted": True,
                "flag_split_event_context": False,
                "flag_band_edge": False,
            },
            {
                "row_id": "b",
                "ticker": "AAPL",
                "asof_date": "2025-07-08",
                "expiry_date": "2025-07-11",
                "K": 195.0,
                "pRN": 0.44,
                "rel_spread_median": 0.02,
                "n_chain_used": 90,
                "flag_prn_monotone_adjusted": False,
                "flag_split_event_context": True,
                "flag_band_edge": False,
            },
            {
                "row_id": "c",
                "ticker": "AAPL",
                "asof_date": "2025-07-09",
                "expiry_date": "2025-07-11",
                "K": 200.0,
                "pRN": 0.47,
                "rel_spread_median": 0.02,
                "n_chain_used": 90,
                "flag_prn_monotone_adjusted": False,
                "flag_split_event_context": False,
                "flag_band_edge": True,
            },
        ]
    ).to_csv(training_path, index=False)

    preview = build_cleanup_preview(
        training_path,
        CleanupCriteria(
            quality_buckets=["clean"],
            min_quality_issue_count=None,
        ),
    )

    assert preview.rows_to_drop == 2
    assert {row.row_id for row in preview.sample_rows} == {"a", "b"}
    assert all(row.quality_issue_count == 0 for row in preview.sample_rows)

    monkeypatch.setattr(datasets, "_resolve_project_path", lambda value: Path(value))
    monkeypatch.setattr(datasets, "DATASET_BASE_DIRS", [tmp_path])
    audit = datasets.audit_dataset_file(str(training_path))
    aapl = next(item for item in audit.top_problem_tickers if item.ticker == "AAPL")

    assert aapl.avg_issue_count == pytest.approx(1 / 3, rel=0, abs=1e-4)
    assert aapl.flagged_share == pytest.approx(1 / 3, rel=0, abs=1e-6)
    assert aapl.clean_share == pytest.approx(2 / 3, rel=0, abs=1e-6)
    assert aapl.watch_share == pytest.approx(1 / 3, rel=0, abs=1e-6)
    assert aapl.noisy_share == pytest.approx(0.0, rel=0, abs=1e-6)
    flag_counts = {item.name: item.count for item in audit.quality_flags}
    assert flag_counts["flag_prn_monotone_adjusted"] == 1
    assert flag_counts["flag_split_event_context"] == 1


def test_audit_dataset_derives_only_ascending_rv_ratios_for_legacy_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_path = tmp_path / "training-legacy-rv.csv"
    pd.DataFrame(
        [
            {
                "row_id": "a",
                "ticker": "AAPL",
                "asof_date": "2025-07-07",
                "expiry_date": "2025-07-11",
                "K": 190.0,
                "pRN": 0.41,
                "rv5": 0.20,
                "rv10": 0.25,
                "rv20": 0.40,
                "rv20_over_rv10": 99.0,
                "rv20_over_rv5": 99.0,
                "rv10_over_rv5": 99.0,
                "quality_issue_count": 0,
                "rel_spread_median": 0.02,
                "n_chain_used": 90,
            },
            {
                "row_id": "b",
                "ticker": "AAPL",
                "asof_date": "2025-07-08",
                "expiry_date": "2025-07-11",
                "K": 195.0,
                "pRN": 0.44,
                "rv5": 0.30,
                "rv10": 0.45,
                "rv20": 0.60,
                "rv20_over_rv10": 88.0,
                "rv20_over_rv5": 88.0,
                "rv10_over_rv5": 88.0,
                "quality_issue_count": 1,
                "rel_spread_median": 0.05,
                "n_chain_used": 85,
            },
        ]
    ).to_csv(training_path, index=False)

    monkeypatch.setattr(datasets, "_resolve_project_path", lambda value: Path(value))
    monkeypatch.setattr(datasets, "DATASET_BASE_DIRS", [tmp_path])
    audit = datasets.audit_dataset_file(str(training_path))
    feature_set = set(audit.available_rv_features)
    numeric_names = {metric.name for metric in audit.numeric_distributions}
    rv_feature_audit = {item.feature: item for item in audit.rv_feature_audit}

    assert {"rv5", "rv10", "rv20", "rv5_over_rv10", "rv5_over_rv20", "rv10_over_rv20"} <= feature_set
    assert "rv20_over_rv10" not in feature_set
    assert "rv20_over_rv5" not in feature_set
    assert "rv10_over_rv5" not in feature_set

    assert {"rv5_over_rv10", "rv5_over_rv20", "rv10_over_rv20"} <= numeric_names
    assert "rv20_over_rv10" not in numeric_names
    assert "rv20_over_rv5" not in numeric_names
    assert "rv10_over_rv5" not in numeric_names
    assert set(rv_feature_audit) == feature_set
    assert all(item.finite_row_count == 2 for item in rv_feature_audit.values())
    assert all(item.finite_row_share == pytest.approx(1.0, rel=0, abs=1e-6) for item in rv_feature_audit.values())
    assert all(item.buckets == [] for item in rv_feature_audit.values())


def test_audit_dataset_builds_rv_tercile_quality_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_path = tmp_path / "training-rv-audit.csv"
    pd.DataFrame(
        [
            {
                "row_id": "r1",
                "ticker": "AAPL",
                "asof_date": "2025-07-07",
                "expiry_date": "2025-07-11",
                "K": 190.0,
                "pRN": 0.41,
                "rv5": 0.10,
                "rv10": 0.20,
                "rv20": 0.30,
                "quality_issue_count": 0,
                "rel_spread_median": 0.02,
                "n_chain_used": 90,
            },
            {
                "row_id": "r2",
                "ticker": "AAPL",
                "asof_date": "2025-07-08",
                "expiry_date": "2025-07-11",
                "K": 191.0,
                "pRN": 0.42,
                "rv5": 0.20,
                "rv10": 0.30,
                "rv20": 0.40,
                "quality_issue_count": 0,
                "rel_spread_median": 0.03,
                "n_chain_used": 88,
            },
            {
                "row_id": "r3",
                "ticker": "AAPL",
                "asof_date": "2025-07-09",
                "expiry_date": "2025-07-11",
                "K": 192.0,
                "pRN": 0.43,
                "rv5": 0.30,
                "rv10": 0.40,
                "rv20": 0.50,
                "quality_issue_count": 1,
                "rel_spread_median": 0.04,
                "n_chain_used": 86,
            },
            {
                "row_id": "r4",
                "ticker": "AAPL",
                "asof_date": "2025-07-10",
                "expiry_date": "2025-07-11",
                "K": 193.0,
                "pRN": 0.44,
                "rv5": 0.40,
                "rv10": 0.50,
                "rv20": 0.60,
                "quality_issue_count": 1,
                "rel_spread_median": 0.05,
                "n_chain_used": 84,
            },
            {
                "row_id": "r5",
                "ticker": "AAPL",
                "asof_date": "2025-07-11",
                "expiry_date": "2025-07-18",
                "K": 194.0,
                "pRN": 0.45,
                "rv5": 0.50,
                "rv10": 0.60,
                "rv20": 0.70,
                "quality_issue_count": 3,
                "rel_spread_median": 0.08,
                "n_chain_used": 70,
            },
            {
                "row_id": "r6",
                "ticker": "AAPL",
                "asof_date": "2025-07-14",
                "expiry_date": "2025-07-18",
                "K": 195.0,
                "pRN": 0.46,
                "rv5": 0.60,
                "rv10": 0.70,
                "rv20": 0.80,
                "quality_issue_count": 4,
                "rel_spread_median": 0.09,
                "n_chain_used": 68,
            },
        ]
    ).to_csv(training_path, index=False)

    monkeypatch.setattr(datasets, "_resolve_project_path", lambda value: Path(value))
    monkeypatch.setattr(datasets, "DATASET_BASE_DIRS", [tmp_path])
    audit = datasets.audit_dataset_file(str(training_path))
    rv_feature_audit = {item.feature: item for item in audit.rv_feature_audit}

    assert audit.available_rv_features == [
        "rv5",
        "rv10",
        "rv20",
        "rv5_over_rv10",
        "rv5_over_rv20",
        "rv10_over_rv20",
    ]
    assert set(rv_feature_audit) == set(audit.available_rv_features)

    rv5_audit = rv_feature_audit["rv5"]
    assert rv5_audit.finite_row_count == 6
    assert rv5_audit.finite_row_share == pytest.approx(1.0, rel=0, abs=1e-6)
    assert [bucket.label for bucket in rv5_audit.buckets] == ["low", "mid", "high"]
    assert [bucket.row_count for bucket in rv5_audit.buckets] == [2, 2, 2]
    assert rv5_audit.buckets[0].value_min == pytest.approx(0.10, rel=0, abs=1e-6)
    assert rv5_audit.buckets[0].value_max == pytest.approx(0.20, rel=0, abs=1e-6)
    assert rv5_audit.buckets[0].avg_issue_count == pytest.approx(0.0, rel=0, abs=1e-6)
    assert rv5_audit.buckets[0].flagged_share == pytest.approx(0.0, rel=0, abs=1e-6)
    assert rv5_audit.buckets[1].avg_issue_count == pytest.approx(1.0, rel=0, abs=1e-6)
    assert rv5_audit.buckets[1].flagged_share == pytest.approx(1.0, rel=0, abs=1e-6)
    assert rv5_audit.buckets[2].avg_issue_count == pytest.approx(3.5, rel=0, abs=1e-6)
    assert rv5_audit.buckets[2].flagged_share == pytest.approx(1.0, rel=0, abs=1e-6)

    ratio_audit = rv_feature_audit["rv5_over_rv10"]
    assert [bucket.row_count for bucket in ratio_audit.buckets] == [2, 2, 2]
    assert ratio_audit.buckets[0].avg_issue_count == pytest.approx(0.0, rel=0, abs=1e-6)
    assert ratio_audit.buckets[1].avg_issue_count == pytest.approx(1.0, rel=0, abs=1e-6)
    assert ratio_audit.buckets[2].avg_issue_count == pytest.approx(3.5, rel=0, abs=1e-6)
