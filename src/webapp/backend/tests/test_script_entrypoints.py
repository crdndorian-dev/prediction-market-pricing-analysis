from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "src" / "webapp" / "backend"
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"
ENTRYPOINTS_ROOT = SCRIPTS_ROOT / "entrypoints"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from app.services.script_entrypoints import CANONICAL_SCRIPT_ENTRYPOINTS


@pytest.mark.parametrize("entrypoint", CANONICAL_SCRIPT_ENTRYPOINTS)
def test_canonical_script_entrypoints_exist(entrypoint) -> None:
    assert entrypoint.path.exists()
    assert entrypoint.path.is_file()


@pytest.mark.parametrize(
    "public_name",
    [entrypoint.public_name for entrypoint in CANONICAL_SCRIPT_ENTRYPOINTS],
)
def test_public_script_wrappers_support_help(public_name: str) -> None:
    script_path = ENTRYPOINTS_ROOT / public_name
    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "usage" in (result.stdout or result.stderr).lower()


@pytest.mark.parametrize(
    "module_name",
    [
        "app.services.datasets",
        "app.services.calibrate_models",
        "app.services.market_map",
        "app.services.markets",
        "app.services.polymarket_history",
        "app.services.polymarket_quality",
        "app.services.polymarket_run_prn",
    ],
)
def test_backend_services_import_after_reorg(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_option_chain_weighting_module_exports_public_api() -> None:
    module = importlib.import_module("feature_engineering.option_chain.weighting_v3")

    assert hasattr(module, "apply_weighting_v3")
    assert hasattr(module, "drop_weight_columns")
    assert getattr(module, "WEIGHTING_VERSION", None) == "v3"


def test_option_chain_calibration_defaults_follow_registry_contract() -> None:
    module = importlib.import_module("model_training.calibration.calibrate_v2_core")

    assert module.DEFAULT_PRN_FEATURES == [
        "x_logit_prn",
        "log_m_fwd",
        "rv20",
        "rel_spread_median",
        "dividend_yield",
    ]


def test_option_chain_calibration_parser_drops_enable_x_abs_m_flag() -> None:
    module = importlib.import_module("model_training.calibration.calibrate_v2_core")

    parser = module._build_calibration_arg_parser()
    help_text = parser.format_help()

    assert "--enable-x-abs-m" not in help_text


def test_option_chain_resolved_args_snapshot_has_static_diagnostics_contract() -> None:
    module = importlib.import_module("model_training.calibration.calibrate_v2_core")

    parser = module._build_calibration_arg_parser()
    args = parser.parse_args([])
    snapshot = module._resolved_args_snapshot(args)

    assert snapshot["diagnostics"] == {
        "split_timeline": False,
        "per_fold_delta_chart": False,
        "per_group_delta_distribution": False,
        "skip_test_metrics": False,
    }


def test_option_chain_run_calibration_from_cache_uses_cached_train_df_for_feature_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("model_training.calibration.calibrate_v2_core")

    parser = module._build_calibration_arg_parser()
    args = parser.parse_args(["--csv", "ignored.csv", "--out-dir", str(tmp_path / "out")])
    train_df = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "week_friday": ["2026-03-06"],
            "outcome_ST_gt_K": [1],
        }
    )
    empty_df = train_df.iloc[0:0].copy()
    cache = module.CalibrationCache(
        df_base=train_df,
        target_col="outcome_ST_gt_K",
        week_col="week_friday",
        ticker_col="ticker",
        train_idx=train_df.index,
        test_idx=empty_df.index,
        train_fit_idx=train_df.index,
        val_idx=empty_df.index,
        train_df=train_df,
        test_df=empty_df,
        train_fit_df=train_df,
        val_df=empty_df,
        train_pos=pd.Series([0], index=train_df.index),
        fit_pos=np.array([0]),
        val_pos=np.array([], dtype=int),
        split_group_series=None,
        split_group_key=None,
        split_group_dropped_train_rows=0,
        split_group_dropped_train_fit_rows=0,
        embargo_mode="disabled",
        embargo_date_col_used=None,
        embargo_rows_dropped_train=0,
        embargo_rows_dropped_train_fit=0,
        walk_forward_folds=[],
        val_split_info={},
        split_overlap={},
        split_ranges={},
        split_composition_rows=[],
        train_weights_raw=np.array([1.0]),
        weight_source="uniform",
        group_key=None,
        group_key_source=None,
        foundation_set=set(),
        active_filters={},
        trainer_warnings=[],
        requested_numeric_features=[],
        requested_categorical_features=[],
    )

    monkeypatch.setattr(
        module,
        "_resolve_requested_feature_lists",
        lambda args, available_columns, **kwargs: (["missing_feature"], []),
    )

    result = module.run_calibration_from_cache(
        cache,
        args,
        tmp_path / "out",
        n_bins=10,
        eceq_bins=10,
        fast_trial=False,
        skip_test_metrics=False,
    )

    assert result.exit_code == 1


def test_option_chain_auto_search_defaults_follow_registry_contract() -> None:
    module = importlib.import_module("orchestration.calibration.auto_calibrate_logit_model_v2")

    assert module.DEFAULT_FEATURE_SETS == [
        ["x_logit_prn"],
        ["x_logit_prn", "rv20"],
        ["x_logit_prn", "abs_log_m_fwd"],
        ["x_logit_prn", "rv20", "abs_log_m_fwd"],
        ["x_logit_prn", "rv20", "abs_log_m_fwd", "rel_spread_median"],
    ]


def test_option_chain_progress_parser_contract_remains_stable() -> None:
    module = importlib.import_module("app.services.datasets")

    line = (
        "[PROGRESS] 30/80 jobs | groups_kept=12 | rows=144 | "
        "last=NVDA week=2025-07-14 asof_target=2025-07-16"
    )
    match = module._PROGRESS_RE.search(line)

    assert match is not None
    assert match.group(1) == "30"
    assert match.group(2) == "80"
    assert match.group(3) == "12"
    assert match.group(4) == "144"
    assert match.group(5) == "NVDA"
    assert match.group(6) == "2025-07-14"
    assert match.group(7) == "2025-07-16"


def test_option_chain_progress_update_interval_defaults() -> None:
    module = importlib.import_module("dataset_building.option_chain.build_historic_dataset_v1_0")

    assert module.progress_update_interval(0) == 1
    assert module.progress_update_interval(10) == 1
    assert module.progress_update_interval(11) == 10
    assert module.progress_update_interval(80) == 10


def test_option_chain_live_telemetry_accumulator_tracks_group_checks_and_drop_reasons() -> None:
    module = importlib.import_module("dataset_building.option_chain.build_historic_dataset_v1_0")

    telemetry = module.build_live_telemetry(["AAPL", "NVDA"], planned_jobs_per_ticker=4)
    kept_rows = [
        {
            "flag_asof_close_fallback": True,
            "flag_expiry_close_fallback": False,
            "flag_expiry_saturday_fallback": True,
            "flag_quote_close_fallback": True,
            "flag_low_chain_used": True,
            "flag_wide_rel_spread": False,
            "quality_issue_count": 0,
            "quality_bucket": "clean",
        },
        {
            "flag_asof_close_fallback": False,
            "flag_expiry_close_fallback": False,
            "flag_expiry_saturday_fallback": False,
            "flag_quote_close_fallback": False,
            "flag_low_chain_used": False,
            "flag_wide_rel_spread": True,
            "quality_issue_count": 2,
            "quality_bucket": "watch",
        },
        {
            "flag_asof_close_fallback": False,
            "flag_expiry_close_fallback": True,
            "flag_expiry_saturday_fallback": False,
            "flag_quote_close_fallback": False,
            "flag_low_chain_used": False,
            "flag_wide_rel_spread": False,
            "quality_issue_count": 4,
            "quality_bucket": "noisy",
        },
    ]

    module.update_live_telemetry_for_job(telemetry, "AAPL", kept_rows, None)
    module.update_live_telemetry_for_job(
        telemetry,
        "AAPL",
        [],
        {"drop_reason": "empty_option_chain"},
    )
    module.update_live_telemetry_for_job(
        telemetry,
        "NVDA",
        [],
        {"drop_reason": "missing_ST"},
    )

    assert telemetry["group_checks"]["asof_close_fallback"] == 1
    assert telemetry["group_checks"]["expiry_saturday_fallback"] == 1
    assert telemetry["group_checks"]["quote_close_fallback"] == 1
    assert telemetry["group_checks"]["low_chain_used"] == 1
    assert telemetry["group_checks"]["expiry_close_fallback"] == 0
    assert telemetry["group_checks"]["wide_rel_spread"] == 0

    ticker_map = {item["ticker"]: item for item in telemetry["tickers"]}
    assert ticker_map["AAPL"]["completed_jobs"] == 2
    assert ticker_map["AAPL"]["planned_jobs"] == 4
    assert ticker_map["AAPL"]["kept_groups"] == 1
    assert ticker_map["AAPL"]["rows"] == 3
    assert ticker_map["AAPL"]["issue_count_sum"] == 6
    assert ticker_map["AAPL"]["flagged_rows"] == 2
    assert ticker_map["AAPL"]["fallback_rows"] == 2
    assert ticker_map["AAPL"]["wide_spread_rows"] == 1
    assert ticker_map["AAPL"]["clean_rows"] == 1
    assert ticker_map["AAPL"]["watch_rows"] == 1
    assert ticker_map["AAPL"]["noisy_rows"] == 1
    assert ticker_map["AAPL"]["drop_reasons"] == {"empty_option_chain": 1}
    assert ticker_map["NVDA"]["completed_jobs"] == 1
    assert ticker_map["NVDA"]["planned_jobs"] == 4
    assert ticker_map["NVDA"]["kept_groups"] == 0
    assert ticker_map["NVDA"]["rows"] == 0
    assert ticker_map["NVDA"]["drop_reasons"] == {"missing_ST": 1}
    assert telemetry["drop_reasons"] == {
        "empty_option_chain": 1,
        "missing_ST": 1,
    }


def test_dataset_job_status_includes_live_telemetry() -> None:
    services = importlib.import_module("app.services.datasets")
    models = importlib.import_module("app.models.datasets")

    job = services.DatasetJob(
        "job-telemetry",
        models.DatasetRunRequest(start="2025-01-01", end="2025-01-31"),
    )
    line = (
        '[LIVE] {"phase":"building","drop_reasons":{"empty_option_chain":2},'
        '"group_checks":{"asof_close_fallback":1,"expiry_close_fallback":0,'
        '"expiry_saturday_fallback":1,"quote_close_fallback":0,'
        '"low_chain_used":1,"wide_rel_spread":0},'
        '"tickers":[{"ticker":"NVDA","completed_jobs":3,"planned_jobs":8,'
        '"kept_groups":2,"rows":24,"issue_count_sum":18,'
        '"flagged_rows":12,"fallback_rows":4,"wide_spread_rows":3,'
        '"clean_rows":12,"watch_rows":9,"noisy_rows":3,'
        '"drop_reasons":{"empty_option_chain":1}}]}'
    )

    job._record_line("stdout", f"{line}\n")
    status = job.to_status()

    assert status.telemetry is not None
    assert status.telemetry.phase == "building"
    assert status.telemetry.drop_reasons == {"empty_option_chain": 2}
    assert status.telemetry.group_checks.asof_close_fallback == 1
    assert status.telemetry.group_checks.expiry_saturday_fallback == 1
    assert len(status.telemetry.tickers) == 1
    assert status.telemetry.tickers[0].ticker == "NVDA"
    assert status.telemetry.tickers[0].completed_jobs == 3
    assert status.telemetry.tickers[0].kept_groups == 2
    assert status.telemetry.tickers[0].rows == 24
    assert status.telemetry.tickers[0].issue_count_sum == 18
    assert status.telemetry.tickers[0].flagged_rows == 12
    assert status.telemetry.tickers[0].fallback_rows == 4
    assert status.telemetry.tickers[0].wide_spread_rows == 3
    assert status.telemetry.tickers[0].clean_rows == 12
    assert status.telemetry.tickers[0].watch_rows == 9
    assert status.telemetry.tickers[0].noisy_rows == 3
