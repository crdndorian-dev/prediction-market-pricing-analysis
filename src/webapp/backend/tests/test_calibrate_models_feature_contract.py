from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ROOT = REPO_ROOT / "src" / "webapp" / "backend"
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


services = importlib.import_module("app.services.calibrate_models")
models = importlib.import_module("app.models.calibrate_models")


def _write_option_chain_training_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "ticker,week_friday,outcome_ST_gt_K,pRN,log_m_fwd,abs_log_m_fwd,rv20,rel_spread_median,dividend_yield,spot_scale_used,quality_issue_count,quality_bucket",
                "AAPL,2026-03-06,1,0.52,0.08,0.08,0.24,0.012,0.004,spot,0,clean",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_get_dataset_features_returns_raw_columns_and_registry_backed_selectable_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = _write_option_chain_training_csv(tmp_path / "training-contract.csv")
    monkeypatch.setattr(services, "_resolve_project_path", lambda _: dataset_path)
    monkeypatch.setattr(services, "_ensure_dataset_path", lambda _: None)

    response = services.get_dataset_features("ignored/training-contract.csv")

    assert response.available_columns == [
        "ticker",
        "week_friday",
        "outcome_ST_gt_K",
        "pRN",
        "log_m_fwd",
        "abs_log_m_fwd",
        "rv20",
        "rel_spread_median",
        "dividend_yield",
        "spot_scale_used",
        "quality_issue_count",
        "quality_bucket",
    ]
    selectable_names = [feature.name for feature in response.selectable_features]
    assert selectable_names == [
        "log_m_fwd",
        "abs_log_m_fwd",
        "rv20",
        "rel_spread_median",
        "dividend_yield",
        "quality_issue_count",
        "spot_scale_used",
    ]
    assert "rv20_sqrtT" not in response.available_columns
    assert "rv20_sqrtT" not in selectable_names
    assert "log_m_fwd_over_volT" not in selectable_names
    assert "quality_bucket" not in selectable_names


@pytest.mark.parametrize(
    ("feature_field", "feature_value", "error_fragment"),
    [
        ("features", "x_logit_prn,rv20_sqrtT", "Unsupported numeric features requested: rv20_sqrtT."),
        (
            "features",
            "x_logit_prn,log_m_fwd_over_volT",
            "Unsupported numeric features requested: log_m_fwd_over_volT.",
        ),
        (
            "features",
            "x_logit_prn,log_m_fwd,abs_log_m_fwd",
            "Mutually exclusive numeric features requested: log_m_fwd, abs_log_m_fwd.",
        ),
        ("features", "x_logit_prn,x_abs_m", "Unsupported numeric features requested: x_abs_m."),
        ("categorical_features", "quality_bucket", "Unsupported categorical features requested: quality_bucket."),
        (
            "categorical_features",
            "sanity_regime",
            "Unsupported categorical features requested: sanity_regime.",
        ),
    ],
)
def test_build_config_payload_rejects_removed_or_unknown_features(
    tmp_path: Path,
    feature_field: str,
    feature_value: str,
    error_fragment: str,
) -> None:
    dataset_path = _write_option_chain_training_csv(tmp_path / "training-invalid.csv")
    payload = models.CalibrateModelRunRequest(csv="ignored/training-invalid.csv", **{feature_field: feature_value})

    with pytest.raises(ValueError, match=error_fragment):
        services._build_config_payload(
            payload,
            dataset_path=dataset_path,
            out_dir=tmp_path / "out",
        )


def test_build_config_payload_uses_registry_default_feature_order(tmp_path: Path) -> None:
    dataset_path = _write_option_chain_training_csv(tmp_path / "training-defaults.csv")
    payload = models.CalibrateModelRunRequest(csv="ignored/training-defaults.csv")

    config_payload = services._build_config_payload(
        payload,
        dataset_path=dataset_path,
        out_dir=tmp_path / "out",
    )

    assert config_payload["features"] == "x_logit_prn,log_m_fwd,rv20,rel_spread_median,dividend_yield"
    assert config_payload["categorical_features"] is None


def test_auto_feature_sets_normalize_to_registry_defaults(tmp_path: Path) -> None:
    dataset_path = _write_option_chain_training_csv(tmp_path / "training-auto.csv")

    feature_sets = services._normalize_auto_feature_sets(None, dataset_path=dataset_path)

    assert feature_sets == [
        ["x_logit_prn"],
        ["x_logit_prn", "rv20"],
        ["x_logit_prn", "abs_log_m_fwd"],
        ["x_logit_prn", "rv20", "abs_log_m_fwd"],
        ["x_logit_prn", "rv20", "abs_log_m_fwd", "rel_spread_median"],
    ]


def test_legacy_request_flags_are_rejected() -> None:
    with pytest.raises(ValueError, match="enable_x_abs_m is no longer supported"):
        models.CalibrateModelRunRequest(csv="ignored.csv", enable_x_abs_m=True)

    with pytest.raises(ValueError, match="allow_risky_features is no longer supported"):
        models.AutoSearchConfig(allow_risky_features=True)
