from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

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


stata_diagnostics = importlib.import_module("model_training.calibration.stata_diagnostics")
services = importlib.import_module("app.services.calibrate_models")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _make_split_frame(
    *,
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
) -> pd.DataFrame:
    ticker = np.where(x1 >= 0.0, "AAPL", "NVDA")
    return pd.DataFrame(
        {
            "feature_one": x1,
            "feature_two": x2,
            "ticker": ticker,
            "outcome_ST_gt_K": y.astype(int),
            "pRN": baseline,
        }
    )


def test_build_and_write_stata_diagnostics_writes_expected_artifacts(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)

    x1_train = np.linspace(-2.0, 2.0, 48)
    x2_train = np.sin(np.linspace(-1.0, 1.0, 48) * np.pi)
    p_train = _sigmoid(-0.15 + 1.0 * x1_train - 0.65 * x2_train)
    y_train = rng.binomial(1, p_train)
    baseline_train = np.clip(0.5 + 0.12 * np.tanh(x1_train / 2.5), 0.02, 0.98)

    x1_val = np.linspace(-1.8, 1.8, 24)
    x2_val = np.cos(np.linspace(-1.0, 1.0, 24) * np.pi / 2.0)
    p_val = _sigmoid(-0.1 + 0.9 * x1_val - 0.55 * x2_val)
    y_val = rng.binomial(1, p_val)
    baseline_val = np.clip(0.5 + 0.1 * np.tanh(x1_val / 2.0), 0.02, 0.98)

    x1_test = np.linspace(-1.6, 1.6, 24)
    x2_test = np.sin(np.linspace(-0.7, 0.7, 24) * np.pi)
    p_test = _sigmoid(-0.05 + 0.95 * x1_test - 0.45 * x2_test)
    y_test = rng.binomial(1, p_test)
    baseline_test = np.clip(0.5 + 0.1 * np.tanh(x1_test / 2.0), 0.02, 0.98)

    train_df = _make_split_frame(x1=x1_train, x2=x2_train, y=y_train, baseline=baseline_train)
    val_df = _make_split_frame(x1=x1_val, x2=x2_val, y=y_val, baseline=baseline_val)
    test_df = _make_split_frame(x1=x1_test, x2=x2_test, y=y_test, baseline=baseline_test)

    payload = stata_diagnostics.build_and_write_stata_diagnostics(
        out_dir=tmp_path,
        model_id="stata-diag-test",
        estimator="sklearn_logit",
        baseline_name="pRN",
        best_c=1.0,
        penalty="l2",
        solver="lbfgs",
        threshold_default=0.5,
        threshold_operating=None,
        production_coefficients=[1.0, -0.65],
        production_intercept=-0.15,
        production_feature_names=["feature_one", "feature_two"],
        shadow_exog=train_df[["feature_one", "feature_two"]].to_numpy(dtype=float),
        shadow_endog=train_df["outcome_ST_gt_K"].to_numpy(dtype=float),
        shadow_sample_weight=np.ones(len(train_df), dtype=float),
        shadow_feature_names=["feature_one", "feature_two"],
        numeric_features=["feature_one", "feature_two"],
        categorical_features=["ticker"],
        requested_numeric_features=["feature_one", "feature_two"],
        requested_categorical_features=["ticker", "dropped_indicator"],
        train_fit_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_fit_pred=p_train,
        val_pred=p_val,
        test_pred=p_test,
        target_col="outcome_ST_gt_K",
        n_bins=5,
        eceq_bins=5,
        fold_delta_rows=[
            {"fold": 1, "delta_logloss": -0.018},
            {"fold": 2, "delta_logloss": -0.007},
            {"fold": 3, "delta_logloss": 0.004},
        ],
        trainer_warnings=["Synthetic diagnostic warning"],
    )

    assert payload["schema_version"] == 1
    assert payload["header"]["model_id"] == "stata-diag-test"
    assert payload["header"]["inference_basis"] == "shadow_statsmodels_glm"
    assert [section["id"] for section in payload["sections"]] == [
        "global_significance",
        "coefficient_quality",
        "goodness_of_fit",
        "predictive_quality",
        "probabilistic_quality",
        "calibration",
        "stability_robustness",
        "data_specification_quality",
    ]
    assert payload["production_coefficient_table"] is not None
    assert payload["coefficient_table"] is not None
    assert payload["marginal_effects_table"] is not None
    assert payload["calibration_curve"] is not None
    assert "Synthetic diagnostic warning" in payload["warnings"]
    assert {row["series"] for row in payload["calibration_curve"]["rows"]} == {"model", "baseline"}
    assert payload["production_coefficient_table"]["basis"] == "production_sklearn_final_model"
    assert payload["production_coefficient_table"]["fit_scope"] == "train"
    assert payload["production_coefficient_table"]["rows"][0]["feature_name"] == "const"
    assert payload["production_coefficient_table"]["rows"][0]["coefficient"] == pytest.approx(-0.15)
    assert payload["production_coefficient_table"]["rows"][1]["feature_name"] == "feature_one"
    assert payload["production_coefficient_table"]["rows"][1]["coefficient"] == pytest.approx(1.0)
    assert payload["production_coefficient_table"]["rows"][2]["feature_name"] == "feature_two"
    assert payload["production_coefficient_table"]["rows"][2]["coefficient"] == pytest.approx(-0.65)

    diagnostics_path = tmp_path / "diagnostics_table.json"
    coefficients_path = tmp_path / "coefficient_diagnostics.csv"
    marginal_effects_path = tmp_path / "marginal_effects.csv"
    reliability_path = tmp_path / "reliability_bins.csv"

    assert diagnostics_path.exists()
    assert coefficients_path.exists()
    assert marginal_effects_path.exists()
    assert reliability_path.exists()

    persisted_payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert persisted_payload["header"]["feature_counts"]["transformed"] == 2
    assert persisted_payload["production_coefficient_table"]["subtitle"] == (
        "Production final sklearn model coefficients. Matches the displayed equation exactly."
    )
    data_quality_rows = {
        row["key"]: row
        for section in persisted_payload["sections"]
        if section["id"] == "data_specification_quality"
        for row in section["rows"]
    }
    assert data_quality_rows["dropped_features"]["train_fit"] == 1.0
    assert "dropped_indicator" in (data_quality_rows["dropped_features"]["note"] or "")

    coefficient_frame = pd.read_csv(coefficients_path)
    assert {"feature_name", "coefficient", "std_error", "odds_ratio"}.issubset(coefficient_frame.columns)
    assert "feature_one" in set(coefficient_frame["feature_name"])

    marginal_effect_frame = pd.read_csv(marginal_effects_path)
    assert {"feature_name", "ame", "std_error"}.issubset(marginal_effect_frame.columns)

    reliability_frame = pd.read_csv(reliability_path)
    assert {"split", "series", "bin", "pred_mean", "obs_rate", "abs_gap"}.issubset(reliability_frame.columns)
    assert {"train_fit", "val", "test"}.issubset(set(reliability_frame["split"]))


def test_service_loader_accepts_diagnostics_payload_and_whitelists_artifacts(tmp_path: Path) -> None:
    diagnostics_payload = {
        "schema_version": 1,
        "header": {"model_id": "example"},
        "sections": [],
        "calibration_curve": {
            "split": "val",
            "binning": "equal_mass",
            "rows": [
                {
                    "bin": 1,
                    "n": 10,
                    "pred_mean": 0.42,
                    "obs_rate": 0.50,
                    "abs_gap": 0.08,
                }
            ],
        },
    }
    diagnostics_path = tmp_path / "diagnostics_table.json"
    diagnostics_path.write_text(json.dumps(diagnostics_payload), encoding="utf-8")

    loaded = services._load_stata_diagnostics(diagnostics_path)
    assert loaded is not None
    assert loaded["calibration_curve"]["rows"][0]["series"] == "model"
    assert services._load_stata_diagnostics(tmp_path / "missing.json") is None
    assert "diagnostics_table.json" in services.SELECTED_MODEL_IMPORTANT_FILES
    assert "coefficient_diagnostics.csv" in services.SELECTED_MODEL_IMPORTANT_FILES
    assert "marginal_effects.csv" in services.SELECTED_MODEL_IMPORTANT_FILES


def test_service_loader_backfills_calibration_curve_series_from_reliability_bins(tmp_path: Path) -> None:
    diagnostics_payload = {
        "schema_version": 1,
        "header": {"model_id": "example"},
        "sections": [],
        "calibration_curve": {
            "split": "test",
            "binning": "equal_mass",
            "rows": [
                {
                    "bin": 1,
                    "n": 12,
                    "pred_mean": 0.31,
                    "obs_rate": 0.28,
                    "abs_gap": 0.03,
                }
            ],
        },
    }
    diagnostics_path = tmp_path / "diagnostics_table.json"
    diagnostics_path.write_text(json.dumps(diagnostics_payload), encoding="utf-8")

    pd.DataFrame(
        [
            {"split": "test", "series": "model", "bin": 1, "n": 12, "pred_mean": 0.31, "obs_rate": 0.28, "abs_gap": 0.03},
            {"split": "test", "series": "baseline", "bin": 1, "n": 12, "pred_mean": 0.38, "obs_rate": 0.28, "abs_gap": 0.10},
        ]
    ).to_csv(tmp_path / "reliability_bins.csv", index=False)

    loaded = services._load_stata_diagnostics(diagnostics_path)
    assert loaded is not None
    assert {row["series"] for row in loaded["calibration_curve"]["rows"]} == {"model", "baseline"}
    assert loaded["calibration_curve"]["rows"][1]["pred_mean"] == pytest.approx(0.38)


def test_service_loader_backfills_production_coefficient_table_from_metadata(tmp_path: Path) -> None:
    diagnostics_payload = {
        "schema_version": 1,
        "header": {"model_id": "example"},
        "sections": [],
        "coefficient_table": {
            "basis": "shadow_statsmodels_glm",
            "rows": [
                {
                    "feature_name": "feature_one",
                    "display_name": "Feature one",
                    "feature_group": "core",
                    "coefficient": 0.12,
                }
            ],
        },
    }
    diagnostics_path = tmp_path / "diagnostics_table.json"
    diagnostics_path.write_text(json.dumps(diagnostics_payload), encoding="utf-8")

    metadata = {
        "coefficients": [1.1624, -0.0276, 0.1169],
        "intercept": 0.3726,
        "feature_names_out": ["x_logit_prn", "abs_log_m", "rv10"],
    }

    loaded = services._load_stata_diagnostics(diagnostics_path, metadata=metadata)
    assert loaded is not None
    assert loaded["production_coefficient_table"]["basis"] == "production_sklearn_final_model"
    assert loaded["production_coefficient_table"]["rows"][0]["feature_name"] == "const"
    assert loaded["production_coefficient_table"]["rows"][0]["coefficient"] == pytest.approx(0.3726)
    assert [row["feature_name"] for row in loaded["production_coefficient_table"]["rows"][1:]] == [
        "x_logit_prn",
        "abs_log_m",
        "rv10",
    ]
    assert loaded["production_coefficient_table"]["rows"][1]["coefficient"] == pytest.approx(1.1624)
    assert loaded["coefficient_table"]["fit_scope"] == "train_fit"
    assert "transformed train_fit design matrix" in loaded["coefficient_table"]["subtitle"]


def test_build_and_write_stata_diagnostics_raises_clean_warning_when_statsmodels_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "feature_one": [0.1, -0.2],
            "outcome_ST_gt_K": [1, 0],
            "pRN": [0.55, 0.45],
        }
    )

    monkeypatch.setattr(stata_diagnostics, "sm", None)
    monkeypatch.setattr(stata_diagnostics, "variance_inflation_factor", None)
    monkeypatch.setattr(
        stata_diagnostics,
        "_STATSMODELS_IMPORT_ERROR",
        ModuleNotFoundError("No module named 'statsmodels'"),
    )

    with pytest.raises(RuntimeError, match="^Stata diagnostics skipped: statsmodels dependency unavailable\\."):
        stata_diagnostics.build_and_write_stata_diagnostics(
            out_dir=tmp_path,
            model_id="stata-diag-missing-statsmodels",
            estimator="sklearn_logit",
        baseline_name="pRN",
        best_c=1.0,
        penalty="l2",
        solver="lbfgs",
        threshold_default=0.5,
        threshold_operating=None,
        production_coefficients=[0.1],
        production_intercept=0.2,
        production_feature_names=["feature_one"],
        shadow_exog=np.asarray([[0.1], [-0.2]], dtype=float),
        shadow_endog=np.asarray([1.0, 0.0], dtype=float),
        shadow_sample_weight=np.ones(2, dtype=float),
            shadow_feature_names=["feature_one"],
            numeric_features=["feature_one"],
            categorical_features=[],
            requested_numeric_features=["feature_one"],
            requested_categorical_features=[],
            train_fit_df=frame,
            val_df=frame,
            test_df=frame,
            train_fit_pred=np.asarray([0.6, 0.4], dtype=float),
            val_pred=np.asarray([0.6, 0.4], dtype=float),
            test_pred=np.asarray([0.6, 0.4], dtype=float),
            target_col="outcome_ST_gt_K",
            n_bins=2,
            eceq_bins=2,
            fold_delta_rows=[],
            trainer_warnings=[],
        )
