import json
import threading
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_SCRIPT = REPO_ROOT / "src" / "scripts" / "03-auto-calibrate-logit-model-v1.1.py"

sys.path.insert(0, str(REPO_ROOT / "src" / "webapp" / "backend"))

from app.services.parity_check import check_artifact_parity  # noqa: E402


def load_auto_module():
    spec = spec_from_file_location("auto_calibrate", AUTO_SCRIPT)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUTO = load_auto_module()
CATALOG_PATH = REPO_ROOT / "config" / "autocalibrate_feature_catalog.json"
TEMPLATE_PATH = REPO_ROOT / "config" / "metadata_schema_template.json"


def test_feature_catalog_loads():
    catalog = AUTO.load_catalog(CATALOG_PATH)
    assert "groups" in catalog
    assert "hyperparameters" in catalog
    assert "base_features_always_included" in catalog


def test_enumerate_option_only_excludes_pm_features():
    catalog = AUTO.load_catalog(CATALOG_PATH)
    combos, _ = AUTO.enumerate_feature_combos(catalog, mode="option_only")
    pm_features = set()
    for group in catalog["groups"]:
        if group["name"].startswith("pm_"):
            pm_features.update(group["options"])
    for combo in combos:
        selected = {feat for feats in combo.values() for feat in feats}
        assert not (selected & pm_features)


def test_enumerate_mixed_includes_pm_signal():
    catalog = AUTO.load_catalog(CATALOG_PATH)
    combos, _ = AUTO.enumerate_feature_combos(catalog, mode="mixed")
    for combo in combos:
        selected = {feat for feats in combo.values() for feat in feats}
        assert "pm_mid" in selected


def test_x_m_dependency_enforced():
    catalog = AUTO.load_catalog(CATALOG_PATH)
    combos, _ = AUTO.enumerate_feature_combos(catalog, mode="option_only")
    for combo in combos:
        selected = {feat for feats in combo.values() for feat in feats}
        if "x_m" in selected:
            assert "log_m_fwd" in selected


def test_rv20_mutual_exclusion():
    catalog = AUTO.load_catalog(CATALOG_PATH)
    combos, _ = AUTO.enumerate_feature_combos(catalog, mode="option_only")
    for combo in combos:
        selected = {feat for feats in combo.values() for feat in feats}
        assert not ("rv20" in selected and "rv20_sqrtT" in selected)


def test_trial_id_determinism():
    config_a = {
        "mode": "option_only",
        "feature_choices": {"volatility": ["rv20"], "moneyness": []},
        "ticker_intercepts": "none",
        "train_decay_half_life_weeks": 0,
        "calibrate": "none",
        "foundation_enabled": False,
        "foundation_weight": 1.0,
    }
    config_b = {
        "mode": "option_only",
        "feature_choices": {"volatility": [], "moneyness": ["log_m_fwd"]},
        "ticker_intercepts": "none",
        "train_decay_half_life_weeks": 0,
        "calibrate": "none",
        "foundation_enabled": False,
        "foundation_weight": 1.0,
    }
    assert AUTO.make_trial_id(config_a) == AUTO.make_trial_id(config_a)
    assert AUTO.make_trial_id(config_a) != AUTO.make_trial_id(config_b)


def test_trial_grid_sorted_deterministically():
    catalog = AUTO.load_catalog(CATALOG_PATH)
    combos1, _ = AUTO.enumerate_feature_combos(catalog, mode="option_only")
    combos2, _ = AUTO.enumerate_feature_combos(catalog, mode="option_only")
    grid1 = AUTO.enumerate_trial_grid(combos1, catalog, mode="option_only")
    grid2 = AUTO.enumerate_trial_grid(combos2, catalog, mode="option_only")
    assert grid1 == grid2


def test_parity_check_detects_missing_key(tmp_path: Path):
    meta_path = REPO_ROOT / "src" / "data" / "models" / "1dte-logit-model" / "trials" / (
        "001__C=0.05__feat=min__ti=non_foundation__tx=0__decay=0__cal=none__foundation=off__bbd10b43"
    ) / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    metadata.pop("csv", None)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    warnings = check_artifact_parity(tmp_path, TEMPLATE_PATH)
    assert any("Missing key" in warning and "metadata.csv" in warning for warning in warnings)


def test_run_trial_skips_existing_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    result_path = trial_dir / "trial_result.json"
    result_path.write_text(json.dumps({"status": "success", "objective": 0.1}))

    def _fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(AUTO.subprocess, "run", _fail_run)
    result = AUTO.run_trial(trial_dir, ["echo", "hi"], 0)
    assert result["status"] == "success"


def test_run_trial_isolates_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    def _mock_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(AUTO.subprocess, "run", _mock_run)
    result = AUTO.run_trial(trial_dir, ["echo", "hi"], 0)
    assert result["status"] == "failed"


def test_progress_json_atomic_write(tmp_path: Path):
    def _writer():
        AUTO.write_progress(tmp_path, 10, 1, 0, 0.5, None)

    threads = [threading.Thread(target=_writer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["trials_total"] == 10


@pytest.mark.integration
def test_autocalibrate_smoke(tmp_path: Path):
    dataset_path = tmp_path / "tiny.csv"
    rows = []
    import datetime as dt
    start = dt.date(2024, 1, 5)
    for i in range(30):
        week = start + dt.timedelta(weeks=i)
        rows.append({
            "label": 1 if i % 2 == 0 else 0,
            "week_friday": week.isoformat(),
            "ticker": "SPY",
            "x_logit_prn": 0.1 + (i * 0.01),
            "log_m_fwd": -0.2 + (i * 0.01),
            "abs_log_m_fwd": 0.2 + (i * 0.01),
            "rv20": 0.3 + (i * 0.01),
            "rv20_sqrtT": 0.3 + (i * 0.01),
            "pRN": 0.5,
        })

    import csv
    with dataset_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "auto_out"
    cmd = [
        sys.executable,
        str(AUTO_SCRIPT),
        "--csv",
        str(dataset_path),
        "--out-dir",
        str(out_dir),
        "--mode",
        "option_only",
        "--max-trials",
        "3",
    ]
    result = AUTO.subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0

    progress_path = out_dir / "progress.json"
    assert progress_path.exists()
    progress = json.loads(progress_path.read_text())
    assert progress.get("stage") == "done"

    leaderboard_path = out_dir / "leaderboard.csv"
    assert leaderboard_path.exists()
    with leaderboard_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3

    metadata_path = out_dir / "metadata.json"
    assert metadata_path.exists()

    warnings = check_artifact_parity(out_dir, TEMPLATE_PATH)
    assert isinstance(warnings, list)
