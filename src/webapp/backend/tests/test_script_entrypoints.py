from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

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
