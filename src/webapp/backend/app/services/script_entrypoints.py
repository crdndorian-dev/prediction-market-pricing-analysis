from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPTS_DIR = BASE_DIR / "src" / "scripts"
ENTRYPOINTS_DIR = SCRIPTS_DIR / "entrypoints"


@dataclass(frozen=True)
class ScriptEntrypoint:
    public_name: str
    path: Path
    description: str


def _entrypoint(public_name: str, description: str) -> ScriptEntrypoint:
    return ScriptEntrypoint(
        public_name=public_name,
        path=ENTRYPOINTS_DIR / public_name,
        description=description,
    )


OPTION_CHAIN_DATASET_SCRIPT = _entrypoint(
    "option-chain-build-historic-dataset.py",
    "Option-chain historic dataset builder",
)
OPTION_CHAIN_CLEANUP_SCRIPT = _entrypoint(
    "option-chain-clean-training-dataset.py",
    "Option-chain training dataset cleanup",
)
CALIBRATE_MODEL_SCRIPT = _entrypoint(
    "calibrate-logit-model.py",
    "Calibration model trainer",
)
AUTO_CALIBRATE_MODEL_SCRIPT = _entrypoint(
    "auto-calibrate-logit-model.py",
    "Auto-calibration orchestrator",
)
POLYMARKET_WEEKLY_HISTORY_SCRIPT = _entrypoint(
    "polymarket-weekly-history.py",
    "Polymarket weekly history pipeline",
)
POLYMARKET_BUILD_FEATURES_SCRIPT = _entrypoint(
    "polymarket-build-features.py",
    "Polymarket feature builder",
)
POLYMARKET_MARKET_MAP_SCRIPT = _entrypoint(
    "polymarket-market-map.py",
    "Polymarket market-map builder",
)
POLYMARKET_FETCH_SNAPSHOT_SCRIPT = _entrypoint(
    "polymarket-fetch-snapshot.py",
    "Polymarket snapshot fetcher",
)
POLYMARKET_MARKETS_REFRESH_SCRIPT = _entrypoint(
    "polymarket-markets-refresh.py",
    "Polymarket markets refresh pipeline",
)
POLYMARKET_RUN_PRN_REFRESH_SCRIPT = _entrypoint(
    "polymarket-run-prn-refresh.py",
    "Polymarket run-local pRN refresh",
)

CANONICAL_SCRIPT_ENTRYPOINTS = (
    OPTION_CHAIN_DATASET_SCRIPT,
    OPTION_CHAIN_CLEANUP_SCRIPT,
    CALIBRATE_MODEL_SCRIPT,
    AUTO_CALIBRATE_MODEL_SCRIPT,
    POLYMARKET_WEEKLY_HISTORY_SCRIPT,
    POLYMARKET_BUILD_FEATURES_SCRIPT,
    POLYMARKET_MARKET_MAP_SCRIPT,
    POLYMARKET_FETCH_SNAPSHOT_SCRIPT,
    POLYMARKET_MARKETS_REFRESH_SCRIPT,
    POLYMARKET_RUN_PRN_REFRESH_SCRIPT,
)
