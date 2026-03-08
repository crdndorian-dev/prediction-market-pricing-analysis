from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


BASE_DIR = Path(__file__).resolve().parents[4]
ENV_FILE = BASE_DIR / ".env"
SUBGRAPH_ENV_SAMPLE = BASE_DIR / "config" / "polymarket_subgraph.env.sample"


def _load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            env[key] = value
    return env


def load_project_env() -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for path in (SUBGRAPH_ENV_SAMPLE, ENV_FILE):
        merged.update(_load_env_file(path))
    merged.update({key: value for key, value in os.environ.items() if value is not None})
    return merged


def get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    env = load_project_env()
    return env.get(name, default)


def get_analysis_database_url() -> Optional[str]:
    return get_setting("POLYMARKET_ANALYSIS_DATABASE_URL")
