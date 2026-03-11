from __future__ import annotations

import sys
from pathlib import Path


def _looks_like_repo_root(path: Path) -> bool:
    return (
        (path / "README.md").exists()
        and (path / "src").is_dir()
        and (path / "src" / "scripts").is_dir()
    )


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if _looks_like_repo_root(candidate):
            return candidate
    raise RuntimeError(f"Could not resolve repository root from {start or __file__}.")


REPO_ROOT = find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = SRC_ROOT / "scripts"


def prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
