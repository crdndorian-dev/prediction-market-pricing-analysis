from __future__ import annotations

from pathlib import Path
from typing import Iterable


_REPO_ANCHORS = ("src", "config")


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve(strict=False)
        key = resolved.as_posix()
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _resolve_input_path(path_value: str | Path, repo_root: Path) -> Path:
    raw = Path(path_value).expanduser()
    if raw.is_absolute():
        return raw.resolve(strict=False)
    if raw.parts and raw.parts[0] in _REPO_ANCHORS:
        return (repo_root / raw).resolve(strict=False)
    return (Path.cwd() / raw).resolve(strict=False)


def serialize_portable_repo_path(path_value: str | Path | None, repo_root: Path) -> str | None:
    if path_value is None:
        return None
    resolved = _resolve_input_path(path_value, repo_root)
    repo_root = repo_root.resolve(strict=False)
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _relocate_repo_path(raw: Path, repo_root: Path) -> Path | None:
    resolved = raw.resolve(strict=False)
    repo_root = repo_root.resolve(strict=False)
    try:
        resolved.relative_to(repo_root)
        return resolved
    except ValueError:
        pass
    parts = resolved.parts
    for anchor in _REPO_ANCHORS:
        if anchor not in parts:
            continue
        anchor_index = parts.index(anchor)
        candidate = (repo_root / Path(*parts[anchor_index:])).resolve(strict=False)
        if candidate != resolved:
            return candidate
    return None


def manifest_path_candidates(path_value: str | Path | None, repo_root: Path) -> list[Path]:
    if path_value is None:
        return []
    raw = Path(path_value).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        relocated = _relocate_repo_path(raw, repo_root)
        if relocated is not None:
            candidates.append(relocated)
        candidates.append(raw.resolve(strict=False))
    else:
        candidates.append((repo_root / raw).resolve(strict=False))
        candidates.append((Path.cwd() / raw).resolve(strict=False))
    return _dedupe_paths(candidates)
