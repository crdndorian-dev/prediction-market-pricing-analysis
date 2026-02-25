#!/usr/bin/env python3
"""Safely clean generated output files with a conservative allow/deny policy.

Default behavior is DRY RUN. Use --apply to actually delete.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Sequence, Tuple

# ---- Safety policies ----
CODE_DIRS = {
    "src",
    "app",
    "web",
    "backend",
    "scripts",
    "config",
    "docs",
    "vendor",
}

DEPENDENCY_DIRS = {
    "node_modules",
    "venv",
    ".venv",
    "__pypackages__",
}

PROTECTED_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".json",
    ".yml",
    ".yaml",
    ".toml",
    ".md",
}

DEFAULT_OUTPUT_ROOT_CANDIDATES = [
    "data",
    "outputs",
    "output",
    "artifacts",
    "runs",
    "cache",
    ".cache",
    "tmp",
    "logs",
    "out",
    "build",
    "dist",
]

GENERATED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "cache",
    ".cache",
    "tmp",
    "logs",
    "artifacts",
    "outputs",
    "output",
    "runs",
}

GENERATED_FILE_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}

# Denylist for deletion (only delete files with these extensions unless in a generated dir)
DELETABLE_EXTS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".pkl",
    ".pickle",
    ".joblib",
    ".npy",
    ".npz",
    ".pyc",
    ".log",
    ".txt",
    ".html",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
}

# Allowlist for option-chain dataset related CSVs
DEFAULT_KEEP_PATTERNS = [
    "src/data/raw/option-chain/**/*.csv",
    "src/data/raw/option-chains/**/*.csv",
    "data/raw/option-chain/**/*.csv",
    "data/raw/option-chains/**/*.csv",
    "src/data/raw/options_chain_dataset.csv",
    "src/data/raw/options-chain-dataset.csv",
    "data/raw/options_chain_dataset.csv",
    "data/raw/options-chain-dataset.csv",
    "src/data/raw/option_chain_historic_dataset_drops.csv",
    "src/data/raw/option-chain-historic-dataset-drops.csv",
    "data/raw/option_chain_historic_dataset_drops.csv",
    "data/raw/option-chain-historic-dataset-drops.csv",
]


@dataclass(frozen=True)
class DeleteItem:
    path: Path
    size: int


def parse_multi_value(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return []
    out: List[str] = []
    for v in values:
        for part in v.split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def find_repo_root(start: Path) -> Optional[Path]:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
        if (p / "README.md").exists() and (p / "src").exists():
            return p
    return None


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def is_within_code_dir(rel: PurePosixPath) -> bool:
    return len(rel.parts) > 0 and rel.parts[0] in CODE_DIRS


def is_within_dependency_dir(rel: PurePosixPath) -> bool:
    return len(rel.parts) > 0 and rel.parts[0] in DEPENDENCY_DIRS


def match_keep_pattern(rel_posix: str, name: str, pattern: str) -> bool:
    if "/" in pattern:
        return PurePosixPath(rel_posix).match(pattern)
    return PurePosixPath(name).match(pattern)


def is_keep(rel_posix: str, name: str, keep_patterns: Sequence[str]) -> bool:
    for pattern in keep_patterns:
        if match_keep_pattern(rel_posix, name, pattern):
            return True
    return False


def has_generated_evidence(rel: PurePosixPath, name: str, suffix: str) -> bool:
    if name in GENERATED_FILE_NAMES:
        return True
    if suffix in DELETABLE_EXTS:
        return True
    if any(part in GENERATED_DIR_NAMES for part in rel.parts):
        return True
    return False


def format_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Safely clean generated output files (dry run by default)."
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this flag, runs in dry-run mode.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Allow permanent deletion when send2trash is unavailable.",
    )
    ap.add_argument(
        "--include-dirs",
        action="append",
        help="Repo-relative or absolute dirs to scan (comma-separated or repeatable).",
    )
    ap.add_argument(
        "--exclude-dirs",
        action="append",
        help="Repo-relative or absolute dirs to skip (comma-separated or repeatable).",
    )
    ap.add_argument(
        "--keep-pattern",
        action="append",
        help="Additional keep pattern (glob). Can be repeated.",
    )
    ap.add_argument(
        "--manifest",
        action="store_true",
        help="Write a manifest file of deletion candidates.",
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="How many sample paths to print (default: 25).",
    )
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    repo_root = find_repo_root(Path.cwd())
    if repo_root is None:
        print("ERROR: Could not detect repo root (.git or README.md+src). Aborting.")
        return 2

    include_dirs_raw = parse_multi_value(args.include_dirs)
    exclude_dirs_raw = parse_multi_value(args.exclude_dirs)
    keep_patterns = DEFAULT_KEEP_PATTERNS + parse_multi_value(args.keep_pattern)

    include_dirs: List[Path] = []
    if include_dirs_raw:
        for d in include_dirs_raw:
            p = Path(d)
            if not p.is_absolute():
                p = repo_root / p
            include_dirs.append(p.resolve())
    else:
        for name in DEFAULT_OUTPUT_ROOT_CANDIDATES:
            p = (repo_root / name)
            if p.exists() and p.is_dir():
                include_dirs.append(p.resolve())
        # Include top-level cache dirs if present
        for name in ["__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"]:
            p = (repo_root / name)
            if p.exists() and p.is_dir():
                include_dirs.append(p.resolve())

    exclude_dirs: List[Path] = []
    for d in exclude_dirs_raw:
        p = Path(d)
        if not p.is_absolute():
            p = repo_root / p
        exclude_dirs.append(p.resolve())

    # Always exclude code and dependency dirs
    for name in CODE_DIRS.union(DEPENDENCY_DIRS).union({".git"}):
        p = repo_root / name
        if p.exists() and p.is_dir():
            exclude_dirs.append(p.resolve())

    # Guard: refuse to scan code dirs
    for d in include_dirs:
        if not is_within(d, repo_root):
            print(f"ERROR: Include dir is outside repo root: {d}")
            return 2
        rel = PurePosixPath(d.relative_to(repo_root).as_posix())
        if is_within_code_dir(rel) or is_within_dependency_dir(rel):
            print(f"ERROR: Refusing to scan code/dependency dir: {d}")
            return 2

    if not include_dirs:
        print("No include dirs found. Nothing to do.")
        return 0

    delete_items: List[DeleteItem] = []
    kept = 0
    protected = 0
    unproven = 0
    skipped_symlink = 0

    def is_excluded(path: Path) -> bool:
        for ex in exclude_dirs:
            if is_within(path, ex):
                return True
        return False

    for root in include_dirs:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            try:
                if path.is_symlink():
                    skipped_symlink += 1
                    continue
                if path.is_dir():
                    continue
                if not is_within(path.resolve(), repo_root):
                    continue
            except OSError:
                continue

            if is_excluded(path):
                continue

            rel = PurePosixPath(path.relative_to(repo_root).as_posix())
            if is_within_code_dir(rel) or is_within_dependency_dir(rel):
                protected += 1
                continue

            suffix = path.suffix.lower()
            if suffix in PROTECTED_EXTS:
                protected += 1
                continue

            rel_posix = rel.as_posix()
            if is_keep(rel_posix, path.name, keep_patterns):
                kept += 1
                continue

            if not has_generated_evidence(rel, path.name, suffix):
                unproven += 1
                continue

            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            delete_items.append(DeleteItem(path=path, size=size))

    # Also consider repo-root generated junk files (e.g., .DS_Store)
    for name in GENERATED_FILE_NAMES:
        path = repo_root / name
        if not path.exists() or not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(repo_root).as_posix())
        suffix = path.suffix.lower()
        if suffix in PROTECTED_EXTS:
            protected += 1
            continue
        rel_posix = rel.as_posix()
        if is_keep(rel_posix, path.name, keep_patterns):
            kept += 1
            continue
        if not has_generated_evidence(rel, path.name, suffix):
            unproven += 1
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        delete_items.append(DeleteItem(path=path, size=size))

    delete_items.sort(key=lambda x: (-x.size, x.path.as_posix()))

    total_size = sum(item.size for item in delete_items)

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Repo root: {repo_root}")
    print("Include dirs:")
    for d in include_dirs:
        print(f"  - {d}")
    print("Exclude dirs:")
    for d in sorted(set(exclude_dirs)):
        print(f"  - {d}")

    print(f"Candidates: {len(delete_items)} files")
    print(f"Total size: {format_bytes(total_size)}")
    print(f"Kept by allowlist: {kept}")
    print(f"Protected (code/config): {protected}")
    print(f"Unproven (no generated evidence): {unproven}")
    print(f"Skipped symlinks: {skipped_symlink}")

    if delete_items:
        print("Sample: ")
        for item in delete_items[: max(0, args.sample_size)]:
            rel_path = item.path.relative_to(repo_root)
            print(f"  - {format_bytes(item.size):>9}  {rel_path}")

    manifest_path = None
    if args.manifest:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = repo_root / f"cleanup_manifest_{ts}.txt"
        with manifest_path.open("w", encoding="utf-8") as fh:
            for item in delete_items:
                rel_path = item.path.relative_to(repo_root)
                fh.write(f"{item.size}\t{rel_path.as_posix()}\n")
        print(f"Manifest written: {manifest_path}")

    if not args.apply:
        return 0

    if not delete_items:
        print("Nothing to delete.")
        return 0

    # Prefer trash if available
    try:
        from send2trash import send2trash  # type: ignore
    except Exception:
        send2trash = None

    if send2trash is None and not args.force:
        print("ERROR: send2trash not available. Re-run with --apply --force to delete permanently.")
        return 2

    deleted = 0
    errors: List[Tuple[Path, str]] = []

    for item in delete_items:
        try:
            if send2trash is not None:
                send2trash(str(item.path))
            else:
                item.path.unlink()
            deleted += 1
        except Exception as exc:
            errors.append((item.path, str(exc)))

    print(f"Deleted: {deleted} files")
    if errors:
        print("Errors:")
        for path, msg in errors[:20]:
            rel_path = path.relative_to(repo_root)
            print(f"  - {rel_path}: {msg}")
        if len(errors) > 20:
            print(f"  ... {len(errors) - 20} more")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
