from __future__ import annotations

import csv
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import queue
import threading
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Set, Tuple
from uuid import uuid4
from urllib.parse import urlparse

from app.models.datasets import (
    DatasetFileSummary,
    DatasetJobProgress,
    DatasetJobStatus,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetRunRequest,
    DatasetRunResponse,
    DatasetRunSummary,
)

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "01-option-chain-build-historic-dataset-v1.0.py"

DEFAULT_OUT_DIR = "src/data/raw/option-chain"
DEFAULT_OUT_NAME = "pRN__history__mon_thu__PM10__v1.6.0.csv"
DEFAULT_DROPS_NAME = "pRN__history__mon_thu__drops__v1.6.0.csv"
DEFAULT_THETA_JAR = "ThetaTerminalv3.jar"
TRAINING_META_NAME = "training_selection.json"


def _unique_dirs(paths: List[Path]) -> List[Path]:
    seen = set()
    unique: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path.absolute())
    return unique


DATASET_BASE_DIRS = _unique_dirs(
    [
        BASE_DIR / DEFAULT_OUT_DIR,
        BASE_DIR / "data" / "raw" / "option-chain",
    ],
)


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = BASE_DIR / path
    path = path.absolute()
    try:
        path.relative_to(BASE_DIR)
    except ValueError as exc:
        raise ValueError("Path must be inside the project root.") from exc
    return path


def _dataset_base_dirs(existing_only: bool = True) -> List[Path]:
    dirs = [path for path in DATASET_BASE_DIRS if path.exists()] if existing_only else DATASET_BASE_DIRS[:]
    if not dirs:
        dirs = [DATASET_BASE_DIRS[0]]
    return dirs


def _dataset_display_base_dir() -> Path:
    return _dataset_base_dirs()[0]


def _find_dataset_base_for_path(path: Path) -> Path:
    for base in DATASET_BASE_DIRS:
        try:
            path.relative_to(base)
            return base
        except ValueError:
            continue
    raise ValueError("Path must be under src/data/raw/option-chain or data/raw/option-chain.")


def _is_port_open(host: str, port: int, timeout_s: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _theta_is_available(theta_url: str) -> bool:
    parsed = urlparse(theta_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    return _is_port_open(host, port)


def _ensure_theta_running(theta_url: str) -> None:
    if _theta_is_available(theta_url):
        return

    cmd = None
    jar_path = os.environ.get("THETA_TERMINAL_JAR")
    workdir = os.environ.get("THETA_TERMINAL_WORKDIR")
    creds_path = os.environ.get("THETA_TERMINAL_CREDS")
    log_path = os.environ.get("THETA_TERMINAL_LOG", "~/theta_terminal.log")

    candidate_paths = []
    candidate_paths.append(BASE_DIR / "vendor" / "theta" / DEFAULT_THETA_JAR)
    if jar_path:
        candidate_paths.append(Path(os.path.expanduser(jar_path)))
    candidate_paths.append(BASE_DIR / DEFAULT_THETA_JAR)

    jar = next((p for p in candidate_paths if p.exists()), None)
    if jar:
        cmd = f"java -jar {shlex.quote(str(jar))}"
        workdir = str(jar.parent)

    if not cmd:
        cmd = os.environ.get("THETA_TERMINAL_CMD")

    if not cmd:
        raise RuntimeError(
            "Theta Terminal is not reachable. Provide THETA_TERMINAL_CMD or set "
            "THETA_TERMINAL_JAR (and optionally THETA_TERMINAL_WORKDIR) so the web app "
            "can start it automatically."
        )

    if not creds_path:
        creds_candidates = [
            Path.home() / "Downloads" / "creds.txt",
            BASE_DIR / "vendor" / "theta" / "creds.txt",
            BASE_DIR / "creds.txt",
        ]
        creds = next((p for p in creds_candidates if p.exists()), None)
        if creds:
            creds_path = str(creds)

    if creds_path and "--creds-file" not in cmd:
        cmd = f"{cmd} --creds-file {shlex.quote(os.path.expanduser(creds_path))}"

    log_file = Path(os.path.expanduser(log_path)).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    stdout_target = log_file.open("ab")
    stderr_target = stdout_target
    subprocess.Popen(
        shlex.split(cmd),
        cwd=os.path.expanduser(workdir) if workdir else None,
        stdout=stdout_target,
        stderr=stderr_target,
    )

    wait_s = float(os.environ.get("THETA_TERMINAL_STARTUP_WAIT", "12"))
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if _theta_is_available(theta_url):
            return
        time.sleep(0.5)

    raise RuntimeError(
        "Theta Terminal did not become available in time. Check the local "
        "Theta Terminal app or increase THETA_TERMINAL_STARTUP_WAIT."
    )


def _add_value(cmd: List[str], flag: str, value: Optional[object]) -> None:
    if value is None:
        return
    text = str(value).strip()
    if not text:
        return
    cmd.extend([flag, text])


def _add_flag(cmd: List[str], flag: str, enabled: Optional[bool]) -> None:
    if enabled:
        cmd.append(flag)


def _add_optional_bool(cmd: List[str], flag: str, neg_flag: str, value: Optional[bool]) -> None:
    if value is None:
        return
    cmd.append(flag if value else neg_flag)

def _to_kebab_case(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    raw = re.sub(r'[\s_]+', '-', raw)
    raw = re.sub(r'[^a-zA-Z0-9-]', '', raw)
    raw = re.sub(r'-{2,}', '-', raw)
    return raw.strip('-').lower()


def _validate_payload(payload: DatasetRunRequest) -> Tuple[str, str]:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Dataset script not found at {SCRIPT_PATH}")

    start_value = payload.start.strip()
    end_value = payload.end.strip()
    if not start_value or not end_value:
        raise ValueError("start and end dates are required.")

    min_start = date(2023, 6, 1)
    max_end = date.today()
    try:
        start_date = date.fromisoformat(start_value)
        end_date = date.fromisoformat(end_value)
    except ValueError as exc:
        raise ValueError("start and end must be YYYY-MM-DD.") from exc

    if start_date < min_start:
        raise ValueError("start date must be on or after 2023-06-01.")
    if end_date > max_end:
        raise ValueError(f"end date must be on or before {max_end.isoformat()}.")
    if end_date < start_date:
        raise ValueError("end date must be on or after start date.")

    if payload.dataset_name:
        kebab = _to_kebab_case(payload.dataset_name)
        if not kebab:
            raise ValueError("dataset_name must contain at least one alphanumeric character.")
        if len(kebab) > 140:
            raise ValueError("dataset_name is too long (max 140 characters).")

    return start_value, end_value


def _build_dataset_command(payload: DatasetRunRequest) -> Tuple[List[str], Path, str, str]:
    start_value, end_value = _validate_payload(payload)

    dataset_name = _to_kebab_case(payload.dataset_name or "")

    if dataset_name:
        out_name = f"legacy-{dataset_name}.csv"
        drops_name = f"drops-{dataset_name}.csv"
        out_dir = _resolve_project_path(DEFAULT_OUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_name = payload.out_name or DEFAULT_OUT_NAME
        drops_name = payload.drops_name or f"{Path(out_name).stem}-drops.csv"
        dataset_dir = Path(DEFAULT_OUT_DIR) / Path(out_name).stem
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_dir = _resolve_project_path(str(dataset_dir))

    theta_url = payload.theta_base_url or "http://127.0.0.1:25503/v3"
    _ensure_theta_running(theta_url)

    cmd: List[str] = [
        sys.executable,
        str(SCRIPT_PATH),
        "--out-dir",
        str(out_dir),
        "--out-name",
        out_name,
        "--start",
        start_value,
        "--end",
        end_value,
    ]

    _add_value(cmd, "--dataset-name", dataset_name if dataset_name else None)
    if dataset_name:
        _add_value(cmd, "--run-dir-name", dataset_name)
    else:
        _add_value(cmd, "--run-dir-name", payload.run_dir_name)
    _add_value(cmd, "--schedule-mode", payload.schedule_mode)
    _add_value(cmd, "--expiry-weekdays", payload.expiry_weekdays)
    _add_value(cmd, "--asof-weekdays", payload.asof_weekdays)
    _add_value(cmd, "--dte-list", payload.dte_list)
    _add_value(cmd, "--dte-min", payload.dte_min)
    _add_value(cmd, "--dte-max", payload.dte_max)
    _add_value(cmd, "--dte-step", payload.dte_step)

    _add_optional_bool(cmd, "--write-snapshot", "--no-write-snapshot", payload.write_snapshot)
    _add_optional_bool(cmd, "--write-prn-view", "--no-write-prn-view", payload.write_prn_view)
    _add_optional_bool(cmd, "--write-train-view", "--no-write-train-view", payload.write_train_view)
    _add_optional_bool(cmd, "--write-legacy", "--no-write-legacy", payload.write_legacy)
    _add_value(cmd, "--prn-version", payload.prn_version)
    _add_value(cmd, "--prn-config-hash", payload.prn_config_hash)
    if not dataset_name:
        _add_value(cmd, "--train-view-name", payload.train_view_name)

    _add_value(cmd, "--tickers", payload.tickers)
    _add_value(cmd, "--theta-base-url", payload.theta_base_url)
    _add_value(cmd, "--stock-source", payload.stock_source)
    _add_value(cmd, "--timeout-s", payload.timeout_s)
    _add_value(cmd, "--r", payload.r)

    _add_value(cmd, "--max-abs-logm", payload.max_abs_logm)
    _add_value(cmd, "--max-abs-logm-cap", payload.max_abs_logm_cap)
    _add_value(cmd, "--band-widen-step", payload.band_widen_step)
    _add_flag(cmd, "--no-adaptive-band", payload.no_adaptive_band)
    _add_value(cmd, "--max-band-strikes", payload.max_band_strikes)

    _add_value(cmd, "--min-band-strikes", payload.min_band_strikes)
    _add_value(cmd, "--min-band-prn-strikes", payload.min_band_prn_strikes)

    _add_value(cmd, "--strike-range", payload.strike_range)
    _add_flag(cmd, "--no-retry-full-chain", payload.no_retry_full_chain)
    _add_flag(cmd, "--no-sat-expiry-fallback", payload.no_sat_expiry_fallback)
    _add_value(cmd, "--threads", payload.threads)

    _add_optional_bool(cmd, "--prefer-bidask", "--no-prefer-bidask", payload.prefer_bidask)
    _add_value(cmd, "--min-trade-count", payload.min_trade_count)
    _add_value(cmd, "--min-volume", payload.min_volume)

    _add_value(cmd, "--min-chain-used-hard", payload.min_chain_used_hard)
    _add_value(cmd, "--max-rel-spread-median-hard", payload.max_rel_spread_median_hard)
    _add_flag(cmd, "--hard-drop-close-fallback", payload.hard_drop_close_fallback)

    _add_value(cmd, "--min-prn-train", payload.min_prn_train)
    _add_value(cmd, "--max-prn-train", payload.max_prn_train)

    _add_flag(cmd, "--no-split-adjust", payload.no_split_adjust)

    _add_value(cmd, "--dividend-source", payload.dividend_source)
    _add_value(cmd, "--dividend-lookback-days", payload.dividend_lookback_days)
    _add_value(cmd, "--dividend-yield-default", payload.dividend_yield_default)
    _add_flag(cmd, "--no-forward-moneyness", payload.no_forward_moneyness)

    _add_flag(cmd, "--no-group-weights", payload.no_group_weights)
    _add_flag(cmd, "--no-ticker-weights", payload.no_ticker_weights)
    _add_flag(cmd, "--no-soft-quality-weight", payload.no_soft_quality_weight)

    _add_value(cmd, "--rv-lookback-days", payload.rv_lookback_days)

    _add_optional_bool(cmd, "--cache", "--no-cache", payload.cache)

    _add_flag(cmd, "--write-drops", payload.write_drops)
    if payload.drops_name:
        _add_value(cmd, "--drops-name", drops_name)

    _add_flag(cmd, "--sanity-report", payload.sanity_report)
    _add_flag(cmd, "--sanity-drop", payload.sanity_drop)
    _add_value(cmd, "--sanity-abs-logm-max", payload.sanity_abs_logm_max)
    _add_value(cmd, "--sanity-k-over-s-min", payload.sanity_k_over_s_min)
    _add_value(cmd, "--sanity-k-over-s-max", payload.sanity_k_over_s_max)

    _add_flag(cmd, "--verbose-skips", payload.verbose_skips)

    return cmd, out_dir, out_name, drops_name


def _build_run_response(
    *,
    ok: bool,
    out_dir: Path,
    out_name: str,
    drops_name: str,
    stdout: str,
    stderr: str,
    duration_s: float,
    command: List[str],
    write_drops: bool,
) -> DatasetRunResponse:
    output_path = out_dir / out_name
    drops_path = out_dir / drops_name if write_drops else None
    output_file = (
        str(output_path.relative_to(BASE_DIR)) if output_path.exists() else None
    )
    drops_file = (
        str(drops_path.relative_to(BASE_DIR))
        if drops_path and drops_path.exists()
        else None
    )

    return DatasetRunResponse(
        ok=ok,
        out_dir=str(out_dir.relative_to(BASE_DIR)),
        out_name=out_name,
        output_file=output_file,
        drops_file=drops_file,
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
        command=command,
    )


def _file_summary(path: Path) -> DatasetFileSummary:
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    return DatasetFileSummary(
        name=path.name,
        path=str(path.relative_to(BASE_DIR)),
        size_bytes=path.stat().st_size,
        last_modified=mtime,
    )


_PROGRESS_RE = re.compile(
    r"\[PROGRESS\]\s+(\d+)\/(\d+)\s+jobs\s+\|\s+groups_kept=(\d+)\s+\|\s+rows=(\d+)\s+\|\s+last=([A-Za-z0-9._-]+)\s+week=([0-9-]+)\s+asof_target=([0-9-]+)"
)
_OUT_LINE_RE = re.compile(r"\[OUT\]\s+base=(\S+)\s+run_dir=(\S+)")


def _extract_run_dir_from_output(stdout: str) -> Optional[Path]:
    for line in stdout.splitlines():
        match = _OUT_LINE_RE.search(line)
        if match:
            try:
                return Path(match.group(2)).resolve()
            except Exception:
                return None
    return None


def _read_training_selection(run_dir: Path) -> Optional[Dict[str, str]]:
    meta_path = run_dir / TRAINING_META_NAME
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return {str(k): str(v) for k, v in payload.items() if v is not None}

def _training_target_name(run_dir: Path) -> str:
    return f"{run_dir.name}.csv"


def _resolve_training_candidate(
    run_dir: Path,
    payload: DatasetRunRequest,
    out_name: str,
) -> Optional[Path]:
    # New convention: training-*.csv
    for tf in sorted(run_dir.glob("training-*.csv")):
        if tf.is_file():
            return tf
    # Legacy fallback
    train_view_name = (payload.train_view_name or "train_view.csv").strip() or "train_view.csv"
    legacy_name = out_name
    selection = (payload.training_dataset or "").strip().lower()
    if selection == "train_view":
        candidates = [train_view_name, legacy_name]
    elif selection == "legacy":
        candidates = [legacy_name, train_view_name]
    else:
        candidates = [train_view_name, legacy_name]
    for name in candidates:
        candidate = run_dir / name
        if candidate.exists() and _is_csv_file(candidate):
            return candidate
    for item in sorted(run_dir.iterdir()):
        if not _is_csv_file(item):
            continue
        if "drop" in item.name.lower():
            continue
        return item
    return None


def _ensure_training_file_name(
    run_dir: Path,
    payload: DatasetRunRequest,
    out_name: str,
) -> Optional[Path]:
    candidate = _resolve_training_candidate(run_dir, payload, out_name)
    if candidate is None or not candidate.exists():
        return None
    # New convention files are already correctly named
    if candidate.name.startswith("training-"):
        return candidate
    # Legacy: rename to {run_dir.name}.csv
    target_name = _training_target_name(run_dir)
    target_path = run_dir / target_name
    if candidate.name == target_name:
        return candidate
    if target_path.exists():
        return target_path
    try:
        candidate.rename(target_path)
        return target_path
    except Exception:
        return candidate


def _write_training_selection(run_dir: Path, payload: DatasetRunRequest, out_name: str) -> None:
    try:
        dataset_name = _to_kebab_case(payload.dataset_name or "")
        training_path = _ensure_training_file_name(run_dir, payload, out_name)
        training_file = training_path.name if training_path else None
        meta: Dict[str, Optional[str]] = {
            "training_file": training_file,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if dataset_name:
            meta["dataset_name"] = dataset_name
        else:
            train_view_name = (payload.train_view_name or "train_view.csv").strip() or "train_view.csv"
            selection = (payload.training_dataset or "").strip().lower()
            if selection not in {"legacy", "train_view"}:
                selection = ""
            meta["training_dataset"] = selection or None
            meta["train_view_file"] = train_view_name
            meta["legacy_file"] = out_name
        (run_dir / TRAINING_META_NAME).write_text(json.dumps(meta, indent=2))
    except Exception:
        return


def _is_csv_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".csv"


def _select_training_file(
    run_dir: Path,
    dataset_file: Optional[DatasetFileSummary],
) -> Optional[DatasetFileSummary]:
    # New convention: training-*.csv
    for tf in sorted(run_dir.glob("training-*.csv")):
        if tf.is_file():
            return _file_summary(tf)
    # Legacy: {run_dir.name}.csv
    target = run_dir / _training_target_name(run_dir)
    if target.exists() and _is_csv_file(target):
        return _file_summary(target)
    meta = _read_training_selection(run_dir)
    if meta:
        training_name = meta.get("training_file")
        if training_name:
            candidate = run_dir / training_name
            if candidate.exists() and _is_csv_file(candidate):
                return _file_summary(candidate)

    default_train_view = run_dir / "train_view.csv"
    if default_train_view.exists():
        return _file_summary(default_train_view)

    return dataset_file


def _collect_run_files(
    run_dir: Path,
) -> Tuple[
    Optional[DatasetFileSummary],
    Optional[DatasetFileSummary],
    Optional[DatasetFileSummary],
    List[DatasetFileSummary],
]:
    dataset_file: Optional[DatasetFileSummary] = None
    drops_file: Optional[DatasetFileSummary] = None
    files: List[DatasetFileSummary] = []
    for item in sorted(run_dir.iterdir()):
        if not _is_csv_file(item):
            continue
        summary = _file_summary(item)
        files.append(summary)
        lower_name = item.name.lower()
        if "drop" in lower_name:
            if drops_file is None:
                drops_file = summary
            continue
        if dataset_file is None:
            dataset_file = summary
    training_file = _select_training_file(run_dir, dataset_file)
    return dataset_file, drops_file, training_file, files


def _run_summary_from_dir(run_dir: Path) -> DatasetRunSummary:
    dataset_file, drops_file, training_file, files = _collect_run_files(run_dir)
    last_modified = datetime.fromtimestamp(run_dir.stat().st_mtime, tz=timezone.utc).isoformat()
    relative_path = str(run_dir.relative_to(BASE_DIR))
    return DatasetRunSummary(
        id=relative_path,
        run_dir=relative_path,
        dataset_file=dataset_file,
        drops_file=drops_file,
        training_file=training_file,
        files=files,
        last_modified=last_modified,
    )


def _iter_run_dirs() -> List[Path]:
    run_dirs: List[Path] = []
    seen: Set[Path] = set()
    for base_dir in _dataset_base_dirs():
        if not base_dir.exists():
            continue
        for dataset_dir in sorted(base_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            found_run = False
            for run_dir in sorted(dataset_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                resolved_run = run_dir.resolve()
                if resolved_run in seen:
                    continue
                seen.add(resolved_run)
                run_dirs.append(run_dir)
                found_run = True
            if not found_run:
                resolved_dataset = dataset_dir.resolve()
                if resolved_dataset in seen:
                    continue
                seen.add(resolved_dataset)
                run_dirs.append(dataset_dir)
    return run_dirs


def list_dataset_runs() -> DatasetListResponse:
    run_dirs = _iter_run_dirs()
    runs = [_run_summary_from_dir(run_dir) for run_dir in run_dirs]
    runs.sort(key=lambda entry: entry.last_modified or "", reverse=True)
    base_dir_path = _dataset_display_base_dir()
    return DatasetListResponse(
        base_dir=str(base_dir_path.relative_to(BASE_DIR)),
        runs=runs,
    )


def _resolve_dataset_directory(path_value: str) -> Path:
    path = _resolve_project_path(path_value)
    _find_dataset_base_for_path(path)
    if not path.exists():
        raise KeyError(path_value)
    if not path.is_dir():
        raise ValueError("Run path must point to a directory.")
    return path


def _resolve_dataset_file(path_value: str) -> Path:
    path = _resolve_project_path(path_value)
    _find_dataset_base_for_path(path)
    return path

def get_dataset_file_path(path_value: str) -> Path:
    path = _resolve_dataset_file(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError(f"File not found: {path}")
    return path


def delete_dataset_run(run_dir_path: str) -> DatasetRunSummary:
    target_dir = _resolve_dataset_directory(run_dir_path)
    summary = _run_summary_from_dir(target_dir)
    shutil.rmtree(target_dir)
    # Clean up empty parent directory (legacy nested structure)
    parent = target_dir.parent
    base_dirs = {b.resolve() for b in DATASET_BASE_DIRS}
    if parent.resolve() not in base_dirs and parent.exists() and parent.is_dir():
        try:
            remaining = list(parent.iterdir())
            if not remaining:
                parent.rmdir()
        except Exception:
            pass
    return summary

_CSV_PURPOSE_PREFIXES = ("training-", "snapshot-", "prn-view-", "legacy-", "drops-")


def _rename_csv_files(directory: Path, new_kebab_name: str) -> None:
    """Rename all CSVs in directory to follow {purpose}-{new_kebab_name}.csv convention."""
    for item in sorted(directory.iterdir()):
        if not _is_csv_file(item):
            continue
        lowered = item.name.lower()
        # New convention: files matching {purpose}-*.csv get their suffix updated
        matched_prefix = None
        for prefix in _CSV_PURPOSE_PREFIXES:
            if lowered.startswith(prefix):
                matched_prefix = prefix
                break
        if matched_prefix:
            new_name = f"{matched_prefix}{new_kebab_name}.csv"
            if item.name != new_name:
                target = directory / new_name
                if not target.exists():
                    try:
                        item.rename(target)
                    except Exception:
                        pass
            continue
        # Legacy files: try to identify purpose and rename
        if lowered == "train_view.csv":
            new_name = f"training-{new_kebab_name}.csv"
        elif lowered == "snapshot.csv":
            new_name = f"snapshot-{new_kebab_name}.csv"
        elif lowered == "prn_view.csv":
            new_name = f"prn-view-{new_kebab_name}.csv"
        elif "drop" in lowered:
            new_name = f"drops-{new_kebab_name}.csv"
        elif lowered == f"{directory.name.lower()}.csv":
            # Legacy renamed training file ({old_dir_name}.csv)
            new_name = f"training-{new_kebab_name}.csv"
        else:
            new_name = f"legacy-{new_kebab_name}.csv"
        if item.name != new_name:
            target = directory / new_name
            if not target.exists():
                try:
                    item.rename(target)
                except Exception:
                    pass


def rename_dataset_run(run_dir_path: str, new_name: str) -> DatasetRunSummary:
    target_dir = _resolve_dataset_directory(run_dir_path)
    sanitized = Path(new_name).name.strip()
    if not sanitized:
        raise ValueError("New directory name cannot be empty.")
    if sanitized != new_name:
        raise ValueError("New directory name must not include path separators.")
    if sanitized.lower().endswith(".csv"):
        raise ValueError("Directory name must not include a .csv extension.")
    if sanitized in {".", ".."}:
        raise ValueError("Invalid directory name.")

    new_kebab = _to_kebab_case(sanitized)
    if not new_kebab:
        raise ValueError("Name must contain at least one alphanumeric character.")

    parent_dir = target_dir.parent
    new_dir = parent_dir / new_kebab
    if new_dir.exists() and new_dir != target_dir:
        raise ValueError("Target directory already exists.")

    # Rename CSVs inside the directory first (while directory still has old name)
    _rename_csv_files(target_dir, new_kebab)

    # Rename the directory itself
    if new_dir != target_dir:
        target_dir.rename(new_dir)

    # Update training metadata
    _dataset_file, _drops_file, training_file, _files = _collect_run_files(new_dir)
    meta: Dict[str, Optional[str]] = {
        "dataset_name": new_kebab,
        "training_file": training_file.name if training_file else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (new_dir / TRAINING_META_NAME).write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    return _run_summary_from_dir(new_dir)


def _read_csv_head(path: Path, limit: int) -> Tuple[List[str], List[Dict[str, Optional[str]]]]:
    rows: List[Dict[str, Optional[str]]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append({key: row.get(key) for key in headers})
    return headers, rows


def _read_csv_tail(path: Path, limit: int) -> Tuple[List[str], List[Dict[str, Optional[str]]], int]:
    buffer: deque[Dict[str, Optional[str]]] = deque(maxlen=limit)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        row_count = 0
        for row in reader:
            row_count += 1
            buffer.append({key: row.get(key) for key in headers})
    return headers, list(buffer), row_count


def preview_dataset_file(
    path_value: str,
    *,
    limit: int = 20,
    mode: str = "head",
) -> DatasetPreviewResponse:
    path = get_dataset_file_path(path_value)
    sanitized_limit = max(1, min(limit, 100))
    normalized_mode = mode.lower()
    if normalized_mode not in {"head", "tail"}:
        raise ValueError("mode must be 'head' or 'tail'")

    if normalized_mode == "tail":
        headers, rows, row_count = _read_csv_tail(path, sanitized_limit)
    else:
        headers, rows = _read_csv_head(path, sanitized_limit)
        row_count = None

    return DatasetPreviewResponse(
        file=_file_summary(path),
        headers=headers,
        rows=rows,
        row_count=row_count,
        mode=normalized_mode,
        limit=sanitized_limit,
    )


def run_dataset(payload: DatasetRunRequest) -> DatasetRunResponse:
    cmd, out_dir, out_name, drops_name = _build_dataset_command(payload)

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    duration_s = round(time.monotonic() - start, 3)

    if result.returncode == 0:
        run_dir = _extract_run_dir_from_output(result.stdout)
        if run_dir and run_dir.exists():
            _write_training_selection(run_dir, payload, out_name)

    return _build_run_response(
        ok=result.returncode == 0,
        out_dir=out_dir,
        out_name=out_name,
        drops_name=drops_name,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=duration_s,
        command=cmd,
        write_drops=bool(payload.write_drops),
    )



class DatasetJob:
    LOG_LIMIT = 500

    def __init__(self, job_id: str, payload: DatasetRunRequest):
        self.job_id = job_id
        self.payload = payload
        self.status = "queued"
        self.progress: Optional[DatasetJobProgress] = None
        self.stdout_lines: Deque[str] = deque(maxlen=self.LOG_LIMIT)
        self.stderr_lines: Deque[str] = deque(maxlen=self.LOG_LIMIT)
        self.result: Optional[DatasetRunResponse] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._cancel_requested = False
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._out_dir: Optional[Path] = None
        self.run_dir_path: Optional[Path] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancel_requested = True
        proc = self._proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()

    def to_status(self) -> DatasetJobStatus:
        return DatasetJobStatus(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            stdout=list(self.stdout_lines),
            stderr=list(self.stderr_lines),
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    def _run(self) -> None:
        try:
            cmd, out_dir, out_name, drops_name = _build_dataset_command(self.payload)
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
            self.finished_at = datetime.utcnow()
            return

        self._out_dir = out_dir
        self.started_at = datetime.utcnow()
        self.status = "running"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        start = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        self._proc = proc

        lines_queue: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()

        def _reader(stream, kind: str) -> None:
            for line in iter(stream.readline, ""):
                lines_queue.put((kind, line))
            lines_queue.put((kind, None))

        threads = [
            threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True),
            threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()

        done_streams = 0
        while done_streams < 2:
            kind, line = lines_queue.get()
            if line is None:
                done_streams += 1
                continue
            self._record_line(kind, line)
            if self._cancel_requested and proc.poll() is None:
                proc.terminate()
        for thread in threads:
            thread.join()

        return_code = proc.wait()
        duration_s = round(time.monotonic() - start, 3)
        self.result = _build_run_response(
            ok=return_code == 0,
            out_dir=out_dir,
            out_name=out_name,
            drops_name=drops_name,
            stdout="".join(self.stdout_lines),
            stderr="".join(self.stderr_lines),
            duration_s=duration_s,
            command=cmd,
            write_drops=bool(self.payload.write_drops),
        )
        self.finished_at = datetime.utcnow()

        if self._cancel_requested:
            self.status = "cancelled"
            self._cleanup_run_dir()
        else:
            self.status = "finished" if return_code == 0 else "failed"
            if self.status == "failed":
                self.error = (self.result.stderr or "").strip() or "Dataset run failed."
            elif self.status == "finished":
                run_dir = self.run_dir_path
                if run_dir is None:
                    run_dir = _extract_run_dir_from_output(self.result.stdout or "")
                if run_dir and run_dir.exists():
                    _write_training_selection(run_dir, self.payload, out_name)

    def _record_line(self, kind: str, line: str) -> None:
        target = self.stdout_lines if kind == "stdout" else self.stderr_lines
        target.append(line)
        clean_line = line.strip()
        self._parse_progress(clean_line)
        self._capture_run_dir(clean_line)

    def _parse_progress(self, line: str) -> None:
        match = _PROGRESS_RE.search(line)
        if not match:
            return
        self.progress = DatasetJobProgress(
            done=int(match.group(1)),
            total=int(match.group(2)),
            groups=int(match.group(3)),
            rows=int(match.group(4)),
            lastTicker=match.group(5),
            lastWeek=match.group(6),
            lastAsof=match.group(7),
        )

    def _capture_run_dir(self, line: str) -> None:
        match = _OUT_LINE_RE.search(line)
        if not match:
            return
        try:
            self.run_dir_path = Path(match.group(2)).resolve()
        except Exception:
            self.run_dir_path = None

    def _cleanup_run_dir(self) -> None:
        if not self.run_dir_path or not self._out_dir:
            return
        try:
            resolved_run_dir = self.run_dir_path.resolve()
            resolved_base = self._out_dir.resolve()
            resolved_run_dir.relative_to(resolved_base)
        except Exception:
            return
        if resolved_run_dir.exists():
            shutil.rmtree(resolved_run_dir, ignore_errors=True)


class DatasetJobManager:
    def __init__(self):
        self._jobs: Dict[str, DatasetJob] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: DatasetRunRequest) -> str:
        job_id = uuid4().hex
        job = DatasetJob(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> DatasetJobStatus:
        job = self._get_job(job_id)
        return job.to_status()

    def list_jobs(self) -> List[DatasetJobStatus]:
        with self._lock:
            return [job.to_status() for job in self._jobs.values()]

    def cancel_job(self, job_id: str) -> DatasetJobStatus:
        job = self._get_job(job_id)
        job.cancel()
        return job.to_status()

    def _get_job(self, job_id: str) -> DatasetJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


JOB_MANAGER = DatasetJobManager()


def start_dataset_job(payload: DatasetRunRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    return JOB_MANAGER.start_job(payload)


def get_dataset_job(job_id: str) -> DatasetJobStatus:
    return JOB_MANAGER.get_status(job_id)


def cancel_dataset_job(job_id: str) -> DatasetJobStatus:
    return JOB_MANAGER.cancel_job(job_id)
