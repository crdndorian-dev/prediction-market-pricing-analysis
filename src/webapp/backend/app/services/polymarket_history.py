from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.models.polymarket_history import (
    PolymarketHistoryProgress,
    PolymarketHistoryJobStatus,
    PolymarketHistoryRunRequest,
    PolymarketHistoryRunResponse,
    PolymarketRunFeaturesRequest,
    PolymarketRunFeaturesResponse,
)
from app.services.process_runtime import (
    ManagedProcessHandle,
    spawn_managed_process,
    terminate_managed_process,
)

BASE_DIR = Path(__file__).resolve().parents[5]
SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "02-polymarket-weekly-history-v1.0.py"
FEATURES_SCRIPT_PATH = BASE_DIR / "src" / "scripts" / "02-polymarket-build-features-v1.0.py"
DEFAULT_OUT_DIR = BASE_DIR / "src" / "data" / "raw" / "polymarket" / "weekly_history"
RUNS_DIR = DEFAULT_OUT_DIR / "runs"
DEFAULT_EVENT_URLS_FILE = BASE_DIR / "config" / "polymarket_event_urls.csv"
DIM_MARKET_WEEKLY_PATH = BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market_weekly.csv"
DEFAULT_BARS_DIR = BASE_DIR / "src" / "data" / "analysis" / "polymarket" / "bars_history"
ENV_FILE = BASE_DIR / ".env"
ENV_SAMPLE_FILE = BASE_DIR / "config" / "polymarket_subgraph.env.sample"
MAX_RUN_DIR_NAME_LEN = 140
LEGACY_DECISION_FEATURES_CSV = "decision_features.csv"

_HISTORY_COMPLETE_RE = re.compile(
    r"\[Weekly History\] Market complete (?P<current>\d+)/(?P<total>\d+)\s+job_id=(?P<job_id>[^\s]+)\s+status=(?P<status>\w+)"
)
_FEATURE_PROGRESS_RE = re.compile(
    r"\[features\] PROGRESS (?P<current>\d+)/(?P<total>\d+)\s+step=(?P<step>[^\s]+)"
)


class JobProgressTracker:
    def __init__(self) -> None:
        self._total: Optional[int] = None
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._failure_flag = False
        self._lock = threading.Lock()

    def set_total(self, total: int) -> None:
        if total <= 0:
            return
        with self._lock:
            if self._total is None or total > self._total:
                self._total = total

    def mark_completed(self, job_id: str, *, failed: bool = False) -> None:
        if not job_id:
            return
        with self._lock:
            if job_id in self._completed:
                return
            self._completed.add(job_id)
            if failed:
                self._failed.add(job_id)

    def mark_failed(self, job_id: Optional[str] = None) -> None:
        if job_id:
            self.mark_completed(job_id, failed=True)
            return
        with self._lock:
            self._failure_flag = True

    def snapshot(self) -> Optional[PolymarketHistoryProgress]:
        with self._lock:
            if not self._total:
                return None
            completed = min(len(self._completed), self._total)
            failed = max(len(self._failed), 1 if self._failure_flag else 0)
            failed = min(failed, completed)
            status = "running"
            if completed >= self._total:
                status = "failed" if failed > 0 else "completed"
            return PolymarketHistoryProgress(
                total=self._total,
                completed=completed,
                failed=failed,
                status=status,
            )


def _load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    env: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('\"')
        if key:
            env[key] = value
    return env


def _apply_subgraph_env(env: Dict[str, str]) -> None:
    if ENV_FILE.exists():
        env.update(_load_env_file(ENV_FILE))
        return
    if ENV_SAMPLE_FILE.exists():
        env.update(_load_env_file(ENV_SAMPLE_FILE))


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


def _normalize_tickers(tickers: Optional[List[str]]) -> Optional[List[str]]:
    if tickers is None:
        return None
    cleaned = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
    if not cleaned:
        raise ValueError("Tickers list is empty after cleaning.")
    return cleaned


def _parse_run_id(stdout: str) -> Optional[str]:
    for line in stdout.splitlines():
        if "run_id=" not in line:
            continue
        match = re.search(r"run_id=([0-9A-Za-z_-]+)", line)
        if match:
            return match.group(1)
    return None


def _to_kebab_case(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    raw = re.sub(r"[\s_]+", "-", raw)
    raw = re.sub(r"[^a-zA-Z0-9-]", "", raw)
    raw = re.sub(r"-{2,}", "-", raw)
    return raw.strip("-").lower()


def _sanitize_run_dir_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw != Path(raw).name or raw in {".", ".."}:
        raise ValueError("run_dir_name must be a single directory name (no path separators).")
    kebab = _to_kebab_case(raw)
    if not kebab:
        raise ValueError("run_dir_name must contain at least one alphanumeric character.")
    if len(kebab) > MAX_RUN_DIR_NAME_LEN:
        raise ValueError(f"run_dir_name is too long (max {MAX_RUN_DIR_NAME_LEN} characters).")
    return kebab


def _list_run_file_names(run_dir: Path) -> List[str]:
    if not run_dir.exists():
        return []
    return sorted([item.name for item in run_dir.iterdir() if item.is_file()])


def _decision_features_prefixed_csv_name(run_dir: Path) -> str:
    return f"{run_dir.name}-decision-features.csv"


def _decision_features_csv_candidates(run_dir: Path) -> List[Path]:
    return [
        run_dir / _decision_features_prefixed_csv_name(run_dir),
        run_dir / LEGACY_DECISION_FEATURES_CSV,
    ]


def _find_existing_decision_features_csv(run_dir: Path) -> Optional[Path]:
    for candidate in _decision_features_csv_candidates(run_dir):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _write_event_urls_file(out_dir: Path, event_urls: List[str]) -> Path:
    tmp_dir = out_dir / "event_sources"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"event_urls_{uuid4().hex}.txt"
    path.write_text("\n".join(event_urls))
    return path


def _build_history_command(
    payload: PolymarketHistoryRunRequest,
) -> tuple[List[str], Dict[str, str], Path]:
    if not SCRIPT_PATH.exists():
        raise RuntimeError(f"Weekly history script not found at {SCRIPT_PATH}")

    out_dir = _resolve_project_path(str(payload.out_dir)) if payload.out_dir else DEFAULT_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [sys.executable, str(SCRIPT_PATH), "--out-dir", str(out_dir)]
    sanitized_run_dir_name = _sanitize_run_dir_name(payload.run_dir_name)
    if sanitized_run_dir_name:
        candidate_run_dir = out_dir / "runs" / sanitized_run_dir_name
        if candidate_run_dir.exists():
            raise ValueError(f"Run directory already exists: {candidate_run_dir.relative_to(BASE_DIR)}")
        payload.run_dir_name = sanitized_run_dir_name
        cmd.extend(["--run-id", sanitized_run_dir_name])

    tickers = _normalize_tickers(payload.tickers)
    if tickers:
        cmd.extend(["--tickers", ",".join(tickers)])

    if payload.tickers_csv:
        tickers_csv = _resolve_project_path(payload.tickers_csv)
        if not tickers_csv.exists():
            raise ValueError(f"tickers_csv not found: {tickers_csv}")
        cmd.extend(["--tickers-csv", str(tickers_csv)])

    event_urls_used = False
    if payload.event_urls:
        cleaned = [value.strip() for value in payload.event_urls if value and value.strip()]
        if cleaned:
            event_urls_file = _write_event_urls_file(out_dir, cleaned)
            cmd.extend(["--event-urls-file", str(event_urls_file)])
            event_urls_used = True

    if payload.event_urls_file:
        event_urls_file = _resolve_project_path(payload.event_urls_file)
        if not event_urls_file.exists():
            raise ValueError(f"event_urls_file not found: {event_urls_file}")
        cmd.extend(["--event-urls-file", str(event_urls_file)])
        event_urls_used = True

    if not event_urls_used and DEFAULT_EVENT_URLS_FILE.exists():
        cmd.extend(["--event-urls-file", str(DEFAULT_EVENT_URLS_FILE)])

    if payload.start_date:
        cmd.extend(["--start-date", payload.start_date])
    if payload.end_date:
        cmd.extend(["--end-date", payload.end_date])

    if payload.fidelity_min is not None:
        cmd.extend(["--fidelity-min", str(payload.fidelity_min)])

    if payload.bars_freqs:
        cmd.extend(["--bars-freqs", payload.bars_freqs])

    if payload.bars_dir:
        bars_dir = _resolve_project_path(payload.bars_dir)
        cmd.extend(["--bars-dir", str(bars_dir)])

    if payload.dim_market_out:
        dim_path = _resolve_project_path(payload.dim_market_out)
        cmd.extend(["--dim-market-out", str(dim_path)])

    if payload.fact_trade_dir:
        fact_dir = _resolve_project_path(payload.fact_trade_dir)
        cmd.extend(["--fact-trade-dir", str(fact_dir)])

    if payload.include_subgraph:
        cmd.append("--include-subgraph")

    if payload.max_subgraph_entities is not None:
        cmd.extend(["--max-subgraph-entities", str(payload.max_subgraph_entities)])

    if payload.dry_run:
        cmd.append("--dry-run")

    env = {**os.environ}
    _apply_subgraph_env(env)
    existing = env.get("PYTHONPATH")
    root = str(BASE_DIR)
    if existing:
        env["PYTHONPATH"] = os.pathsep.join([existing, root])
    else:
        env["PYTHONPATH"] = root
    return cmd, env, out_dir


def _collect_run_files(out_dir: Path, run_id: Optional[str]) -> tuple[Optional[Path], List[str]]:
    run_dir: Optional[Path] = None
    if run_id:
        candidate = out_dir / "runs" / run_id
        if candidate.exists():
            run_dir = candidate

    files: List[str] = []
    if run_dir and run_dir.exists():
        files = _list_run_file_names(run_dir)
    return run_dir, files


# ---------------------------------------------------------------------------
# Run management helpers (Phase 0 + 1)
# ---------------------------------------------------------------------------


def _safe_json_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _load_run_manifest(run_dir: Path) -> Dict[str, Any]:
    return _safe_json_load(run_dir / "manifest.json") or {}


def _resolve_manifest_path(value: Optional[str], *, fallback: Path, label: str, must_exist: bool = True) -> Path:
    path = _resolve_project_path(value) if value else fallback
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_manifest_date(manifest: Dict[str, Any], key: str) -> Optional[str]:
    value = manifest.get(key)
    if value:
        return value
    pipeline_args = manifest.get("pipeline_args")
    if isinstance(pipeline_args, dict):
        value = pipeline_args.get(key)
    return value if value else None


def _expand_dim_market_candidates(path: Path) -> List[Path]:
    candidates = [path]
    suffix = path.suffix.lower()
    if suffix == ".csv":
        candidates.append(path.with_suffix(".parquet"))
    elif suffix == ".parquet":
        candidates.append(path.with_suffix(".csv"))
    return candidates


def _find_dim_market_for_run(run_dir: Path, manifest: Dict[str, Any]) -> Path:
    candidates: List[Path] = []

    raw_value = manifest.get("dim_market") or manifest.get("dim_market_out")
    if raw_value:
        try:
            resolved = _resolve_project_path(str(raw_value))
            candidates.extend(_expand_dim_market_candidates(resolved))
        except ValueError:
            path = Path(str(raw_value)).expanduser()
            candidates.extend(_expand_dim_market_candidates(path))

    candidates.extend(
        [
            run_dir / "dim_market_weekly.csv",
            run_dir / f"{run_dir.name}-dim-market-weekly.csv",
            run_dir / "dim_market.csv",
            run_dir / f"{run_dir.name}-dim-market.csv",
            run_dir / "dim_market.parquet",
            run_dir / f"{run_dir.name}-dim-market.parquet",
            run_dir / f"{run_dir.name}-dim-market-weekly.parquet",
        ]
    )

    candidates.extend(
        [
            DIM_MARKET_WEEKLY_PATH,
            DIM_MARKET_WEEKLY_PATH.with_suffix(".parquet"),
            BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market.parquet",
            BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market.csv",
        ]
    )

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    details = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"dim_market not found. Checked: {details}")


def _atomic_json_write(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically via temp + rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


def _artifact_entry(file_path: Path) -> Dict[str, Any]:
    """Build an artifact inventory entry for a file."""
    entry: Dict[str, Any] = {"size_bytes": file_path.stat().st_size}
    if file_path.suffix.lower() == ".csv":
        try:
            with file_path.open(newline="") as handle:
                entry["rows"] = sum(1 for _ in handle) - 1  # subtract header
        except Exception:
            pass
    return entry


def _build_artifact_inventory(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan run dir and build artifact inventory."""
    inventory: Dict[str, Dict[str, Any]] = {}
    if not run_dir or not run_dir.exists():
        return inventory
    for item in run_dir.iterdir():
        if item.is_file() and item.name != "manifest.json":
            inventory[item.name] = _artifact_entry(item)
    return inventory


def _enhance_manifest(
    run_dir: Path,
    *,
    status: str,
    duration_s: float,
    pipeline_args: Optional[Dict[str, Any]] = None,
    error_summary: Optional[str] = None,
    features_built: bool = False,
) -> None:
    """Enrich the existing manifest.json with run management metadata."""
    manifest_path = run_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}

    manifest["status"] = status
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["duration_s"] = duration_s
    manifest.setdefault("label", None)
    manifest.setdefault("pinned", False)

    if pipeline_args:
        manifest["pipeline_args"] = pipeline_args
    if error_summary:
        manifest["error_summary"] = error_summary

    manifest["features_built"] = features_built
    manifest["artifacts"] = _build_artifact_inventory(run_dir)

    _atomic_json_write(manifest_path, manifest)


def _update_features_manifest(run_dir: Path, *, features_built: bool) -> None:
    manifest_path = run_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}
    manifest["features_built"] = features_built
    manifest["artifacts"] = _build_artifact_inventory(run_dir)
    _atomic_json_write(manifest_path, manifest)


def _update_latest_pointer(run_id: str) -> None:
    """Write latest.json pointing to the given run, atomically."""
    latest_path = DEFAULT_OUT_DIR / "latest.json"
    payload = {
        "run_id": run_id,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_by": "pipeline",
    }
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(latest_path, payload)


def _copy_dim_market_to_run(run_dir: Path) -> None:
    """Copy dim_market_weekly.csv into the run dir for provenance."""
    if DIM_MARKET_WEEKLY_PATH.exists() and run_dir and run_dir.exists():
        dest = run_dir / "dim_market_weekly.csv"
        if not dest.exists():
            try:
                shutil.copy2(DIM_MARKET_WEEKLY_PATH, dest)
            except OSError:
                pass


def _pipeline_args_from_payload(payload: PolymarketHistoryRunRequest) -> Dict[str, Any]:
    """Extract pipeline args from the request payload for manifest storage."""
    args: Dict[str, Any] = {}
    if payload.tickers:
        args["tickers"] = payload.tickers
    if payload.start_date:
        args["start_date"] = payload.start_date
    if payload.end_date:
        args["end_date"] = payload.end_date
    if payload.fidelity_min is not None:
        args["fidelity_min"] = payload.fidelity_min
    if payload.bars_freqs:
        args["bars_freqs"] = payload.bars_freqs
    if payload.run_dir_name:
        sanitized_run_dir_name = _sanitize_run_dir_name(payload.run_dir_name)
        if sanitized_run_dir_name:
            args["run_dir_name"] = sanitized_run_dir_name
    args["include_subgraph"] = payload.include_subgraph
    args["build_features"] = payload.build_features
    if payload.prn_dataset:
        args["prn_dataset"] = payload.prn_dataset
    args["skip_subgraph_labels"] = payload.skip_subgraph_labels
    return args


def get_latest_pointer() -> Optional[Dict[str, Any]]:
    """Read the latest.json pointer file."""
    return _safe_json_load(DEFAULT_OUT_DIR / "latest.json")


def get_latest_run_id() -> Optional[str]:
    """Return the run_id from latest.json, or None."""
    pointer = get_latest_pointer()
    if pointer:
        return pointer.get("run_id")
    return None


def _clear_latest_pointer_if_matches(run_id: str) -> None:
    latest_path = DEFAULT_OUT_DIR / "latest.json"
    pointer = _safe_json_load(latest_path)
    if not pointer or pointer.get("run_id") != run_id:
        return
    try:
        latest_path.unlink()
    except FileNotFoundError:
        return


def _ensure_run_features_csv(run_dir: Path) -> Optional[Path]:
    existing_csv = _find_existing_decision_features_csv(run_dir)
    if existing_csv:
        return existing_csv

    csv_path = run_dir / _decision_features_prefixed_csv_name(run_dir)
    parquet_path = run_dir / "decision_features.parquet"
    if not parquet_path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        df.to_csv(csv_path, index=False)
    except Exception:
        return None
    return csv_path if csv_path.exists() else None


def _is_valid_run_filename(filename: str) -> bool:
    return bool(filename) and "/" not in filename and "\\" not in filename and ".." not in filename


def _resolve_run_file_path(run_dir: Path, filename: str) -> Path:
    """Resolve a safe file path under a run dir, with legacy/new decision_features.csv aliases."""
    if not _is_valid_run_filename(filename):
        raise ValueError("Invalid filename.")
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_dir.name}")

    prefixed_decision_name = _decision_features_prefixed_csv_name(run_dir)
    if filename in {LEGACY_DECISION_FEATURES_CSV, prefixed_decision_name}:
        _ensure_run_features_csv(run_dir)
        direct = run_dir / filename
        if direct.exists() and direct.is_file():
            return direct
        existing_decision_csv = _find_existing_decision_features_csv(run_dir)
        if existing_decision_csv and existing_decision_csv.exists():
            return existing_decision_csv

    file_path = run_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File not found in run directory: {filename}")
    return file_path


def _rename_run_csv_files(run_dir: Path) -> None:
    """Normalize top-level run CSV names to {run_dir}-{csv-type}.csv without failing the job."""
    if not run_dir.exists() or not run_dir.is_dir():
        return

    prefix = f"{run_dir.name}-"
    for item in sorted(run_dir.iterdir(), key=lambda path: path.name):
        if not item.is_file() or item.suffix.lower() != ".csv":
            continue
        if item.name.startswith(prefix):
            continue

        normalized_stem = _to_kebab_case(item.stem) or "csv"
        target_name = f"{prefix}{normalized_stem}.csv"
        if item.name == target_name:
            continue

        target_path = run_dir / target_name
        if target_path.exists():
            continue
        try:
            item.rename(target_path)
        except OSError:
            continue


def _resolve_features_output_path(run_dir: Path) -> Optional[Path]:
    features_csv = _find_existing_decision_features_csv(run_dir)
    if features_csv:
        return features_csv
    parquet_path = run_dir / "decision_features.parquet"
    if parquet_path.exists():
        return parquet_path
    return None


def _build_run_csv_files(run_dir: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    csv_entries: Dict[str, Dict[str, Any]] = {}

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        for name, meta in artifacts.items():
            if not isinstance(name, str) or not name.lower().endswith(".csv"):
                continue
            if isinstance(meta, dict):
                size_bytes = meta.get("size_bytes")
                row_count = meta.get("rows")
            else:
                size_bytes = None
                row_count = None
            entry: Dict[str, Any] = {"name": name, "size_bytes": int(size_bytes) if isinstance(size_bytes, int) else 0}
            if isinstance(row_count, int):
                entry["row_count"] = max(0, row_count)
            csv_entries[name] = entry

    for item in sorted(run_dir.iterdir(), key=lambda path: path.name):
        if not item.is_file() or item.suffix.lower() != ".csv":
            continue
        existing = csv_entries.get(item.name)
        row_count = existing.get("row_count") if existing else None
        entry: Dict[str, Any] = {
            "name": item.name,
            "size_bytes": item.stat().st_size,
        }
        if isinstance(row_count, int):
            entry["row_count"] = row_count
        csv_entries[item.name] = entry

    parquet_path = run_dir / "decision_features.parquet"
    if parquet_path.exists() and _find_existing_decision_features_csv(run_dir) is None:
        synthetic_name = _decision_features_prefixed_csv_name(run_dir)
        csv_entries.setdefault(
            synthetic_name,
            {
                "name": synthetic_name,
                "size_bytes": 0,
            },
        )

    return [csv_entries[name] for name in sorted(csv_entries)]


def build_run_decision_features(
    run_id: str,
    payload: PolymarketRunFeaturesRequest,
) -> PolymarketRunFeaturesResponse:
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")

    start = time.monotonic()
    existing_csv = _find_existing_decision_features_csv(run_dir)
    if existing_csv:
        features_manifest_path = run_dir / "feature_manifest.json"
        _update_features_manifest(run_dir, features_built=True)
        return PolymarketRunFeaturesResponse(
            ok=True,
            run_id=run_id,
            run_dir=str(run_dir.relative_to(BASE_DIR)),
            features_built=True,
            features_path=str(existing_csv.relative_to(run_dir)),
            features_manifest_path=(
                str(features_manifest_path.relative_to(run_dir))
                if features_manifest_path.exists()
                else None
            ),
            stdout="Decision features already present.",
            stderr="",
            duration_s=round(time.monotonic() - start, 3),
            command=[],
        )

    generated_csv = _ensure_run_features_csv(run_dir)
    if generated_csv:
        features_manifest_path = run_dir / "feature_manifest.json"
        _update_features_manifest(run_dir, features_built=True)
        return PolymarketRunFeaturesResponse(
            ok=True,
            run_id=run_id,
            run_dir=str(run_dir.relative_to(BASE_DIR)),
            features_built=True,
            features_path=str(generated_csv.relative_to(run_dir)),
            features_manifest_path=(
                str(features_manifest_path.relative_to(run_dir))
                if features_manifest_path.exists()
                else None
            ),
            stdout="Generated decision features CSV from existing parquet output.",
            stderr="",
            duration_s=round(time.monotonic() - start, 3),
            command=[],
        )

    manifest = _load_run_manifest(run_dir)
    bars_dir_value = manifest.get("bars_dir")
    bars_dir = _resolve_manifest_path(
        str(bars_dir_value) if bars_dir_value else None,
        fallback=DEFAULT_BARS_DIR,
        label="bars_dir",
    )
    if not bars_dir.is_dir():
        raise ValueError(f"bars_dir must be a directory: {bars_dir}")
    dim_market_path = _find_dim_market_for_run(run_dir, manifest)

    start_date = _resolve_manifest_date(manifest, "start_date")
    end_date = _resolve_manifest_date(manifest, "end_date")

    features_payload = PolymarketHistoryRunRequest(
        start_date=start_date,
        end_date=end_date,
        prn_dataset=payload.prn_dataset,
        skip_subgraph_labels=payload.skip_subgraph_labels,
    )
    cmd, env = _build_features_command(
        features_payload,
        bars_dir,
        dim_market_path,
        run_dir,
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    duration_s = round(time.monotonic() - start, 3)

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    if result.returncode != 0:
        return PolymarketRunFeaturesResponse(
            ok=False,
            run_id=run_id,
            run_dir=str(run_dir.relative_to(BASE_DIR)),
            features_built=False,
            features_path=None,
            features_manifest_path=None,
            stdout=stdout,
            stderr=stderr or "Decision features build failed.",
            duration_s=duration_s,
            command=cmd,
        )

    parquet_path = run_dir / "decision_features.parquet"
    csv_path = run_dir / LEGACY_DECISION_FEATURES_CSV
    if parquet_path.exists() and not csv_path.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
        except Exception:
            pass

    try:
        _rename_run_csv_files(run_dir)
    except Exception:
        pass

    features_path = _resolve_features_output_path(run_dir)
    features_manifest_path = run_dir / "feature_manifest.json"
    features_built = features_path is not None

    if features_built:
        try:
            _update_features_manifest(run_dir, features_built=True)
        except Exception:
            pass

    return PolymarketRunFeaturesResponse(
        ok=features_built,
        run_id=run_id,
        run_dir=str(run_dir.relative_to(BASE_DIR)),
        features_built=features_built,
        features_path=(
            str(features_path.relative_to(run_dir))
            if features_path is not None
            else None
        ),
        features_manifest_path=(
            str(features_manifest_path.relative_to(run_dir))
            if features_manifest_path.exists()
            else None
        ),
        stdout=stdout,
        stderr=stderr,
        duration_s=duration_s,
        command=cmd,
    )


def get_pipeline_run_file_path(run_id: str, filename: str) -> Path:
    """Resolve a file path under a run directory for safe download/open."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Run not found: {run_id}")
    return _resolve_run_file_path(run_dir, filename)


def _read_csv_head_preview(
    path: Path,
    limit: int,
) -> tuple[List[str], List[Dict[str, Optional[str]]]]:
    rows: List[Dict[str, Optional[str]]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            rows.append({key: row.get(key) for key in headers})
    return headers, rows


def _read_csv_tail_preview(
    path: Path,
    limit: int,
) -> tuple[List[str], List[Dict[str, Optional[str]]], int]:
    buffer: deque[Dict[str, Optional[str]]] = deque(maxlen=limit)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        row_count = 0
        for row in reader:
            row_count += 1
            buffer.append({key: row.get(key) for key in headers})
    return headers, list(buffer), row_count


def _preview_csv_file(
    csv_path: Path,
    *,
    filename: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    if csv_path.suffix.lower() != ".csv":
        raise ValueError("Preview is only supported for CSV files.")

    sanitized_limit = max(1, min(limit, 100))
    normalized_mode = mode.lower()
    if normalized_mode not in {"head", "tail"}:
        raise ValueError("mode must be 'head' or 'tail'")

    try:
        if normalized_mode == "tail":
            headers, rows, row_count = _read_csv_tail_preview(csv_path, sanitized_limit)
        else:
            headers, rows = _read_csv_head_preview(csv_path, sanitized_limit)
            row_count = None
    except Exception as exc:
        raise ValueError(f"Error reading CSV file: {exc}") from exc

    return {
        "filename": filename,
        "headers": headers,
        "rows": rows,
        "row_count": row_count,
        "mode": normalized_mode,
        "limit": sanitized_limit,
    }


# ---------------------------------------------------------------------------
# Run management: list / rename / set-active / delete
# ---------------------------------------------------------------------------


def list_pipeline_runs() -> List[Dict[str, Any]]:
    """List all pipeline runs with manifest data, newest first."""
    if not RUNS_DIR.exists():
        return []
    latest_run_id = get_latest_run_id()
    runs: List[Dict[str, Any]] = []
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        manifest = _safe_json_load(run_dir / "manifest.json") or {}
        status = manifest.get("status", "unknown")
        if status == "failed":
            try:
                _clear_latest_pointer_if_matches(run_dir.name)
                shutil.rmtree(run_dir)
            except Exception:
                pass
            continue
        size_bytes = sum(
            item.stat().st_size for item in run_dir.iterdir() if item.is_file()
        )
        runs.append({
            "run_id": run_dir.name,
            "run_dir": str(run_dir.relative_to(BASE_DIR)),
            "label": manifest.get("label"),
            "status": status,
            "created_at_utc": manifest.get("created_at_utc"),
            "finished_at_utc": manifest.get("finished_at_utc"),
            "duration_s": manifest.get("duration_s"),
            "tickers": manifest.get("tickers"),
            "start_date": manifest.get("start_date"),
            "end_date": manifest.get("end_date"),
            "markets": manifest.get("markets"),
            "price_rows": manifest.get("price_rows"),
            "features_built": manifest.get("features_built", False),
            "pinned": manifest.get("pinned", False),
            "is_active": run_dir.name == latest_run_id,
            "artifact_count": len(manifest.get("artifacts", {})),
            "csv_files": _build_run_csv_files(run_dir, manifest),
            "size_bytes": size_bytes,
            "error_summary": manifest.get("error_summary"),
        })
    runs.sort(key=lambda r: r.get("created_at_utc") or "", reverse=True)
    return runs


def rename_pipeline_run(run_id: str, label: str) -> Dict[str, Any]:
    """Set the user-facing label for a run (stored in manifest, folder unchanged)."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    manifest_path = run_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}
    manifest["label"] = label.strip() if label else None
    _atomic_json_write(manifest_path, manifest)
    return {"run_id": run_id, "label": manifest["label"]}


def set_active_run(run_id: str) -> Dict[str, Any]:
    """Set a run as the active/default run via latest.json."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    _update_latest_pointer(run_id)
    return {"run_id": run_id, "active": True}


def delete_pipeline_run(run_id: str) -> Dict[str, Any]:
    """Delete a pipeline run directory. Prevents deleting the active run."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    latest_run_id = get_latest_run_id()
    if run_id == latest_run_id:
        raise ValueError(
            "Cannot delete the currently active run. "
            "Set a different run as active first."
        )
    shutil.rmtree(run_dir)
    return {"run_id": run_id, "deleted": True}


def toggle_pin_run(run_id: str) -> Dict[str, Any]:
    """Toggle the pinned state of a run."""
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    manifest_path = run_dir / "manifest.json"
    manifest = _safe_json_load(manifest_path) or {}
    manifest["pinned"] = not manifest.get("pinned", False)
    _atomic_json_write(manifest_path, manifest)
    return {"run_id": run_id, "pinned": manifest["pinned"]}


def get_runs_storage_summary() -> Dict[str, Any]:
    """Get total storage used by all runs."""
    if not RUNS_DIR.exists():
        return {"total_runs": 0, "total_size_bytes": 0, "total_size_mb": 0}
    total = 0
    count = 0
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        count += 1
        for item in run_dir.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    return {
        "total_runs": count,
        "total_size_bytes": total,
        "total_size_mb": round(total / (1024 * 1024), 2),
    }


def _build_features_command(
    payload: PolymarketHistoryRunRequest,
    bars_dir: Path,
    dim_market_path: Path,
    out_dir: Path,
) -> tuple[List[str], Dict[str, str]]:
    if not FEATURES_SCRIPT_PATH.exists():
        raise RuntimeError(f"Features script not found: {FEATURES_SCRIPT_PATH}")

    cmd: List[str] = [sys.executable, str(FEATURES_SCRIPT_PATH)]
    cmd.extend(["--bars-dir", str(bars_dir)])
    cmd.extend(["--dim-market", str(dim_market_path)])
    cmd.extend(["--out-dir", str(out_dir)])

    if payload.prn_dataset:
        prn_path = _resolve_project_path(payload.prn_dataset)
        cmd.extend(["--prn-dataset", str(prn_path)])

    if payload.start_date:
        cmd.extend(["--start-date", payload.start_date])
    if payload.end_date:
        cmd.extend(["--end-date", payload.end_date])
    if payload.skip_subgraph_labels:
        cmd.append("--skip-subgraph-labels")

    env = {**os.environ}
    existing = env.get("PYTHONPATH")
    root = str(BASE_DIR)
    if existing:
        env["PYTHONPATH"] = os.pathsep.join([existing, root])
    else:
        env["PYTHONPATH"] = root

    return cmd, env


def _build_features(
    payload: PolymarketHistoryRunRequest,
    bars_dir: Path,
    dim_market_path: Path,
    out_dir: Path,
) -> tuple[Optional[Path], Optional[Path], str]:
    """Build decision features using script 8."""
    cmd, env = _build_features_command(payload, bars_dir, dim_market_path, out_dir)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    if result.returncode != 0:
        return None, None, result.stderr or result.stdout

    # Check which format was created
    parquet_path = out_dir / "decision_features.parquet"
    csv_path = out_dir / LEGACY_DECISION_FEATURES_CSV
    manifest_path = out_dir / "feature_manifest.json"

    # If parquet exists but CSV doesn't, create CSV from parquet for preview
    if parquet_path.exists() and not csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            df.to_csv(csv_path, index=False)
        except Exception:
            pass  # If conversion fails, just use parquet

    features_path = csv_path if csv_path.exists() else parquet_path if parquet_path.exists() else None

    return (
        features_path,
        manifest_path if manifest_path.exists() else None,
        result.stdout,
    )


def run_polymarket_history(
    payload: PolymarketHistoryRunRequest,
) -> PolymarketHistoryRunResponse:
    cmd, env, out_dir = _build_history_command(payload)

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    duration_s = round(time.monotonic() - start, 3)

    run_id = _parse_run_id(result.stdout)
    run_dir, files = _collect_run_files(out_dir, run_id)
    if result.returncode == 0 and run_dir and run_dir.exists():
        try:
            _copy_dim_market_to_run(run_dir)
        except Exception:
            pass
        try:
            _rename_run_csv_files(run_dir)
        except Exception:
            pass
        files = _list_run_file_names(run_dir)

    return PolymarketHistoryRunResponse(
        ok=result.returncode == 0,
        run_id=run_id,
        out_dir=str(out_dir.relative_to(BASE_DIR)),
        run_dir=str(run_dir.relative_to(BASE_DIR)) if run_dir else None,
        files=files,
        stdout=result.stdout,
        stderr=result.stderr,
        duration_s=duration_s,
        command=cmd,
    )


class PolymarketHistoryJob:
    def __init__(self, job_id: str, payload: PolymarketHistoryRunRequest) -> None:
        self.job_id = job_id
        self.payload = payload
        self.status = "queued"
        self.phase: Optional[str] = None
        self.result: Optional[PolymarketHistoryRunResponse] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.finished_at: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._features_process: Optional[subprocess.Popen[str]] = None
        self._process_handle: Optional[ManagedProcessHandle] = None
        self._features_process_handle: Optional[ManagedProcessHandle] = None
        self._cancel_requested = False
        self._history_progress = JobProgressTracker()
        self._features_progress = JobProgressTracker()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        if self.status in {"finished", "failed", "cancelled"}:
            return
        self._cancel_requested = True
        for proc, handle in (
            (self._process, self._process_handle),
            (self._features_process, self._features_process_handle),
        ):
            if proc and proc.poll() is None and handle is not None:
                terminate_managed_process(handle, term_timeout_s=5.0, kill_timeout_s=5.0)
                continue
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

    def to_status(self) -> PolymarketHistoryJobStatus:
        return PolymarketHistoryJobStatus(
            job_id=self.job_id,
            status=self.status,
            phase=self.phase,
            progress=self._history_progress.snapshot(),
            features_progress=self._features_progress.snapshot(),
            result=self.result,
            error=self.error,
            started_at=self.started_at,
            finished_at=self.finished_at,
        )

    def _update_history_progress(self, line: str) -> None:
        match = _HISTORY_COMPLETE_RE.search(line)
        if not match:
            return
        total = int(match.group("total"))
        job_id = match.group("job_id")
        status = match.group("status").lower()
        failed = status not in {"ok", "success", "completed"}
        self._history_progress.set_total(total)
        self._history_progress.mark_completed(job_id, failed=failed)

    def _update_features_progress(self, line: str) -> None:
        match = _FEATURE_PROGRESS_RE.search(line)
        if not match:
            return
        total = int(match.group("total"))
        step_id = match.group("step")
        self._features_progress.set_total(total)
        self._features_progress.mark_completed(step_id)

    def _run(self) -> None:
        self.started_at = datetime.utcnow()
        if self._cancel_requested:
            self.status = "cancelled"
            self.error = "Weekly history run cancelled."
            self.finished_at = datetime.utcnow()
            return
        self.status = "running"
        self.phase = "history"
        try:
            cmd, env, out_dir = _build_history_command(self.payload)
            start = time.monotonic()
            handle = spawn_managed_process(
                cmd,
                job_id=self.job_id,
                service="polymarket_history",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                env=env,
            )
            proc = handle.process
            if proc is None:
                raise RuntimeError("Failed to start polymarket history process.")
            self._process = proc
            self._process_handle = handle

            stdout_lines: List[str] = []
            stderr_lines: List[str] = []
            run_id: Optional[str] = None
            run_dir: Optional[Path] = None
            files: List[str] = []
            features_built = False
            features_path: Optional[Path] = None
            features_manifest_path: Optional[Path] = None

            def snapshot_result(ok: bool) -> None:
                self.result = PolymarketHistoryRunResponse(
                    ok=ok,
                    run_id=run_id,
                    out_dir=str(out_dir.relative_to(BASE_DIR)),
                    run_dir=str(run_dir.relative_to(BASE_DIR)) if run_dir else None,
                    files=list(files),
                    stdout=''.join(stdout_lines),
                    stderr=''.join(stderr_lines),
                    duration_s=round(time.monotonic() - start, 3),
                    command=cmd,
                    features_built=features_built,
                    features_path=(
                        str(features_path.relative_to(run_dir))
                        if features_path and run_dir
                        else None
                    ),
                    features_manifest_path=(
                        str(features_manifest_path.relative_to(run_dir))
                        if features_manifest_path and run_dir
                        else None
                    ),
                )

            def read_stdout():
                nonlocal run_id
                if proc.stdout:
                    for line in iter(proc.stdout.readline, ''):
                        if line:
                            stdout_lines.append(line)
                            self._update_history_progress(line)
                            parsed = _parse_run_id(''.join(stdout_lines))
                            if parsed:
                                run_id = parsed
                            snapshot_result(ok=False)
                    proc.stdout.close()

            def read_stderr():
                if proc.stderr:
                    for line in iter(proc.stderr.readline, ''):
                        if line:
                            stderr_lines.append(line)
                            snapshot_result(ok=False)
                    proc.stderr.close()

            stdout_thread = threading.Thread(target=read_stdout, daemon=True)
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()

            proc.wait()
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)

            run_id = _parse_run_id(stdout) or run_id
            run_dir, files = _collect_run_files(out_dir, run_id)
            ok = proc.returncode == 0 and not self._cancel_requested

            if ok and self.payload.build_features and run_dir:
                self.phase = "features"
                stdout_lines.append("\n[Features] Building decision features...\n")
                snapshot_result(ok=False)

                if self._cancel_requested:
                    stdout_lines.append("[Features] Cancel requested before feature build.\n")
                else:
                    bars_dir_arg = (
                        Path(self.payload.bars_dir)
                        if self.payload.bars_dir
                        else BASE_DIR / "src" / "data" / "analysis" / "polymarket" / "bars_history"
                    )
                    dim_market_arg = (
                        Path(self.payload.dim_market_out)
                        if self.payload.dim_market_out
                        else BASE_DIR / "src" / "data" / "models" / "polymarket" / "dim_market_weekly.csv"
                    )
                    features_cmd, features_env = _build_features_command(
                        self.payload,
                        bars_dir_arg,
                        dim_market_arg,
                        run_dir,
                    )
                    features_handle = spawn_managed_process(
                        features_cmd,
                        job_id=self.job_id,
                        service="polymarket_features",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        env=features_env,
                    )
                    features_proc = features_handle.process
                    if features_proc is None:
                        raise RuntimeError("Failed to start feature build process.")
                    self._features_process = features_proc
                    self._features_process_handle = features_handle

                    def read_features_stdout():
                        if features_proc.stdout:
                            for line in iter(features_proc.stdout.readline, ''):
                                if line:
                                    stdout_lines.append(line)
                                    self._update_features_progress(line)
                                    snapshot_result(ok=False)
                            features_proc.stdout.close()

                    def read_features_stderr():
                        if features_proc.stderr:
                            for line in iter(features_proc.stderr.readline, ''):
                                if line:
                                    stderr_lines.append(line)
                                    snapshot_result(ok=False)
                            features_proc.stderr.close()

                    features_stdout_thread = threading.Thread(
                        target=read_features_stdout,
                        daemon=True,
                    )
                    features_stderr_thread = threading.Thread(
                        target=read_features_stderr,
                        daemon=True,
                    )
                    features_stdout_thread.start()
                    features_stderr_thread.start()

                    features_proc.wait()
                    features_stdout_thread.join(timeout=1)
                    features_stderr_thread.join(timeout=1)

                    if self._cancel_requested:
                        stdout_lines.append("\n[Features] Feature build cancelled.\n")
                        self._features_progress.mark_failed()
                    elif features_proc.returncode != 0:
                        stdout_lines.append("\n[Features] Failed to build features.\n")
                        self._features_progress.mark_failed()
                    else:
                        parquet_path = run_dir / "decision_features.parquet"
                        csv_path = run_dir / LEGACY_DECISION_FEATURES_CSV
                        manifest_path = run_dir / "feature_manifest.json"

                        if parquet_path.exists() and not csv_path.exists():
                            try:
                                import pandas as pd
                                df = pd.read_parquet(parquet_path)
                                df.to_csv(csv_path, index=False)
                            except Exception:
                                pass

                        features_path = _resolve_features_output_path(run_dir)
                        features_manifest_path = (
                            manifest_path if manifest_path.exists() else None
                        )

                        if features_path:
                            features_built = True
                            stdout_lines.append(
                                f"\n[Features] Built successfully: {features_path.name}\n"
                            )
                            files.append(features_path.name)
                            if features_manifest_path:
                                files.append(features_manifest_path.name)
                        else:
                            stdout_lines.append(
                                "\n[Features] Completed but output files not found.\n"
                            )

                    self._features_process = None
                    self._features_process_handle = None

            if run_dir and run_dir.exists():
                if ok and run_id:
                    try:
                        _copy_dim_market_to_run(run_dir)
                    except Exception:
                        pass
                try:
                    _rename_run_csv_files(run_dir)
                except Exception:
                    pass
                files = _list_run_file_names(run_dir)
                if features_manifest_path and not features_manifest_path.exists():
                    features_manifest_path = None
                features_path = _resolve_features_output_path(run_dir)

            duration_s = round(time.monotonic() - start, 3)
            stdout = ''.join(stdout_lines)
            stderr = ''.join(stderr_lines)

            self.result = PolymarketHistoryRunResponse(
                ok=ok,
                run_id=run_id,
                out_dir=str(out_dir.relative_to(BASE_DIR)),
                run_dir=str(run_dir.relative_to(BASE_DIR)) if run_dir else None,
                files=files,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration_s,
                command=cmd,
                features_built=features_built,
                features_path=(
                    str(features_path.relative_to(run_dir))
                    if features_path and run_dir
                    else None
                ),
                features_manifest_path=(
                    str(features_manifest_path.relative_to(run_dir))
                    if features_manifest_path and run_dir
                    else None
                ),
            )

            if self._cancel_requested:
                self.status = "cancelled"
                self.error = "Weekly history run cancelled."
            elif ok:
                self.status = "finished"
            else:
                self.status = "failed"
                self.error = (stderr or "").strip() or "Weekly history run failed."

            # --- Phase 0+1: post-run hooks (enhance manifest, latest pointer) ---
            if run_dir and run_dir.exists():
                try:
                    final_status = "cancelled" if self._cancel_requested else ("success" if ok else "failed")
                    _enhance_manifest(
                        run_dir,
                        status=final_status,
                        duration_s=duration_s,
                        pipeline_args=_pipeline_args_from_payload(self.payload),
                        error_summary=self.error if not ok else None,
                        features_built=features_built,
                    )
                    if ok and run_id:
                        _update_latest_pointer(run_id)
                except Exception:
                    pass  # never let post-run hooks break the job status

            # Failed runs are auto-pruned so they never appear in history directories.
            if self.status == "failed" and run_dir and run_dir.exists():
                try:
                    if run_id:
                        _clear_latest_pointer_if_matches(run_id)
                    shutil.rmtree(run_dir)
                except Exception:
                    pass
        except Exception as exc:
            self.status = "failed"
            self.error = str(exc)
        finally:
            self._process = None
            self._features_process = None
            self._process_handle = None
            self._features_process_handle = None
            self.finished_at = datetime.utcnow()


class PolymarketHistoryJobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, PolymarketHistoryJob] = {}
        self._lock = threading.Lock()

    def start_job(self, payload: PolymarketHistoryRunRequest) -> str:
        job_id = uuid4().hex
        job = PolymarketHistoryJob(job_id, payload)
        with self._lock:
            self._jobs[job_id] = job
        job.start()
        return job_id

    def get_status(self, job_id: str) -> PolymarketHistoryJobStatus:
        job = self._get_job(job_id)
        return job.to_status()

    def list_jobs(self) -> List[PolymarketHistoryJobStatus]:
        with self._lock:
            return [job.to_status() for job in self._jobs.values()]

    def cancel_job(self, job_id: str) -> PolymarketHistoryJobStatus:
        job = self._get_job(job_id)
        job.cancel()
        return job.to_status()

    def _get_job(self, job_id: str) -> PolymarketHistoryJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


POLYMARKET_HISTORY_JOB_MANAGER = PolymarketHistoryJobManager()


def start_polymarket_history_job(payload: PolymarketHistoryRunRequest) -> str:
    from app.services.job_guard import ensure_no_active_jobs

    ensure_no_active_jobs()
    return POLYMARKET_HISTORY_JOB_MANAGER.start_job(payload)


def get_polymarket_history_job(job_id: str) -> PolymarketHistoryJobStatus:
    return POLYMARKET_HISTORY_JOB_MANAGER.get_status(job_id)


def cancel_polymarket_history_job(job_id: str) -> PolymarketHistoryJobStatus:
    return POLYMARKET_HISTORY_JOB_MANAGER.cancel_job(job_id)


def get_csv_preview(
    job_id: str,
    filename: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    """Read and preview a CSV file from a completed job run directory."""
    job_status = POLYMARKET_HISTORY_JOB_MANAGER.get_status(job_id)

    if not job_status.result or not job_status.result.run_dir:
        raise FileNotFoundError("Job has no run directory yet.")

    run_dir = BASE_DIR / job_status.result.run_dir
    csv_path = _resolve_run_file_path(run_dir, filename)

    return _preview_csv_file(csv_path, filename=filename, limit=limit, mode=mode)


def get_run_csv_preview(
    run_id: str,
    filename: str,
    limit: int = 20,
    mode: str = "head",
) -> Dict[str, Any]:
    """Read and preview a CSV file from a persisted run directory."""
    csv_path = get_pipeline_run_file_path(run_id, filename)
    return _preview_csv_file(csv_path, filename=filename, limit=limit, mode=mode)
