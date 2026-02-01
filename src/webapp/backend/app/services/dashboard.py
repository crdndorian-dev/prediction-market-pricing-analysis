from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[5]
DATA_DIR = BASE_DIR / "src" / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DIR = DATA_DIR / "raw"

_DATASET_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "summary": None}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _load_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "model": row.get("model"),
                    "split": row.get("split"),
                    "logloss": _safe_float(row.get("logloss")),
                    "brier": _safe_float(row.get("brier")),
                    "ece": _safe_float(row.get("ece")),
                    "n": row.get("n"),
                    "weight_sum": _safe_float(row.get("weight_sum")),
                }
            )
    return rows


def _pick_best(rows: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    candidates = [row for row in rows if row.get(metric) is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row[metric])


def _summarize_metrics(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    preferred = [row for row in rows if row.get("split") == "test"]
    pool = preferred if preferred else rows
    best_logloss = _pick_best(pool, "logloss") or pool[0]
    if best_logloss is None:
        return None
    return {
        "model": best_logloss.get("model"),
        "split": best_logloss.get("split"),
        "logloss": best_logloss.get("logloss"),
        "brier": best_logloss.get("brier"),
        "ece": best_logloss.get("ece"),
        "n": best_logloss.get("n"),
        "weight_sum": best_logloss.get("weight_sum"),
    }


def _latest_file(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    files = [item for item in directory.iterdir() if item.is_file()]
    if not files:
        return None
    return max(files, key=lambda item: item.stat().st_mtime)


def _iso_from_mtime(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _calc_days_since(date_value: Optional[datetime]) -> Optional[int]:
    if not date_value:
        return None
    now = datetime.now(timezone.utc).date()
    delta = now - date_value.date()
    return delta.days


def _parse_date_str(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _find_dataset_file(raw_dir: Path) -> Optional[Path]:
    preferred = raw_dir / "options-chain-dataset.csv"
    if preferred.exists():
        return preferred
    if not raw_dir.exists():
        return None
    candidates = [
        item
        for item in raw_dir.iterdir()
        if item.is_file() and item.suffix == ".csv" and "drops" not in item.name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        # Skip header
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _summarize_dataset(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    cached = _DATASET_CACHE
    if cached.get("path") == str(path) and cached.get("mtime") == mtime:
        return cached.get("summary")

    row_count = 0
    tickers = set()
    date_min = None
    date_max = None
    date_column = None

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            if "week_monday" in reader.fieldnames:
                date_column = "week_monday"
            elif "asof_date" in reader.fieldnames:
                date_column = "asof_date"
            elif "week_friday" in reader.fieldnames:
                date_column = "week_friday"

        for row in reader:
            row_count += 1
            ticker = row.get("ticker")
            if ticker:
                tickers.add(ticker)
            if date_column:
                parsed = _parse_date_str(row.get(date_column))
                if parsed:
                    if date_min is None or parsed < date_min:
                        date_min = parsed
                    if date_max is None or parsed > date_max:
                        date_max = parsed

    summary = {
        "fileName": path.name,
        "path": str(path.relative_to(BASE_DIR)),
        "sizeMB": round(path.stat().st_size / (1024 * 1024), 2),
        "rowCount": row_count,
        "columnCount": len(reader.fieldnames or []),
        "tickerCount": len(tickers),
        "dateRange": {
            "column": date_column,
            "start": date_min.date().isoformat() if date_min else None,
            "end": date_max.date().isoformat() if date_max else None,
        },
        "lastModified": _iso_from_mtime(mtime),
    }

    _DATASET_CACHE.update({"path": str(path), "mtime": mtime, "summary": summary})
    return summary


def _dataset_label(metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    csv_path = metadata.get("csv")
    if not csv_path:
        return None
    return Path(csv_path).name


def _focus_label(calibration: Optional[str]) -> str:
    if not calibration:
        return "Calibration run"
    if calibration == "platt":
        return "Logit + Platt calibration"
    if calibration == "none":
        return "Baseline pRN calibration"
    return f"Calibration: {calibration}"


def _run_status(metrics: Optional[Dict[str, Any]]) -> str:
    return "Success" if metrics else "Warning"


def _get_runs() -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not MODELS_DIR.exists():
        return runs
    for model_dir in MODELS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        metadata_path = model_dir / "metadata.json"
        metrics_path = model_dir / "metrics.csv"
        metadata = _load_json(metadata_path)
        metrics_rows = _load_metrics(metrics_path)
        metrics_summary = _summarize_metrics(metrics_rows)
        mtime_candidates = [
            model_dir.stat().st_mtime,
            metadata_path.stat().st_mtime if metadata_path.exists() else 0,
            metrics_path.stat().st_mtime if metrics_path.exists() else 0,
        ]
        latest_mtime = max(mtime_candidates)
        runs.append(
            {
                "id": model_dir.name,
                "dataset": _dataset_label(metadata),
                "calibration": metadata.get("calibration") if metadata else None,
                "focus": _focus_label(metadata.get("calibration") if metadata else None),
                "status": _run_status(metrics_summary),
                "metrics": metrics_summary,
                "modified_at": _iso_from_mtime(latest_mtime),
                "metadata": metadata or {},
            }
        )
    runs.sort(key=lambda run: run["modified_at"], reverse=True)
    return runs


def _build_signal_bars(runs: List[Dict[str, Any]]) -> List[int]:
    values: List[float] = []
    for run in runs[:10]:
        metrics = run.get("metrics")
        if metrics and metrics.get("ece") is not None:
            values.append(metrics["ece"])
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [70 for _ in values]
    bars: List[int] = []
    for value in values:
        normalized = 30 + 70 * (max_v - value) / (max_v - min_v)
        bars.append(int(round(normalized)))
    return bars


def _best_metrics(rows: List[Dict[str, Any]], split: str) -> Optional[Dict[str, Any]]:
    pool = [row for row in rows if row.get("split") == split]
    if not pool:
        return None
    best = _pick_best(pool, "logloss")
    if not best:
        return None
    return {
        "model": best.get("model"),
        "split": best.get("split"),
        "logloss": best.get("logloss"),
        "brier": best.get("brier"),
        "ece": best.get("ece"),
    }


def build_dashboard_payload() -> Dict[str, Any]:
    runs = _get_runs()
    latest = runs[0] if runs else None
    latest_meta = latest.get("metadata") if latest else None
    latest_metrics = latest.get("metrics") if latest else None

    split_info = latest_meta.get("split_info") if latest_meta else None
    test_range = split_info.get("test_weeks_range") if split_info else None
    test_end = _parse_date(test_range[1]) if test_range else None

    dataset_file = _find_dataset_file(RAW_DIR)
    dataset_summary = _summarize_dataset(dataset_file) if dataset_file else None
    raw_latest = _latest_file(RAW_DIR)
    raw_label = raw_latest.name if raw_latest else None
    raw_date = (
        datetime.fromtimestamp(raw_latest.stat().st_mtime, tz=timezone.utc)
        if raw_latest
        else None
    )

    data_date = test_end or raw_date
    data_days = _calc_days_since(data_date)

    readiness = []
    if dataset_summary:
        readiness.append(
            {
                "title": "Ingestion snapshot",
                "detail": (
                    "1-option-chain-build-historic-dataset-v1.0.py"
                    f" · {dataset_summary.get('fileName')} · {dataset_summary.get('rowCount', 0)} rows"
                ),
                "status": "Ready",
                "progress": 100,
            }
        )
    else:
        readiness.append(
            {
                "title": "Ingestion snapshot",
                "detail": "No dataset found in src/data/raw",
                "status": "Missing",
                "progress": 0,
            }
        )

    if latest_meta:
        calibration = latest_meta.get("calibration", "unknown")
        features = latest_meta.get("features") or latest_meta.get("features_used")
        feature_count = len(features) if isinstance(features, list) else None
        detail = f"2-calibrate-logit-model-v1.5.py · Calibration: {calibration}"
        if feature_count is not None:
            detail = f"{detail} · {feature_count} features"
        readiness.append(
            {
                "title": "Calibration model",
                "detail": detail,
                "status": "Ready",
                "progress": 100,
            }
        )
    else:
        readiness.append(
            {
                "title": "Calibration model",
                "detail": "No metadata.json found",
                "status": "Missing",
                "progress": 0,
            }
        )

    if latest_metrics:
        readiness.append(
            {
                "title": "Analysis outputs",
                "detail": "metrics.csv available · model diagnostics",
                "status": "Ready",
                "progress": 100,
            }
        )
    else:
        readiness.append(
            {
                "title": "Analysis outputs",
                "detail": "metrics.csv missing",
                "status": "Needs review",
                "progress": 45,
            }
        )

    recent_runs = []
    for run in runs[:3]:
        recent_runs.append(
            {
                "id": run.get("id"),
                "dataset": run.get("dataset") or "Unknown dataset",
                "focus": run.get("focus"),
                "status": run.get("status"),
                "time": run.get("modified_at"),
            }
        )

    drops_file = RAW_DIR / "option-chain-historic-dataset-drops.csv"
    drops_summary = (
        {
            "fileName": drops_file.name,
            "path": str(drops_file.relative_to(BASE_DIR)),
            "rowCount": _count_csv_rows(drops_file),
        }
        if drops_file.exists()
        else None
    )

    best_test = None
    best_val = None
    for run in runs:
        metrics_rows = _load_metrics(MODELS_DIR / run["id"] / "metrics.csv")
        candidate_test = _best_metrics(metrics_rows, "test")
        candidate_val = _best_metrics(metrics_rows, "val")
        if candidate_test and (
            best_test is None or candidate_test["logloss"] < best_test["logloss"]
        ):
            best_test = candidate_test
        if candidate_val and (
            best_val is None or candidate_val["logloss"] < best_val["logloss"]
        ):
            best_val = candidate_val

    payload = {
        "asOf": datetime.now(timezone.utc).isoformat(),
        "hero": {
            "dataFreshnessDate": data_date.date().isoformat() if data_date else None,
            "dataFreshnessDays": data_days,
            "dataSourceLabel": _dataset_label(latest_meta) or raw_label,
            "calibrationEce": latest_metrics.get("ece") if latest_metrics else None,
            "calibrationModel": latest_metrics.get("model") if latest_metrics else None,
            "calibrationSplit": latest_metrics.get("split") if latest_metrics else None,
            "lastRunId": latest.get("id") if latest else None,
            "lastRunTime": latest.get("modified_at") if latest else None,
            "lastRunSummary": latest.get("focus") if latest else None,
        },
        "readiness": readiness,
        "runQueue": [],
        "recentRuns": recent_runs,
        "datasetSummary": dataset_summary,
        "dropsSummary": drops_summary,
        "modelSummary": {
            "modelCount": len(runs),
            "latestModel": {
                "id": latest.get("id") if latest else None,
                "calibration": latest_meta.get("calibration") if latest_meta else None,
                "dataset": _dataset_label(latest_meta),
                "modifiedAt": latest.get("modified_at") if latest else None,
                "metrics": latest_metrics,
            },
            "bestTest": best_test,
            "bestVal": best_val,
        },
        "calibrationSnapshot": {
            "logloss": latest_metrics.get("logloss") if latest_metrics else None,
            "brier": latest_metrics.get("brier") if latest_metrics else None,
            "ece": latest_metrics.get("ece") if latest_metrics else None,
            "model": latest_metrics.get("model") if latest_metrics else None,
            "split": latest_metrics.get("split") if latest_metrics else None,
        },
        "signalBars": _build_signal_bars(runs),
    }

    return payload
