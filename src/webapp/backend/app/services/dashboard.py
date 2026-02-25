from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.services.job_guard import list_active_jobs

BASE_DIR = Path(__file__).resolve().parents[5]
DATA_DIR = BASE_DIR / "src" / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DIR = DATA_DIR / "raw"
SUBGRAPH_RUNS_DIR = RAW_DIR / "polymarket" / "subgraph" / "runs"
BACKTESTS_DIR = DATA_DIR / "analysis" / "backtests"
SIGNALS_DIR = DATA_DIR / "analysis" / "signals"
POLYMARKET_MODELS_DIR = MODELS_DIR / "polymarket"
MIXED_MODELS_DIR = MODELS_DIR / "mixed"
BARS_DIR = DATA_DIR / "analysis" / "polymarket" / "bars"
DIM_MARKET_PATHS = (
    POLYMARKET_MODELS_DIR / "dim_market.parquet",
    POLYMARKET_MODELS_DIR / "dim_market.csv",
)
FEATURES_PATHS = (
    POLYMARKET_MODELS_DIR / "decision_features.parquet",
    POLYMARKET_MODELS_DIR / "decision_features.csv",
)
FEATURES_MANIFEST = POLYMARKET_MODELS_DIR / "feature_manifest.json"

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


def _latest_csv_file(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    files = [
        item
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() == ".csv"
    ]
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
    if candidates:
        return max(candidates, key=lambda item: item.stat().st_mtime)
    nested = [
        item
        for item in raw_dir.rglob("*.csv")
        if item.is_file() and "drops" not in item.name
    ]
    if not nested:
        return None
    return max(nested, key=lambda item: item.stat().st_mtime)


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


def _count_parquet_rows(path: Path) -> Optional[int]:
    try:
        import pyarrow.parquet as pq  # type: ignore

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore

        return int(len(pd.read_parquet(path)))
    except Exception:
        return None


def _count_table_rows(path: Path) -> Optional[int]:
    if path.suffix.lower() == ".parquet":
        return _count_parquet_rows(path)
    return _count_csv_rows(path)


def _pick_existing(paths: tuple[Path, ...]) -> Optional[Path]:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def _summarize_subgraph() -> Optional[Dict[str, Any]]:
    if not SUBGRAPH_RUNS_DIR.exists():
        return None
    latest_run: Optional[Path] = None
    latest_manifest: Optional[Dict[str, Any]] = None
    latest_key: Optional[str] = None

    for run_dir in SUBGRAPH_RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        manifest = _load_json(run_dir / "manifest.json") or {}
        run_time = manifest.get("finished_at_utc") or manifest.get("started_at_utc")
        sort_key = run_time or _iso_from_mtime(run_dir.stat().st_mtime)
        if latest_key is None or sort_key > latest_key:
            latest_key = sort_key
            latest_run = run_dir
            latest_manifest = manifest

    if not latest_run:
        return None

    return {
        "latestRunId": latest_run.name,
        "latestRunTime": (
            latest_manifest.get("finished_at_utc")
            or latest_manifest.get("started_at_utc")
            if latest_manifest
            else _iso_from_mtime(latest_run.stat().st_mtime)
        ),
        "latestQuery": latest_manifest.get("query_name") if latest_manifest else None,
        "totalEntities": latest_manifest.get("total_entities") if latest_manifest else None,
    }


def _summarize_market_map() -> Optional[Dict[str, Any]]:
    path = _pick_existing(DIM_MARKET_PATHS)
    if not path:
        return None
    row_count = _count_table_rows(path)
    return {
        "fileName": path.name,
        "path": str(path.relative_to(BASE_DIR)),
        "rowCount": row_count,
        "lastModified": _iso_from_mtime(path.stat().st_mtime),
    }


def _summarize_bars() -> Optional[Dict[str, Any]]:
    if not BARS_DIR.exists():
        return None
    freqs: Dict[str, int] = {}
    latest_mtime: Optional[float] = None
    has_files = False
    for freq in ["1m", "5m", "1h"]:
        files = list((BARS_DIR / freq).glob("market_id=*/date=*/bars.csv"))
        if files:
            has_files = True
        total = 0
        for path in files:
            total += _count_csv_rows(path)
            if latest_mtime is None or path.stat().st_mtime > latest_mtime:
                latest_mtime = path.stat().st_mtime
        freqs[freq] = total
    if not has_files:
        return None
    return {
        "barsDir": str(BARS_DIR.relative_to(BASE_DIR)),
        "freqs": freqs,
        "lastModified": _iso_from_mtime(latest_mtime) if latest_mtime else None,
    }


def _summarize_features() -> Optional[Dict[str, Any]]:
    path = _pick_existing(FEATURES_PATHS)
    if not path:
        return None
    manifest = _load_json(FEATURES_MANIFEST) or {}
    features = manifest.get("features")
    feature_count = len(features) if isinstance(features, list) else None
    created_at = manifest.get("created_at_utc") if isinstance(manifest, dict) else None
    return {
        "fileName": path.name,
        "path": str(path.relative_to(BASE_DIR)),
        "lastModified": _iso_from_mtime(path.stat().st_mtime),
        "featureCount": feature_count,
        "createdAtUtc": created_at,
    }


def _parse_mixed_run_time(run_id: str) -> Optional[str]:
    candidate = run_id
    if candidate.startswith("mixed-"):
        candidate = candidate[len("mixed-"):]
    try:
        dt = datetime.strptime(candidate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _summarize_mixed_model() -> Optional[Dict[str, Any]]:
    if not MIXED_MODELS_DIR.exists():
        return None

    run_dirs = [item for item in MIXED_MODELS_DIR.iterdir() if item.is_dir()]
    if not run_dirs:
        return None

    latest_run: Optional[Path] = None
    latest_key: Optional[str] = None
    latest_meta: Optional[Dict[str, Any]] = None

    for run_dir in run_dirs:
        metadata = _load_json(run_dir / "metadata.json") or {}
        run_time = (
            metadata.get("trained_at_utc")
            or _parse_mixed_run_time(run_dir.name)
            or _iso_from_mtime(run_dir.stat().st_mtime)
        )
        if latest_key is None or run_time > latest_key:
            latest_key = run_time
            latest_run = run_dir
            latest_meta = metadata

    if not latest_run:
        return None

    return {
        "runCount": len(run_dirs),
        "latestRunId": latest_run.name,
        "latestRunTime": latest_key,
        "modelType": latest_meta.get("model_type") if latest_meta else None,
        "rowCount": latest_meta.get("rows") if latest_meta else None,
    }


def _parse_backtest_run_time(run_id: str) -> Optional[str]:
    candidate = run_id
    if candidate.startswith("backtest-"):
        candidate = candidate[len("backtest-"):]
    try:
        dt = datetime.strptime(candidate, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return None


def _summarize_backtests() -> Optional[Dict[str, Any]]:
    if not BACKTESTS_DIR.exists():
        return None

    latest_run: Optional[Path] = None
    latest_key: Optional[str] = None
    latest_metrics: Optional[Dict[str, Any]] = None

    for run_dir in BACKTESTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        run_time = _parse_backtest_run_time(run_dir.name)
        sort_key = run_time or _iso_from_mtime(run_dir.stat().st_mtime)
        if latest_key is None or sort_key > latest_key:
            latest_key = sort_key
            latest_run = run_dir
            latest_metrics = _load_json(run_dir / "metrics.json")

    if not latest_run:
        return None

    metrics_summary = None
    if isinstance(latest_metrics, dict):
        metrics_summary = {
            "trades": latest_metrics.get("trades"),
            "hitRate": latest_metrics.get("hit_rate"),
            "sharpeLike": latest_metrics.get("sharpe_like"),
            "maxDrawdown": latest_metrics.get("max_drawdown"),
        }

    return {
        "latestRunId": latest_run.name,
        "latestRunTime": latest_key,
        "metrics": metrics_summary,
    }


def _summarize_signals() -> Optional[Dict[str, Any]]:
    if not SIGNALS_DIR.exists():
        return None

    run_dirs = [item for item in SIGNALS_DIR.iterdir() if item.is_dir()]
    if not run_dirs:
        return None

    latest_run = max(run_dirs, key=lambda item: item.stat().st_mtime)
    summary = _load_json(latest_run / "summary.json")
    signals_csv = latest_run / "signals.csv"
    row_count = _count_csv_rows(signals_csv) if signals_csv.exists() else None

    latest_time = None
    if isinstance(summary, dict):
        latest_time = summary.get("created_at_utc")
    if not latest_time:
        latest_time = _iso_from_mtime(latest_run.stat().st_mtime)

    return {
        "latestRunId": latest_run.name,
        "latestRunTime": latest_time,
        "rowCount": row_count,
    }


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
                    "01-option-chain-build-historic-dataset-v1.0.py"
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
        detail = f"03-calibrate-logit-model-v1.5.py · Calibration: {calibration}"
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
    subgraph_summary = _summarize_subgraph()
    market_map_summary = _summarize_market_map()
    bars_summary = _summarize_bars()
    features_summary = _summarize_features()
    mixed_model_summary = _summarize_mixed_model()
    backtest_summary = _summarize_backtests()
    signals_summary = _summarize_signals()

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
        "runQueue": list_active_jobs(),
        "recentRuns": recent_runs,
        "datasetSummary": dataset_summary,
        "dropsSummary": drops_summary,
        "subgraphSummary": subgraph_summary,
        "marketMapSummary": market_map_summary,
        "barsSummary": bars_summary,
        "featuresSummary": features_summary,
        "mixedModelSummary": mixed_model_summary,
        "backtestSummary": backtest_summary,
        "signalsSummary": signals_summary,
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
