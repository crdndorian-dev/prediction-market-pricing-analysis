#!/usr/bin/env python3
"""
Backfill realized-volatility features into existing option-chain history artifacts.

This one-off script updates the main dataset artifacts in place:
- training-main-dataset.csv: add rv5, rv10
- legacy-main-dataset.csv: add rv5, rv10
- snapshot-main-dataset.csv: add rv5, rv10, rv20

The realized-vol computations reuse the same historical close-loading logic as the
option-chain history builder so the backfill stays consistent with future builds.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILDER_SCRIPT_PATH = REPO_ROOT / "src" / "scripts" / "01-option-chain-build-historic-dataset-v1.0.py"
DEFAULT_DATASET_DIR = REPO_ROOT / "src" / "data" / "raw" / "option-chain" / "main-dataset"


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    columns: Tuple[Tuple[str, int], ...]
    insert_before: str


ARTIFACT_SPECS: Tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        name="training-main-dataset.csv",
        columns=(("rv5", 5), ("rv10", 10)),
        insert_before="rv20",
    ),
    ArtifactSpec(
        name="legacy-main-dataset.csv",
        columns=(("rv5", 5), ("rv10", 10)),
        insert_before="rv20",
    ),
    ArtifactSpec(
        name="snapshot-main-dataset.csv",
        columns=(("rv5", 5), ("rv10", 10), ("rv20", 20)),
        insert_before="moneyness_ref",
    ),
)


def _load_builder_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("option_chain_history_builder", BUILDER_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load builder module from {BUILDER_SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalize_spot_scale(value: object) -> str:
    token = str(value or "").strip().lower()
    return "raw" if token == "raw" else "split_adj"


def _normalize_asof_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=False)
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"Found {bad} invalid asof_date values while backfilling volatility features.")
    return parsed.dt.date


def _make_key(ticker: str, asof_used: date, spot_scale_used: str) -> str:
    return f"{str(ticker).strip().upper()}|{asof_used.isoformat()}|{_normalize_spot_scale(spot_scale_used)}"


def _collect_unique_keys(dataset_dir: Path, specs: Iterable[ArtifactSpec]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for spec in specs:
        path = dataset_dir / spec.name
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["ticker", "asof_date", "spot_scale_used"])
        if frame.empty:
            continue
        frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
        frame["spot_scale_used"] = frame["spot_scale_used"].astype("string").fillna("split_adj").map(_normalize_spot_scale)
        frame["asof_used"] = _normalize_asof_date(frame["asof_date"])
        frames.append(frame[["ticker", "asof_used", "spot_scale_used"]])
    if not frames:
        raise FileNotFoundError("No dataset artifacts found to backfill.")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates()
    combined.sort_values(["ticker", "asof_used", "spot_scale_used"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _build_feature_maps(
    *,
    keys: pd.DataFrame,
    builder: ModuleType,
    lookbacks: Iterable[int],
    stock_source: str,
) -> Dict[str, Dict[str, float]]:
    tickers = sorted({str(value) for value in keys["ticker"].dropna().tolist() if str(value).strip()})
    if not tickers:
        raise ValueError("No tickers found in dataset artifacts.")
    start = min(keys["asof_used"].tolist())
    end = max(keys["asof_used"].tolist())
    cfg = builder.Config(stock_source=str(stock_source))
    theta = builder.ThetaClient(cfg.theta_base_url, timeout_s=cfg.timeout_s, verbose=False)
    raw_histories, adj_histories, _split_counts = builder.preload_stock_closes(
        theta=theta,
        tickers=tickers,
        start=start,
        end=end,
        cfg=cfg,
        stock_source=stock_source,
    )

    feature_days = {
        f"rv{int(lookback)}": int(lookback)
        for lookback in sorted(set(int(value) for value in lookbacks))
    }
    feature_maps: Dict[str, Dict[str, float]] = {
        feature_name: {}
        for feature_name in feature_days
    }
    for row in keys.itertuples(index=False):
        ticker = str(row.ticker)
        asof_used = row.asof_used
        spot_scale = _normalize_spot_scale(row.spot_scale_used)
        close_map = raw_histories.get(ticker, {}) if spot_scale == "raw" else adj_histories.get(ticker, {})
        key = _make_key(ticker, asof_used, spot_scale)
        for feature_name, days in feature_days.items():
            feature_maps[feature_name][key] = builder.realized_vol_proxy(close_map, asof_used, days)
    return feature_maps


def _reorder_columns(columns: List[str], new_columns: List[str], insert_before: str) -> List[str]:
    existing = [column for column in columns if column not in new_columns]
    if insert_before in existing:
        idx = existing.index(insert_before)
        return existing[:idx] + new_columns + existing[idx:]
    return existing + new_columns


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, suffix=".csv") as handle:
        temp_path = Path(handle.name)
    try:
        frame.to_csv(temp_path, index=False)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _backfill_artifact(
    *,
    path: Path,
    spec: ArtifactSpec,
    feature_maps: Dict[str, Dict[str, float]],
) -> None:
    frame = pd.read_csv(path)
    if frame.empty:
        print(f"[SKIP] {path.name}: empty file")
        return

    frame["ticker"] = frame["ticker"].astype("string").str.upper().str.strip()
    asof_dates = _normalize_asof_date(frame["asof_date"])
    spot_scale = frame["spot_scale_used"].astype("string").fillna("split_adj").map(_normalize_spot_scale)
    keys = [
        _make_key(ticker, asof_used, spot_scale_used)
        for ticker, asof_used, spot_scale_used in zip(frame["ticker"], asof_dates, spot_scale)
    ]
    key_series = pd.Series(keys, index=frame.index)

    inserted_columns = [column for column, _days in spec.columns]
    for column, _days in spec.columns:
        frame[column] = key_series.map(feature_maps[column])

    ordered_columns = _reorder_columns(frame.columns.tolist(), inserted_columns, spec.insert_before)
    frame = frame[ordered_columns]
    _write_csv_atomic(path, frame)
    print(f"[WRITE] {path.name}: rows={len(frame)} columns_added={','.join(inserted_columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill rv5/rv10 option-chain history features in place.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing main-dataset CSV artifacts.",
    )
    parser.add_argument(
        "--stock-source",
        type=str,
        default="auto",
        choices=["yfinance", "theta", "auto"],
        help="Historical close source used for realized-vol backfill.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    builder = _load_builder_module()
    keys = _collect_unique_keys(dataset_dir, ARTIFACT_SPECS)
    lookbacks = [days for spec in ARTIFACT_SPECS for _column, days in spec.columns]
    feature_maps = _build_feature_maps(
        keys=keys,
        builder=builder,
        lookbacks=lookbacks,
        stock_source=args.stock_source,
    )

    for spec in ARTIFACT_SPECS:
        path = dataset_dir / spec.name
        if not path.exists():
            print(f"[SKIP] {spec.name}: file not found")
            continue
        _backfill_artifact(path=path, spec=spec, feature_maps=feature_maps)


if __name__ == "__main__":
    main()
