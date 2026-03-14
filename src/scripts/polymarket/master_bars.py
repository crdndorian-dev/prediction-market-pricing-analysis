from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

MASTER_BAR_FREQS: tuple[str, ...] = ("1h", "1d")
MASTER_BAR_COLUMNS: List[str] = [
    "timestamp_utc",
    "market_id",
    "event_id",
    "event_slug",
    "market_slug",
    "ticker",
    "threshold",
    "week_friday",
    "expiry_date_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "bar_source",
    "written_by_run_id",
    "schema_version",
]
MASTER_BAR_KEY_COLUMNS: List[str] = ["market_id", "timestamp_utc"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def master_bar_path(bars_dir: Path, freq: str) -> Path:
    return bars_dir / f"{freq}.csv"


def normalize_master_bars(
    bars: pd.DataFrame,
    *,
    allowed_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if bars.empty:
        columns = list(allowed_columns or MASTER_BAR_COLUMNS)
        return pd.DataFrame(columns=columns)

    df = bars.copy()
    df["market_id"] = df.get("market_id").astype(str)
    df["timestamp_utc"] = pd.to_datetime(df.get("timestamp_utc"), utc=True, errors="coerce")
    df = df.dropna(subset=["market_id", "timestamp_utc"])

    for column in ("open", "high", "low", "close", "volume", "trade_count", "threshold"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["timestamp_utc"] = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    target_columns = list(allowed_columns or MASTER_BAR_COLUMNS)
    for column in target_columns:
        if column not in df.columns:
            df[column] = None
    df = df.reindex(columns=target_columns)
    df = df.drop_duplicates(subset=MASTER_BAR_KEY_COLUMNS, keep="last")
    df = df.sort_values(MASTER_BAR_KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    return df


def _read_master_bars(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MASTER_BAR_COLUMNS)
    df = pd.read_csv(path, dtype={"market_id": str}, low_memory=False)
    return normalize_master_bars(df)


def upsert_master_bars(bars: pd.DataFrame, bars_dir: Path, freq: str) -> Dict[str, object]:
    if freq not in MASTER_BAR_FREQS:
        raise ValueError(f"Unsupported master bars freq: {freq}")

    path = master_bar_path(bars_dir, freq)
    ensure_dir(path.parent)

    incoming = normalize_master_bars(bars)
    rows_written = len(incoming)
    if incoming.empty:
        return {
            "name": path.name,
            "path": str(path),
            "frequency": freq,
            "rows_written": 0,
            "rows_total": len(_read_master_bars(path)) if path.exists() else 0,
        }

    existing = _read_master_bars(path)
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = normalize_master_bars(merged)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    merged.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)

    return {
        "name": path.name,
        "path": str(path),
        "frequency": freq,
        "rows_written": rows_written,
        "rows_total": len(merged),
    }


def build_master_bar_rows(
    bars: pd.DataFrame,
    market_metadata: pd.DataFrame,
    *,
    bar_source: str,
    written_by_run_id: str,
    schema_version: str,
) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=MASTER_BAR_COLUMNS)

    metadata_columns = [
        "market_id",
        "event_id",
        "event_slug",
        "market_slug",
        "ticker",
        "threshold",
        "week_friday",
        "expiry_date_utc",
    ]
    metadata = market_metadata.reindex(columns=metadata_columns).drop_duplicates(subset=["market_id"])
    metadata["market_id"] = metadata["market_id"].astype(str)

    out = bars.copy()
    out["market_id"] = out["market_id"].astype(str)
    out = out.merge(metadata, on="market_id", how="left")
    out["bar_source"] = bar_source
    out["written_by_run_id"] = written_by_run_id
    out["schema_version"] = schema_version
    return normalize_master_bars(out)


def summarize_master_bar_file(path: Path, freq: str) -> Optional[Dict[str, object]]:
    if not path.exists() or not path.is_file():
        return None
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for _ in handle:
            row_count += 1
    return {
        "name": path.name,
        "path": str(path),
        "frequency": freq,
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
        "last_modified": path.stat().st_mtime,
    }


def master_bar_artifacts(bars_dir: Path) -> List[Dict[str, object]]:
    artifacts: List[Dict[str, object]] = []
    for freq in MASTER_BAR_FREQS:
        summary = summarize_master_bar_file(master_bar_path(bars_dir, freq), freq)
        if summary is not None:
            artifacts.append(summary)
    return artifacts


def migrate_partitioned_bars_to_master(bars_dir: Path) -> Dict[str, Dict[str, object]]:
    ensure_dir(bars_dir)
    summaries: Dict[str, Dict[str, object]] = {}
    old_dirs: List[Path] = []
    for freq in MASTER_BAR_FREQS:
        freq_dir = bars_dir / freq
        target_path = master_bar_path(bars_dir, freq)
        if not freq_dir.exists():
            summaries[freq] = {
                "name": target_path.name,
                "path": str(target_path),
                "frequency": freq,
                "rows_written": 0,
                "rows_total": len(_read_master_bars(target_path)) if target_path.exists() else 0,
            }
            continue

        frames: List[pd.DataFrame] = []
        for part_path in sorted(freq_dir.glob("market_id=*/date=*/bars.csv")):
            try:
                frame = pd.read_csv(part_path, dtype={"market_id": str}, low_memory=False)
            except Exception:
                continue
            if frame.empty:
                continue
            frames.append(frame)

        merged = normalize_master_bars(
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MASTER_BAR_COLUMNS)
        )
        tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
        merged.to_csv(tmp_path, index=False)
        os.replace(tmp_path, target_path)
        summaries[freq] = {
            "name": target_path.name,
            "path": str(target_path),
            "frequency": freq,
            "rows_written": len(merged),
            "rows_total": len(merged),
        }
        old_dirs.append(freq_dir)

    for old_dir in old_dirs:
        if old_dir.exists():
            shutil.rmtree(old_dir)

    return summaries
