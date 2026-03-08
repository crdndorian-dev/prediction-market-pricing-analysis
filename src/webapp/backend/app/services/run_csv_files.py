from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


PRESERVED_RUNTIME_CSVS = frozenset(
    {
        "price_history.csv",
        "weekly_markets.csv",
        "weekly_events.csv",
        "markets_prn_hourly.csv",
        "dim_market_weekly.csv",
        "snapshot_daily.csv",
    }
)

_ROW_IDENTITY_OPTIONS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    "price_history.csv": (
        ("timestamp_utc", "market_id", "token_id", "token_role"),
        ("timestamp_utc", "market_id", "token_role"),
        ("timestamp_utc", "ticker", "threshold", "token_role"),
    ),
    "weekly_markets.csv": (
        ("market_id",),
        ("market_slug",),
        ("event_id", "ticker", "threshold", "week_friday"),
    ),
    "markets_prn_hourly.csv": (
        ("market_id", "timestamp_utc"),
        ("ticker", "threshold", "timestamp_utc", "week_friday"),
    ),
}


def to_kebab_case(value: str) -> str:
    raw = value.strip()
    if not raw:
        return ""
    raw = re.sub(r"[\s_]+", "-", raw)
    raw = re.sub(r"[^a-zA-Z0-9-]", "", raw)
    raw = re.sub(r"-{2,}", "-", raw)
    return raw.strip("-").lower()


def prefixed_run_csv_name(run_dir: Path, filename: str) -> str:
    stem = Path(filename).stem
    normalized_stem = to_kebab_case(stem) or "csv"
    return f"{run_dir.name}-{normalized_stem}.csv"


def get_run_csv_paths(run_dir: Path, filename: str) -> List[Path]:
    candidates = [run_dir / filename, run_dir / prefixed_run_csv_name(run_dir, filename)]
    paths: List[Path] = []
    for path in candidates:
        if path.exists() and path.is_file() and path not in paths:
            paths.append(path)
    return paths


def primary_run_csv_path(run_dir: Path, filename: str) -> Optional[Path]:
    paths = get_run_csv_paths(run_dir, filename)
    return paths[0] if paths else None


def has_run_csv(run_dir: Path, filename: str) -> bool:
    return primary_run_csv_path(run_dir, filename) is not None


def combined_run_csv_size(run_dir: Path, filename: str) -> int:
    return sum(path.stat().st_size for path in get_run_csv_paths(run_dir, filename))


def dedupe_columns_for_filename(
    filename: str,
    columns: Sequence[str],
) -> Optional[List[str]]:
    column_set = set(columns)
    for option in _ROW_IDENTITY_OPTIONS.get(filename, ()):
        if all(column in column_set for column in option):
            return list(option)
    return None


def dedupe_merged_dataframe(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    if df.empty:
        return df
    dedupe_columns = dedupe_columns_for_filename(filename, list(df.columns))
    if dedupe_columns:
        return df.drop_duplicates(subset=dedupe_columns, keep="first").reset_index(drop=True)
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def _row_identity_from_mapping(filename: str, row: Mapping[str, object]) -> Tuple[str, ...]:
    for option in _ROW_IDENTITY_OPTIONS.get(filename, ()):
        values: List[str] = []
        for column in option:
            value = row.get(column)
            if value is None or value == "":
                values = []
                break
            values.append(str(value))
        if values:
            return (filename, *values)
    return (
        filename,
        *(f"{key}={value}" for key, value in sorted((key, row.get(key, "")) for key in row.keys())),
    )


def iter_deduped_csv_rows(paths: Sequence[Path], filename: str) -> Iterator[dict]:
    seen = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                identity = _row_identity_from_mapping(filename, row)
                if identity in seen:
                    continue
                seen.add(identity)
                yield row
