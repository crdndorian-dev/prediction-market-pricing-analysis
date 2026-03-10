from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


IMMUTABLE_REFRESH_COLUMNS = {
    "ingested_at_utc",
    "last_refreshed_at_utc",
    "created_at_utc",
}


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: Sequence[str]


pm_raw_price_history = TableSpec(
    name="pm_raw_price_history",
    columns=(
        "run_id",
        "timestamp_utc",
        "market_id",
        "token_id",
        "token_role",
        "price",
        "ingested_at_utc",
    ),
)

pm_raw_run_manifest = TableSpec(
    name="pm_raw_run_manifest",
    columns=(
        "run_id",
        "status",
        "label",
        "last_refreshed_at_utc",
    ),
)


def default_upsert_update_columns(table: TableSpec, conflict_columns: Iterable[str]) -> List[str]:
    excluded = set(conflict_columns) | set(IMMUTABLE_REFRESH_COLUMNS)
    return [column for column in table.columns if column not in excluded]
