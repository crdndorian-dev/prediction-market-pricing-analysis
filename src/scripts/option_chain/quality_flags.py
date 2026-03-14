from __future__ import annotations

from datetime import date
from typing import AbstractSet, Iterable, Mapping

INFORMATIONAL_QUALITY_FLAGS = frozenset(
    {
        "flag_prn_monotone_adjusted",
        "flag_split_event_context",
    }
)


def counted_quality_flag_columns(flag_columns: Iterable[str]) -> list[str]:
    return [column for column in flag_columns if column not in INFORMATIONAL_QUALITY_FLAGS]


def compute_quality_issue_count(quality_flags: Mapping[str, object]) -> int:
    return sum(
        1
        for name, enabled in quality_flags.items()
        if name not in INFORMATIONAL_QUALITY_FLAGS and bool(enabled)
    )


def is_split_context_date(asof_date: date, split_dates: AbstractSet[date]) -> bool:
    return asof_date in split_dates
