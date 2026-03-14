#!/usr/bin/env python3
"""Refresh exact run-local pRN data for an existing Polymarket history run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from typing import Iterable, Optional, Set

from support.script_paths import REPO_ROOT, SCRIPTS_ROOT, SRC_ROOT, prepend_sys_path

BACKEND_ROOT = SRC_ROOT / "webapp" / "backend"

for candidate in (REPO_ROOT, SRC_ROOT, SCRIPTS_ROOT, BACKEND_ROOT):
    prepend_sys_path(candidate)

from app.services.polymarket_run_prn import (
    refresh_run_local_prn_dataset,
    resolve_polymarket_run_dir,
)
from app.services.script_entrypoints import POLYMARKET_MARKETS_REFRESH_SCRIPT

MARKETS_REFRESH_SCRIPT = POLYMARKET_MARKETS_REFRESH_SCRIPT.path


def _parse_week_set(raw: Optional[str]) -> Optional[Set[date]]:
    if not raw:
        return None
    weeks: Set[date] = set()
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        weeks.add(date.fromisoformat(value))
    return weeks or None


def _sorted_rebuild_weeks(
    *,
    affected_weeks: Iterable[date],
    missing_markets_pairs: Iterable[tuple[str, date]],
) -> list[date]:
    weeks = {week for week in affected_weeks}
    weeks.update(week for _, week in missing_markets_pairs)
    return sorted(weeks)


def _command_output_tail(*, stdout: str, stderr: str, max_lines: int = 20) -> str:
    lines = [
        line.rstrip()
        for line in [*(stdout or "").splitlines(), *(stderr or "").splitlines()]
        if line and line.strip()
    ]
    if not lines:
        return ""
    tail = lines[-max_lines:]
    return "\n".join(tail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh exact run-local pRN data for a polymarket weekly-history run.",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Weekly-history run id. Defaults to latest.")
    parser.add_argument(
        "--prn-dataset",
        type=str,
        default=None,
        help="Optional source training CSV to seed/run exact pRN backfill from.",
    )
    parser.add_argument(
        "--week-fridays",
        type=str,
        default=None,
        help="Optional comma-separated YYYY-MM-DD list to scope the refresh.",
    )
    parser.add_argument(
        "--no-refresh-markets",
        action="store_true",
        help="Only update the run-local pRN dataset; do not rebuild markets_prn_hourly/snapshot artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = resolve_polymarket_run_dir(args.run_id)
    week_filter = _parse_week_set(args.week_fridays)

    print(f"[Run pRN] run_id={run_dir.name}", flush=True)
    if week_filter:
        print(
            f"[Run pRN] week_filter={','.join(sorted(week.isoformat() for week in week_filter))}",
            flush=True,
        )

    result = refresh_run_local_prn_dataset(
        run_dir,
        week_fridays=week_filter,
        explicit_prn_dataset=args.prn_dataset,
    )
    print(
        "[Run pRN] training_file="
        f"{result.training_path} "
        f"seeded_from_source={result.seeded_from_source} "
        f"used_inferred_defaults={result.used_inferred_defaults}",
        flush=True,
    )
    print(
        "[Run pRN] coverage "
        f"required_pairs={len(result.required_pairs)} "
        f"missing_before={len(result.missing_pairs_before)} "
        f"missing_after={len(result.missing_pairs_after)} "
        f"markets_missing_before={len(result.missing_markets_pairs_before)}",
        flush=True,
    )

    if result.missing_pairs_after:
        sample = ", ".join(
            f"{ticker}:{week.isoformat()}"
            for ticker, week in sorted(result.missing_pairs_after)[:10]
        )
        print(
            "[Run pRN] WARNING unresolved exact pRN coverage remains "
            f"missing_pairs_after={len(result.missing_pairs_after)} "
            f"sample={sample}",
            flush=True,
        )

    if args.no_refresh_markets:
        print("[Run pRN] Skipping markets artifact refresh.", flush=True)
        return

    rebuild_weeks = _sorted_rebuild_weeks(
        affected_weeks=result.affected_week_fridays,
        missing_markets_pairs=result.missing_markets_pairs_before,
    )
    if not rebuild_weeks:
        print("[Run pRN] markets_prn_hourly already covers the requested scope.", flush=True)
        return

    for week in rebuild_weeks:
        cmd = [
            sys.executable,
            str(MARKETS_REFRESH_SCRIPT),
            "--run-id",
            run_dir.name,
            "--week-friday",
            week.isoformat(),
            "--replace-week",
            "--skip-build-features-append",
            "--prn-dataset",
            str(result.training_path),
        ]
        print(f"[Run pRN] Rebuilding markets week={week.isoformat()}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
        if proc.returncode != 0:
            tail = _command_output_tail(stdout=proc.stdout or "", stderr=proc.stderr or "")
            message = f"Markets refresh rebuild failed for week {week.isoformat()}."
            if tail:
                message = f"{message}\n{tail}"
            raise RuntimeError(message)

    print("[Run pRN] Refresh complete.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Run pRN] ERROR {exc}", file=sys.stderr, flush=True)
        raise
