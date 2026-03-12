#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "src" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from polymarket.weekly_history_io import (  # noqa: E402
    build_master_from_legacy_partitions,
    cleanup_legacy_partition_dirs,
)


DEFAULT_BARS_DIR = REPO_ROOT / "src" / "data" / "analysis" / "polymarket" / "bars_history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Polymarket legacy bars_history partitions into hourly/daily master CSV files."
    )
    parser.add_argument("--bars-dir", type=str, default=str(DEFAULT_BARS_DIR))
    parser.add_argument(
        "--keep-legacy",
        action="store_true",
        help="Build master CSVs but keep the legacy 1h/ and 1d/ partition directories.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing hourly_master.csv / daily_master.csv if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bars_dir = Path(args.bars_dir)

    results = []
    for freq in ("1h", "1d"):
        result = build_master_from_legacy_partitions(
            bars_dir,
            freq,
            overwrite=bool(args.force),
        )
        results.append(result)

    if not args.keep_legacy:
        cleanup_legacy_partition_dirs(bars_dir, freqs=["1h", "1d"])

    print(
        json.dumps(
            {
                "bars_dir": str(bars_dir),
                "cleanup_performed": not args.keep_legacy,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
