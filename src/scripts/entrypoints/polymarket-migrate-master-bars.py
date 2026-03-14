#!/usr/bin/env python3
"""Migrate partitioned Polymarket bars_history trees into master 1h/1d CSVs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from support.script_paths import REPO_ROOT, SCRIPTS_ROOT as SUPPORT_SCRIPTS_ROOT, SRC_ROOT, prepend_sys_path

prepend_sys_path(REPO_ROOT)
prepend_sys_path(SRC_ROOT)
prepend_sys_path(SUPPORT_SCRIPTS_ROOT)

from polymarket.master_bars import migrate_partitioned_bars_to_master


DEFAULT_BARS_DIR = REPO_ROOT / "src" / "data" / "analysis" / "polymarket" / "bars_history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate partitioned Polymarket bars into master CSVs.")
    parser.add_argument("--bars-dir", type=str, default=str(DEFAULT_BARS_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bars_dir = Path(args.bars_dir)
    summary = migrate_partitioned_bars_to_master(bars_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
