#!/usr/bin/env python3
"""
calibrate-logit-model.py

Option-chain probabilistic calibration trainer.
"""

from __future__ import annotations

import argparse
import sys

from support.script_paths import REPO_ROOT, SRC_ROOT, prepend_sys_path

prepend_sys_path(REPO_ROOT)
prepend_sys_path(SRC_ROOT)

from calibration.calibrate_v2_core import main as core_main

SCRIPT_VERSION = "v2.0.0"


def build_cli_contract_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model-kind",
        choices=["calibrate", "mixed", "both"],
        default="calibrate",
    )
    parser.add_argument(
        "--training-mode",
        choices=["pretrain", "finetune", "joint", "two_stage"],
        default="two_stage",
    )
    parser.add_argument(
        "--feature-sources",
        choices=["options", "pm", "both"],
        default="both",
    )
    parser.add_argument(
        "--mixed-model",
        choices=["residual", "blend"],
        default="residual",
    )
    return parser


def main(*, entry_script: str | None = None) -> None:
    core_main(entry_script=entry_script or __file__)


if __name__ == "__main__":
    main()
