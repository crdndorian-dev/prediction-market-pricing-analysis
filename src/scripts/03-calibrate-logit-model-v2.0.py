#!/usr/bin/env python3
"""
03-calibrate-logit-model-v2.0.py

Option-chain probabilistic calibration trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from calibration.calibrate_v2_core import main as core_main

SCRIPT_VERSION = "v2.0.0"


if __name__ == "__main__":
    core_main(entry_script=__file__)
