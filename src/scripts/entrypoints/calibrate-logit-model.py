#!/usr/bin/env python3
"""Canonical public entrypoint for the v2 calibration trainer."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from support.legacy_facade import load_impl_module, reexport_module_globals

_IMPL = load_impl_module(__file__, "model_training/calibration/calibrate_logit_model_v2.py")
reexport_module_globals(globals(), _IMPL)


if __name__ == "__main__":
    main(entry_script=__file__)
