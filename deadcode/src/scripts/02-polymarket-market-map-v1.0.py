#!/usr/bin/env python3
"""Legacy compatibility wrapper. Canonical public entrypoint: polymarket-market-map.py."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _legacy_facade import load_impl_module, reexport_module_globals

_IMPL = load_impl_module(__file__, "polymarket/pipelines/market_map_v1.py")
reexport_module_globals(globals(), _IMPL)


if __name__ == "__main__":
    main()
