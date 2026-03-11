from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, MutableMapping

from support.script_paths import SCRIPTS_ROOT

_SKIP_EXPORTS = {
    "__builtins__",
    "__cached__",
    "__file__",
    "__loader__",
    "__name__",
    "__package__",
    "__spec__",
}


def _module_name_for_path(path: Path) -> str:
    stem = "_".join(path.with_suffix("").parts[-4:])
    sanitized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)
    return f"_scripts_impl_{sanitized}"


def load_impl_module(legacy_file: str, relative_impl_path: str) -> ModuleType:
    legacy_path = Path(legacy_file).resolve()
    impl_path = (SCRIPTS_ROOT / relative_impl_path).resolve()
    if not impl_path.exists():
        raise FileNotFoundError(f"Implementation module not found: {impl_path}")

    module_name = _module_name_for_path(impl_path)
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, impl_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load implementation module: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def reexport_module_globals(namespace: MutableMapping[str, Any], module: ModuleType) -> None:
    namespace["__doc__"] = module.__doc__
    for name, value in vars(module).items():
        if name in _SKIP_EXPORTS:
            continue
        namespace[name] = value
