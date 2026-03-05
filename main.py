"""Repo-root bootstrap for the combined Jarvis runtime."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def main() -> None:
    module = _load_src_main_module()
    module.main()


def _load_src_main_module() -> ModuleType:
    src_dir = Path(__file__).resolve().parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    module_path = src_dir / "main.py"
    spec = importlib.util.spec_from_file_location("jarvis_src_main", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load main module from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
