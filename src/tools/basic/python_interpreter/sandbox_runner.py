"""Runner executed inside the python_interpreter bubblewrap sandbox."""

from __future__ import annotations

import builtins
import importlib.metadata
import json
import os
import re
import resource
import runpy
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path

_CANONICAL_NAME_PATTERN = re.compile(r"[-_.]+")
_BLOCKED_OS_FUNCTIONS = (
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "fork",
    "forkpty",
    "popen",
    "posix_spawn",
    "posix_spawnp",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
)


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError("python_interpreter sandbox runner requires one config-path arg.")

    config = _load_config(Path(sys.argv[1]))
    _apply_resource_limits(config)
    _install_process_guards()
    _install_import_guard(config)

    script_path = Path(str(config["script_path"]))
    args = [str(argument) for argument in config.get("args", []) or []]
    sys.argv = [str(script_path), *args]
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    runpy.run_path(str(script_path), run_name="__main__")


def _load_config(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_resource_limits(config: dict[str, object]) -> None:
    _set_limit(resource.RLIMIT_AS, int(config["memory_limit_bytes"]))
    _set_limit(resource.RLIMIT_FSIZE, int(config["file_size_limit_bytes"]))
    _set_limit(resource.RLIMIT_CPU, int(config["cpu_time_limit_seconds"]))
    _set_limit(resource.RLIMIT_CORE, 0)
    _set_limit(resource.RLIMIT_NOFILE, 64)
    sys.dont_write_bytecode = True


def _set_limit(limit_name: int, value: int) -> None:
    resource.setrlimit(limit_name, (value, value))


def _install_process_guards() -> None:
    def _deny_process_spawn(*args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise PermissionError("python_interpreter does not allow spawning child processes.")

    for function_name in _BLOCKED_OS_FUNCTIONS:
        if hasattr(os, function_name):
            setattr(os, function_name, _deny_process_spawn)

    _patch_low_level_process_modules(_deny_process_spawn)
    _patch_subprocess_module(_deny_process_spawn)


def _patch_low_level_process_modules(
    deny_process_spawn: Callable[..., None],
) -> None:
    try:
        import _posixsubprocess
    except ImportError:
        return

    if hasattr(_posixsubprocess, "fork_exec"):
        setattr(_posixsubprocess, "fork_exec", deny_process_spawn)


def _patch_subprocess_module(
    deny_process_spawn: Callable[..., None],
) -> None:
    try:
        import subprocess
    except ImportError:
        return

    class _DeniedPopen:
        def __init__(self, *args: object, **kwargs: object) -> None:
            deny_process_spawn(*args, **kwargs)

    subprocess.Popen = _DeniedPopen
    for attribute_name in (
        "call",
        "check_call",
        "check_output",
        "getoutput",
        "getstatusoutput",
        "run",
    ):
        if hasattr(subprocess, attribute_name):
            setattr(subprocess, attribute_name, deny_process_spawn)


def _install_import_guard(config: dict[str, object]) -> None:
    blocked_roots = {
        str(root).strip()
        for root in config.get("blocked_import_roots", []) or []
        if str(root).strip()
    }
    allowed_root_distributions = {
        _canonicalize_name(str(name))
        for name in config.get("allowed_packages", []) or []
        if str(name).strip()
    }
    installed_distributions = _installed_distributions()
    missing_distributions = sorted(
        name for name in allowed_root_distributions if name not in installed_distributions
    )
    if missing_distributions:
        raise RuntimeError(
            "Configured python_interpreter packages are missing from the dedicated venv: "
            + ", ".join(missing_distributions)
        )

    workspace_root = Path("/workspace").resolve(strict=False)
    trusted_roots = (
        workspace_root,
        *_stdlib_roots(),
        *_site_package_roots(),
    )
    validated_modules: set[str] = set()
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> object:
        if level == 0 and name:
            root_name = name.split(".", 1)[0]
            if root_name in blocked_roots:
                raise ImportError(f"Import of '{root_name}' is blocked in python_interpreter.")

        before_modules = set(sys.modules)
        module = original_import(name, globals, locals, fromlist, level)
        _validate_loaded_modules(
            before_modules=before_modules,
            validated_modules=validated_modules,
            trusted_roots=trusted_roots,
            workspace_root=workspace_root,
        )
        if level == 0 and name:
            _validate_module_name(
                name.split(".", 1)[0],
                validated_modules=validated_modules,
                trusted_roots=trusted_roots,
                workspace_root=workspace_root,
            )
        return module

    builtins.__import__ = guarded_import


def _validate_loaded_modules(
    *,
    before_modules: set[str],
    validated_modules: set[str],
    trusted_roots: tuple[Path, ...],
    workspace_root: Path,
) -> None:
    for module_name in sorted(set(sys.modules) - before_modules):
        if module_name in validated_modules:
            continue
        module = sys.modules.get(module_name)
        if module is None:
            continue
        _raise_if_untrusted_module(
            module_name=module_name,
            module=module,
            trusted_roots=trusted_roots,
            workspace_root=workspace_root,
        )
        validated_modules.add(module_name)


def _validate_module_name(
    module_name: str,
    *,
    validated_modules: set[str],
    trusted_roots: tuple[Path, ...],
    workspace_root: Path,
) -> None:
    if module_name in validated_modules:
        return
    module = sys.modules.get(module_name)
    if module is None:
        return
    _raise_if_untrusted_module(
        module_name=module_name,
        module=module,
        trusted_roots=trusted_roots,
        workspace_root=workspace_root,
    )
    validated_modules.add(module_name)


def _raise_if_untrusted_module(
    *,
    module_name: str,
    module: object,
    trusted_roots: tuple[Path, ...],
    workspace_root: Path,
) -> None:
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None)
    if origin in {None, "built-in", "frozen"}:
        search_locations = getattr(spec, "submodule_search_locations", None)
        if not search_locations:
            return
    else:
        origin_path = Path(str(origin)).resolve(strict=False)
        if _is_under_trusted_root(origin_path, trusted_roots):
            if origin_path == workspace_root or origin_path.is_relative_to(workspace_root):
                if not _is_safe_workspace_module_origin(origin_path):
                    raise ImportError(
                        f"Native extension import '{module_name}' is not allowed from /workspace.",
                    )
            return
        raise ImportError(
            f"Import of '{module_name}' is not allowed in python_interpreter.",
        )

    search_locations = getattr(spec, "submodule_search_locations", None)
    if search_locations is None:
        return

    resolved_locations = [
        Path(str(location)).resolve(strict=False)
        for location in search_locations
    ]
    if resolved_locations and all(
        _is_under_trusted_root(location, trusted_roots)
        for location in resolved_locations
    ):
        return

    raise ImportError(
        f"Import of '{module_name}' is not allowed in python_interpreter.",
    )


def _installed_distributions() -> dict[str, importlib.metadata.Distribution]:
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        distributions[_canonicalize_name(name)] = distribution
    return distributions


def _canonicalize_name(name: str) -> str:
    return _CANONICAL_NAME_PATTERN.sub("-", name).strip().lower()


def _stdlib_roots() -> tuple[Path, ...]:
    roots = {
        Path(path).resolve(strict=False)
        for path in (
            sysconfig.get_path("stdlib"),
            sysconfig.get_path("platstdlib"),
        )
        if path
    }
    return tuple(sorted(roots, key=str))


def _site_package_roots() -> tuple[Path, ...]:
    roots = {
        Path(path).resolve(strict=False)
        for path in (
            sysconfig.get_path("purelib"),
            sysconfig.get_path("platlib"),
        )
        if path
    }
    return tuple(sorted(roots, key=str))


def _is_under_trusted_root(path: Path, trusted_roots: tuple[Path, ...]) -> bool:
    for trusted_root in trusted_roots:
        try:
            path.relative_to(trusted_root)
            return True
        except ValueError:
            continue
    return False


def _is_safe_workspace_module_origin(path: Path) -> bool:
    return path.suffix in {"", ".py", ".pyc", ".pyi"}


if __name__ == "__main__":
    main()
