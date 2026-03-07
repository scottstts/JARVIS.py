"""Runner executed inside the python_interpreter bubblewrap sandbox."""

from __future__ import annotations

import builtins
import importlib.metadata
import importlib.util
import json
import os
import re
import resource
import runpy
import sys
import sysconfig
from pathlib import Path

_CANONICAL_NAME_PATTERN = re.compile(r"[-_.]+")
_REQUIREMENT_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
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

    allowed_distributions = _resolve_allowed_distribution_closure(
        root_distributions=allowed_root_distributions,
        installed_distributions=installed_distributions,
    )
    root_to_distributions = {
        root: [_canonicalize_name(dist_name) for dist_name in distribution_names]
        for root, distribution_names in importlib.metadata.packages_distributions().items()
    }
    stdlib_roots = _stdlib_roots()
    site_package_roots = _site_package_roots()
    checked_roots: dict[str, bool] = {}
    original_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> object:
        if level > 0 or not name:
            return original_import(name, globals, locals, fromlist, level)
        root_name = name.split(".", 1)[0]
        is_allowed = checked_roots.get(root_name)
        if is_allowed is None:
            _raise_if_disallowed_import(
                root_name=root_name,
                blocked_roots=blocked_roots,
                allowed_distributions=allowed_distributions,
                root_to_distributions=root_to_distributions,
                stdlib_roots=stdlib_roots,
                site_package_roots=site_package_roots,
            )
            checked_roots[root_name] = True
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def _raise_if_disallowed_import(
    *,
    root_name: str,
    blocked_roots: set[str],
    allowed_distributions: set[str],
    root_to_distributions: dict[str, list[str]],
    stdlib_roots: tuple[Path, ...],
    site_package_roots: tuple[Path, ...],
) -> None:
    if root_name in blocked_roots:
        raise ImportError(f"Import of '{root_name}' is blocked in python_interpreter.")

    if root_name in sys.builtin_module_names:
        return

    spec = importlib.util.find_spec(root_name)
    if spec is None:
        return
    if spec.origin in {None, "built-in", "frozen"}:
        return

    origin_path = Path(spec.origin).resolve(strict=False)
    if any(origin_path.is_relative_to(site_root) for site_root in site_package_roots):
        distributions = root_to_distributions.get(root_name, [])
        if any(distribution in allowed_distributions for distribution in distributions):
            return
        raise ImportError(
            f"Third-party import '{root_name}' is not available in python_interpreter.",
        )

    if any(origin_path.is_relative_to(stdlib_root) for stdlib_root in stdlib_roots):
        return

    raise ImportError(
        f"Import of '{root_name}' is not allowed in python_interpreter.",
    )


def _resolve_allowed_distribution_closure(
    *,
    root_distributions: set[str],
    installed_distributions: dict[str, importlib.metadata.Distribution],
) -> set[str]:
    allowed_distributions: set[str] = set()
    pending = list(root_distributions)

    while pending:
        distribution_name = pending.pop()
        if distribution_name in allowed_distributions:
            continue

        distribution = installed_distributions.get(distribution_name)
        if distribution is None:
            continue

        allowed_distributions.add(distribution_name)
        for requirement in distribution.requires or ():
            dependency_name = _extract_requirement_name(requirement)
            if dependency_name is None:
                continue
            if dependency_name in installed_distributions and dependency_name not in allowed_distributions:
                pending.append(dependency_name)

    return allowed_distributions


def _installed_distributions() -> dict[str, importlib.metadata.Distribution]:
    distributions: dict[str, importlib.metadata.Distribution] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        distributions[_canonicalize_name(name)] = distribution
    return distributions


def _extract_requirement_name(requirement: str) -> str | None:
    match = _REQUIREMENT_NAME_PATTERN.match(requirement)
    if match is None:
        return None
    return _canonicalize_name(match.group(1))


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


if __name__ == "__main__":
    main()
