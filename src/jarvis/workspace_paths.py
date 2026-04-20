"""Shared workspace path resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path

_CONTAINER_WORKSPACE = Path("/workspace")


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def _host_workspace_error_message() -> str:
    return (
        "AGENT_WORKSPACE must be explicitly set for host runs. "
        "The default '/workspace' path is only valid inside the container. "
        "Set AGENT_WORKSPACE to a real host path, typically the same value as AGENT_ROOT."
    )


def resolve_workspace_dir(*, error_type: type[Exception] = ValueError) -> Path:
    """Resolve the active workspace root for the current runtime."""

    explicit_workspace = _optional_env("AGENT_WORKSPACE")
    if explicit_workspace is not None:
        return Path(explicit_workspace).expanduser()

    if _running_in_container():
        return _CONTAINER_WORKSPACE

    raise error_type(_host_workspace_error_message())


def resolve_workspace_child(
    *,
    env_name: str,
    configured_default: str | None,
    workspace_dir: Path,
    child_name: str,
) -> Path:
    """Resolve a workspace-derived child path with explicit env overrides."""

    explicit_path = _optional_env(env_name)
    if explicit_path is not None:
        return Path(explicit_path).expanduser()

    if _optional_env("AGENT_WORKSPACE") is not None:
        return workspace_dir / child_name

    if configured_default is not None and configured_default.strip():
        return Path(configured_default).expanduser()

    return workspace_dir / child_name
