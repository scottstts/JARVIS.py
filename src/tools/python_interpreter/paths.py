"""Workspace path helpers for the python_interpreter tool."""

from __future__ import annotations

import re
from pathlib import Path

from ..types import ToolExecutionContext

_GLOB_PATTERN = re.compile(r"[*?\[]")


class PythonInterpreterPathError(ValueError):
    """Raised when a python_interpreter path violates workspace policy."""


def resolve_workspace_path(
    raw_path: str,
    *,
    context: ToolExecutionContext,
    require_exists: bool,
) -> tuple[Path, Path]:
    normalized = raw_path.strip()
    if not normalized:
        raise PythonInterpreterPathError("Path cannot be empty.")
    if normalized == "-":
        raise PythonInterpreterPathError("Path '-' is not allowed.")
    if normalized.startswith("~") or _GLOB_PATTERN.search(normalized):
        raise PythonInterpreterPathError(
            f"Shell-expanded path '{normalized}' is not allowed.",
        )

    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate

    if require_exists:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PythonInterpreterPathError(
                f"Path '{normalized}' does not exist.",
            ) from exc
    else:
        try:
            resolved_parent = candidate.parent.resolve(strict=True)
        except FileNotFoundError as exc:
            raise PythonInterpreterPathError(
                f"Parent directory for '{normalized}' does not exist.",
            ) from exc
        resolved = resolved_parent / candidate.name

    workspace = context.workspace_dir.resolve(strict=False)
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as exc:
        raise PythonInterpreterPathError(
            f"Path '{normalized}' must stay inside {context.workspace_dir}.",
        ) from exc

    return resolved, relative if relative.parts else Path(".")
