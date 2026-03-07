"""Policy checks for the python_interpreter tool."""

from __future__ import annotations

from typing import Any

from ..config import ToolSettings
from ..types import ToolExecutionContext, ToolPolicyDecision
from .paths import (
    contains_protected_relative_descendant,
    is_protected_relative_path,
    resolve_workspace_path,
)

_MAX_ARGS = 32
_MAX_ARG_CHARS = 512


class PythonInterpreterPolicy:
    """Restricts python_interpreter to explicit, sandboxable workspace access."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def authorize(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        code = _normalize_nullable_string(arguments.get("code"))
        script_path = _normalize_nullable_string(arguments.get("script_path"))
        args = _normalize_string_list(arguments.get("args"))
        read_paths = _normalize_string_list(arguments.get("read_paths"))
        write_paths = _normalize_string_list(arguments.get("write_paths"))

        if bool(code) == bool(script_path):
            return ToolPolicyDecision(
                allowed=False,
                reason="python_interpreter requires exactly one of 'code' or 'script_path'.",
            )

        if code is not None and len(code) > self._settings.python_interpreter_max_code_chars:
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "python_interpreter code length must be <= "
                    f"{self._settings.python_interpreter_max_code_chars} characters."
                ),
            )

        if len(args) > _MAX_ARGS:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"python_interpreter allows at most {_MAX_ARGS} args.",
            )
        for argument in args:
            if len(argument) > _MAX_ARG_CHARS:
                return ToolPolicyDecision(
                    allowed=False,
                    reason=(
                        "python_interpreter arg length must be <= "
                        f"{_MAX_ARG_CHARS} characters."
                    ),
                )

        if len(read_paths) > self._settings.python_interpreter_max_paths:
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "python_interpreter allows at most "
                    f"{self._settings.python_interpreter_max_paths} read_paths."
                ),
            )
        if len(write_paths) > self._settings.python_interpreter_max_paths:
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "python_interpreter allows at most "
                    f"{self._settings.python_interpreter_max_paths} write_paths."
                ),
            )

        if script_path is not None:
            script_decision = self._authorize_script_path(script_path=script_path, context=context)
            if not script_decision.allowed:
                return script_decision

        for raw_path in read_paths:
            decision = self._authorize_read_path(raw_path=raw_path, context=context)
            if not decision.allowed:
                return decision

        for raw_path in write_paths:
            decision = self._authorize_write_path(raw_path=raw_path, context=context)
            if not decision.allowed:
                return decision

        return ToolPolicyDecision(allowed=True)

    def _authorize_script_path(
        self,
        *,
        script_path: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        try:
            resolved, relative = resolve_workspace_path(
                script_path,
                context=context,
                require_exists=True,
            )
        except ValueError as exc:
            return ToolPolicyDecision(allowed=False, reason=str(exc))

        if not resolved.is_file():
            return ToolPolicyDecision(
                allowed=False,
                reason="python_interpreter 'script_path' must point to a file.",
            )
        if is_protected_relative_path(relative):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "python_interpreter does not allow scripts inside protected workspace "
                    "paths or .env paths."
                ),
            )
        return ToolPolicyDecision(allowed=True)

    def _authorize_read_path(
        self,
        *,
        raw_path: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        try:
            resolved, relative = resolve_workspace_path(
                raw_path,
                context=context,
                require_exists=True,
            )
        except ValueError as exc:
            return ToolPolicyDecision(allowed=False, reason=str(exc))

        if not resolved.is_file() and not resolved.is_dir():
            return ToolPolicyDecision(
                allowed=False,
                reason=f"python_interpreter read path '{raw_path}' must be a file or directory.",
            )
        if is_protected_relative_path(relative):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "python_interpreter read paths may not target protected workspace "
                    "paths or .env paths."
                ),
            )
        return ToolPolicyDecision(allowed=True)

    def _authorize_write_path(
        self,
        *,
        raw_path: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        try:
            resolved, relative = resolve_workspace_path(
                raw_path,
                context=context,
                require_exists=True,
            )
        except ValueError as exc:
            return ToolPolicyDecision(allowed=False, reason=str(exc))

        if not resolved.is_file() and not resolved.is_dir():
            return ToolPolicyDecision(
                allowed=False,
                reason=f"python_interpreter write path '{raw_path}' must be a file or directory.",
            )
        if is_protected_relative_path(relative):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "python_interpreter write paths may not target protected workspace "
                    "paths or .env paths."
                ),
            )
        if contains_protected_relative_descendant(relative):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "python_interpreter write paths may not be ancestors of protected "
                    "workspace paths. Use a narrower writable directory."
                ),
            )
        return ToolPolicyDecision(allowed=True)


def _normalize_nullable_string(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [str(value).strip()]
    return [str(item).strip() for item in value if str(item).strip()]
