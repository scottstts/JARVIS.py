"""Policy checks for the python_interpreter tool."""

from __future__ import annotations

from typing import Any

from ...config import ToolSettings
from ...types import ToolExecutionContext, ToolPolicyDecision
from .paths import resolve_workspace_path

_MAX_ARGS = 32
_MAX_ARG_CHARS = 512


class PythonInterpreterPolicy:
    """Validates structured python_interpreter requests before direct execution."""

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

        if script_path is not None:
            script_decision = self._authorize_script_path(script_path=script_path, context=context)
            if not script_decision.allowed:
                return script_decision

        return ToolPolicyDecision(allowed=True)

    def _authorize_script_path(
        self,
        *,
        script_path: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        try:
            resolved, _ = resolve_workspace_path(
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
