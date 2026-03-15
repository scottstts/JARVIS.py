"""Bash-specific policy implementation."""

from __future__ import annotations

import re
from typing import Any

from ...config import ToolSettings
from ...types import ToolExecutionContext, ToolPolicyDecision

_INSTALL_KEYWORDS = (
    "apt install",
    "apt-get install",
    "apt-get update",
    "pip install",
    "pip3 install",
    "uv tool install",
    "uv pip install",
    "npm install -g",
    "npm uninstall -g",
    "pnpm add -g",
    "yarn global add",
    "cargo install",
    "go install",
    "brew install",
    "brew tap",
    "gem install",
    "pipx install",
    "poetry add",
)
_COMMAND_PREFIX = r"(?:^|[;&|()]\s*|\n\s*)"
_OPTIONAL_SUDO = r"(?:sudo\s+)?"
_OPTIONAL_ENV_WRAPPER = r"(?:env\s+)?"
_OPTIONAL_ENV_ASSIGNMENTS = r"(?:[a-z_][a-z0-9_]*=[^\s;&|()]+\s+)*"
_PYTHON_EXECUTABLE = r"(?:python(?:\d+(?:\.\d+)*)?|/(?:[\w.-]+/)*python(?:\d+(?:\.\d+)*)?)"
_HARD_DENY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            _COMMAND_PREFIX
            + _OPTIONAL_SUDO
            + _OPTIONAL_ENV_WRAPPER
            + _OPTIONAL_ENV_ASSIGNMENTS
            + _PYTHON_EXECUTABLE
            + r"\b"
        ),
        "Direct Python execution via bash is denied; use python_interpreter instead.",
    ),
    (
        re.compile(
            _COMMAND_PREFIX
            + _OPTIONAL_SUDO
            + _OPTIONAL_ENV_WRAPPER
            + _OPTIONAL_ENV_ASSIGNMENTS
            + r"uv\s+run(?:\s+[^\n;&|()]+)*\s+python(?:\d+(?:\.\d+)*)?\b"
        ),
        "Direct Python execution via bash is denied; use python_interpreter instead.",
    ),
    (
        re.compile(
            _COMMAND_PREFIX
            + _OPTIONAL_SUDO
            + r"apt(?:-get)?\s+(?:[^;&|()\n]+\s+)?(?:upgrade|full-upgrade|dist-upgrade)\b"
        ),
        "tool_runtime OS upgrade commands are denied.",
    ),
    (
        re.compile(_COMMAND_PREFIX + _OPTIONAL_SUDO + r"do-release-upgrade\b"),
        "tool_runtime OS upgrade commands are denied.",
    ),
    (
        re.compile(
            _COMMAND_PREFIX
            + _OPTIONAL_SUDO
            + r"(?:systemctl|service|init|telinit|reboot|shutdown|poweroff|halt)\b"
        ),
        "tool_runtime service and init control commands are denied.",
    ),
    (
        re.compile(
            _COMMAND_PREFIX
            + _OPTIONAL_SUDO
            + r"(?:mount|umount|swapon|swapoff|modprobe|insmod)\b"
        ),
        "tool_runtime mount, kernel, and low-level admin commands are denied.",
    ),
    (
        re.compile(_COMMAND_PREFIX + _OPTIONAL_SUDO + r"sysctl\s+-w\b"),
        "tool_runtime mount, kernel, and low-level admin commands are denied.",
    ),
    (
        re.compile(_COMMAND_PREFIX + _OPTIONAL_SUDO + r"(?:docker|podman|nerdctl)\b"),
        "tool_runtime container-runtime recursion is denied.",
    ),
)
_SYSTEM_DESTINATION_PATTERN = re.compile(
    r"\b(?:install|cp|mv|ln|chmod|chown)\b[^\n]*\s"
    r"(/usr/local/bin\S*|/usr/bin\S*|/bin/\S*|/sbin/\S*|/etc/\S*|/opt/\S*|/var/\S*|/root/\S*)\s*$"
)
_SYSTEM_REDIRECT_PATTERN = re.compile(
    r"(?:^|[;&|])[^#\n]*(?:>|>>)\s*"
    r"(/usr/local/bin\S*|/usr/bin\S*|/bin/\S*|/sbin/\S*|/etc/\S*|/opt/\S*|/var/\S*|/root/\S*)"
)


class BashCommandPolicy:
    """Thin validation for the sandboxed bash runtime."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    def authorize(
        self,
        *,
        command: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        _ = context

        if not command.strip():
            return ToolPolicyDecision(allowed=False, reason="bash command cannot be empty.")
        if "\x00" in command:
            return ToolPolicyDecision(
                allowed=False,
                reason="bash command cannot contain null bytes.",
            )

        hard_deny_reason = _hard_deny_reason(command)
        if hard_deny_reason is not None:
            return ToolPolicyDecision(allowed=False, reason=hard_deny_reason)

        if self._settings.bash_dangerously_skip_permission:
            return ToolPolicyDecision(allowed=True)

        detector_reason = _approval_detector_reason(command)
        if detector_reason is None:
            return ToolPolicyDecision(allowed=True)

        approved_action = context.approved_action or {}
        if (
            approved_action.get("kind") == "bash_command"
            and approved_action.get("command") == command
        ):
            return ToolPolicyDecision(allowed=True)

        summary = str(arguments.get("approval_summary", "")).strip()
        details = str(arguments.get("approval_details", "")).strip()
        inspection_url = str(arguments.get("inspection_url", "")).strip()
        if not summary:
            summary = "Run a bash command that mutates the isolated tool_runtime container."
        if not details:
            details = (
                "I want to run a bash command that appears to install, build, or modify "
                "system-level tooling inside the isolated tool_runtime container. "
                "Review the exact command before approving."
            )

        return ToolPolicyDecision(
            allowed=False,
            reason="bash command requires explicit approval.",
            approval_request={
                "kind": "bash_command",
                "summary": summary,
                "details": details,
                "inspection_url": inspection_url or None,
                "command": command,
                "detector_reason": detector_reason,
                "target_runtime": "tool_runtime",
                "runtime_location": "tool_runtime_container",
            },
        )


def _hard_deny_reason(command: str) -> str | None:
    lowered = command.lower()
    for pattern, reason in _HARD_DENY_PATTERNS:
        if pattern.search(lowered):
            return reason
    return None


def _approval_detector_reason(command: str) -> str | None:
    lowered = " ".join(command.strip().lower().split())
    if not lowered:
        return None

    if any(keyword in lowered for keyword in _INSTALL_KEYWORDS):
        return "matched install or package-manager mutation pattern for tool_runtime"

    if ("curl " in lowered or "wget " in lowered) and (
        "| sh" in lowered
        or "| bash" in lowered
        or "| zsh" in lowered
        or "| python" in lowered
        or "| python3" in lowered
    ):
        return "matched remote installer pipeline pattern for tool_runtime"

    if _SYSTEM_DESTINATION_PATTERN.search(lowered):
        return "matched non-workspace system-path mutation pattern in tool_runtime"

    if _SYSTEM_REDIRECT_PATTERN.search(lowered):
        return "matched non-workspace system-path mutation pattern in tool_runtime"

    return None
