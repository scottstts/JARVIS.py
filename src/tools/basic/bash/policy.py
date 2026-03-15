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
    "python -m pip install",
    "python3 -m pip install",
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
_SYSTEM_DESTINATION_PATTERN = re.compile(
    r"\b(?:install|cp|mv|ln|chmod|chown)\b[^\n]*\s"
    r"(/usr/local/bin\S*|/usr/bin\S*|/bin/\S*|/sbin/\S*|/etc/\S*|/opt/\S*)\s*$"
)
_SYSTEM_REDIRECT_PATTERN = re.compile(
    r"(?:^|[;&|])[^#\n]*(?:>|>>)\s*"
    r"(/usr/local/bin\S*|/usr/bin\S*|/bin/\S*|/sbin/\S*|/etc/\S*|/opt/\S*)"
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
            summary = "Run a bash command that changes the system or installs tooling."
        if not details:
            details = (
                "I want to run a bash command that appears to install, build, or modify "
                "system-level tooling. Review the exact command before approving."
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
            },
        )


def _approval_detector_reason(command: str) -> str | None:
    lowered = " ".join(command.strip().lower().split())
    if not lowered:
        return None

    if any(keyword in lowered for keyword in _INSTALL_KEYWORDS):
        return "matched install or package-manager command pattern"

    if ("curl " in lowered or "wget " in lowered) and (
        "| sh" in lowered
        or "| bash" in lowered
        or "| zsh" in lowered
        or "| python" in lowered
        or "| python3" in lowered
    ):
        return "matched remote installer pipeline pattern"

    if _SYSTEM_DESTINATION_PATTERN.search(lowered):
        return "matched system-path mutation pattern"

    if _SYSTEM_REDIRECT_PATTERN.search(lowered):
        return "matched system-path mutation pattern"

    return None
