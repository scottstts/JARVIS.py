"""Policy checks for the send_file tool."""

from __future__ import annotations

import re
from pathlib import Path

from ...types import ToolExecutionContext, ToolPolicyDecision

_GLOB_PATTERN = re.compile(r"[*?\[]")


class SendFilePolicy:
    """Restricts send_file to explicit non-secret files inside the workspace."""

    def authorize(
        self,
        *,
        path: str,
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        raw_path = path.strip()
        if not raw_path:
            return ToolPolicyDecision(
                allowed=False,
                reason="send_file requires a non-empty 'path'.",
            )
        if raw_path == "-":
            return ToolPolicyDecision(
                allowed=False,
                reason="send_file path '-' is not allowed.",
            )
        if raw_path.startswith("~") or _GLOB_PATTERN.search(raw_path):
            return ToolPolicyDecision(
                allowed=False,
                reason=f"send_file does not allow shell-expanded path '{raw_path}'.",
            )

        resolved = _resolve_workspace_relative_path(raw_path, context)
        if not _is_within_workspace(resolved, context.workspace_dir):
            return ToolPolicyDecision(
                allowed=False,
                reason=f"send_file may only read files inside {context.workspace_dir}.",
            )
        if _contains_dot_env_path(resolved):
            return ToolPolicyDecision(
                allowed=False,
                reason="send_file does not allow sending .env files or paths inside .env directories.",
            )

        return ToolPolicyDecision(allowed=True)


def _resolve_workspace_relative_path(raw_path: str, context: ToolExecutionContext) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve(strict=False)


def _is_within_workspace(path: Path, workspace_dir: Path) -> bool:
    workspace = workspace_dir.resolve(strict=False)
    try:
        path.relative_to(workspace)
        return True
    except ValueError:
        return False


def _contains_dot_env_path(path: Path) -> bool:
    return any(part == ".env" for part in path.parts)
