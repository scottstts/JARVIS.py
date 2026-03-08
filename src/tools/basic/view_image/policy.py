"""Policy checks for the view_image tool."""

from __future__ import annotations

import re
from pathlib import Path

from ...types import ToolExecutionContext, ToolPolicyDecision

_GLOB_PATTERN = re.compile(r"[*?\[]")


class ViewImagePolicy:
    """Restricts view_image to explicit workspace-relative image paths."""

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
                reason="view_image requires a non-empty 'path'.",
            )
        if raw_path == "-":
            return ToolPolicyDecision(
                allowed=False,
                reason="view_image path '-' is not allowed.",
            )
        if raw_path.startswith("~") or _GLOB_PATTERN.search(raw_path):
            return ToolPolicyDecision(
                allowed=False,
                reason=f"view_image does not allow shell-expanded path '{raw_path}'.",
            )

        resolved = _resolve_workspace_relative_path(raw_path, context)
        if not _is_within_workspace(resolved, context.workspace_dir):
            return ToolPolicyDecision(
                allowed=False,
                reason=f"view_image may only read files inside {context.workspace_dir}.",
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
