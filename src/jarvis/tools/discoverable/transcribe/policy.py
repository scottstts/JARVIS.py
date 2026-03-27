"""Policy checks for the transcribe tool."""

from __future__ import annotations

from typing import Any

from ...types import ToolExecutionContext, ToolPolicyDecision
from .shared import (
    GLOB_PATTERN,
    SUPPORTED_AUDIO_FORMATS,
    detect_supported_audio_format,
    is_within_workspace,
    normalize_audio_path,
    resolve_workspace_relative_path,
)


class TranscribePolicy:
    """Restricts transcribe to one explicit supported workspace audio path."""

    def authorize(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        audio_path = normalize_audio_path(arguments.get("audio_path"))
        if audio_path is None:
            return ToolPolicyDecision(
                allowed=False,
                reason="transcribe requires a non-empty 'audio_path'.",
            )
        if audio_path == "-":
            return ToolPolicyDecision(
                allowed=False,
                reason="transcribe audio_path '-' is not allowed.",
            )
        if audio_path.startswith("~") or GLOB_PATTERN.search(audio_path):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "transcribe does not allow shell-expanded "
                    f"audio_path '{audio_path}'."
                ),
            )

        resolved_audio_path = resolve_workspace_relative_path(audio_path, context)
        if not is_within_workspace(resolved_audio_path, context.workspace_dir):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "transcribe may only read input files inside "
                    f"{context.workspace_dir}."
                ),
            )

        if detect_supported_audio_format(resolved_audio_path) is None:
            supported = ", ".join(SUPPORTED_AUDIO_FORMATS)
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "transcribe audio_path must use a supported extension: "
                    f"{supported}."
                ),
            )

        return ToolPolicyDecision(allowed=True)
