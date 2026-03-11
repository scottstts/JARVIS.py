"""Policy checks for the generate_edit_image tool."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ...types import ToolExecutionContext, ToolPolicyDecision

_ALLOWED_PROVIDERS = {"gemini", "openai"}
_GLOB_PATTERN = re.compile(r"[*?\[]")
_MAX_PROMPT_CHARS = 8_000
_MAX_PROMPT_WORDS = 1_200
_OPENAI_QUALITY_VALUES = {"low", "medium", "high"}
_GEMINI_RESOLUTION_VALUES = {"512", "1K", "2K", "4K"}


class GenerateEditImagePolicy:
    """Restricts generate_edit_image to explicit workspace image edits."""

    def authorize(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        raw_prompt = str(arguments.get("prompt", "")).strip()
        if not raw_prompt:
            return ToolPolicyDecision(
                allowed=False,
                reason="generate_edit_image requires a non-empty 'prompt'.",
            )
        if len(raw_prompt) > _MAX_PROMPT_CHARS:
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image prompt length must be <= "
                    f"{_MAX_PROMPT_CHARS} characters."
                ),
            )
        if len(raw_prompt.split()) > _MAX_PROMPT_WORDS:
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image prompt length must be <= "
                    f"{_MAX_PROMPT_WORDS} words."
                ),
            )

        output_path = _normalize_optional_string(arguments.get("output_path"))
        if output_path is None:
            return ToolPolicyDecision(
                allowed=False,
                reason="generate_edit_image requires a non-empty 'output_path'.",
            )
        if output_path == "-":
            return ToolPolicyDecision(
                allowed=False,
                reason="generate_edit_image output_path '-' is not allowed.",
            )
        if output_path.startswith("~") or _GLOB_PATTERN.search(output_path):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image does not allow shell-expanded "
                    f"output_path '{output_path}'."
                ),
            )

        resolved_output_path = _resolve_workspace_relative_path(output_path, context)
        if not _is_within_workspace(resolved_output_path, context.workspace_dir):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image may only write output files inside "
                    f"{context.workspace_dir}."
                ),
            )

        provider = _normalize_provider(arguments.get("provider"))
        if provider not in _ALLOWED_PROVIDERS:
            allowed = ", ".join(sorted(_ALLOWED_PROVIDERS))
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image provider must be one of: "
                    f"{allowed}."
                ),
            )

        quality = arguments.get("quality")
        if quality is not None and _normalize_openai_quality(quality) is None:
            allowed = ", ".join(sorted(_OPENAI_QUALITY_VALUES))
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image quality must be one of: "
                    f"{allowed}."
                ),
            )

        resolution = arguments.get("resolution")
        if resolution is not None and _normalize_gemini_resolution(resolution) is None:
            allowed = ", ".join(sorted(_GEMINI_RESOLUTION_VALUES))
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image resolution must be one of: "
                    f"{allowed}."
                ),
            )

        image_path = _normalize_optional_string(arguments.get("image_path"))
        if image_path is None:
            return ToolPolicyDecision(allowed=True)
        if image_path == "-":
            return ToolPolicyDecision(
                allowed=False,
                reason="generate_edit_image image_path '-' is not allowed.",
            )
        if image_path.startswith("~") or _GLOB_PATTERN.search(image_path):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image does not allow shell-expanded "
                    f"image_path '{image_path}'."
                ),
            )

        resolved = _resolve_workspace_relative_path(image_path, context)
        if not _is_within_workspace(resolved, context.workspace_dir):
            return ToolPolicyDecision(
                allowed=False,
                reason=(
                    "generate_edit_image may only read input images inside "
                    f"{context.workspace_dir}."
                ),
            )
        return ToolPolicyDecision(allowed=True)


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_provider(value: Any) -> str:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return "gemini"
    return normalized.lower()


def _normalize_openai_quality(value: Any) -> str | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered in _OPENAI_QUALITY_VALUES:
        return lowered
    return None


def _normalize_gemini_resolution(value: Any) -> str | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None
    if normalized == "512":
        return normalized
    upper = normalized.upper()
    if upper in _GEMINI_RESOLUTION_VALUES:
        return upper
    return None


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
