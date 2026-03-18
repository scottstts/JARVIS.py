"""View-image tool definition and execution runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llm import ToolDefinition

from ...config import ToolSettings
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_SUPPORTED_IMAGE_MEDIA_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


class ViewImageToolExecutor:
    """Validates a workspace image and exposes it for the next model turn."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        raw_path = str(arguments["path"]).strip()
        raw_detail = str(arguments.get("detail", "auto")).strip() or "auto"
        image_path = _resolve_workspace_relative_path(raw_path, context)

        if not image_path.exists():
            return _view_image_error(
                call_id=call_id,
                raw_path=raw_path,
                reason="file does not exist.",
            )
        if not image_path.is_file():
            return _view_image_error(
                call_id=call_id,
                raw_path=raw_path,
                reason="path must point to a file.",
            )

        media_type = _detect_supported_image_media_type(image_path)
        if media_type is None:
            return _view_image_error(
                call_id=call_id,
                raw_path=raw_path,
                reason=(
                    "unsupported image type. Supported types shared across all provider "
                    "adapters: image/png, image/jpeg, image/webp."
                ),
            )
        if media_type == "unreadable":
            return _view_image_error(
                call_id=call_id,
                raw_path=raw_path,
                reason="file could not be read.",
            )

        detail = raw_detail if raw_detail in {"low", "high", "auto", "original"} else "auto"
        content = (
            "Image attachment prepared\n"
            f"path: {image_path}\n"
            f"media_type: {media_type}\n"
            f"detail: {detail}\n"
            "The image will be attached to the next model turn in this conversation."
        )
        metadata = {
            "path": str(image_path),
            "media_type": media_type,
            "detail": detail,
            "image_attachment": {
                "path": str(image_path),
                "media_type": media_type,
                "detail": detail,
            },
        }
        return ToolExecutionResult(
            call_id=call_id,
            name="view_image",
            ok=True,
            content=content,
            metadata=metadata,
        )


def build_view_image_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the view_image registry entry."""

    return RegisteredTool(
        name="view_image",
        exposure="basic",
        definition=ToolDefinition(
            name="view_image",
            description=_build_view_image_tool_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to a workspace image file that should be attached to the next "
                            "model turn. Use the local_path provided in prior user messages."
                        ),
                    },
                    "detail": {
                        "type": "string",
                        "enum": ["auto", "low", "high", "original"],
                        "description": (
                            "Optional image detail hint for providers that support image detail "
                            "levels. Use 'auto' unless high fidelity is clearly needed."
                        ),
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        executor=ViewImageToolExecutor(),
    )


def _build_view_image_tool_description(settings: ToolSettings) -> str:
    return (
        "Attach a workspace image file to the next model call so you can inspect it with "
        "multimodal vision. Only files inside "
        f"{settings.workspace_dir} are allowed. "
        "Use this when you need to visually inspect a local image. "
        "Supported image types: PNG, JPEG, and WEBP. "
        "The attachment is only guaranteed for the current tool-followup turn, so call "
        "view_image again later if you still need the image."
    )


def _resolve_workspace_relative_path(raw_path: str, context: ToolExecutionContext) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve(strict=False)


def _detect_supported_image_media_type(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return "unreadable"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _view_image_error(
    *,
    call_id: str,
    raw_path: str,
    reason: str,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id=call_id,
        name="view_image",
        ok=False,
        content=(
            "View image failed\n"
            f"path: {raw_path}\n"
            f"reason: {reason}"
        ),
        metadata={
            "path": raw_path,
            "error": reason,
        },
    )
