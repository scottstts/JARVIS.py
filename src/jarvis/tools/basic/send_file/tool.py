"""Send-file tool definition and execution runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jarvis.llm import ToolDefinition
from jarvis.ui.telegram import UIConfigurationError, send_telegram_file
from jarvis.ui.telegram.api import TelegramAPIError

from ...config import ToolSettings
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult


class SendFileToolExecutor:
    """Sends a local workspace file to the user's Telegram chat."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        raw_path = str(arguments["path"]).strip()
        file_path = _resolve_workspace_relative_path(raw_path, context)
        caption = _normalize_optional_string(arguments.get("caption"))
        filename = _normalize_optional_string(arguments.get("filename"))

        if not file_path.exists():
            return _send_file_error(
                call_id=call_id,
                raw_path=raw_path,
                reason="file does not exist.",
            )
        if not file_path.is_file():
            return _send_file_error(
                call_id=call_id,
                raw_path=raw_path,
                reason="path must point to a file.",
            )

        try:
            telegram_response = await send_telegram_file(
                file_path=file_path,
                caption=caption,
                filename=filename,
                route_id=context.route_id,
            )
        except (TelegramAPIError, UIConfigurationError, ValueError, OSError) as exc:
            return _send_file_error(
                call_id=call_id,
                raw_path=raw_path,
                reason=str(exc),
            )

        resolved_chat_id = telegram_response.get("chat_id")
        resolved_filename = filename or file_path.name
        content_lines = [
            "File sent to Telegram",
            f"path: {file_path}",
            f"filename: {resolved_filename}",
        ]
        if caption is not None:
            content_lines.append(f"caption: {caption}")
        if isinstance(resolved_chat_id, int):
            content_lines.append(f"chat_id: {resolved_chat_id}")
        if context.route_id is not None:
            content_lines.append(f"route_id: {context.route_id}")

        metadata: dict[str, Any] = {
            "path": str(file_path),
            "filename": resolved_filename,
            "caption": caption,
            "route_id": context.route_id,
            "telegram_response": telegram_response,
        }
        if isinstance(resolved_chat_id, int):
            metadata["chat_id"] = resolved_chat_id

        return ToolExecutionResult(
            call_id=call_id,
            name="send_file",
            ok=True,
            content="\n".join(content_lines),
            metadata=metadata,
        )


def build_send_file_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the send_file registry entry."""

    return RegisteredTool(
        name="send_file",
        exposure="basic",
        definition=ToolDefinition(
            name="send_file",
            description=_build_send_file_tool_description(settings),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace file to send to the user's Telegram chat.",
                    },
                    "caption": {
                        "type": "string",
                        "description": "Optional Telegram caption.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional upload filename.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        executor=SendFileToolExecutor(),
        allowed_agent_kinds=("main",),
    )


def _build_send_file_tool_description(settings: ToolSettings) -> str:
    return (
        "Send a local workspace file to the user's Telegram chat as a document. "
        f"Only files inside {settings.workspace_dir} are allowed. "
        "Use this when the user asks you to deliver a generated or existing workspace file "
        "back to Telegram. "
        "Never use it for .env files or paths outside the workspace."
    )


def _resolve_workspace_relative_path(raw_path: str, context: ToolExecutionContext) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve(strict=False)


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _send_file_error(
    *,
    call_id: str,
    raw_path: str,
    reason: str,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id=call_id,
        name="send_file",
        ok=False,
        content=(
            "Send file failed\n"
            f"path: {raw_path}\n"
            f"reason: {reason}"
        ),
        metadata={
            "path": raw_path,
            "error": reason,
        },
    )
