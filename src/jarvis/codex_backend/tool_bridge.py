"""Dynamic-tool bridge between Jarvis tools and Codex app-server."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from jarvis.llm import ToolCall, ToolDefinition
from jarvis.tools import ToolExecutionResult

from .types import CodexProtocolError

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


class CodexToolBridge:
    """Builds dynamic tool specs and serializes tool results back to Codex."""

    def __init__(self, *, tool_definitions_provider) -> None:
        self._tool_definitions_provider = tool_definitions_provider

    def build_dynamic_tools(
        self,
        *,
        activated_discoverable_tool_names: Sequence[str],
    ) -> list[dict[str, Any]]:
        definitions = self._tool_definitions_provider(activated_discoverable_tool_names)
        return [self._dynamic_tool_spec(definition) for definition in definitions]

    def build_tool_call(
        self,
        *,
        call_id: str,
        tool_name: str,
        arguments: object,
    ) -> ToolCall:
        if not isinstance(arguments, dict):
            raise CodexProtocolError(
                f"Codex dynamic tool arguments for '{tool_name}' must be a JSON object."
            )
        return ToolCall(
            call_id=call_id,
            name=tool_name,
            arguments=dict(arguments),
            raw_arguments="",
        )

    def build_tool_response(self, result: ToolExecutionResult) -> dict[str, Any]:
        return {
            "success": result.ok,
            "contentItems": self._content_items_for_result(result),
        }

    def build_tool_rejection_response(self, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "contentItems": [{"type": "inputText", "text": message}],
        }

    def _dynamic_tool_spec(self, definition: ToolDefinition) -> dict[str, Any]:
        return {
            "name": definition.name,
            "description": definition.description or definition.name,
            "inputSchema": dict(definition.input_schema),
        }

    def _content_items_for_result(self, result: ToolExecutionResult) -> list[dict[str, str]]:
        items = [{"type": "inputText", "text": result.content}]
        advisory_text = self._advisory_text_for_result(result)
        if advisory_text is not None:
            items.append({"type": "inputText", "text": advisory_text})
        image_attachment = result.metadata.get("image_attachment")
        if isinstance(image_attachment, dict):
            data_url = _image_attachment_to_data_url(image_attachment)
            if data_url is not None:
                items.append(
                    {
                        "type": "inputImage",
                        "imageUrl": data_url,
                    }
                )
        return items

    def _advisory_text_for_result(self, result: ToolExecutionResult) -> str | None:
        if result.name == "bash":
            status = str(result.metadata.get("status") or result.metadata.get("state") or "").strip()
            mode = str(result.metadata.get("mode", "")).strip()
            if status == "running" and (
                mode == "background" or bool(result.metadata.get("promoted_to_background"))
            ):
                return (
                    "Codex runtime note: this detached bash job is now being monitored by the "
                    "Jarvis orchestrator. Do not continue this turn or call more tools here."
                )
        if result.metadata.get("subagent_control") is not True:
            return None
        action = str(result.metadata.get("subagent_action", "")).strip()
        if action in {"invoke", "step_in"} and str(result.metadata.get("status", "")).strip() in {
            "running",
            "waiting_background",
            "awaiting_approval",
        }:
            return (
                "Codex runtime note: the subagent is now running independently and Jarvis is "
                "yielding control back to the route orchestrator. Do not call "
                "`subagent_monitor` or any other tool again in this turn."
            )
        if action == "monitor" and result.metadata.get("changed") is False:
            return (
                "Codex runtime note: no subagent state changed. Do not poll `subagent_monitor` "
                "again in this turn. Wait for the next orchestrator system update instead."
            )
        return None


def _image_attachment_to_data_url(image_attachment: dict[str, Any]) -> str | None:
    raw_path = image_attachment.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    media_type = image_attachment.get("media_type")
    if not isinstance(media_type, str) or not media_type.strip():
        media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        return None
    try:
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return None
    return f"data:{media_type};base64,{payload}"
