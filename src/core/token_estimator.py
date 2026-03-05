"""Heuristic token estimation for preflight budget checks."""

from __future__ import annotations

import json
import math
from typing import Any

from llm.types import (
    ImagePart,
    LLMMessage,
    LLMRequest,
    TextPart,
    ToolChoiceMode,
)


def estimate_request_input_tokens(request: LLMRequest) -> int:
    """Estimate request input tokens from the fully assembled payload."""
    payload = {
        "instructions": request.instructions,
        "messages": [_serialize_message(message) for message in request.messages],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "strict": tool.strict,
                "input_schema": dict(tool.input_schema),
            }
            for tool in request.tools
        ],
        "tool_choice": _serialize_tool_choice(request),
        "parallel_tool_calls": request.parallel_tool_calls,
        "metadata": dict(request.metadata) if request.metadata is not None else None,
        "max_output_tokens": request.max_output_tokens,
        "temperature": request.temperature,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Roughly 4 characters per token; used only for preflight heuristics.
    return max(1, math.ceil(len(serialized) / 4))


def _serialize_message(message: LLMMessage) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            parts.append(
                {
                    "type": "image",
                    "image_url": part.image_url,
                    "file_id": part.file_id,
                    "detail": part.detail,
                }
            )
    return {"role": message.role, "parts": parts}


def _serialize_tool_choice(request: LLMRequest) -> dict[str, Any]:
    if request.tool_choice.mode == ToolChoiceMode.TOOL:
        return {"mode": request.tool_choice.mode.value, "tool_name": request.tool_choice.tool_name}
    return {"mode": request.tool_choice.mode.value}
