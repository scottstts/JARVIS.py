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
    ToolCall,
    ToolChoiceMode,
    ToolResultPart,
)

_IMAGE_DETAIL_TOKEN_ESTIMATES = {
    "low": 128,
    "auto": 256,
    "high": 384,
    "original": 512,
}


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
    text_token_estimate = max(1, math.ceil(len(serialized) / 4))
    image_token_estimate = sum(_estimate_message_image_tokens(message) for message in request.messages)
    return text_token_estimate + image_token_estimate


def _serialize_message(message: LLMMessage) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for part in message.parts:
        if isinstance(part, TextPart):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            parts.append(
                {
                    "type": "image",
                    # Avoid letting raw base64 payloads dominate the estimate.
                    "image_url": "<image-url>" if part.image_url is not None else None,
                    "file_id": "<file-id>" if part.file_id is not None else None,
                    "detail": part.detail,
                }
            )
        elif isinstance(part, ToolCall):
            parts.append(
                {
                    "type": "tool_call",
                    "call_id": part.call_id,
                    "name": part.name,
                    "arguments": part.arguments,
                    "provider_metadata": part.provider_metadata,
                }
            )
        elif isinstance(part, ToolResultPart):
            parts.append(
                {
                    "type": "tool_result",
                    "call_id": part.call_id,
                    "name": part.name,
                    "content": part.content,
                    "is_error": part.is_error,
                }
            )
    return {"role": message.role, "parts": parts}


def _estimate_message_image_tokens(message: LLMMessage) -> int:
    return sum(
        _estimate_image_tokens(part)
        for part in message.parts
        if isinstance(part, ImagePart)
    )


def _estimate_image_tokens(part: ImagePart) -> int:
    return _IMAGE_DETAIL_TOKEN_ESTIMATES.get(part.detail, _IMAGE_DETAIL_TOKEN_ESTIMATES["auto"])


def _serialize_tool_choice(request: LLMRequest) -> dict[str, Any]:
    if request.tool_choice.mode == ToolChoiceMode.TOOL:
        return {"mode": request.tool_choice.mode.value, "tool_name": request.tool_choice.tool_name}
    return {"mode": request.tool_choice.mode.value}
