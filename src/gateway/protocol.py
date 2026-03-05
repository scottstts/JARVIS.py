"""Wire protocol helpers for gateway websocket events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ProtocolError(ValueError):
    """Raised when websocket payload violates protocol contract."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True, frozen=True)
class ClientUserMessage:
    text: str


def parse_client_event(payload: Any, *, max_message_chars: int) -> ClientUserMessage:
    if not isinstance(payload, dict):
        raise ProtocolError(
            code="invalid_payload",
            message="Payload must be a JSON object.",
        )

    event_type = payload.get("type")
    if event_type != "user_message":
        raise ProtocolError(
            code="unsupported_event_type",
            message="Only 'user_message' events are supported.",
        )

    raw_text = payload.get("text")
    if not isinstance(raw_text, str):
        raise ProtocolError(
            code="invalid_message_text",
            message="'text' must be a string.",
        )
    if not raw_text.strip():
        raise ProtocolError(
            code="empty_message",
            message="'text' cannot be empty.",
        )
    if len(raw_text) > max_message_chars:
        raise ProtocolError(
            code="message_too_large",
            message=f"'text' exceeds max length of {max_message_chars} chars.",
        )

    return ClientUserMessage(text=raw_text)


def build_ready_event(*, route_id: str, session_id: str | None) -> dict[str, Any]:
    return {
        "type": "ready",
        "route_id": route_id,
        "session_id": session_id,
    }


def build_assistant_message_event(
    *,
    session_id: str,
    text: str,
    command: str | None,
    compaction_performed: bool,
) -> dict[str, Any]:
    return {
        "type": "assistant_message",
        "session_id": session_id,
        "text": text,
        "command": command,
        "compaction_performed": compaction_performed,
    }


def build_assistant_delta_event(*, session_id: str, delta: str) -> dict[str, Any]:
    return {
        "type": "assistant_delta",
        "session_id": session_id,
        "delta": delta,
    }


def build_error_event(*, code: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "code": code,
        "message": message,
    }
