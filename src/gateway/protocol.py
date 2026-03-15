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


@dataclass(slots=True, frozen=True)
class ClientStopTurn:
    pass


@dataclass(slots=True, frozen=True)
class ClientApprovalResponse:
    approval_id: str
    approved: bool


def parse_client_event(
    payload: Any,
    *,
    max_message_chars: int,
) -> ClientUserMessage | ClientStopTurn | ClientApprovalResponse:
    if not isinstance(payload, dict):
        raise ProtocolError(
            code="invalid_payload",
            message="Payload must be a JSON object.",
        )

    event_type = payload.get("type")
    if event_type == "stop_turn":
        return ClientStopTurn()
    if event_type == "approval_response":
        raw_approval_id = payload.get("approval_id")
        if not isinstance(raw_approval_id, str) or not raw_approval_id.strip():
            raise ProtocolError(
                code="invalid_approval_id",
                message="'approval_id' must be a non-empty string.",
            )
        approved = payload.get("approved")
        if not isinstance(approved, bool):
            raise ProtocolError(
                code="invalid_approval_decision",
                message="'approved' must be a boolean.",
            )
        return ClientApprovalResponse(
            approval_id=raw_approval_id.strip(),
            approved=approved,
        )
    if event_type != "user_message":
        raise ProtocolError(
            code="unsupported_event_type",
            message=(
                "Only 'user_message', 'stop_turn', and 'approval_response' events are supported."
            ),
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
) -> dict[str, Any]:
    return {
        "type": "assistant_message",
        "session_id": session_id,
        "text": text,
    }


def build_tool_call_event(
    *,
    session_id: str,
    tool_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "session_id": session_id,
        "tool_names": list(tool_names),
    }


def build_approval_request_event(
    *,
    session_id: str,
    approval_id: str,
    kind: str,
    summary: str,
    details: str,
    command: str | None,
    tool_name: str | None,
    inspection_url: str | None,
) -> dict[str, Any]:
    return {
        "type": "approval_request",
        "session_id": session_id,
        "approval_id": approval_id,
        "kind": kind,
        "summary": summary,
        "details": details,
        "command": command,
        "tool_name": tool_name,
        "inspection_url": inspection_url,
    }


def build_assistant_delta_event(*, session_id: str, delta: str) -> dict[str, Any]:
    return {
        "type": "assistant_delta",
        "session_id": session_id,
        "delta": delta,
    }


def build_turn_done_event(
    *,
    session_id: str,
    response_text: str,
    command: str | None,
    compaction_performed: bool,
    interrupted: bool,
) -> dict[str, Any]:
    return {
        "type": "turn_done",
        "session_id": session_id,
        "response_text": response_text,
        "command": command,
        "compaction_performed": compaction_performed,
        "interrupted": interrupted,
    }


def build_stop_ack_event(*, stop_requested: bool) -> dict[str, Any]:
    return {
        "type": "stop_ack",
        "stop_requested": stop_requested,
    }


def build_approval_ack_event(*, resolved: bool) -> dict[str, Any]:
    return {
        "type": "approval_ack",
        "resolved": resolved,
    }


def build_error_event(*, code: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "code": code,
        "message": message,
    }
