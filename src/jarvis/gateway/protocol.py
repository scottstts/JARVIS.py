"""Wire protocol helpers for gateway websocket events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .route_events import (
    RouteApprovalRequestEvent,
    RouteAuthRequiredEvent,
    RouteAssistantDeltaEvent,
    RouteAssistantMessageEvent,
    RouteErrorEvent,
    RouteEvent,
    RouteLocalNoticeEvent,
    RouteSystemNoticeEvent,
    RouteTaskStatusEvent,
    RouteToolCallEvent,
    RouteTurnStartedEvent,
    RouteTurnDoneEvent,
)


class ProtocolError(ValueError):
    """Raised when websocket payload violates protocol contract."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True, frozen=True)
class ClientUserMessage:
    text: str
    client_message_id: str


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

    raw_client_message_id = payload.get("client_message_id")
    if not isinstance(raw_client_message_id, str) or not raw_client_message_id.strip():
        raise ProtocolError(
            code="invalid_client_message_id",
            message="'client_message_id' must be a non-empty string.",
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

    return ClientUserMessage(
        text=raw_text,
        client_message_id=raw_client_message_id.strip(),
    )


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
    turn_id: str | None = None,
    turn_kind: str | None = None,
    client_message_id: str | None = None,
    command: str | None,
    compaction_performed: bool,
    interrupted: bool,
    interruption_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "turn_done",
        "session_id": session_id,
        "turn_id": turn_id,
        "turn_kind": turn_kind,
        "client_message_id": client_message_id,
        "response_text": response_text,
        "command": command,
        "compaction_performed": compaction_performed,
        "interrupted": interrupted,
        "interruption_reason": interruption_reason,
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


def build_route_event_payload(event: RouteEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event.type,
        "event_id": event.event_id,
        "created_at": event.created_at,
        "route_id": event.route_id,
        "session_id": event.session_id,
        "turn_id": event.turn_id,
        "turn_kind": event.turn_kind,
        "client_message_id": event.client_message_id,
        "agent_kind": event.agent_kind,
        "agent_name": event.agent_name,
        "subagent_id": event.subagent_id,
    }
    if isinstance(event, RouteTurnStartedEvent):
        return payload
    if isinstance(event, RouteAssistantDeltaEvent):
        payload["delta"] = event.delta
        return payload
    if isinstance(event, RouteAssistantMessageEvent):
        payload["text"] = event.text
        return payload
    if isinstance(event, RouteToolCallEvent):
        payload["tool_names"] = list(event.tool_names)
        return payload
    if isinstance(event, RouteApprovalRequestEvent):
        payload.update(
            {
                "approval_id": event.approval_id,
                "kind": event.kind,
                "summary": event.summary,
                "details": event.details,
                "command": event.command,
                "tool_name": event.tool_name,
                "inspection_url": event.inspection_url,
            }
        )
        return payload
    if isinstance(event, RouteAuthRequiredEvent):
        payload.update(
            {
                "provider": event.provider,
                "auth_kind": event.auth_kind,
                "login_id": event.login_id,
                "auth_url": event.auth_url,
                "message": event.message,
            }
        )
        return payload
    if isinstance(event, RouteTurnDoneEvent):
        payload.update(
            {
                "response_text": event.response_text,
                "command": event.command,
                "compaction_performed": event.compaction_performed,
                "interrupted": event.interrupted,
                "approval_rejected": event.approval_rejected,
                "interruption_reason": event.interruption_reason,
            }
        )
        return payload
    if isinstance(event, RouteLocalNoticeEvent):
        payload.update(
            {
                "notice_kind": event.notice_kind,
                "text": event.text,
            }
        )
        return payload
    if isinstance(event, RouteSystemNoticeEvent):
        payload.update(
            {
                "notice_kind": event.notice_kind,
                "text": event.text,
            }
        )
        return payload
    if isinstance(event, RouteTaskStatusEvent):
        payload.update(
            {
                "active": event.active,
                "reason": event.reason,
            }
        )
        return payload
    if isinstance(event, RouteErrorEvent):
        payload.update(
            {
                "code": event.code,
                "message": event.message,
            }
        )
        return payload
    raise TypeError(f"Unsupported route event type: {type(event).__name__}")
