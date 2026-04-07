"""Route-scoped outbound event models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from jarvis.core.agent_loop import AgentKind

RouteEventType = Literal[
    "turn_started",
    "assistant_delta",
    "assistant_message",
    "turn_done",
    "tool_call",
    "approval_request",
    "auth_required",
    "local_notice",
    "system_notice",
    "error",
]
RouteTurnKind = Literal["user", "runtime"]


@dataclass(slots=True, frozen=True)
class RouteEventBase:
    route_id: str
    agent_kind: AgentKind
    agent_name: str
    session_id: str | None = None
    turn_id: str | None = None
    turn_kind: RouteTurnKind | None = None
    client_message_id: str | None = None
    subagent_id: str | None = None
    public: bool = True
    event_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=lambda: _utc_now_iso())


@dataclass(slots=True, frozen=True)
class RouteTurnStartedEvent(RouteEventBase):
    type: Literal["turn_started"] = "turn_started"


@dataclass(slots=True, frozen=True)
class RouteAssistantDeltaEvent(RouteEventBase):
    delta: str = ""
    type: Literal["assistant_delta"] = "assistant_delta"


@dataclass(slots=True, frozen=True)
class RouteAssistantMessageEvent(RouteEventBase):
    text: str = ""
    type: Literal["assistant_message"] = "assistant_message"


@dataclass(slots=True, frozen=True)
class RouteToolCallEvent(RouteEventBase):
    tool_names: tuple[str, ...] = ()
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True, frozen=True)
class RouteApprovalRequestEvent(RouteEventBase):
    approval_id: str = ""
    kind: str = ""
    summary: str = ""
    details: str = ""
    command: str | None = None
    tool_name: str | None = None
    inspection_url: str | None = None
    type: Literal["approval_request"] = "approval_request"


@dataclass(slots=True, frozen=True)
class RouteAuthRequiredEvent(RouteEventBase):
    provider: str = ""
    auth_kind: str = ""
    login_id: str = ""
    auth_url: str = ""
    message: str = ""
    type: Literal["auth_required"] = "auth_required"


@dataclass(slots=True, frozen=True)
class RouteTurnDoneEvent(RouteEventBase):
    response_text: str = ""
    command: str | None = None
    compaction_performed: bool = False
    interrupted: bool = False
    approval_rejected: bool = False
    interruption_reason: str | None = None
    type: Literal["turn_done"] = "turn_done"


@dataclass(slots=True, frozen=True)
class RouteLocalNoticeEvent(RouteEventBase):
    notice_kind: str = ""
    text: str = ""
    type: Literal["local_notice"] = "local_notice"


@dataclass(slots=True, frozen=True)
class RouteSystemNoticeEvent(RouteEventBase):
    notice_kind: str = ""
    text: str = ""
    type: Literal["system_notice"] = "system_notice"


@dataclass(slots=True, frozen=True)
class RouteErrorEvent(RouteEventBase):
    code: str = ""
    message: str = ""
    type: Literal["error"] = "error"


RouteEvent = (
    RouteTurnStartedEvent
    | RouteAssistantDeltaEvent
    | RouteAssistantMessageEvent
    | RouteToolCallEvent
    | RouteApprovalRequestEvent
    | RouteAuthRequiredEvent
    | RouteTurnDoneEvent
    | RouteLocalNoticeEvent
    | RouteSystemNoticeEvent
    | RouteErrorEvent
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
