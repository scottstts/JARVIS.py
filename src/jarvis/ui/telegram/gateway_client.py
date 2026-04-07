"""Persistent websocket bridge client for the gateway wire protocol."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4


class GatewayBridgeError(RuntimeError):
    """Raised when gateway websocket interaction fails."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True, frozen=True)
class GatewayRouteEventBase:
    event_id: str = ""
    created_at: str = ""
    route_id: str = ""
    session_id: str | None = None
    turn_id: str | None = None
    turn_kind: str | None = None
    client_message_id: str | None = None
    agent_kind: str = "main"
    agent_name: str = "Jarvis"
    subagent_id: str | None = None


@dataclass(slots=True, frozen=True)
class GatewayTurnStartedEvent(GatewayRouteEventBase):
    type: str = "turn_started"


@dataclass(slots=True, frozen=True)
class GatewayDeltaEvent(GatewayRouteEventBase):
    delta: str = ""
    type: str = "assistant_delta"


@dataclass(slots=True, frozen=True)
class GatewayMessageEvent(GatewayRouteEventBase):
    text: str = ""
    type: str = "assistant_message"


@dataclass(slots=True, frozen=True)
class GatewayToolCallEvent(GatewayRouteEventBase):
    tool_names: tuple[str, ...] = ()
    type: str = "tool_call"


@dataclass(slots=True, frozen=True)
class GatewayApprovalRequestEvent(GatewayRouteEventBase):
    approval_id: str = ""
    kind: str = ""
    summary: str = ""
    details: str = ""
    command: str | None = None
    tool_name: str | None = None
    inspection_url: str | None = None
    type: str = "approval_request"


@dataclass(slots=True, frozen=True)
class GatewayAuthRequiredEvent(GatewayRouteEventBase):
    provider: str = ""
    auth_kind: str = ""
    login_id: str = ""
    auth_url: str = ""
    message: str = ""
    type: str = "auth_required"


@dataclass(slots=True, frozen=True)
class GatewayTurnDoneEvent(GatewayRouteEventBase):
    response_text: str = ""
    command: str | None = None
    compaction_performed: bool = False
    interrupted: bool = False
    approval_rejected: bool = False
    interruption_reason: str | None = None
    type: str = "turn_done"


@dataclass(slots=True, frozen=True)
class GatewayLocalNoticeEvent(GatewayRouteEventBase):
    notice_kind: str = ""
    text: str = ""
    type: str = "local_notice"


@dataclass(slots=True, frozen=True)
class GatewaySystemNoticeEvent(GatewayRouteEventBase):
    notice_kind: str = ""
    text: str = ""
    type: str = "system_notice"


@dataclass(slots=True, frozen=True)
class GatewayErrorEvent(GatewayRouteEventBase):
    code: str = ""
    message: str = ""
    type: str = "error"


GatewayRouteEvent = (
    GatewayTurnStartedEvent
    | GatewayDeltaEvent
    | GatewayMessageEvent
    | GatewayToolCallEvent
    | GatewayApprovalRequestEvent
    | GatewayAuthRequiredEvent
    | GatewayTurnDoneEvent
    | GatewayLocalNoticeEvent
    | GatewaySystemNoticeEvent
    | GatewayErrorEvent
)


class GatewayRouteSession:
    """Maintains one persistent websocket connection for a single route."""

    def __init__(
        self,
        *,
        route_id: str,
        websocket_base_url: str,
        connect_timeout_seconds: float,
    ) -> None:
        self.route_id = route_id
        self._websocket_base_url = websocket_base_url.rstrip("/")
        self._connect_timeout_seconds = connect_timeout_seconds
        self._connection: Any | None = None
        self._socket: Any | None = None
        self._ready_session_id: str | None = None
        self._events: asyncio.Queue[GatewayRouteEvent] = asyncio.Queue()
        self._reader_task: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._approval_lock = asyncio.Lock()
        self._pending_stop_future: asyncio.Future[bool] | None = None
        self._pending_approval_future: asyncio.Future[bool] | None = None

    @property
    def ready_session_id(self) -> str | None:
        return self._ready_session_id

    async def connect(self) -> None:
        if self._socket is not None:
            return
        websocket_url = f"{self._websocket_base_url}/{self.route_id}"
        connect = _resolve_websocket_connect()
        try:
            connection = connect(
                websocket_url,
                open_timeout=self._connect_timeout_seconds,
                close_timeout=self._connect_timeout_seconds,
            )
            socket = await connection.__aenter__()
        except Exception:
            raise GatewayBridgeError(
                code="gateway_unavailable",
                message="Could not communicate with the gateway websocket.",
            ) from None

        ready_payload = await _recv_json(socket)
        ready_type = ready_payload.get("type")
        if ready_type == "error":
            raise GatewayBridgeError(
                code=str(ready_payload.get("code", "gateway_error")),
                message=str(ready_payload.get("message", "Gateway returned an error.")),
            )
        if ready_type != "ready":
            raise GatewayBridgeError(
                code="invalid_ready_event",
                message="Gateway did not send a ready event.",
            )
        self._connection = connection
        self._ready_session_id = (
            str(ready_payload["session_id"])
            if ready_payload.get("session_id") is not None
            else None
        )
        self._socket = socket
        self._reader_task = asyncio.create_task(
            self._reader_loop(),
            name=f"jarvis-gateway-route-session-{self.route_id}",
        )

    async def aclose(self) -> None:
        reader_task = self._reader_task
        if reader_task is not None:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
        socket = self._socket
        connection = self._connection
        self._socket = None
        self._connection = None
        self._reader_task = None
        if connection is not None:
            try:
                await connection.__aexit__(None, None, None)
            except Exception:
                pass
        elif socket is not None:
            try:
                await socket.close()
            except Exception:
                pass

    async def send_user_message(self, *, text: str, client_message_id: str) -> None:
        await self._ensure_connected()
        async with self._send_lock:
            await self._socket.send(
                json.dumps(
                    {
                        "type": "user_message",
                        "text": text,
                        "client_message_id": client_message_id,
                    }
                )
            )

    async def request_stop(self) -> bool:
        await self._ensure_connected()
        async with self._stop_lock:
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._pending_stop_future = future
            async with self._send_lock:
                await self._socket.send(json.dumps({"type": "stop_turn"}))
            return await future

    async def submit_approval(self, *, approval_id: str, approved: bool) -> bool:
        await self._ensure_connected()
        async with self._approval_lock:
            future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
            self._pending_approval_future = future
            async with self._send_lock:
                await self._socket.send(
                    json.dumps(
                        {
                            "type": "approval_response",
                            "approval_id": approval_id,
                            "approved": approved,
                        }
                    )
                )
            return await future

    async def events(self) -> AsyncIterator[GatewayRouteEvent]:
        await self._ensure_connected()
        while True:
            yield await self._events.get()

    async def _ensure_connected(self) -> None:
        if self._socket is None:
            await self.connect()

    async def _reader_loop(self) -> None:
        socket = self._socket
        if socket is None:
            return
        try:
            while True:
                payload = await _recv_json(socket)
                event_type = payload.get("type")
                if event_type == "stop_ack":
                    future = self._pending_stop_future
                    self._pending_stop_future = None
                    if future is not None and not future.done():
                        future.set_result(bool(payload.get("stop_requested", False)))
                    continue
                if event_type == "approval_ack":
                    future = self._pending_approval_future
                    self._pending_approval_future = None
                    if future is not None and not future.done():
                        future.set_result(bool(payload.get("resolved", False)))
                    continue
                if event_type == "error" and payload.get("event_id") is None:
                    raise GatewayBridgeError(
                        code=str(payload.get("code", "gateway_error")),
                        message=str(payload.get("message", "Gateway returned an error.")),
                    )
                await self._events.put(_parse_route_event(payload))
        except asyncio.CancelledError:
            raise
        except GatewayBridgeError as exc:
            if self._pending_stop_future is not None and not self._pending_stop_future.done():
                self._pending_stop_future.set_result(False)
            if self._pending_approval_future is not None and not self._pending_approval_future.done():
                self._pending_approval_future.set_result(False)
            await self._events.put(
                GatewayErrorEvent(
                    event_id="",
                    created_at="",
                    route_id=self.route_id,
                    session_id=None,
                    agent_kind="main",
                    agent_name="Jarvis",
                    subagent_id=None,
                    code=exc.code,
                    message=exc.message,
                )
            )
        except Exception:
            if self._pending_stop_future is not None and not self._pending_stop_future.done():
                self._pending_stop_future.set_result(False)
            if self._pending_approval_future is not None and not self._pending_approval_future.done():
                self._pending_approval_future.set_result(False)
            await self._events.put(
                GatewayErrorEvent(
                    event_id="",
                    created_at="",
                    route_id=self.route_id,
                    session_id=None,
                    agent_kind="main",
                    agent_name="Jarvis",
                    subagent_id=None,
                    code="gateway_unavailable",
                    message="Could not communicate with the gateway websocket.",
                )
            )


class GatewayWebSocketClient:
    """Factory and compatibility wrapper for route-scoped gateway sessions."""

    def __init__(
        self,
        *,
        websocket_base_url: str,
        connect_timeout_seconds: float = 15.0,
    ) -> None:
        self._websocket_base_url = websocket_base_url.rstrip("/")
        self._connect_timeout_seconds = connect_timeout_seconds

    async def connect_route(self, *, route_id: str) -> GatewayRouteSession:
        session = GatewayRouteSession(
            route_id=route_id,
            websocket_base_url=self._websocket_base_url,
            connect_timeout_seconds=self._connect_timeout_seconds,
        )
        await session.connect()
        return session

    async def stream_turn(self, *, route_id: str, user_text: str) -> AsyncIterator[GatewayRouteEvent]:
        session = await self.connect_route(route_id=route_id)
        client_message_id = uuid4().hex
        matched_turn_id: str | None = None
        try:
            await session.send_user_message(
                text=user_text,
                client_message_id=client_message_id,
            )
            async for event in session.events():
                if isinstance(event, GatewayErrorEvent):
                    raise GatewayBridgeError(
                        code=event.code or "gateway_error",
                        message=event.message or "Gateway returned an error.",
                    )
                if isinstance(event, GatewayTurnStartedEvent):
                    if event.client_message_id != client_message_id:
                        continue
                    matched_turn_id = event.turn_id
                    continue
                if matched_turn_id is None:
                    if event.client_message_id != client_message_id:
                        continue
                else:
                    if event.turn_id is not None and event.turn_id != matched_turn_id:
                        continue
                    if (
                        event.turn_id is None
                        and event.client_message_id is not None
                        and event.client_message_id != client_message_id
                    ):
                        continue
                yield event
                if isinstance(event, GatewayTurnDoneEvent):
                    return
        finally:
            await session.aclose()

    async def request_stop(self, *, route_id: str) -> bool:
        session = await self.connect_route(route_id=route_id)
        try:
            return await session.request_stop()
        finally:
            await session.aclose()

    async def submit_approval(
        self,
        *,
        route_id: str,
        approval_id: str,
        approved: bool,
    ) -> bool:
        session = await self.connect_route(route_id=route_id)
        try:
            return await session.submit_approval(
                approval_id=approval_id,
                approved=approved,
            )
        finally:
            await session.aclose()


def _resolve_websocket_connect():
    try:
        import websockets
    except ModuleNotFoundError as exc:
        raise GatewayBridgeError(
            code="websocket_client_missing",
            message="Python package 'websockets' is required for the Telegram UI.",
        ) from exc

    return websockets.connect


async def _recv_json(socket: Any) -> dict[str, Any]:
    try:
        raw_payload = await asyncio.wait_for(socket.recv(), timeout=None)
    except asyncio.CancelledError:
        raise

    if not isinstance(raw_payload, str):
        raise GatewayBridgeError(
            code="invalid_gateway_payload",
            message="Gateway payload must be a JSON text frame.",
        )

    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise GatewayBridgeError(
            code="invalid_gateway_payload",
            message="Gateway payload was not valid JSON.",
        ) from exc

    if not isinstance(parsed, dict):
        raise GatewayBridgeError(
            code="invalid_gateway_payload",
            message="Gateway payload must be a JSON object.",
        )
    return parsed


def _parse_route_event(payload: dict[str, Any]) -> GatewayRouteEvent:
    base_kwargs = {
        "event_id": str(payload.get("event_id", "")),
        "created_at": str(payload.get("created_at", "")),
        "route_id": str(payload.get("route_id", "")),
        "session_id": (
            str(payload["session_id"])
            if payload.get("session_id") is not None
            else None
        ),
        "turn_id": (
            str(payload["turn_id"])
            if payload.get("turn_id") is not None
            else None
        ),
        "turn_kind": (
            str(payload["turn_kind"])
            if payload.get("turn_kind") is not None
            else None
        ),
        "client_message_id": (
            str(payload["client_message_id"])
            if payload.get("client_message_id") is not None
            else None
        ),
        "agent_kind": str(payload.get("agent_kind", "")),
        "agent_name": str(payload.get("agent_name", "")),
        "subagent_id": (
            str(payload["subagent_id"])
            if payload.get("subagent_id") is not None
            else None
        ),
    }
    event_type = str(payload.get("type", ""))
    if event_type == "turn_started":
        return GatewayTurnStartedEvent(**base_kwargs)
    if event_type == "assistant_delta":
        return GatewayDeltaEvent(**base_kwargs, delta=str(payload.get("delta", "")))
    if event_type == "assistant_message":
        return GatewayMessageEvent(**base_kwargs, text=str(payload.get("text", "")))
    if event_type == "tool_call":
        raw_tool_names = payload.get("tool_names", [])
        tool_names: list[str] = []
        if isinstance(raw_tool_names, list):
            for raw_name in raw_tool_names:
                name = str(raw_name).strip()
                if name:
                    tool_names.append(name)
        return GatewayToolCallEvent(**base_kwargs, tool_names=tuple(tool_names))
    if event_type == "approval_request":
        return GatewayApprovalRequestEvent(
            **base_kwargs,
            approval_id=str(payload.get("approval_id", "")),
            kind=str(payload.get("kind", "")),
            summary=str(payload.get("summary", "")),
            details=str(payload.get("details", "")),
            command=(
                str(payload["command"])
                if payload.get("command") is not None
                else None
            ),
            tool_name=(
                str(payload["tool_name"])
                if payload.get("tool_name") is not None
                else None
            ),
            inspection_url=(
                str(payload["inspection_url"])
                if payload.get("inspection_url") is not None
                else None
            ),
        )
    if event_type == "auth_required":
        return GatewayAuthRequiredEvent(
            **base_kwargs,
            provider=str(payload.get("provider", "")),
            auth_kind=str(payload.get("auth_kind", "")),
            login_id=str(payload.get("login_id", "")),
            auth_url=str(payload.get("auth_url", "")),
            message=str(payload.get("message", "")),
        )
    if event_type == "turn_done":
        return GatewayTurnDoneEvent(
            **base_kwargs,
            response_text=str(payload.get("response_text", "")),
            command=(
                str(payload["command"])
                if payload.get("command") is not None
                else None
            ),
            compaction_performed=bool(payload.get("compaction_performed", False)),
            interrupted=bool(payload.get("interrupted", False)),
            approval_rejected=bool(payload.get("approval_rejected", False)),
            interruption_reason=(
                str(payload["interruption_reason"])
                if payload.get("interruption_reason") is not None
                else None
            ),
        )
    if event_type == "local_notice":
        return GatewayLocalNoticeEvent(
            **base_kwargs,
            notice_kind=str(payload.get("notice_kind", "")),
            text=str(payload.get("text", "")),
        )
    if event_type == "system_notice":
        return GatewaySystemNoticeEvent(
            **base_kwargs,
            notice_kind=str(payload.get("notice_kind", "")),
            text=str(payload.get("text", "")),
        )
    if event_type == "error":
        return GatewayErrorEvent(
            **base_kwargs,
            code=str(payload.get("code", "gateway_error")),
            message=str(payload.get("message", "Gateway returned an error.")),
        )
    raise GatewayBridgeError(
        code="invalid_gateway_payload",
        message=f"Unsupported gateway event type: {event_type}",
    )
