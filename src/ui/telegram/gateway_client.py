"""Websocket bridge client for the gateway wire protocol."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


class GatewayBridgeError(RuntimeError):
    """Raised when gateway websocket interaction fails."""

    def __init__(self, *, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(slots=True, frozen=True)
class GatewayDeltaEvent:
    session_id: str
    delta: str
    type: str = "assistant_delta"


@dataclass(slots=True, frozen=True)
class GatewayMessageEvent:
    session_id: str
    text: str
    type: str = "assistant_message"


@dataclass(slots=True, frozen=True)
class GatewayToolCallEvent:
    session_id: str
    tool_names: tuple[str, ...]
    type: str = "tool_call"


@dataclass(slots=True, frozen=True)
class GatewayApprovalRequestEvent:
    session_id: str
    approval_id: str
    kind: str
    summary: str
    details: str
    command: str | None = None
    tool_name: str | None = None
    inspection_url: str | None = None
    type: str = "approval_request"


@dataclass(slots=True, frozen=True)
class GatewayTurnDoneEvent:
    session_id: str
    response_text: str
    command: str | None = None
    compaction_performed: bool = False
    interrupted: bool = False
    type: str = "turn_done"


GatewayTurnEvent = (
    GatewayDeltaEvent
    | GatewayMessageEvent
    | GatewayToolCallEvent
    | GatewayApprovalRequestEvent
    | GatewayTurnDoneEvent
)


class GatewayWebSocketClient:
    """Minimal websocket client for one-turn gateway interactions."""

    def __init__(
        self,
        *,
        websocket_base_url: str,
        connect_timeout_seconds: float = 15.0,
    ) -> None:
        self._websocket_base_url = websocket_base_url.rstrip("/")
        self._connect_timeout_seconds = connect_timeout_seconds

    async def stream_turn(self, *, route_id: str, user_text: str) -> AsyncIterator[GatewayTurnEvent]:
        websocket_url = f"{self._websocket_base_url}/{route_id}"
        connect = _resolve_websocket_connect()

        try:
            async with connect(
                websocket_url,
                open_timeout=self._connect_timeout_seconds,
                close_timeout=self._connect_timeout_seconds,
            ) as socket:
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

                await socket.send(
                    json.dumps(
                        {
                            "type": "user_message",
                            "text": user_text,
                        }
                    )
                )

                while True:
                    payload = await _recv_json(socket)
                    event_type = payload.get("type")
                    if event_type == "assistant_delta":
                        yield GatewayDeltaEvent(
                            session_id=str(payload.get("session_id", "")),
                            delta=str(payload.get("delta", "")),
                        )
                        continue
                    if event_type == "assistant_message":
                        yield GatewayMessageEvent(
                            session_id=str(payload.get("session_id", "")),
                            text=str(payload.get("text", "")),
                        )
                        continue
                    if event_type == "tool_call":
                        raw_tool_names = payload.get("tool_names", [])
                        tool_names: list[str] = []
                        if isinstance(raw_tool_names, list):
                            for raw_name in raw_tool_names:
                                name = str(raw_name).strip()
                                if name:
                                    tool_names.append(name)
                        yield GatewayToolCallEvent(
                            session_id=str(payload.get("session_id", "")),
                            tool_names=tuple(tool_names),
                        )
                        continue
                    if event_type == "approval_request":
                        yield GatewayApprovalRequestEvent(
                            session_id=str(payload.get("session_id", "")),
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
                        continue
                    if event_type == "turn_done":
                        yield GatewayTurnDoneEvent(
                            session_id=str(payload.get("session_id", "")),
                            response_text=str(payload.get("response_text", "")),
                            command=(
                                str(payload["command"])
                                if payload.get("command") is not None
                                else None
                            ),
                            compaction_performed=bool(payload.get("compaction_performed", False)),
                            interrupted=bool(payload.get("interrupted", False)),
                        )
                        return
                    if event_type == "error":
                        raise GatewayBridgeError(
                            code=str(payload.get("code", "gateway_error")),
                            message=str(payload.get("message", "Gateway returned an error.")),
                        )
        except GatewayBridgeError:
            raise
        except Exception:  # pragma: no cover - exception types depend on websocket lib
            raise GatewayBridgeError(
                code="gateway_unavailable",
                message="Could not communicate with the gateway websocket.",
            ) from None

    async def request_stop(self, *, route_id: str) -> bool:
        websocket_url = f"{self._websocket_base_url}/{route_id}"
        connect = _resolve_websocket_connect()

        try:
            async with connect(
                websocket_url,
                open_timeout=self._connect_timeout_seconds,
                close_timeout=self._connect_timeout_seconds,
            ) as socket:
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

                await socket.send(json.dumps({"type": "stop_turn"}))
                payload = await _recv_json(socket)
                event_type = payload.get("type")
                if event_type == "stop_ack":
                    return bool(payload.get("stop_requested", False))
                if event_type == "error":
                    raise GatewayBridgeError(
                        code=str(payload.get("code", "gateway_error")),
                        message=str(payload.get("message", "Gateway returned an error.")),
                    )
                raise GatewayBridgeError(
                    code="invalid_stop_ack",
                    message="Gateway did not send a stop acknowledgement.",
                )
        except GatewayBridgeError:
            raise
        except Exception:  # pragma: no cover - exception types depend on websocket lib
            raise GatewayBridgeError(
                code="gateway_unavailable",
                message="Could not communicate with the gateway websocket.",
            ) from None

    async def submit_approval(
        self,
        *,
        route_id: str,
        approval_id: str,
        approved: bool,
    ) -> bool:
        websocket_url = f"{self._websocket_base_url}/{route_id}"
        connect = _resolve_websocket_connect()

        try:
            async with connect(
                websocket_url,
                open_timeout=self._connect_timeout_seconds,
                close_timeout=self._connect_timeout_seconds,
            ) as socket:
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

                await socket.send(
                    json.dumps(
                        {
                            "type": "approval_response",
                            "approval_id": approval_id,
                            "approved": approved,
                        }
                    )
                )

                payload = await _recv_json(socket)
                event_type = payload.get("type")
                if event_type == "approval_ack":
                    return bool(payload.get("resolved", False))
                if event_type == "error":
                    raise GatewayBridgeError(
                        code=str(payload.get("code", "gateway_error")),
                        message=str(payload.get("message", "Gateway returned an error.")),
                    )
                raise GatewayBridgeError(
                    code="invalid_approval_ack",
                    message="Gateway did not send an approval acknowledgement.",
                )
        except GatewayBridgeError:
            raise
        except Exception:  # pragma: no cover - exception types depend on websocket lib
            raise GatewayBridgeError(
                code="gateway_unavailable",
                message="Could not communicate with the gateway websocket.",
            ) from None


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
