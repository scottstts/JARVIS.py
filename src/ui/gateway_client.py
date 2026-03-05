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
class GatewayDoneEvent:
    session_id: str
    text: str
    command: str | None = None
    compaction_performed: bool = False
    type: str = "assistant_message"


GatewayTurnEvent = GatewayDeltaEvent | GatewayDoneEvent


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
                        yield GatewayDoneEvent(
                            session_id=str(payload.get("session_id", "")),
                            text=str(payload.get("text", "")),
                            command=(
                                str(payload["command"])
                                if payload.get("command") is not None
                                else None
                            ),
                            compaction_performed=bool(payload.get("compaction_performed", False)),
                        )
                        return
                    if event_type == "error":
                        raise GatewayBridgeError(
                            code=str(payload.get("code", "gateway_error")),
                            message=str(payload.get("message", "Gateway returned an error.")),
                        )
        except GatewayBridgeError:
            raise
        except Exception as exc:  # pragma: no cover - exception types depend on websocket lib
            raise GatewayBridgeError(
                code="gateway_unavailable",
                message="Could not communicate with the gateway websocket.",
            ) from exc


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
