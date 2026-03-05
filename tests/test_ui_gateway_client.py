"""Unit tests for gateway websocket client event mapping."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from ui.gateway_client import GatewayBridgeError, GatewayDeltaEvent, GatewayDoneEvent, GatewayWebSocketClient


class _FakeSocket:
    def __init__(self, incoming: list[str]) -> None:
        self._incoming = incoming
        self.sent: list[str] = []

    async def recv(self) -> str:
        if not self._incoming:
            raise RuntimeError("No more frames.")
        return self._incoming.pop(0)

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


class _FakeConnection:
    def __init__(self, socket: _FakeSocket) -> None:
        self._socket = socket

    async def __aenter__(self) -> _FakeSocket:
        return self._socket

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = exc_type, exc, tb
        return False


class GatewayWebSocketClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_turn_maps_delta_and_done_events(self) -> None:
        socket = _FakeSocket(
            incoming=[
                json.dumps({"type": "ready", "route_id": "tg_1", "session_id": None}),
                json.dumps({"type": "assistant_delta", "session_id": "s1", "delta": "hel"}),
                json.dumps({"type": "assistant_message", "session_id": "s1", "text": "hello"}),
            ]
        )
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        with patch(
            "ui.gateway_client._resolve_websocket_connect",
            return_value=lambda *args, **kwargs: _FakeConnection(socket),
        ):
            events = [event async for event in client.stream_turn(route_id="tg_1", user_text="hi")]

        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], GatewayDeltaEvent)
        self.assertIsInstance(events[1], GatewayDoneEvent)
        self.assertEqual(events[1].text, "hello")
        outbound = json.loads(socket.sent[0])
        self.assertEqual(outbound, {"type": "user_message", "text": "hi"})

    async def test_stream_turn_raises_on_gateway_error_event(self) -> None:
        socket = _FakeSocket(
            incoming=[
                json.dumps({"type": "ready", "route_id": "tg_1", "session_id": None}),
                json.dumps({"type": "error", "code": "internal_error", "message": "boom"}),
            ]
        )
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        with patch(
            "ui.gateway_client._resolve_websocket_connect",
            return_value=lambda *args, **kwargs: _FakeConnection(socket),
        ):
            with self.assertRaises(GatewayBridgeError) as context:
                _ = [event async for event in client.stream_turn(route_id="tg_1", user_text="hi")]
        self.assertEqual(context.exception.code, "internal_error")
