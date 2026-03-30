"""Unit tests for gateway websocket client event mapping."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jarvis.ui.telegram.gateway_client import (
    GatewayApprovalRequestEvent,
    GatewayBridgeError,
    GatewayDeltaEvent,
    GatewayLocalNoticeEvent,
    GatewayMessageEvent,
    GatewayToolCallEvent,
    GatewayTurnDoneEvent,
    GatewayWebSocketClient,
)


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
                json.dumps(
                    {
                        "type": "turn_started",
                        "session_id": "s1",
                        "turn_id": "turn_1",
                        "turn_kind": "user",
                        "client_message_id": "msg_1",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant_delta",
                        "session_id": "s1",
                        "turn_id": "turn_1",
                        "delta": "hel",
                    }
                ),
                json.dumps(
                    {
                        "type": "tool_call",
                        "session_id": "s1",
                        "turn_id": "turn_1",
                        "tool_names": ["bash"],
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant_message",
                        "session_id": "s1",
                        "turn_id": "turn_1",
                        "text": "hello",
                    }
                ),
                json.dumps(
                    {
                        "type": "turn_done",
                        "session_id": "s1",
                        "turn_id": "turn_1",
                        "response_text": "hello",
                        "interrupted": True,
                    }
                ),
            ]
        )
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        with patch(
            "jarvis.ui.telegram.gateway_client._resolve_websocket_connect",
            return_value=lambda *args, **kwargs: _FakeConnection(socket),
        ), patch(
            "jarvis.ui.telegram.gateway_client.uuid4",
            return_value=SimpleNamespace(hex="msg_1"),
        ):
            events = [event async for event in client.stream_turn(route_id="tg_1", user_text="hi")]

        self.assertEqual(len(events), 4)
        self.assertIsInstance(events[0], GatewayDeltaEvent)
        self.assertIsInstance(events[1], GatewayToolCallEvent)
        self.assertIsInstance(events[2], GatewayMessageEvent)
        self.assertIsInstance(events[3], GatewayTurnDoneEvent)
        self.assertEqual(events[1].tool_names, ("bash",))
        self.assertEqual(events[2].text, "hello")
        self.assertEqual(events[3].response_text, "hello")
        self.assertTrue(events[3].interrupted)
        outbound = json.loads(socket.sent[0])
        self.assertEqual(outbound["type"], "user_message")
        self.assertEqual(outbound["text"], "hi")
        self.assertEqual(outbound["client_message_id"], "msg_1")

    async def test_request_stop_returns_acknowledged_state(self) -> None:
        socket = _FakeSocket(
            incoming=[
                json.dumps({"type": "ready", "route_id": "tg_1", "session_id": None}),
                json.dumps({"type": "stop_ack", "stop_requested": True}),
            ]
        )
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        with patch(
            "jarvis.ui.telegram.gateway_client._resolve_websocket_connect",
            return_value=lambda *args, **kwargs: _FakeConnection(socket),
        ):
            stop_requested = await client.request_stop(route_id="tg_1")

        self.assertTrue(stop_requested)
        outbound = json.loads(socket.sent[0])
        self.assertEqual(outbound, {"type": "stop_turn"})

    async def test_stream_turn_maps_approval_request_events(self) -> None:
        socket = _FakeSocket(
            incoming=[
                json.dumps({"type": "ready", "route_id": "tg_1", "session_id": None}),
                json.dumps(
                    {
                        "type": "turn_started",
                        "session_id": "s1",
                        "turn_id": "turn_1",
                        "turn_kind": "user",
                        "client_message_id": "msg_1",
                    }
                ),
                json.dumps(
                    {
                        "type": "approval_request",
                        "session_id": "s1",
                        "turn_id": "turn_1",
                        "approval_id": "approval_1",
                        "kind": "bash_command",
                        "summary": "Install a CLI.",
                        "details": "Need to install a CLI for this task.",
                        "command": "curl https://example.com/install.sh | sh",
                        "tool_name": "bash",
                        "inspection_url": "https://example.com",
                    }
                ),
                json.dumps(
                    {
                        "type": "turn_done",
                        "session_id": "s1",
                        "turn_id": "turn_1",
                        "response_text": "",
                    }
                ),
            ]
        )
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        with patch(
            "jarvis.ui.telegram.gateway_client._resolve_websocket_connect",
            return_value=lambda *args, **kwargs: _FakeConnection(socket),
        ), patch(
            "jarvis.ui.telegram.gateway_client.uuid4",
            return_value=SimpleNamespace(hex="msg_1"),
        ):
            events = [event async for event in client.stream_turn(route_id="tg_1", user_text="hi")]

        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], GatewayApprovalRequestEvent)
        self.assertEqual(events[0].approval_id, "approval_1")
        self.assertEqual(events[0].tool_name, "bash")
        self.assertEqual(events[1].response_text, "")

    async def test_stream_turn_keeps_local_notice_before_turn_started(self) -> None:
        socket = _FakeSocket(
            incoming=[
                json.dumps({"type": "ready", "route_id": "tg_1", "session_id": None}),
                json.dumps(
                    {
                        "type": "local_notice",
                        "session_id": "s1",
                        "turn_kind": "user",
                        "client_message_id": "msg_1",
                        "notice_kind": "compaction_started",
                        "text": "Compacting...",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant_message",
                        "session_id": "s1",
                        "turn_kind": "user",
                        "client_message_id": "msg_1",
                        "text": "Context compacted into a new session.",
                    }
                ),
                json.dumps(
                    {
                        "type": "turn_done",
                        "session_id": "s1",
                        "turn_kind": "user",
                        "client_message_id": "msg_1",
                        "response_text": "Context compacted into a new session.",
                        "command": "/compact",
                    }
                ),
            ]
        )
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        with patch(
            "jarvis.ui.telegram.gateway_client._resolve_websocket_connect",
            return_value=lambda *args, **kwargs: _FakeConnection(socket),
        ), patch(
            "jarvis.ui.telegram.gateway_client.uuid4",
            return_value=SimpleNamespace(hex="msg_1"),
        ):
            events = [event async for event in client.stream_turn(route_id="tg_1", user_text="/compact")]

        self.assertEqual(len(events), 3)
        self.assertIsInstance(events[0], GatewayLocalNoticeEvent)
        self.assertEqual(events[0].notice_kind, "compaction_started")
        self.assertEqual(events[0].text, "Compacting...")
        self.assertIsInstance(events[1], GatewayMessageEvent)
        self.assertIsInstance(events[2], GatewayTurnDoneEvent)

    async def test_submit_approval_returns_acknowledged_state(self) -> None:
        socket = _FakeSocket(
            incoming=[
                json.dumps({"type": "ready", "route_id": "tg_1", "session_id": None}),
                json.dumps({"type": "approval_ack", "resolved": True}),
            ]
        )
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        with patch(
            "jarvis.ui.telegram.gateway_client._resolve_websocket_connect",
            return_value=lambda *args, **kwargs: _FakeConnection(socket),
        ):
            resolved = await client.submit_approval(
                route_id="tg_1",
                approval_id="approval_1",
                approved=False,
            )

        self.assertTrue(resolved)
        outbound = json.loads(socket.sent[0])
        self.assertEqual(
            outbound,
            {
                "type": "approval_response",
                "approval_id": "approval_1",
                "approved": False,
            },
        )

    async def test_stream_turn_raises_on_gateway_error_event(self) -> None:
        socket = _FakeSocket(
            incoming=[
                json.dumps({"type": "ready", "route_id": "tg_1", "session_id": None}),
                json.dumps({"type": "error", "code": "internal_error", "message": "boom"}),
            ]
        )
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        with patch(
            "jarvis.ui.telegram.gateway_client._resolve_websocket_connect",
            return_value=lambda *args, **kwargs: _FakeConnection(socket),
        ):
            with self.assertRaises(GatewayBridgeError) as context:
                _ = [event async for event in client.stream_turn(route_id="tg_1", user_text="hi")]
        self.assertEqual(context.exception.code, "internal_error")

    async def test_stream_turn_hides_route_id_when_connect_fails(self) -> None:
        client = GatewayWebSocketClient(websocket_base_url="ws://localhost:8080/ws")

        def fail_connect(*args, **kwargs):
            _ = (args, kwargs)
            raise RuntimeError("connect failed for ws://localhost:8080/ws/tg_123")

        with patch(
            "jarvis.ui.telegram.gateway_client._resolve_websocket_connect",
            return_value=fail_connect,
        ):
            with self.assertRaises(GatewayBridgeError) as context:
                _ = [event async for event in client.stream_turn(route_id="tg_123", user_text="hi")]

        self.assertEqual(context.exception.code, "gateway_unavailable")
        self.assertTrue(context.exception.__suppress_context__)
        self.assertIsNone(context.exception.__cause__)
        self.assertNotIn("tg_123", str(context.exception))
