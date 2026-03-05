"""Websocket behavior tests for gateway Starlette app."""

from __future__ import annotations

import unittest

from core import AgentTextDeltaEvent, AgentTurnDoneEvent, ContextBudgetError
from gateway import GatewaySettings, create_app
from starlette.testclient import TestClient


class _FakeRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.active_sessions: dict[str, str | None] = {}

    def active_session_id(self, route_id: str) -> str | None:
        return self.active_sessions.get(route_id)

    async def stream_turn(self, route_id: str, user_text: str):
        self.calls.append((route_id, user_text))
        if user_text == "budget":
            raise ContextBudgetError("budget exceeded")
        if user_text == "boom":
            raise RuntimeError("unexpected")
        yield AgentTextDeltaEvent(
            session_id=f"{route_id}-session",
            delta="echo:",
        )
        yield AgentTurnDoneEvent(
            session_id=f"{route_id}-session",
            response_text=f"echo:{user_text}",
        )


class GatewayAppTests(unittest.TestCase):
    def test_ready_event_then_assistant_reply(self) -> None:
        router = _FakeRouter()
        router.active_sessions["dm_1"] = None
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=router,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_1") as socket:
                ready = socket.receive_json()
                self.assertEqual(ready["type"], "ready")
                self.assertEqual(ready["route_id"], "dm_1")
                self.assertIsNone(ready["session_id"])

                socket.send_json({"type": "user_message", "text": "hello"})
                delta = socket.receive_json()
                self.assertEqual(delta["type"], "assistant_delta")
                self.assertEqual(delta["session_id"], "dm_1-session")
                self.assertEqual(delta["delta"], "echo:")

                reply = socket.receive_json()
                self.assertEqual(reply["type"], "assistant_message")
                self.assertEqual(reply["session_id"], "dm_1-session")
                self.assertEqual(reply["text"], "echo:hello")
                self.assertEqual(router.calls, [("dm_1", "hello")])

    def test_invalid_json_emits_error_event(self) -> None:
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=_FakeRouter(),
        )
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_2") as socket:
                _ = socket.receive_json()  # ready
                socket.send_text("{bad-json")
                error = socket.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["code"], "invalid_json")

    def test_protocol_error_emits_error_event(self) -> None:
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=5),
            router=_FakeRouter(),
        )
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_3") as socket:
                _ = socket.receive_json()  # ready
                socket.send_json({"type": "user_message", "text": "too long"})
                error = socket.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["code"], "message_too_large")

    def test_invalid_route_id_is_rejected(self) -> None:
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=_FakeRouter(),
        )
        with TestClient(app) as client:
            with client.websocket_connect("/ws/bad%20id") as socket:
                error = socket.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["code"], "invalid_route_id")

    def test_context_budget_error_is_mapped(self) -> None:
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=_FakeRouter(),
        )
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_4") as socket:
                _ = socket.receive_json()  # ready
                socket.send_json({"type": "user_message", "text": "budget"})
                error = socket.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["code"], "context_budget_exceeded")

    def test_internal_error_is_mapped(self) -> None:
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=_FakeRouter(),
        )
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_5") as socket:
                _ = socket.receive_json()  # ready
                socket.send_json({"type": "user_message", "text": "boom"})
                error = socket.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["code"], "internal_error")
