"""Websocket behavior tests for gateway Starlette app."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from jarvis.core import (
    AgentAssistantMessageEvent,
    AgentApprovalRequestEvent,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentTurnDoneEvent,
    ContextBudgetError,
)
from jarvis.gateway import GatewaySettings, create_app
from jarvis.gateway.app import _build_default_router, _send_json_if_open
from jarvis.llm import ProviderTimeoutError
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from tests.helpers import build_core_settings


class _FakeRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.active_sessions: dict[str, str | None] = {}
        self.stop_requests: list[str] = []
        self.approval_resolutions: list[tuple[str, str, bool]] = []

    def active_session_id(self, route_id: str) -> str | None:
        return self.active_sessions.get(route_id)

    def request_stop(self, route_id: str) -> bool:
        self.stop_requests.append(route_id)
        return route_id == "dm_stop"

    def resolve_approval(self, route_id: str, approval_id: str, approved: bool) -> bool:
        self.approval_resolutions.append((route_id, approval_id, approved))
        return route_id == "dm_approve" and approval_id == "approval_1"

    async def stream_turn(self, route_id: str, user_text: str):
        self.calls.append((route_id, user_text))
        if user_text == "budget":
            raise ContextBudgetError("budget exceeded")
        if user_text == "timeout":
            raise ProviderTimeoutError("Request timed out.")
        if user_text == "boom":
            raise RuntimeError("unexpected")
        if user_text == "tool":
            yield AgentToolCallEvent(
                session_id=f"{route_id}-session",
                tool_names=("bash",),
            )
            yield AgentTurnDoneEvent(
                session_id=f"{route_id}-session",
                response_text="",
            )
            return
        if user_text == "approval":
            yield AgentApprovalRequestEvent(
                session_id=f"{route_id}-session",
                approval_id="approval_1",
                kind="bash_command",
                summary="Install a CLI.",
                details="I want to install a CLI for this task.",
                command="curl https://example.com/install.sh | sh",
                inspection_url="https://example.com",
            )
            yield AgentTurnDoneEvent(
                session_id=f"{route_id}-session",
                response_text="",
                interrupted=True,
            )
            return
        yield AgentTextDeltaEvent(
            session_id=f"{route_id}-session",
            delta="echo:",
        )
        yield AgentAssistantMessageEvent(
            session_id=f"{route_id}-session",
            text=f"echo:{user_text}",
        )
        yield AgentTurnDoneEvent(
            session_id=f"{route_id}-session",
            response_text=f"echo:{user_text}",
        )


class GatewayAppTests(unittest.TestCase):
    def test_default_router_stores_main_transcripts_under_jarvis_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core_settings = build_core_settings(root_dir=Path(tmp))

            with patch("jarvis.gateway.app.RouteRuntime") as runtime_cls:
                runtime_cls.return_value.active_session_id.return_value = None
                router = _build_default_router(
                    core_settings=core_settings,
                    llm_service=object(),  # type: ignore[arg-type]
                )

                self.assertIsNone(router.active_session_id("tg_123"))

            runtime_core_settings = runtime_cls.call_args.kwargs["core_settings"]
            self.assertEqual(
                runtime_core_settings.transcript_archive_dir,
                core_settings.transcript_archive_dir / "jarvis" / "tg_123",
            )

    def test_default_app_lifespan_healthchecks_remote_tool_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "jarvis.gateway.app.ensure_remote_tool_runtime_healthy",
                new=AsyncMock(),
            ) as healthcheck:
                with patch.dict(
                    "os.environ",
                    {"JARVIS_TOOL_RUNTIME_BASE_URL": "http://tool_runtime:8081"},
                    clear=False,
                ):
                    app = create_app(
                        gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
                        core_settings=build_core_settings(root_dir=Path(tmp)),
                        llm_service=object(),  # type: ignore[arg-type]
                    )

                    with TestClient(app):
                        pass

        self.assertTrue(healthcheck.await_count >= 1)

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

                socket.send_json(
                    {
                        "type": "user_message",
                        "text": "hello",
                        "client_message_id": "msg_1",
                    }
                )
                delta = socket.receive_json()
                self.assertEqual(delta["type"], "assistant_delta")
                self.assertEqual(delta["session_id"], "dm_1-session")
                self.assertEqual(delta["delta"], "echo:")

                reply = socket.receive_json()
                self.assertEqual(reply["type"], "assistant_message")
                self.assertEqual(reply["session_id"], "dm_1-session")
                self.assertEqual(reply["text"], "echo:hello")

                done = socket.receive_json()
                self.assertEqual(done["type"], "turn_done")
                self.assertEqual(done["session_id"], "dm_1-session")
                self.assertEqual(done["response_text"], "echo:hello")
                self.assertEqual(router.calls, [("dm_1", "hello")])

    def test_tool_call_event_is_forwarded(self) -> None:
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=_FakeRouter(),
        )

        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_tool") as socket:
                _ = socket.receive_json()  # ready
                socket.send_json(
                    {
                        "type": "user_message",
                        "text": "tool",
                        "client_message_id": "msg_1",
                    }
                )

                tool_event = socket.receive_json()
                self.assertEqual(tool_event["type"], "tool_call")
                self.assertEqual(tool_event["session_id"], "dm_tool-session")
                self.assertEqual(tool_event["tool_names"], ["bash"])

                done = socket.receive_json()
                self.assertEqual(done["type"], "turn_done")

    def test_approval_request_event_is_forwarded(self) -> None:
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=_FakeRouter(),
        )

        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_approval") as socket:
                _ = socket.receive_json()  # ready
                socket.send_json(
                    {
                        "type": "user_message",
                        "text": "approval",
                        "client_message_id": "msg_1",
                    }
                )

                approval_event = socket.receive_json()
                self.assertEqual(approval_event["type"], "approval_request")
                self.assertEqual(approval_event["session_id"], "dm_approval-session")
                self.assertEqual(approval_event["approval_id"], "approval_1")
                self.assertEqual(approval_event["kind"], "bash_command")
                self.assertEqual(
                    approval_event["command"],
                    "curl https://example.com/install.sh | sh",
                )

                done = socket.receive_json()
                self.assertEqual(done["type"], "turn_done")
                self.assertTrue(done["interrupted"])

    def test_stop_turn_event_is_acknowledged(self) -> None:
        router = _FakeRouter()
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=router,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_stop") as socket:
                _ = socket.receive_json()  # ready
                socket.send_json({"type": "stop_turn"})
                ack = socket.receive_json()
                self.assertEqual(ack["type"], "stop_ack")
                self.assertTrue(ack["stop_requested"])
                self.assertEqual(router.stop_requests, ["dm_stop"])

    def test_approval_response_event_is_acknowledged(self) -> None:
        router = _FakeRouter()
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=router,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_approve") as socket:
                _ = socket.receive_json()  # ready
                socket.send_json(
                    {
                        "type": "approval_response",
                        "approval_id": "approval_1",
                        "approved": True,
                    }
                )
                ack = socket.receive_json()
                self.assertEqual(ack["type"], "approval_ack")
                self.assertTrue(ack["resolved"])
                self.assertEqual(
                    router.approval_resolutions,
                    [("dm_approve", "approval_1", True)],
                )

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
                socket.send_json(
                    {
                        "type": "user_message",
                        "text": "too long",
                        "client_message_id": "msg_1",
                    }
                )
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
                socket.send_json(
                    {
                        "type": "user_message",
                        "text": "budget",
                        "client_message_id": "msg_1",
                    }
                )
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
                socket.send_json(
                    {
                        "type": "user_message",
                        "text": "boom",
                        "client_message_id": "msg_1",
                    }
                )
                error = socket.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["code"], "internal_error")

    def test_provider_timeout_is_mapped(self) -> None:
        app = create_app(
            gateway_settings=GatewaySettings(websocket_path="/ws", max_message_chars=50),
            router=_FakeRouter(),
        )
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dm_6") as socket:
                _ = socket.receive_json()  # ready
                socket.send_json(
                    {
                        "type": "user_message",
                        "text": "timeout",
                        "client_message_id": "msg_1",
                    }
                )
                error = socket.receive_json()
                self.assertEqual(error["type"], "error")
                self.assertEqual(error["code"], "provider_timeout")

    def test_send_json_if_open_returns_false_on_disconnect(self) -> None:
        class _DisconnectingWebSocket:
            async def send_json(self, payload):
                raise WebSocketDisconnect(code=1006)

        result = asyncio.run(
            _send_json_if_open(
                _DisconnectingWebSocket(),
                {"type": "error", "code": "internal_error"},
            )
        )

        self.assertFalse(result)

    def test_send_json_if_open_returns_false_after_close(self) -> None:
        class _ClosedWebSocket:
            async def send_json(self, payload):
                raise RuntimeError('Cannot call "send" once a close message has been sent.')

        result = asyncio.run(
            _send_json_if_open(
                _ClosedWebSocket(),
                {"type": "error", "code": "internal_error"},
            )
        )

        self.assertFalse(result)
