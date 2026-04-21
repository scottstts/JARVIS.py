"""Unit tests for websocket payload protocol helpers."""

from __future__ import annotations

import unittest

from jarvis.gateway.protocol import ProtocolError, build_route_event_payload, parse_client_event
from jarvis.gateway.route_events import RouteAuthRequiredEvent, RouteTaskStatusEvent
from jarvis.ui.telegram.gateway_client import (
    GatewayAuthRequiredEvent,
    GatewayTaskStatusEvent,
    _parse_route_event,
)


class GatewayProtocolTests(unittest.TestCase):
    def test_parse_user_message_success(self) -> None:
        parsed = parse_client_event(
            {
                "type": "user_message",
                "text": "hello",
                "client_message_id": "msg_1",
            },
            max_message_chars=100,
        )
        self.assertEqual(parsed.text, "hello")
        self.assertEqual(parsed.client_message_id, "msg_1")

    def test_payload_must_be_object(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            parse_client_event(["user_message"], max_message_chars=100)
        self.assertEqual(context.exception.code, "invalid_payload")

    def test_type_must_be_user_message(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            parse_client_event({"type": "ping"}, max_message_chars=100)
        self.assertEqual(context.exception.code, "unsupported_event_type")

    def test_parse_stop_turn_success(self) -> None:
        parsed = parse_client_event(
            {"type": "stop_turn"},
            max_message_chars=100,
        )
        self.assertEqual(type(parsed).__name__, "ClientStopTurn")

    def test_parse_approval_response_success(self) -> None:
        parsed = parse_client_event(
            {"type": "approval_response", "approval_id": "approval_1", "approved": True},
            max_message_chars=100,
        )
        self.assertEqual(type(parsed).__name__, "ClientApprovalResponse")
        self.assertEqual(parsed.approval_id, "approval_1")
        self.assertTrue(parsed.approved)

    def test_parse_approval_response_validates_payload(self) -> None:
        with self.assertRaises(ProtocolError) as missing_id:
            parse_client_event(
                {"type": "approval_response", "approval_id": "", "approved": True},
                max_message_chars=100,
            )
        self.assertEqual(missing_id.exception.code, "invalid_approval_id")

        with self.assertRaises(ProtocolError) as invalid_decision:
            parse_client_event(
                {"type": "approval_response", "approval_id": "approval_1", "approved": "yes"},
                max_message_chars=100,
            )
        self.assertEqual(invalid_decision.exception.code, "invalid_approval_decision")

    def test_text_must_be_non_empty_string(self) -> None:
        with self.assertRaises(ProtocolError) as non_string:
            parse_client_event(
                {
                    "type": "user_message",
                    "text": 123,
                    "client_message_id": "msg_1",
                },
                max_message_chars=100,
            )
        self.assertEqual(non_string.exception.code, "invalid_message_text")

        with self.assertRaises(ProtocolError) as empty:
            parse_client_event(
                {
                    "type": "user_message",
                    "text": "   ",
                    "client_message_id": "msg_1",
                },
                max_message_chars=100,
            )
        self.assertEqual(empty.exception.code, "empty_message")

    def test_message_length_is_limited(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            parse_client_event(
                {
                    "type": "user_message",
                    "text": "hello",
                    "client_message_id": "msg_1",
                },
                max_message_chars=4,
            )
        self.assertEqual(context.exception.code, "message_too_large")

    def test_client_message_id_is_required(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            parse_client_event(
                {"type": "user_message", "text": "hello"},
                max_message_chars=100,
            )
        self.assertEqual(context.exception.code, "invalid_client_message_id")

    def test_auth_required_event_round_trips_through_gateway_payload(self) -> None:
        payload = build_route_event_payload(
            RouteAuthRequiredEvent(
                route_id="route_1",
                agent_kind="main",
                agent_name="Jarvis",
                session_id="session_1",
                provider="codex",
                auth_kind="openai_oauth",
                login_id="login_1",
                auth_url="https://auth.example/login",
                message="Open the browser login URL.",
            )
        )

        parsed = _parse_route_event(payload)

        self.assertIsInstance(parsed, GatewayAuthRequiredEvent)
        self.assertEqual(parsed.provider, "codex")
        self.assertEqual(parsed.auth_kind, "openai_oauth")
        self.assertEqual(parsed.login_id, "login_1")
        self.assertEqual(parsed.auth_url, "https://auth.example/login")
        self.assertEqual(parsed.message, "Open the browser login URL.")

    def test_task_status_event_round_trips_through_gateway_payload(self) -> None:
        payload = build_route_event_payload(
            RouteTaskStatusEvent(
                route_id="route_1",
                agent_kind="main",
                agent_name="Jarvis",
                active=True,
                reason="user_message_queued",
            )
        )

        parsed = _parse_route_event(payload)

        self.assertIsInstance(parsed, GatewayTaskStatusEvent)
        self.assertTrue(parsed.active)
        self.assertEqual(parsed.reason, "user_message_queued")
