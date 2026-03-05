"""Unit tests for websocket payload protocol helpers."""

from __future__ import annotations

import unittest

from gateway.protocol import ProtocolError, parse_client_event


class GatewayProtocolTests(unittest.TestCase):
    def test_parse_user_message_success(self) -> None:
        parsed = parse_client_event(
            {"type": "user_message", "text": "hello"},
            max_message_chars=100,
        )
        self.assertEqual(parsed.text, "hello")

    def test_payload_must_be_object(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            parse_client_event(["user_message"], max_message_chars=100)
        self.assertEqual(context.exception.code, "invalid_payload")

    def test_type_must_be_user_message(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            parse_client_event({"type": "ping"}, max_message_chars=100)
        self.assertEqual(context.exception.code, "unsupported_event_type")

    def test_text_must_be_non_empty_string(self) -> None:
        with self.assertRaises(ProtocolError) as non_string:
            parse_client_event({"type": "user_message", "text": 123}, max_message_chars=100)
        self.assertEqual(non_string.exception.code, "invalid_message_text")

        with self.assertRaises(ProtocolError) as empty:
            parse_client_event({"type": "user_message", "text": "   "}, max_message_chars=100)
        self.assertEqual(empty.exception.code, "empty_message")

    def test_message_length_is_limited(self) -> None:
        with self.assertRaises(ProtocolError) as context:
            parse_client_event(
                {"type": "user_message", "text": "hello"},
                max_message_chars=4,
            )
        self.assertEqual(context.exception.code, "message_too_large")
