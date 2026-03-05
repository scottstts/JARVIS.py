"""Unit tests for Telegram->gateway bridge behavior."""

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ui.config import UISettings
from ui.gateway_client import GatewayBridgeError, GatewayDeltaEvent, GatewayDoneEvent
from ui.telegram_api import DraftMessage
from ui.telegram_bot import (
    IncomingTextMessage,
    TelegramGatewayBridge,
    parse_incoming_text_message,
    route_id_for_chat,
    split_telegram_message,
)


@dataclass(slots=True)
class _SentMessage:
    chat_id: int
    text: str


@dataclass(slots=True)
class _SentDraft:
    chat_id: int
    draft: DraftMessage


class _FakeTelegramClient:
    def __init__(self, updates: list[dict[str, Any]] | None = None) -> None:
        self._updates = updates or []
        self.sent_messages: list[_SentMessage] = []
        self.sent_drafts: list[_SentDraft] = []
        self.closed = False

    async def get_me(self) -> dict[str, Any]:
        return {"id": 1, "username": "jarvis_test_bot"}

    async def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        _ = offset, timeout_seconds, limit
        updates = self._updates
        self._updates = []
        return updates

    async def send_message(self, *, chat_id: int, text: str) -> dict[str, Any]:
        self.sent_messages.append(_SentMessage(chat_id=chat_id, text=text))
        return {"message_id": len(self.sent_messages)}

    async def send_message_draft(self, *, chat_id: int, draft: DraftMessage) -> bool:
        self.sent_drafts.append(_SentDraft(chat_id=chat_id, draft=draft))
        return True

    async def aclose(self) -> None:
        self.closed = True


class _FakeGatewayClient:
    def __init__(
        self,
        *,
        events: list[GatewayDeltaEvent | GatewayDoneEvent] | None = None,
        error: GatewayBridgeError | None = None,
    ) -> None:
        self._events = events or []
        self._error = error
        self.calls: list[tuple[str, str]] = []

    async def stream_turn(self, *, route_id: str, user_text: str) -> AsyncIterator[Any]:
        self.calls.append((route_id, user_text))
        if self._error is not None:
            raise self._error
        for event in self._events:
            yield event


def _settings() -> UISettings:
    return UISettings(
        telegram_token="test-token",
        stream_draft_min_interval_seconds=0.0,
        stream_draft_min_chars=1,
    )


class TelegramBotBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_poll_once_processes_private_text_update(self) -> None:
        updates = [
            {
                "update_id": 10,
                "message": {
                    "message_id": 100,
                    "chat": {"id": 123, "type": "private"},
                    "text": "hello",
                },
            }
        ]
        telegram = _FakeTelegramClient(updates=updates)
        gateway = _FakeGatewayClient(
            events=[GatewayDoneEvent(session_id="s", text="pong")],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        next_offset = await bridge.poll_once(offset=None)

        self.assertEqual(next_offset, 11)
        self.assertEqual(gateway.calls, [("tg_123", "hello")])
        self.assertEqual([message.text for message in telegram.sent_messages], ["pong"])

    async def test_poll_once_advances_offset_for_ignored_update(self) -> None:
        telegram = _FakeTelegramClient(
            updates=[
                {
                    "update_id": 20,
                    "edited_message": {
                        "chat": {"id": 123, "type": "private"},
                        "text": "ignored",
                    },
                }
            ]
        )
        gateway = _FakeGatewayClient(
            events=[GatewayDoneEvent(session_id="s", text="pong")],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        next_offset = await bridge.poll_once(offset=10)

        self.assertEqual(next_offset, 21)
        self.assertEqual(gateway.calls, [])
        self.assertEqual(telegram.sent_messages, [])

    async def test_handle_message_streams_drafts_then_final_message(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="stream-"),
                GatewayDeltaEvent(session_id="session", delta="reply"),
                GatewayDoneEvent(session_id="session", text="stream-reply"),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual([draft.draft.text for draft in telegram.sent_drafts], ["stream-", "stream-reply"])
        self.assertEqual([message.text for message in telegram.sent_messages], ["stream-reply"])

    async def test_handle_message_ignores_non_private_chat(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[GatewayDoneEvent(session_id="session", text="should-not-send")],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=-10, chat_type="group", text="hi"),
        )

        self.assertEqual(gateway.calls, [])
        self.assertEqual(telegram.sent_messages, [])
        self.assertEqual(telegram.sent_drafts, [])

    async def test_long_final_text_is_split(self) -> None:
        long_text = "a" * 5000
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[GatewayDoneEvent(session_id="session", text=long_text)],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=9, chat_type="private", text="hi"),
        )

        self.assertEqual(len(telegram.sent_messages), 2)
        self.assertEqual(len(telegram.sent_messages[0].text), 4096)
        self.assertEqual(len(telegram.sent_messages[1].text), 904)

    async def test_gateway_error_sends_error_message(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            error=GatewayBridgeError(code="internal_error", message="gateway failed"),
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=9, chat_type="private", text="hi"),
        )

        self.assertEqual([message.text for message in telegram.sent_messages], ["gateway failed"])


class TelegramBotHelpersTests(unittest.TestCase):
    def test_parse_incoming_text_message(self) -> None:
        parsed = parse_incoming_text_message(
            {
                "update_id": 12,
                "message": {
                    "chat": {"id": 321, "type": "private"},
                    "text": "  hello ",
                },
            }
        )
        self.assertIsNotNone(parsed)
        if parsed is None:
            self.fail("Expected parsed incoming message.")
        self.assertEqual(parsed.update_id, 12)
        self.assertEqual(parsed.chat_id, 321)
        self.assertEqual(parsed.text, "hello")

    def test_route_id_for_negative_chat_id(self) -> None:
        self.assertEqual(route_id_for_chat(-123), "tg_n123")

    def test_split_telegram_message(self) -> None:
        chunks = split_telegram_message("abc", max_chars=2)
        self.assertEqual(chunks, ["ab", "c"])
