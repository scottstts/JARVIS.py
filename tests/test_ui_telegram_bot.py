"""Unit tests for Telegram->gateway bridge behavior."""

from __future__ import annotations

import unittest
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ui.telegram.api import DraftMessage, TelegramAPIError
from ui.telegram.bot import (
    IncomingTextMessage,
    TelegramGatewayBridge,
    parse_incoming_text_message,
    route_id_for_chat,
    split_telegram_message,
)
from ui.telegram.config import UISettings
from ui.telegram.gateway_client import GatewayBridgeError, GatewayDeltaEvent, GatewayDoneEvent


@dataclass(slots=True)
class _SentMessage:
    chat_id: int
    text: str
    parse_mode: str | None


@dataclass(slots=True)
class _SentDraft:
    chat_id: int
    draft: DraftMessage
    parse_mode: str | None


class _FakeTelegramClient:
    def __init__(
        self,
        updates: list[dict[str, Any]] | None = None,
        *,
        draft_errors: list[TelegramAPIError] | None = None,
        message_errors: list[TelegramAPIError] | None = None,
    ) -> None:
        self._updates = updates or []
        self._draft_errors = draft_errors or []
        self._message_errors = message_errors or []
        self.sent_messages: list[_SentMessage] = []
        self.sent_drafts: list[_SentDraft] = []
        self.message_attempts = 0
        self.draft_attempts = 0
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

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
    ) -> dict[str, Any]:
        self.message_attempts += 1
        if self._message_errors:
            raise self._message_errors.pop(0)
        self.sent_messages.append(
            _SentMessage(chat_id=chat_id, text=text, parse_mode=parse_mode)
        )
        return {"message_id": len(self.sent_messages)}

    async def send_message_draft(
        self,
        *,
        chat_id: int,
        draft: DraftMessage,
        parse_mode: str | None = None,
    ) -> bool:
        self.draft_attempts += 1
        if self._draft_errors:
            raise self._draft_errors.pop(0)
        self.sent_drafts.append(
            _SentDraft(chat_id=chat_id, draft=draft, parse_mode=parse_mode)
        )
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
                    "from": {"id": 123},
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

    async def test_poll_once_ignores_private_message_from_unauthorized_user(self) -> None:
        updates = [
            {
                "update_id": 10,
                "message": {
                    "message_id": 100,
                    "from": {"id": 123},
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
            settings=UISettings(
                telegram_token="test-token",
                telegram_allowed_user_id=999,
                stream_draft_min_interval_seconds=0.0,
                stream_draft_min_chars=1,
            ),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        next_offset = await bridge.poll_once(offset=None)

        self.assertEqual(next_offset, 11)
        self.assertEqual(gateway.calls, [])
        self.assertEqual(telegram.sent_messages, [])

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
        self.assertEqual([draft.parse_mode for draft in telegram.sent_drafts], ["HTML", "HTML"])
        self.assertEqual([message.parse_mode for message in telegram.sent_messages], ["HTML"])

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

    async def test_handle_message_ignores_private_chat_when_owner_gate_does_not_match(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[GatewayDoneEvent(session_id="session", text="should-not-send")],
        )
        bridge = TelegramGatewayBridge(
            settings=UISettings(
                telegram_token="test-token",
                telegram_allowed_user_id=777,
                stream_draft_min_interval_seconds=0.0,
                stream_draft_min_chars=1,
            ),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(
                update_id=1,
                chat_id=123,
                chat_type="private",
                text="hi",
                sender_user_id=123,
            ),
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

    async def test_draft_rate_limit_disables_drafts_for_current_and_next_turn(self) -> None:
        telegram = _FakeTelegramClient(
            draft_errors=[
                TelegramAPIError(
                    code="telegram_api_error_429",
                    message="Too Many Requests",
                    retry_after_seconds=3,
                )
            ]
        )
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="stream-"),
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
        await bridge.handle_message(
            IncomingTextMessage(update_id=2, chat_id=777, chat_type="private", text="again"),
        )

        self.assertEqual(telegram.draft_attempts, 1)
        self.assertEqual(telegram.sent_drafts, [])
        self.assertEqual([message.text for message in telegram.sent_messages], ["stream-reply", "stream-reply"])

    async def test_final_message_renders_markdown_as_html(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDoneEvent(
                    session_id="session",
                    text="**bold** and `code` with [link](https://example.com)",
                )
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=9, chat_type="private", text="hi"),
        )

        self.assertEqual(len(telegram.sent_messages), 1)
        self.assertEqual(telegram.sent_messages[0].parse_mode, "HTML")
        self.assertEqual(
            telegram.sent_messages[0].text,
            '<b>bold</b> and <code>code</code> with <a href="https://example.com">link</a>',
        )

    async def test_final_message_falls_back_to_plain_text_on_formatting_error(self) -> None:
        telegram = _FakeTelegramClient(
            message_errors=[
                TelegramAPIError(
                    code="telegram_api_error_400",
                    message="Bad Request: can't parse entities",
                )
            ]
        )
        gateway = _FakeGatewayClient(
            events=[GatewayDoneEvent(session_id="session", text="**bold**")],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=9, chat_type="private", text="hi"),
        )

        self.assertEqual(telegram.message_attempts, 2)
        self.assertEqual(len(telegram.sent_messages), 1)
        self.assertEqual(telegram.sent_messages[0].text, "**bold**")
        self.assertIsNone(telegram.sent_messages[0].parse_mode)


class TelegramBotHelpersTests(unittest.TestCase):
    def test_parse_incoming_text_message(self) -> None:
        parsed = parse_incoming_text_message(
            {
                "update_id": 12,
                "message": {
                    "from": {"id": 321},
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
        self.assertEqual(parsed.sender_user_id, 321)
        self.assertEqual(parsed.text, "hello")

    def test_route_id_for_negative_chat_id(self) -> None:
        self.assertEqual(route_id_for_chat(-123), "tg_n123")

    def test_split_telegram_message(self) -> None:
        chunks = split_telegram_message("abc", max_chars=2)
        self.assertEqual(chunks, ["ab", "c"])
