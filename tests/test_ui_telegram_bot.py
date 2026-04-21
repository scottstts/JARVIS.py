"""Unit tests for Telegram->gateway bridge behavior."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import SimpleNamespace

from jarvis.ui.telegram.api import DraftMessage, TelegramAPIError, TelegramRemoteFile
from jarvis.ui.telegram.bot import (
    IncomingTelegramApprovalCallback,
    IncomingTelegramFile,
    IncomingTextMessage,
    TelegramGatewayBridge,
    _next_stream_chunk,
    _clear_directory_contents,
    chat_id_for_route_id,
    parse_incoming_approval_callback,
    parse_incoming_message,
    parse_incoming_text_message,
    route_id_for_chat,
    split_telegram_message,
)
from jarvis.ui.telegram.config import UISettings
from jarvis.ui.telegram.gateway_client import (
    GatewayApprovalRequestEvent,
    GatewayBridgeError,
    GatewayDeltaEvent,
    GatewayErrorEvent,
    GatewayLocalNoticeEvent,
    GatewayMessageEvent,
    GatewaySystemNoticeEvent,
    GatewayTaskStatusEvent,
    GatewayToolCallEvent,
    GatewayTurnStartedEvent,
    GatewayTurnDoneEvent,
)


@dataclass(slots=True)
class _SentMessage:
    chat_id: int
    text: str
    parse_mode: str | None
    reply_markup: dict[str, Any] | None = None


@dataclass(slots=True)
class _SentDraft:
    chat_id: int
    draft: DraftMessage
    parse_mode: str | None


@dataclass(slots=True)
class _DownloadedFile:
    remote_file_path: str
    destination_path: Path


@dataclass(slots=True)
class _SentDocument:
    chat_id: int
    file_path: Path
    caption: str | None
    filename: str | None


@dataclass(slots=True)
class _EditedMessage:
    chat_id: int
    message_id: int
    text: str
    parse_mode: str | None
    reply_markup: dict[str, Any] | None


@dataclass(slots=True)
class _AnsweredCallback:
    callback_query_id: str
    text: str | None
    show_alert: bool


@dataclass(slots=True)
class _SentChatAction:
    chat_id: int
    action: str


class _FakeTelegramClient:
    def __init__(
        self,
        updates: list[dict[str, Any]] | None = None,
        *,
        remote_files: dict[str, TelegramRemoteFile] | None = None,
        download_payloads: dict[str, bytes] | None = None,
        draft_errors: list[TelegramAPIError] | None = None,
        message_errors: list[TelegramAPIError] | None = None,
    ) -> None:
        self._updates = updates or []
        self._remote_files = remote_files or {}
        self._download_payloads = download_payloads or {}
        self._draft_errors = draft_errors or []
        self._message_errors = message_errors or []
        self.sent_messages: list[_SentMessage] = []
        self.sent_drafts: list[_SentDraft] = []
        self.downloaded_files: list[_DownloadedFile] = []
        self.sent_documents: list[_SentDocument] = []
        self.edited_messages: list[_EditedMessage] = []
        self.answered_callbacks: list[_AnsweredCallback] = []
        self.sent_chat_actions: list[_SentChatAction] = []
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

    async def get_file(self, *, file_id: str) -> TelegramRemoteFile:
        return self._remote_files[file_id]

    async def download_file(
        self,
        *,
        remote_file_path: str,
        destination_path: str | Path,
    ) -> Path:
        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._download_payloads.get(remote_file_path, b""))
        self.downloaded_files.append(
            _DownloadedFile(
                remote_file_path=remote_file_path,
                destination_path=destination,
            )
        )
        return destination

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.message_attempts += 1
        if not text.strip():
            raise TelegramAPIError(
                code="telegram_api_error_400",
                message="Bad Request: text must be non-empty",
            )
        if self._message_errors:
            raise self._message_errors.pop(0)
        self.sent_messages.append(
            _SentMessage(
                chat_id=chat_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        )
        return {"message_id": len(self.sent_messages)}

    async def edit_message_text(
        self,
        *,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.edited_messages.append(
            _EditedMessage(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
        )
        return {"message_id": message_id}

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> bool:
        self.answered_callbacks.append(
            _AnsweredCallback(
                callback_query_id=callback_query_id,
                text=text,
                show_alert=show_alert,
            )
        )
        return True

    async def send_chat_action(
        self,
        *,
        chat_id: int,
        action: str,
    ) -> bool:
        self.sent_chat_actions.append(
            _SentChatAction(chat_id=chat_id, action=action)
        )
        return True

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

    async def send_document(
        self,
        *,
        chat_id: int,
        file_path: str | Path,
        caption: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        self.sent_documents.append(
            _SentDocument(
                chat_id=chat_id,
                file_path=Path(file_path),
                caption=caption,
                filename=filename,
            )
        )
        return {"message_id": len(self.sent_documents)}

    async def aclose(self) -> None:
        self.closed = True


class _FakeGatewayClient:
    def __init__(
        self,
        *,
        events: list[
            GatewayDeltaEvent
            | GatewayMessageEvent
            | GatewayToolCallEvent
            | GatewayApprovalRequestEvent
            | GatewayTurnDoneEvent
        ] | None = None,
        error: GatewayBridgeError | None = None,
        stop_error: GatewayBridgeError | None = None,
        approval_error: GatewayBridgeError | None = None,
        stop_requested: bool = False,
        approval_resolved: bool = True,
    ) -> None:
        self._events = events or []
        self._error = error
        self._stop_error = stop_error
        self._approval_error = approval_error
        self._stop_requested = stop_requested
        self._approval_resolved = approval_resolved
        self.calls: list[tuple[str, str]] = []
        self.stop_calls: list[str] = []
        self.approval_calls: list[tuple[str, str, bool]] = []

    async def stream_turn(self, *, route_id: str, user_text: str) -> AsyncIterator[Any]:
        self.calls.append((route_id, user_text))
        if self._error is not None:
            raise self._error
        for event in self._events:
            yield event

    async def request_stop(self, *, route_id: str) -> bool:
        self.stop_calls.append(route_id)
        if self._stop_error is not None:
            raise self._stop_error
        return self._stop_requested

    async def submit_approval(
        self,
        *,
        route_id: str,
        approval_id: str,
        approved: bool,
    ) -> bool:
        self.approval_calls.append((route_id, approval_id, approved))
        if self._approval_error is not None:
            raise self._approval_error
        return self._approval_resolved


class _BlockingGatewayClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.stop_calls: list[str] = []
        self.approval_calls: list[tuple[str, str, bool]] = []
        self.stream_started = asyncio.Event()
        self.release_turn = asyncio.Event()

    async def stream_turn(self, *, route_id: str, user_text: str) -> AsyncIterator[Any]:
        self.calls.append((route_id, user_text))
        self.stream_started.set()
        await self.release_turn.wait()
        yield GatewayTurnDoneEvent(
            session_id="session",
            response_text="",
            interrupted=True,
        )

    async def request_stop(self, *, route_id: str) -> bool:
        self.stop_calls.append(route_id)
        return True

    async def submit_approval(
        self,
        *,
        route_id: str,
        approval_id: str,
        approved: bool,
    ) -> bool:
        self.approval_calls.append((route_id, approval_id, approved))
        return True


class _FakeClosableRouteSession:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class _PersistentFakeRouteSession:
    def __init__(self) -> None:
        self.sent_messages: list[tuple[str, str]] = []
        self.stop_requested = False
        self.closed = False
        self._events: asyncio.Queue[Any] = asyncio.Queue()

    async def send_user_message(self, *, text: str, client_message_id: str) -> None:
        self.sent_messages.append((text, client_message_id))

    async def request_stop(self) -> bool:
        self.stop_requested = True
        return True

    async def submit_approval(self, *, approval_id: str, approved: bool) -> bool:
        _ = approval_id, approved
        return True

    async def events(self) -> AsyncIterator[Any]:
        while True:
            yield await self._events.get()

    async def emit(self, event: Any) -> None:
        await self._events.put(event)

    async def aclose(self) -> None:
        self.closed = True


class _PersistentFakeGatewayClient:
    def __init__(self, session: _PersistentFakeRouteSession) -> None:
        self._session = session
        self.connected_routes: list[str] = []

    async def connect_route(self, *, route_id: str) -> _PersistentFakeRouteSession:
        self.connected_routes.append(route_id)
        return self._session


def _settings(**overrides: object) -> UISettings:
    values: dict[str, object] = {
        "telegram_token": "test-token",
        "telegram_allowed_user_id": 777,
        "stream_transport": "edit",
        "stream_chunk_idle_flush_seconds": 0.0,
        "stream_chunk_min_chars": 1,
        "stream_chunk_max_chars": 4096,
        "stream_typing_indicator_interval_seconds": 4.0,
    }
    values.update(overrides)
    return UISettings(**values)


class TelegramBotBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_poll_once_processes_private_text_update(self) -> None:
        updates = [
            {
                "update_id": 10,
                "message": {
                    "message_id": 100,
                    "from": {"id": 123, "is_bot": False},
                    "chat": {"id": 123, "type": "private"},
                    "text": "hello",
                },
            }
        ]
        telegram = _FakeTelegramClient(updates=updates)
        gateway = _FakeGatewayClient(
            events=[
                GatewayMessageEvent(session_id="s", text="pong"),
                GatewayTurnDoneEvent(session_id="s", response_text="pong"),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(telegram_allowed_user_id=123),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        next_offset = await bridge.poll_once(offset=None)
        await bridge.wait_for_chat_idle(123)

        self.assertEqual(next_offset, 11)
        self.assertEqual(gateway.calls, [("tg_123", "hello")])
        self.assertEqual([message.text for message in telegram.sent_messages], ["pong"])

    async def test_poll_once_downloads_file_and_forwards_metadata_text(self) -> None:
        updates = [
            {
                "update_id": 10,
                "message": {
                    "message_id": 100,
                    "from": {"id": 123, "is_bot": False},
                    "chat": {"id": 123, "type": "private"},
                    "caption": "please review",
                    "document": {
                        "file_id": "file-1",
                        "file_unique_id": "unique-1",
                        "file_name": "report.pdf",
                        "mime_type": "application/pdf",
                        "file_size": 42,
                    },
                },
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            telegram = _FakeTelegramClient(
                updates=updates,
                remote_files={
                    "file-1": TelegramRemoteFile(
                        file_id="file-1",
                        file_unique_id="unique-1",
                        file_path="documents/report.pdf",
                        file_size=42,
                    )
                },
                download_payloads={"documents/report.pdf": b"%PDF-1.4"},
            )
            gateway = _FakeGatewayClient(
                events=[
                    GatewayMessageEvent(session_id="s", text="pong"),
                    GatewayTurnDoneEvent(session_id="s", response_text="pong"),
                ],
            )
            bridge = TelegramGatewayBridge(
                settings=_settings(
                    telegram_allowed_user_id=123,
                    telegram_temp_dir=Path(tmp),
                ),
                telegram_client=telegram,
                gateway_client=gateway,
            )

            await bridge.poll_once(offset=None)
            await bridge.wait_for_chat_idle(123)

            self.assertEqual(len(telegram.downloaded_files), 1)
            downloaded_path = telegram.downloaded_files[0].destination_path
            self.assertTrue(downloaded_path.exists())
            self.assertEqual(downloaded_path.read_bytes(), b"%PDF-1.4")
            self.assertEqual(gateway.calls[0][0], "tg_123")
            self.assertIn("filename: report.pdf", gateway.calls[0][1])
            self.assertIn("telegram_media_type: document", gateway.calls[0][1])
            self.assertIn("mime_type: application/pdf", gateway.calls[0][1])
            self.assertIn("size_bytes: 42", gateway.calls[0][1])
            self.assertIn(f"local_path: {downloaded_path}", gateway.calls[0][1])
            self.assertIn("caption: please review", gateway.calls[0][1])

    async def test_poll_once_ignores_private_message_from_unauthorized_user(self) -> None:
        updates = [
            {
                "update_id": 10,
                "message": {
                    "message_id": 100,
                    "from": {"id": 123, "is_bot": False},
                    "chat": {"id": 123, "type": "private"},
                    "text": "hello",
                },
            }
        ]
        telegram = _FakeTelegramClient(updates=updates)
        gateway = _FakeGatewayClient(
            events=[
                GatewayMessageEvent(session_id="s", text="pong"),
                GatewayTurnDoneEvent(session_id="s", response_text="pong"),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(telegram_allowed_user_id=999),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        with self.assertLogs("ui.telegram.bot", level="WARNING") as captured_logs:
            next_offset = await bridge.poll_once(offset=None)

        self.assertEqual(next_offset, 11)
        self.assertEqual(gateway.calls, [])
        self.assertEqual(telegram.sent_messages, [])
        self.assertEqual(
            captured_logs.output,
            ["WARNING:ui.telegram.bot:Ignoring unauthorized Telegram private message."],
        )

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
            events=[
                GatewayMessageEvent(session_id="s", text="pong"),
                GatewayTurnDoneEvent(session_id="s", response_text="pong"),
            ],
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

    async def test_handle_approval_callback_submits_decision_and_updates_message(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(approval_resolved=True)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_approval_callback(
            IncomingTelegramApprovalCallback(
                update_id=1,
                callback_query_id="callback_1",
                chat_id=777,
                message_id=42,
                sender_user_id=777,
                approval_id="approval_1",
                approved=True,
                message_text="Approval required\nsummary: Install a CLI.",
            )
        )

        self.assertEqual(gateway.approval_calls, [("tg_777", "approval_1", True)])
        self.assertEqual(len(telegram.answered_callbacks), 1)
        self.assertEqual(telegram.answered_callbacks[0].text, "Approved")
        self.assertEqual(len(telegram.edited_messages), 1)
        self.assertIn("Status: Approved", telegram.edited_messages[0].text)

    async def test_handle_approval_callback_stale_request_is_silent(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(approval_resolved=False)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_approval_callback(
            IncomingTelegramApprovalCallback(
                update_id=1,
                callback_query_id="callback_1",
                chat_id=777,
                message_id=42,
                sender_user_id=777,
                approval_id="approval_1",
                approved=True,
                message_text="Approval required\nsummary: Install a CLI.",
            )
        )

        self.assertEqual(gateway.approval_calls, [("tg_777", "approval_1", True)])
        self.assertEqual(len(telegram.answered_callbacks), 1)
        self.assertIsNone(telegram.answered_callbacks[0].text)
        self.assertFalse(telegram.answered_callbacks[0].show_alert)
        self.assertEqual(telegram.sent_messages, [])
        self.assertEqual(telegram.edited_messages, [])

    async def test_handle_message_streams_single_official_message_via_edits(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="stream-"),
                GatewayDeltaEvent(session_id="session", delta="reply"),
                GatewayMessageEvent(session_id="session", text="stream-reply"),
                GatewayTurnDoneEvent(session_id="session", response_text="stream-reply"),
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

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["stream-"],
        )
        self.assertEqual(
            [message.parse_mode for message in telegram.sent_messages],
            ["HTML"],
        )
        self.assertEqual(
            [message.text for message in telegram.edited_messages],
            ["stream-reply"],
        )
        self.assertEqual(
            [message.parse_mode for message in telegram.edited_messages],
            ["HTML"],
        )
        self.assertEqual(telegram.sent_drafts, [])

    async def test_handle_message_streams_drafts_when_draft_transport_enabled(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="stream-"),
                GatewayDeltaEvent(session_id="session", delta="reply"),
                GatewayMessageEvent(session_id="session", text="stream-reply"),
                GatewayTurnDoneEvent(session_id="session", response_text="stream-reply"),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_transport="draft"),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(
            [draft.draft.text for draft in telegram.sent_drafts],
            ["stream-", "stream-reply"],
        )
        self.assertEqual(
            [draft.parse_mode for draft in telegram.sent_drafts],
            ["HTML", "HTML"],
        )
        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["stream-reply"],
        )

    async def test_handle_message_suppresses_blank_final_turn(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayTurnDoneEvent(session_id="session", response_text=""),
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

        self.assertEqual(telegram.sent_messages, [])
        self.assertEqual(telegram.sent_drafts, [])

    async def test_handle_message_preserves_multiple_assistant_segments(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="Working on it."),
                GatewayMessageEvent(session_id="session", text="Working on it."),
                GatewayDeltaEvent(session_id="session", delta="Done."),
                GatewayMessageEvent(session_id="session", text="Done."),
                GatewayTurnDoneEvent(session_id="session", response_text="Done."),
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

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["Working on it.", "Done."],
        )
        self.assertEqual(telegram.edited_messages, [])
        self.assertEqual(telegram.sent_drafts, [])

    async def test_handle_message_emits_normalized_tool_notice(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="Working on it."),
                GatewayMessageEvent(session_id="session", text="Working on it."),
                GatewayToolCallEvent(session_id="session", tool_names=("bash",)),
                GatewayMessageEvent(session_id="session", text="Done."),
                GatewayTurnDoneEvent(session_id="session", response_text="Done."),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_chunk_min_chars=999),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["Working on it.", "🔧 <b>Jarvis</b> used <b>bash</b> tool.", "Done."],
        )

    async def test_handle_message_preserves_underscores_in_tool_notice(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayToolCallEvent(
                    session_id="session",
                    tool_names=("generate_edit_image",),
                ),
                GatewayMessageEvent(session_id="session", text="Done."),
                GatewayTurnDoneEvent(session_id="session", response_text="Done."),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_chunk_min_chars=999),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            [
                "🔧 <b>Jarvis</b> used <b>generate_edit_image</b> tool.",
                "Done.",
            ],
        )
        self.assertEqual(
            [message.parse_mode for message in telegram.sent_messages],
            ["HTML", "HTML"],
        )

    async def test_handle_message_collapses_only_consecutive_duplicate_tool_notices(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayToolCallEvent(
                    session_id="session",
                    tool_names=(
                        "bash",
                        "bash",
                        "web_fetch",
                        "web_fetch",
                        "bash",
                    ),
                ),
                GatewayMessageEvent(session_id="session", text="Done."),
                GatewayTurnDoneEvent(session_id="session", response_text="Done."),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_chunk_min_chars=999),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            [
                "🔧 <b>Jarvis</b> used <b>bash</b> tool.",
                "🔧 <b>Jarvis</b> used <b>web_fetch</b> tool.",
                "🔧 <b>Jarvis</b> used <b>bash</b> tool.",
                "Done.",
            ],
        )

    async def test_handle_message_flushes_pending_stream_text_before_tool_notice(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="Working on it."),
                GatewayToolCallEvent(session_id="session", tool_names=("bash",)),
                GatewayMessageEvent(session_id="session", text="Working on it."),
                GatewayMessageEvent(session_id="session", text="Done."),
                GatewayTurnDoneEvent(session_id="session", response_text="Done."),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_chunk_min_chars=999),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["Working on it.", "🔧 <b>Jarvis</b> used <b>bash</b> tool.", "Done."],
        )

    async def test_handle_message_sends_approval_request_with_inline_keyboard(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayApprovalRequestEvent(
                    session_id="session",
                    approval_id="approval_1",
                    kind="bash_command",
                    summary="Install a CLI.",
                    details="I want to install a CLI for this task.",
                    command="curl https://example.com/install.sh | sh",
                    tool_name="bash",
                    inspection_url="https://example.com",
                ),
                GatewayTurnDoneEvent(
                    session_id="session",
                    response_text="",
                    interrupted=True,
                ),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_chunk_min_chars=999),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(len(telegram.sent_messages), 1)
        approval_message = telegram.sent_messages[0]
        self.assertEqual(approval_message.parse_mode, "HTML")
        self.assertEqual(
            approval_message.text,
            "<b>Jarvis requests approval</b>\n"
            "<b>summary:</b> Install a CLI.\n"
            "<b>details:</b> I want to install a CLI for this task.\n"
            "<b>tool_name:</b> bash\n"
            "<b>command:</b>\n"
            "<pre>curl https://example.com/install.sh | sh</pre>\n"
            "<b>inspect:</b> https://example.com",
        )
        self.assertEqual(
            approval_message.reply_markup,
            {
                "inline_keyboard": [
                    [
                        {
                            "text": "Approve",
                            "callback_data": "appr:1:approval_1",
                        },
                        {
                            "text": "Reject",
                            "callback_data": "appr:0:approval_1",
                        },
                    ]
                ]
            },
        )

    async def test_approval_status_edit_preserves_original_html_formatting(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayApprovalRequestEvent(
                    session_id="session",
                    approval_id="approval_1",
                    kind="bash_command",
                    summary="Install a CLI.",
                    details="I want to install a CLI for this task.",
                    command="curl https://example.com/install.sh | sh",
                    tool_name="bash",
                    inspection_url="https://example.com",
                ),
                GatewayTurnDoneEvent(
                    session_id="session",
                    response_text="",
                    interrupted=True,
                ),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_chunk_min_chars=999),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        await bridge.handle_approval_callback(
            IncomingTelegramApprovalCallback(
                update_id=2,
                callback_query_id="callback_1",
                chat_id=777,
                message_id=1,
                sender_user_id=777,
                approval_id="approval_1",
                approved=True,
                message_text="Approval required\nsummary: Install a CLI.",
            )
        )

        self.assertEqual(gateway.approval_calls, [("tg_777", "approval_1", True)])
        self.assertEqual(len(telegram.edited_messages), 1)
        self.assertEqual(telegram.edited_messages[0].parse_mode, "HTML")
        self.assertEqual(
            telegram.edited_messages[0].text,
            "<b>Jarvis requests approval</b>\n"
            "<b>summary:</b> Install a CLI.\n"
            "<b>details:</b> I want to install a CLI for this task.\n"
            "<b>tool_name:</b> bash\n"
            "<b>command:</b>\n"
            "<pre>curl https://example.com/install.sh | sh</pre>\n"
            "<b>inspect:</b> https://example.com\n\n"
            "<b>Status:</b> Approved",
        )

    async def test_handle_message_preserves_tool_notice_when_tool_event_arrives_first(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayToolCallEvent(session_id="session", tool_names=("bash",)),
                GatewayDeltaEvent(session_id="session", delta="Working on it."),
                GatewayMessageEvent(session_id="session", text="Working on it."),
                GatewayTurnDoneEvent(session_id="session", response_text="Working on it."),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_chunk_min_chars=999),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["🔧 <b>Jarvis</b> used <b>bash</b> tool.", "Working on it."],
        )

    async def test_handle_message_skips_whitespace_only_draft_payloads(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="\n\n"),
                GatewayMessageEvent(session_id="session", text="pong"),
                GatewayTurnDoneEvent(session_id="session", response_text="pong"),
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

        self.assertEqual(telegram.draft_attempts, 0)
        self.assertEqual(telegram.sent_drafts, [])
        self.assertEqual([message.text for message in telegram.sent_messages], ["pong"])

    async def test_dispatch_message_sends_typing_indicator_before_first_chunk(self) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )
        client_message_id = session.sent_messages[0][1]
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
            )
        )

        await asyncio.sleep(0.6)

        self.assertEqual(
            [(action.chat_id, action.action) for action in telegram.sent_chat_actions],
            [(777, "typing")],
        )

        await session.emit(
            GatewayDeltaEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                delta="hello",
            )
        )
        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                text="hello",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                response_text="hello",
            )
        )
        await bridge.wait_for_chat_idle(777)

        self.assertEqual([message.text for message in telegram.sent_messages], ["hello"])
        self.assertEqual(len(telegram.sent_chat_actions), 1)

    async def test_dispatch_message_resumes_typing_after_mid_turn_tool_notice(self) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )
        client_message_id = session.sent_messages[0][1]
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
            )
        )

        await asyncio.sleep(0.6)
        self.assertEqual(len(telegram.sent_chat_actions), 1)

        await session.emit(
            GatewayToolCallEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                tool_names=("bash",),
            )
        )
        await asyncio.sleep(0)
        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["🔧 <b>Jarvis</b> used <b>bash</b> tool."],
        )

        await asyncio.sleep(0.6)
        self.assertEqual(
            [(action.chat_id, action.action) for action in telegram.sent_chat_actions],
            [(777, "typing"), (777, "typing")],
        )

        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                text="done",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                response_text="done",
            )
        )
        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["🔧 <b>Jarvis</b> used <b>bash</b> tool.", "done"],
        )

    async def test_task_status_keeps_typing_after_user_turn_done(self) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_typing_indicator_interval_seconds=0.05),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )
        client_message_id = session.sent_messages[0][1]
        await session.emit(
            GatewayTaskStatusEvent(
                route_id="tg_777",
                active=True,
                reason="user_message_queued",
            )
        )
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
            )
        )

        await asyncio.sleep(0.08)
        self.assertGreaterEqual(len(telegram.sent_chat_actions), 1)

        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                response_text="",
                interrupted=True,
            )
        )
        await bridge.wait_for_chat_idle(777)

        actions_after_user_turn = len(telegram.sent_chat_actions)
        await asyncio.sleep(0.08)
        self.assertGreater(len(telegram.sent_chat_actions), actions_after_user_turn)

        await session.emit(
            GatewayTaskStatusEvent(
                route_id="tg_777",
                active=False,
                reason="turn_worker_idle",
            )
        )
        actions_after_task_done = len(telegram.sent_chat_actions)
        await asyncio.sleep(0.08)
        self.assertEqual(len(telegram.sent_chat_actions), actions_after_task_done)

    async def test_dispatch_message_repeats_tool_notice_when_same_tool_is_used_in_next_turn(
        self,
    ) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="first"),
        )
        first_client_message_id = session.sent_messages[0][1]
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
            )
        )
        await session.emit(
            GatewayToolCallEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                tool_names=("bash", "bash"),
            )
        )
        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                text="Done 1.",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                response_text="Done 1.",
            )
        )
        await bridge.wait_for_chat_idle(777)

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=2, chat_id=777, chat_type="private", text="second"),
        )
        second_client_message_id = session.sent_messages[1][1]
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
            )
        )
        await session.emit(
            GatewayToolCallEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                tool_names=("bash",),
            )
        )
        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                text="Done 2.",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                response_text="Done 2.",
            )
        )
        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            [
                "🔧 <b>Jarvis</b> used <b>bash</b> tool.",
                "Done 1.",
                "🔧 <b>Jarvis</b> used <b>bash</b> tool.",
                "Done 2.",
            ],
        )

    async def test_background_subagent_tool_notices_collapse_only_consecutive_duplicates(
        self,
    ) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )
        client_message_id = session.sent_messages[0][1]
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
            )
        )
        await session.emit(
            GatewayToolCallEvent(
                route_id="tg_777",
                session_id="sub_session",
                turn_id="sub_turn_1",
                agent_kind="subagent",
                agent_name="Friday",
                subagent_id="sub_1",
                tool_names=("bash", "bash", "web_fetch", "bash"),
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                response_text="",
                interrupted=True,
            )
        )
        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            [
                "🔧 <b>Friday</b> used <b>bash</b> tool.",
                "🔧 <b>Friday</b> used <b>web_fetch</b> tool.",
                "🔧 <b>Friday</b> used <b>bash</b> tool.",
            ],
        )

    async def test_handle_message_skips_whitespace_only_assistant_segment(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="  "),
                GatewayMessageEvent(session_id="session", text="  "),
                GatewayDeltaEvent(session_id="session", delta="done"),
                GatewayMessageEvent(session_id="session", text="done"),
                GatewayTurnDoneEvent(session_id="session", response_text="done"),
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

        self.assertEqual([message.text for message in telegram.sent_messages], ["done"])
        self.assertEqual(telegram.sent_drafts, [])

    async def test_handle_message_skips_invisible_unicode_only_assistant_segment(self) -> None:
        telegram = _FakeTelegramClient()
        invisible_text = "\u200b\u200d\ufeff"
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta=invisible_text),
                GatewayMessageEvent(session_id="session", text=invisible_text),
                GatewayDeltaEvent(session_id="session", delta="done"),
                GatewayMessageEvent(session_id="session", text="done"),
                GatewayTurnDoneEvent(session_id="session", response_text="done"),
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

        self.assertEqual([message.text for message in telegram.sent_messages], ["done"])
        self.assertEqual(telegram.sent_drafts, [])

    async def test_handle_message_falls_back_to_plain_text_for_empty_html_drafts(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayDeltaEvent(session_id="session", delta="> "),
                GatewayMessageEvent(session_id="session", text="pong"),
                GatewayTurnDoneEvent(session_id="session", response_text="pong"),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_transport="draft"),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(telegram.draft_attempts, 1)
        self.assertEqual(len(telegram.sent_drafts), 1)
        self.assertEqual(telegram.sent_drafts[0].draft.text, ">")
        self.assertIsNone(telegram.sent_drafts[0].parse_mode)
        self.assertEqual([message.text for message in telegram.sent_messages], ["pong"])

    async def test_handle_message_ignores_non_private_chat(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayMessageEvent(session_id="session", text="should-not-send"),
                GatewayTurnDoneEvent(session_id="session", response_text="should-not-send"),
            ],
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
            events=[
                GatewayMessageEvent(session_id="session", text="should-not-send"),
                GatewayTurnDoneEvent(session_id="session", response_text="should-not-send"),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(telegram_allowed_user_id=777),
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

    async def test_send_file_to_owner_uses_allowed_user_id(self) -> None:
        telegram = _FakeTelegramClient()
        bridge = TelegramGatewayBridge(
            settings=_settings(telegram_allowed_user_id=777),
            telegram_client=telegram,
            gateway_client=_FakeGatewayClient(),
        )

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "note.txt"
            file_path.write_text("hello", encoding="utf-8")

            await bridge.send_file_to_owner(
                file_path=file_path,
                caption="attached",
                filename="custom.txt",
            )

        self.assertEqual(len(telegram.sent_documents), 1)
        self.assertEqual(telegram.sent_documents[0].chat_id, 777)
        self.assertEqual(telegram.sent_documents[0].caption, "attached")
        self.assertEqual(telegram.sent_documents[0].filename, "custom.txt")

    async def test_long_final_text_is_split(self) -> None:
        long_text = "a" * 5000
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayMessageEvent(session_id="session", text=long_text),
                GatewayTurnDoneEvent(session_id="session", response_text=long_text),
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

        self.assertEqual(len(telegram.sent_messages), 2)
        self.assertEqual(len(telegram.sent_messages[0].text), 4096)
        self.assertEqual(len(telegram.sent_messages[1].text), 904)

    async def test_gateway_error_sends_runtime_error_message_for_active_turn(self) -> None:
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
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["❌ Error occurred. Try again."],
        )

    async def test_handle_message_does_not_send_placeholder_for_interrupted_turn(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayTurnDoneEvent(
                    session_id="session",
                    response_text="",
                    interrupted=True,
                )
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

        self.assertEqual(telegram.sent_messages, [])

    async def test_dispatch_message_routes_stop_outside_active_turn_queue(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _BlockingGatewayClient()
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="hi"),
        )
        await gateway.stream_started.wait()

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=2, chat_id=777, chat_type="private", text="/stop"),
        )
        gateway.release_turn.set()
        await bridge.wait_for_chat_idle(777)

        self.assertEqual(gateway.calls, [("tg_777", "hi")])
        self.assertEqual(gateway.stop_calls, ["tg_777"])
        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["⚙️ <b>System:</b> Stop requested. I will stop after the current step."],
        )
        self.assertEqual(
            [message.parse_mode for message in telegram.sent_messages],
            ["HTML"],
        )

    async def test_dispatch_message_submits_second_message_immediately_while_first_turn_is_active(
        self,
    ) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        first = IncomingTextMessage(
            update_id=1,
            chat_id=777,
            chat_type="private",
            text="first",
        )
        second = IncomingTextMessage(
            update_id=2,
            chat_id=777,
            chat_type="private",
            text="second",
        )

        await bridge.dispatch_message(first)
        await bridge.dispatch_message(second)

        self.assertEqual(gateway.connected_routes, ["tg_777"])
        self.assertEqual(
            [text for text, _client_message_id in session.sent_messages],
            ["first", "second"],
        )
        first_client_message_id = session.sent_messages[0][1]
        second_client_message_id = session.sent_messages[1][1]

        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="partial",
                interrupted=True,
                interruption_reason="superseded_by_user_message",
            )
        )
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
            )
        )
        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                text="handled second",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="handled second",
            )
        )

        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["handled second"],
        )

    async def test_stop_command_suppresses_in_flight_route_events_until_next_user_turn(self) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="first"),
        )
        first_client_message_id = session.sent_messages[0][1]

        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
            )
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=2, chat_id=777, chat_type="private", text="/stop"),
        )

        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                text="suppressed reply",
            )
        )
        await session.emit(
            GatewayToolCallEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                tool_names=("bash",),
            )
        )
        await session.emit(
            GatewaySystemNoticeEvent(
                route_id="tg_777",
                session_id="session_1",
                agent_kind="subagent",
                agent_name="Ultron",
                subagent_id="sub_1",
                notice_kind="subagent_completed",
                text="completed.",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="suppressed reply",
                interrupted=True,
                interruption_reason="user_stop",
            )
        )

        await bridge.wait_for_chat_idle(777)

        self.assertTrue(session.stop_requested)
        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["⚙️ <b>System:</b> Stop requested. I will stop after the current step."],
        )

    async def test_stop_command_resumes_output_only_after_next_user_turn_event(self) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="first"),
        )
        first_client_message_id = session.sent_messages[0][1]
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
            )
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=2, chat_id=777, chat_type="private", text="/stop"),
        )
        await bridge.dispatch_message(
            IncomingTextMessage(update_id=3, chat_id=777, chat_type="private", text="resume"),
        )
        second_client_message_id = session.sent_messages[1][1]

        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                text="stale after stop",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="stale after stop",
                interrupted=True,
                interruption_reason="user_stop",
            )
        )
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
            )
        )
        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                text="handled second",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_2",
                turn_kind="user",
                client_message_id=second_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="handled second",
            )
        )

        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            [
                "⚙️ <b>System:</b> Stop requested. I will stop after the current step.",
                "handled second",
            ],
        )

    async def test_stop_command_resumes_new_command_without_turn_started_event(self) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="first"),
        )
        first_client_message_id = session.sent_messages[0][1]
        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
            )
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=2, chat_id=777, chat_type="private", text="/stop"),
        )
        await bridge.dispatch_message(
            IncomingTextMessage(update_id=3, chat_id=777, chat_type="private", text="/new"),
        )
        new_client_message_id = session.sent_messages[1][1]

        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=first_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="",
                interrupted=True,
                interruption_reason="user_stop",
            )
        )
        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session_2",
                turn_kind="user",
                client_message_id=new_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                text="Started a new session.",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_2",
                turn_kind="user",
                client_message_id=new_client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="Started a new session.",
                command="/new",
            )
        )

        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            [
                "⚙️ <b>System:</b> Stop requested. I will stop after the current step.",
                "⚙️ <b>System:</b> Started a new session.",
            ],
        )

    async def test_dispatch_message_submits_mid_turn_file_message_immediately(self) -> None:
        telegram = _FakeTelegramClient(
            remote_files={
                "file_1": TelegramRemoteFile(
                    file_id="file_1",
                    file_path="documents/spec.txt",
                    file_unique_id="uniq_1",
                    file_size=12,
                )
            },
            download_payloads={"documents/spec.txt": b"hello world\n"},
        )
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(
                update_id=1,
                chat_id=777,
                chat_type="private",
                text="first",
            )
        )
        await bridge.dispatch_message(
            IncomingTextMessage(
                update_id=2,
                chat_id=777,
                chat_type="private",
                text="please inspect",
                file_attachment=IncomingTelegramFile(
                    telegram_media_type="document",
                    file_id="file_1",
                    file_unique_id="uniq_1",
                    original_file_name="spec.txt",
                    mime_type="text/plain",
                    size_bytes=12,
                ),
            )
        )

        self.assertEqual(gateway.connected_routes, ["tg_777"])
        self.assertEqual(len(session.sent_messages), 2)
        self.assertEqual(session.sent_messages[0][0], "first")
        file_turn_text = session.sent_messages[1][0]
        self.assertIn("User sent a Telegram file.", file_turn_text)
        self.assertIn("filename: spec.txt", file_turn_text)
        self.assertIn("telegram_media_type: document", file_turn_text)
        self.assertIn("caption: please inspect", file_turn_text)
        self.assertEqual(
            [download.remote_file_path for download in telegram.downloaded_files],
            ["documents/spec.txt"],
        )

    async def test_new_command_sends_telegram_only_session_notice(self) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(
                update_id=1,
                chat_id=777,
                chat_type="private",
                text="/new",
            )
        )

        self.assertEqual(len(session.sent_messages), 1)
        client_message_id = session.sent_messages[0][1]

        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id=None,
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                text="Started a new session.",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id=None,
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="Started a new session.",
                command="/new",
            )
        )

        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["⚙️ <b>System:</b> Started a new session."],
        )
        self.assertEqual(
            [message.parse_mode for message in telegram.sent_messages],
            ["HTML"],
        )

    async def test_new_command_with_body_keeps_notice_and_response_separate(self) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(
                update_id=1,
                chat_id=777,
                chat_type="private",
                text="/new continue here",
            )
        )

        self.assertEqual(len(session.sent_messages), 1)
        client_message_id = session.sent_messages[0][1]

        await session.emit(
            GatewayTurnStartedEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
            )
        )
        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                text="continuing in the new session",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_id="turn_1",
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="continuing in the new session",
                command="/new",
            )
        )

        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            [
                "⚙️ <b>System:</b> Started a new session.",
                "continuing in the new session",
            ],
        )

    async def test_compact_command_sends_local_notices_and_suppresses_duplicate_plain_text(
        self,
    ) -> None:
        telegram = _FakeTelegramClient()
        session = _PersistentFakeRouteSession()
        gateway = _PersistentFakeGatewayClient(session)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.dispatch_message(
            IncomingTextMessage(
                update_id=1,
                chat_id=777,
                chat_type="private",
                text="/compact",
            )
        )

        self.assertEqual(len(session.sent_messages), 1)
        client_message_id = session.sent_messages[0][1]

        await session.emit(
            GatewayLocalNoticeEvent(
                route_id="tg_777",
                session_id="session_1",
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                notice_kind="compaction_started",
                text="Compacting...",
            )
        )
        await session.emit(
            GatewayLocalNoticeEvent(
                route_id="tg_777",
                session_id="session_2",
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                notice_kind="compaction_completed",
                text="Context compacted into a new session.",
            )
        )
        await session.emit(
            GatewayMessageEvent(
                route_id="tg_777",
                session_id="session_2",
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                text="Context compacted into a new session.",
            )
        )
        await session.emit(
            GatewayTurnDoneEvent(
                route_id="tg_777",
                session_id="session_2",
                turn_kind="user",
                client_message_id=client_message_id,
                agent_kind="main",
                agent_name="Jarvis",
                response_text="Context compacted into a new session.",
                command="/compact",
            )
        )

        await bridge.wait_for_chat_idle(777)

        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            [
                "⚙️ <b>System:</b> Compacting...",
                "⚙️ <b>System:</b> Context compacted into a new session.",
            ],
        )
        self.assertEqual(
            [message.parse_mode for message in telegram.sent_messages],
            ["HTML", "HTML"],
        )

    async def test_stop_command_gateway_error_is_suppressed(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            stop_error=GatewayBridgeError(
                code="gateway_unavailable",
                message="Could not communicate with the gateway websocket.",
            )
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="/stop"),
        )

        self.assertEqual(gateway.stop_calls, ["tg_777"])
        self.assertEqual(telegram.sent_messages, [])

    async def test_stop_command_without_active_turn_is_silent(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(stop_requested=False)
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="/stop"),
        )

        self.assertEqual(gateway.stop_calls, ["tg_777"])
        self.assertEqual(telegram.sent_messages, [])

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
                GatewayMessageEvent(session_id="session", text="stream-reply"),
                GatewayTurnDoneEvent(session_id="session", response_text="stream-reply"),
            ],
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_transport="draft"),
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
        self.assertEqual(
            [message.text for message in telegram.sent_messages],
            ["stream-reply", "stream-reply"],
        )

    async def test_background_subagent_system_notice_is_sent_to_chat(self) -> None:
        telegram = _FakeTelegramClient()
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=_FakeGatewayClient(),
        )

        await bridge._handle_background_route_event(
            chat_id=777,
            event=GatewaySystemNoticeEvent(
                route_id="route_1",
                session_id="session_1",
                agent_kind="subagent",
                agent_name="Ultron",
                subagent_id="sub_1",
                notice_kind="subagent_completed",
                text="completed.",
            ),
        )

        self.assertEqual(len(telegram.sent_messages), 1)
        self.assertEqual(telegram.sent_messages[0].parse_mode, "HTML")
        self.assertEqual(
            telegram.sent_messages[0].text,
            "⚙️ <b>System:</b> <b>Ultron</b> completed.",
        )

    async def test_background_local_notice_is_sent_to_chat(self) -> None:
        telegram = _FakeTelegramClient()
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=_FakeGatewayClient(),
        )

        await bridge._handle_background_route_event(
            chat_id=777,
            event=GatewayLocalNoticeEvent(
                route_id="route_1",
                session_id="session_1",
                turn_kind="runtime",
                agent_kind="main",
                agent_name="Jarvis",
                notice_kind="compaction_started",
                text="Compacting...",
            ),
        )

        self.assertEqual(len(telegram.sent_messages), 1)
        self.assertEqual(telegram.sent_messages[0].parse_mode, "HTML")
        self.assertEqual(
            telegram.sent_messages[0].text,
            "⚙️ <b>System:</b> Compacting...",
        )

    async def test_background_main_turn_done_is_sent_to_chat(self) -> None:
        telegram = _FakeTelegramClient()
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=_FakeGatewayClient(),
        )

        await bridge._handle_background_route_event(
            chat_id=777,
            event=GatewayTurnDoneEvent(
                route_id="route_1",
                session_id="session_1",
                agent_kind="main",
                agent_name="Jarvis",
                response_text="Ultron finished. I verified the result and cleaned it up.",
            ),
        )

        self.assertEqual(len(telegram.sent_messages), 1)
        self.assertEqual(
            telegram.sent_messages[0].text,
            "Ultron finished. I verified the result and cleaned it up.",
        )

    async def test_background_gateway_error_is_suppressed(self) -> None:
        telegram = _FakeTelegramClient()
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=_FakeGatewayClient(),
        )

        await bridge._handle_background_route_event(
            chat_id=777,
            event=GatewayErrorEvent(
                route_id="route_1",
                session_id="session_1",
                agent_kind="main",
                agent_name="Jarvis",
                code="internal_error",
                message="gateway failed",
            ),
        )

        self.assertEqual(telegram.sent_messages, [])

    async def test_aclose_tolerates_worker_tasks_removing_themselves_from_tracking_dicts(self) -> None:
        telegram = _FakeTelegramClient()
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=_FakeGatewayClient(),
        )
        fake_route_session = _FakeClosableRouteSession()

        async def chat_worker() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                bridge._chat_workers.pop(777, None)

        async def route_worker() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                bridge._route_sessions.pop(777, None)
                await fake_route_session.aclose()

        bridge._chat_workers[777] = asyncio.create_task(chat_worker())
        bridge._route_sessions[777] = SimpleNamespace(
            session=fake_route_session,
            event_task=asyncio.create_task(route_worker()),
        )

        await bridge.aclose()

        self.assertEqual(bridge._chat_workers, {})
        self.assertEqual(bridge._route_sessions, {})
        self.assertTrue(fake_route_session.closed)
        self.assertTrue(telegram.closed)

    async def test_final_message_renders_markdown_as_html(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            events=[
                GatewayMessageEvent(
                    session_id="session",
                    text="**bold** and `code` with [link](https://example.com)",
                ),
                GatewayTurnDoneEvent(
                    session_id="session",
                    response_text="**bold** and `code` with [link](https://example.com)",
                ),
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
            events=[
                GatewayMessageEvent(session_id="session", text="**bold**"),
                GatewayTurnDoneEvent(session_id="session", response_text="**bold**"),
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

        self.assertEqual(telegram.message_attempts, 2)
        self.assertEqual(len(telegram.sent_messages), 1)
        self.assertEqual(telegram.sent_messages[0].text, "**bold**")
        self.assertIsNone(telegram.sent_messages[0].parse_mode)

    async def test_route_event_worker_logs_telegram_delivery_errors_without_relabeling_them(
        self,
    ) -> None:
        telegram = _FakeTelegramClient(
            message_errors=[
                TelegramAPIError(
                    code="telegram_http_error",
                    message=(
                        "Telegram request failed for method 'sendMessage': "
                        "ReadTimeout: timed out"
                    ),
                )
            ]
        )
        session = _PersistentFakeRouteSession()
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=_PersistentFakeGatewayClient(session),
        )

        await bridge.dispatch_message(
            IncomingTextMessage(update_id=1, chat_id=777, chat_type="private", text="/new"),
        )
        client_message_id = session.sent_messages[0][1]

        with self.assertLogs("ui.telegram.bot", level="ERROR") as captured_logs:
            await session.emit(
                GatewayTurnStartedEvent(
                    route_id="tg_777",
                    session_id="session_1",
                    turn_id="turn_1",
                    turn_kind="user",
                    client_message_id=client_message_id,
                    agent_kind="main",
                    agent_name="Jarvis",
                )
            )
            await bridge.wait_for_chat_idle(777)

        self.assertEqual(telegram.sent_messages, [])
        self.assertNotIn(777, bridge._active_turn_by_chat)
        self.assertNotIn(777, bridge._submitted_turns_by_chat)
        self.assertTrue(
            any(
                "Telegram route event delivery failed for chat 777 "
                "(code=telegram_http_error, message=Telegram request failed for method "
                "'sendMessage': ReadTimeout: timed out)."
                in line
                for line in captured_logs.output
            )
        )
        self.assertFalse(
            any("gateway_unavailable" in line for line in captured_logs.output)
        )

    async def test_approval_callback_gateway_error_is_suppressed(self) -> None:
        telegram = _FakeTelegramClient()
        gateway = _FakeGatewayClient(
            approval_error=GatewayBridgeError(
                code="gateway_unavailable",
                message="Could not communicate with the gateway websocket.",
            )
        )
        bridge = TelegramGatewayBridge(
            settings=_settings(),
            telegram_client=telegram,
            gateway_client=gateway,
        )

        await bridge.handle_approval_callback(
            IncomingTelegramApprovalCallback(
                update_id=1,
                callback_query_id="callback_1",
                chat_id=777,
                message_id=42,
                sender_user_id=777,
                approval_id="approval_1",
                approved=True,
                message_text="Approval required",
            )
        )

        self.assertEqual(gateway.approval_calls, [("tg_777", "approval_1", True)])
        self.assertEqual(len(telegram.answered_callbacks), 1)
        self.assertIsNone(telegram.answered_callbacks[0].text)
        self.assertFalse(telegram.answered_callbacks[0].show_alert)
        self.assertEqual(telegram.sent_messages, [])


class TelegramBotHelpersTests(unittest.TestCase):
    def test_next_stream_chunk_waits_for_idle_flush(self) -> None:
        self.assertIsNone(
            _next_stream_chunk(
                current_text="hello world, this is enough text",
                published_text="",
                segment_started_monotonic=10.0,
                last_publish_monotonic=0.0,
                min_chars=10,
                max_chars=100,
                idle_flush_seconds=1.0,
                now_monotonic=10.5,
            )
        )
        self.assertEqual(
            _next_stream_chunk(
                current_text="hello world, this is enough text",
                published_text="",
                segment_started_monotonic=10.0,
                last_publish_monotonic=0.0,
                min_chars=10,
                max_chars=100,
                idle_flush_seconds=1.0,
                now_monotonic=11.0,
            ),
            "hello world, this is enough text",
        )

    def test_next_stream_chunk_prefers_paragraph_boundary(self) -> None:
        self.assertEqual(
            _next_stream_chunk(
                current_text="First paragraph.\n\nSecond paragraph keeps going.",
                published_text="",
                segment_started_monotonic=5.0,
                last_publish_monotonic=0.0,
                min_chars=10,
                max_chars=80,
                idle_flush_seconds=0.0,
                now_monotonic=5.0,
            ),
            "First paragraph.\n\n",
        )

    def test_next_stream_chunk_avoids_open_fenced_code_blocks(self) -> None:
        self.assertIsNone(
            _next_stream_chunk(
                current_text="```python\nprint('hello')\n",
                published_text="",
                segment_started_monotonic=1.0,
                last_publish_monotonic=0.0,
                min_chars=5,
                max_chars=80,
                idle_flush_seconds=0.0,
                now_monotonic=1.0,
            )
        )

    def test_draft_backoff_increases_interval_and_success_decays_it(self) -> None:
        bridge = TelegramGatewayBridge(
            settings=_settings(stream_chunk_idle_flush_seconds=1.0),
            telegram_client=_FakeTelegramClient(),
            gateway_client=_FakeGatewayClient(),
        )

        bridge._record_draft_backoff(
            chat_id=777,
            exc=TelegramAPIError(
                code="telegram_api_error_429",
                message="Too Many Requests",
                retry_after_seconds=3,
            ),
        )
        backed_off = bridge._draft_min_interval_for_chat(777)
        self.assertGreater(backed_off, 1.0)

        bridge._record_draft_success(chat_id=777)
        self.assertLess(bridge._draft_min_interval_for_chat(777), backed_off)
        self.assertGreaterEqual(bridge._draft_min_interval_for_chat(777), 1.0)

    def test_parse_incoming_text_message(self) -> None:
        parsed = parse_incoming_text_message(
            {
                "update_id": 12,
                "message": {
                    "from": {"id": 321, "is_bot": False},
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

    def test_parse_incoming_message_extracts_document_metadata(self) -> None:
        parsed = parse_incoming_message(
            {
                "update_id": 14,
                "message": {
                    "from": {"id": 321, "is_bot": False},
                    "chat": {"id": 321, "type": "private"},
                    "caption": "review this",
                    "document": {
                        "file_id": "file-1",
                        "file_unique_id": "unique-1",
                        "file_name": "report.pdf",
                        "mime_type": "application/pdf",
                        "file_size": 42,
                    },
                },
            }
        )
        self.assertIsNotNone(parsed)
        if parsed is None:
            self.fail("Expected parsed incoming message.")
        self.assertEqual(parsed.text, "review this")
        self.assertEqual(
            parsed.file_attachment,
            IncomingTelegramFile(
                telegram_media_type="document",
                file_id="file-1",
                file_unique_id="unique-1",
                original_file_name="report.pdf",
                mime_type="application/pdf",
                size_bytes=42,
                extra_metadata=(),
            ),
        )

    def test_parse_incoming_message_rejects_sender_chat_updates(self) -> None:
        parsed = parse_incoming_message(
            {
                "update_id": 15,
                "message": {
                    "from": {"id": 321, "is_bot": False},
                    "sender_chat": {"id": -1000},
                    "chat": {"id": 321, "type": "private"},
                    "text": "hello",
                },
            }
        )
        self.assertIsNone(parsed)

    def test_parse_incoming_message_rejects_bot_sender(self) -> None:
        parsed = parse_incoming_message(
            {
                "update_id": 16,
                "message": {
                    "from": {"id": 321, "is_bot": True},
                    "chat": {"id": 321, "type": "private"},
                    "text": "hello",
                },
            }
        )
        self.assertIsNone(parsed)

    def test_parse_incoming_message_rejects_private_sender_chat_id_mismatch(self) -> None:
        parsed = parse_incoming_message(
            {
                "update_id": 17,
                "message": {
                    "from": {"id": 321, "is_bot": False},
                    "chat": {"id": 999, "type": "private"},
                    "text": "hello",
                },
            }
        )
        self.assertIsNone(parsed)

    def test_parse_incoming_approval_callback(self) -> None:
        parsed = parse_incoming_approval_callback(
            {
                "update_id": 18,
                "callback_query": {
                    "id": "callback_1",
                    "from": {"id": 777, "is_bot": False},
                    "data": "appr:0:approval_1",
                    "message": {
                        "message_id": 42,
                        "chat": {"id": 777, "type": "private"},
                        "text": "Approval required\nsummary: Install a CLI.",
                    },
                },
            }
        )

        self.assertIsNotNone(parsed)
        if parsed is None:
            self.fail("Expected parsed approval callback.")
        self.assertEqual(parsed.callback_query_id, "callback_1")
        self.assertEqual(parsed.chat_id, 777)
        self.assertEqual(parsed.message_id, 42)
        self.assertEqual(parsed.approval_id, "approval_1")
        self.assertFalse(parsed.approved)

    def test_clear_directory_contents_removes_files_and_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "note.txt").write_text("hello", encoding="utf-8")
            nested = root / "nested"
            nested.mkdir()
            (nested / "child.txt").write_text("world", encoding="utf-8")

            _clear_directory_contents(root)

            self.assertEqual(list(root.iterdir()), [])

    def test_route_id_for_negative_chat_id(self) -> None:
        self.assertEqual(route_id_for_chat(-123), "tg_n123")

    def test_chat_id_for_route_id(self) -> None:
        self.assertEqual(chat_id_for_route_id("tg_123"), 123)
        self.assertEqual(chat_id_for_route_id("tg_n123"), -123)
        self.assertIsNone(chat_id_for_route_id("ws_123"))
        self.assertIsNone(chat_id_for_route_id("tg_bad"))

    def test_split_telegram_message(self) -> None:
        chunks = split_telegram_message("abc", max_chars=2)
        self.assertEqual(chunks, ["ab", "c"])

    def test_split_telegram_message_returns_no_chunks_for_blank_text(self) -> None:
        self.assertEqual(split_telegram_message(" \n\t "), [])

    def test_split_telegram_message_returns_no_chunks_for_invisible_unicode(self) -> None:
        self.assertEqual(split_telegram_message("\u200b\u200d\ufeff"), [])
