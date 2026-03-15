"""Telegram bot runtime that bridges inbound messages to the gateway websocket."""

from __future__ import annotations

import asyncio
import html
import logging
import re
import shutil
import time
import unicodedata
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from .api import (
    DraftMessage,
    TelegramAPIError,
    TelegramBotAPIClient,
    TelegramRemoteFile,
)
from .config import UIConfigurationError, UISettings
from .formatting import render_markdown_to_telegram_html
from .gateway_client import (
    GatewayApprovalRequestEvent,
    GatewayBridgeError,
    GatewayDeltaEvent,
    GatewayMessageEvent,
    GatewayToolCallEvent,
    GatewayTurnDoneEvent,
    GatewayWebSocketClient,
)

LOGGER = logging.getLogger(__name__)
_GENERIC_ERROR_REPLY = "I hit an internal error while processing that message."
_FILE_DOWNLOAD_ERROR_REPLY = "I couldn't download that file from Telegram."
_FILENAME_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_APPROVAL_CALLBACK_PREFIX = "appr"
_APPROVAL_COMMAND_PREVIEW_CHARS = 1500


class GatewayClientLike(Protocol):
    async def stream_turn(self, *, route_id: str, user_text: str) -> AsyncIterator[Any]:
        """Streams one gateway turn."""

    async def request_stop(self, *, route_id: str) -> bool:
        """Requests cooperative stop for the active turn on the given route."""

    async def submit_approval(
        self,
        *,
        route_id: str,
        approval_id: str,
        approved: bool,
    ) -> bool:
        """Resolves one pending approval request for the active route."""


class TelegramClientLike(Protocol):
    async def get_me(self) -> dict[str, Any]:
        """Returns Telegram bot profile."""

    async def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Long-polls Telegram updates."""

    async def get_file(self, *, file_id: str) -> TelegramRemoteFile:
        """Resolves a Telegram file identifier to a downloadable file path."""

    async def download_file(
        self,
        *,
        remote_file_path: str,
        destination_path: str | Path,
    ) -> Path:
        """Downloads a Telegram file to a local destination path."""

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Sends a standard Telegram message."""

    async def edit_message_text(
        self,
        *,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Edits an existing Telegram message."""

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> bool:
        """Acknowledges a Telegram callback query."""

    async def send_message_draft(
        self,
        *,
        chat_id: int,
        draft: DraftMessage,
        parse_mode: str | None = None,
    ) -> bool:
        """Sends/updates Telegram draft message content."""

    async def send_document(
        self,
        *,
        chat_id: int,
        file_path: str | Path,
        caption: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        """Sends a local file to a Telegram chat as a document."""

    async def aclose(self) -> None:
        """Releases client resources."""


@dataclass(slots=True, frozen=True)
class IncomingTelegramFile:
    telegram_media_type: str
    file_id: str
    file_unique_id: str | None = None
    original_file_name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    extra_metadata: tuple[tuple[str, str], ...] = ()


@dataclass(slots=True, frozen=True)
class IncomingTelegramMessage:
    update_id: int
    chat_id: int
    chat_type: str
    text: str
    sender_user_id: int | None = None
    sender_is_bot: bool = False
    has_sender_chat: bool = False
    file_attachment: IncomingTelegramFile | None = None


IncomingTextMessage = IncomingTelegramMessage


@dataclass(slots=True, frozen=True)
class IncomingTelegramApprovalCallback:
    update_id: int
    callback_query_id: str
    chat_id: int
    message_id: int
    sender_user_id: int
    approval_id: str
    approved: bool
    message_text: str | None = None


class TelegramGatewayBridge:
    """Runs the Telegram->Gateway bridge loop."""

    def __init__(
        self,
        *,
        settings: UISettings | None = None,
        telegram_client: TelegramClientLike | None = None,
        gateway_client: GatewayClientLike | None = None,
    ) -> None:
        self._settings = settings or UISettings.from_env()
        self._telegram = telegram_client or TelegramBotAPIClient(
            token=self._settings.telegram_token,
            api_base_url=self._settings.telegram_api_base_url,
        )
        self._gateway = gateway_client or GatewayWebSocketClient(
            websocket_base_url=self._settings.gateway_ws_base_url,
            connect_timeout_seconds=self._settings.gateway_connect_timeout_seconds,
        )
        self._next_draft_ids: dict[int, int] = {}
        self._draft_retry_until_by_chat: dict[int, float] = {}
        self._draft_min_interval_by_chat: dict[int, float] = {}
        self._chat_queues: dict[int, asyncio.Queue[IncomingTelegramMessage]] = {}
        self._chat_workers: dict[int, asyncio.Task[None]] = {}
        self._approval_message_html_by_key: dict[tuple[int, int], str] = {}

    async def run_forever(self) -> None:
        bot_profile = await self._telegram.get_me()
        LOGGER.info(
            "Telegram bridge online for bot @%s",
            bot_profile.get("username", "unknown"),
        )
        LOGGER.info(
            "Telegram temp dir is %s",
            self._settings.telegram_temp_dir,
        )
        LOGGER.info(
            "Telegram bridge restricted to the configured owner.",
        )

        next_offset: int | None = None
        while True:
            try:
                next_offset = await self.poll_once(offset=next_offset)
            except asyncio.CancelledError:
                raise
            except TelegramAPIError:
                LOGGER.exception("Telegram API polling failed; retrying after backoff.")
                await asyncio.sleep(self._settings.poll_error_backoff_seconds)
            except Exception:
                LOGGER.exception(
                    "Unexpected error in polling loop; retrying after backoff."
                )
                await asyncio.sleep(self._settings.poll_error_backoff_seconds)

    async def poll_once(self, *, offset: int | None) -> int | None:
        updates = await self._telegram.get_updates(
            offset=offset,
            timeout_seconds=self._settings.telegram_poll_timeout_seconds,
            limit=self._settings.telegram_poll_limit,
        )
        next_offset = offset
        for update in updates:
            if isinstance(update, dict):
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    next_offset = max(next_offset or 0, update_id + 1)
            callback = parse_incoming_approval_callback(update)
            if callback is not None:
                await self.handle_approval_callback(callback)
                continue
            message = parse_incoming_message(update)
            if message is None:
                continue
            await self.dispatch_message(message)
        return next_offset

    async def handle_approval_callback(
        self,
        callback: IncomingTelegramApprovalCallback,
    ) -> None:
        if self._settings.telegram_allowed_user_id != callback.sender_user_id:
            LOGGER.warning("Ignoring unauthorized Telegram approval callback.")
            return

        route_id = route_id_for_chat(callback.chat_id)
        try:
            resolved = await self._gateway.submit_approval(
                route_id=route_id,
                approval_id=callback.approval_id,
                approved=callback.approved,
            )
        except GatewayBridgeError:
            LOGGER.exception("Failed to submit approval callback to the gateway.")
            await self._telegram.answer_callback_query(
                callback_query_id=callback.callback_query_id,
                text="Could not contact the gateway.",
                show_alert=True,
            )
            return

        if not resolved:
            await self._telegram.answer_callback_query(
                callback_query_id=callback.callback_query_id,
                text="This approval request is no longer pending.",
                show_alert=True,
            )
            return

        decision_text = "Approved" if callback.approved else "Rejected"
        await self._telegram.answer_callback_query(
            callback_query_id=callback.callback_query_id,
            text=decision_text,
        )
        message_key = (callback.chat_id, callback.message_id)
        stored_html = self._approval_message_html_by_key.pop(message_key, None)
        if stored_html is not None:
            updated_text = f"{stored_html}\n\n<b>Status:</b> {html.escape(decision_text)}"
            parse_mode = "HTML"
        else:
            base_text = callback.message_text or "Approval request"
            updated_text = f"{base_text}\n\nStatus: {decision_text}"
            parse_mode = None
        try:
            await self._telegram.edit_message_text(
                chat_id=callback.chat_id,
                message_id=callback.message_id,
                text=updated_text,
                parse_mode=parse_mode,
            )
        except TelegramAPIError:
            LOGGER.exception("Failed to edit Telegram approval message status.")

    async def dispatch_message(self, message: IncomingTelegramMessage) -> None:
        if message.chat_type != "private":
            return
        if not _is_authorized_private_message(
            message=message,
            allowed_user_id=self._settings.telegram_allowed_user_id,
        ):
            LOGGER.warning("Ignoring unauthorized Telegram private message.")
            return
        if message.file_attachment is None and _is_stop_command_text(message.text):
            await self._handle_stop_command(message)
            return

        queue = self._chat_queues.get(message.chat_id)
        if queue is None:
            queue = asyncio.Queue()
            self._chat_queues[message.chat_id] = queue
        await queue.put(message)

        worker = self._chat_workers.get(message.chat_id)
        if worker is None or worker.done():
            self._chat_workers[message.chat_id] = asyncio.create_task(
                self._chat_worker(message.chat_id),
                name=f"jarvis-telegram-chat-{message.chat_id}",
            )

    async def _chat_worker(self, chat_id: int) -> None:
        queue = self._chat_queues[chat_id]
        try:
            while True:
                message = await queue.get()
                try:
                    await self.handle_message(message)
                finally:
                    queue.task_done()
                if queue.empty():
                    return
        except asyncio.CancelledError:
            raise
        finally:
            worker = self._chat_workers.get(chat_id)
            current_task = asyncio.current_task()
            if worker is current_task:
                self._chat_workers.pop(chat_id, None)
            if queue.empty():
                self._chat_queues.pop(chat_id, None)

    async def handle_message(self, message: IncomingTelegramMessage) -> None:
        if message.chat_type != "private":
            return
        if not _is_authorized_private_message(
            message=message,
            allowed_user_id=self._settings.telegram_allowed_user_id,
        ):
            LOGGER.warning("Ignoring unauthorized Telegram private message.")
            return
        if message.file_attachment is None and _is_stop_command_text(message.text):
            await self._handle_stop_command(message)
            return

        user_text = message.text
        if message.file_attachment is not None:
            try:
                user_text = await self._build_file_turn_text(message)
            except TelegramAPIError:
                LOGGER.exception("Telegram file download failed.")
                await self._send_final_text(
                    chat_id=message.chat_id,
                    text=_FILE_DOWNLOAD_ERROR_REPLY,
                )
                return

        if not user_text:
            return

        route_id = route_id_for_chat(message.chat_id)
        current_draft_id = self._next_draft_id_for_chat(message.chat_id)

        accumulated_text = ""
        last_draft_text = ""
        last_draft_at = 0.0
        segment_started_at = time.monotonic()
        delivered_any_segment = False
        pending_finalized_text_for_dedup: str | None = None
        drafts_enabled = (
            time.monotonic()
            >= self._draft_retry_until_by_chat.get(message.chat_id, 0.0)
        )

        try:
            async for event in self._gateway.stream_turn(
                route_id=route_id,
                user_text=user_text,
            ):
                if isinstance(event, GatewayDeltaEvent):
                    if not event.delta:
                        continue
                    accumulated_text += event.delta
                    if not drafts_enabled:
                        continue
                    now = time.monotonic()
                    if _should_flush_draft(
                        current_text=accumulated_text,
                        last_sent_text=last_draft_text,
                        last_sent_monotonic=last_draft_at,
                        segment_started_monotonic=segment_started_at,
                        min_chars=self._settings.stream_draft_min_chars,
                        min_interval_seconds=self._draft_min_interval_for_chat(
                            message.chat_id
                        ),
                        now_monotonic=now,
                    ):
                        try:
                            await self._send_draft(
                                chat_id=message.chat_id,
                                draft_id=current_draft_id,
                                text=accumulated_text,
                            )
                            last_draft_text = accumulated_text
                            last_draft_at = now
                            self._record_draft_success(chat_id=message.chat_id)
                        except TelegramAPIError as exc:
                            drafts_enabled = False
                            self._record_draft_backoff(
                                chat_id=message.chat_id,
                                exc=exc,
                            )
                            if exc.retry_after_seconds is not None:
                                LOGGER.warning(
                                    "sendMessageDraft rate-limited; pausing drafts for %ss.",
                                    exc.retry_after_seconds,
                                )
                            else:
                                LOGGER.exception(
                                    "sendMessageDraft failed; continuing this turn without drafts."
                                )
                    continue

                if isinstance(event, GatewayMessageEvent):
                    final_text = _coalesce_visible_text(event.text, accumulated_text)
                    if (
                        final_text is not None
                        and final_text != pending_finalized_text_for_dedup
                    ):
                        await self._send_final_text(chat_id=message.chat_id, text=final_text)
                        delivered_any_segment = True
                    pending_finalized_text_for_dedup = None
                    accumulated_text = ""
                    last_draft_text = ""
                    last_draft_at = 0.0
                    segment_started_at = time.monotonic()
                    current_draft_id = self._next_draft_id_for_chat(message.chat_id)
                    continue

                if isinstance(event, GatewayToolCallEvent):
                    flushed_text = _coalesce_visible_text(accumulated_text)
                    if flushed_text is not None:
                        await self._send_final_text(
                            chat_id=message.chat_id,
                            text=flushed_text,
                        )
                        delivered_any_segment = True
                        pending_finalized_text_for_dedup = flushed_text
                    for tool_name in event.tool_names:
                        await self._send_html_message(
                            chat_id=message.chat_id,
                            html_text=_format_tool_usage_notice(tool_name),
                        )
                        delivered_any_segment = True
                    accumulated_text = ""
                    last_draft_text = ""
                    last_draft_at = 0.0
                    segment_started_at = time.monotonic()
                    current_draft_id = self._next_draft_id_for_chat(message.chat_id)
                    continue

                if isinstance(event, GatewayApprovalRequestEvent):
                    flushed_text = _coalesce_visible_text(accumulated_text)
                    if flushed_text is not None:
                        await self._send_final_text(
                            chat_id=message.chat_id,
                            text=flushed_text,
                        )
                        delivered_any_segment = True
                        pending_finalized_text_for_dedup = flushed_text
                    await self._send_approval_request_message(
                        chat_id=message.chat_id,
                        approval=event,
                    )
                    accumulated_text = ""
                    last_draft_text = ""
                    last_draft_at = 0.0
                    segment_started_at = time.monotonic()
                    current_draft_id = self._next_draft_id_for_chat(message.chat_id)
                    continue

                if isinstance(event, GatewayTurnDoneEvent):
                    if event.interrupted:
                        return
                    if not delivered_any_segment:
                        final_text = (
                            _coalesce_visible_text(
                                event.response_text,
                                accumulated_text,
                            )
                            or "(No response text.)"
                        )
                        await self._send_final_text(
                            chat_id=message.chat_id,
                            text=final_text,
                        )
                    return

            await self._send_final_text(
                chat_id=message.chat_id,
                text=_GENERIC_ERROR_REPLY,
            )
        except GatewayBridgeError as exc:
            LOGGER.exception("Gateway bridge failed.")
            await self._send_final_text(
                chat_id=message.chat_id,
                text=exc.message if exc.message else _GENERIC_ERROR_REPLY,
            )
        except TelegramAPIError:
            LOGGER.exception("Telegram API send failed.")
        except Exception:
            LOGGER.exception("Unexpected bridge error.")
            try:
                await self._send_final_text(
                    chat_id=message.chat_id,
                    text=_GENERIC_ERROR_REPLY,
                )
            except TelegramAPIError:
                LOGGER.exception("Failed to send fallback error text.")

    async def aclose(self) -> None:
        for worker in self._chat_workers.values():
            worker.cancel()
        for worker in self._chat_workers.values():
            with suppress(asyncio.CancelledError):
                await worker
        self._chat_workers.clear()
        self._chat_queues.clear()
        await self._telegram.aclose()

    async def wait_for_chat_idle(self, chat_id: int) -> None:
        queue = self._chat_queues.get(chat_id)
        if queue is not None:
            await queue.join()
        worker = self._chat_workers.get(chat_id)
        if worker is not None:
            await worker

    async def _handle_stop_command(self, message: IncomingTelegramMessage) -> None:
        route_id = route_id_for_chat(message.chat_id)
        try:
            stop_requested = await self._gateway.request_stop(route_id=route_id)
        except GatewayBridgeError as exc:
            LOGGER.exception("Gateway stop request failed.")
            await self._send_final_text(
                chat_id=message.chat_id,
                text=exc.message if exc.message else _GENERIC_ERROR_REPLY,
            )
            return

        if stop_requested:
            await self._send_final_text(
                chat_id=message.chat_id,
                text="Stop requested. I will stop after the current step.",
            )
            return
        await self._send_final_text(
            chat_id=message.chat_id,
            text="Nothing is currently running.",
        )

    async def send_file(
        self,
        *,
        chat_id: int,
        file_path: str | Path,
        caption: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        return await self._telegram.send_document(
            chat_id=chat_id,
            file_path=file_path,
            caption=caption,
            filename=filename,
        )

    async def send_file_to_owner(
        self,
        *,
        file_path: str | Path,
        caption: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        allowed_user_id = self._settings.telegram_allowed_user_id
        if allowed_user_id is None:
            raise UIConfigurationError(
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID must be set to send files to the Telegram owner."
            )
        return await self.send_file(
            chat_id=allowed_user_id,
            file_path=file_path,
            caption=caption,
            filename=filename,
        )

    async def _build_file_turn_text(
        self,
        message: IncomingTelegramMessage,
    ) -> str:
        attachment = message.file_attachment
        if attachment is None:
            return message.text

        remote_file = await self._telegram.get_file(file_id=attachment.file_id)
        destination_path = _build_download_destination(
            temp_dir=self._settings.telegram_temp_dir,
            chat_id=message.chat_id,
            update_id=message.update_id,
            attachment=attachment,
            remote_file=remote_file,
        )
        local_path = await self._telegram.download_file(
            remote_file_path=remote_file.file_path,
            destination_path=destination_path,
        )
        return _format_incoming_file_message(
            attachment=attachment,
            local_path=local_path,
            remote_file=remote_file,
            caption=message.text,
        )

    async def _send_draft(self, *, chat_id: int, draft_id: int, text: str) -> None:
        max_chars = self._settings.telegram_max_message_chars
        draft_text = text
        if len(draft_text) > max_chars:
            draft_text = f"{draft_text[: max_chars - 1]}…"
        plain_draft_text = _coalesce_visible_text(draft_text)
        if plain_draft_text is None:
            return

        rendered_text = render_markdown_to_telegram_html(plain_draft_text)
        if _has_visible_telegram_text(rendered_text):
            draft = DraftMessage(id=draft_id, text=rendered_text)
            try:
                await self._telegram.send_message_draft(
                    chat_id=chat_id,
                    draft=draft,
                    parse_mode="HTML",
                )
                return
            except TelegramAPIError as exc:
                if not _is_formatting_error(exc):
                    raise

        await self._telegram.send_message_draft(
            chat_id=chat_id,
            draft=DraftMessage(id=draft_id, text=plain_draft_text),
        )

    async def _send_final_text(self, *, chat_id: int, text: str) -> None:
        normalized_text = _coalesce_visible_text(text) or "(No response text.)"
        chunks = split_telegram_message(
            normalized_text,
            max_chars=self._settings.telegram_max_message_chars,
        )
        for chunk in chunks:
            await self._send_formatted_message(chat_id=chat_id, text=chunk)

    async def _send_formatted_message(self, *, chat_id: int, text: str) -> None:
        plain_text = text.strip()
        if not plain_text:
            return

        formatted_text = render_markdown_to_telegram_html(plain_text)
        if _has_visible_telegram_text(formatted_text):
            try:
                await self._telegram.send_message(
                    chat_id=chat_id,
                    text=formatted_text,
                    parse_mode="HTML",
                )
                return
            except TelegramAPIError as exc:
                if not _is_formatting_error(exc):
                    raise

        await self._telegram.send_message(chat_id=chat_id, text=plain_text)

    async def _send_html_message(self, *, chat_id: int, html_text: str) -> None:
        normalized_html = html_text.strip()
        if not normalized_html:
            return

        try:
            await self._telegram.send_message(
                chat_id=chat_id,
                text=normalized_html,
                parse_mode="HTML",
            )
            return
        except TelegramAPIError as exc:
            if not _is_formatting_error(exc):
                raise

        plain_text = _coalesce_visible_text(
            html.unescape(_HTML_TAG_PATTERN.sub("", normalized_html))
        )
        if plain_text is None:
            return
        await self._telegram.send_message(chat_id=chat_id, text=plain_text)

    async def _send_approval_request_message(
        self,
        *,
        chat_id: int,
        approval: GatewayApprovalRequestEvent,
    ) -> None:
        message_text = _format_approval_request_message(approval)
        reply_markup = {
            "inline_keyboard": [
                [
                    {
                        "text": "Approve",
                        "callback_data": _build_approval_callback_data(
                            approval_id=approval.approval_id,
                            approved=True,
                        ),
                    },
                    {
                        "text": "Reject",
                        "callback_data": _build_approval_callback_data(
                            approval_id=approval.approval_id,
                            approved=False,
                        ),
                    },
                ]
            ]
        }
        response = await self._telegram.send_message(
            chat_id=chat_id,
            text=message_text,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )
        message_id = response.get("message_id")
        if isinstance(message_id, int):
            self._approval_message_html_by_key[(chat_id, message_id)] = message_text

    def _record_draft_backoff(self, *, chat_id: int, exc: TelegramAPIError) -> None:
        if exc.retry_after_seconds is None:
            return
        base_interval = self._settings.stream_draft_min_interval_seconds
        current_interval = self._draft_min_interval_by_chat.get(chat_id, base_interval)
        backed_off_interval = max(
            base_interval,
            current_interval * 1.5,
            min(2.5, max(base_interval, exc.retry_after_seconds / 2)),
        )
        self._draft_min_interval_by_chat[chat_id] = backed_off_interval
        self._draft_retry_until_by_chat[chat_id] = (
            time.monotonic() + max(1, exc.retry_after_seconds)
        )

    def _record_draft_success(self, *, chat_id: int) -> None:
        base_interval = self._settings.stream_draft_min_interval_seconds
        current_interval = self._draft_min_interval_by_chat.get(chat_id)
        if current_interval is None or current_interval <= base_interval:
            self._draft_min_interval_by_chat.pop(chat_id, None)
            return
        next_interval = max(base_interval, current_interval * 0.9)
        if next_interval <= base_interval + 0.05:
            self._draft_min_interval_by_chat.pop(chat_id, None)
            return
        self._draft_min_interval_by_chat[chat_id] = next_interval

    def _draft_min_interval_for_chat(self, chat_id: int) -> float:
        return self._draft_min_interval_by_chat.get(
            chat_id,
            self._settings.stream_draft_min_interval_seconds,
        )

    def _next_draft_id_for_chat(self, chat_id: int) -> int:
        next_id = self._next_draft_ids.get(chat_id, 1)
        self._next_draft_ids[chat_id] = next_id + 1
        return next_id


def parse_incoming_message(update: dict[str, Any]) -> IncomingTelegramMessage | None:
    if not isinstance(update, dict):
        return None
    update_id = update.get("update_id")
    if not isinstance(update_id, int):
        return None

    message = update.get("message")
    if not isinstance(message, dict):
        return None

    text = ""
    if isinstance(message.get("text"), str):
        text = message["text"].strip()
    elif isinstance(message.get("caption"), str):
        text = message["caption"].strip()

    from_user = message.get("from")
    if not isinstance(from_user, dict):
        return None
    sender_user_id = from_user.get("id")
    sender_is_bot = from_user.get("is_bot")
    if not isinstance(sender_user_id, int):
        return None
    if not isinstance(sender_is_bot, bool):
        return None
    if sender_is_bot:
        return None
    if message.get("sender_chat") is not None:
        return None

    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_id = chat.get("id")
    chat_type = chat.get("type")
    if not isinstance(chat_id, int):
        return None
    if not isinstance(chat_type, str):
        return None
    if chat_type == "private" and chat_id != sender_user_id:
        return None

    file_attachment = _extract_file_attachment(message)
    if not text and file_attachment is None:
        return None

    return IncomingTelegramMessage(
        update_id=update_id,
        chat_id=chat_id,
        chat_type=chat_type,
        text=text,
        sender_user_id=sender_user_id,
        sender_is_bot=sender_is_bot,
        has_sender_chat=False,
        file_attachment=file_attachment,
    )


def parse_incoming_text_message(update: dict[str, Any]) -> IncomingTextMessage | None:
    return parse_incoming_message(update)


def parse_incoming_approval_callback(
    update: dict[str, Any],
) -> IncomingTelegramApprovalCallback | None:
    if not isinstance(update, dict):
        return None
    update_id = update.get("update_id")
    if not isinstance(update_id, int):
        return None

    callback_query = update.get("callback_query")
    if not isinstance(callback_query, dict):
        return None

    callback_query_id = callback_query.get("id")
    if not isinstance(callback_query_id, str) or not callback_query_id.strip():
        return None

    from_user = callback_query.get("from")
    if not isinstance(from_user, dict):
        return None
    sender_user_id = from_user.get("id")
    if not isinstance(sender_user_id, int):
        return None

    message = callback_query.get("message")
    if not isinstance(message, dict):
        return None
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_id = chat.get("id")
    if not isinstance(chat_id, int):
        return None
    message_id = message.get("message_id")
    if not isinstance(message_id, int):
        return None

    raw_callback_data = callback_query.get("data")
    if not isinstance(raw_callback_data, str):
        return None
    parsed = _parse_approval_callback_data(raw_callback_data)
    if parsed is None:
        return None
    approved, approval_id = parsed

    message_text = None
    raw_text = message.get("text")
    if isinstance(raw_text, str):
        message_text = raw_text

    return IncomingTelegramApprovalCallback(
        update_id=update_id,
        callback_query_id=callback_query_id.strip(),
        chat_id=chat_id,
        message_id=message_id,
        sender_user_id=sender_user_id,
        approval_id=approval_id,
        approved=approved,
        message_text=message_text,
    )


def route_id_for_chat(chat_id: int) -> str:
    chat_segment = str(chat_id)
    if chat_segment.startswith("-"):
        chat_segment = f"n{chat_segment[1:]}"
    return f"tg_{chat_segment}"


def chat_id_for_route_id(route_id: str) -> int | None:
    normalized = route_id.strip()
    if not normalized.startswith("tg_"):
        return None

    chat_segment = normalized.removeprefix("tg_")
    if not chat_segment:
        return None
    if chat_segment.startswith("n"):
        digits = chat_segment[1:]
        if digits.isdigit():
            return -int(digits)
        return None
    if chat_segment.isdigit():
        return int(chat_segment)
    return None


def _build_approval_callback_data(*, approval_id: str, approved: bool) -> str:
    decision = "1" if approved else "0"
    return f"{_APPROVAL_CALLBACK_PREFIX}:{decision}:{approval_id}"


def _parse_approval_callback_data(raw_value: str) -> tuple[bool, str] | None:
    prefix = f"{_APPROVAL_CALLBACK_PREFIX}:"
    if not raw_value.startswith(prefix):
        return None
    parts = raw_value.split(":", maxsplit=2)
    if len(parts) != 3:
        return None
    decision, approval_id = parts[1], parts[2].strip()
    if not approval_id:
        return None
    if decision == "1":
        return True, approval_id
    if decision == "0":
        return False, approval_id
    return None


def _format_approval_request_message(approval: GatewayApprovalRequestEvent) -> str:
    escaped_summary = html.escape(approval.summary.strip() or "Approval requested.")
    lines = [
        "<b>Approval required</b>",
        f"<b>summary:</b> {escaped_summary}",
    ]
    details = approval.details.strip()
    if details:
        lines.append(f"<b>details:</b> {html.escape(details)}")
    tool_name = (approval.tool_name or "").strip()
    if tool_name:
        lines.append(f"<b>tool_name:</b> {html.escape(tool_name)}")
    command = (approval.command or "").strip()
    if command:
        lines.append("<b>command:</b>")
        lines.append(f"<pre>{html.escape(_truncate_approval_command(command))}</pre>")
    inspection_url = (approval.inspection_url or "").strip()
    if inspection_url:
        lines.append(f"<b>inspect:</b> {html.escape(inspection_url)}")
    return "\n".join(lines)


def _truncate_approval_command(command: str) -> str:
    if len(command) <= _APPROVAL_COMMAND_PREVIEW_CHARS:
        return command
    return command[: _APPROVAL_COMMAND_PREVIEW_CHARS - 15] + "\n...[truncated]..."


def _extract_file_attachment(message: dict[str, Any]) -> IncomingTelegramFile | None:
    document = message.get("document")
    if isinstance(document, dict):
        return _build_file_attachment(
            "document",
            document,
            extra_keys=("thumbnail",),
        )

    photo = message.get("photo")
    if isinstance(photo, list):
        best_photo = _largest_photo_size(photo)
        if best_photo is not None:
            return _build_file_attachment("photo", best_photo)

    for media_type in (
        "video",
        "audio",
        "voice",
        "animation",
        "video_note",
        "sticker",
    ):
        payload = message.get(media_type)
        if isinstance(payload, dict):
            return _build_file_attachment(media_type, payload)

    return None


def _build_file_attachment(
    telegram_media_type: str,
    payload: dict[str, Any],
    *,
    extra_keys: tuple[str, ...] = (),
) -> IncomingTelegramFile | None:
    file_id = payload.get("file_id")
    if not isinstance(file_id, str) or not file_id.strip():
        return None

    extra_metadata: list[tuple[str, str]] = []
    for key in (
        "width",
        "height",
        "duration",
        "performer",
        "title",
        "emoji",
        "set_name",
        "type",
        *extra_keys,
    ):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            extra_metadata.append((key, str(value)))

    file_unique_id = payload.get("file_unique_id")
    original_file_name = payload.get("file_name")
    mime_type = payload.get("mime_type")
    size_bytes = payload.get("file_size")

    return IncomingTelegramFile(
        telegram_media_type=telegram_media_type,
        file_id=file_id,
        file_unique_id=(
            str(file_unique_id) if file_unique_id is not None else None
        ),
        original_file_name=(
            str(original_file_name)
            if original_file_name is not None
            else None
        ),
        mime_type=str(mime_type) if mime_type is not None else None,
        size_bytes=size_bytes if isinstance(size_bytes, int) else None,
        extra_metadata=tuple(extra_metadata),
    )


def _largest_photo_size(photo_sizes: list[Any]) -> dict[str, Any] | None:
    best_photo: dict[str, Any] | None = None
    best_area = -1
    best_file_size = -1
    for candidate in photo_sizes:
        if not isinstance(candidate, dict):
            continue
        file_id = candidate.get("file_id")
        if not isinstance(file_id, str) or not file_id.strip():
            continue
        width = candidate.get("width")
        height = candidate.get("height")
        area = width * height if isinstance(width, int) and isinstance(height, int) else -1
        file_size = candidate.get("file_size")
        normalized_file_size = file_size if isinstance(file_size, int) else -1
        if area > best_area or (
            area == best_area and normalized_file_size > best_file_size
        ):
            best_photo = candidate
            best_area = area
            best_file_size = normalized_file_size
    return best_photo


def _effective_private_user_id(message: IncomingTelegramMessage) -> int | None:
    if message.sender_user_id is not None:
        return message.sender_user_id
    if message.chat_type == "private":
        return message.chat_id
    return None


def _is_authorized_private_message(
    *,
    message: IncomingTelegramMessage,
    allowed_user_id: int | None,
) -> bool:
    if allowed_user_id is None:
        return False
    if message.chat_type != "private":
        return False
    if message.sender_is_bot or message.has_sender_chat:
        return False
    effective_user_id = _effective_private_user_id(message)
    return effective_user_id == allowed_user_id and message.chat_id == allowed_user_id


def _build_download_destination(
    *,
    temp_dir: Path,
    chat_id: int,
    update_id: int,
    attachment: IncomingTelegramFile,
    remote_file: TelegramRemoteFile,
) -> Path:
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_name = (
        attachment.original_file_name
        or Path(remote_file.file_path).name
        or f"{attachment.telegram_media_type}-{update_id}"
    )
    safe_name = _sanitize_filename(file_name)
    return temp_dir / f"tg-{chat_id}-{update_id}-{safe_name}"


def _sanitize_filename(file_name: str) -> str:
    base_name = Path(file_name).name.strip()
    if not base_name:
        return "telegram-file"

    suffix = "".join(Path(base_name).suffixes)
    stem = base_name[: -len(suffix)] if suffix else base_name
    normalized_stem = _FILENAME_SANITIZE_PATTERN.sub("_", stem).strip("._")
    normalized_suffix = _FILENAME_SANITIZE_PATTERN.sub("", suffix)
    normalized_stem = normalized_stem or "telegram-file"
    return f"{normalized_stem[:80]}{normalized_suffix[:20]}"


def _format_incoming_file_message(
    *,
    attachment: IncomingTelegramFile,
    local_path: Path,
    remote_file: TelegramRemoteFile,
    caption: str,
) -> str:
    lines = [
        "User sent a Telegram file.",
        f"filename: {attachment.original_file_name or local_path.name}",
        f"telegram_media_type: {attachment.telegram_media_type}",
        f"local_path: {local_path}",
    ]
    if attachment.mime_type is not None:
        lines.append(f"mime_type: {attachment.mime_type}")
    if attachment.size_bytes is not None:
        lines.append(f"size_bytes: {attachment.size_bytes}")
    elif remote_file.file_size is not None:
        lines.append(f"size_bytes: {remote_file.file_size}")
    if attachment.file_unique_id is not None:
        lines.append(f"telegram_file_unique_id: {attachment.file_unique_id}")
    lines.append(f"telegram_remote_path: {remote_file.file_path}")
    for key, value in attachment.extra_metadata:
        lines.append(f"{key}: {value}")
    if caption:
        lines.append(f"caption: {caption}")
    return "\n".join(lines)


def _clear_directory_contents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)


def _remove_stale_temp_entries(path: Path, *, keep_since: datetime) -> None:
    path.mkdir(parents=True, exist_ok=True)
    keep_since_timestamp = keep_since.timestamp()
    for entry in path.iterdir():
        if entry.stat().st_mtime >= keep_since_timestamp:
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)


def _seconds_until_next_local_midnight(now: datetime) -> float:
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return max((next_midnight - now).total_seconds(), 1.0)


async def _maintain_temp_dir(temp_dir: Path) -> None:
    await asyncio.to_thread(
        _remove_stale_temp_entries,
        temp_dir,
        keep_since=datetime.now().astimezone().replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ),
    )
    while True:
        now = datetime.now().astimezone()
        await asyncio.sleep(_seconds_until_next_local_midnight(now))
        await asyncio.to_thread(_clear_directory_contents, temp_dir)
        LOGGER.info("Cleared Telegram temp dir %s", temp_dir)


def _should_flush_draft(
    *,
    current_text: str,
    last_sent_text: str,
    last_sent_monotonic: float,
    segment_started_monotonic: float,
    min_chars: int,
    min_interval_seconds: float,
    now_monotonic: float,
) -> bool:
    if not current_text or current_text == last_sent_text:
        return False

    unsent_chars = len(current_text) - len(last_sent_text)
    if unsent_chars <= 0:
        return False

    if last_sent_monotonic == 0.0:
        if unsent_chars < min_chars:
            return False
        return (now_monotonic - segment_started_monotonic) >= min_interval_seconds

    if (now_monotonic - last_sent_monotonic) < min_interval_seconds:
        return False

    return True


def _has_visible_telegram_text(text: str) -> bool:
    return _has_effective_text(_HTML_TAG_PATTERN.sub("", text))


def _format_tool_usage_notice(tool_name: str) -> str:
    normalized_name = html.escape(tool_name.strip() or "unknown")
    return f"🔧 Used <b>{normalized_name}</b> tool."


def _is_stop_command_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.split(maxsplit=1)[0] == "/stop"


def _coalesce_visible_text(*candidates: str) -> str | None:
    for candidate in candidates:
        normalized = _strip_invisible_edges(candidate)
        if _has_effective_text(normalized):
            return normalized
    return None


def split_telegram_message(text: str, *, max_chars: int = 4_096) -> list[str]:
    if not _has_effective_text(text):
        return ["(No response text.)"]
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            newline_break = text.rfind("\n", start, end)
            if newline_break > start:
                end = newline_break + 1
        chunk = text[start:end]
        if _has_effective_text(chunk):
            chunks.append(chunk)
        start = end
    return chunks or ["(No response text.)"]


def _has_effective_text(text: str) -> bool:
    return any(_is_effectively_visible_character(char) for char in text)


def _strip_invisible_edges(text: str) -> str:
    start = 0
    end = len(text)
    while start < end and not _is_effectively_visible_character(text[start]):
        start += 1
    while end > start and not _is_effectively_visible_character(text[end - 1]):
        end -= 1
    return text[start:end]


def _is_effectively_visible_character(char: str) -> bool:
    if char.isspace():
        return False
    return unicodedata.category(char)[0] not in {"C", "M"}


def _is_formatting_error(exc: TelegramAPIError) -> bool:
    if exc.code != "telegram_api_error_400":
        return False
    lowered = exc.message.lower()
    formatting_markers = (
        "parse entities",
        "unsupported start tag",
        "unexpected end tag",
        "can't find end tag",
    )
    return any(marker in lowered for marker in formatting_markers)


async def send_owner_telegram_file(
    *,
    file_path: str | Path,
    caption: str | None = None,
    filename: str | None = None,
    settings: UISettings | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or UISettings.from_env()
    allowed_user_id = resolved_settings.telegram_allowed_user_id
    if allowed_user_id is None:
        raise UIConfigurationError(
            "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID must be set to send files to the Telegram owner."
        )

    client = TelegramBotAPIClient(
        token=resolved_settings.telegram_token,
        api_base_url=resolved_settings.telegram_api_base_url,
    )
    try:
        return await client.send_document(
            chat_id=allowed_user_id,
            file_path=file_path,
            caption=caption,
            filename=filename,
        )
    finally:
        await client.aclose()


async def send_telegram_file(
    *,
    file_path: str | Path,
    caption: str | None = None,
    filename: str | None = None,
    route_id: str | None = None,
    chat_id: int | None = None,
    settings: UISettings | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or UISettings.from_env()
    resolved_chat_id = _resolve_delivery_chat_id(
        route_id=route_id,
        chat_id=chat_id,
        owner_chat_id=resolved_settings.telegram_allowed_user_id,
    )

    client = TelegramBotAPIClient(
        token=resolved_settings.telegram_token,
        api_base_url=resolved_settings.telegram_api_base_url,
    )
    try:
        result = await client.send_document(
            chat_id=resolved_chat_id,
            file_path=file_path,
            caption=caption,
            filename=filename,
        )
    finally:
        await client.aclose()

    if isinstance(result, dict):
        enriched_result = dict(result)
        enriched_result.setdefault("chat_id", resolved_chat_id)
        return enriched_result
    return {
        "chat_id": resolved_chat_id,
        "result": result,
    }


async def run_telegram_ui(settings: UISettings | None = None) -> None:
    bridge = TelegramGatewayBridge(settings=settings)
    cleanup_task = asyncio.create_task(
        _maintain_temp_dir(bridge._settings.telegram_temp_dir),
        name="jarvis-telegram-temp-maintenance",
    )
    try:
        await bridge.run_forever()
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        await bridge.aclose()


def _resolve_delivery_chat_id(
    *,
    route_id: str | None,
    chat_id: int | None,
    owner_chat_id: int | None,
) -> int:
    if chat_id is not None:
        return chat_id
    if route_id is not None:
        resolved = chat_id_for_route_id(route_id)
        if resolved is not None:
            return resolved
        if owner_chat_id is None:
            raise UIConfigurationError(
                "send_file could not map the active route to a Telegram chat and no owner fallback is configured."
            )
        return owner_chat_id
    if owner_chat_id is None:
        raise UIConfigurationError(
            "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID must be set when send_file has no Telegram route context."
        )
    return owner_chat_id
