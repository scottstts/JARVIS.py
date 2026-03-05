"""Telegram bot runtime that bridges inbound messages to the gateway websocket."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from .config import UISettings
from .gateway_client import GatewayBridgeError, GatewayDeltaEvent, GatewayDoneEvent, GatewayWebSocketClient
from .telegram_api import DraftMessage, TelegramAPIError, TelegramBotAPIClient

LOGGER = logging.getLogger(__name__)
_GENERIC_ERROR_REPLY = "I hit an internal error while processing that message."


class GatewayClientLike(Protocol):
    async def stream_turn(self, *, route_id: str, user_text: str) -> AsyncIterator[Any]:
        """Streams one gateway turn."""


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

    async def send_message(self, *, chat_id: int, text: str) -> dict[str, Any]:
        """Sends a standard Telegram message."""

    async def send_message_draft(self, *, chat_id: int, draft: DraftMessage) -> bool:
        """Sends/updates Telegram draft message content."""

    async def aclose(self) -> None:
        """Releases client resources."""


@dataclass(slots=True, frozen=True)
class IncomingTextMessage:
    update_id: int
    chat_id: int
    chat_type: str
    text: str


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

    async def run_forever(self) -> None:
        bot_profile = await self._telegram.get_me()
        LOGGER.info(
            "Telegram bridge online for bot @%s",
            bot_profile.get("username", "unknown"),
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
                LOGGER.exception("Unexpected error in polling loop; retrying after backoff.")
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
            message = parse_incoming_text_message(update)
            if message is None:
                continue
            await self.handle_message(message)
        return next_offset

    async def handle_message(self, message: IncomingTextMessage) -> None:
        if message.chat_type != "private":
            return
        route_id = route_id_for_chat(message.chat_id)
        draft_id = self._next_draft_id_for_chat(message.chat_id)

        accumulated_text = ""
        last_draft_text = ""
        last_draft_at = 0.0
        drafts_enabled = True

        try:
            async for event in self._gateway.stream_turn(route_id=route_id, user_text=message.text):
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
                        min_chars=self._settings.stream_draft_min_chars,
                        min_interval_seconds=self._settings.stream_draft_min_interval_seconds,
                        now_monotonic=now,
                    ):
                        try:
                            await self._send_draft(
                                chat_id=message.chat_id,
                                draft_id=draft_id,
                                text=accumulated_text,
                            )
                            last_draft_text = accumulated_text
                            last_draft_at = now
                        except TelegramAPIError:
                            drafts_enabled = False
                            LOGGER.exception(
                                "sendMessageDraft failed; continuing this turn without drafts."
                            )
                    continue

                if isinstance(event, GatewayDoneEvent):
                    final_text = event.text or "(No response text.)"
                    await self._send_final_text(chat_id=message.chat_id, text=final_text)
                    return

            await self._send_final_text(
                chat_id=message.chat_id,
                text=_GENERIC_ERROR_REPLY,
            )
        except GatewayBridgeError as exc:
            LOGGER.exception("Gateway bridge failed for chat_id=%s", message.chat_id)
            await self._send_final_text(
                chat_id=message.chat_id,
                text=exc.message if exc.message else _GENERIC_ERROR_REPLY,
            )
        except TelegramAPIError:
            LOGGER.exception("Telegram API send failed for chat_id=%s", message.chat_id)
        except Exception:
            LOGGER.exception("Unexpected bridge error for chat_id=%s", message.chat_id)
            try:
                await self._send_final_text(
                    chat_id=message.chat_id,
                    text=_GENERIC_ERROR_REPLY,
                )
            except TelegramAPIError:
                LOGGER.exception("Failed to send fallback error text for chat_id=%s", message.chat_id)

    async def aclose(self) -> None:
        await self._telegram.aclose()

    async def _send_draft(self, *, chat_id: int, draft_id: int, text: str) -> None:
        max_chars = self._settings.telegram_max_message_chars
        draft_text = text
        if len(draft_text) > max_chars:
            draft_text = f"{draft_text[: max_chars - 1]}…"
        draft_text = draft_text or "."
        await self._telegram.send_message_draft(
            chat_id=chat_id,
            draft=DraftMessage(id=draft_id, text=draft_text),
        )

    async def _send_final_text(self, *, chat_id: int, text: str) -> None:
        chunks = split_telegram_message(text, max_chars=self._settings.telegram_max_message_chars)
        for chunk in chunks:
            await self._telegram.send_message(chat_id=chat_id, text=chunk)

    def _next_draft_id_for_chat(self, chat_id: int) -> int:
        next_id = self._next_draft_ids.get(chat_id, 1)
        self._next_draft_ids[chat_id] = next_id + 1
        return next_id


def parse_incoming_text_message(update: dict[str, Any]) -> IncomingTextMessage | None:
    if not isinstance(update, dict):
        return None
    update_id = update.get("update_id")
    if not isinstance(update_id, int):
        return None

    message = update.get("message")
    if not isinstance(message, dict):
        return None
    if not isinstance(message.get("text"), str):
        return None
    text = message["text"].strip()
    if not text:
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

    return IncomingTextMessage(
        update_id=update_id,
        chat_id=chat_id,
        chat_type=chat_type,
        text=text,
    )


def route_id_for_chat(chat_id: int) -> str:
    chat_segment = str(chat_id)
    if chat_segment.startswith("-"):
        chat_segment = f"n{chat_segment[1:]}"
    return f"tg_{chat_segment}"


def _should_flush_draft(
    *,
    current_text: str,
    last_sent_text: str,
    last_sent_monotonic: float,
    min_chars: int,
    min_interval_seconds: float,
    now_monotonic: float,
) -> bool:
    if not current_text or current_text == last_sent_text:
        return False

    unsent_chars = len(current_text) - len(last_sent_text)
    if unsent_chars >= min_chars:
        return True

    if last_sent_monotonic == 0.0:
        return True

    elapsed = now_monotonic - last_sent_monotonic
    return elapsed >= min_interval_seconds


def split_telegram_message(text: str, *, max_chars: int = 4_096) -> list[str]:
    if not text:
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
        chunks.append(text[start:end])
        start = end
    return chunks


async def run_telegram_ui(settings: UISettings | None = None) -> None:
    bridge = TelegramGatewayBridge(settings=settings)
    try:
        await bridge.run_forever()
    finally:
        await bridge.aclose()
