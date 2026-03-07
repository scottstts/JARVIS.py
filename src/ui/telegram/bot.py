"""Telegram bot runtime that bridges inbound messages to the gateway websocket."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
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
    GatewayBridgeError,
    GatewayDeltaEvent,
    GatewayDoneEvent,
    GatewayWebSocketClient,
)

LOGGER = logging.getLogger(__name__)
_GENERIC_ERROR_REPLY = "I hit an internal error while processing that message."
_FILE_DOWNLOAD_ERROR_REPLY = "I couldn't download that file from Telegram."
_FILENAME_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


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
    ) -> dict[str, Any]:
        """Sends a standard Telegram message."""

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
    file_attachment: IncomingTelegramFile | None = None


IncomingTextMessage = IncomingTelegramMessage


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
        if self._settings.telegram_allowed_user_id is None:
            LOGGER.warning(
                "Telegram owner gate is disabled; any private chat can reach the bot."
            )
        else:
            LOGGER.info(
                "Telegram bridge restricted to user_id=%s",
                self._settings.telegram_allowed_user_id,
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
            message = parse_incoming_message(update)
            if message is None:
                continue
            await self.handle_message(message)
        return next_offset

    async def handle_message(self, message: IncomingTelegramMessage) -> None:
        if message.chat_type != "private":
            return
        if not _is_authorized_private_message(
            message=message,
            allowed_user_id=self._settings.telegram_allowed_user_id,
        ):
            LOGGER.warning(
                "Ignoring unauthorized Telegram message from user_id=%s chat_id=%s",
                _effective_private_user_id(message),
                message.chat_id,
            )
            return

        user_text = message.text
        if message.file_attachment is not None:
            try:
                user_text = await self._build_file_turn_text(message)
            except TelegramAPIError:
                LOGGER.exception(
                    "Telegram file download failed for chat_id=%s",
                    message.chat_id,
                )
                await self._send_final_text(
                    chat_id=message.chat_id,
                    text=_FILE_DOWNLOAD_ERROR_REPLY,
                )
                return

        if not user_text:
            return

        route_id = route_id_for_chat(message.chat_id)
        draft_id = self._next_draft_id_for_chat(message.chat_id)

        accumulated_text = ""
        last_draft_text = ""
        last_draft_at = 0.0
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
                        except TelegramAPIError as exc:
                            drafts_enabled = False
                            self._record_draft_backoff(
                                chat_id=message.chat_id,
                                exc=exc,
                            )
                            if exc.retry_after_seconds is not None:
                                LOGGER.warning(
                                    "sendMessageDraft rate-limited for chat_id=%s; pausing drafts for %ss.",
                                    message.chat_id,
                                    exc.retry_after_seconds,
                                )
                            else:
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
                LOGGER.exception(
                    "Failed to send fallback error text for chat_id=%s",
                    message.chat_id,
                )

    async def aclose(self) -> None:
        await self._telegram.aclose()

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
        plain_draft_text = draft_text.strip()
        if not plain_draft_text:
            return

        rendered_text = render_markdown_to_telegram_html(draft_text)
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
        chunks = split_telegram_message(
            text,
            max_chars=self._settings.telegram_max_message_chars,
        )
        for chunk in chunks:
            await self._send_formatted_message(chat_id=chat_id, text=chunk)

    async def _send_formatted_message(self, *, chat_id: int, text: str) -> None:
        formatted_text = render_markdown_to_telegram_html(text)
        try:
            await self._telegram.send_message(
                chat_id=chat_id,
                text=formatted_text,
                parse_mode="HTML",
            )
        except TelegramAPIError as exc:
            if not _is_formatting_error(exc):
                raise
            await self._telegram.send_message(chat_id=chat_id, text=text)

    def _record_draft_backoff(self, *, chat_id: int, exc: TelegramAPIError) -> None:
        if exc.retry_after_seconds is None:
            return
        self._draft_retry_until_by_chat[chat_id] = (
            time.monotonic() + max(1, exc.retry_after_seconds)
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

    sender_user_id: int | None = None
    from_user = message.get("from")
    if isinstance(from_user, dict):
        from_user_id = from_user.get("id")
        if isinstance(from_user_id, int):
            sender_user_id = from_user_id

    chat = message.get("chat")
    if not isinstance(chat, dict):
        return None
    chat_id = chat.get("id")
    chat_type = chat.get("type")
    if not isinstance(chat_id, int):
        return None
    if not isinstance(chat_type, str):
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
        file_attachment=file_attachment,
    )


def parse_incoming_text_message(update: dict[str, Any]) -> IncomingTextMessage | None:
    return parse_incoming_message(update)


def route_id_for_chat(chat_id: int) -> str:
    chat_segment = str(chat_id)
    if chat_segment.startswith("-"):
        chat_segment = f"n{chat_segment[1:]}"
    return f"tg_{chat_segment}"


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
        return True
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


def _has_visible_telegram_text(text: str) -> bool:
    if not text.strip():
        return False
    return bool(_HTML_TAG_PATTERN.sub("", text).strip())


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
