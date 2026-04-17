"""Thin async wrapper around Telegram Bot HTTP API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class TelegramAPIError(RuntimeError):
    """Raised when Telegram API returns an error response."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


@dataclass(slots=True, frozen=True)
class DraftMessage:
    id: int
    text: str


@dataclass(slots=True, frozen=True)
class TelegramRemoteFile:
    file_id: str
    file_path: str
    file_unique_id: str | None = None
    file_size: int | None = None


def _redact_token(value: str, *, token: str) -> str:
    if not token:
        return value
    return value.replace(token, "[REDACTED]")


def _describe_request_error(exc: httpx.RequestError, *, token: str) -> str:
    detail = _redact_token(str(exc).strip(), token=token)
    if detail:
        return f"{type(exc).__name__}: {detail}"
    return type(exc).__name__


def _sanitize_request_error(
    exc: httpx.RequestError,
    *,
    token: str,
) -> httpx.RequestError:
    sanitized_message = _redact_token(str(exc).strip(), token=token)
    if sanitized_message:
        exc.args = (sanitized_message,)
    return exc


class TelegramBotAPIClient:
    """Async client facade using httpx under the hood."""

    def __init__(
        self,
        *,
        token: str,
        api_base_url: str = "https://api.telegram.org",
        request_timeout_seconds: float = 20.0,
        file_transfer_timeout_seconds: float = 120.0,
    ) -> None:
        self._token = token
        self._api_base_url = api_base_url.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._file_transfer_timeout_seconds = max(
            file_transfer_timeout_seconds,
            request_timeout_seconds,
        )
        self._session = httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._session.aclose()

    async def get_me(self) -> dict[str, Any]:
        result = await self._call_method("getMe", payload={})
        if not isinstance(result, dict):
            raise TelegramAPIError(
                code="invalid_telegram_response",
                message="Telegram getMe returned an unexpected payload shape.",
            )
        return result

    async def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout_seconds,
            "limit": limit,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset

        request_timeout = self._request_timeout_seconds + timeout_seconds + 5
        result = await self._call_method(
            "getUpdates",
            payload=payload,
            request_timeout_seconds=request_timeout,
        )
        if not isinstance(result, list):
            raise TelegramAPIError(
                code="invalid_telegram_response",
                message="Telegram getUpdates returned an unexpected payload shape.",
            )
        return [item for item in result if isinstance(item, dict)]

    async def get_file(self, *, file_id: str) -> TelegramRemoteFile:
        result = await self._call_method("getFile", payload={"file_id": file_id})
        if not isinstance(result, dict):
            raise TelegramAPIError(
                code="invalid_telegram_response",
                message="Telegram getFile returned an unexpected payload shape.",
            )
        file_path = result.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            raise TelegramAPIError(
                code="invalid_telegram_response",
                message="Telegram getFile did not include a file_path.",
            )
        file_unique_id = result.get("file_unique_id")
        file_size = result.get("file_size")
        return TelegramRemoteFile(
            file_id=(
                str(result.get("file_id"))
                if result.get("file_id") is not None
                else file_id
            ),
            file_path=file_path,
            file_unique_id=(
                str(file_unique_id) if file_unique_id is not None else None
            ),
            file_size=file_size if isinstance(file_size, int) else None,
        )

    async def download_file(
        self,
        *,
        remote_file_path: str,
        destination_path: str | Path,
    ) -> Path:
        destination = Path(destination_path).expanduser()
        url = f"{self._api_base_url}/file/bot{self._token}/{remote_file_path.lstrip('/')}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with self._session.stream(
                "GET",
                url,
                timeout=self._file_transfer_timeout_seconds,
            ) as response:
                if response.status_code >= 400:
                    raise TelegramAPIError(
                        code="telegram_http_error",
                        message=f"Telegram file download failed for path '{remote_file_path}'.",
                        status_code=response.status_code,
                    )
                with destination.open("wb") as handle:
                    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                        if chunk:
                            handle.write(chunk)
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                code="telegram_http_error",
                message=(
                    f"Telegram file download failed for path '{remote_file_path}': "
                    f"{_describe_request_error(exc, token=self._token)}"
                ),
            ) from _sanitize_request_error(exc, token=self._token)
        return destination

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        result = await self._call_method("sendMessage", payload=payload)
        if not isinstance(result, dict):
            raise TelegramAPIError(
                code="invalid_telegram_response",
                message="Telegram sendMessage returned an unexpected payload shape.",
            )
        return result

    async def edit_message_text(
        self,
        *,
        chat_id: int,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        result = await self._call_method("editMessageText", payload=payload)
        if not isinstance(result, dict):
            raise TelegramAPIError(
                code="invalid_telegram_response",
                message="Telegram editMessageText returned an unexpected payload shape.",
            )
        return result

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> bool:
        payload: dict[str, Any] = {
            "callback_query_id": callback_query_id,
            "show_alert": show_alert,
        }
        if text is not None:
            payload["text"] = text
        result = await self._call_method("answerCallbackQuery", payload=payload)
        return bool(result)

    async def send_chat_action(
        self,
        *,
        chat_id: int,
        action: str,
    ) -> bool:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "action": action,
        }
        result = await self._call_method("sendChatAction", payload=payload)
        return bool(result)

    async def send_message_draft(
        self,
        *,
        chat_id: int,
        draft: DraftMessage,
        parse_mode: str | None = None,
    ) -> bool:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "draft_id": draft.id,
            "text": draft.text,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        result = await self._call_method("sendMessageDraft", payload=payload)
        return bool(result)

    async def send_document(
        self,
        *,
        chat_id: int,
        file_path: str | Path,
        caption: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self._api_base_url}/bot{self._token}/sendDocument"
        data: dict[str, str] = {"chat_id": str(chat_id)}
        if caption is not None:
            data["caption"] = caption

        resolved_file_path = Path(file_path).expanduser()
        with resolved_file_path.open("rb") as handle:
            upload_name = filename or resolved_file_path.name
            files = {"document": (upload_name, handle)}
            try:
                response = await self._session.post(
                    url,
                    data=data,
                    files=files,
                    timeout=self._file_transfer_timeout_seconds,
                )
            except httpx.RequestError as exc:
                raise TelegramAPIError(
                    code="telegram_http_error",
                    message=(
                        "Telegram request failed for method 'sendDocument': "
                        f"{_describe_request_error(exc, token=self._token)}"
                    ),
                ) from _sanitize_request_error(exc, token=self._token)

        result = self._parse_method_response(response, method="sendDocument")
        if not isinstance(result, dict):
            raise TelegramAPIError(
                code="invalid_telegram_response",
                message="Telegram sendDocument returned an unexpected payload shape.",
            )
        return result

    async def _call_method(
        self,
        method: str,
        *,
        payload: dict[str, Any],
        request_timeout_seconds: float | None = None,
    ) -> Any:
        timeout = (
            request_timeout_seconds
            if request_timeout_seconds is not None
            else self._request_timeout_seconds
        )
        url = f"{self._api_base_url}/bot{self._token}/{method}"
        try:
            response = await self._session.post(url, json=payload, timeout=timeout)
        except httpx.RequestError as exc:
            raise TelegramAPIError(
                code="telegram_http_error",
                message=(
                    f"Telegram request failed for method '{method}': "
                    f"{_describe_request_error(exc, token=self._token)}"
                ),
            ) from _sanitize_request_error(exc, token=self._token)

        return self._parse_method_response(response, method=method)

    def _parse_method_response(
        self,
        response: Any,
        *,
        method: str,
    ) -> Any:
        try:
            parsed = response.json()
        except ValueError as exc:
            raise TelegramAPIError(
                code="telegram_http_error",
                message=f"Telegram returned non-JSON response for method '{method}'.",
                status_code=response.status_code,
            ) from exc

        if not isinstance(parsed, dict):
            raise TelegramAPIError(
                code="telegram_http_error",
                message=f"Telegram returned unexpected payload for method '{method}'.",
                status_code=response.status_code,
            )

        if response.status_code >= 400 or not parsed.get("ok", False):
            description = str(parsed.get("description", "Telegram API error"))
            error_code = parsed.get("error_code")
            retry_after_seconds: int | None = None
            parameters = parsed.get("parameters")
            if isinstance(parameters, dict):
                retry_after = parameters.get("retry_after")
                if isinstance(retry_after, int):
                    retry_after_seconds = retry_after
                elif isinstance(retry_after, float):
                    retry_after_seconds = int(retry_after)
            raise TelegramAPIError(
                code=(
                    f"telegram_api_error_{error_code}"
                    if error_code is not None
                    else "telegram_api_error"
                ),
                message=description,
                status_code=response.status_code,
                retry_after_seconds=retry_after_seconds,
            )

        return parsed.get("result")
