"""Thin async wrapper around Telegram Bot HTTP API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import requests


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


class TelegramBotAPIClient:
    """Async client facade using requests under the hood."""

    def __init__(
        self,
        *,
        token: str,
        api_base_url: str = "https://api.telegram.org",
        request_timeout_seconds: float = 20.0,
    ) -> None:
        self._token = token
        self._api_base_url = api_base_url.rstrip("/")
        self._request_timeout_seconds = request_timeout_seconds
        self._session = requests.Session()

    async def aclose(self) -> None:
        await asyncio.to_thread(self._session.close)

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
            "allowed_updates": ["message"],
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

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        result = await self._call_method("sendMessage", payload=payload)
        if not isinstance(result, dict):
            raise TelegramAPIError(
                code="invalid_telegram_response",
                message="Telegram sendMessage returned an unexpected payload shape.",
            )
        return result

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
        return await asyncio.to_thread(
            self._call_method_blocking,
            method,
            payload,
            timeout,
        )

    def _call_method_blocking(
        self,
        method: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> Any:
        url = f"{self._api_base_url}/bot{self._token}/{method}"
        try:
            response = self._session.post(url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise TelegramAPIError(
                code="telegram_http_error",
                message=f"Telegram request failed for method '{method}'.",
            ) from exc

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
