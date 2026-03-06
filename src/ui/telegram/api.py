"""Thin async wrapper around Telegram Bot HTTP API."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(slots=True, frozen=True)
class TelegramRemoteFile:
    file_id: str
    file_path: str
    file_unique_id: str | None = None
    file_size: int | None = None


class TelegramBotAPIClient:
    """Async client facade using requests under the hood."""

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
        await asyncio.to_thread(
            self._download_file_blocking,
            remote_file_path,
            destination,
        )
        return destination

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

    async def send_document(
        self,
        *,
        chat_id: int,
        file_path: str | Path,
        caption: str | None = None,
        filename: str | None = None,
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            self._send_document_blocking,
            chat_id,
            Path(file_path).expanduser(),
            caption,
            filename,
        )
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

        return self._parse_method_response(response, method=method)

    def _send_document_blocking(
        self,
        chat_id: int,
        file_path: Path,
        caption: str | None,
        filename: str | None,
    ) -> Any:
        url = f"{self._api_base_url}/bot{self._token}/sendDocument"
        data: dict[str, str] = {"chat_id": str(chat_id)}
        if caption is not None:
            data["caption"] = caption

        with file_path.open("rb") as handle:
            upload_name = filename or file_path.name
            files = {"document": (upload_name, handle)}
            try:
                response = self._session.post(
                    url,
                    data=data,
                    files=files,
                    timeout=self._file_transfer_timeout_seconds,
                )
            except requests.RequestException as exc:
                raise TelegramAPIError(
                    code="telegram_http_error",
                    message="Telegram request failed for method 'sendDocument'.",
                ) from exc

        return self._parse_method_response(response, method="sendDocument")

    def _download_file_blocking(
        self,
        remote_file_path: str,
        destination_path: Path,
    ) -> None:
        url = f"{self._api_base_url}/file/bot{self._token}/{remote_file_path.lstrip('/')}"
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = self._session.get(
                url,
                stream=True,
                timeout=self._file_transfer_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise TelegramAPIError(
                code="telegram_http_error",
                message=f"Telegram file download failed for path '{remote_file_path}'.",
            ) from exc

        if response.status_code >= 400:
            raise TelegramAPIError(
                code="telegram_http_error",
                message=f"Telegram file download failed for path '{remote_file_path}'.",
                status_code=response.status_code,
            )

        with destination_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    handle.write(chunk)

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
