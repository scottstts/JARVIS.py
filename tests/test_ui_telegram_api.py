"""Unit tests for Telegram Bot API client behavior."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from ui.telegram.api import (
    DraftMessage,
    TelegramAPIError,
    TelegramBotAPIClient,
    TelegramRemoteFile,
)


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: object | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._body = body

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("No JSON payload configured.")
        return self._payload

    def iter_content(self, chunk_size: int) -> list[bytes]:
        _ = chunk_size
        return [self._body]


class _FakeSession:
    def __init__(
        self,
        *,
        post_responses: list[_FakeResponse] | None = None,
        get_responses: list[_FakeResponse] | None = None,
    ) -> None:
        self._post_responses = post_responses or []
        self._get_responses = get_responses or []
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
        data: dict[str, object] | None = None,
        files: dict[str, object] | None = None,
        timeout: float,
    ) -> _FakeResponse:
        self.calls.append(
            {
                "method": "POST",
                "url": url,
                "json": json,
                "data": data,
                "files": files,
                "timeout": timeout,
            }
        )
        if not self._post_responses:
            raise AssertionError("No fake POST response available.")
        return self._post_responses.pop(0)

    def get(self, url: str, *, stream: bool, timeout: float) -> _FakeResponse:
        self.calls.append(
            {
                "method": "GET",
                "url": url,
                "stream": stream,
                "timeout": timeout,
            }
        )
        if not self._get_responses:
            raise AssertionError("No fake GET response available.")
        return self._get_responses.pop(0)

    def close(self) -> None:
        return None


class TelegramBotAPIClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_message_includes_parse_mode(self) -> None:
        session = _FakeSession(
            post_responses=[
                _FakeResponse(
                    status_code=200,
                    payload={"ok": True, "result": {"message_id": 1}},
                )
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        await client.send_message(chat_id=123, text="<b>hi</b>", parse_mode="HTML")

        self.assertEqual(session.calls[0]["json"]["parse_mode"], "HTML")

    async def test_send_message_draft_includes_parse_mode(self) -> None:
        session = _FakeSession(
            post_responses=[
                _FakeResponse(status_code=200, payload={"ok": True, "result": True})
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        await client.send_message_draft(
            chat_id=123,
            draft=DraftMessage(id=7, text="<b>hi</b>"),
            parse_mode="HTML",
        )

        self.assertEqual(session.calls[0]["json"]["parse_mode"], "HTML")

    async def test_get_file_returns_remote_file_metadata(self) -> None:
        session = _FakeSession(
            post_responses=[
                _FakeResponse(
                    status_code=200,
                    payload={
                        "ok": True,
                        "result": {
                            "file_id": "file-1",
                            "file_unique_id": "unique-1",
                            "file_path": "documents/report.pdf",
                            "file_size": 42,
                        },
                    },
                )
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        result = await client.get_file(file_id="file-1")

        self.assertEqual(
            result,
            TelegramRemoteFile(
                file_id="file-1",
                file_unique_id="unique-1",
                file_path="documents/report.pdf",
                file_size=42,
            ),
        )

    async def test_download_file_writes_to_destination(self) -> None:
        session = _FakeSession(
            get_responses=[
                _FakeResponse(
                    status_code=200,
                    body=b"hello world",
                )
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "report.pdf"
            result = await client.download_file(
                remote_file_path="documents/report.pdf",
                destination_path=destination,
            )

            self.assertEqual(result, destination)
            self.assertEqual(destination.read_bytes(), b"hello world")
            self.assertIn("/file/bottoken/documents/report.pdf", session.calls[0]["url"])

    async def test_send_document_posts_multipart_file(self) -> None:
        session = _FakeSession(
            post_responses=[
                _FakeResponse(
                    status_code=200,
                    payload={"ok": True, "result": {"message_id": 5}},
                )
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.txt"
            path.write_text("hello", encoding="utf-8")

            await client.send_document(
                chat_id=123,
                file_path=path,
                caption="attached",
                filename="custom-name.txt",
            )

        self.assertEqual(session.calls[0]["data"]["chat_id"], "123")
        self.assertEqual(session.calls[0]["data"]["caption"], "attached")
        uploaded_file = session.calls[0]["files"]["document"]
        self.assertEqual(uploaded_file[0], "custom-name.txt")

    async def test_raises_retry_after_on_rate_limit(self) -> None:
        session = _FakeSession(
            post_responses=[
                _FakeResponse(
                    status_code=429,
                    payload={
                        "ok": False,
                        "error_code": 429,
                        "description": "Too Many Requests: retry later",
                        "parameters": {"retry_after": 5},
                    },
                )
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        with self.assertRaises(TelegramAPIError) as context:
            await client.send_message(chat_id=123, text="hi")

        self.assertEqual(context.exception.code, "telegram_api_error_429")
        self.assertEqual(context.exception.retry_after_seconds, 5)
        self.assertEqual(context.exception.status_code, 429)
