"""Unit tests for Telegram Bot API client behavior."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx

from jarvis.ui.telegram.api import (
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

    async def aiter_bytes(self, chunk_size: int) -> Any:
        _ = chunk_size
        if self._body:
            yield self._body


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> bool:
        _ = (exc_type, exc, tb)
        return False


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

    async def post(
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

    def stream(self, method: str, url: str, *, timeout: float) -> _FakeStream:
        self.calls.append(
            {
                "method": method,
                "url": url,
                "timeout": timeout,
            }
        )
        if not self._get_responses:
            raise AssertionError("No fake stream response available.")
        return _FakeStream(self._get_responses.pop(0))

    async def aclose(self) -> None:
        return None


class _BlockingSession(_FakeSession):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self._release = asyncio.Event()

    async def post(
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
        self.started.set()
        await self._release.wait()
        raise AssertionError("Blocking session should be cancelled before completing.")


class _RequestErrorSession:
    def __init__(self, exc: httpx.RequestError) -> None:
        self._exc = exc

    async def post(
        self,
        url: str,
        *,
        json: dict[str, object] | None = None,
        data: dict[str, object] | None = None,
        files: dict[str, object] | None = None,
        timeout: float,
    ) -> _FakeResponse:
        _ = (url, json, data, files, timeout)
        raise self._exc

    def stream(self, method: str, url: str, *, timeout: float) -> _FakeStream:
        _ = (method, url, timeout)
        raise self._exc

    async def aclose(self) -> None:
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

    async def test_send_message_includes_reply_markup(self) -> None:
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

        await client.send_message(
            chat_id=123,
            text="approve?",
            reply_markup={"inline_keyboard": [[{"text": "Approve", "callback_data": "x"}]]},
        )

        self.assertEqual(
            session.calls[0]["json"]["reply_markup"],
            {"inline_keyboard": [[{"text": "Approve", "callback_data": "x"}]]},
        )

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

    async def test_get_updates_requests_callback_queries(self) -> None:
        session = _FakeSession(
            post_responses=[
                _FakeResponse(
                    status_code=200,
                    payload={"ok": True, "result": []},
                )
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        await client.get_updates(offset=None, timeout_seconds=30, limit=100)

        self.assertEqual(
            session.calls[0]["json"]["allowed_updates"],
            ["message", "callback_query"],
        )

    async def test_edit_message_text_includes_reply_markup(self) -> None:
        session = _FakeSession(
            post_responses=[
                _FakeResponse(
                    status_code=200,
                    payload={"ok": True, "result": {"message_id": 7}},
                )
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        await client.edit_message_text(
            chat_id=123,
            message_id=7,
            text="updated",
            reply_markup={"inline_keyboard": [[{"text": "Approved", "callback_data": "done"}]]},
        )

        self.assertEqual(session.calls[0]["url"].split("/")[-1], "editMessageText")
        self.assertEqual(session.calls[0]["json"]["message_id"], 7)
        self.assertEqual(
            session.calls[0]["json"]["reply_markup"],
            {"inline_keyboard": [[{"text": "Approved", "callback_data": "done"}]]},
        )

    async def test_answer_callback_query_posts_expected_payload(self) -> None:
        session = _FakeSession(
            post_responses=[
                _FakeResponse(
                    status_code=200,
                    payload={"ok": True, "result": True},
                )
            ]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        result = await client.answer_callback_query(
            callback_query_id="callback_1",
            text="Approved",
            show_alert=False,
        )

        self.assertTrue(result)
        self.assertEqual(session.calls[0]["url"].split("/")[-1], "answerCallbackQuery")
        self.assertEqual(session.calls[0]["json"]["callback_query_id"], "callback_1")
        self.assertEqual(session.calls[0]["json"]["text"], "Approved")

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

    async def test_get_updates_cancellation_propagates(self) -> None:
        session = _BlockingSession()
        client = TelegramBotAPIClient(token="token")
        client._session = session

        task = asyncio.create_task(
            client.get_updates(offset=None, timeout_seconds=30, limit=100)
        )
        await session.started.wait()

        task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_send_message_preserves_transport_error_details_without_leaking_token(
        self,
    ) -> None:
        session = _RequestErrorSession(
            httpx.RequestError(
                "network down",
                request=httpx.Request(
                    "POST",
                    "https://api.telegram.org/bot123456:secret/sendMessage",
                ),
            )
        )
        client = TelegramBotAPIClient(token="123456:secret")
        client._session = session

        with self.assertRaises(TelegramAPIError) as context:
            await client.send_message(chat_id=123, text="hi")

        self.assertIn("RequestError: network down", str(context.exception))
        self.assertNotIn("123456:secret", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, httpx.RequestError)
        self.assertEqual(str(context.exception.__cause__), "network down")

    async def test_download_file_preserves_transport_error_details_without_leaking_token(
        self,
    ) -> None:
        session = _RequestErrorSession(
            httpx.RequestError(
                "network down",
                request=httpx.Request(
                    "GET",
                    "https://api.telegram.org/file/bot123456:secret/documents/report.pdf",
                ),
            )
        )
        client = TelegramBotAPIClient(token="123456:secret")
        client._session = session

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TelegramAPIError) as context:
                await client.download_file(
                    remote_file_path="documents/report.pdf",
                    destination_path=Path(tmp) / "report.pdf",
                )

        self.assertIn("RequestError: network down", str(context.exception))
        self.assertNotIn("123456:secret", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, httpx.RequestError)
        self.assertEqual(str(context.exception.__cause__), "network down")

    async def test_send_document_preserves_transport_error_details_without_leaking_token(
        self,
    ) -> None:
        session = _RequestErrorSession(
            httpx.RequestError(
                "network down",
                request=httpx.Request(
                    "POST",
                    "https://api.telegram.org/bot123456:secret/sendDocument",
                ),
            )
        )
        client = TelegramBotAPIClient(token="123456:secret")
        client._session = session

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "report.txt"
            file_path.write_text("hello", encoding="utf-8")

            with self.assertRaises(TelegramAPIError) as context:
                await client.send_document(chat_id=123, file_path=file_path)

        self.assertIn("RequestError: network down", str(context.exception))
        self.assertNotIn("123456:secret", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, httpx.RequestError)
        self.assertEqual(str(context.exception.__cause__), "network down")
