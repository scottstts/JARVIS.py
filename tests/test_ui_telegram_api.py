"""Unit tests for Telegram Bot API client behavior."""

from __future__ import annotations

import unittest

from ui.telegram.api import DraftMessage, TelegramAPIError, TelegramBotAPIClient


class _FakeResponse:
    def __init__(self, *, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, object], float]] = []

    def post(self, url: str, json: dict[str, object], timeout: float) -> _FakeResponse:
        self.calls.append((url, json, timeout))
        if not self._responses:
            raise AssertionError("No fake response available.")
        return self._responses.pop(0)

    def close(self) -> None:
        return None


class TelegramBotAPIClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_message_includes_parse_mode(self) -> None:
        session = _FakeSession(
            [_FakeResponse(status_code=200, payload={"ok": True, "result": {"message_id": 1}})]
        )
        client = TelegramBotAPIClient(token="token")
        client._session = session

        await client.send_message(chat_id=123, text="<b>hi</b>", parse_mode="HTML")

        self.assertEqual(session.calls[0][1]["parse_mode"], "HTML")

    async def test_send_message_draft_includes_parse_mode(self) -> None:
        session = _FakeSession([_FakeResponse(status_code=200, payload={"ok": True, "result": True})])
        client = TelegramBotAPIClient(token="token")
        client._session = session

        await client.send_message_draft(
            chat_id=123,
            draft=DraftMessage(id=7, text="<b>hi</b>"),
            parse_mode="HTML",
        )

        self.assertEqual(session.calls[0][1]["parse_mode"], "HTML")

    async def test_raises_retry_after_on_rate_limit(self) -> None:
        session = _FakeSession(
            [
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
