"""Unit tests for Codex websocket client transport behavior."""

from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jarvis.codex_backend.client import CodexClient
from jarvis.codex_backend.types import CodexConnectionError
from websockets import InvalidStatus
from websockets.datastructures import Headers
from websockets.http11 import Response


class _DummyInvalidURI(Exception):
    pass


class _DummyInvalidHandshake(Exception):
    pass


class _DummyConnectionClosedError(Exception):
    pass


class _FakeSocket:
    def __init__(self, incoming: list[str]) -> None:
        self._incoming = list(incoming)
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def recv(self) -> str:
        if self._incoming:
            return self._incoming.pop(0)
        await asyncio.Future()

    async def close(self) -> None:
        return None


class _FakeConnection:
    def __init__(self, socket: _FakeSocket) -> None:
        self._socket = socket

    async def __aenter__(self) -> _FakeSocket:
        return self._socket

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        _ = (exc_type, exc, tb)
        return False


class CodexClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_sends_bearer_authorization_header_when_configured(self) -> None:
        captured: dict[str, object] = {}
        socket = _FakeSocket(incoming=[json.dumps({"id": 1, "result": {"ok": True}})])

        def _connect(uri: str, **kwargs):
            captured["uri"] = uri
            captured["kwargs"] = kwargs
            return _FakeConnection(socket)

        websockets_stub = SimpleNamespace(
            connect=_connect,
            InvalidURI=_DummyInvalidURI,
            InvalidStatus=InvalidStatus,
            InvalidHandshake=_DummyInvalidHandshake,
            ConnectionClosedError=_DummyConnectionClosedError,
        )

        client = CodexClient(
            ws_url="ws://host.docker.internal:4500",
            on_notification=_noop_notification,
            on_server_request=_noop_server_request,
            ws_bearer_token="secret-token",
        )

        with patch("jarvis.codex_backend.client._resolve_websockets", return_value=websockets_stub):
            await client.connect()
            await client.aclose()

        self.assertEqual(captured["uri"], "ws://host.docker.internal:4500")
        kwargs = captured["kwargs"]
        self.assertIsInstance(kwargs, dict)
        self.assertEqual(
            kwargs["additional_headers"],
            {"Authorization": "Bearer secret-token"},
        )

    async def test_connect_maps_transport_failures_to_codex_connection_error(self) -> None:
        def _connect(_uri: str, **_kwargs):
            raise OSError("connect failed")

        websockets_stub = SimpleNamespace(
            connect=_connect,
            InvalidURI=_DummyInvalidURI,
            InvalidStatus=InvalidStatus,
            InvalidHandshake=_DummyInvalidHandshake,
            ConnectionClosedError=_DummyConnectionClosedError,
        )

        client = CodexClient(
            ws_url="ws://host.docker.internal:4500",
            on_notification=_noop_notification,
            on_server_request=_noop_server_request,
        )

        with patch("jarvis.codex_backend.client._resolve_websockets", return_value=websockets_stub):
            with self.assertRaises(CodexConnectionError) as context:
                await client.connect()

        message = str(context.exception)
        self.assertIn("ws://host.docker.internal:4500", message)
        self.assertIn("127.0.0.1", message)
        self.assertIn("JARVIS_CODEX_WS_BEARER_TOKEN", message)

    async def test_connect_maps_unauthorized_handshake_to_codex_connection_error(self) -> None:
        def _connect(_uri: str, **_kwargs):
            raise InvalidStatus(
                Response(
                    status_code=401,
                    reason_phrase="Unauthorized",
                    headers=Headers(),
                )
            )

        websockets_stub = SimpleNamespace(
            connect=_connect,
            InvalidURI=_DummyInvalidURI,
            InvalidStatus=InvalidStatus,
            InvalidHandshake=_DummyInvalidHandshake,
            ConnectionClosedError=_DummyConnectionClosedError,
        )

        client = CodexClient(
            ws_url="ws://host.docker.internal:4500",
            on_notification=_noop_notification,
            on_server_request=_noop_server_request,
        )

        with patch("jarvis.codex_backend.client._resolve_websockets", return_value=websockets_stub):
            with self.assertRaises(CodexConnectionError) as context:
                await client.connect()

        message = str(context.exception)
        self.assertIn("HTTP 401", message)
        self.assertIn("JARVIS_CODEX_WS_BEARER_TOKEN", message)


async def _noop_notification(_method: str, _params: dict[str, object]) -> None:
    return None


async def _noop_server_request(_method: str, _params: dict[str, object]) -> object:
    return {}
