"""Low-level WebSocket JSON-RPC client for Codex app-server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlparse

from jarvis.logging_setup import get_application_logger

from .types import CodexConfigurationError, CodexConnectionError, CodexProtocolError

LOGGER = get_application_logger(__name__)

NotificationHandler = Callable[[str, dict[str, Any]], Awaitable[None]]
ServerRequestHandler = Callable[[str, dict[str, Any]], Awaitable[object]]


def _resolve_websockets():
    try:
        import websockets
    except ModuleNotFoundError as exc:
        raise CodexProtocolError(
            "Python package 'websockets' is required for the Codex backend."
        ) from exc
    return websockets


class CodexClient:
    """Maintains one Codex app-server JSON-RPC connection."""

    def __init__(
        self,
        *,
        ws_url: str,
        on_notification: NotificationHandler,
        on_server_request: ServerRequestHandler,
        client_name: str = "Jarvis",
        client_version: str = "0",
        ws_bearer_token: str | None = None,
    ) -> None:
        self._ws_url = ws_url
        self._on_notification = on_notification
        self._on_server_request = on_server_request
        self._client_name = client_name
        self._client_version = client_version
        self._ws_bearer_token = ws_bearer_token
        self._connection: Any | None = None
        self._socket: Any | None = None
        self._send_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._pending_requests: dict[int, asyncio.Future[object]] = {}
        self._next_request_id = 1

    async def connect(self) -> None:
        if self._socket is not None:
            return
        websockets = _resolve_websockets()
        try:
            connection = websockets.connect(
                self._ws_url,
                **self._connection_kwargs(),
            )
            socket = await connection.__aenter__()
        except websockets.InvalidURI as exc:
            raise CodexConfigurationError(
                f"JARVIS_CODEX_WS_URL is invalid: {self._ws_url}"
            ) from exc
        except websockets.InvalidStatus as exc:
            raise self._build_handshake_error(
                exc=exc,
                status_code=exc.response.status_code,
            ) from exc
        except websockets.InvalidHandshake as exc:
            raise self._build_handshake_error(exc=exc) from exc
        except (OSError, TimeoutError) as exc:
            raise self._build_transport_error(exc) from exc
        self._connection = connection
        self._socket = socket
        initialize_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "clientInfo": {
                    "name": self._client_name,
                    "version": self._client_version,
                },
                "capabilities": {
                    "experimentalApi": True,
                },
            },
        }
        self._next_request_id = 2
        try:
            async with self._send_lock:
                await socket.send(json.dumps(initialize_payload))
            initialize_message = await _recv_json(socket)
        except asyncio.CancelledError:
            raise
        except (websockets.ConnectionClosedError, OSError, TimeoutError) as exc:
            await self.aclose()
            raise self._build_transport_error(exc) from exc
        if initialize_message.get("id") != 1:
            await self.aclose()
            raise CodexProtocolError("Codex initialize response id did not match the request.")
        if initialize_message.get("error") is not None:
            error = initialize_message["error"]
            if isinstance(error, dict):
                message = str(error.get("message", "Codex initialize failed."))
            else:
                message = "Codex initialize failed."
            await self.aclose()
            raise CodexProtocolError(message)
        initialize_response = initialize_message.get("result")
        if not isinstance(initialize_response, dict):
            await self.aclose()
            raise CodexProtocolError("Codex initialize response must be a JSON object.")
        await self._send_notification_locked("initialized", None)
        self._reader_task = asyncio.create_task(
            self._reader_loop(),
            name="jarvis-codex-reader",
        )

    def _connection_kwargs(self) -> dict[str, object]:
        headers = self._authorization_headers()
        if headers is None:
            return {}
        return {"additional_headers": headers}

    def _authorization_headers(self) -> dict[str, str] | None:
        token = self._ws_bearer_token
        if token is None:
            return None
        return {"Authorization": f"Bearer {token}"}

    def _build_transport_error(self, exc: Exception) -> CodexConnectionError:
        parts = [f"Could not connect to Codex app-server at {self._ws_url}."]
        parts.extend(self._connectivity_hints())
        parts.append(f"Underlying error: {exc}")
        return CodexConnectionError(" ".join(parts))

    def _build_handshake_error(
        self,
        *,
        exc: Exception,
        status_code: int | None = None,
    ) -> CodexConnectionError:
        if status_code is None:
            parts = [f"Codex app-server at {self._ws_url} rejected the websocket handshake."]
        else:
            parts = [
                (
                    f"Codex app-server at {self._ws_url} rejected the websocket handshake "
                    f"with HTTP {status_code}."
                )
            ]
        parts.extend(self._connectivity_hints())
        parts.append(f"Underlying error: {exc}")
        return CodexConnectionError(" ".join(parts))

    def _connectivity_hints(self) -> list[str]:
        hints: list[str] = []
        if self._is_non_loopback_target():
            hints.append(
                "If Jarvis is running in Docker, a host app-server bound only to "
                "127.0.0.1 is not reachable from jarvis_runtime."
            )
            hints.append(
                "Run Codex app-server on a host-reachable websocket listener instead."
            )
            if self._ws_bearer_token is None:
                hints.append(
                    "For host listeners started with `codex app-server --ws-auth "
                    "capability-token`, set the same token in `JARVIS_CODEX_WS_BEARER_TOKEN`."
                )
        return hints

    def _is_non_loopback_target(self) -> bool:
        host = urlparse(self._ws_url).hostname
        if host is None:
            return False
        return host not in {"127.0.0.1", "::1", "localhost"}

    async def aclose(self) -> None:
        reader_task = self._reader_task
        self._reader_task = None
        if reader_task is not None:
            reader_task.cancel()
            try:
                await reader_task
            except asyncio.CancelledError:
                pass
        for future in tuple(self._pending_requests.values()):
            if not future.done():
                future.set_exception(CodexProtocolError("Codex connection closed."))
        self._pending_requests.clear()
        connection = self._connection
        socket = self._socket
        self._connection = None
        self._socket = None
        if connection is not None:
            try:
                await connection.__aexit__(None, None, None)
            except Exception:
                LOGGER.debug("Codex websocket close via context manager failed.", exc_info=True)
        elif socket is not None:
            try:
                await socket.close()
            except Exception:
                LOGGER.debug("Codex websocket direct close failed.", exc_info=True)

    async def request(self, method: str, params: dict[str, Any] | None) -> object:
        await self.connect()
        async with self._request_lock:
            return await self._send_request_locked(method, params)

    async def notify(self, method: str, params: dict[str, Any] | None) -> None:
        await self.connect()
        async with self._send_lock:
            await self._send_notification_locked(method, params)

    async def _send_request_locked(self, method: str, params: dict[str, Any] | None) -> object:
        socket = self._socket
        if socket is None:
            raise CodexProtocolError("Codex websocket is not connected.")
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[object] = asyncio.get_running_loop().create_future()
        self._pending_requests[request_id] = future
        async with self._send_lock:
            await socket.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    }
                )
            )
        return await future

    async def _send_notification_locked(
        self,
        method: str,
        params: dict[str, Any] | None,
    ) -> None:
        socket = self._socket
        if socket is None:
            raise CodexProtocolError("Codex websocket is not connected.")
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        await socket.send(json.dumps(payload))

    async def _reader_loop(self) -> None:
        socket = self._socket
        if socket is None:
            return
        try:
            while True:
                payload = await _recv_json(socket)
                if "id" in payload and "method" not in payload:
                    self._resolve_response(payload)
                    continue
                if "id" in payload and "method" in payload:
                    await self._handle_server_request(payload)
                    continue
                method = payload.get("method")
                params = payload.get("params", {})
                if not isinstance(method, str) or not isinstance(params, dict):
                    raise CodexProtocolError("Codex notification payload is invalid.")
                await self._on_notification(method, params)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("Codex websocket reader failed.")
            self._fail_all_pending(CodexProtocolError("Codex websocket reader failed."))
        finally:
            self._socket = None
            self._connection = None

    def _resolve_response(self, payload: dict[str, Any]) -> None:
        raw_id = payload.get("id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise CodexProtocolError("Codex response id must be an integer.")
        future = self._pending_requests.pop(raw_id, None)
        if future is None:
            return
        if "error" in payload and payload["error"] is not None:
            error = payload["error"]
            if isinstance(error, dict):
                message = str(error.get("message", "Codex request failed."))
            else:
                message = "Codex request failed."
            future.set_exception(CodexProtocolError(message))
            return
        future.set_result(payload.get("result"))

    async def _handle_server_request(self, payload: dict[str, Any]) -> None:
        raw_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params", {})
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise CodexProtocolError("Codex server request id must be an integer.")
        if not isinstance(method, str) or not isinstance(params, dict):
            raise CodexProtocolError("Codex server request payload is invalid.")
        try:
            result = await self._on_server_request(method, params)
            response = {
                "jsonrpc": "2.0",
                "id": raw_id,
                "result": result,
            }
        except Exception as exc:
            LOGGER.exception("Codex server request handler failed for %s.", method)
            response = {
                "jsonrpc": "2.0",
                "id": raw_id,
                "error": {
                    "code": -32000,
                    "message": str(exc) or type(exc).__name__,
                },
            }
        socket = self._socket
        if socket is None:
            return
        async with self._send_lock:
            await socket.send(json.dumps(response))

    def _fail_all_pending(self, exc: Exception) -> None:
        for future in tuple(self._pending_requests.values()):
            if future.done():
                continue
            future.set_exception(exc)
        self._pending_requests.clear()


async def _recv_json(socket: Any) -> dict[str, Any]:
    raw_payload = await socket.recv()
    if not isinstance(raw_payload, str):
        raise CodexProtocolError("Codex websocket payload must be a JSON text frame.")
    try:
        parsed = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise CodexProtocolError("Codex websocket payload was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise CodexProtocolError("Codex websocket payload must be a JSON object.")
    return parsed
