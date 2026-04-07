"""Route-scoped Codex client coordination shared by main and subagent actors."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING

from jarvis.logging_setup import get_application_logger

from .auth import CodexRouteAuthenticator
from .client import CodexClient
from .config import CodexBackendSettings
from .types import CodexAuthChallenge, CodexProtocolError

if TYPE_CHECKING:
    from .actor_runtime import CodexActorRuntime

LOGGER = get_application_logger(__name__)


class CodexRouteCoordinator:
    """Owns one route-scoped Codex connection and dispatches by thread id."""

    def __init__(
        self,
        *,
        settings: CodexBackendSettings,
        client_factory: Callable[[str, object, object], CodexClient] | None = None,
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory
        self._client: CodexClient | None = None
        self._client_lock = asyncio.Lock()
        self._actors_by_thread_id: dict[str, CodexActorRuntime] = {}
        self._authenticator: CodexRouteAuthenticator | None = None

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        self._authenticator = None
        self._actors_by_thread_id.clear()
        if client is not None:
            await client.aclose()

    async def ensure_client(self) -> CodexClient:
        client = self._client
        if client is not None:
            return client
        async with self._client_lock:
            client = self._client
            if client is not None:
                return client
            factory = self._client_factory or self._default_client_factory
            client = factory(
                self._settings.ws_url,
                self._handle_notification,
                self._handle_server_request,
            )
            await client.connect()
            self._client = client
            self._authenticator = CodexRouteAuthenticator(client=client)
            return client

    async def request(self, method: str, params: dict[str, Any] | None) -> object:
        client = await self.ensure_client()
        return await client.request(method, params)

    async def notify(self, method: str, params: dict[str, Any] | None) -> None:
        client = await self.ensure_client()
        await client.notify(method, params)

    async def ensure_authenticated(
        self,
        *,
        on_challenge: Callable[[CodexAuthChallenge], Awaitable[None]],
    ) -> None:
        await self.ensure_client()
        authenticator = self._authenticator
        if authenticator is None:
            raise CodexProtocolError("Codex authenticator is unavailable.")
        await authenticator.ensure_authenticated(on_challenge=on_challenge)

    def register_actor(self, *, thread_id: str, actor: "CodexActorRuntime") -> None:
        self._actors_by_thread_id[thread_id] = actor

    def unregister_actor(self, *, thread_id: str, actor: "CodexActorRuntime") -> None:
        current = self._actors_by_thread_id.get(thread_id)
        if current is actor:
            self._actors_by_thread_id.pop(thread_id, None)

    async def _handle_notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "account/login/completed":
            authenticator = self._authenticator
            if authenticator is None:
                return
            await authenticator.handle_login_completed(
                login_id=(
                    str(params.get("loginId"))
                    if params.get("loginId") is not None
                    else None
                ),
                success=bool(params.get("success", False)),
                error=(
                    str(params.get("error"))
                    if params.get("error") is not None
                    else None
                ),
            )
            return
        thread_id = _thread_id_from_payload(params)
        if thread_id is None:
            return
        actor = self._actors_by_thread_id.get(thread_id)
        if actor is None:
            LOGGER.debug(
                "Ignoring Codex notification %s for unregistered thread %s.",
                method,
                thread_id,
            )
            return
        await actor.handle_notification(method, params)

    async def _handle_server_request(self, method: str, params: dict[str, Any]) -> object:
        thread_id = _thread_id_from_payload(params)
        if thread_id is None:
            raise CodexProtocolError(
                f"Codex server request '{method}' did not include a threadId."
            )
        actor = self._actors_by_thread_id.get(thread_id)
        if actor is None:
            raise CodexProtocolError(
                f"Codex server request '{method}' targeted unknown thread '{thread_id}'."
            )
        return await actor.handle_server_request(method, params)

    def _default_client_factory(
        self,
        ws_url: str,
        notification_handler,
        server_request_handler,
    ) -> CodexClient:
        return CodexClient(
            ws_url=ws_url,
            on_notification=notification_handler,
            on_server_request=server_request_handler,
            ws_bearer_token=self._settings.ws_bearer_token,
        )


def _thread_id_from_payload(payload: dict[str, Any]) -> str | None:
    raw_thread_id = payload.get("threadId")
    if raw_thread_id is None:
        return None
    thread_id = str(raw_thread_id).strip()
    return thread_id or None
