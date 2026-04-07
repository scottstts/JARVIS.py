"""Route-scoped ChatGPT OAuth coordination for the Codex backend."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from .client import CodexClient
from .types import CodexAuthChallenge, CodexAuthenticationError, CodexProtocolError

AuthChallengeCallback = Callable[[CodexAuthChallenge], Awaitable[None]]


class CodexRouteAuthenticator:
    """Serializes first-run browser login across one route connection."""

    def __init__(self, *, client: CodexClient) -> None:
        self._client = client
        self._lock = asyncio.Lock()
        self._pending_login_future: asyncio.Future[None] | None = None
        self._pending_login_id: str | None = None

    async def ensure_authenticated(
        self,
        *,
        on_challenge: AuthChallengeCallback,
    ) -> None:
        account = await self._client.request("account/read", {"refreshToken": False})
        if _account_ready(account):
            return

        async with self._lock:
            if self._pending_login_future is None or self._pending_login_future.done():
                response = await self._client.request(
                    "account/login/start",
                    {"type": "chatgpt"},
                )
                login = _parse_login_response(response)
                self._pending_login_id = login.login_id
                self._pending_login_future = asyncio.get_running_loop().create_future()
                await on_challenge(login)
            pending = self._pending_login_future

        if pending is None:
            raise CodexAuthenticationError("Codex login did not start correctly.")
        await pending

        refreshed = await self._client.request("account/read", {"refreshToken": True})
        if not _account_ready(refreshed):
            raise CodexAuthenticationError("Codex login completed but no authenticated account is available.")

    async def handle_login_completed(
        self,
        *,
        login_id: str | None,
        success: bool,
        error: str | None,
    ) -> None:
        async with self._lock:
            pending = self._pending_login_future
            expected_login_id = self._pending_login_id
            if pending is None or pending.done():
                return
            if expected_login_id is not None and login_id not in {None, expected_login_id}:
                return
            self._pending_login_future = None
            self._pending_login_id = None
            if success:
                pending.set_result(None)
                return
            pending.set_exception(
                CodexAuthenticationError(error or "Codex browser login failed.")
            )


def _account_ready(response: object) -> bool:
    if not isinstance(response, dict):
        return False
    if response.get("requiresOpenaiAuth") is False:
        return True
    return response.get("account") is not None


def _parse_login_response(response: object) -> CodexAuthChallenge:
    if not isinstance(response, dict):
        raise CodexProtocolError("Codex login response must be a JSON object.")
    if response.get("type") != "chatgpt":
        raise CodexProtocolError("Jarvis only supports Codex ChatGPT OAuth login flow.")
    login_id = str(response.get("loginId", "")).strip()
    auth_url = str(response.get("authUrl", "")).strip()
    if not login_id or not auth_url:
        raise CodexProtocolError("Codex ChatGPT login response is missing loginId or authUrl.")
    return CodexAuthChallenge(login_id=login_id, auth_url=auth_url)

