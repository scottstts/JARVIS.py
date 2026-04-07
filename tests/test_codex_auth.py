"""Unit tests for Codex route authentication flow."""

from __future__ import annotations

import asyncio
import unittest

from jarvis.codex_backend.auth import CodexRouteAuthenticator


class _FakeCodexClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[str, object]] = []

    async def request(self, method: str, params: object) -> object:
        self.requests.append((method, params))
        if not self._responses:
            raise AssertionError(f"Unexpected request with no queued response: {method}")
        return self._responses.pop(0)


class CodexRouteAuthenticatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_login_flow_emits_auth_challenge_and_waits_for_completion(self) -> None:
        client = _FakeCodexClient(
            responses=[
                {"account": None, "requiresOpenaiAuth": True},
                {
                    "type": "chatgpt",
                    "loginId": "login_1",
                    "authUrl": "https://auth.example/login",
                },
                {"account": {"email": "user@example.com"}, "requiresOpenaiAuth": True},
            ]
        )
        authenticator = CodexRouteAuthenticator(client=client)
        challenges: list[tuple[str, str]] = []

        async def on_challenge(challenge) -> None:
            challenges.append((challenge.login_id, challenge.auth_url))

        task = asyncio.create_task(
            authenticator.ensure_authenticated(on_challenge=on_challenge)
        )
        await asyncio.sleep(0)
        await authenticator.handle_login_completed(
            login_id="login_1",
            success=True,
            error=None,
        )
        await task

        self.assertEqual(
            challenges,
            [("login_1", "https://auth.example/login")],
        )
        self.assertEqual(
            client.requests,
            [
                ("account/read", {"refreshToken": False}),
                ("account/login/start", {"type": "chatgpt"}),
                ("account/read", {"refreshToken": True}),
            ],
        )

