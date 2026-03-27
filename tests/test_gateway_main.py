"""Tests for the standalone gateway entrypoint."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import jarvis.gateway.__main__ as gateway_main
from jarvis.gateway import GatewaySettings


class GatewayMainEntrypointTests(unittest.TestCase):
    def test_main_disables_access_logs(self) -> None:
        settings = GatewaySettings(
            host="127.0.0.1",
            port=8080,
            websocket_path="/ws",
            max_message_chars=32_000,
        )

        with patch("jarvis.gateway.__main__.GatewaySettings.from_env", return_value=settings):
            with patch("jarvis.gateway.__main__.uvicorn.run") as run:
                gateway_main.main()

        run.assert_called_once_with(
            "jarvis.gateway.app:build_asgi_app",
            factory=True,
            host="127.0.0.1",
            port=8080,
            access_log=False,
        )
