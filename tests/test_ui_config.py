"""Unit tests for UI runtime settings."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ui.telegram.config import UIConfigurationError, UISettings


class UISettingsTests(unittest.TestCase):
    def test_requires_telegram_token(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(UIConfigurationError):
                UISettings.from_env()

    def test_derives_gateway_ws_url_from_gateway_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_GATEWAY_HOST": "0.0.0.0",
                "JARVIS_GATEWAY_PORT": "8181",
                "JARVIS_GATEWAY_WS_PATH": "/socket",
            },
            clear=True,
        ):
            settings = UISettings.from_env()
        self.assertEqual(settings.gateway_ws_base_url, "ws://127.0.0.1:8181/socket")

    def test_accepts_explicit_ui_gateway_ws_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_GATEWAY_WS_BASE_URL": "wss://example.com/ws",
            },
            clear=True,
        ):
            settings = UISettings.from_env()
        self.assertEqual(settings.gateway_ws_base_url, "wss://example.com/ws")

    def test_rejects_invalid_poll_limit(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_TELEGRAM_POLL_LIMIT": "200",
            },
            clear=True,
        ):
            with self.assertRaises(UIConfigurationError):
                UISettings.from_env()
