"""Unit tests for UI runtime settings."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
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
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "777",
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
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "777",
                "JARVIS_UI_GATEWAY_WS_BASE_URL": "wss://example.com/ws",
            },
            clear=True,
        ):
            settings = UISettings.from_env()
        self.assertEqual(settings.gateway_ws_base_url, "wss://example.com/ws")

    def test_requires_allowed_telegram_user_id(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
            },
            clear=True,
        ):
            with self.assertRaises(UIConfigurationError):
                UISettings.from_env()

    def test_reads_allowed_telegram_user_id(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "8793811805",
            },
            clear=True,
        ):
            settings = UISettings.from_env()
        self.assertEqual(settings.telegram_allowed_user_id, 8793811805)

    def test_rejects_blank_allowed_telegram_user_id(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "  ",
            },
            clear=True,
        ):
            with self.assertRaises(UIConfigurationError):
                UISettings.from_env()

    def test_defaults_telegram_temp_dir_from_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "777",
            },
            clear=True,
        ):
            settings = UISettings.from_env()
        self.assertEqual(settings.telegram_temp_dir, Path("/workspace/temp"))

    def test_requires_agent_workspace_for_host_runs(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "777",
            },
            clear=True,
        ), patch("workspace_paths._running_in_container", return_value=False):
            with self.assertRaisesRegex(
                UIConfigurationError,
                "AGENT_WORKSPACE must be explicitly set for host runs",
            ):
                UISettings.from_env()

    def test_uses_explicit_agent_workspace_for_telegram_temp_dir_default(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "777",
                "AGENT_WORKSPACE": "/tmp/jarvis-host-workspace",
            },
            clear=True,
        ), patch("workspace_paths._running_in_container", return_value=False):
            settings = UISettings.from_env()
        self.assertEqual(settings.telegram_temp_dir, Path("/tmp/jarvis-host-workspace/temp"))

    def test_accepts_explicit_telegram_temp_dir(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "777",
                "JARVIS_UI_TELEGRAM_TEMP_DIR": "/tmp/jarvis-telegram",
            },
            clear=True,
        ):
            settings = UISettings.from_env()
        self.assertEqual(settings.telegram_temp_dir, Path("/tmp/jarvis-telegram"))

    def test_rejects_invalid_poll_limit(self) -> None:
        with patch.dict(
            os.environ,
            {
                "TELEGRAM_TOKEN": "token",
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID": "777",
                "JARVIS_UI_TELEGRAM_POLL_LIMIT": "200",
            },
            clear=True,
        ):
            with self.assertRaises(UIConfigurationError):
                UISettings.from_env()

    def test_repr_hides_telegram_token_and_user_id(self) -> None:
        settings = UISettings(
            telegram_token="123456:abcdef-secret",
            telegram_allowed_user_id=8793811805,
        )

        rendered = repr(settings)

        self.assertNotIn("123456:abcdef-secret", rendered)
        self.assertNotIn("8793811805", rendered)
