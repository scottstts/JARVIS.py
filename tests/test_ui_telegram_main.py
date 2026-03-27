"""Tests for the standalone Telegram UI entrypoint."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import jarvis.ui.telegram.__main__ as telegram_main


class TelegramMainEntrypointTests(unittest.TestCase):
    def test_main_configures_logging_before_starting_ui(self) -> None:
        async def fake_run_telegram_ui() -> None:
            return None

        with patch("jarvis.ui.telegram.__main__.configure_application_logging") as configure_logging:
            with patch("jarvis.ui.telegram.__main__.run_telegram_ui", side_effect=fake_run_telegram_ui):
                telegram_main.main()

        configure_logging.assert_called_once_with()
