"""Tests for shared application logging setup."""

from __future__ import annotations

import logging
import sys
import unittest
from unittest.mock import patch

from logging_setup import (
    CollapseTelegramDraftRequestFilter,
    SensitiveDataFormatter,
    _redact_sensitive_text,
    configure_application_logging,
)

_FAKE_TELEGRAM_BOT_TOKEN = "1234567890:TEST_TOKEN_VALUE_FOR_REDACTION_ABCDE"


class LoggingSetupTests(unittest.TestCase):
    def test_collapse_telegram_draft_request_filter_suppresses_consecutive_drafts(self) -> None:
        filter_ = CollapseTelegramDraftRequestFilter()

        first_draft = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=(
                'HTTP Request: POST '
                'https://api.telegram.org/bot[REDACTED]/sendMessageDraft '
                '"HTTP/1.1 200 OK"'
            ),
            args=(),
            exc_info=None,
        )
        second_draft = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=2,
            msg=(
                'HTTP Request: POST '
                'https://api.telegram.org/bot[REDACTED]/sendMessageDraft '
                '"HTTP/1.1 200 OK"'
            ),
            args=(),
            exc_info=None,
        )
        provider_request = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=3,
            msg=(
                'HTTP Request: POST '
                'https://api.openai.com/v1/responses '
                '"HTTP/1.1 200 OK"'
            ),
            args=(),
            exc_info=None,
        )
        next_draft = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=4,
            msg=(
                'HTTP Request: POST '
                'https://api.telegram.org/bot[REDACTED]/sendMessageDraft '
                '"HTTP/1.1 200 OK"'
            ),
            args=(),
            exc_info=None,
        )

        self.assertTrue(filter_.filter(first_draft))
        self.assertFalse(filter_.filter(second_draft))
        self.assertTrue(filter_.filter(provider_request))
        self.assertTrue(filter_.filter(next_draft))

    def test_collapse_telegram_draft_request_filter_keeps_same_decision_for_same_record(self) -> None:
        filter_ = CollapseTelegramDraftRequestFilter()
        draft = logging.LogRecord(
            name="httpx",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=(
                'HTTP Request: POST '
                'https://api.telegram.org/bot[REDACTED]/sendMessageDraft '
                '"HTTP/1.1 200 OK"'
            ),
            args=(),
            exc_info=None,
        )

        self.assertTrue(filter_.filter(draft))
        self.assertTrue(filter_.filter(draft))

    def test_configure_application_logging_keeps_http_client_loggers_enabled(self) -> None:
        httpx_logger = logging.getLogger("httpx")
        httpcore_logger = logging.getLogger("httpcore")
        original_httpx_level = httpx_logger.level
        original_httpcore_level = httpcore_logger.level

        try:
            httpx_logger.setLevel(logging.INFO)
            httpcore_logger.setLevel(logging.INFO)

            with patch("logging_setup.logging.basicConfig") as basic_config:
                configure_application_logging()

            basic_config.assert_called_once_with(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
            self.assertEqual(httpx_logger.level, logging.INFO)
            self.assertEqual(httpcore_logger.level, logging.INFO)
        finally:
            httpx_logger.setLevel(original_httpx_level)
            httpcore_logger.setLevel(original_httpcore_level)

    def test_configure_application_logging_installs_draft_filter_and_formatter(self) -> None:
        root_logger = logging.getLogger()
        handler = logging.StreamHandler()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers = [handler]

        try:
            with patch("logging_setup.logging.basicConfig") as basic_config:
                configure_application_logging()

            basic_config.assert_called_once_with(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
            self.assertIsInstance(handler.formatter, SensitiveDataFormatter)
            self.assertTrue(
                any(
                    isinstance(filter_, CollapseTelegramDraftRequestFilter)
                    for filter_ in handler.filters
                )
            )
        finally:
            root_logger.handlers = original_handlers

    def test_redact_sensitive_text_hides_telegram_bot_token(self) -> None:
        text = (
            'HTTP Request: POST '
            f"https://api.telegram.org/bot{_FAKE_TELEGRAM_BOT_TOKEN}/getMe "
            '"HTTP/1.1 200 OK"'
        )

        redacted = _redact_sensitive_text(text)

        self.assertNotIn(_FAKE_TELEGRAM_BOT_TOKEN, redacted)
        self.assertIn("https://api.telegram.org/bot[REDACTED]/getMe", redacted)

    def test_redact_sensitive_text_preserves_ai_provider_request_urls(self) -> None:
        text = (
            'HTTP Request: POST '
            'https://api.openai.com/v1/responses '
            '"HTTP/1.1 200 OK"'
        )

        self.assertEqual(_redact_sensitive_text(text), text)

    def test_sensitive_data_formatter_redacts_exception_text(self) -> None:
        formatter = SensitiveDataFormatter("%(levelname)s %(message)s")
        try:
            raise RuntimeError(f"telegram token leaked: {_FAKE_TELEGRAM_BOT_TOKEN}")
        except RuntimeError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname=__file__,
                lineno=1,
                msg="request failed",
                args=(),
                exc_info=exc_info,
            )

        rendered = formatter.format(record)

        self.assertNotIn(_FAKE_TELEGRAM_BOT_TOKEN, rendered)
        self.assertIn("[REDACTED]", rendered)
