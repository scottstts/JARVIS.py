"""Shared logging configuration for Jarvis entrypoints."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
import logging
import re
import threading

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_TELEGRAM_BOT_TOKEN_PATTERN = re.compile(r"\d{6,}:[A-Za-z0-9_-]{20,}")
_SUPPRESS_INTERNAL_HTTPX_REQUEST_LOGS: ContextVar[bool] = ContextVar(
    "jarvis_suppress_internal_httpx_request_logs",
    default=False,
)


def _redact_sensitive_text(text: str) -> str:
    return _TELEGRAM_BOT_TOKEN_PATTERN.sub("[REDACTED]", text)


class SensitiveDataFormatter(logging.Formatter):
    """Formatter that redacts credentials from rendered log output."""

    def format(self, record: logging.LogRecord) -> str:
        return _redact_sensitive_text(super().format(record))


class CollapseTelegramDraftRequestFilter(logging.Filter):
    """Suppresses consecutive Telegram draft request logs after the first line."""

    def __init__(self) -> None:
        super().__init__()
        self._draft_streak_active = False
        self._last_record_id: int | None = None
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        with self._lock:
            current_record_id = id(record)
            if self._last_record_id == current_record_id:
                return bool(getattr(record, "_jarvis_allow_record", True))

            message = record.getMessage()
            is_draft_request = (
                record.name == "httpx"
                and "api.telegram.org" in message
                and "/sendMessageDraft" in message
            )

            if is_draft_request:
                allow = not self._draft_streak_active
                self._draft_streak_active = True
            else:
                allow = True
                self._draft_streak_active = False

            self._last_record_id = current_record_id
            setattr(record, "_jarvis_allow_record", allow)
            return allow


class SuppressInternalHttpxRequestFilter(logging.Filter):
    """Hide HTTP transport chatter for Jarvis's own container RPC calls."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.name == "httpx"
            and record.levelno <= logging.INFO
            and _SUPPRESS_INTERNAL_HTTPX_REQUEST_LOGS.get()
        )


@contextmanager
def suppress_internal_httpx_request_logs() -> Iterator[None]:
    """Suppress routine httpx INFO records inside an internal RPC operation."""

    token = _SUPPRESS_INTERNAL_HTTPX_REQUEST_LOGS.set(True)
    try:
        yield
    finally:
        _SUPPRESS_INTERNAL_HTTPX_REQUEST_LOGS.reset(token)


def get_application_logger(module_name: str) -> logging.Logger:
    """Preserve pre-refactor logger names after moving into the jarvis package."""

    return logging.getLogger(module_name.removeprefix("jarvis."))


def configure_application_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=_LOG_FORMAT,
    )

    root_logger = logging.getLogger()
    draft_filter = CollapseTelegramDraftRequestFilter()
    internal_httpx_filter = SuppressInternalHttpxRequestFilter()
    for handler in root_logger.handlers:
        handler.setFormatter(SensitiveDataFormatter(_LOG_FORMAT))
        handler.addFilter(draft_filter)
        handler.addFilter(internal_httpx_filter)
