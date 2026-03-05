"""Runtime settings for the Telegram UI bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass


class UIConfigurationError(ValueError):
    """Raised when UI runtime settings are invalid."""


def _required_env(name: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        raise UIConfigurationError(f"{name} must be explicitly set in the environment.")
    return raw.strip()


def _optional_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    return value or default


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise UIConfigurationError(f"{name} must be an integer, got: {raw}") from exc


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise UIConfigurationError(f"{name} must be a float, got: {raw}") from exc


def _normalize_ws_path(path: str) -> str:
    trimmed = path.strip()
    if not trimmed:
        raise UIConfigurationError("JARVIS_GATEWAY_WS_PATH cannot be empty.")
    if not trimmed.startswith("/"):
        trimmed = f"/{trimmed}"
    if trimmed != "/":
        trimmed = trimmed.rstrip("/")
    return trimmed


def _normalize_gateway_host(host: str) -> str:
    trimmed = host.strip()
    if not trimmed:
        raise UIConfigurationError("JARVIS_GATEWAY_HOST cannot be empty when deriving gateway URL.")
    if trimmed in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return trimmed


def _normalize_gateway_ws_base_url(raw_url: str) -> str:
    trimmed = raw_url.strip().rstrip("/")
    if not trimmed:
        raise UIConfigurationError("JARVIS_UI_GATEWAY_WS_BASE_URL cannot be empty.")
    if not (trimmed.startswith("ws://") or trimmed.startswith("wss://")):
        raise UIConfigurationError("JARVIS_UI_GATEWAY_WS_BASE_URL must start with ws:// or wss://.")
    return trimmed


def _derive_gateway_ws_base_url() -> str:
    host = _normalize_gateway_host(_optional_env("JARVIS_GATEWAY_HOST", "127.0.0.1"))
    port = _parse_int_env("JARVIS_GATEWAY_PORT", 8080)
    if port <= 0 or port > 65_535:
        raise UIConfigurationError("JARVIS_GATEWAY_PORT must be between 1 and 65535.")
    ws_path = _normalize_ws_path(_optional_env("JARVIS_GATEWAY_WS_PATH", "/ws"))
    return f"ws://{host}:{port}{ws_path}"


@dataclass(slots=True, frozen=True)
class UISettings:
    telegram_token: str
    telegram_api_base_url: str = "https://api.telegram.org"
    telegram_poll_timeout_seconds: int = 30
    telegram_poll_limit: int = 100
    poll_error_backoff_seconds: float = 2.0
    gateway_ws_base_url: str = "ws://127.0.0.1:8080/ws"
    gateway_connect_timeout_seconds: float = 15.0
    stream_draft_min_interval_seconds: float = 0.4
    stream_draft_min_chars: int = 20
    telegram_max_message_chars: int = 4_096

    def __post_init__(self) -> None:
        if not self.telegram_token.strip():
            raise UIConfigurationError("TELEGRAM_TOKEN cannot be empty.")
        if not self.telegram_api_base_url.startswith("https://"):
            raise UIConfigurationError("TELEGRAM_API_BASE_URL must start with https://.")
        if self.telegram_poll_timeout_seconds <= 0:
            raise UIConfigurationError("JARVIS_UI_TELEGRAM_POLL_TIMEOUT_SECONDS must be > 0.")
        if self.telegram_poll_limit <= 0 or self.telegram_poll_limit > 100:
            raise UIConfigurationError("JARVIS_UI_TELEGRAM_POLL_LIMIT must be between 1 and 100.")
        if self.poll_error_backoff_seconds <= 0:
            raise UIConfigurationError("JARVIS_UI_POLL_ERROR_BACKOFF_SECONDS must be > 0.")
        if self.gateway_connect_timeout_seconds <= 0:
            raise UIConfigurationError(
                "JARVIS_UI_GATEWAY_CONNECT_TIMEOUT_SECONDS must be > 0."
            )
        if self.stream_draft_min_interval_seconds < 0:
            raise UIConfigurationError(
                "JARVIS_UI_STREAM_DRAFT_MIN_INTERVAL_SECONDS must be >= 0."
            )
        if self.stream_draft_min_chars <= 0:
            raise UIConfigurationError("JARVIS_UI_STREAM_DRAFT_MIN_CHARS must be > 0.")
        if self.telegram_max_message_chars <= 0:
            raise UIConfigurationError("JARVIS_UI_TELEGRAM_MAX_MESSAGE_CHARS must be > 0.")
        _normalize_gateway_ws_base_url(self.gateway_ws_base_url)

    @classmethod
    def from_env(cls) -> "UISettings":
        gateway_raw = os.getenv("JARVIS_UI_GATEWAY_WS_BASE_URL")
        gateway_ws_base_url = (
            _normalize_gateway_ws_base_url(gateway_raw)
            if gateway_raw is not None
            else _derive_gateway_ws_base_url()
        )

        return cls(
            telegram_token=_required_env("TELEGRAM_TOKEN"),
            telegram_api_base_url=_optional_env(
                "TELEGRAM_API_BASE_URL",
                "https://api.telegram.org",
            ).rstrip("/"),
            telegram_poll_timeout_seconds=_parse_int_env(
                "JARVIS_UI_TELEGRAM_POLL_TIMEOUT_SECONDS",
                30,
            ),
            telegram_poll_limit=_parse_int_env("JARVIS_UI_TELEGRAM_POLL_LIMIT", 100),
            poll_error_backoff_seconds=_parse_float_env(
                "JARVIS_UI_POLL_ERROR_BACKOFF_SECONDS",
                2.0,
            ),
            gateway_ws_base_url=gateway_ws_base_url,
            gateway_connect_timeout_seconds=_parse_float_env(
                "JARVIS_UI_GATEWAY_CONNECT_TIMEOUT_SECONDS",
                15.0,
            ),
            stream_draft_min_interval_seconds=_parse_float_env(
                "JARVIS_UI_STREAM_DRAFT_MIN_INTERVAL_SECONDS",
                0.4,
            ),
            stream_draft_min_chars=_parse_int_env("JARVIS_UI_STREAM_DRAFT_MIN_CHARS", 20),
            telegram_max_message_chars=_parse_int_env(
                "JARVIS_UI_TELEGRAM_MAX_MESSAGE_CHARS",
                4_096,
            ),
        )
