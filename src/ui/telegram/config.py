"""Runtime settings for the Telegram UI bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import settings as app_settings


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


def _parse_optional_int_env(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        raise UIConfigurationError(f"{name} must be an integer when set.")
    try:
        return int(value)
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


def _parse_path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if not value:
        raise UIConfigurationError(f"{name} cannot be empty when set.")
    return Path(value).expanduser()


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
    host = _normalize_gateway_host(
        _optional_env("JARVIS_GATEWAY_HOST", app_settings.JARVIS_GATEWAY_HOST)
    )
    port = _parse_int_env("JARVIS_GATEWAY_PORT", app_settings.JARVIS_GATEWAY_PORT)
    if port <= 0 or port > 65_535:
        raise UIConfigurationError("JARVIS_GATEWAY_PORT must be between 1 and 65535.")
    ws_path = _normalize_ws_path(
        _optional_env("JARVIS_GATEWAY_WS_PATH", app_settings.JARVIS_GATEWAY_WS_PATH)
    )
    return f"ws://{host}:{port}{ws_path}"


_DEFAULT_GATEWAY_WS_BASE_URL = (
    _normalize_gateway_ws_base_url(app_settings.JARVIS_UI_GATEWAY_WS_BASE_URL)
    if app_settings.JARVIS_UI_GATEWAY_WS_BASE_URL is not None
    else _normalize_gateway_ws_base_url(
        f"ws://{_normalize_gateway_host(app_settings.JARVIS_GATEWAY_HOST)}:"
        f"{app_settings.JARVIS_GATEWAY_PORT}"
        f"{_normalize_ws_path(app_settings.JARVIS_GATEWAY_WS_PATH)}"
    )
)


def _default_telegram_temp_dir() -> Path:
    configured = app_settings.JARVIS_UI_TELEGRAM_TEMP_DIR
    if configured is not None and configured.strip():
        return Path(configured).expanduser()
    return Path("/workspace/temp")


_DEFAULT_TELEGRAM_TEMP_DIR = _default_telegram_temp_dir()


@dataclass(slots=True, frozen=True)
class UISettings:
    telegram_token: str
    telegram_api_base_url: str = app_settings.TELEGRAM_API_BASE_URL
    telegram_allowed_user_id: int | None = app_settings.JARVIS_UI_TELEGRAM_ALLOWED_USER_ID
    telegram_temp_dir: Path = _DEFAULT_TELEGRAM_TEMP_DIR
    telegram_poll_timeout_seconds: int = app_settings.JARVIS_UI_TELEGRAM_POLL_TIMEOUT_SECONDS
    telegram_poll_limit: int = app_settings.JARVIS_UI_TELEGRAM_POLL_LIMIT
    poll_error_backoff_seconds: float = app_settings.JARVIS_UI_POLL_ERROR_BACKOFF_SECONDS
    gateway_ws_base_url: str = _DEFAULT_GATEWAY_WS_BASE_URL
    gateway_connect_timeout_seconds: float = app_settings.JARVIS_UI_GATEWAY_CONNECT_TIMEOUT_SECONDS
    stream_draft_min_interval_seconds: float = (
        app_settings.JARVIS_UI_STREAM_DRAFT_MIN_INTERVAL_SECONDS
    )
    stream_draft_min_chars: int = app_settings.JARVIS_UI_STREAM_DRAFT_MIN_CHARS
    telegram_max_message_chars: int = app_settings.JARVIS_UI_TELEGRAM_MAX_MESSAGE_CHARS

    def __post_init__(self) -> None:
        if not self.telegram_token.strip():
            raise UIConfigurationError("TELEGRAM_TOKEN cannot be empty.")
        if not self.telegram_api_base_url.startswith("https://"):
            raise UIConfigurationError("TELEGRAM_API_BASE_URL must start with https://.")
        if self.telegram_allowed_user_id is not None and self.telegram_allowed_user_id <= 0:
            raise UIConfigurationError("JARVIS_UI_TELEGRAM_ALLOWED_USER_ID must be > 0.")
        if not str(self.telegram_temp_dir).strip():
            raise UIConfigurationError("JARVIS_UI_TELEGRAM_TEMP_DIR cannot be empty.")
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
        if gateway_raw is None:
            gateway_raw = app_settings.JARVIS_UI_GATEWAY_WS_BASE_URL
        gateway_ws_base_url = (
            _normalize_gateway_ws_base_url(gateway_raw)
            if gateway_raw is not None
            else _derive_gateway_ws_base_url()
        )

        return cls(
            telegram_token=_required_env("TELEGRAM_TOKEN"),
            telegram_api_base_url=_optional_env(
                "TELEGRAM_API_BASE_URL",
                app_settings.TELEGRAM_API_BASE_URL,
            ).rstrip("/"),
            telegram_allowed_user_id=_parse_optional_int_env(
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID",
                app_settings.JARVIS_UI_TELEGRAM_ALLOWED_USER_ID,
            ),
            telegram_temp_dir=_parse_path_env(
                "JARVIS_UI_TELEGRAM_TEMP_DIR",
                _DEFAULT_TELEGRAM_TEMP_DIR,
            ),
            telegram_poll_timeout_seconds=_parse_int_env(
                "JARVIS_UI_TELEGRAM_POLL_TIMEOUT_SECONDS",
                app_settings.JARVIS_UI_TELEGRAM_POLL_TIMEOUT_SECONDS,
            ),
            telegram_poll_limit=_parse_int_env(
                "JARVIS_UI_TELEGRAM_POLL_LIMIT",
                app_settings.JARVIS_UI_TELEGRAM_POLL_LIMIT,
            ),
            poll_error_backoff_seconds=_parse_float_env(
                "JARVIS_UI_POLL_ERROR_BACKOFF_SECONDS",
                app_settings.JARVIS_UI_POLL_ERROR_BACKOFF_SECONDS,
            ),
            gateway_ws_base_url=gateway_ws_base_url,
            gateway_connect_timeout_seconds=_parse_float_env(
                "JARVIS_UI_GATEWAY_CONNECT_TIMEOUT_SECONDS",
                app_settings.JARVIS_UI_GATEWAY_CONNECT_TIMEOUT_SECONDS,
            ),
            stream_draft_min_interval_seconds=_parse_float_env(
                "JARVIS_UI_STREAM_DRAFT_MIN_INTERVAL_SECONDS",
                app_settings.JARVIS_UI_STREAM_DRAFT_MIN_INTERVAL_SECONDS,
            ),
            stream_draft_min_chars=_parse_int_env(
                "JARVIS_UI_STREAM_DRAFT_MIN_CHARS",
                app_settings.JARVIS_UI_STREAM_DRAFT_MIN_CHARS,
            ),
            telegram_max_message_chars=_parse_int_env(
                "JARVIS_UI_TELEGRAM_MAX_MESSAGE_CHARS",
                app_settings.JARVIS_UI_TELEGRAM_MAX_MESSAGE_CHARS,
            ),
        )
