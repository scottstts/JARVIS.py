"""Runtime settings for the Telegram UI bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from jarvis.gateway.config import GatewaySettings
from jarvis.workspace_paths import resolve_workspace_child, resolve_workspace_dir


class UIConfigurationError(ValueError):
    """Raised when UI runtime settings are invalid."""


_DEFAULT_TELEGRAM_API_BASE_URL = "https://api.telegram.org"
_DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS = 30
_DEFAULT_TELEGRAM_POLL_LIMIT = 100
_DEFAULT_POLL_ERROR_BACKOFF_SECONDS = 2.0
_DEFAULT_GATEWAY_CONNECT_TIMEOUT_SECONDS = 15.0
_DEFAULT_STREAM_TRANSPORT = "edit"
_DEFAULT_STREAM_CHUNK_IDLE_FLUSH_SECONDS = 1.2
_DEFAULT_STREAM_CHUNK_MIN_CHARS = 80
_DEFAULT_STREAM_CHUNK_MAX_CHARS = 350
_DEFAULT_STREAM_TYPING_INDICATOR_INTERVAL_SECONDS = 4.0
_DEFAULT_TELEGRAM_MAX_MESSAGE_CHARS = 4096
_DEFAULT_TELEGRAM_TEMP_DIR = Path("/workspace/temp")


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


def _required_int_env(name: str, default: int | None) -> int:
    value = _parse_optional_int_env(name, default)
    if value is None:
        raise UIConfigurationError(f"{name} must be explicitly set in the environment.")
    return value


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
    settings = GatewaySettings.from_env()
    host = _normalize_gateway_host(settings.host)
    return f"ws://{host}:{settings.port}{_normalize_ws_path(settings.websocket_path)}"


_DEFAULT_GATEWAY_WS_BASE_URL = _derive_gateway_ws_base_url()


@dataclass(slots=True, frozen=True)
class UISettings:
    telegram_token: str = field(repr=False)
    telegram_api_base_url: str = _DEFAULT_TELEGRAM_API_BASE_URL
    telegram_allowed_user_id: int | None = field(default=None, repr=False)
    telegram_temp_dir: Path = _DEFAULT_TELEGRAM_TEMP_DIR
    telegram_poll_timeout_seconds: int = _DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS
    telegram_poll_limit: int = _DEFAULT_TELEGRAM_POLL_LIMIT
    poll_error_backoff_seconds: float = _DEFAULT_POLL_ERROR_BACKOFF_SECONDS
    gateway_ws_base_url: str = _DEFAULT_GATEWAY_WS_BASE_URL
    gateway_connect_timeout_seconds: float = _DEFAULT_GATEWAY_CONNECT_TIMEOUT_SECONDS
    stream_transport: str = _DEFAULT_STREAM_TRANSPORT
    stream_chunk_idle_flush_seconds: float = _DEFAULT_STREAM_CHUNK_IDLE_FLUSH_SECONDS
    stream_chunk_min_chars: int = _DEFAULT_STREAM_CHUNK_MIN_CHARS
    stream_chunk_max_chars: int = _DEFAULT_STREAM_CHUNK_MAX_CHARS
    stream_typing_indicator_interval_seconds: float = (
        _DEFAULT_STREAM_TYPING_INDICATOR_INTERVAL_SECONDS
    )
    telegram_max_message_chars: int = _DEFAULT_TELEGRAM_MAX_MESSAGE_CHARS

    def __post_init__(self) -> None:
        if not self.telegram_token.strip():
            raise UIConfigurationError("TELEGRAM_TOKEN cannot be empty.")
        if not self.telegram_api_base_url.startswith("https://"):
            raise UIConfigurationError("TELEGRAM_API_BASE_URL must start with https://.")
        if self.telegram_allowed_user_id is None:
            raise UIConfigurationError(
                "JARVIS_UI_TELEGRAM_ALLOWED_USER_ID must be explicitly set."
            )
        if self.telegram_allowed_user_id <= 0:
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
        if self.stream_transport not in {"edit", "draft"}:
            raise UIConfigurationError(
                "JARVIS_UI_STREAM_TRANSPORT must be either 'edit' or 'draft'."
            )
        if self.stream_chunk_idle_flush_seconds < 0:
            raise UIConfigurationError(
                "JARVIS_UI_STREAM_DRAFT_MIN_INTERVAL_SECONDS must be >= 0."
            )
        if self.stream_chunk_min_chars <= 0:
            raise UIConfigurationError("JARVIS_UI_STREAM_DRAFT_MIN_CHARS must be > 0.")
        if self.stream_chunk_max_chars < self.stream_chunk_min_chars:
            raise UIConfigurationError(
                "JARVIS_UI_STREAM_CHUNK_MAX_CHARS must be >= JARVIS_UI_STREAM_DRAFT_MIN_CHARS."
            )
        if self.stream_typing_indicator_interval_seconds <= 0:
            raise UIConfigurationError(
                "JARVIS_UI_STREAM_TYPING_INDICATOR_INTERVAL_SECONDS must be > 0."
            )
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
        telegram_token = _required_env("TELEGRAM_TOKEN")
        telegram_allowed_user_id = _required_int_env("JARVIS_UI_TELEGRAM_ALLOWED_USER_ID", None)
        workspace_dir = resolve_workspace_dir(error_type=UIConfigurationError)
        default_telegram_temp_dir = resolve_workspace_child(
            env_name="JARVIS_UI_TELEGRAM_TEMP_DIR",
            configured_default=None,
            workspace_dir=workspace_dir,
            child_name="temp",
        )

        return cls(
            telegram_token=telegram_token,
            telegram_api_base_url=_optional_env(
                "TELEGRAM_API_BASE_URL",
                _DEFAULT_TELEGRAM_API_BASE_URL,
            ).rstrip("/"),
            telegram_allowed_user_id=telegram_allowed_user_id,
            telegram_temp_dir=_parse_path_env(
                "JARVIS_UI_TELEGRAM_TEMP_DIR",
                default_telegram_temp_dir,
            ),
            telegram_poll_timeout_seconds=_parse_int_env(
                "JARVIS_UI_TELEGRAM_POLL_TIMEOUT_SECONDS",
                _DEFAULT_TELEGRAM_POLL_TIMEOUT_SECONDS,
            ),
            telegram_poll_limit=_parse_int_env(
                "JARVIS_UI_TELEGRAM_POLL_LIMIT",
                _DEFAULT_TELEGRAM_POLL_LIMIT,
            ),
            poll_error_backoff_seconds=_parse_float_env(
                "JARVIS_UI_POLL_ERROR_BACKOFF_SECONDS",
                _DEFAULT_POLL_ERROR_BACKOFF_SECONDS,
            ),
            gateway_ws_base_url=gateway_ws_base_url,
            gateway_connect_timeout_seconds=_parse_float_env(
                "JARVIS_UI_GATEWAY_CONNECT_TIMEOUT_SECONDS",
                _DEFAULT_GATEWAY_CONNECT_TIMEOUT_SECONDS,
            ),
            stream_transport=_optional_env(
                "JARVIS_UI_STREAM_TRANSPORT",
                _DEFAULT_STREAM_TRANSPORT,
            ),
            stream_chunk_idle_flush_seconds=_parse_float_env(
                "JARVIS_UI_STREAM_DRAFT_MIN_INTERVAL_SECONDS",
                _DEFAULT_STREAM_CHUNK_IDLE_FLUSH_SECONDS,
            ),
            stream_chunk_min_chars=_parse_int_env(
                "JARVIS_UI_STREAM_DRAFT_MIN_CHARS",
                _DEFAULT_STREAM_CHUNK_MIN_CHARS,
            ),
            stream_chunk_max_chars=_parse_int_env(
                "JARVIS_UI_STREAM_CHUNK_MAX_CHARS",
                _DEFAULT_STREAM_CHUNK_MAX_CHARS,
            ),
            stream_typing_indicator_interval_seconds=_parse_float_env(
                "JARVIS_UI_STREAM_TYPING_INDICATOR_INTERVAL_SECONDS",
                _DEFAULT_STREAM_TYPING_INDICATOR_INTERVAL_SECONDS,
            ),
            telegram_max_message_chars=_parse_int_env(
                "JARVIS_UI_TELEGRAM_MAX_MESSAGE_CHARS",
                _DEFAULT_TELEGRAM_MAX_MESSAGE_CHARS,
            ),
        )
