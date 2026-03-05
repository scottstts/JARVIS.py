"""Runtime settings for the websocket gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass


class GatewayConfigurationError(ValueError):
    """Raised when gateway runtime settings are invalid."""


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
        raise GatewayConfigurationError(f"{name} must be an integer, got: {raw}") from exc


def _normalize_ws_path(path: str) -> str:
    trimmed = path.strip()
    if not trimmed:
        raise GatewayConfigurationError("JARVIS_GATEWAY_WS_PATH cannot be empty.")
    if not trimmed.startswith("/"):
        trimmed = f"/{trimmed}"
    if trimmed != "/":
        trimmed = trimmed.rstrip("/")
    return trimmed


@dataclass(slots=True, frozen=True)
class GatewaySettings:
    host: str = "0.0.0.0"
    port: int = 8080
    websocket_path: str = "/ws"
    max_message_chars: int = 32_000

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise GatewayConfigurationError("JARVIS_GATEWAY_HOST cannot be empty.")
        if self.port <= 0 or self.port > 65_535:
            raise GatewayConfigurationError("JARVIS_GATEWAY_PORT must be between 1 and 65535.")
        if self.max_message_chars <= 0:
            raise GatewayConfigurationError("JARVIS_GATEWAY_MAX_MESSAGE_CHARS must be > 0.")
        if not self.websocket_path.startswith("/"):
            raise GatewayConfigurationError("JARVIS_GATEWAY_WS_PATH must start with '/'.")

    @property
    def websocket_route_path(self) -> str:
        if self.websocket_path == "/":
            return "/{route_id}"
        return f"{self.websocket_path}/{{route_id}}"

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        return cls(
            host=_optional_env("JARVIS_GATEWAY_HOST", "0.0.0.0"),
            port=_parse_int_env("JARVIS_GATEWAY_PORT", 8080),
            websocket_path=_normalize_ws_path(_optional_env("JARVIS_GATEWAY_WS_PATH", "/ws")),
            max_message_chars=_parse_int_env("JARVIS_GATEWAY_MAX_MESSAGE_CHARS", 32_000),
        )
