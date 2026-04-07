"""Shared types and errors for the Codex backend."""

from __future__ import annotations

from dataclasses import dataclass


class CodexBackendError(RuntimeError):
    """Base error for Codex backend failures."""


class CodexConfigurationError(CodexBackendError):
    """Raised when Codex backend settings are incomplete or invalid."""


class CodexProtocolError(CodexBackendError):
    """Raised when the Codex app-server protocol is violated."""


class CodexConnectionError(CodexBackendError):
    """Raised when Jarvis cannot connect to the Codex app-server transport."""


class CodexAuthenticationError(CodexBackendError):
    """Raised when ChatGPT OAuth cannot be completed."""


class CodexNativeCapabilityError(CodexBackendError):
    """Raised when Codex emits native items Jarvis does not support owning."""


@dataclass(slots=True, frozen=True)
class CodexAuthChallenge:
    """Browser-login challenge returned by Codex app-server."""

    login_id: str
    auth_url: str
