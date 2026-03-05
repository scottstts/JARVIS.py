"""Normalized exception hierarchy for the LLM service layer."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM service errors."""


class LLMConfigurationError(LLMError):
    """Raised when service/provider configuration is invalid."""


class ProviderNotFoundError(LLMError):
    """Raised when a provider key is not registered."""


class UnsupportedCapabilityError(LLMError):
    """Raised when a request asks for unsupported provider capability."""


class ProviderAuthenticationError(LLMError):
    """Raised when upstream auth/permission fails."""


class ProviderRateLimitError(LLMError):
    """Raised when upstream rate limits a request."""


class ProviderBadRequestError(LLMError):
    """Raised when request payload is rejected by upstream API."""


class ProviderTemporaryError(LLMError):
    """Raised for retryable upstream/service failures."""


class ProviderTimeoutError(ProviderTemporaryError):
    """Raised when upstream request times out."""


class ProviderResponseError(LLMError):
    """Raised for unexpected provider-side response/protocol failures."""


class StreamProtocolError(LLMError):
    """Raised when streaming response flow is malformed or incomplete."""


class ToolCallValidationError(LLMError):
    """Raised when model tool-call output fails parsing or schema validation."""

    def __init__(
        self,
        message: str,
        *,
        tool_name: str | None = None,
        call_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.call_id = call_id


def is_retryable_error(error: Exception) -> bool:
    """True when request can be safely retried at the service layer."""
    return isinstance(
        error,
        (
            ProviderRateLimitError,
            ProviderTemporaryError,
        ),
    )
