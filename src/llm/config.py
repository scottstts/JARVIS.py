"""Configuration models for provider-specific LLM settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import LLMConfigurationError


def _required_env(name: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        raise LLMConfigurationError(f"{name} must be explicitly set in the environment.")
    return raw.strip()


def _optional_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    return value or None


def _optional_lower_env(name: str) -> str | None:
    value = _optional_env(name)
    if value is None:
        return None
    return value.lower()


def _optional_choice_env(name: str, allowed: set[str]) -> str | None:
    value = _optional_env(name)
    if value is None:
        return None
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise LLMConfigurationError(f"{name} must be one of: {allowed_values}. Got: {value}")
    return value


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be an integer, got: {raw}") from exc


def _parse_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be a float, got: {raw}") from exc


def _parse_optional_int_env(name: str) -> int | None:
    value = _optional_env(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be an integer, got: {value}") from exc


def _parse_optional_float_env(name: str) -> float | None:
    value = _optional_env(name)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be a float, got: {value}") from exc


@dataclass(slots=True, frozen=True)
class EmbeddingSettings:
    provider: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise LLMConfigurationError("JARVIS_EMBEDDING_PROVIDER cannot be empty.")
        if self.provider not in {"openai", "gemini", "openrouter"}:
            raise LLMConfigurationError(
                "JARVIS_EMBEDDING_PROVIDER must be one of: openai, gemini, openrouter."
            )
        if not self.model.strip():
            raise LLMConfigurationError("JARVIS_EMBEDDING_MODEL cannot be empty.")

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        return cls(
            provider=_required_env("JARVIS_EMBEDDING_PROVIDER"),
            model=_required_env("JARVIS_EMBEDDING_MODEL"),
        )


@dataclass(slots=True, frozen=True)
class OpenAIProviderSettings:
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    chat_model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    text_verbosity: str | None = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise LLMConfigurationError("JARVIS_OPENAI_MAX_OUTPUT_TOKENS must be > 0.")

    @classmethod
    def from_env(cls) -> "OpenAIProviderSettings":
        return cls(
            api_key=_optional_env("OPENAI_API_KEY"),
            base_url=_optional_env("OPENAI_BASE_URL"),
            organization=_optional_env("OPENAI_ORG_ID"),
            project=_optional_env("OPENAI_PROJECT_ID"),
            chat_model=_optional_env("JARVIS_OPENAI_CHAT_MODEL"),
            temperature=_parse_optional_float_env("JARVIS_OPENAI_TEMPERATURE"),
            max_output_tokens=_parse_optional_int_env("JARVIS_OPENAI_MAX_OUTPUT_TOKENS"),
            reasoning_effort=_optional_choice_env(
                "JARVIS_OPENAI_REASONING_EFFORT",
                {"none", "minimal", "low", "medium", "high", "xhigh"},
            ),
            reasoning_summary=_optional_choice_env(
                "JARVIS_OPENAI_REASONING_SUMMARY",
                {"auto", "concise", "detailed"},
            ),
            text_verbosity=_optional_choice_env(
                "JARVIS_OPENAI_TEXT_VERBOSITY",
                {"low", "medium", "high"},
            ),
        )


@dataclass(slots=True, frozen=True)
class AnthropicProviderSettings:
    api_key: str | None = None
    base_url: str | None = None
    chat_model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_mode: str | None = None
    thinking_budget_tokens: int | None = None
    effort: str | None = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise LLMConfigurationError("JARVIS_ANTHROPIC_MAX_OUTPUT_TOKENS must be > 0.")
        if self.thinking_budget_tokens is not None and self.thinking_budget_tokens <= 0:
            raise LLMConfigurationError("JARVIS_ANTHROPIC_THINKING_BUDGET_TOKENS must be > 0.")

    @classmethod
    def from_env(cls) -> "AnthropicProviderSettings":
        return cls(
            api_key=_optional_env("ANTHROPIC_API_KEY"),
            base_url=_optional_env("ANTHROPIC_BASE_URL"),
            chat_model=_optional_env("JARVIS_ANTHROPIC_CHAT_MODEL"),
            temperature=_parse_optional_float_env("JARVIS_ANTHROPIC_TEMPERATURE"),
            max_output_tokens=_parse_optional_int_env("JARVIS_ANTHROPIC_MAX_OUTPUT_TOKENS"),
            thinking_mode=(
                _optional_lower_env("JARVIS_ANTHROPIC_THINKING_MODE")
                or _optional_lower_env("JARVIS_ANTHROPIC_THINKING_TYPE")
            ),
            thinking_budget_tokens=_parse_optional_int_env(
                "JARVIS_ANTHROPIC_THINKING_BUDGET_TOKENS"
            ),
            effort=_optional_lower_env("JARVIS_ANTHROPIC_EFFORT"),
        )


@dataclass(slots=True, frozen=True)
class GeminiProviderSettings:
    api_key: str | None = None
    chat_model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_level: str | None = None
    thinking_budget: int | None = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise LLMConfigurationError("JARVIS_GEMINI_MAX_OUTPUT_TOKENS must be > 0.")
        if self.thinking_budget is not None and self.thinking_budget <= 0:
            raise LLMConfigurationError("JARVIS_GEMINI_THINKING_BUDGET must be > 0.")

    @classmethod
    def from_env(cls) -> "GeminiProviderSettings":
        return cls(
            api_key=_optional_env("GOOGLE_API_KEY"),
            chat_model=_optional_env("JARVIS_GEMINI_CHAT_MODEL"),
            temperature=_parse_optional_float_env("JARVIS_GEMINI_TEMPERATURE"),
            max_output_tokens=_parse_optional_int_env("JARVIS_GEMINI_MAX_OUTPUT_TOKENS"),
            thinking_level=_optional_lower_env("JARVIS_GEMINI_THINKING_LEVEL"),
            thinking_budget=_parse_optional_int_env("JARVIS_GEMINI_THINKING_BUDGET"),
        )


@dataclass(slots=True, frozen=True)
class OpenRouterProviderSettings:
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    chat_model: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    site_url: str | None = None
    app_name: str | None = None

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise LLMConfigurationError("JARVIS_OPENROUTER_MAX_OUTPUT_TOKENS must be > 0.")

    @classmethod
    def from_env(cls) -> "OpenRouterProviderSettings":
        return cls(
            api_key=_optional_env("OPENROUTER_API_KEY"),
            base_url=_optional_env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1",
            chat_model=_optional_env("JARVIS_OPENROUTER_CHAT_MODEL"),
            temperature=_parse_optional_float_env("JARVIS_OPENROUTER_TEMPERATURE"),
            max_output_tokens=_parse_optional_int_env("JARVIS_OPENROUTER_MAX_OUTPUT_TOKENS"),
            site_url=_optional_env("OPENROUTER_SITE_URL"),
            app_name=_optional_env("OPENROUTER_APP_NAME"),
        )


@dataclass(slots=True, frozen=True)
class LLMSettings:
    default_provider: str = ""
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings.from_env)
    request_timeout_seconds: float = 60.0
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.5
    openai: OpenAIProviderSettings = field(default_factory=OpenAIProviderSettings.from_env)
    anthropic: AnthropicProviderSettings = field(default_factory=AnthropicProviderSettings.from_env)
    gemini: GeminiProviderSettings = field(default_factory=GeminiProviderSettings.from_env)
    openrouter: OpenRouterProviderSettings = field(default_factory=OpenRouterProviderSettings.from_env)

    def __post_init__(self) -> None:
        if not self.default_provider.strip():
            raise LLMConfigurationError("JARVIS_LLM_DEFAULT_PROVIDER cannot be empty.")
        if self.default_provider not in {"openai", "anthropic", "gemini", "openrouter"}:
            raise LLMConfigurationError(
                "JARVIS_LLM_DEFAULT_PROVIDER must be one of: openai, anthropic, gemini, openrouter."
            )
        if self.retry_attempts < 0:
            raise LLMConfigurationError("JARVIS_LLM_RETRY_ATTEMPTS must be >= 0.")
        if self.retry_backoff_seconds < 0:
            raise LLMConfigurationError("JARVIS_LLM_RETRY_BACKOFF_SECONDS must be >= 0.")

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            default_provider=_required_env("JARVIS_LLM_DEFAULT_PROVIDER"),
            embedding=EmbeddingSettings.from_env(),
            request_timeout_seconds=_parse_float_env("JARVIS_LLM_TIMEOUT_SECONDS", 60.0),
            retry_attempts=_parse_int_env("JARVIS_LLM_RETRY_ATTEMPTS", 2),
            retry_backoff_seconds=_parse_float_env("JARVIS_LLM_RETRY_BACKOFF_SECONDS", 0.5),
            openai=OpenAIProviderSettings.from_env(),
            anthropic=AnthropicProviderSettings.from_env(),
            gemini=GeminiProviderSettings.from_env(),
            openrouter=OpenRouterProviderSettings.from_env(),
        )
