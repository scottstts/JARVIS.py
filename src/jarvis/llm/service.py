"""Provider-agnostic service facade for LLM generation and embeddings."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import AsyncIterator, Iterable, TypeVar

from .config import LLMSettings
from .errors import (
    LLMConfigurationError,
    ProviderNotFoundError,
    ProviderTimeoutError,
    UnsupportedCapabilityError,
    is_retryable_error,
)
from .protocols import LLMProvider
from .types import (
    EmbeddingRequest,
    EmbeddingResponse,
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)

T = TypeVar("T")


class ProviderRegistry:
    """Holds provider adapters keyed by provider name."""

    def __init__(self, providers: Iterable[LLMProvider] | None = None) -> None:
        self._providers: dict[str, LLMProvider] = {}
        if providers:
            for provider in providers:
                self.register(provider)

    def register(self, provider: LLMProvider) -> None:
        if provider.name in self._providers:
            raise LLMConfigurationError(f"Provider '{provider.name}' already registered.")
        self._providers[provider.name] = provider

    def get(self, provider_name: str) -> LLMProvider:
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ProviderNotFoundError(f"Provider '{provider_name}' is not registered.")
        return provider

    def all(self) -> tuple[LLMProvider, ...]:
        return tuple(self._providers.values())


class LLMService:
    """Entry point used by core agent loop for LLM/embedding operations."""

    def __init__(
        self,
        *,
        settings: LLMSettings | None = None,
        providers: Iterable[LLMProvider] | None = None,
    ) -> None:
        self.settings = settings or LLMSettings.from_env()
        if providers is None:
            from .providers.anthropic_provider import AnthropicProvider
            from .providers.gemini_provider import GeminiProvider
            from .providers.grok_provider import GrokProvider
            from .providers.lmstudio_provider import LMStudioProvider
            from .providers.openai_provider import OpenAIProvider
            from .providers.openrouter_provider import OpenRouterProvider

            providers = (
                OpenAIProvider(
                    settings=self.settings.openai,
                    default_timeout_seconds=self.settings.request_timeout_seconds,
                ),
                AnthropicProvider(
                    settings=self.settings.anthropic,
                    default_timeout_seconds=self.settings.request_timeout_seconds,
                ),
                GeminiProvider(
                    settings=self.settings.gemini,
                    default_timeout_seconds=self.settings.request_timeout_seconds,
                ),
                GrokProvider(
                    settings=self.settings.grok,
                    default_timeout_seconds=self.settings.request_timeout_seconds,
                ),
                OpenRouterProvider(
                    settings=self.settings.openrouter,
                    default_timeout_seconds=self.settings.request_timeout_seconds,
                ),
                LMStudioProvider(
                    settings=self.settings.lmstudio,
                    default_timeout_seconds=self.settings.request_timeout_seconds,
                ),
            )
        self.registry = ProviderRegistry(providers)

    async def aclose(self) -> None:
        providers = self.registry.all()
        if not providers:
            return
        await asyncio.gather(*(provider.aclose() for provider in providers))

    async def generate(self, request: LLMRequest) -> LLMResponse:
        resolved = self._resolve_generate_request(request)
        if resolved.provider is None:
            raise LLMConfigurationError("Provider resolution failed.")
        self._reject_backend_routed_provider(resolved.provider)
        provider = self.registry.get(resolved.provider)
        self._assert_generation_capabilities(provider, resolved)

        async def attempt() -> LLMResponse:
            return await self._run_with_optional_timeout(
                resolved.timeout_seconds,
                provider.generate(resolved),
            )

        return await self._run_with_retries(attempt)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        resolved = self._resolve_generate_request(request)
        if resolved.provider is None:
            raise LLMConfigurationError("Provider resolution failed.")
        self._reject_backend_routed_provider(resolved.provider)
        provider = self.registry.get(resolved.provider)
        self._assert_generation_capabilities(provider, resolved, require_streaming=True)

        attempts = max(1, self.settings.retry_attempts + 1)
        for attempt_index in range(attempts):
            emitted_any = False
            try:
                stream = provider.stream_generate(resolved)
                iterator = (
                    self._iter_with_per_event_timeout(stream, resolved.timeout_seconds)
                    if resolved.timeout_seconds is not None
                    else stream
                )
                async for event in iterator:
                    emitted_any = True
                    yield event
                return
            except Exception as exc:
                should_retry = (
                    attempt_index < attempts - 1
                    and not emitted_any
                    and is_retryable_error(exc)
                )
                if not should_retry:
                    raise
                await asyncio.sleep(self._retry_delay_seconds(attempt_index))

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        resolved = self._resolve_embedding_request(request)
        if resolved.provider is None:
            raise LLMConfigurationError("Provider resolution failed.")
        self._reject_backend_routed_provider(resolved.provider)
        provider = self.registry.get(resolved.provider)
        if not provider.capabilities.embeddings:
            raise UnsupportedCapabilityError(
                f"Provider '{provider.name}' does not support embeddings."
            )

        async def attempt() -> EmbeddingResponse:
            return await self._run_with_optional_timeout(
                resolved.timeout_seconds,
                provider.embed(resolved),
            )

        return await self._run_with_retries(attempt)

    async def _run_with_retries(self, operation: Callable[[], Awaitable[T]]) -> T:
        attempts = max(1, self.settings.retry_attempts + 1)
        for attempt_index in range(attempts):
            try:
                return await operation()
            except Exception as exc:
                should_retry = (
                    attempt_index < attempts - 1 and is_retryable_error(exc)
                )
                if not should_retry:
                    raise
                await asyncio.sleep(self._retry_delay_seconds(attempt_index))

        raise RuntimeError("Retry loop exited unexpectedly.")

    def _reject_backend_routed_provider(self, provider: str) -> None:
        if provider != "codex":
            return
        raise LLMConfigurationError(
            "Provider 'codex' is handled by the Codex backend, not by LLMService."
        )

    async def _run_with_optional_timeout(
        self,
        timeout_seconds: float | None,
        awaitable: Awaitable[T],
    ) -> T:
        if timeout_seconds is None:
            return await awaitable
        try:
            return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise ProviderTimeoutError("Request timed out.") from exc

    async def _iter_with_per_event_timeout(
        self,
        stream: AsyncIterator[LLMStreamEvent],
        timeout_seconds: float | None,
    ) -> AsyncIterator[LLMStreamEvent]:
        if timeout_seconds is None:
            async for event in stream:
                yield event
            return

        iterator = stream.__aiter__()
        while True:
            try:
                event = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_seconds)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                aclose = getattr(iterator, "aclose", None)
                if callable(aclose):
                    await aclose()
                raise ProviderTimeoutError("Request timed out.") from exc
            yield event

    def _resolve_generate_request(self, request: LLMRequest) -> LLMRequest:
        provider = request.provider or self.settings.default_provider
        model: str | None = request.model
        temperature: float | None = request.temperature
        max_output_tokens: int | None = request.max_output_tokens

        if provider == "openai":
            model = model or self.settings.openai.chat_model
            temperature = (
                temperature
                if temperature is not None
                else self.settings.openai.temperature
            )
            max_output_tokens = (
                max_output_tokens
                if max_output_tokens is not None
                else self.settings.openai.max_output_tokens
            )
        elif provider == "anthropic":
            model = model or self.settings.anthropic.chat_model
            temperature = (
                temperature
                if temperature is not None
                else self.settings.anthropic.temperature
            )
            max_output_tokens = (
                max_output_tokens
                if max_output_tokens is not None
                else self.settings.anthropic.max_output_tokens
            )
        elif provider == "gemini":
            model = model or self.settings.gemini.chat_model
            temperature = (
                temperature
                if temperature is not None
                else self.settings.gemini.temperature
            )
            max_output_tokens = (
                max_output_tokens
                if max_output_tokens is not None
                else self.settings.gemini.max_output_tokens
            )
        elif provider == "grok":
            model = model or self.settings.grok.chat_model
            temperature = (
                temperature
                if temperature is not None
                else self.settings.grok.temperature
            )
            max_output_tokens = (
                max_output_tokens
                if max_output_tokens is not None
                else self.settings.grok.max_output_tokens
            )
        elif provider == "openrouter":
            model = model or self.settings.openrouter.chat_model
            temperature = (
                temperature
                if temperature is not None
                else self.settings.openrouter.temperature
            )
            max_output_tokens = (
                max_output_tokens
                if max_output_tokens is not None
                else self.settings.openrouter.max_output_tokens
            )
        elif provider == "codex":
            pass
        elif provider == "lmstudio":
            pass

        if model is None and provider not in {"codex", "lmstudio"}:
            raise LLMConfigurationError(
                f"No chat model configured for provider '{provider}'."
            )

        return replace(
            request,
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def _resolve_embedding_request(self, request: EmbeddingRequest) -> EmbeddingRequest:
        provider = request.provider or self.settings.embedding.provider
        model = request.model or self.settings.embedding.model
        return replace(request, provider=provider, model=model)

    def _assert_generation_capabilities(
        self,
        provider: LLMProvider,
        request: LLMRequest,
        *,
        require_streaming: bool = False,
    ) -> None:
        if request.tools and not provider.capabilities.tools:
            raise UnsupportedCapabilityError(
                f"Provider '{provider.name}' does not support tool calls."
            )

        has_image_input = any(
            isinstance(part, ImagePart)
            for message in request.messages
            for part in message.parts
        )
        if has_image_input and not provider.capabilities.image_input:
            raise UnsupportedCapabilityError(
                f"Provider '{provider.name}' does not support image inputs."
            )

        if require_streaming and not provider.capabilities.streaming:
            raise UnsupportedCapabilityError(
                f"Provider '{provider.name}' does not support streaming."
            )

    def _retry_delay_seconds(self, attempt_index: int) -> float:
        return self.settings.retry_backoff_seconds * (2**attempt_index)
