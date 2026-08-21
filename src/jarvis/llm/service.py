"""Provider-agnostic service facade for LLM generation and embeddings."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TypeVar

from .config import LLMSettings
from .errors import (
    LLMError,
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
    ProviderActivityEvent,
    ProviderStreamEvent,
)

T = TypeVar("T")


@dataclass(slots=True)
class _StreamAttemptState:
    provider: str
    model: str | None
    request_started_at: float
    attempt_started_at: float
    accepted: bool = False
    emitted_output: bool = False
    last_provider_event_type: str | None = None
    last_normalized_event_type: str | None = None
    response_id: str | None = None


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
                    read_timeout_seconds=self.settings.read_timeout_seconds,
                    connect_timeout_seconds=self.settings.connect_timeout_seconds,
                ),
                AnthropicProvider(
                    settings=self.settings.anthropic,
                    read_timeout_seconds=self.settings.read_timeout_seconds,
                    connect_timeout_seconds=self.settings.connect_timeout_seconds,
                ),
                GeminiProvider(
                    settings=self.settings.gemini,
                    read_timeout_seconds=self.settings.read_timeout_seconds,
                    connect_timeout_seconds=self.settings.connect_timeout_seconds,
                ),
                GrokProvider(
                    settings=self.settings.grok,
                    read_timeout_seconds=self.settings.read_timeout_seconds,
                    connect_timeout_seconds=self.settings.connect_timeout_seconds,
                ),
                OpenRouterProvider(
                    settings=self.settings.openrouter,
                    read_timeout_seconds=self.settings.read_timeout_seconds,
                    connect_timeout_seconds=self.settings.connect_timeout_seconds,
                ),
                LMStudioProvider(
                    settings=self.settings.lmstudio,
                    read_timeout_seconds=self.settings.read_timeout_seconds,
                    connect_timeout_seconds=self.settings.connect_timeout_seconds,
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
            return await provider.generate(resolved)

        return await self._run_with_retries(
            attempt,
            deadline_seconds=resolved.deadline_seconds,
            provider=resolved.provider,
            model=resolved.model,
        )

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        resolved = self._resolve_generate_request(request)
        if resolved.provider is None:
            raise LLMConfigurationError("Provider resolution failed.")
        self._reject_backend_routed_provider(resolved.provider)
        provider = self.registry.get(resolved.provider)
        self._assert_generation_capabilities(provider, resolved, require_streaming=True)

        loop = asyncio.get_running_loop()
        request_started_at = loop.time()
        deadline_at = (
            request_started_at + resolved.deadline_seconds
            if resolved.deadline_seconds is not None
            else None
        )
        attempts = max(1, self.settings.retry_attempts + 1)
        for attempt_index in range(attempts):
            state = _StreamAttemptState(
                provider=resolved.provider,
                model=resolved.model,
                request_started_at=request_started_at,
                attempt_started_at=loop.time(),
            )
            try:
                stream = provider.stream_generate(resolved)
                async for event in self._iter_stream_with_deadline(
                    stream,
                    deadline_at=deadline_at,
                    deadline_seconds=resolved.deadline_seconds,
                    state=state,
                ):
                    yield event
                return
            except Exception as exc:
                self._enrich_stream_error(exc, state=state)
                should_retry = (
                    attempt_index < attempts - 1
                    and not state.emitted_output
                    and (
                        not state.accepted
                        or self._is_retry_safe_after_acceptance(exc)
                    )
                    and self._is_safe_to_retry(exc)
                )
                if not should_retry:
                    raise
                await self._sleep_before_retry(
                    self._retry_delay_seconds(attempt_index),
                    deadline_at=deadline_at,
                    deadline_seconds=resolved.deadline_seconds,
                    provider=resolved.provider,
                    model=resolved.model,
                    request_started_at=request_started_at,
                )

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
            return await provider.embed(resolved)

        return await self._run_with_retries(
            attempt,
            deadline_seconds=resolved.deadline_seconds,
            provider=resolved.provider,
            model=resolved.model,
        )

    async def _run_with_retries(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        deadline_seconds: float | None,
        provider: str,
        model: str | None,
    ) -> T:
        loop = asyncio.get_running_loop()
        request_started_at = loop.time()
        deadline_at = (
            request_started_at + deadline_seconds
            if deadline_seconds is not None
            else None
        )
        attempts = max(1, self.settings.retry_attempts + 1)
        for attempt_index in range(attempts):
            try:
                return await self._run_with_request_deadline(
                    operation,
                    deadline_at=deadline_at,
                    deadline_seconds=deadline_seconds,
                    provider=provider,
                    model=model,
                    request_started_at=request_started_at,
                )
            except Exception as exc:
                self._enrich_request_error(
                    exc,
                    provider=provider,
                    model=model,
                    request_started_at=request_started_at,
                )
                should_retry = (
                    attempt_index < attempts - 1 and self._is_safe_to_retry(exc)
                )
                if not should_retry:
                    raise
                await self._sleep_before_retry(
                    self._retry_delay_seconds(attempt_index),
                    deadline_at=deadline_at,
                    deadline_seconds=deadline_seconds,
                    provider=provider,
                    model=model,
                    request_started_at=request_started_at,
                )

        raise RuntimeError("Retry loop exited unexpectedly.")

    def _reject_backend_routed_provider(self, provider: str) -> None:
        if provider != "codex":
            return
        raise LLMConfigurationError(
            "Provider 'codex' is handled by the Codex backend, not by LLMService."
        )

    async def _run_with_request_deadline(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        deadline_at: float | None,
        deadline_seconds: float | None,
        provider: str,
        model: str | None,
        request_started_at: float,
    ) -> T:
        if deadline_at is None:
            return await operation()
        remaining = deadline_at - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise self._request_deadline_error(
                deadline_seconds=deadline_seconds,
                provider=provider,
                model=model,
                request_started_at=request_started_at,
            )
        operation_task = asyncio.ensure_future(operation())
        try:
            done, _ = await asyncio.wait((operation_task,), timeout=remaining)
            if operation_task in done:
                return await operation_task
            operation_task.cancel()
            with suppress(asyncio.CancelledError):
                await operation_task
            raise self._request_deadline_error(
                deadline_seconds=deadline_seconds,
                provider=provider,
                model=model,
                request_started_at=request_started_at,
            )
        except BaseException:
            if not operation_task.done():
                operation_task.cancel()
                with suppress(asyncio.CancelledError):
                    await operation_task
            raise

    async def _iter_stream_with_deadline(
        self,
        stream: AsyncIterator[ProviderStreamEvent],
        *,
        deadline_at: float | None,
        deadline_seconds: float | None,
        state: _StreamAttemptState,
    ) -> AsyncIterator[LLMStreamEvent]:
        iterator = stream.__aiter__()
        completed = False
        try:
            while True:
                try:
                    if deadline_at is None:
                        event = await iterator.__anext__()
                    else:
                        remaining = deadline_at - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise self._request_deadline_error(
                                deadline_seconds=deadline_seconds,
                                provider=state.provider,
                                model=state.model,
                                request_started_at=state.request_started_at,
                                state=state,
                            )
                        next_task = asyncio.ensure_future(iterator.__anext__())
                        try:
                            done, _ = await asyncio.wait((next_task,), timeout=remaining)
                            if next_task not in done:
                                next_task.cancel()
                                with suppress(asyncio.CancelledError):
                                    await next_task
                                raise self._request_deadline_error(
                                    deadline_seconds=deadline_seconds,
                                    provider=state.provider,
                                    model=state.model,
                                    request_started_at=state.request_started_at,
                                    state=state,
                                )
                            event = await next_task
                        except BaseException:
                            if not next_task.done():
                                next_task.cancel()
                                with suppress(asyncio.CancelledError):
                                    await next_task
                            raise
                except StopAsyncIteration:
                    completed = True
                    return

                state.accepted = True
                if isinstance(event, ProviderActivityEvent):
                    state.last_provider_event_type = event.provider_event_type
                    if event.response_id is not None:
                        state.response_id = event.response_id
                    continue

                state.emitted_output = True
                state.last_normalized_event_type = event.type
                state.last_provider_event_type = event.type
                if event.type == "done" and event.response.response_id is not None:
                    state.response_id = event.response.response_id
                yield event
        finally:
            if not completed:
                await self._close_async_iterator(iterator)

    async def _close_async_iterator(
        self,
        iterator: AsyncIterator[ProviderStreamEvent],
    ) -> None:
        close = getattr(iterator, "aclose", None)
        if not callable(close):
            return
        with suppress(Exception):
            result = close()
            if inspect.isawaitable(result):
                await result

    async def _sleep_before_retry(
        self,
        delay_seconds: float,
        *,
        deadline_at: float | None,
        deadline_seconds: float | None,
        provider: str,
        model: str | None,
        request_started_at: float,
    ) -> None:
        if deadline_at is None:
            await asyncio.sleep(delay_seconds)
            return
        remaining = deadline_at - asyncio.get_running_loop().time()
        if remaining <= delay_seconds:
            if remaining > 0:
                await asyncio.sleep(remaining)
            raise self._request_deadline_error(
                deadline_seconds=deadline_seconds,
                provider=provider,
                model=model,
                request_started_at=request_started_at,
            )
        await asyncio.sleep(delay_seconds)

    def _request_deadline_error(
        self,
        *,
        deadline_seconds: float | None,
        provider: str,
        model: str | None,
        request_started_at: float,
        state: _StreamAttemptState | None = None,
    ) -> ProviderTimeoutError:
        now = asyncio.get_running_loop().time()
        metadata: dict[str, object] = {
            "timeout_kind": "request_deadline",
            "request_deadline_seconds": deadline_seconds,
            "elapsed_seconds": max(0.0, now - request_started_at),
            "provider": provider,
            "model": model,
        }
        if state is not None:
            metadata.update(
                {
                    "accepted": state.accepted,
                    "emitted_output": state.emitted_output,
                    "attempt_elapsed_seconds": max(0.0, now - state.attempt_started_at),
                    "last_provider_event_type": state.last_provider_event_type,
                    "last_normalized_event_type": state.last_normalized_event_type,
                    "response_id": state.response_id,
                }
            )
        return ProviderTimeoutError("Request deadline exceeded.", metadata=metadata)

    def _enrich_stream_error(
        self,
        error: Exception,
        *,
        state: _StreamAttemptState,
    ) -> None:
        if not isinstance(error, LLMError):
            return
        now = asyncio.get_running_loop().time()
        error.metadata.setdefault("provider", state.provider)
        error.metadata.setdefault("model", state.model)
        error.metadata.setdefault(
            "elapsed_seconds",
            max(0.0, now - state.request_started_at),
        )
        error.metadata.setdefault(
            "attempt_elapsed_seconds",
            max(0.0, now - state.attempt_started_at),
        )
        error.metadata.setdefault("accepted", state.accepted)
        error.metadata.setdefault("emitted_output", state.emitted_output)
        error.metadata.setdefault("last_provider_event_type", state.last_provider_event_type)
        error.metadata.setdefault(
            "last_normalized_event_type",
            state.last_normalized_event_type,
        )
        error.metadata.setdefault("response_id", state.response_id)

    def _enrich_request_error(
        self,
        error: Exception,
        *,
        provider: str,
        model: str | None,
        request_started_at: float,
    ) -> None:
        if not isinstance(error, LLMError):
            return
        error.metadata.setdefault("provider", provider)
        error.metadata.setdefault("model", model)
        error.metadata.setdefault(
            "elapsed_seconds",
            max(0.0, asyncio.get_running_loop().time() - request_started_at),
        )

    def _resolve_generate_request(self, request: LLMRequest) -> LLMRequest:
        provider = request.provider or self.settings.default_provider
        model: str | None = request.model
        temperature: float | None = request.temperature
        max_output_tokens: int | None = request.max_output_tokens
        deadline_seconds = (
            request.deadline_seconds
            if request.deadline_seconds is not None
            else self.settings.request_deadline_seconds
        )

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
            deadline_seconds=deadline_seconds,
        )

    def _resolve_embedding_request(self, request: EmbeddingRequest) -> EmbeddingRequest:
        provider = request.provider or self.settings.embedding.provider
        model = request.model or self.settings.embedding.model
        deadline_seconds = (
            request.deadline_seconds
            if request.deadline_seconds is not None
            else self.settings.request_deadline_seconds
        )
        return replace(
            request,
            provider=provider,
            model=model,
            deadline_seconds=deadline_seconds,
        )

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

    def _is_safe_to_retry(self, error: Exception) -> bool:
        if not is_retryable_error(error):
            return False
        if self._is_retry_safe_after_acceptance(error):
            return True
        if not isinstance(error, ProviderTimeoutError):
            return True
        return error.metadata.get("timeout_kind") in {"connect", "pool"}

    def _is_retry_safe_after_acceptance(self, error: Exception) -> bool:
        return (
            isinstance(error, LLMError)
            and error.metadata.get("retry_safe_after_acceptance") is True
        )
