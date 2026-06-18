"""Lifecycle tests for LLMService provider cleanup."""

from __future__ import annotations

import asyncio
import unittest

from jarvis.llm.config import EmbeddingSettings, LLMSettings
from jarvis.llm.errors import LLMConfigurationError, ProviderTemporaryError, ProviderTimeoutError
from jarvis.llm.protocols import ProviderCapabilities
from jarvis.llm.service import LLMService
from jarvis.llm.types import (
    DoneEvent,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    StreamActivityEvent,
    TextDeltaEvent,
)


class _FakeProvider:
    def __init__(self, name: str) -> None:
        self._name = name
        self.closed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    async def generate(self, request):
        raise NotImplementedError

    async def stream_generate(self, request):
        raise NotImplementedError
        yield None

    async def embed(self, request):
        raise NotImplementedError

    async def aclose(self) -> None:
        self.closed = True


class _SlowProvider(_FakeProvider):
    async def generate(self, request):
        await asyncio.sleep(3600)
        raise AssertionError("unreachable")

    async def stream_generate(self, request):
        await asyncio.sleep(3600)
        yield None


class _DelayedStreamProvider(_FakeProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.request_timeout_seconds: float | None = None

    async def stream_generate(self, request):
        self.request_timeout_seconds = request.timeout_seconds
        await asyncio.sleep(0.02)
        yield DoneEvent(
            response=LLMResponse(
                provider=self.name,
                model=request.model or "delayed-model",
                text="complete",
                tool_calls=[],
                finish_reason="stop",
                usage=LLMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        )


class _RetryableStreamProvider(_FakeProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.stream_attempts = 0

    async def stream_generate(self, request):
        self.stream_attempts += 1
        if self.stream_attempts == 1:
            raise ProviderTemporaryError("transient stream failure")
        yield DoneEvent(
            response=LLMResponse(
                provider=self.name,
                model=request.model or "retry-model",
                text="recovered",
                tool_calls=[],
                finish_reason="stop",
                usage=LLMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        )


class _ActivityThenRetryableStreamProvider(_RetryableStreamProvider):
    async def stream_generate(self, request):
        self.stream_attempts += 1
        if self.stream_attempts == 1:
            yield StreamActivityEvent(source="reasoning")
            raise ProviderTemporaryError("transient stream failure")
        yield DoneEvent(
            response=LLMResponse(
                provider=self.name,
                model=request.model or "retry-model",
                text="recovered",
                tool_calls=[],
                finish_reason="stop",
                usage=LLMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        )


class _PartialOutputThenRetryableStreamProvider(_FakeProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.stream_attempts = 0

    async def stream_generate(self, request):
        self.stream_attempts += 1
        yield TextDeltaEvent(delta="partial")
        raise ProviderTemporaryError("transient stream failure")


class _ReasoningActivityProvider(_FakeProvider):
    async def stream_generate(self, request):
        for _index in range(3):
            await asyncio.sleep(0.04)
            yield StreamActivityEvent(source="reasoning")
        yield DoneEvent(
            response=LLMResponse(
                provider=self.name,
                model=request.model or "reasoning-model",
                text="complete",
                tool_calls=[],
                finish_reason="stop",
                usage=LLMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        )


class LLMServiceLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_aclose_calls_aclose_on_all_registered_providers(self) -> None:
        provider_one = _FakeProvider("openai")
        provider_two = _FakeProvider("anthropic")
        settings = LLMSettings(
            default_provider="openai",
            embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
        )
        service = LLMService(
            settings=settings,
            providers=(provider_one, provider_two),
        )

        await service.aclose()
        self.assertTrue(provider_one.closed)
        self.assertTrue(provider_two.closed)

    async def test_default_registry_includes_lmstudio_and_grok_providers(self) -> None:
        service = LLMService(
            settings=LLMSettings(
                default_provider="lmstudio",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            )
        )

        try:
            self.assertEqual(service.registry.get("lmstudio").name, "lmstudio")
            self.assertEqual(service.registry.get("grok").name, "grok")
        finally:
            await service.aclose()

    async def test_generate_maps_service_timeout_to_provider_timeout(self) -> None:
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(_SlowProvider("openai"),),
        )

        with self.assertRaises(ProviderTimeoutError):
            await service.generate(
                LLMRequest(
                    messages=(LLMMessage.text("user", "hello"),),
                    generation_timeout_seconds=0.01,
                )
            )

    async def test_stream_generate_leaves_request_timeout_to_provider_transport(self) -> None:
        provider = _DelayedStreamProvider("openai")
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(provider,),
        )

        events = [
            event
            async for event in service.stream_generate(
                LLMRequest(
                    messages=(LLMMessage.text("user", "hello"),),
                    timeout_seconds=0.001,
                )
            )
        ]

        self.assertEqual(provider.request_timeout_seconds, 0.001)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], DoneEvent)

    async def test_stream_generate_retries_retryable_pre_output_errors(self) -> None:
        provider = _RetryableStreamProvider("openai")
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                retry_attempts=1,
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(provider,),
        )

        events = [
            event
            async for event in service.stream_generate(
                LLMRequest(
                    model="gpt-5.4-2026-03-05",
                    messages=(LLMMessage.text("user", "hello"),),
                )
            )
        ]

        self.assertEqual(provider.stream_attempts, 2)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], DoneEvent)
        self.assertEqual(events[0].response.text, "recovered")

    async def test_stream_generate_reasoning_activity_does_not_block_retry(self) -> None:
        provider = _ActivityThenRetryableStreamProvider("openrouter")
        service = LLMService(
            settings=LLMSettings(
                default_provider="openrouter",
                retry_attempts=1,
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(provider,),
        )

        events = [
            event
            async for event in service.stream_generate(
                LLMRequest(
                    model="z-ai/glm-5.2",
                    messages=(LLMMessage.text("user", "hello"),),
                )
            )
        ]

        self.assertEqual(provider.stream_attempts, 2)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], DoneEvent)

    async def test_stream_generate_filters_reasoning_activity(self) -> None:
        service = LLMService(
            settings=LLMSettings(
                default_provider="openrouter",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(_ReasoningActivityProvider("openrouter"),),
        )

        events = [
            event
            async for event in service.stream_generate(
                LLMRequest(
                    model="z-ai/glm-5.2",
                    messages=(LLMMessage.text("user", "hello"),),
                )
            )
        ]

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], DoneEvent)

    async def test_stream_generate_does_not_retry_after_visible_output(self) -> None:
        provider = _PartialOutputThenRetryableStreamProvider("openrouter")
        service = LLMService(
            settings=LLMSettings(
                default_provider="openrouter",
                retry_attempts=2,
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(provider,),
        )

        events = []
        with self.assertRaisesRegex(ProviderTemporaryError, "transient stream failure"):
            async for event in service.stream_generate(
                LLMRequest(
                    model="z-ai/glm-5.2",
                    messages=(LLMMessage.text("user", "hello"),),
                )
            ):
                events.append(event)

        self.assertEqual(provider.stream_attempts, 1)
        self.assertEqual(
            [event.delta for event in events if isinstance(event, TextDeltaEvent)],
            ["partial"],
        )

    async def test_stream_generate_enforces_optional_generation_deadline(self) -> None:
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(_SlowProvider("openai"),),
        )

        with self.assertRaisesRegex(ProviderTimeoutError, "Generation timed out"):
            async for _event in service.stream_generate(
                LLMRequest(
                    messages=(LLMMessage.text("user", "hello"),),
                    generation_timeout_seconds=0.01,
                )
            ):
                pass

    async def test_generate_rejects_codex_backend_provider_with_clear_error(self) -> None:
        service = LLMService(
            settings=LLMSettings(
                default_provider="codex",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(_FakeProvider("openai"),),
        )

        with self.assertRaisesRegex(LLMConfigurationError, "Codex backend"):
            await service.generate(
                LLMRequest(
                    messages=(LLMMessage.text("user", "hello"),),
                )
            )
