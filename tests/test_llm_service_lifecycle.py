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
    ProviderActivityEvent,
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


class _SuccessfulProvider(_FakeProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.request_deadline_seconds: float | None = None

    async def generate(self, request):
        self.request_deadline_seconds = request.deadline_seconds
        return LLMResponse(
            provider=self.name,
            model=request.model or "success-model",
            text="complete",
            tool_calls=[],
            finish_reason="stop",
            usage=LLMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
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


class _PartialOutputThenRetryableStreamProvider(_FakeProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.stream_attempts = 0

    async def stream_generate(self, request):
        self.stream_attempts += 1
        yield TextDeltaEvent(delta="partial")
        raise ProviderTemporaryError("transient stream failure")


class _ActivityThenRetryableStreamProvider(_FakeProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.stream_attempts = 0

    async def stream_generate(self, request):
        self.stream_attempts += 1
        yield ProviderActivityEvent(
            provider_event_type="response.created",
            response_id="resp_accepted",
        )
        raise ProviderTemporaryError("failure after provider acceptance")


class _ContinuouslyActiveStreamProvider(_FakeProvider):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.stream_closed = False

    async def stream_generate(self, request):
        try:
            while True:
                await asyncio.sleep(0.005)
                yield ProviderActivityEvent(
                    provider_event_type="reasoning.delta",
                    response_id="resp_active",
                )
        finally:
            self.stream_closed = True


class _TimeoutThenSuccessfulProvider(_SuccessfulProvider):
    def __init__(self, name: str, *, timeout_kind: str) -> None:
        super().__init__(name)
        self.timeout_kind = timeout_kind
        self.generate_attempts = 0

    async def generate(self, request):
        self.generate_attempts += 1
        if self.generate_attempts == 1:
            raise ProviderTimeoutError(
                "transport timeout",
                metadata={"timeout_kind": self.timeout_kind},
            )
        return await super().generate(request)


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
                    deadline_seconds=0.01,
                )
            )

    async def test_generate_resolves_default_request_deadline(self) -> None:
        provider = _SuccessfulProvider("openai")
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(provider,),
        )

        response = await service.generate(
            LLMRequest(messages=(LLMMessage.text("user", "hello"),))
        )

        self.assertEqual(response.text, "complete")
        self.assertEqual(provider.request_deadline_seconds, 3600.0)

    async def test_stream_generate_maps_request_deadline_to_provider_timeout(self) -> None:
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                embedding=EmbeddingSettings(provider="openai", model="text-embedding-test"),
            ),
            providers=(_SlowProvider("openai"),),
        )

        with self.assertRaises(ProviderTimeoutError):
            async for _event in service.stream_generate(
                LLMRequest(
                    messages=(LLMMessage.text("user", "hello"),),
                    deadline_seconds=0.01,
                )
            ):
                pass

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

    async def test_stream_activity_is_internal_and_prevents_blind_retry(self) -> None:
        provider = _ActivityThenRetryableStreamProvider("openai")
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                retry_attempts=2,
                embedding=EmbeddingSettings(
                    provider="openai",
                    model="text-embedding-test",
                ),
            ),
            providers=(provider,),
        )

        events = []
        with self.assertRaises(ProviderTemporaryError) as raised:
            async for event in service.stream_generate(
                LLMRequest(
                    model="gpt-5.4-2026-03-05",
                    messages=(LLMMessage.text("user", "hello"),),
                )
            ):
                events.append(event)

        self.assertEqual(events, [])
        self.assertEqual(provider.stream_attempts, 1)
        self.assertTrue(raised.exception.metadata["accepted"])
        self.assertFalse(raised.exception.metadata["emitted_output"])
        self.assertEqual(
            raised.exception.metadata["last_provider_event_type"],
            "response.created",
        )
        self.assertEqual(raised.exception.metadata["response_id"], "resp_accepted")

    async def test_stream_activity_does_not_extend_absolute_deadline(self) -> None:
        provider = _ContinuouslyActiveStreamProvider("openai")
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                retry_attempts=2,
                embedding=EmbeddingSettings(
                    provider="openai",
                    model="text-embedding-test",
                ),
            ),
            providers=(provider,),
        )

        with self.assertRaises(ProviderTimeoutError) as raised:
            async for _event in service.stream_generate(
                LLMRequest(
                    model="gpt-5.4-2026-03-05",
                    messages=(LLMMessage.text("user", "hello"),),
                    deadline_seconds=0.03,
                )
            ):
                pass

        self.assertEqual(raised.exception.metadata["timeout_kind"], "request_deadline")
        self.assertEqual(raised.exception.metadata["request_deadline_seconds"], 0.03)
        self.assertTrue(raised.exception.metadata["accepted"])
        self.assertFalse(raised.exception.metadata["emitted_output"])
        self.assertEqual(
            raised.exception.metadata["last_provider_event_type"],
            "reasoning.delta",
        )
        self.assertTrue(provider.stream_closed)

    async def test_generate_retries_connect_timeout(self) -> None:
        provider = _TimeoutThenSuccessfulProvider("openai", timeout_kind="connect")
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                retry_attempts=1,
                retry_backoff_seconds=0,
                embedding=EmbeddingSettings(
                    provider="openai",
                    model="text-embedding-test",
                ),
            ),
            providers=(provider,),
        )

        response = await service.generate(
            LLMRequest(messages=(LLMMessage.text("user", "hello"),))
        )

        self.assertEqual(response.text, "complete")
        self.assertEqual(provider.generate_attempts, 2)

    async def test_generate_does_not_blindly_retry_read_timeout(self) -> None:
        provider = _TimeoutThenSuccessfulProvider(
            "openai",
            timeout_kind="read_idle",
        )
        service = LLMService(
            settings=LLMSettings(
                default_provider="openai",
                retry_attempts=2,
                retry_backoff_seconds=0,
                embedding=EmbeddingSettings(
                    provider="openai",
                    model="text-embedding-test",
                ),
            ),
            providers=(provider,),
        )

        with self.assertRaises(ProviderTimeoutError):
            await service.generate(
                LLMRequest(messages=(LLMMessage.text("user", "hello"),))
            )

        self.assertEqual(provider.generate_attempts, 1)

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
