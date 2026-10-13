"""Native streaming tests for the Gemini provider adapter."""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from google.genai import errors as genai_errors

from jarvis.llm.config import GeminiProviderSettings
from jarvis.llm.errors import (
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderTemporaryError,
)
from jarvis.llm.providers.gemini_provider import GeminiProvider
from jarvis.llm.types import (
    DoneEvent,
    LLMMessage,
    LLMRequest,
    ProviderActivityEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolDefinition,
    UsageDeltaEvent,
)


class _FakeGeminiStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> "_FakeGeminiStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _FakeGeminiModels:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.stream_kwargs: dict[str, Any] | None = None

    async def generate_content(self, **kwargs: Any) -> Any:
        self.stream_kwargs = kwargs
        return self._chunks[-1]

    async def generate_content_stream(self, **kwargs: Any) -> _FakeGeminiStream:
        self.stream_kwargs = kwargs
        return _FakeGeminiStream(self._chunks)


def _candidate(*parts: Any, finish_reason: str | None = None) -> Any:
    return SimpleNamespace(
        content=SimpleNamespace(parts=list(parts)),
        finish_reason=(
            SimpleNamespace(value=finish_reason)
            if finish_reason is not None
            else None
        ),
    )


class GeminiProviderStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_cache_falls_back_for_generate_and_stream(self) -> None:
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                provider = GeminiProvider(
                    settings=GeminiProviderSettings(),
                    read_timeout_seconds=60.0,
                )
                request = LLMRequest(
                    model="gemini-4-argon",
                    messages=(
                        LLMMessage.text("system", "Turn context"),
                        LLMMessage.text("user", "Hello"),
                    ),
                    cached_content_messages=(
                        LLMMessage.text("system", "Stable system"),
                        LLMMessage.text("user", "Prior user"),
                        LLMMessage.text("assistant", "Prior assistant"),
                    ),
                    tools=(ToolDefinition(name="bash", input_schema={"type": "object"}),),
                    temperature=0.5,
                    max_output_tokens=100,
                )
                fake_models = _FakeGeminiModels([
                    SimpleNamespace(
                        candidates=[_candidate(SimpleNamespace(text="Done", thought=False))],
                    ),
                ])
                create_cache = AsyncMock(side_effect=genai_errors.ClientError(
                    400,
                    {"error": {"message": "Explicit caching is not supported for models/gemini-4-argon."}},
                ))
                provider._client = SimpleNamespace(  # type: ignore[assignment]
                    aio=SimpleNamespace(
                        models=fake_models,
                        caches=SimpleNamespace(create=create_cache),
                    ),
                )

                for _ in range(2):
                    if streaming:
                        events = [event async for event in provider.stream_generate(request)]
                        response = events[-1].response
                    else:
                        response = await provider.generate(request)
                    self.assertEqual(response.text, "Done")
                    self.assertNotIn("cached_content_name", response.provider_metadata)
                    kwargs = fake_models.stream_kwargs
                    self.assertEqual(kwargs["contents"], [
                        {"role": "user", "parts": [{"text": "Prior user"}]},
                        {"role": "model", "parts": [{"text": "Prior assistant"}]},
                        {"role": "user", "parts": [{"text": "Hello"}]},
                    ])
                    config = kwargs["config"]
                    self.assertNotIn("cached_content", config)
                    self.assertEqual(config["system_instruction"], "Stable system\n\nTurn context")
                    self.assertEqual(config["tools"][0]["function_declarations"][0]["name"], "bash")
                    self.assertIn("tool_config", config)
                    self.assertEqual(config["temperature"], 0.5)
                    self.assertEqual(config["max_output_tokens"], 100)
                create_cache.assert_awaited_once()
                self.assertEqual(len(request.cached_content_messages), 3)

                # Capability detection is per model; supported models still cache.
                create_cache.side_effect = None
                create_cache.return_value = SimpleNamespace(name="cachedContents/supported")
                supported = replace(request, model="gemini-3-flash-preview")
                response = await provider.generate(supported)
                self.assertEqual(create_cache.await_count, 2)
                self.assertEqual(response.provider_metadata["cached_content_name"], "cachedContents/supported")
                self.assertEqual(fake_models.stream_kwargs["config"]["cached_content"], "cachedContents/supported")
                self.assertNotIn("tools", fake_models.stream_kwargs["config"])

    async def test_other_cache_errors_propagate_without_generation(self) -> None:
        for code, message, expected in (
            (400, "Invalid tool schema", ProviderBadRequestError),
            (403, "Explicit caching is not supported", ProviderAuthenticationError),
            (429, "Quota exceeded", ProviderRateLimitError),
            (500, "Internal error", ProviderTemporaryError),
        ):
            with self.subTest(code=code):
                provider = GeminiProvider(
                    settings=GeminiProviderSettings(),
                    read_timeout_seconds=60.0,
                )
                error_type = genai_errors.ServerError if code >= 500 else genai_errors.ClientError
                create_cache = AsyncMock(side_effect=error_type(code, {"error": {"message": message}}))
                generate = AsyncMock()
                provider._client = SimpleNamespace(  # type: ignore[assignment]
                    aio=SimpleNamespace(
                        models=SimpleNamespace(generate_content=generate, generate_content_stream=generate),
                        caches=SimpleNamespace(create=create_cache),
                    ),
                )
                request = LLMRequest(
                    model="gemini-4-argon",
                    messages=(LLMMessage.text("user", "Hello"),),
                    cached_content_messages=(LLMMessage.text("system", "Stable system"),),
                )
                with self.assertRaises(expected):
                    await provider.generate(request)
                with self.assertRaises(expected):
                    _ = [event async for event in provider.stream_generate(request)]
                self.assertEqual(create_cache.await_count, 2)
                generate.assert_not_awaited()

    async def test_stream_generate_aggregates_native_chunks_and_signatures(self) -> None:
        provider = GeminiProvider(
            settings=GeminiProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="gemini-3-flash-preview",
            messages=(LLMMessage.text("user", "run pwd"),),
            tools=(
                ToolDefinition(
                    name="bash",
                    input_schema={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                ),
            ),
        )

        chunks = [
            SimpleNamespace(
                response_id="resp_123",
                model_version="gemini-3-flash-preview-001",
                usage_metadata=None,
                candidates=[
                    _candidate(
                        SimpleNamespace(
                            text="private reasoning",
                            thought=True,
                            thought_signature=b"reasoning-signature",
                        )
                    )
                ],
            ),
            SimpleNamespace(
                response_id="resp_123",
                model_version="gemini-3-flash-preview-001",
                usage_metadata=None,
                candidates=[_candidate(SimpleNamespace(text="Done", thought=False))],
            ),
            SimpleNamespace(
                response_id="resp_123",
                model_version="gemini-3-flash-preview-001",
                usage_metadata=None,
                candidates=[
                    _candidate(
                        SimpleNamespace(
                            text=None,
                            thought=False,
                            thought_signature=b"tool-signature",
                            function_call=SimpleNamespace(
                                id="call_1",
                                name="bash",
                                args={"command": "pwd"},
                            ),
                        )
                    )
                ],
            ),
            SimpleNamespace(
                response_id="resp_123",
                model_version="gemini-3-flash-preview-001",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                    total_token_count=15,
                ),
                candidates=[
                    _candidate(
                        SimpleNamespace(
                            text="",
                            thought=False,
                            thought_signature=b"final-signature",
                        ),
                        finish_reason="STOP",
                    )
                ],
            ),
        ]
        fake_models = _FakeGeminiModels(chunks)
        provider._client = SimpleNamespace(  # type: ignore[assignment]
            aio=SimpleNamespace(models=fake_models)
        )

        streamed = [event async for event in provider.stream_generate(request)]

        self.assertIsNotNone(fake_models.stream_kwargs)
        self.assertEqual(fake_models.stream_kwargs["model"], request.model)
        self.assertEqual(
            [event.delta for event in streamed if isinstance(event, TextDeltaEvent)],
            ["Done"],
        )
        self.assertNotIn("private reasoning", repr(streamed))
        tool_events = [
            event for event in streamed if isinstance(event, ToolCallDeltaEvent)
        ]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].call_id, "call_1")
        self.assertEqual(tool_events[0].arguments_delta, '{"command": "pwd"}')
        self.assertGreaterEqual(
            len(
                [
                    event
                    for event in streamed
                    if isinstance(event, ProviderActivityEvent)
                ]
            ),
            2,
        )
        self.assertEqual(
            len([event for event in streamed if isinstance(event, UsageDeltaEvent)]),
            1,
        )
        self.assertIsInstance(streamed[-1], DoneEvent)
        done = streamed[-1].response
        self.assertEqual(done.text, "Done")
        self.assertEqual(done.finish_reason, "tool_calls")
        self.assertEqual(done.response_id, "resp_123")
        self.assertEqual(len(done.provider_metadata["thought_signatures_b64"]), 3)
        self.assertIn("thought_signature_b64", done.tool_calls[0].provider_metadata)

