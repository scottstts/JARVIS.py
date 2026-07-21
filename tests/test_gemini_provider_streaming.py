"""Native streaming tests for the Gemini provider adapter."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from jarvis.llm.config import GeminiProviderSettings
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

