"""Unit tests for OpenRouter provider streaming behavior."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from jarvis.llm.config import OpenRouterProviderSettings
from jarvis.llm.errors import ProviderResponseError
from jarvis.llm.providers.openrouter_provider import OpenRouterProvider
from jarvis.llm.types import (
    DoneEvent,
    LLMMessage,
    LLMRequest,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolDefinition,
    UsageDeltaEvent,
)


class _FakeStreamingResponse:
    def __init__(
        self,
        *,
        lines: list[str],
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._lines = lines
        self.status_code = status_code
        self.text = text
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line if decode_unicode else line.encode("utf-8")

    def close(self) -> None:
        self.closed = True


class OpenRouterProviderStreamingTests(unittest.TestCase):
    def test_stream_generate_emits_text_usage_and_done_events(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="openai/gpt-4o-mini",
            messages=(LLMMessage.text("user", "hello"),),
        )

        captured_request: dict[str, object] = {}
        response = _FakeStreamingResponse(
            lines=[
                ": OPENROUTER PROCESSING",
                "",
                self._sse_chunk(
                    {
                        "id": "gen_123",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Hel"},
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                "",
                self._sse_chunk(
                    {
                        "id": "gen_123",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "lo"},
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                "",
                self._sse_chunk(
                    {
                        "id": "gen_123",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ),
                "",
                self._sse_chunk(
                    {
                        "id": "gen_123",
                        "model": "openai/gpt-4o-mini",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                ),
                "",
                "data: [DONE]",
                "",
            ]
        )

        def fake_post(*args, **kwargs):
            captured_request.update(kwargs)
            return response

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch("jarvis.llm.providers.openrouter_provider.requests.post", side_effect=fake_post):
                events = asyncio.run(self._collect_events(provider, request))

        self.assertTrue(captured_request["stream"])
        self.assertTrue(captured_request["json"]["stream"])
        self.assertEqual(
            [event.delta for event in events if isinstance(event, TextDeltaEvent)],
            ["Hel", "lo"],
        )

        usage_events = [event for event in events if isinstance(event, UsageDeltaEvent)]
        self.assertEqual(len(usage_events), 1)
        self.assertEqual(usage_events[0].usage.total_tokens, 5)

        self.assertIsInstance(events[-1], DoneEvent)
        done = events[-1]
        self.assertEqual(done.response.text, "Hello")
        self.assertEqual(done.response.finish_reason, "stop")
        self.assertEqual(done.response.usage.total_tokens, 5)
        self.assertEqual(done.response.response_id, "gen_123")
        self.assertTrue(response.closed)

    def test_stream_generate_assembles_streamed_tool_calls(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="openai/gpt-4o-mini",
            messages=(LLMMessage.text("user", "run pwd"),),
            tools=(
                ToolDefinition(
                    name="bash",
                    description="Run bash.",
                    input_schema={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                ),
            ),
        )

        response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "gen_456",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "bash",
                                                "arguments": '{"command"',
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                "",
                self._sse_chunk(
                    {
                        "id": "gen_456",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": ':"pwd"}'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                "",
                self._sse_chunk(
                    {
                        "id": "gen_456",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                ),
                "",
                "data: [DONE]",
                "",
            ]
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "jarvis.llm.providers.openrouter_provider.requests.post",
                return_value=response,
            ):
                events = asyncio.run(self._collect_events(provider, request))

        tool_events = [event for event in events if isinstance(event, ToolCallDeltaEvent)]
        self.assertEqual(
            [event.arguments_delta for event in tool_events],
            ['{"command"', ':"pwd"}'],
        )

        self.assertIsInstance(events[-1], DoneEvent)
        done = events[-1]
        self.assertEqual(done.response.finish_reason, "tool_calls")
        self.assertEqual(len(done.response.tool_calls), 1)
        self.assertEqual(done.response.tool_calls[0].call_id, "call_1")
        self.assertEqual(done.response.tool_calls[0].name, "bash")
        self.assertEqual(done.response.tool_calls[0].arguments, {"command": "pwd"})

    def test_stream_generate_preserves_utf8_text(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="openai/gpt-4o-mini",
            messages=(LLMMessage.text("user", "list tools"),),
        )

        response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "gen_utf8",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "bash — run commands"},
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                "",
                self._sse_chunk(
                    {
                        "id": "gen_utf8",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                ),
                "",
                "data: [DONE]",
                "",
            ]
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "jarvis.llm.providers.openrouter_provider.requests.post",
                return_value=response,
            ):
                events = asyncio.run(self._collect_events(provider, request))

        self.assertEqual(
            [event.delta for event in events if isinstance(event, TextDeltaEvent)],
            ["bash — run commands"],
        )
        self.assertIsInstance(events[-1], DoneEvent)
        done = events[-1]
        self.assertEqual(done.response.text, "bash — run commands")

    def test_stream_generate_raises_on_stream_error_chunk(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="openai/gpt-4o-mini",
            messages=(LLMMessage.text("user", "hello"),),
        )

        response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "gen_err",
                        "model": "openai/gpt-4o-mini",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "error",
                            }
                        ],
                        "error": {"message": "Upstream failed"},
                    }
                ),
                "",
            ]
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "jarvis.llm.providers.openrouter_provider.requests.post",
                return_value=response,
            ):
                with self.assertRaisesRegex(ProviderResponseError, "Upstream failed"):
                    asyncio.run(self._collect_events(provider, request))

    async def _collect_events(
        self,
        provider: OpenRouterProvider,
        request: LLMRequest,
    ) -> list[object]:
        return [event async for event in provider.stream_generate(request)]

    def _sse_chunk(self, payload: dict[str, object]) -> str:
        return f"data: {json.dumps(payload)}"
