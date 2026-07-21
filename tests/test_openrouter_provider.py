"""Unit tests for OpenRouter provider streaming behavior."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from jarvis.llm.config import OpenRouterProviderSettings
from jarvis.llm.errors import (
    ProviderResponseError,
    ProviderTemporaryError,
)
from jarvis.llm.providers.openrouter_provider import OpenRouterProvider
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


class _FakeStreamingResponse:
    def __init__(
        self,
        *,
        lines: list[str],
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._lines = lines
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line if decode_unicode else line.encode("utf-8")

    def close(self) -> None:
        self.closed = True


class _FakeJsonResponse:
    def __init__(
        self,
        *,
        data: dict[str, object],
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self._data = data
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    def json(self) -> dict[str, object]:
        return self._data


class OpenRouterProviderStreamingTests(unittest.TestCase):
    def test_request_context_includes_openrouter_attribution_headers(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(
                api_key="test-key",
                site_url="https://jarvis.example",
                app_name="Jarvis",
            ),
            read_timeout_seconds=60.0,
        )

        _url, headers, timeout = provider._build_request_context(
            endpoint="/chat/completions",
        )

        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["HTTP-Referer"], "https://jarvis.example")
        self.assertEqual(headers["X-OpenRouter-Cache"], "true")
        self.assertEqual(headers["X-OpenRouter-Title"], "Jarvis")
        self.assertEqual(headers["X-Title"], "Jarvis")
        self.assertEqual(timeout, (30.0, 60.0))

    def test_generate_sends_response_cache_header_and_surfaces_cache_metadata(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="openai/gpt-4o-mini",
            messages=(LLMMessage.text("user", "hello"),),
        )

        captured_request: dict[str, object] = {}
        response = _FakeJsonResponse(
            data={
                "id": "gen_123",
                "model": "openai/gpt-4o-mini",
                "choices": [
                    {
                        "message": {"content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
            headers={
                "X-OpenRouter-Cache-Status": "HIT",
                "X-OpenRouter-Cache-Age": "12",
                "X-OpenRouter-Cache-TTL": "300",
                "X-Generation-Id": "gen_header_123",
            },
        )

        def fake_post(*args, **kwargs):
            captured_request.update(kwargs)
            return response

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "jarvis.llm.providers.openrouter_provider.requests.post",
                side_effect=fake_post,
            ):
                result = asyncio.run(provider.generate(request))

        headers = captured_request["headers"]
        self.assertEqual(headers["X-OpenRouter-Cache"], "true")
        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.provider_metadata["openrouter_cache_status"], "HIT")
        self.assertEqual(result.provider_metadata["openrouter_cache_age_seconds"], 12)
        self.assertEqual(result.provider_metadata["openrouter_cache_ttl_seconds"], 300)
        self.assertEqual(result.provider_metadata["openrouter_generation_id"], "gen_header_123")

    def test_generate_preserves_http_504_metadata(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="z-ai/glm-5.2",
            messages=(LLMMessage.text("user", "hello"),),
        )
        response = _FakeJsonResponse(
            data={
                "error": {
                    "code": 504,
                    "message": "Upstream idle timeout exceeded",
                    "metadata": {
                        "error_type": "timeout",
                        "provider_code": "upstream_idle_timeout",
                        "provider_name": "Z.ai",
                    },
                }
            },
            status_code=504,
            headers={"X-Generation-Id": "gen_header_504"},
        )

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch(
                "jarvis.llm.providers.openrouter_provider.requests.post",
                return_value=response,
            ):
                with self.assertRaises(ProviderTemporaryError) as caught:
                    asyncio.run(provider.generate(request))

        self.assertEqual(caught.exception.metadata["generation_id"], "gen_header_504")
        self.assertEqual(caught.exception.metadata["http_code"], 504)
        self.assertEqual(caught.exception.metadata["error_type"], "timeout")

    def test_anthropic_model_payload_enables_prompt_cache_and_sticky_session(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="anthropic/claude-4.5-sonnet",
            messages=(
                LLMMessage.text("system", "System prompt"),
                LLMMessage.text("user", "hello"),
            ),
            prompt_cache_key="session_123",
        )

        payload = provider._build_chat_payload(request)

        self.assertEqual(payload["cache_control"], {"type": "ephemeral"})
        self.assertEqual(payload["session_id"], "session_123")
        self.assertNotIn("provider", payload)

    def test_non_anthropic_model_payload_omits_prompt_cache_control(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="openai/gpt-4o-mini",
            messages=(LLMMessage.text("user", "hello"),),
            prompt_cache_key="session_123",
        )

        payload = provider._build_chat_payload(request)

        self.assertNotIn("cache_control", payload)
        self.assertEqual(payload["session_id"], "session_123")

    def test_stream_generate_emits_text_usage_and_done_events(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
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
            ],
            headers={
                "X-OpenRouter-Cache-Status": "HIT",
                "X-OpenRouter-Cache-Age": "12",
                "X-OpenRouter-Cache-TTL": "300",
                "X-Generation-Id": "gen_header_123",
            },
        )

        def fake_post(*args, **kwargs):
            captured_request.update(kwargs)
            return response

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with patch("jarvis.llm.providers.openrouter_provider.requests.post", side_effect=fake_post):
                events = asyncio.run(self._collect_events(provider, request))

        self.assertTrue(captured_request["stream"])
        self.assertTrue(captured_request["json"]["stream"])
        self.assertEqual(captured_request["headers"]["X-OpenRouter-Cache"], "true")
        activity_types = [
            event.provider_event_type
            for event in events
            if isinstance(event, ProviderActivityEvent)
        ]
        self.assertIn("http.response_headers", activity_types)
        self.assertIn("OPENROUTER PROCESSING", activity_types)
        self.assertIn("sse.done", activity_types)
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
        self.assertEqual(done.response.provider_metadata["openrouter_cache_status"], "HIT")
        self.assertEqual(done.response.provider_metadata["openrouter_cache_age_seconds"], 12)
        self.assertEqual(done.response.provider_metadata["openrouter_cache_ttl_seconds"], 300)
        self.assertEqual(
            done.response.provider_metadata["openrouter_generation_id"],
            "gen_header_123",
        )
        self.assertTrue(response.closed)

    def test_stream_generate_assembles_streamed_tool_calls(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
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
            read_timeout_seconds=60.0,
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

    def test_stream_generate_ignores_private_reasoning_chunks(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="z-ai/glm-5.2",
            messages=(LLMMessage.text("user", "hello"),),
        )
        response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "gen_reasoning",
                        "model": "z-ai/glm-5.2",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "reasoning_details": [
                                        {
                                            "type": "reasoning.text",
                                            "text": "private reasoning",
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
                        "id": "gen_reasoning",
                        "model": "z-ai/glm-5.2",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Hello"},
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                "",
                self._sse_chunk(
                    {
                        "id": "gen_reasoning",
                        "model": "z-ai/glm-5.2",
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

        self.assertNotIn("private reasoning", repr(events))
        self.assertEqual(
            [event.delta for event in events if isinstance(event, TextDeltaEvent)],
            ["Hello"],
        )

    def test_stream_generate_preserves_structured_error_metadata(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="z-ai/glm-5.2",
            messages=(LLMMessage.text("user", "hello"),),
        )
        cases = (
            ("timeout", 504),
            ("provider_unavailable", 502),
            ("provider_overloaded", 503),
            ("rate_limit_exceeded", 429),
        )

        for error_type, http_code in cases:
            with self.subTest(error_type=error_type):
                response = _FakeStreamingResponse(
                    lines=[
                        self._sse_chunk(
                            {
                                "id": f"gen_{error_type}",
                                "provider": "Z.ai",
                                "model": "z-ai/glm-5.2",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "error",
                                    }
                                ],
                                "error": {
                                    "code": http_code,
                                    "message": f"{error_type} message",
                                    "metadata": {
                                        "error_type": error_type,
                                        "provider_code": f"upstream_{error_type}",
                                        "provider_name": "Z.ai",
                                    },
                                },
                            }
                        ),
                        "",
                    ],
                    headers={"X-Generation-Id": f"header_{error_type}"},
                )

                with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
                    with patch(
                        "jarvis.llm.providers.openrouter_provider.requests.post",
                        return_value=response,
                    ):
                        with self.assertRaises(ProviderResponseError) as caught:
                            asyncio.run(self._collect_events(provider, request))

                self.assertIs(type(caught.exception), ProviderResponseError)
                self.assertEqual(
                    caught.exception.metadata,
                    {
                        "generation_id": f"header_{error_type}",
                        "response_id": f"gen_{error_type}",
                        "provider_name": "Z.ai",
                        "http_code": http_code,
                        "error_type": error_type,
                        "upstream_provider_code": f"upstream_{error_type}",
                    },
                )

    def test_stream_generate_raises_on_stream_error_chunk(self) -> None:
        provider = OpenRouterProvider(
            settings=OpenRouterProviderSettings(),
            read_timeout_seconds=60.0,
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
