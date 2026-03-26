"""Unit tests for LM Studio provider discovery and streaming behavior."""

from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from llm.config import LMStudioProviderSettings
from llm.errors import LLMConfigurationError
from llm.providers.lmstudio_provider import LMStudioProvider
from llm.types import (
    DoneEvent,
    LLMMessage,
    LLMRequest,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolDefinition,
    UsageDeltaEvent,
)


class _FakeJsonResponse:
    def __init__(
        self,
        payload: dict[str, object],
        *,
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self) -> dict[str, object]:
        return self._payload


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


class LMStudioProviderTests(unittest.TestCase):
    def test_stream_generate_discovers_loaded_model_and_uses_local_endpoints(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(base_url="http://localhost:1234/v1"),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            messages=(LLMMessage.text("user", "hello"),),
        )

        model_list_response = _FakeJsonResponse(
            {
                "models": [
                    {
                        "type": "llm",
                        "publisher": "ibm",
                        "key": "granite-4-micro",
                        "loaded_instances": [{"id": "granite-live"}],
                        "capabilities": {
                            "vision": False,
                            "trained_for_tool_use": True,
                        },
                    }
                ]
            }
        )
        stream_response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "gen_123",
                        "model": "granite-live",
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
                        "model": "granite-live",
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
                        "model": "granite-live",
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
                        "model": "granite-live",
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

        get_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        post_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def fake_get(*args, **kwargs):
            get_calls.append((args, kwargs))
            return model_list_response

        def fake_post(*args, **kwargs):
            post_calls.append((args, kwargs))
            return stream_response

        with patch("llm.providers.lmstudio_provider._running_in_container", return_value=False):
            with patch("llm.providers.lmstudio_provider.requests.get", side_effect=fake_get):
                with patch("llm.providers.lmstudio_provider.requests.post", side_effect=fake_post):
                    events = asyncio.run(self._collect_events(provider, request))

        self.assertEqual(get_calls[0][0][0], "http://localhost:1234/api/v1/models")
        self.assertEqual(post_calls[0][0][0], "http://localhost:1234/v1/chat/completions")
        self.assertEqual(
            get_calls[0][1]["headers"],
            {"Content-Type": "application/json"},
        )
        self.assertEqual(
            post_calls[0][1]["headers"],
            {"Content-Type": "application/json"},
        )
        self.assertEqual(post_calls[0][1]["json"]["model"], "granite-live")

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
        self.assertEqual(done.response.model, "granite-live")
        self.assertEqual(done.response.finish_reason, "stop")
        self.assertTrue(stream_response.closed)

    def test_loopback_base_url_rewrites_to_host_docker_internal_in_container(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(base_url="http://127.0.0.1:1234"),
            default_timeout_seconds=60.0,
        )

        with patch("llm.providers.lmstudio_provider._running_in_container", return_value=True):
            self.assertEqual(
                provider._server_base_url(),
                "http://host.docker.internal:1234",
            )

    def test_generate_raises_when_no_loaded_llm_is_available(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(messages=(LLMMessage.text("user", "hello"),))

        with patch(
            "llm.providers.lmstudio_provider.requests.get",
            return_value=_FakeJsonResponse(
                {
                    "models": [
                        {
                            "type": "llm",
                            "publisher": "ibm",
                            "key": "granite-4-micro",
                            "loaded_instances": [],
                            "capabilities": {
                                "vision": False,
                                "trained_for_tool_use": True,
                            },
                        }
                    ]
                }
            ),
        ):
            with self.assertRaisesRegex(LLMConfigurationError, "loaded LLM available"):
                asyncio.run(provider.generate(request))

    def test_generate_raises_when_multiple_loaded_llms_match_request(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(messages=(LLMMessage.text("user", "hello"),))

        with patch(
            "llm.providers.lmstudio_provider.requests.get",
            return_value=_FakeJsonResponse(
                {
                    "models": [
                        {
                            "type": "llm",
                            "publisher": "ibm",
                            "key": "granite-4-micro",
                            "loaded_instances": [{"id": "granite-live"}],
                            "capabilities": {
                                "vision": False,
                                "trained_for_tool_use": True,
                            },
                        },
                        {
                            "type": "llm",
                            "publisher": "qwen",
                            "key": "qwen3-14b",
                            "loaded_instances": [{"id": "qwen-live"}],
                            "capabilities": {
                                "vision": False,
                                "trained_for_tool_use": True,
                            },
                        },
                    ]
                }
            ),
        ):
            with self.assertRaisesRegex(LLMConfigurationError, "multiple loaded LLMs"):
                asyncio.run(provider.generate(request))

    def test_generate_requires_loaded_tool_capable_model_for_tool_requests(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
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

        with patch(
            "llm.providers.lmstudio_provider.requests.get",
            return_value=_FakeJsonResponse(
                {
                    "models": [
                        {
                            "type": "llm",
                            "publisher": "ibm",
                            "key": "granite-4-micro",
                            "loaded_instances": [{"id": "granite-live"}],
                            "capabilities": {
                                "vision": False,
                                "trained_for_tool_use": False,
                            },
                        }
                    ]
                }
            ),
        ):
            with self.assertRaisesRegex(LLMConfigurationError, "tool-capable"):
                asyncio.run(provider.generate(request))

    def test_stream_generate_assembles_tool_calls(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
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

        model_list_response = _FakeJsonResponse(
            {
                "models": [
                    {
                        "type": "llm",
                        "publisher": "ibm",
                        "key": "granite-4-micro",
                        "loaded_instances": [{"id": "granite-live"}],
                        "capabilities": {
                            "vision": False,
                            "trained_for_tool_use": True,
                        },
                    }
                ]
            }
        )
        stream_response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "gen_456",
                        "model": "granite-live",
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
                        "model": "granite-live",
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
                        "model": "granite-live",
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

        with patch(
            "llm.providers.lmstudio_provider.requests.get",
            return_value=model_list_response,
        ):
            with patch(
                "llm.providers.lmstudio_provider.requests.post",
                return_value=stream_response,
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

    def test_normalize_chat_response_supports_legacy_function_call_shape(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="granite-live",
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

        response = provider._normalize_chat_response(
            request=request,
            response_json={
                "id": "resp_123",
                "model": "granite-live",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "function_call": {
                                "name": "bash",
                                "arguments": '{"command":"pwd"}',
                            },
                        },
                        "finish_reason": "function_call",
                    }
                ],
            },
        )

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "bash")
        self.assertEqual(response.tool_calls[0].arguments, {"command": "pwd"})

    async def _collect_events(self, provider: LMStudioProvider, request: LLMRequest) -> list[object]:
        events: list[object] = []
        async for event in provider.stream_generate(request):
            events.append(event)
        return events

    def _sse_chunk(self, payload: dict[str, object]) -> str:
        return f"data: {json.dumps(payload)}"
