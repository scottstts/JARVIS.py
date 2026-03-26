"""Unit tests for LM Studio provider discovery, statefulness, and streaming."""

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
    ToolCall,
    ToolCallDeltaEvent,
    ToolDefinition,
    ToolResultPart,
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
    def test_stream_generate_discovers_loaded_model_and_uses_responses_endpoint(self) -> None:
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
                self._sse_event(
                    {
                        "type": "response.created",
                        "response": {
                            "id": "resp_123",
                            "model": "granite-live",
                            "status": "in_progress",
                        },
                    }
                ),
                "",
                self._sse_event({"type": "response.output_text.delta", "delta": "Hel"}),
                "",
                self._sse_event({"type": "response.output_text.delta", "delta": "lo"}),
                "",
                self._sse_event(
                    {
                        "type": "response.completed",
                        "response": self._text_response_payload(
                            response_id="resp_123",
                            model="granite-live",
                            text="Hello",
                            usage={
                                "input_tokens": 3,
                                "output_tokens": 2,
                                "total_tokens": 5,
                            },
                        ),
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
        self.assertEqual(post_calls[0][0][0], "http://localhost:1234/v1/responses")
        self.assertEqual(
            get_calls[0][1]["headers"],
            {"Content-Type": "application/json"},
        )
        self.assertEqual(
            post_calls[0][1]["headers"],
            {"Content-Type": "application/json"},
        )
        self.assertEqual(post_calls[0][1]["json"]["model"], "granite-live")
        self.assertEqual(post_calls[0][1]["json"]["input"][0]["role"], "user")
        self.assertTrue(post_calls[0][1]["json"]["store"])

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
        self.assertEqual(done.response.response_id, "resp_123")
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
            tools=(self._bash_tool(),),
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

    def test_generate_uses_previous_response_id_for_append_only_history(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )

        first_request = LLMRequest(
            model="loaded-model",
            messages=(LLMMessage.text("user", "hello"),),
        )
        second_request = LLMRequest(
            model="loaded-model",
            messages=(
                LLMMessage.text("user", "hello"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        post_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        responses = [
            _FakeJsonResponse(
                self._text_response_payload(
                    response_id="resp_1",
                    model="loaded-model",
                    text="Hi there",
                )
            ),
            _FakeJsonResponse(
                self._text_response_payload(
                    response_id="resp_2",
                    model="loaded-model",
                    text="Still here",
                    previous_response_id="resp_1",
                )
            ),
        ]

        def fake_post(*args, **kwargs):
            post_calls.append((args, kwargs))
            return responses.pop(0)

        with patch("llm.providers.lmstudio_provider.requests.post", side_effect=fake_post):
            asyncio.run(provider.generate(first_request))
            asyncio.run(provider.generate(second_request))

        self.assertNotIn("previous_response_id", post_calls[0][1]["json"])
        self.assertEqual(
            post_calls[0][1]["json"]["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        )
        self.assertEqual(post_calls[1][1]["json"]["previous_response_id"], "resp_1")
        self.assertEqual(
            post_calls[1][1]["json"]["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Second turn"}],
                }
            ],
        )

    def test_generate_uses_previous_response_id_for_tool_result_followup(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        bash_tool = self._bash_tool()

        first_request = LLMRequest(
            model="loaded-model",
            messages=(LLMMessage.text("user", "run pwd"),),
            tools=(bash_tool,),
        )
        second_request = LLMRequest(
            model="loaded-model",
            messages=(
                LLMMessage.text("user", "run pwd"),
                LLMMessage(
                    role="assistant",
                    parts=(
                        ToolCall(
                            call_id="bash_1",
                            name="bash",
                            arguments={"command": "pwd"},
                            raw_arguments='{"command":"pwd"}',
                        ),
                    ),
                ),
                LLMMessage(
                    role="tool",
                    parts=(
                        ToolResultPart(
                            call_id="bash_1",
                            name="bash",
                            content="Bash execution result\nstatus: success",
                        ),
                    ),
                ),
            ),
            tools=(bash_tool,),
        )

        post_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        responses = [
            _FakeJsonResponse(
                self._tool_call_response_payload(
                    response_id="resp_tool_1",
                    model="loaded-model",
                    call_id="bash_1",
                    name="bash",
                    raw_arguments='{"command":"pwd"}',
                )
            ),
            _FakeJsonResponse(
                self._text_response_payload(
                    response_id="resp_tool_2",
                    model="loaded-model",
                    text="done",
                    previous_response_id="resp_tool_1",
                )
            ),
        ]

        def fake_post(*args, **kwargs):
            post_calls.append((args, kwargs))
            return responses.pop(0)

        with patch("llm.providers.lmstudio_provider.requests.post", side_effect=fake_post):
            asyncio.run(provider.generate(first_request))
            asyncio.run(provider.generate(second_request))

        self.assertEqual(post_calls[1][1]["json"]["previous_response_id"], "resp_tool_1")
        self.assertEqual(
            post_calls[1][1]["json"]["input"],
            [
                {
                    "type": "function_call_output",
                    "call_id": "bash_1",
                    "output": "Bash execution result\nstatus: success",
                }
            ],
        )

    def test_generate_retries_full_history_when_previous_response_id_is_stale(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )

        first_request = LLMRequest(
            model="loaded-model",
            messages=(LLMMessage.text("user", "hello"),),
        )
        second_request = LLMRequest(
            model="loaded-model",
            messages=(
                LLMMessage.text("user", "hello"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        post_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        responses: list[object] = [
            _FakeJsonResponse(
                self._text_response_payload(
                    response_id="resp_1",
                    model="loaded-model",
                    text="Hi there",
                )
            ),
            _FakeJsonResponse(
                {},
                status_code=400,
                text='{"error":{"message":"previous_response_id not found"}}',
            ),
            _FakeJsonResponse(
                self._text_response_payload(
                    response_id="resp_2",
                    model="loaded-model",
                    text="Recovered",
                )
            ),
        ]

        def fake_post(*args, **kwargs):
            post_calls.append((args, kwargs))
            response = responses.pop(0)
            if isinstance(response, _FakeJsonResponse) and response.status_code >= 400:
                return response
            return response

        with patch("llm.providers.lmstudio_provider.requests.post", side_effect=fake_post):
            asyncio.run(provider.generate(first_request))
            response = asyncio.run(provider.generate(second_request))

        self.assertEqual(response.text, "Recovered")
        self.assertEqual(post_calls[1][1]["json"]["previous_response_id"], "resp_1")
        self.assertNotIn("previous_response_id", post_calls[2][1]["json"])
        self.assertEqual(
            post_calls[2][1]["json"]["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi there"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Second turn"}],
                },
            ],
        )

    def test_generate_falls_back_to_full_history_when_prefix_does_not_match(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )

        first_request = LLMRequest(
            model="loaded-model",
            messages=(LLMMessage.text("user", "hello"),),
        )
        rewritten_history_request = LLMRequest(
            model="loaded-model",
            messages=(
                LLMMessage.text("user", "hello"),
                LLMMessage.text("assistant", "A different reply"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        post_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        responses = [
            _FakeJsonResponse(
                self._text_response_payload(
                    response_id="resp_1",
                    model="loaded-model",
                    text="Hi there",
                )
            ),
            _FakeJsonResponse(
                self._text_response_payload(
                    response_id="resp_2",
                    model="loaded-model",
                    text="Fresh pass",
                )
            ),
        ]

        def fake_post(*args, **kwargs):
            post_calls.append((args, kwargs))
            return responses.pop(0)

        with patch("llm.providers.lmstudio_provider.requests.post", side_effect=fake_post):
            asyncio.run(provider.generate(first_request))
            asyncio.run(provider.generate(rewritten_history_request))

        self.assertNotIn("previous_response_id", post_calls[1][1]["json"])
        self.assertEqual(len(post_calls[1][1]["json"]["input"]), 3)

    def test_stream_generate_assembles_tool_calls(self) -> None:
        provider = LMStudioProvider(
            settings=LMStudioProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="granite-live",
            messages=(LLMMessage.text("user", "run pwd"),),
            tools=(self._bash_tool(),),
        )

        stream_response = _FakeStreamingResponse(
            lines=[
                self._sse_event(
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "call_1",
                            "name": "bash",
                        },
                    }
                ),
                "",
                self._sse_event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": '{"command"',
                    }
                ),
                "",
                self._sse_event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": ':"pwd"}',
                    }
                ),
                "",
                self._sse_event(
                    {
                        "type": "response.completed",
                        "response": self._tool_call_response_payload(
                            response_id="resp_456",
                            model="granite-live",
                            call_id="call_1",
                            name="bash",
                            raw_arguments='{"command":"pwd"}',
                        ),
                    }
                ),
                "",
                "data: [DONE]",
                "",
            ]
        )

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

    async def _collect_events(self, provider: LMStudioProvider, request: LLMRequest) -> list[object]:
        events: list[object] = []
        async for event in provider.stream_generate(request):
            events.append(event)
        return events

    def _sse_event(self, payload: dict[str, object]) -> str:
        return f"data: {json.dumps(payload)}"

    def _bash_tool(self) -> ToolDefinition:
        return ToolDefinition(
            name="bash",
            description="Run bash.",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    def _text_response_payload(
        self,
        *,
        response_id: str,
        model: str,
        text: str,
        previous_response_id: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": response_id,
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
        }
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if usage is not None:
            payload["usage"] = usage
        return payload

    def _tool_call_response_payload(
        self,
        *,
        response_id: str,
        model: str,
        call_id: str,
        name: str,
        raw_arguments: str,
    ) -> dict[str, object]:
        return {
            "id": response_id,
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": f"item_{call_id}",
                    "call_id": call_id,
                    "name": name,
                    "arguments": raw_arguments,
                }
            ],
        }
