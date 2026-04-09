"""Unit tests for Grok provider request shaping and streaming behavior."""

from __future__ import annotations

import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from jarvis.llm.config import GrokProviderSettings
from jarvis.llm.errors import ProviderResponseError
from jarvis.llm.providers.grok_provider import GrokProvider
from jarvis.llm.types import (
    DoneEvent,
    ImagePart,
    LLMMessage,
    LLMRequest,
    TextDeltaEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolDefinition,
    ToolResultPart,
    UsageDeltaEvent,
)


class _FakeAsyncStream:
    def __init__(self, events: list[object]) -> None:
        self._events = list(events)

    def __aiter__(self) -> "_FakeAsyncStream":
        return self

    async def __anext__(self) -> object:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


class _FakeResponsesResource:
    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


class _FakeClient:
    def __init__(self, result: object) -> None:
        self.responses = _FakeResponsesResource(result)
        self.with_options_calls: list[dict[str, object]] = []

    def with_options(self, **kwargs) -> "_FakeClient":
        self.with_options_calls.append(kwargs)
        return self


class GrokProviderTests(unittest.TestCase):
    def test_build_response_create_kwargs_preserves_system_messages_in_order(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(
                LLMMessage.text("system", "Program prompt"),
                LLMMessage.text("user", "Hello"),
                LLMMessage.text("system", "Turn context"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)
        roles = [item["role"] for item in kwargs["input"] if item["type"] == "message"]
        content_types = [
            item["content"][0]["type"]
            for item in kwargs["input"]
            if item["type"] == "message"
        ]

        self.assertEqual(roles, ["system", "user", "system", "assistant", "user"])
        self.assertEqual(
            content_types,
            ["input_text", "input_text", "input_text", "output_text", "input_text"],
        )

    def test_assistant_history_prefers_persisted_response_output_items_for_replay(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-reasoning",
            messages=(
                LLMMessage(
                    role="assistant",
                    parts=(ToolCall(
                        call_id="bash_1",
                        name="bash",
                        arguments={"command": "pwd"},
                        raw_arguments='{"command":"pwd"}',
                    ),),
                    metadata={
                        "provider": "grok",
                        "provider_metadata": {
                            "response_output": [
                                {
                                    "type": "reasoning",
                                    "status": "completed",
                                    "encrypted_content": "enc_blob",
                                },
                                {
                                    "type": "function_call",
                                    "call_id": "bash_1",
                                    "name": "bash",
                                    "arguments": '{"command":"pwd"}',
                                },
                            ]
                        },
                    },
                ),
            ),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)

        self.assertEqual(
            kwargs["input"],
            [
                {
                    "type": "reasoning",
                    "status": "completed",
                    "encrypted_content": "enc_blob",
                },
                {
                    "type": "function_call",
                    "call_id": "bash_1",
                    "name": "bash",
                    "arguments": '{"command":"pwd"}',
                },
            ],
        )

    def test_tool_roundtrip_uses_function_call_and_function_call_output_items(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(
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
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)
        content = kwargs["input"]

        self.assertEqual(content[0]["type"], "function_call")
        self.assertEqual(content[0]["call_id"], "bash_1")
        self.assertEqual(content[1]["type"], "function_call_output")
        self.assertEqual(content[1]["call_id"], "bash_1")

    def test_image_input_uses_input_image_items(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        image_url = f"data:image/png;base64,{base64.b64encode(b'png-bytes').decode('ascii')}"
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(
                LLMMessage(
                    role="user",
                    parts=(ImagePart(image_url=image_url, detail="original"),),
                ),
            ),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)
        image_item = kwargs["input"][0]["content"][0]

        self.assertEqual(image_item["type"], "input_image")
        self.assertEqual(image_item["image_url"], image_url)
        self.assertEqual(image_item["detail"], "high")

    def test_reasoning_models_request_encrypted_reasoning_and_disable_store(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)

        self.assertEqual(kwargs["include"], ["reasoning.encrypted_content"])
        self.assertFalse(kwargs["store"])

    def test_non_reasoning_models_omit_encrypted_reasoning_include(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)

        self.assertNotIn("include", kwargs)
        self.assertFalse(kwargs["store"])

    def test_generate_uses_prompt_cache_header(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
            prompt_cache_key="conv_123",
        )
        response = self._build_response(
            output_text="Hello",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ],
        )
        fake_client = _FakeClient(response)

        with patch.object(provider, "_client_instance", AsyncMock(return_value=fake_client)):
            normalized = asyncio.run(provider.generate(request))

        self.assertEqual(
            fake_client.with_options_calls[0]["default_headers"],
            {"x-grok-conv-id": "conv_123"},
        )
        self.assertEqual(normalized.text, "Hello")

    def test_normalize_response_persists_response_output_and_usage_details(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
        )
        response = self._build_response(
            output_text=None,
            output=[
                {
                    "type": "reasoning",
                    "status": "completed",
                    "encrypted_content": "enc_blob",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
            ],
            cached_tokens=2,
            reasoning_tokens=7,
        )

        normalized = provider._normalize_response(request=request, response=response)

        self.assertEqual(normalized.text, "Hello")
        self.assertEqual(normalized.provider_metadata["response_output"][0]["type"], "reasoning")
        self.assertEqual(normalized.provider_metadata["cached_tokens"], 2)
        self.assertEqual(normalized.provider_metadata["reasoning_tokens"], 7)
        self.assertEqual(normalized.usage.total_tokens, 5)

    def test_stream_generate_emits_text_usage_and_done_events(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
            prompt_cache_key="conv_abc123",
        )
        response = self._build_response(
            response_id="resp_123",
            output_text="Hello",
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                }
            ],
            cached_tokens=2,
        )
        stream = _FakeAsyncStream(
            [
                SimpleNamespace(type="response.output_text.delta", delta="Hel"),
                SimpleNamespace(type="response.output_text.delta", delta="lo"),
                SimpleNamespace(type="response.completed", response=response),
            ]
        )
        fake_client = _FakeClient(stream)

        with patch.object(provider, "_client_instance", AsyncMock(return_value=fake_client)):
            events = asyncio.run(self._collect_events(provider, request))

        self.assertEqual(
            fake_client.with_options_calls[0]["default_headers"],
            {"x-grok-conv-id": "conv_abc123"},
        )
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
        self.assertEqual(done.response.provider_metadata["cached_tokens"], 2)

    def test_stream_generate_assembles_tool_call_events(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
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
        response = self._build_response(
            response_id="resp_tool_123",
            output_text="",
            output=[
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "bash",
                    "arguments": '{"command":"pwd"}',
                }
            ],
        )
        stream = _FakeAsyncStream(
            [
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        type="function_call",
                        id="fc_1",
                        call_id="call_1",
                        name="bash",
                    ),
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.done",
                    item_id="fc_1",
                    name="bash",
                    arguments='{"command":"pwd"}',
                ),
                SimpleNamespace(type="response.completed", response=response),
            ]
        )
        fake_client = _FakeClient(stream)

        with patch.object(provider, "_client_instance", AsyncMock(return_value=fake_client)):
            events = asyncio.run(self._collect_events(provider, request))

        tool_events = [event for event in events if isinstance(event, ToolCallDeltaEvent)]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].call_id, "call_1")
        self.assertEqual(tool_events[0].arguments_delta, '{"command":"pwd"}')

        self.assertIsInstance(events[-1], DoneEvent)
        done = events[-1]
        self.assertEqual(done.response.finish_reason, "tool_calls")
        self.assertEqual(len(done.response.tool_calls), 1)
        self.assertEqual(done.response.tool_calls[0].call_id, "call_1")

    def test_stream_generate_raises_on_failed_event(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
        )
        stream = _FakeAsyncStream(
            [
                SimpleNamespace(
                    type="response.failed",
                    response=SimpleNamespace(error=SimpleNamespace(message="Upstream failed")),
                )
            ]
        )
        fake_client = _FakeClient(stream)

        with patch.object(provider, "_client_instance", AsyncMock(return_value=fake_client)):
            with self.assertRaisesRegex(ProviderResponseError, "Upstream failed"):
                asyncio.run(self._collect_events(provider, request))

    async def _collect_events(
        self,
        provider: GrokProvider,
        request: LLMRequest,
    ) -> list[object]:
        return [event async for event in provider.stream_generate(request)]

    def _build_response(
        self,
        *,
        response_id: str = "resp_1",
        model: str = "grok-4.20-0309-non-reasoning",
        status: str = "completed",
        output_text: str | None,
        output: list[dict[str, object]],
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=response_id,
            model=model,
            status=status,
            output_text=output_text,
            output=[SimpleNamespace(**item) for item in output],
            usage=SimpleNamespace(
                input_tokens=3,
                output_tokens=2,
                total_tokens=5,
                input_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
                output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
            ),
            incomplete_details=None,
        )
