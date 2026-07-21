"""Unit tests for Grok provider request shaping and streaming behavior."""

from __future__ import annotations

import asyncio
import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from jarvis.llm.config import GrokProviderSettings
from jarvis.llm.errors import ProviderResponseError
from jarvis.llm.providers.grok_provider import (
    GrokProvider,
    GrokResponseStorageOverflowError,
)
from jarvis.llm.types import (
    DoneEvent,
    ImagePart,
    LLMMessage,
    LLMRequest,
    StatefulContinuation,
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


class _FakeWebSocket:
    def __init__(self, response_batches: list[list[dict[str, object]]]) -> None:
        self._response_batches = [list(batch) for batch in response_batches]
        self._current_events: list[dict[str, object]] = []
        self.sent_payloads: list[dict[str, object]] = []
        self.closed = False

    async def send(self, raw_payload: str) -> None:
        self.sent_payloads.append(json.loads(raw_payload))
        if not self._response_batches:
            raise AssertionError("Unexpected Grok WebSocket request.")
        self._current_events = self._response_batches.pop(0)

    async def recv(self) -> str:
        if not self._current_events:
            raise AssertionError("No queued Grok WebSocket event.")
        return json.dumps(self._current_events.pop(0))

    async def close(self) -> None:
        self.closed = True


class GrokProviderTests(unittest.TestCase):
    def test_build_response_create_kwargs_includes_reasoning_effort(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(reasoning_effort="high"),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.3",
            messages=(LLMMessage.text("user", "Hello"),),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)

        self.assertEqual(kwargs["reasoning"], {"effort": "high"})

    def test_build_response_create_kwargs_preserves_system_messages_in_order(
        self,
    ) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
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
            item["content"][0]["type"] for item in kwargs["input"] if item["type"] == "message"
        ]

        self.assertEqual(roles, ["system", "user", "system", "assistant", "user"])
        self.assertEqual(
            content_types,
            ["input_text", "input_text", "input_text", "output_text", "input_text"],
        )

    def test_assistant_history_prefers_persisted_response_output_items_for_replay(
        self,
    ) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-reasoning",
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

    def test_tool_roundtrip_uses_function_call_and_function_call_output_items(
        self,
    ) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
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
            read_timeout_seconds=60.0,
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

    def test_reasoning_models_request_encrypted_reasoning_and_enable_store(
        self,
    ) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)

        self.assertEqual(kwargs["include"], ["reasoning.encrypted_content"])
        self.assertTrue(kwargs["store"])

    def test_non_reasoning_models_omit_encrypted_reasoning_include(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)

        self.assertNotIn("include", kwargs)
        self.assertTrue(kwargs["store"])

    def test_stateful_continuation_uses_previous_response_id(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(LLMMessage.text("user", "Second turn"),),
            previous_response_id="resp_grok_123",
        )

        kwargs = provider._build_response_create_kwargs(request, stream=False)

        self.assertEqual(kwargs["previous_response_id"], "resp_grok_123")
        self.assertEqual(
            kwargs["input"],
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Second turn"}],
                }
            ],
        )

    def test_generate_uses_responses_prompt_cache_key(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
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

        self.assertEqual(fake_client.responses.calls[0]["prompt_cache_key"], "conv_123")
        self.assertEqual(fake_client.with_options_calls, [])
        self.assertEqual(normalized.text, "Hello")

    def test_normalize_response_persists_response_output_and_usage_details(
        self,
    ) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
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
            read_timeout_seconds=60.0,
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
            fake_client.responses.calls[0]["prompt_cache_key"],
            "conv_abc123",
        )
        self.assertEqual(fake_client.with_options_calls, [])
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
            read_timeout_seconds=60.0,
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
            read_timeout_seconds=60.0,
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

    def test_image_continuation_uses_store_false_websocket(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(api_key="test-key"),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.5",
            messages=(LLMMessage.text("user", "Inspect the attached image."),),
            prompt_cache_key="session_123",
            previous_response_id="resp_durable",
            stateful_continuation=StatefulContinuation(
                session_key="session_123",
                storage_mode="ephemeral",
                durable_response_id="resp_durable",
            ),
        )
        websocket = _FakeWebSocket(
            [
                [
                    {"type": "response.output_text.delta", "delta": "Looks good."},
                    {
                        "type": "response.completed",
                        "response": self._build_websocket_response(
                            response_id="resp_live_1",
                            output_text="Looks good.",
                        ),
                    },
                ]
            ]
        )

        with patch(
            "jarvis.llm.providers.grok_provider.connect",
            AsyncMock(return_value=websocket),
        ):
            events = asyncio.run(self._collect_events(provider, request))

        self.assertEqual(len(websocket.sent_payloads), 1)
        payload = websocket.sent_payloads[0]
        self.assertEqual(payload["type"], "response.create")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["previous_response_id"], "resp_durable")
        self.assertEqual(payload["prompt_cache_key"], "session_123")
        done = events[-1]
        self.assertIsInstance(done, DoneEvent)
        self.assertEqual(done.response.provider_metadata["response_storage_mode"], "ephemeral")
        self.assertEqual(done.response.provider_metadata["durable_response_id"], "resp_durable")
        self.assertEqual(done.response.provider_metadata["websocket_generation"], 1)

    def test_aged_websocket_is_not_evicted_while_session_lock_is_held(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(api_key="test-key"),
            read_timeout_seconds=60.0,
        )
        websocket = _FakeWebSocket([])

        async def exercise() -> None:
            lock = asyncio.Lock()
            await lock.acquire()
            session = SimpleNamespace(
                session_key="session_123",
                websocket=websocket,
                lock=lock,
                opened_at=-1_000_000_000.0,
                last_used_at=-1_000_000_000.0,
                generation=1,
                live_response_id="resp_live_1",
            )
            provider._websocket_sessions["session_123"] = session
            continuation = StatefulContinuation(
                session_key="session_123",
                storage_mode="ephemeral",
                durable_response_id="resp_durable",
            )

            resolved = await provider._websocket_session(continuation)

            self.assertIs(resolved, session)
            self.assertFalse(websocket.closed)
            lock.release()
            await provider.aclose()

        asyncio.run(exercise())

    def test_healthy_websocket_continuation_does_not_materialize_recovery_tail(
        self,
    ) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(api_key="test-key"),
            read_timeout_seconds=60.0,
        )
        websocket = _FakeWebSocket(
            [
                [
                    {
                        "type": "response.completed",
                        "response": self._build_websocket_response(
                            response_id="resp_live_1",
                            output_text="First.",
                        ),
                    }
                ],
                [
                    {
                        "type": "response.completed",
                        "response": self._build_websocket_response(
                            response_id="resp_live_2",
                            output_text="Second.",
                        ),
                    }
                ],
            ]
        )
        first_request = LLMRequest(
            model="grok-4.5",
            messages=(LLMMessage.text("user", "First."),),
            previous_response_id="resp_durable",
            stateful_continuation=StatefulContinuation(
                session_key="session_123",
                storage_mode="ephemeral",
                durable_response_id="resp_durable",
            ),
        )

        def fail_if_materialized():
            raise AssertionError("Healthy Grok continuation materialized its recovery tail.")

        second_request = LLMRequest(
            model="grok-4.5",
            messages=(LLMMessage.text("user", "Second."),),
            previous_response_id="resp_live_1",
            stateful_continuation=StatefulContinuation(
                session_key="session_123",
                storage_mode="ephemeral",
                durable_response_id="resp_durable",
                recovery_message_loader=fail_if_materialized,
            ),
        )

        async def exercise() -> None:
            await provider.generate(first_request)
            response = await provider.generate(second_request)
            self.assertEqual(response.response_id, "resp_live_2")
            await provider.aclose()

        with patch(
            "jarvis.llm.providers.grok_provider.connect",
            AsyncMock(return_value=websocket),
        ):
            asyncio.run(exercise())

        self.assertEqual(len(websocket.sent_payloads), 2)

    def test_websocket_reconnect_warms_bounded_tail_from_durable_anchor(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(api_key="test-key"),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.5",
            messages=(LLMMessage.text("user", "Continue."),),
            prompt_cache_key="session_123",
            previous_response_id="resp_live_1",
            stateful_continuation=StatefulContinuation(
                session_key="session_123",
                storage_mode="ephemeral",
                durable_response_id="resp_durable",
                recovery_messages=(
                    LLMMessage.text("user", "Earlier image/tool tail."),
                    LLMMessage.text("assistant", "Tail restored."),
                ),
                generation=3,
            ),
        )
        websocket = _FakeWebSocket(
            [
                [
                    {
                        "type": "response.completed",
                        "response": self._build_websocket_response(
                            response_id="resp_warmup",
                            output_text="",
                        ),
                    }
                ],
                [
                    {
                        "type": "response.completed",
                        "response": self._build_websocket_response(
                            response_id="resp_live_2",
                            output_text="Continued.",
                        ),
                    }
                ],
            ]
        )

        with patch(
            "jarvis.llm.providers.grok_provider.connect",
            AsyncMock(return_value=websocket),
        ):
            response = asyncio.run(provider.generate(request))

        self.assertEqual(len(websocket.sent_payloads), 2)
        warmup, generation = websocket.sent_payloads
        self.assertFalse(warmup["generate"])
        self.assertEqual(warmup["previous_response_id"], "resp_durable")
        self.assertEqual(len(warmup["input"]), 2)
        self.assertNotIn("generate", generation)
        self.assertEqual(generation["previous_response_id"], "resp_warmup")
        self.assertEqual(response.response_id, "resp_live_2")
        self.assertEqual(response.provider_metadata["websocket_generation"], 4)

    def test_storage_overflow_retries_store_false_and_suppresses_partial_delta(
        self,
    ) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(api_key="test-key"),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.5",
            messages=(LLMMessage.text("user", "Continue."),),
            prompt_cache_key="session_123",
            previous_response_id="resp_durable",
            stateful_continuation=StatefulContinuation(
                session_key="session_123",
                durable_response_id="resp_durable",
            ),
        )
        durable_stream = _FakeAsyncStream(
            [
                SimpleNamespace(type="response.output_text.delta", delta="Hel"),
                SimpleNamespace(
                    type="error",
                    message=(
                        "Response is too large to store. You can avoid this error by "
                        "setting store to false in your request."
                    ),
                ),
            ]
        )
        fake_client = _FakeClient(durable_stream)
        websocket = _FakeWebSocket(
            [
                [
                    {"type": "response.output_text.delta", "delta": "Hello"},
                    {
                        "type": "response.completed",
                        "response": self._build_websocket_response(
                            response_id="resp_live_1",
                            output_text="Hello",
                        ),
                    },
                ]
            ]
        )

        with (
            patch.object(provider, "_client_instance", AsyncMock(return_value=fake_client)),
            patch(
                "jarvis.llm.providers.grok_provider.connect",
                AsyncMock(return_value=websocket),
            ),
        ):
            events = asyncio.run(self._collect_events(provider, request))

        text_events = [event for event in events if isinstance(event, TextDeltaEvent)]
        self.assertEqual([event.delta for event in text_events], ["Hel", "lo"])
        self.assertTrue(fake_client.responses.calls[0]["store"])
        self.assertFalse(websocket.sent_payloads[0]["store"])
        done = events[-1]
        self.assertIsInstance(done, DoneEvent)
        self.assertEqual(done.response.provider_metadata["response_storage_mode"], "ephemeral")

    def test_storage_overflow_suppresses_retried_tool_delta_with_new_call_id(
        self,
    ) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(api_key="test-key"),
            read_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.5",
            messages=(LLMMessage.text("user", "Inspect."),),
            tools=(
                ToolDefinition(
                    name="view_image",
                    input_schema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
            ),
            prompt_cache_key="session_123",
            previous_response_id="resp_durable",
            stateful_continuation=StatefulContinuation(
                session_key="session_123",
                durable_response_id="resp_durable",
            ),
        )
        overflow_message = (
            "Response is too large to store. You can avoid this error by setting store "
            "to false in your request."
        )
        durable_stream = _FakeAsyncStream(
            [
                SimpleNamespace(
                    type="response.output_item.added",
                    item=SimpleNamespace(
                        type="function_call",
                        id="item_old",
                        call_id="call_old",
                        name="view_image",
                    ),
                ),
                SimpleNamespace(
                    type="response.function_call_arguments.done",
                    item_id="item_old",
                    name="view_image",
                    arguments='{"path":"image.png"}',
                ),
                SimpleNamespace(type="error", message=overflow_message),
            ]
        )
        fake_client = _FakeClient(durable_stream)
        websocket = _FakeWebSocket(
            [
                [
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "function_call",
                            "id": "item_new",
                            "call_id": "call_new",
                            "name": "view_image",
                        },
                    },
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": "item_new",
                        "name": "view_image",
                        "arguments": '{"path":"image.png"}',
                    },
                    {
                        "type": "response.completed",
                        "response": {
                            **self._build_websocket_response(
                                response_id="resp_live_1",
                                output_text="",
                            ),
                            "output": [
                                {
                                    "type": "function_call",
                                    "id": "item_new",
                                    "call_id": "call_new",
                                    "name": "view_image",
                                    "arguments": '{"path":"image.png"}',
                                }
                            ],
                        },
                    },
                ]
            ]
        )

        with (
            patch.object(provider, "_client_instance", AsyncMock(return_value=fake_client)),
            patch(
                "jarvis.llm.providers.grok_provider.connect",
                AsyncMock(return_value=websocket),
            ),
        ):
            events = asyncio.run(self._collect_events(provider, request))

        tool_events = [event for event in events if isinstance(event, ToolCallDeltaEvent)]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].call_id, "call_old")
        done = events[-1]
        self.assertIsInstance(done, DoneEvent)
        self.assertEqual(done.response.tool_calls[0].call_id, "call_new")

    def test_storage_overflow_error_has_structured_retry_metadata(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            read_timeout_seconds=60.0,
        )

        error = provider._stream_error_from_event(
            {
                "type": "error",
                "status": 400,
                "error": {
                    "message": (
                        "Response is too large to store. You can avoid this error by setting "
                        "store to false in your request."
                    ),
                    "param": "store",
                },
            }
        )

        self.assertIsInstance(error, GrokResponseStorageOverflowError)
        self.assertEqual(error.metadata["code"], "response_storage_too_large")
        self.assertTrue(error.metadata["retryable_with_store_false"])

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

    def _build_websocket_response(
        self,
        *,
        response_id: str,
        output_text: str,
    ) -> dict[str, object]:
        return {
            "id": response_id,
            "model": "grok-4.5",
            "status": "completed",
            "output_text": output_text,
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": output_text}],
                }
            ],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
            "incomplete_details": None,
        }
