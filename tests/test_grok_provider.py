"""Unit tests for Grok provider request shaping and streaming behavior."""

from __future__ import annotations

import asyncio
import base64
import json
import unittest
from unittest.mock import patch

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


class GrokProviderTests(unittest.TestCase):
    def test_request_context_includes_prompt_cache_header(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(api_key="test-key"),
            default_timeout_seconds=60.0,
        )

        _url, headers, timeout = provider._build_request_context(
            endpoint="/chat/completions",
            timeout_seconds=None,
            prompt_cache_key="conv_123",
        )

        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertEqual(headers["x-grok-conv-id"], "conv_123")
        self.assertEqual(timeout, 60.0)

    def test_build_chat_payload_collapses_system_messages_to_first_message(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(
                LLMMessage.text("system", "Program prompt"),
                LLMMessage.text("system", "Armor prompt"),
                LLMMessage.text("user", "Hello"),
                LLMMessage.text("assistant", "Hi there"),
                LLMMessage.text("user", "Second turn"),
            ),
        )

        payload = provider._build_chat_payload(request, stream=False)

        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(
            payload["messages"][0]["content"],
            "Program prompt\n\nArmor prompt",
        )
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(payload["messages"][2]["role"], "assistant")
        self.assertEqual(payload["messages"][3]["role"], "user")

    def test_tool_roundtrip_uses_openai_compatible_chat_messages(self) -> None:
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

        payload = provider._build_chat_payload(request, stream=False)
        assistant_message = payload["messages"][0]
        tool_message = payload["messages"][1]

        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(assistant_message["tool_calls"][0]["id"], "bash_1")
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "bash_1")

    def test_image_input_uses_image_url_with_normalized_detail(self) -> None:
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

        payload = provider._build_chat_payload(request, stream=False)
        image_part = payload["messages"][0]["content"][0]

        self.assertEqual(image_part["type"], "image_url")
        self.assertEqual(image_part["image_url"]["url"], image_url)
        self.assertEqual(image_part["image_url"]["detail"], "high")

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

        captured_request: dict[str, object] = {}
        response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "chatcmpl_123",
                        "model": "grok-4.20-0309-non-reasoning",
                        "system_fingerprint": "fp_test123",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Hel"},
                                "finish_reason": None,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 1,
                            "total_tokens": 4,
                            "prompt_tokens_details": {"cached_tokens": 0},
                        },
                    }
                ),
                "",
                self._sse_chunk(
                    {
                        "id": "chatcmpl_123",
                        "model": "grok-4.20-0309-non-reasoning",
                        "system_fingerprint": "fp_test123",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "lo"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                            "prompt_tokens_details": {"cached_tokens": 2},
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

        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            with patch("jarvis.llm.providers.grok_provider.requests.post", side_effect=fake_post):
                events = asyncio.run(self._collect_events(provider, request))

        self.assertTrue(captured_request["stream"])
        self.assertTrue(captured_request["json"]["stream"])
        self.assertEqual(captured_request["headers"]["x-grok-conv-id"], "conv_abc123")
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
        self.assertEqual(done.response.provider_metadata["system_fingerprint"], "fp_test123")
        self.assertEqual(done.response.provider_metadata["cached_tokens"], 2)
        self.assertTrue(response.closed)

    def test_stream_generate_assembles_single_chunk_tool_calls(self) -> None:
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

        response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "chatcmpl_456",
                        "model": "grok-4.20-0309-non-reasoning",
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
                                                "arguments": '{"command":"pwd"}',
                                            },
                                        }
                                    ]
                                },
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

        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            with patch(
                "jarvis.llm.providers.grok_provider.requests.post",
                return_value=response,
            ):
                events = asyncio.run(self._collect_events(provider, request))

        tool_events = [event for event in events if isinstance(event, ToolCallDeltaEvent)]
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].arguments_delta, '{"command":"pwd"}')

        self.assertIsInstance(events[-1], DoneEvent)
        done = events[-1]
        self.assertEqual(done.response.finish_reason, "tool_calls")
        self.assertEqual(len(done.response.tool_calls), 1)
        self.assertEqual(done.response.tool_calls[0].call_id, "call_1")
        self.assertEqual(done.response.tool_calls[0].arguments, {"command": "pwd"})

    def test_stream_generate_raises_on_stream_error_chunk(self) -> None:
        provider = GrokProvider(
            settings=GrokProviderSettings(),
            default_timeout_seconds=60.0,
        )
        request = LLMRequest(
            model="grok-4.20-0309-non-reasoning",
            messages=(LLMMessage.text("user", "hello"),),
        )

        response = _FakeStreamingResponse(
            lines=[
                self._sse_chunk(
                    {
                        "id": "chatcmpl_err",
                        "model": "grok-4.20-0309-non-reasoning",
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

        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            with patch(
                "jarvis.llm.providers.grok_provider.requests.post",
                return_value=response,
            ):
                with self.assertRaisesRegex(ProviderResponseError, "Upstream failed"):
                    asyncio.run(self._collect_events(provider, request))

    async def _collect_events(
        self,
        provider: GrokProvider,
        request: LLMRequest,
    ) -> list[object]:
        return [event async for event in provider.stream_generate(request)]

    def _sse_chunk(self, payload: dict[str, object]) -> str:
        return f"data: {json.dumps(payload)}"
