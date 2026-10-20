"""Anthropic SDK integration without live API calls."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx2
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient, Timeout
from anthropic.types import Message
from jsonschema import Draft202012Validator

from jarvis.llm.config import AnthropicProviderSettings
from jarvis.llm.errors import LLMConfigurationError, ProviderTemporaryError, ProviderTimeoutError
from jarvis.llm.providers.anthropic_models import anthropic_model_capabilities
from jarvis.llm.providers.anthropic_provider import AnthropicProvider
from jarvis.llm.types import (
    DoneEvent,
    LLMMessage,
    LLMRequest,
    ProviderActivityEvent,
    TextDeltaEvent,
    TextPart,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    ToolResultPart,
)
from jarvis.llm.validation import TOOL_CALL_VALIDATION_ERROR_METADATA_KEY
from jarvis.tools import ToolRegistry, ToolSettings


def _request(model: str = "claude-opus-5-5", **kwargs: Any) -> LLMRequest:
    return LLMRequest(
        model=model,
        max_output_tokens=8192,
        messages=(LLMMessage.text("system", "Be helpful."), LLMMessage.text("user", "Hello")),
        **kwargs,
    )


def _message() -> Message:
    return Message.model_validate({
        "id": "msg_test", "type": "message", "role": "assistant",
        "model": "claude-opus-5-5", "stop_reason": "tool_use", "stop_sequence": None,
        "content": [
            {"type": "thinking", "thinking": "", "signature": "opaque-signature"},
            {"type": "text", "text": "Checking."},
            {"type": "redacted_thinking", "data": "opaque-redacted"},
            {"type": "tool_use", "id": "tool_1", "name": "lookup", "input": {"query": "hi"}},
            {"type": "text", "text": "One moment."},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5,
                  "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20},
    })


def _sse(message: Message) -> bytes:
    payload = message.model_dump(mode="json", exclude_none=True)
    start = {**payload, "content": [], "stop_reason": None,
             "usage": {**payload["usage"], "output_tokens": 0}}
    events = [{"type": "message_start", "message": start}]
    for index, block in enumerate(payload["content"]):
        events.append({"type": "content_block_start", "index": index, "content_block": block})
        events.append({"type": "content_block_stop", "index": index})
    events.extend([
        {"type": "message_delta", "delta": {"stop_reason": message.stop_reason, "stop_sequence": None},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ])
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


class AnthropicSDKTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_sends_actual_registry_schemas_without_root_combinators(self) -> None:
        tools = ToolRegistry.default(ToolSettings.from_workspace_dir(Path("/workspace"))).basic_definitions()
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                requests: list[dict[str, Any]] = []

                def handler(request: httpx2.Request) -> httpx2.Response:
                    payload = json.loads(request.content)
                    requests.append(payload)
                    for tool in payload["tools"]:
                        self.assertFalse({"allOf", "anyOf", "oneOf"}.intersection(tool["input_schema"]), tool["name"])
                    message = Message.model_validate({
                        **_message().model_dump(), "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "Ready"}],
                    })
                    if streaming:
                        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(message))
                    return httpx2.Response(200, json=message.model_dump(mode="json"))

                def make_client(**kwargs: Any) -> AsyncAnthropic:
                    return AsyncAnthropic(
                        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)), **kwargs,
                    )

                provider = AnthropicProvider(settings=AnthropicProviderSettings(api_key="test-key"), read_timeout_seconds=60)
                self.addAsyncCleanup(provider.aclose)
                with patch("jarvis.llm.providers.anthropic_provider.AsyncAnthropic", make_client):
                    if streaming:
                        events = [event async for event in provider.stream_generate(_request(tools=tools))]
                        response = events[-1].response
                    else:
                        response = await provider.generate(_request(tools=tools))
                self.assertEqual(response.text, "Ready")
                self.assertEqual(response.finish_reason, "stop")
                self.assertEqual(len(requests), 1)
                self.assertIn("file_patch", {tool["name"] for tool in requests[0]["tools"]})

    async def test_constructs_real_sdk_client_with_split_timeout(self) -> None:
        provider = AnthropicProvider(
            settings=AnthropicProviderSettings(api_key="test-key", base_url="https://example.test"),
            read_timeout_seconds=90, connect_timeout_seconds=12,
        )
        self.addAsyncCleanup(provider.aclose)
        client = await provider._client_instance()
        self.assertIsInstance(client.timeout, Timeout)
        self.assertEqual(client.timeout.as_dict(), {
            "connect": 12, "read": 90, "write": 12, "pool": 12,
        })
        self.assertEqual(client.max_retries, 0)
        self.assertIs(await provider._client_instance(), client)

    async def test_generate_and_stream_use_current_sdk_and_preserve_replay(self) -> None:
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                sent: list[dict[str, Any]] = []

                def handler(request: httpx2.Request) -> httpx2.Response:
                    sent.append(json.loads(request.content))
                    if streaming:
                        return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                              content=_sse(_message()))
                    return httpx2.Response(200, json=_message().model_dump(mode="json"))

                def make_client(**kwargs: Any) -> AsyncAnthropic:
                    return AsyncAnthropic(
                        http_client=DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
                        **kwargs,
                    )

                provider = AnthropicProvider(
                    settings=AnthropicProviderSettings(
                        api_key="test-key", thinking_mode="adaptive", effort="xhigh",
                        prompt_cache_ttl="1h",
                    ),
                    read_timeout_seconds=90,
                )
                self.addAsyncCleanup(provider.aclose)
                request = _request(temperature=1, tools=(ToolDefinition(
                    name="lookup", input_schema={"type": "object", "properties": {
                        "query": {"type": "string"}}, "required": ["query"]},
                ),))
                request = replace(request, max_output_tokens=32000)
                with patch("jarvis.llm.providers.anthropic_provider.AsyncAnthropic", make_client):
                    if streaming:
                        events = [event async for event in provider.stream_generate(request)]
                        self.assertIsInstance(events[-1], DoneEvent)
                        response = events[-1].response
                        self.assertEqual("".join(
                            event.delta for event in events if isinstance(event, TextDeltaEvent)
                        ), "Checking.One moment.")
                    else:
                        response = await provider.generate(request)

                self.assertNotIn("temperature", sent[0])
                self.assertEqual(sent[0]["thinking"], {"type": "adaptive"})
                self.assertEqual(sent[0]["output_config"], {"effort": "xhigh"})
                self.assertEqual(sent[0]["cache_control"], {"type": "ephemeral", "ttl": "1h"})
                self.assertEqual(response.usage.input_tokens, 130)
                self.assertEqual(response.usage.total_tokens, 135)
                assistant = LLMMessage(
                    role="assistant", parts=(TextPart(response.text), *response.tool_calls),
                    metadata={"provider": "anthropic", "provider_metadata": response.provider_metadata},
                )
                followup = replace(request, messages=(*request.messages, assistant, LLMMessage(
                    role="tool", parts=(ToolResultPart("tool_1", "lookup", "Found it."),),
                )))
                replay = provider._build_messages_create_kwargs(followup)["messages"][-2]["content"]
                self.assertEqual(replay, _message().model_dump(mode="json", exclude_none=True)["content"])
                self.assertEqual(replay[0]["thinking"], "")
                self.assertEqual(replay[0]["signature"], "opaque-signature")
                self.assertEqual(replay[2]["type"], "redacted_thinking")

    async def test_stream_transport_failures_keep_timeout_classification(self) -> None:
        for exception, expected in (
            (httpx2.ReadTimeout("idle"), ProviderTimeoutError),
            (httpx2.RemoteProtocolError("disconnected"), ProviderTemporaryError),
        ):
            with self.subTest(exception=type(exception).__name__):
                class BrokenStream(httpx2.AsyncByteStream):
                    async def __aiter__(self):
                        yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
                        raise exception

                client = AsyncAnthropic(api_key="test-key", http_client=DefaultAsyncHttpxClient(
                    transport=httpx2.MockTransport(lambda _: httpx2.Response(
                        200, headers={"content-type": "text/event-stream"}, stream=BrokenStream(),
                    )),
                ))
                provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=90)
                provider._client = client
                self.addAsyncCleanup(provider.aclose)
                events = []
                with self.assertRaises(expected) as caught:
                    async for event in provider.stream_generate(_request()):
                        events.append(event)
                self.assertIsInstance(events[0], ProviderActivityEvent)
                if isinstance(caught.exception, ProviderTimeoutError):
                    self.assertEqual(caught.exception.metadata["timeout_kind"], "read_idle")


class AnthropicRequestTests(unittest.TestCase):
    def test_actual_registry_tools_have_anthropic_compatible_root_schemas(self) -> None:
        tools = ToolRegistry.default(ToolSettings.from_workspace_dir(Path("/workspace"))).basic_definitions()
        originals = [deepcopy(tool.input_schema) for tool in tools]
        provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=60)
        kwargs = provider._build_messages_create_kwargs(_request(tools=tools))
        for tool in kwargs["tools"]:
            with self.subTest(tool=tool["name"]):
                schema = tool["input_schema"]
                self.assertEqual(schema["type"], "object")
                self.assertFalse({"allOf", "anyOf", "oneOf"}.intersection(schema))
                Draft202012Validator.check_schema(schema)
        self.assertEqual([tool.input_schema for tool in tools], originals)

    def test_file_preconditions_still_validate_against_original_schema(self) -> None:
        tools = ToolRegistry.default(ToolSettings.from_workspace_dir(Path("/workspace"))).basic_definitions()
        definitions = {tool.name: tool for tool in tools}
        provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=60)
        examples = {
            "file_patch": {"path": "new.txt", "operations": [{"type": "write", "match": "", "replacement": "hi"}]},
            "file_write": {"path": "new.txt", "content": "hi"},
            "file_replace": {"path": "new.txt", "match": "old", "replacement": "new"},
        }
        for name, base in examples.items():
            tool = definitions[name]
            wire_schema = provider._to_anthropic_tool(tool)["input_schema"]
            self.assertNotIn("anyOf", wire_schema)
            self.assertIn("anyOf", tool.input_schema)
            for preconditions in ({}, {"expected_sha256": "a" * 64}, {"expected_file_absent": True}):
                with self.subTest(name=name, preconditions=preconditions):
                    arguments = {**base, **preconditions}
                    self.assertTrue(Draft202012Validator(wire_schema).is_valid(arguments))
                    calls = provider._extract_tool_calls(
                        content_blocks=[Message.model_validate({
                            **_message().model_dump(),
                            "content": [{"type": "tool_use", "id": "edit_1", "name": name, "input": arguments}],
                        }).content[0]], request_tools=tools,
                    )
                    self.assertEqual(calls[0].arguments, arguments)
                    self.assertNotIn(TOOL_CALL_VALIDATION_ERROR_METADATA_KEY, calls[0].provider_metadata)
            for absence in (False, True):
                with self.subTest(name=name, conflicting_absence=absence):
                    arguments = {**base, "expected_sha256": "a" * 64, "expected_file_absent": absence}
                    calls = provider._extract_tool_calls(
                        content_blocks=[Message.model_validate({
                            **_message().model_dump(),
                            "content": [{"type": "tool_use", "id": "edit_1", "name": name, "input": arguments}],
                        }).content[0]], request_tools=tools,
                    )
                    self.assertIn(TOOL_CALL_VALIDATION_ERROR_METADATA_KEY, calls[0].provider_metadata)

    def test_root_combinator_projection_keeps_nested_constraints(self) -> None:
        provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=60)
        for key in ("allOf", "anyOf", "oneOf"):
            with self.subTest(key=key):
                original = {
                    "type": "object", "properties": {"value": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
                    "required": ["value"], "additionalProperties": False,
                    key: [{"properties": {"value": {"type": "string"}}}],
                }
                projected = provider._to_anthropic_tool(ToolDefinition(name="example", input_schema=original))["input_schema"]
                self.assertNotIn(key, projected)
                self.assertEqual(projected["properties"], original["properties"])
                self.assertEqual(projected["required"], ["value"])
                self.assertFalse(projected["additionalProperties"])
                self.assertIn(key, original)
        for schema in (
            {"anyOf": [{"type": "object", "properties": {"x": {"type": "string"}}}]},
            {"type": "object", "properties": {}, "anyOf": [{"properties": {"x": {"type": "string"}}}]},
        ):
            with self.subTest(schema=schema), self.assertRaisesRegex(LLMConfigurationError, "example"):
                provider._to_anthropic_tool(ToolDefinition(name="example", input_schema=schema))

    def test_documented_model_effort_and_adaptive_thinking(self) -> None:
        for model in (
            "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-opus-5-5",
            "claude-sonnet-5", "claude-sonnet-5-5", "claude-haiku-5-5",
            "claude-fable-5", "claude-fable-5-1", "claude-mythos-5", "claude-mythos-5-1",
        ):
            for effort in ("low", "medium", "high", "xhigh", "max"):
                with self.subTest(model=model, effort=effort):
                    provider = AnthropicProvider(settings=AnthropicProviderSettings(
                        thinking_mode="adaptive", effort=effort,
                    ), read_timeout_seconds=60)
                    kwargs = provider._build_messages_create_kwargs(_request(model, temperature=1))
                    self.assertEqual(kwargs["thinking"], {"type": "adaptive"})
                    self.assertEqual(kwargs["output_config"], {"effort": effort})
                    self.assertNotIn("extra_body", kwargs)

    def test_older_models_keep_sampling_and_manual_thinking(self) -> None:
        provider = AnthropicProvider(settings=AnthropicProviderSettings(
            thinking_mode="enabled", thinking_budget_tokens=2048, effort="high",
        ), read_timeout_seconds=60)
        kwargs = provider._build_messages_create_kwargs(_request("claude-opus-4-5", temperature=1))
        self.assertNotIn("extra_body", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["thinking"], {"type": "enabled", "budget_tokens": 2048})
        provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=60)
        kwargs = provider._build_messages_create_kwargs(_request("claude-opus-4-5", temperature=0.3))
        self.assertEqual(kwargs["extra_body"], {"temperature": 0.3})

    def test_adaptive_thinking_allows_forced_tools_on_supported_models(self) -> None:
        for model in ("claude-opus-4-6", "claude-opus-4-7", "claude-opus-5", "claude-sonnet-5",
                      "claude-fable-5", "claude-mythos-5", "claude-mythos-preview", "claude-haiku-5-5"):
            with self.subTest(model=model):
                provider = AnthropicProvider(settings=AnthropicProviderSettings(
                    thinking_mode="adaptive",
                ), read_timeout_seconds=60)
                kwargs = provider._build_messages_create_kwargs(_request(
                    model, tools=(ToolDefinition(name="lookup", input_schema={"type": "object"}),),
                    tool_choice=ToolChoice(mode=ToolChoiceMode.REQUIRED),
                ))
                self.assertEqual(kwargs["tool_choice"], {"type": "any"})

    def test_disabled_thinking_is_sent_explicitly(self) -> None:
        provider = AnthropicProvider(settings=AnthropicProviderSettings(
            thinking_mode="disabled",
        ), read_timeout_seconds=60)
        kwargs = provider._build_messages_create_kwargs(_request("claude-sonnet-5"))
        self.assertEqual(kwargs["thinking"], {"type": "disabled"})

    def test_sonnet_46_supports_max_but_not_xhigh(self) -> None:
        provider = AnthropicProvider(settings=AnthropicProviderSettings(effort="max"), read_timeout_seconds=60)
        self.assertEqual(provider._build_messages_create_kwargs(_request("claude-sonnet-4-6"))[
            "output_config"], {"effort": "max"})
        provider = AnthropicProvider(settings=AnthropicProviderSettings(effort="xhigh"), read_timeout_seconds=60)
        with self.assertRaisesRegex(LLMConfigurationError, "does not support effort"):
            provider._build_messages_create_kwargs(_request("claude-sonnet-4-6"))

    def test_invalid_modern_model_configurations_fail_before_dispatch(self) -> None:
        cases = (
            ("claude-opus-5-5", {"thinking_mode": "disabled"}, "cannot disable"),
            ("claude-opus-5-5", {"thinking_mode": "enabled"}, "requires thinking_mode adaptive"),
            ("claude-opus-5-5", {"thinking_budget_tokens": 2048}, "does not support thinking_budget"),
            ("claude-sonnet-5", {"thinking_mode": "between_tools"}, "does not support between_tools"),
            ("claude-opus-5", {"thinking_mode": "disabled", "effort": "max"}, "requires adaptive"),
            ("claude-haiku-5-5", {"thinking_mode": "disabled", "effort": "xhigh"}, "requires adaptive"),
            ("claude-sonnet-5-5", {"thinking_mode": "between_tools", "effort": "max"}, "requires adaptive"),
            ("claude-opus-4-5", {"thinking_budget_tokens": 8192}, "must be less than"),
        )
        for model, settings, error in cases:
            with self.subTest(model=model, settings=settings):
                provider = AnthropicProvider(settings=AnthropicProviderSettings(**settings), read_timeout_seconds=60)
                with self.assertRaisesRegex(LLMConfigurationError, error):
                    provider._build_messages_create_kwargs(_request(model))
        provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=60)
        with self.assertRaisesRegex(LLMConfigurationError, "custom temperature"):
            provider._build_messages_create_kwargs(_request(temperature=0.3))
        for mode in (ToolChoiceMode.REQUIRED, ToolChoiceMode.TOOL):
            with self.assertRaisesRegex(LLMConfigurationError, "cannot force tool"):
                provider._build_messages_create_kwargs(_request(
                    tools=(ToolDefinition(name="lookup", input_schema={"type": "object"}),),
                    tool_choice=ToolChoice(mode=mode, tool_name="lookup" if mode == ToolChoiceMode.TOOL else None),
                ))

    def test_sonnet_55_between_tools_configuration(self) -> None:
        provider = AnthropicProvider(settings=AnthropicProviderSettings(
            thinking_mode="between_tools", effort="high",
        ), read_timeout_seconds=60)
        self.assertEqual(provider._build_messages_create_kwargs(_request("claude-sonnet-5-5"))[
            "thinking"], {"type": "between_tools"})

    def test_hosted_ids_and_snapshots_do_not_match_unknown_future_models(self) -> None:
        self.assertTrue(anthropic_model_capabilities("anthropic.claude-opus-4-6-v1:0").adaptive)
        self.assertTrue(anthropic_model_capabilities("claude-opus-4-5-20251101").manual_thinking)
        self.assertFalse(anthropic_model_capabilities("claude-opus-5-6").adaptive)

    def test_changed_prefix_or_model_drops_bound_thinking_without_changing_transcript(self) -> None:
        provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=60)
        request = _request()
        response = provider._normalize_message_response(request=request, response=_message())
        assistant = LLMMessage(
            role="assistant", parts=(TextPart(response.text), *response.tool_calls),
            metadata={"provider": "anthropic", "provider_metadata": response.provider_metadata},
        )
        history = (*request.messages, assistant, LLMMessage.text("user", "Continue"))
        transcript = deepcopy(assistant.metadata)
        for changed in (
            replace(request, messages=(LLMMessage.text("system", "Changed"), *history[1:])),
            replace(request, messages=(history[0], LLMMessage.text("user", "Edited"), *history[2:])),
            replace(request, messages=history, model="claude-sonnet-5"),
            replace(request, messages=history, tools=(ToolDefinition(name="new_tool", input_schema={"type": "object"}),)),
            replace(request, messages=(LLMMessage.text("user", "Compacted"), *history[2:])),
        ):
            with self.subTest(changed=changed):
                content = provider._build_messages_create_kwargs(changed)["messages"][-2]["content"]
                self.assertTrue(all(block["type"] not in {"thinking", "redacted_thinking"} for block in content))
                self.assertEqual([block["type"] for block in content], ["text", "tool_use", "text"])
                self.assertEqual(assistant.metadata, transcript)

    def test_multiple_tool_rounds_preserve_or_drop_the_entire_thinking_suffix(self) -> None:
        provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=60)
        request = _request(tools=(ToolDefinition(name="lookup", input_schema={"type": "object"}),))
        for number in (1, 2):
            payload = _message().model_dump(mode="json")
            payload["content"][3]["id"] = f"tool_{number}"
            response = provider._normalize_message_response(request=request, response=Message.model_validate(payload))
            assistant = LLMMessage(
                role="assistant", parts=(TextPart(response.text), *response.tool_calls),
                metadata={"provider": "anthropic", "provider_metadata": response.provider_metadata},
            )
            request = replace(request, messages=(*request.messages, assistant, LLMMessage(
                role="tool", parts=(ToolResultPart(f"tool_{number}", "lookup", "Found it."),),
            )))
        kwargs = provider._build_messages_create_kwargs(request)
        self.assertEqual(sum(
            block["type"] in {"thinking", "redacted_thinking"}
            for message in kwargs["messages"] for block in message["content"]
        ), 4)
        changed = replace(request, messages=(request.messages[0], LLMMessage.text("user", "Edited"), *request.messages[2:]))
        kwargs = provider._build_messages_create_kwargs(changed)
        self.assertFalse(any(
            block["type"] in {"thinking", "redacted_thinking"}
            for message in kwargs["messages"] for block in message["content"]
        ))
        response = provider._normalize_message_response(request=changed, response=_message())
        continued = replace(changed, messages=(*changed.messages, LLMMessage(
            role="assistant", parts=(TextPart(response.text), *response.tool_calls),
            metadata={"provider": "anthropic", "provider_metadata": response.provider_metadata},
        ), LLMMessage.text("user", "Continue")))
        kwargs = provider._build_messages_create_kwargs(continued)
        self.assertEqual(sum(
            block["type"] in {"thinking", "redacted_thinking"}
            for message in kwargs["messages"] for block in message["content"]
        ), 2)

    def test_modified_assistant_text_does_not_restore_stale_content(self) -> None:
        provider = AnthropicProvider(settings=AnthropicProviderSettings(), read_timeout_seconds=60)
        request = _request()
        response = provider._normalize_message_response(request=request, response=_message())
        assistant = LLMMessage(
            role="assistant", parts=(TextPart("Edited reply"), *response.tool_calls),
            metadata={"provider": "anthropic", "provider_metadata": response.provider_metadata},
        )
        kwargs = provider._build_messages_create_kwargs(replace(request, messages=(
            *request.messages, assistant, LLMMessage.text("user", "Continue"),
        )))
        content = kwargs["messages"][-2]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Edited reply"})
        self.assertEqual([block["type"] for block in content], ["text", "tool_use"])

    def test_anthropic_settings_validate_effort_mode_and_budget(self) -> None:
        for kwargs in ({"effort": "minimal"}, {"thinking_mode": "invalid"}, {"thinking_budget_tokens": 100}):
            with self.subTest(kwargs=kwargs), self.assertRaises(LLMConfigurationError):
                AnthropicProviderSettings(**kwargs)
