"""Anthropic provider adapter for normalized LLM requests/responses."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from typing import Any

import httpx2
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncAnthropic,
    AnthropicError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
    Timeout,
)
from pydantic import BaseModel

from ..config import AnthropicProviderSettings
from ..errors import (
    LLMConfigurationError,
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTemporaryError,
    ProviderTimeoutError,
    UnsupportedCapabilityError,
)
from ..protocols import ProviderCapabilities
from ..types import (
    DoneEvent,
    EmbeddingRequest,
    EmbeddingResponse,
    ImagePart,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ProviderActivityEvent,
    ProviderStreamEvent,
    TextDeltaEvent,
    TextPart,
    ToolCall,
    ToolCallDeltaEvent,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    ToolResultPart,
    UsageDeltaEvent,
)
from ..timeouts import ProviderTransportTimeouts, transport_timeout_metadata
from ..validation import build_tool_schema_map, parse_and_validate_tool_call_or_recover
from .anthropic_models import anthropic_model_capabilities


class AnthropicProvider:
    """Provider implementation for Anthropic Messages API."""

    def __init__(
        self,
        *,
        settings: AnthropicProviderSettings,
        read_timeout_seconds: float,
        connect_timeout_seconds: float = 30.0,
    ) -> None:
        self._settings = settings
        self._transport_timeouts = ProviderTransportTimeouts(
            connect_seconds=connect_timeout_seconds,
            read_seconds=read_timeout_seconds,
        )
        self._client: AsyncAnthropic | None = None
        self._client_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools=True,
            embeddings=False,
            image_input=True,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._build_messages_create_kwargs(request)
        client = await self._client_instance()
        try:
            response = await client.messages.create(**kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc
        return self._normalize_message_response(request=request, response=response)

    async def stream_generate(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[ProviderStreamEvent]:
        kwargs = self._build_messages_create_kwargs(request)
        client = await self._client_instance()
        tool_state_by_index: dict[int, tuple[str, str | None]] = {}
        tool_call_ids_with_deltas: set[str] = set()
        response_id: str | None = None
        saw_text_delta = False

        try:
            async with client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    event_type = str(getattr(event, "type", "") or "unknown")
                    event_response_id = _anthropic_event_response_id(event)
                    if event_response_id is not None:
                        response_id = event_response_id

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if getattr(block, "type", None) == "tool_use":
                            index = int(getattr(event, "index", 0))
                            call_id = str(getattr(block, "id", "") or f"tool_{index}")
                            name = _optional_anthropic_string(getattr(block, "name", None))
                            tool_state_by_index[index] = (call_id, name)
                        yield ProviderActivityEvent(
                            provider_event_type=event_type,
                            response_id=response_id,
                        )
                        continue

                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_type = getattr(delta, "type", None)
                        if delta_type == "text_delta":
                            text = str(getattr(delta, "text", "") or "")
                            if text:
                                saw_text_delta = True
                            yield TextDeltaEvent(delta=text)
                            continue
                        if delta_type == "input_json_delta":
                            index = int(getattr(event, "index", 0))
                            call_id, tool_name = tool_state_by_index.get(
                                index,
                                (f"tool_{index}", None),
                            )
                            tool_call_ids_with_deltas.add(call_id)
                            yield ToolCallDeltaEvent(
                                call_id=call_id,
                                tool_name=tool_name,
                                arguments_delta=str(
                                    getattr(delta, "partial_json", "") or ""
                                ),
                            )
                            continue

                    yield ProviderActivityEvent(
                        provider_event_type=event_type,
                        response_id=response_id,
                    )

                final_message = await stream.get_final_message()
        except Exception as exc:
            raise self._map_error(exc) from exc

        normalized = self._normalize_message_response(
            request=request,
            response=final_message,
        )
        if normalized.text and not saw_text_delta:
            yield TextDeltaEvent(delta=normalized.text)
        for tool_call in normalized.tool_calls:
            if tool_call.call_id in tool_call_ids_with_deltas:
                continue
            yield ToolCallDeltaEvent(
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
                arguments_delta=tool_call.raw_arguments,
            )
        if normalized.usage is not None:
            yield UsageDeltaEvent(usage=normalized.usage)
        yield DoneEvent(response=normalized)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise UnsupportedCapabilityError("Provider 'anthropic' does not support embeddings.")

    async def aclose(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is None:
            return
        close = getattr(client, "close", None)
        if close is None:
            return
        maybe = close()
        if inspect.isawaitable(maybe):
            await maybe

    async def _client_instance(self) -> AsyncAnthropic:
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is not None:
                return self._client

            api_key = self._settings.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise LLMConfigurationError(
                    "ANTHROPIC_API_KEY is required for the Anthropic provider."
                )

            self._client = AsyncAnthropic(
                api_key=api_key,
                base_url=self._settings.base_url or os.getenv("ANTHROPIC_BASE_URL"),
                timeout=Timeout(
                    connect=self._transport_timeouts.connect_seconds,
                    read=self._transport_timeouts.read_seconds,
                    write=self._transport_timeouts.connect_seconds,
                    pool=self._transport_timeouts.connect_seconds,
                ),
                max_retries=0,
            )
            return self._client

    def _build_messages_create_kwargs(self, request: LLMRequest) -> dict[str, Any]:
        if request.model is None:
            raise LLMConfigurationError("request.model must be set before provider dispatch.")
        if request.max_output_tokens is None:
            raise LLMConfigurationError(
                "Anthropic requires max_output_tokens. Set JARVIS_ANTHROPIC_MAX_OUTPUT_TOKENS "
                "or provide request.max_output_tokens."
            )

        prompt_cache_ttl = self._settings.prompt_cache_ttl
        tools = [self._to_anthropic_tool(tool) for tool in request.tools]
        system_prompt, messages = self._to_anthropic_messages(
            request.messages,
            prompt_cache_ttl=prompt_cache_ttl,
            model=request.model,
            tools=tools,
        )
        if not messages:
            raise LLMConfigurationError("Anthropic requests require at least one user/assistant message.")

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
        }
        if prompt_cache_ttl is not None:
            kwargs["cache_control"] = {
                "type": "ephemeral",
                "ttl": prompt_cache_ttl,
            }

        if system_prompt is not None:
            kwargs["system"] = system_prompt
        capabilities = anthropic_model_capabilities(request.model)
        if request.temperature is not None:
            if capabilities.sampling:
                # SDK 1.x removed the keyword; older models still accept it in JSON.
                kwargs["extra_body"] = {"temperature": request.temperature}
            elif request.temperature != 1:
                raise LLMConfigurationError(
                    f"{request.model} does not support custom temperature. "
                    "Unset JARVIS_ANTHROPIC_TEMPERATURE or use 1."
                )
        if request.tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = self._to_anthropic_tool_choice(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        thinking = self._to_anthropic_thinking(model=request.model)
        if thinking is not None:
            if thinking["type"] == "enabled" and thinking["budget_tokens"] >= request.max_output_tokens:
                raise LLMConfigurationError(
                    "Anthropic thinking_budget_tokens must be less than max_output_tokens."
                )
            kwargs["thinking"] = thinking

        manual_thinking = thinking is not None and thinking["type"] == "enabled"
        thinking_active = (
            thinking["type"] != "disabled" if thinking else capabilities.thinking_default
        )
        if thinking_active and request.temperature is not None:
            if request.temperature != 1:
                raise LLMConfigurationError("Anthropic thinking requires the default temperature 1.")
            kwargs.pop("extra_body", None)
        if (
            request.tools
            and request.tool_choice.mode in {ToolChoiceMode.REQUIRED, ToolChoiceMode.TOOL}
            and (manual_thinking or not capabilities.forced_tools)
        ):
            raise LLMConfigurationError(
                f"{request.model} cannot force tool use with this thinking configuration; "
                "use tool_choice auto or none."
            )

        output_config = self._to_anthropic_output_config(model=request.model)
        if output_config is not None:
            kwargs["output_config"] = output_config

        return kwargs

    def _to_anthropic_messages(
        self,
        messages: Sequence[LLMMessage],
        *,
        prompt_cache_ttl: str | None,
        model: str,
        tools: list[dict[str, Any]],
    ) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        out_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []
        replay_by_index: dict[int, dict[str, Any]] = {}

        for message in messages:
            if message.role == "system":
                text = _join_text_parts(
                    message.parts,
                    unsupported_message=("Anthropic system history only supports text parts."),
                )
                if text:
                    if _anthropic_system_message_is_global(message):
                        system_parts.append(text)
                    else:
                        if pending_tool_results:
                            out_messages.append(
                                {
                                    "role": "user",
                                    "content": pending_tool_results,
                                }
                            )
                            pending_tool_results = []
                        out_messages.append(
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": text}],
                            }
                        )
                continue

            if message.role == "tool":
                pending_tool_results.extend(self._to_anthropic_tool_result_blocks(message))
                continue

            if pending_tool_results:
                out_messages.append(
                    {
                        "role": "user",
                        "content": pending_tool_results,
                    }
                )
                pending_tool_results = []

            content_blocks = self._to_anthropic_content_blocks(message)
            if not content_blocks:
                continue
            role = "assistant" if message.role == "assistant" else "user"
            replay = _anthropic_replay_metadata(message)
            if replay is not None and (
                _visible_anthropic_blocks(replay["content"])
                == _visible_anthropic_blocks(content_blocks)
            ):
                content_blocks = deepcopy(replay["content"])
                replay_by_index[len(out_messages)] = replay
            out_messages.append(
                {
                    "role": role,
                    "content": content_blocks,
                }
            )

        if pending_tool_results:
            out_messages.append(
                {
                    "role": "user",
                    "content": pending_tool_results,
                }
            )

        if prompt_cache_ttl is None:
            system_prompt = "\n\n".join(system_parts).strip() or None
        else:
            system_prompt = _build_anthropic_system_blocks(
                system_parts,
                prompt_cache_ttl=prompt_cache_ttl,
            )
        prefix = _anthropic_prefix_hasher(system_prompt, tools)
        for index, payload in enumerate(out_messages):
            replay = replay_by_index.get(index)
            if replay is not None:
                preserve_thinking = (
                    replay.get("model") == model
                    and replay.get("prefix") == prefix.hexdigest()
                )
                if not preserve_thinking:
                    payload["content"] = [
                        block for block in payload["content"]
                        if block.get("type") not in {"thinking", "redacted_thinking"}
                    ]
            _update_anthropic_prefix(prefix, payload)
        return system_prompt, out_messages

    def _to_anthropic_content_blocks(self, message: LLMMessage) -> list[dict[str, Any]]:
        content_blocks: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                content_blocks.append({"type": "text", "text": part.text})
            elif isinstance(part, ToolCall):
                if message.role != "assistant":
                    raise LLMConfigurationError(
                        "Tool call history can only appear on assistant messages."
                    )
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": part.call_id,
                        "name": part.name,
                        "input": part.arguments,
                    }
                )
            elif isinstance(part, ImagePart):
                content_blocks.append(
                    {
                        "type": "image",
                        "source": _to_anthropic_image_source(part),
                    }
                )
            else:
                raise LLMConfigurationError(
                    f"Unsupported Anthropic message part type: {type(part).__name__}."
                )
        return content_blocks

    def _to_anthropic_tool_result_blocks(self, message: LLMMessage) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for part in message.parts:
            if not isinstance(part, ToolResultPart):
                raise LLMConfigurationError(
                    "Tool-role messages can only contain tool results for Anthropic."
                )
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": part.call_id,
                "content": part.content,
            }
            if part.is_error:
                block["is_error"] = True
            blocks.append(block)
        return blocks

    def _to_anthropic_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": tool.name,
            "input_schema": _anthropic_tool_input_schema(tool),
        }
        if tool.description is not None:
            payload["description"] = tool.description
        return payload

    def _to_anthropic_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any]:
        if tool_choice.mode == ToolChoiceMode.AUTO:
            return {"type": "auto"}
        if tool_choice.mode == ToolChoiceMode.REQUIRED:
            return {"type": "any"}
        if tool_choice.mode == ToolChoiceMode.NONE:
            return {"type": "none"}
        return {"type": "tool", "name": tool_choice.tool_name}

    def _to_anthropic_thinking(self, *, model: str) -> dict[str, Any] | None:
        thinking_mode = self._settings.thinking_mode
        budget_tokens = self._settings.thinking_budget_tokens
        capabilities = anthropic_model_capabilities(model)

        if thinking_mode in {"disabled", "between_tools"}:
            if thinking_mode == "between_tools" and not capabilities.between_tools:
                raise LLMConfigurationError(f"{model} does not support between_tools thinking.")
            if thinking_mode == "disabled" and not capabilities.can_disable_thinking:
                raise LLMConfigurationError(
                    f"{model} cannot disable thinking; use adaptive thinking and effort"
                    + (", or between_tools for no up-front thinking." if capabilities.between_tools else ".")
                )
            if capabilities.high_effort_requires_thinking and self._settings.effort in {"xhigh", "max"}:
                raise LLMConfigurationError(
                    f"{model} requires adaptive thinking at xhigh/max effort."
                )
            return {"type": thinking_mode}

        if budget_tokens is not None:
            if not capabilities.manual_thinking:
                raise LLMConfigurationError(
                    f"{model} does not support thinking_budget_tokens; use adaptive thinking and effort."
                )
            if thinking_mode != "adaptive":
                return {"type": "enabled", "budget_tokens": budget_tokens}

        if thinking_mode in {"adaptive", "enabled"}:
            if capabilities.adaptive:
                if thinking_mode == "enabled" and not capabilities.manual_thinking:
                    raise LLMConfigurationError(
                        f"{model} requires thinking_mode adaptive instead of enabled."
                    )
                return {"type": "adaptive"}
            if budget_tokens is not None:
                return {"type": "enabled", "budget_tokens": budget_tokens}
            raise LLMConfigurationError(
                f"{model} requires a thinking_budget_tokens value for extended thinking."
            )
        return None

    def _to_anthropic_output_config(self, *, model: str) -> dict[str, Any] | None:
        effort = self._settings.effort
        if effort is None:
            return None
        if effort not in anthropic_model_capabilities(model).efforts:
            raise LLMConfigurationError(f"{model} does not support effort {effort!r}.")
        return {"effort": effort}

    def _normalize_message_response(self, *, request: LLMRequest, response: Any) -> LLMResponse:
        tool_calls = self._extract_tool_calls(
            content_blocks=response.content,
            request_tools=request.tools,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        uncached_input_tokens = getattr(response.usage, "input_tokens", None)
        input_tokens = sum(
            getattr(response.usage, key, 0) or 0
            for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        ) if uncached_input_tokens is not None else None
        output_tokens = getattr(response.usage, "output_tokens", None)
        usage = LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=(input_tokens or 0) + (output_tokens or 0),
        )

        finish_reason = "unknown"
        stop_reason = getattr(response, "stop_reason", None)
        if tool_calls or stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif stop_reason == "max_tokens":
            finish_reason = "length"
        elif stop_reason == "refusal":
            finish_reason = "content_filter"
        elif stop_reason in {"end_turn", "stop_sequence"}:
            finish_reason = "stop"

        usage_metadata = getattr(response, "usage", None)
        cache_creation = getattr(usage_metadata, "cache_creation", None)
        cache_creation_metadata: dict[str, int] | None = None
        if cache_creation is not None:
            cache_creation_metadata = {}
            for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"):
                value = getattr(cache_creation, key, None)
                if isinstance(value, int):
                    cache_creation_metadata[key] = value
            if not cache_creation_metadata:
                cache_creation_metadata = None

        provider_metadata: dict[str, Any] = {"stop_reason": stop_reason}
        if isinstance(uncached_input_tokens, int):
            provider_metadata["uncached_input_tokens"] = uncached_input_tokens
        if response.content and all(isinstance(block, BaseModel) for block in response.content):
            request_kwargs = self._build_messages_create_kwargs(request)
            prefix = _anthropic_prefix_hasher(
                request_kwargs.get("system"), request_kwargs.get("tools", []),
            )
            for message in request_kwargs["messages"]:
                _update_anthropic_prefix(prefix, message)
            provider_metadata["anthropic_replay"] = {
                "model": request.model,
                "prefix": prefix.hexdigest(),
                "content": [block.model_dump(mode="json", exclude_none=True) for block in response.content],
            }
        stop_details = getattr(response, "stop_details", None)
        if isinstance(stop_details, BaseModel):
            provider_metadata["stop_details"] = stop_details.model_dump(mode="json", exclude_none=True)
        cache_read_input_tokens = getattr(usage_metadata, "cache_read_input_tokens", None)
        if isinstance(cache_read_input_tokens, int):
            provider_metadata["cache_read_input_tokens"] = cache_read_input_tokens
        cache_creation_input_tokens = getattr(usage_metadata, "cache_creation_input_tokens", None)
        if isinstance(cache_creation_input_tokens, int):
            provider_metadata["cache_creation_input_tokens"] = cache_creation_input_tokens
        if cache_creation_metadata is not None:
            provider_metadata["cache_creation"] = cache_creation_metadata

        return LLMResponse(
            provider=self.name,
            model=response.model,
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            response_id=response.id,
            provider_metadata=provider_metadata,
        )

    def _extract_tool_calls(
        self,
        *,
        content_blocks: Sequence[Any],
        request_tools: Sequence[ToolDefinition],
    ) -> list[ToolCall]:
        tool_schemas = build_tool_schema_map(request_tools)
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            raw_arguments = json.dumps(block.input or {})
            tool_calls.append(
                parse_and_validate_tool_call_or_recover(
                    call_id=block.id,
                    name=block.name,
                    raw_arguments=raw_arguments,
                    tool_schemas=tool_schemas,
                )
            )
        return tool_calls

    def _map_error(self, error: Exception) -> Exception:
        if isinstance(error, (AuthenticationError, PermissionDeniedError)):
            return ProviderAuthenticationError(str(error))
        if isinstance(error, RateLimitError):
            return ProviderRateLimitError(str(error))
        if isinstance(error, (APITimeoutError, httpx2.TimeoutException)):
            return ProviderTimeoutError(
                str(error),
                metadata=transport_timeout_metadata(
                    error,
                    timeouts=self._transport_timeouts,
                ),
            )
        if isinstance(error, (APIConnectionError, InternalServerError, httpx2.TransportError)):
            return ProviderTemporaryError(str(error))
        if isinstance(error, BadRequestError):
            return ProviderBadRequestError(str(error))
        if isinstance(error, APIStatusError):
            if error.status_code >= 500:
                return ProviderTemporaryError(str(error))
            if error.status_code == 429:
                return ProviderRateLimitError(str(error))
            if error.status_code in {401, 403}:
                return ProviderAuthenticationError(str(error))
            return ProviderBadRequestError(str(error))
        if isinstance(error, AnthropicError):
            return ProviderResponseError(str(error))
        return ProviderResponseError(str(error))


def _anthropic_tool_input_schema(tool: ToolDefinition) -> dict[str, Any]:
    """Project root validation constraints onto Anthropic's object-only wire shape.

    The original ToolDefinition remains authoritative for returned argument
    validation. Nested combinators are supported and must remain intact.
    """
    schema = deepcopy(dict(tool.input_schema))
    combinators = ("allOf", "anyOf", "oneOf")
    if not any(key in schema for key in combinators):
        return schema

    properties = schema.get("properties")
    if schema.get("type") != "object" or not isinstance(properties, dict):
        raise LLMConfigurationError(
            f"Anthropic tool {tool.name!r} requires an object schema with root properties; "
            "root allOf/anyOf/oneOf alone cannot describe its arguments to Anthropic."
        )

    # A branch may narrow a declared property, but must not introduce a field
    # that would disappear from the schema shown to the model.
    pending = [schema]
    while pending:
        branch = pending.pop()
        branch_properties = branch.get("properties", {})
        if isinstance(branch_properties, dict) and not branch_properties.keys() <= properties.keys():
            raise LLMConfigurationError(
                f"Anthropic tool {tool.name!r} must declare every argument in root properties."
            )
        for key in combinators:
            children = branch.get(key)
            if isinstance(children, list):
                pending.extend(child for child in children if isinstance(child, dict))

    for key in combinators:
        schema.pop(key, None)
    return schema


def _anthropic_replay_metadata(message: LLMMessage) -> dict[str, Any] | None:
    if message.role != "assistant" or message.metadata.get("provider") != "anthropic":
        return None
    metadata = message.metadata.get("provider_metadata")
    replay = metadata.get("anthropic_replay") if isinstance(metadata, dict) else None
    if not isinstance(replay, dict):
        return None
    content = replay.get("content")
    if not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
        return None
    return replay


def _visible_anthropic_blocks(content: list[dict[str, Any]]) -> dict[str, Any]:
    text: list[str] = []
    tools: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            text.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tools.append({key: block.get(key) for key in ("type", "id", "name", "input")})
    return {"text": "".join(text), "tools": tools}


def _anthropic_prefix_hasher(system: Any, tools: list[dict[str, Any]]) -> Any:
    prefix = hashlib.sha256()
    _update_anthropic_prefix(prefix, {"system": system, "tools": tools})
    return prefix


def _update_anthropic_prefix(prefix: Any, payload: dict[str, Any]) -> None:
    prefix.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    prefix.update(b"\n")


def _anthropic_event_response_id(event: Any) -> str | None:
    message = getattr(event, "message", None)
    return _optional_anthropic_string(getattr(message, "id", None))


def _optional_anthropic_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _join_text_parts(parts: Sequence[Any], *, unsupported_message: str) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            text_parts.append(part.text)
            continue
        raise LLMConfigurationError(unsupported_message)
    return "\n".join(text_parts).strip()


def _to_anthropic_image_source(part: ImagePart) -> dict[str, Any]:
    data_url_payload = part.data_url_payload()
    if data_url_payload is None:
        raise LLMConfigurationError(
            "Anthropic image input in this layer requires a base64 data URL image."
        )
    media_type, data_base64 = data_url_payload
    return {
        "type": "base64",
        "media_type": media_type,
        "data": data_base64,
    }


def _build_anthropic_system_blocks(
    system_parts: Sequence[str],
    *,
    prompt_cache_ttl: str,
) -> list[dict[str, Any]] | None:
    blocks = [
        {
            "type": "text",
            "text": text,
        }
        for text in system_parts
        if text
    ]
    if not blocks:
        return None
    blocks[-1]["cache_control"] = {
        "type": "ephemeral",
        "ttl": prompt_cache_ttl,
    }
    return blocks


def _anthropic_system_message_is_global(message: LLMMessage) -> bool:
    metadata = message.metadata
    if not metadata:
        return True
    return bool(
        metadata.get("bootstrap_identity")
        or metadata.get("memory_bootstrap")
        or metadata.get("compaction_item")
    )
