"""Grok provider adapter using xAI's OpenAI-compatible Responses API."""

from __future__ import annotations

import asyncio
import copy
import inspect
import json
import os
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from ..config import GrokProviderSettings
from ..errors import (
    LLMConfigurationError,
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTemporaryError,
    ProviderTimeoutError,
    StreamProtocolError,
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
    LLMStreamEvent,
    LLMUsage,
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
from ..validation import build_tool_schema_map, parse_and_validate_tool_call_or_recover


class GrokProvider:
    """Provider implementation for xAI Grok Responses."""

    def __init__(
        self,
        *,
        settings: GrokProviderSettings,
        default_timeout_seconds: float,
    ) -> None:
        self._settings = settings
        self._default_timeout_seconds = default_timeout_seconds
        self._client: AsyncOpenAI | None = None
        self._client_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "grok"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools=True,
            embeddings=False,
            image_input=True,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._build_response_create_kwargs(request, stream=False)
        client = await self._client_for_request(request)
        try:
            response = await client.responses.create(**kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc
        return self._normalize_response(request=request, response=response)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        kwargs = self._build_response_create_kwargs(request, stream=True)
        client = await self._client_for_request(request)

        tool_name_by_item_id: dict[str, str] = {}
        call_id_by_item_id: dict[str, str] = {}
        saw_completion = False

        try:
            stream = await client.responses.create(**kwargs)
            async for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "response.output_text.delta":
                    yield TextDeltaEvent(delta=event.delta)
                    continue

                if event_type in {"response.output_item.added", "response.output_item.done"}:
                    item = event.item
                    if getattr(item, "type", "") == "function_call":
                        item_id = getattr(item, "id", None)
                        if item_id:
                            tool_name_by_item_id[item_id] = item.name
                            call_id_by_item_id[item_id] = item.call_id
                    continue

                if event_type == "response.function_call_arguments.delta":
                    call_id = call_id_by_item_id.get(event.item_id, event.item_id)
                    yield ToolCallDeltaEvent(
                        call_id=call_id,
                        tool_name=tool_name_by_item_id.get(event.item_id),
                        arguments_delta=event.delta,
                    )
                    continue

                if event_type == "response.function_call_arguments.done":
                    call_id = call_id_by_item_id.get(event.item_id, event.item_id)
                    yield ToolCallDeltaEvent(
                        call_id=call_id,
                        tool_name=event.name,
                        arguments_delta=event.arguments,
                    )
                    continue

                if event_type == "response.completed":
                    normalized = self._normalize_response(
                        request=request,
                        response=event.response,
                    )
                    if normalized.usage is not None:
                        yield UsageDeltaEvent(usage=normalized.usage)
                    yield DoneEvent(response=normalized)
                    saw_completion = True
                    continue

                if event_type == "response.failed":
                    raise ProviderResponseError(
                        self._extract_stream_failed_message(getattr(event, "response", None))
                    )

                if event_type == "error":
                    raise ProviderResponseError(event.message)
        except Exception as exc:
            if isinstance(exc, (ProviderResponseError, StreamProtocolError)):
                raise
            raise self._map_error(exc) from exc

        if not saw_completion:
            raise StreamProtocolError("Grok stream closed without a response.completed event.")

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise UnsupportedCapabilityError("Provider 'grok' does not support embeddings.")

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

    async def _client_instance(self) -> AsyncOpenAI:
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is not None:
                return self._client

            api_key = self._settings.api_key or os.getenv("XAI_API_KEY")
            if not api_key:
                raise LLMConfigurationError("XAI_API_KEY is required for the Grok provider.")

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._settings.base_url.rstrip("/"),
                timeout=self._default_timeout_seconds,
                max_retries=0,
            )
            return self._client

    async def _client_for_request(self, request: LLMRequest) -> AsyncOpenAI:
        client = await self._client_instance()
        if request.prompt_cache_key is None:
            return client
        return client.with_options(default_headers={"x-grok-conv-id": request.prompt_cache_key})

    def _build_response_create_kwargs(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        if request.model is None:
            raise LLMConfigurationError("request.model must be set before provider dispatch.")

        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": [
                item
                for message in request.messages
                for item in self._to_grok_input_items(message, model=request.model)
            ],
            "stream": stream,
            "store": False,
            "parallel_tool_calls": request.parallel_tool_calls,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.timeout_seconds is not None:
            kwargs["timeout"] = request.timeout_seconds

        if _grok_model_uses_encrypted_reasoning(request.model):
            kwargs["include"] = ["reasoning.encrypted_content"]

        if request.tools:
            kwargs["tools"] = [self._to_grok_tool(tool) for tool in request.tools]
            kwargs["tool_choice"] = self._to_grok_tool_choice(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        return kwargs

    def _to_grok_input_items(
        self,
        message: LLMMessage,
        *,
        model: str,
    ) -> list[dict[str, Any]]:
        if message.role == "tool":
            return self._to_grok_tool_result_items(message)

        replay_items = self._stored_response_output_items(message)
        if replay_items is not None:
            return replay_items

        content: list[dict[str, Any]] = []
        tool_call_items: list[dict[str, Any]] = []

        for part in message.parts:
            if isinstance(part, TextPart):
                text_part_type = "output_text" if message.role == "assistant" else "input_text"
                content.append(
                    {
                        "type": text_part_type,
                        "text": part.text,
                    }
                )
            elif isinstance(part, ImagePart):
                if message.role == "assistant":
                    raise LLMConfigurationError(
                        "Grok provider does not support assistant image history items."
                    )
                if part.file_id is not None:
                    raise LLMConfigurationError(
                        "Grok provider supports image_url, not file_id."
                    )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": part.image_url,
                        "detail": _normalize_grok_image_detail(part.detail),
                    }
                )
            elif isinstance(part, ToolCall):
                if message.role != "assistant":
                    raise LLMConfigurationError(
                        "Tool call history can only appear on assistant messages."
                    )
                tool_call_items.append(
                    {
                        "type": "function_call",
                        "call_id": part.call_id,
                        "name": part.name,
                        "arguments": part.raw_arguments,
                    }
                )
            else:
                raise LLMConfigurationError(
                    f"Unsupported Grok message part type: {type(part).__name__}."
                )

        items: list[dict[str, Any]] = []
        if content:
            items.append(
                {
                    "type": "message",
                    "role": message.role,
                    "content": content,
                }
            )
        items.extend(tool_call_items)
        return items

    def _stored_response_output_items(self, message: LLMMessage) -> list[dict[str, Any]] | None:
        if message.role != "assistant":
            return None
        if str(message.metadata.get("provider", "")).strip() != self.name:
            return None

        provider_metadata = message.metadata.get("provider_metadata", {})
        if not isinstance(provider_metadata, dict):
            return None

        response_output = provider_metadata.get("response_output")
        if not isinstance(response_output, list):
            return None
        if not all(isinstance(item, dict) for item in response_output):
            return None

        return copy.deepcopy(response_output)

    def _to_grok_tool_result_items(self, message: LLMMessage) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for part in message.parts:
            if not isinstance(part, ToolResultPart):
                raise LLMConfigurationError(
                    "Tool-role messages can only contain tool results for Grok."
                )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": part.call_id,
                    "output": part.content,
                }
            )
        return items

    def _to_grok_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "parameters": dict(tool.input_schema),
        }
        if tool.description is not None:
            payload["description"] = tool.description
        return payload

    def _to_grok_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any] | str:
        if tool_choice.mode == ToolChoiceMode.AUTO:
            return "auto"
        if tool_choice.mode == ToolChoiceMode.REQUIRED:
            return "required"
        if tool_choice.mode == ToolChoiceMode.NONE:
            return "none"
        return {"type": "function", "function": {"name": tool_choice.tool_name}}

    def _normalize_response(self, *, request: LLMRequest, response: Any) -> LLMResponse:
        response_output = _serialize_response_output_items(getattr(response, "output", []))
        tool_calls = self._extract_tool_calls(
            response_output=response_output,
            request_tools=request.tools,
        )
        usage = self._normalize_usage(getattr(response, "usage", None))

        provider_metadata: dict[str, Any] = {
            "status": getattr(response, "status", None),
            "incomplete_reason": _normalize_optional_string(
                getattr(getattr(response, "incomplete_details", None), "reason", None)
            ),
            "response_output": response_output,
        }

        input_tokens_details = _serialize_optional_mapping(
            getattr(getattr(response, "usage", None), "input_tokens_details", None)
        )
        if input_tokens_details is not None:
            provider_metadata["input_tokens_details"] = input_tokens_details
            cached_tokens = input_tokens_details.get("cached_tokens")
            if isinstance(cached_tokens, int):
                provider_metadata["cached_tokens"] = cached_tokens

        output_tokens_details = _serialize_optional_mapping(
            getattr(getattr(response, "usage", None), "output_tokens_details", None)
        )
        if output_tokens_details is not None:
            provider_metadata["output_tokens_details"] = output_tokens_details
            reasoning_tokens = output_tokens_details.get("reasoning_tokens")
            if isinstance(reasoning_tokens, int):
                provider_metadata["reasoning_tokens"] = reasoning_tokens

        return LLMResponse(
            provider=self.name,
            model=getattr(response, "model", None) or request.model or "",
            text=self._extract_response_text(response=response, response_output=response_output),
            tool_calls=tool_calls,
            finish_reason=self._infer_finish_reason(response=response, tool_calls=tool_calls),
            usage=usage,
            response_id=getattr(response, "id", None),
            provider_metadata=provider_metadata,
        )

    def _extract_response_text(
        self,
        *,
        response: Any,
        response_output: Sequence[dict[str, Any]],
    ) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text

        text_parts: list[str] = []
        for item in response_output:
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") not in {"output_text", "text"}:
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)

    def _extract_stream_failed_message(self, response: Any) -> str:
        error = getattr(response, "error", None)
        if error is None:
            return "Grok streaming response failed."
        if isinstance(error, str) and error:
            return error
        message = getattr(error, "message", None)
        if isinstance(message, str) and message:
            return message
        if isinstance(error, dict):
            error_message = error.get("message")
            if isinstance(error_message, str) and error_message:
                return error_message
        return "Grok streaming response failed."

    def _extract_tool_calls(
        self,
        *,
        response_output: Sequence[dict[str, Any]],
        request_tools: Sequence[ToolDefinition],
    ) -> list[ToolCall]:
        tool_schemas = build_tool_schema_map(request_tools)
        parsed_calls: list[ToolCall] = []

        for index, item in enumerate(response_output):
            if item.get("type") != "function_call":
                continue

            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue

            call_id = item.get("call_id") or item.get("id") or f"{name}_{index}"
            raw_arguments = item.get("arguments", "{}")
            if isinstance(raw_arguments, Mapping):
                raw_arguments = json.dumps(raw_arguments, separators=(",", ":"))
            elif not isinstance(raw_arguments, str):
                raw_arguments = str(raw_arguments)

            parsed_calls.append(
                parse_and_validate_tool_call_or_recover(
                    call_id=str(call_id),
                    name=name,
                    raw_arguments=raw_arguments,
                    tool_schemas=tool_schemas,
                )
            )

        return parsed_calls

    def _normalize_usage(self, usage_obj: Any) -> LLMUsage | None:
        input_tokens = _extract_optional_int(usage_obj, "input_tokens")
        output_tokens = _extract_optional_int(usage_obj, "output_tokens")
        total_tokens = _extract_optional_int(usage_obj, "total_tokens")
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _infer_finish_reason(self, *, response: Any, tool_calls: Sequence[ToolCall]) -> str:
        status = getattr(response, "status", None)
        if status == "failed":
            return "error"
        if status == "incomplete":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            if reason == "max_output_tokens":
                return "length"
            if reason == "content_filter":
                return "content_filter"
        if tool_calls:
            return "tool_calls"
        if status == "completed":
            return "stop"
        return "unknown"

    def _map_error(self, error: Exception) -> Exception:
        if isinstance(error, (ProviderResponseError, StreamProtocolError)):
            return error
        if isinstance(error, (AuthenticationError, PermissionDeniedError)):
            return ProviderAuthenticationError(str(error))
        if isinstance(error, RateLimitError):
            return ProviderRateLimitError(str(error))
        if isinstance(error, APITimeoutError):
            return ProviderTimeoutError(str(error))
        if isinstance(error, (APIConnectionError, InternalServerError)):
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
        if isinstance(error, OpenAIError):
            return ProviderResponseError(str(error))
        return ProviderResponseError(str(error))


def _normalize_grok_image_detail(detail: str) -> str:
    if detail == "original":
        return "high"
    if detail in {"auto", "low", "high"}:
        return detail
    raise LLMConfigurationError(
        f"Grok provider does not support image detail '{detail}'."
    )


def _grok_model_uses_encrypted_reasoning(model: str) -> bool:
    normalized = model.strip().lower()
    if not normalized:
        return False
    if "non-reasoning" in normalized:
        return False
    if "reasoning" in normalized or "multi-agent" in normalized:
        return True
    match = re.match(r"^grok-(\d+)", normalized)
    if match is None:
        return False
    return int(match.group(1)) >= 3


def _extract_optional_int(source: Any, field_name: str) -> int | None:
    value: Any
    if isinstance(source, Mapping):
        value = source.get(field_name)
    else:
        value = getattr(source, field_name, None)
    return value if isinstance(value, int) else None


def _normalize_optional_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _serialize_optional_mapping(value: Any) -> dict[str, Any] | None:
    serialized = _serialize_json_compatible(value)
    if isinstance(serialized, dict):
        return serialized
    return None


def _serialize_response_output_items(output: Any) -> list[dict[str, Any]]:
    serialized = _serialize_json_compatible(output)
    if not isinstance(serialized, list):
        return []
    return [item for item in serialized if isinstance(item, dict)]


def _serialize_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serialize_json_compatible(item) for item in value]

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _serialize_json_compatible(to_dict())

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
        return _serialize_json_compatible(dumped)

    if hasattr(value, "__dict__"):
        public_attributes = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if public_attributes:
            return _serialize_json_compatible(public_attributes)

    return str(value)
