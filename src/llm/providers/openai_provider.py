"""OpenAI provider adapter that normalizes Responses API behavior."""

from __future__ import annotations

import asyncio
import copy
import inspect
import os
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

from ..config import OpenAIProviderSettings
from ..errors import (
    LLMConfigurationError,
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTemporaryError,
    ProviderTimeoutError,
    StreamProtocolError,
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
from ..validation import (
    build_tool_schema_map,
    get_tool_schema,
    load_tool_call_arguments,
    validate_tool_call_arguments,
)


class OpenAIProvider:
    """Provider implementation for OpenAI Responses + Embeddings APIs."""

    def __init__(
        self,
        *,
        settings: OpenAIProviderSettings,
        default_timeout_seconds: float,
    ) -> None:
        self._settings = settings
        self._default_timeout_seconds = default_timeout_seconds
        self._client: AsyncOpenAI | None = None
        self._client_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "openai"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools=True,
            embeddings=True,
            image_input=True,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._build_response_create_kwargs(request, stream=False)
        client = await self._client_instance()
        try:
            response = await client.responses.create(**kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc
        return self._normalize_response(request=request, response=response)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        kwargs = self._build_response_create_kwargs(request, stream=True)
        client = await self._client_instance()

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
                    message = "OpenAI streaming response failed."
                    response_error = getattr(event.response, "error", None)
                    if response_error is not None:
                        message = response_error.message
                    raise ProviderResponseError(message)

                if event_type == "error":
                    raise ProviderResponseError(event.message)
        except Exception as exc:
            if isinstance(exc, (ProviderResponseError, StreamProtocolError)):
                raise
            raise self._map_error(exc) from exc

        if not saw_completion:
            raise StreamProtocolError(
                "OpenAI stream closed without a response.completed event."
            )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.model is None:
            raise LLMConfigurationError("No embedding model configured for provider 'openai'.")
        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": request.inputs if isinstance(request.inputs, str) else list(request.inputs),
        }
        if request.dimensions is not None:
            kwargs["dimensions"] = request.dimensions
        if request.user is not None:
            kwargs["user"] = request.user
        if request.timeout_seconds is not None:
            kwargs["timeout"] = request.timeout_seconds

        client = await self._client_instance()
        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc

        usage = LLMUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=None,
            total_tokens=response.usage.total_tokens,
        )
        return EmbeddingResponse(
            provider=self.name,
            model=response.model,
            embeddings=[list(entry.embedding) for entry in response.data],
            usage=usage,
        )

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

            api_key = self._settings.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMConfigurationError(
                    "OPENAI_API_KEY is required for the OpenAI provider."
                )

            self._client = AsyncOpenAI(
                api_key=api_key,
                base_url=self._settings.base_url or os.getenv("OPENAI_BASE_URL"),
                organization=self._settings.organization or os.getenv("OPENAI_ORG_ID"),
                project=self._settings.project or os.getenv("OPENAI_PROJECT_ID"),
                timeout=self._default_timeout_seconds,
                max_retries=0,
            )
            return self._client

    def _build_response_create_kwargs(self, request: LLMRequest, *, stream: bool) -> dict[str, Any]:
        if request.model is None:
            raise LLMConfigurationError("request.model must be set before provider dispatch.")

        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": [
                item
                for message in request.messages
                for item in self._to_openai_input_items(message)
            ],
            "stream": stream,
            "parallel_tool_calls": request.parallel_tool_calls,
        }

        if request.instructions is not None:
            kwargs["instructions"] = request.instructions
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if request.metadata is not None:
            kwargs["metadata"] = dict(request.metadata)
        if request.safety_identifier is not None:
            kwargs["safety_identifier"] = request.safety_identifier
        if request.prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = request.prompt_cache_key
        if request.timeout_seconds is not None:
            kwargs["timeout"] = request.timeout_seconds

        reasoning: dict[str, str] = {}
        if self._settings.reasoning_effort is not None:
            reasoning["effort"] = self._settings.reasoning_effort
        if self._settings.reasoning_summary is not None:
            reasoning["summary"] = self._settings.reasoning_summary
        if reasoning:
            kwargs["reasoning"] = reasoning

        if self._settings.text_verbosity is not None:
            kwargs["text"] = {"verbosity": self._settings.text_verbosity}

        if request.tools:
            kwargs["tools"] = [self._to_openai_tool(tool) for tool in request.tools]
            kwargs["tool_choice"] = self._to_openai_tool_choice(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        return kwargs

    def _to_openai_input_items(self, message: LLMMessage) -> list[dict[str, Any]]:
        if message.role == "tool":
            return self._to_openai_tool_result_items(message)

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
                        "OpenAI provider does not support assistant image history items."
                    )
                image_item: dict[str, Any] = {
                    "type": "input_image",
                    "detail": part.detail,
                }
                if part.image_url is not None:
                    image_item["image_url"] = part.image_url
                elif part.file_id is not None:
                    image_item["file_id"] = part.file_id
                content.append(image_item)
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
                    f"Unsupported OpenAI message part type: {type(part).__name__}."
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

    def _to_openai_tool_result_items(self, message: LLMMessage) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for part in message.parts:
            if not isinstance(part, ToolResultPart):
                raise LLMConfigurationError(
                    "Tool-role messages can only contain tool results for OpenAI."
                )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": part.call_id,
                    "output": part.content,
                }
            )
        return items

    def _to_openai_message(self, message: LLMMessage) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
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
                image_item: dict[str, Any] = {
                    "type": "input_image",
                    "detail": part.detail,
                }
                if part.image_url is not None:
                    image_item["image_url"] = part.image_url
                elif part.file_id is not None:
                    image_item["file_id"] = part.file_id
                content.append(image_item)

        return {
            "type": "message",
            "role": message.role,
            "content": content,
        }

    def _to_openai_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        parameters = dict(tool.input_schema)
        if tool.strict:
            parameters = _normalize_openai_strict_schema(parameters)
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
            "strict": tool.strict,
        }

    def _to_openai_tool_choice(self, tool_choice: ToolChoice) -> Any:
        if tool_choice.mode == ToolChoiceMode.AUTO:
            return "auto"
        if tool_choice.mode == ToolChoiceMode.REQUIRED:
            return "required"
        if tool_choice.mode == ToolChoiceMode.NONE:
            return "none"
        return {
            "type": "function",
            "name": tool_choice.tool_name,
        }

    def _normalize_response(self, *, request: LLMRequest, response: Any) -> LLMResponse:
        tool_calls = self._extract_tool_calls(
            response_output=response.output,
            request_tools=request.tools,
        )
        usage = self._normalize_usage(
            input_tokens=getattr(response.usage, "input_tokens", None) if response.usage else None,
            output_tokens=getattr(response.usage, "output_tokens", None) if response.usage else None,
            total_tokens=getattr(response.usage, "total_tokens", None) if response.usage else None,
        )

        return LLMResponse(
            provider=self.name,
            model=response.model,
            text=response.output_text,
            tool_calls=tool_calls,
            finish_reason=self._infer_finish_reason(response=response, tool_calls=tool_calls),
            usage=usage,
            response_id=response.id,
            provider_metadata={
                "status": response.status,
                "incomplete_reason": (
                    response.incomplete_details.reason
                    if response.incomplete_details is not None
                    else None
                ),
            },
        )

    def _extract_tool_calls(
        self,
        *,
        response_output: Sequence[Any],
        request_tools: Sequence[ToolDefinition],
    ) -> list[ToolCall]:
        tool_schemas = build_tool_schema_map(request_tools)
        tool_definitions = {tool.name: tool for tool in request_tools}
        validated_calls: list[ToolCall] = []
        for item in response_output:
            if getattr(item, "type", None) != "function_call":
                continue

            schema = get_tool_schema(
                call_id=item.call_id,
                name=item.name,
                tool_schemas=tool_schemas,
            )
            arguments = load_tool_call_arguments(
                call_id=item.call_id,
                name=item.name,
                raw_arguments=item.arguments,
            )
            tool = tool_definitions.get(item.name)
            if tool is not None and tool.strict:
                arguments = _sanitize_openai_returned_arguments(arguments, schema)
            validate_tool_call_arguments(
                call_id=item.call_id,
                name=item.name,
                arguments=arguments,
                schema=schema,
            )
            validated_calls.append(
                ToolCall(
                    call_id=item.call_id,
                    name=item.name,
                    arguments=arguments,
                    raw_arguments=item.arguments,
                )
            )
        return validated_calls

    def _normalize_usage(
        self,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
    ) -> LLMUsage | None:
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
            reason = getattr(response.incomplete_details, "reason", None)
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
        if isinstance(error, ProviderResponseError | StreamProtocolError):
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


def _normalize_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(schema)
    return _normalize_openai_schema_node(normalized)


def _normalize_openai_schema_node(node: Any) -> Any:
    if not isinstance(node, dict):
        return node

    if "items" in node:
        node["items"] = _normalize_openai_schema_node(node["items"])

    for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
        value = node.get(key)
        if isinstance(value, list):
            node[key] = [_normalize_openai_schema_node(item) for item in value]

    properties = node.get("properties")
    if isinstance(properties, dict):
        original_required = set(node.get("required", []) or [])
        normalized_properties: dict[str, Any] = {}
        required_names: list[str] = []

        for name, child in properties.items():
            normalized_child = _normalize_openai_schema_node(child)
            if name not in original_required:
                normalized_child = _wrap_schema_as_nullable(normalized_child)
            normalized_properties[name] = normalized_child
            required_names.append(name)

        node["properties"] = normalized_properties
        node["required"] = required_names
        node.setdefault("additionalProperties", False)

    return node


def _wrap_schema_as_nullable(schema: dict[str, Any]) -> dict[str, Any]:
    if _schema_allows_null(schema):
        return schema
    return {
        "anyOf": [
            schema,
            {"type": "null"},
        ]
    }


def _schema_allows_null(schema: dict[str, Any]) -> bool:
    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        return any(_schema_allows_null(item) for item in any_of if isinstance(item, dict))
    return False


def _sanitize_openai_returned_arguments(
    arguments: dict[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    return _sanitize_openai_value(arguments, schema)


def _sanitize_openai_value(value: Any, schema: Mapping[str, Any] | None) -> Any:
    if schema is None:
        return value

    if isinstance(value, dict):
        return _sanitize_openai_object(value, schema)

    items_schema = schema.get("items")
    if isinstance(value, list) and isinstance(items_schema, Mapping):
        return [
            _sanitize_openai_value(item, items_schema)
            for item in value
        ]

    return value


def _sanitize_openai_object(
    value: dict[str, Any],
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    properties = schema.get("properties")
    required = set(schema.get("required", []) or [])
    if not isinstance(properties, Mapping):
        return value

    sanitized: dict[str, Any] = {}
    for key, child_value in value.items():
        child_schema = properties.get(key)
        if (
            child_value is None
            and key not in required
            and isinstance(child_schema, Mapping)
            and not _schema_mapping_allows_null(child_schema)
        ):
            continue

        if isinstance(child_schema, Mapping):
            sanitized[key] = _sanitize_openai_value(child_value, child_schema)
        else:
            sanitized[key] = child_value

    return sanitized


def _schema_mapping_allows_null(schema: Mapping[str, Any]) -> bool:
    return _schema_allows_null(dict(schema))
