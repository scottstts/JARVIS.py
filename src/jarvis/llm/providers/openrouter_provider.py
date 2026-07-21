"""OpenRouter provider adapter using direct HTTP requests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import requests

from ..config import OpenRouterProviderSettings
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
    FinishReason,
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


@dataclass(slots=True, frozen=True)
class _OpenRouterStreamHeaders:
    headers: dict[str, str]


@dataclass(slots=True, frozen=True)
class _OpenRouterStreamDone:
    pass


@dataclass(slots=True, frozen=True)
class _OpenRouterStreamActivity:
    provider_event_type: str


class OpenRouterProvider:
    """Provider implementation for OpenRouter OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        *,
        settings: OpenRouterProviderSettings,
        read_timeout_seconds: float,
        connect_timeout_seconds: float = 30.0,
    ) -> None:
        self._settings = settings
        self._transport_timeouts = ProviderTransportTimeouts(
            connect_seconds=connect_timeout_seconds,
            read_seconds=read_timeout_seconds,
        )

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools=True,
            embeddings=True,
            image_input=True,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.model is None:
            raise LLMConfigurationError("request.model must be set before provider dispatch.")

        payload = self._build_chat_payload(request, stream=False)
        data, response_headers = await self._post_json_with_headers(
            endpoint="/chat/completions",
            payload=payload,
        )
        return self._normalize_chat_response(
            request=request,
            response_json=data,
            response_headers=response_headers,
        )

    async def stream_generate(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[ProviderStreamEvent]:
        if request.model is None:
            raise LLMConfigurationError("request.model must be set before provider dispatch.")

        payload = self._build_chat_payload(request, stream=True)
        accumulated_text: list[str] = []
        streamed_tool_calls: dict[int | str, dict[str, Any]] = {}
        usage: LLMUsage | None = None
        response_id: str | None = None
        response_model = request.model
        raw_finish_reason = "unknown"
        response_header_metadata: dict[str, Any] = {}
        saw_done_sentinel = False
        saw_terminal_choice = False

        async for sse_payload in self._stream_sse_payloads(
            endpoint="/chat/completions",
            payload=payload,
        ):
            if isinstance(sse_payload, _OpenRouterStreamHeaders):
                response_header_metadata = self._extract_response_header_metadata(
                    sse_payload.headers
                )
                generation_id = response_header_metadata.get(
                    "openrouter_generation_id"
                )
                yield ProviderActivityEvent(
                    provider_event_type="http.response_headers",
                    response_id=(
                        str(generation_id) if generation_id is not None else None
                    ),
                )
                continue
            if isinstance(sse_payload, _OpenRouterStreamActivity):
                yield ProviderActivityEvent(
                    provider_event_type=sse_payload.provider_event_type,
                    response_id=response_id,
                )
                continue
            if sse_payload == "[DONE]":
                saw_done_sentinel = True
                yield ProviderActivityEvent(
                    provider_event_type="sse.done",
                    response_id=response_id,
                )
                break

            chunk = self._decode_stream_chunk(sse_payload)
            response_id = chunk.get("id") or response_id
            response_model = chunk.get("model") or response_model
            emitted_semantic_event = False

            error = chunk.get("error")
            if error is not None:
                yield ProviderActivityEvent(
                    provider_event_type="chat.completion.error",
                    response_id=response_id,
                )
                raise self._map_stream_error(
                    error=error,
                    chunk=chunk,
                    response_id=response_id,
                    response_header_metadata=response_header_metadata,
                )

            chunk_usage = self._normalize_usage(chunk.get("usage"))
            if chunk_usage is not None:
                usage = chunk_usage
                emitted_semantic_event = True
                yield UsageDeltaEvent(usage=chunk_usage)

            choices = chunk.get("choices") or []
            for fallback_choice_index, choice in enumerate(choices):
                choice_index = choice.get("index", fallback_choice_index)
                if choice_index not in {0, None}:
                    continue

                delta = choice.get("delta") or {}
                text_delta = self._extract_stream_text(delta.get("content"))
                if text_delta:
                    accumulated_text.append(text_delta)
                    emitted_semantic_event = True
                    yield TextDeltaEvent(delta=text_delta)

                tool_call_events = list(
                    self._extract_stream_tool_call_events(
                        delta.get("tool_calls") or [],
                        tool_call_states=streamed_tool_calls,
                    )
                )
                for tool_call_event in tool_call_events:
                    emitted_semantic_event = True
                    yield tool_call_event

                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    raw_finish_reason = str(finish_reason)
                    saw_terminal_choice = True

            if not emitted_semantic_event:
                yield ProviderActivityEvent(
                    provider_event_type=str(
                        chunk.get("object") or "chat.completion.chunk"
                    ),
                    response_id=response_id,
                )

        if not saw_done_sentinel and not saw_terminal_choice:
            raise StreamProtocolError(
                "OpenRouter stream closed without a terminal chunk or [DONE]."
            )

        response = LLMResponse(
            provider=self.name,
            model=response_model or request.model or "",
            text="".join(accumulated_text),
            tool_calls=self._extract_tool_calls(
                message_tool_calls=self._materialize_stream_tool_calls(streamed_tool_calls),
                request_tools=request.tools,
            ),
            finish_reason=self._map_finish_reason(raw_finish_reason),
            usage=usage,
            response_id=response_id,
            provider_metadata={
                "finish_reason_raw": raw_finish_reason,
                **response_header_metadata,
            },
        )
        yield DoneEvent(response=response)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.model is None:
            raise LLMConfigurationError(
                "No embedding model configured for provider 'openrouter'."
            )

        payload: dict[str, Any] = {
            "model": request.model,
            "input": request.inputs if isinstance(request.inputs, str) else list(request.inputs),
        }
        if request.dimensions is not None:
            payload["dimensions"] = request.dimensions

        data = await self._post_json(
            endpoint="/embeddings",
            payload=payload,
        )

        embeddings = [
            list(item.get("embedding", []))
            for item in data.get("data", [])
        ]
        usage_obj = data.get("usage", {})
        usage = None
        if usage_obj:
            usage = LLMUsage(
                input_tokens=usage_obj.get("prompt_tokens"),
                output_tokens=usage_obj.get("completion_tokens"),
                total_tokens=usage_obj.get("total_tokens"),
            )

        return EmbeddingResponse(
            provider=self.name,
            model=data.get("model", request.model),
            embeddings=embeddings,
            usage=usage,
        )

    async def aclose(self) -> None:
        return

    def _build_chat_payload(self, request: LLMRequest, *, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._to_openrouter_messages(request),
            "stream": stream,
        }
        if request.prompt_cache_key:
            session_id = _openrouter_session_id(request.prompt_cache_key)
            if session_id is not None:
                payload["session_id"] = session_id
        if _openrouter_model_uses_anthropic_system_rules(request.model):
            payload["cache_control"] = {"type": "ephemeral"}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        if self._settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self._settings.reasoning_effort}

        if request.tools:
            payload["tools"] = [self._to_openrouter_tool(tool) for tool in request.tools]
            payload["tool_choice"] = self._to_openrouter_tool_choice(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        return payload

    def _to_openrouter_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        if not _openrouter_model_uses_anthropic_system_rules(request.model):
            return [self._to_openrouter_message(message) for message in request.messages]

        system_parts: list[str] = []
        out_messages: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "system":
                text = _join_text_parts(
                    message.parts,
                    unsupported_message=(
                        "OpenRouter Anthropic-compatible system history only supports "
                        "text parts."
                    ),
                )
                if text:
                    if _openrouter_system_message_is_global(message):
                        system_parts.append(text)
                    else:
                        out_messages.append({"role": "user", "content": text})
                continue

            out_messages.append(self._to_openrouter_message(message))

        if not system_parts:
            return out_messages
        return [
            {"role": "system", "content": "\n\n".join(system_parts)},
            *out_messages,
        ]

    def _to_openrouter_message(self, message: LLMMessage) -> dict[str, Any]:
        role = message.role
        if role == "tool":
            return self._to_openrouter_tool_result_message(message)

        text_parts: list[str] = []
        content: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                text_parts.append(part.text)
                content.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePart):
                if part.file_id is not None:
                    raise LLMConfigurationError(
                        "OpenRouter provider supports image_url, not file_id."
                    )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": part.image_url},
                    }
                )
            elif isinstance(part, ToolCall):
                if role != "assistant":
                    raise LLMConfigurationError(
                        "Tool call history can only appear on assistant messages."
                    )
                tool_calls.append(
                    {
                        "id": part.call_id,
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": part.raw_arguments,
                        },
                    }
                )
            else:
                raise LLMConfigurationError(
                    f"Unsupported OpenRouter message part type: {type(part).__name__}."
                )

        payload: dict[str, Any] = {"role": role}
        if tool_calls:
            payload["tool_calls"] = tool_calls
            payload["content"] = "\n\n".join(text_parts) if text_parts else None
            return payload
        if len(content) == 1 and content[0]["type"] == "text":
            payload["content"] = content[0]["text"]
            return payload
        payload["content"] = content
        return payload

    def _to_openrouter_tool_result_message(self, message: LLMMessage) -> dict[str, Any]:
        if len(message.parts) != 1 or not isinstance(message.parts[0], ToolResultPart):
            raise LLMConfigurationError(
                "Tool-role messages must contain exactly one tool result for OpenRouter."
            )
        part = message.parts[0]
        payload: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": part.call_id,
            "content": part.content,
        }
        if part.name:
            payload["name"] = part.name
        return payload

    def _to_openrouter_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        function_obj: dict[str, Any] = {
            "name": tool.name,
            "parameters": dict(tool.input_schema),
        }
        if tool.description is not None:
            function_obj["description"] = tool.description
        return {"type": "function", "function": function_obj}

    def _to_openrouter_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any] | str:
        if tool_choice.mode == ToolChoiceMode.AUTO:
            return "auto"
        if tool_choice.mode == ToolChoiceMode.REQUIRED:
            return "required"
        if tool_choice.mode == ToolChoiceMode.NONE:
            return "none"
        return {"type": "function", "function": {"name": tool_choice.tool_name}}

    async def _post_json(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        data, _headers = await self._post_json_with_headers(
            endpoint=endpoint,
            payload=payload,
        )
        return data

    async def _post_json_with_headers(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        url, headers, timeout = self._build_request_context(endpoint=endpoint)

        try:
            response = await asyncio.to_thread(
                requests.post,
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(
                str(exc),
                metadata=transport_timeout_metadata(
                    exc,
                    timeouts=self._transport_timeouts,
                ),
            ) from exc
        except requests.ConnectionError as exc:
            raise ProviderTemporaryError(str(exc)) from exc
        except requests.RequestException as exc:
            raise ProviderResponseError(str(exc)) from exc

        if response.status_code >= 400:
            self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError("OpenRouter returned non-JSON response.") from exc
        return data, dict(response.headers)

    async def _stream_sse_payloads(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
    ) -> AsyncIterator[
        str | _OpenRouterStreamHeaders | _OpenRouterStreamActivity
    ]:
        url, headers, timeout = self._build_request_context(endpoint=endpoint)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[
            str
            | _OpenRouterStreamHeaders
            | _OpenRouterStreamActivity
            | Exception
            | _OpenRouterStreamDone
        ] = asyncio.Queue()
        done_sentinel = _OpenRouterStreamDone()
        stop_event = threading.Event()
        response_holder: dict[str, requests.Response | None] = {"response": None}

        def worker() -> None:
            response: requests.Response | None = None
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    stream=True,
                )
                response_holder["response"] = response

                if response.status_code >= 400:
                    self._raise_for_status(response)
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    _OpenRouterStreamHeaders(dict(response.headers)),
                )

                for sse_payload in self._iter_sse_payloads(response):
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, sse_payload)
            except Exception as exc:
                if not stop_event.is_set():
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        self._map_request_exception(exc),
                    )
            finally:
                if response is not None:
                    try:
                        response.close()
                    except requests.RequestException:
                        pass
                if not stop_event.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, done_sentinel)

        thread = threading.Thread(
            target=worker,
            name="openrouter-sse-reader",
            daemon=True,
        )
        thread.start()

        try:
            while True:
                item = await queue.get()
                if isinstance(item, _OpenRouterStreamDone):
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop_event.set()
            response = response_holder["response"]
            if response is not None:
                await asyncio.to_thread(response.close)
            if thread.is_alive():
                await asyncio.to_thread(thread.join, 1.0)

    def _build_request_context(
        self,
        *,
        endpoint: str,
    ) -> tuple[str, dict[str, str], tuple[float, float]]:
        api_key = self._settings.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY is required for the OpenRouter provider.")

        timeout = self._transport_timeouts.as_requests()
        url = f"{self._settings.base_url.rstrip('/')}{endpoint}"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-OpenRouter-Cache": "true",
        }
        if self._settings.site_url:
            headers["HTTP-Referer"] = self._settings.site_url
        if self._settings.app_name:
            headers["X-OpenRouter-Title"] = self._settings.app_name
            headers["X-Title"] = self._settings.app_name
        return url, headers, timeout

    def _iter_sse_payloads(
        self,
        response: requests.Response,
    ) -> Iterator[str | _OpenRouterStreamActivity]:
        data_lines: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=False):
            if isinstance(raw_line, bytes):
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ProviderResponseError(
                        "OpenRouter returned a non-UTF-8 streaming payload."
                    ) from exc
            else:
                line = raw_line
            line = line.rstrip("\r")
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines.clear()
                continue
            if line.startswith(":"):
                activity_name = line[1:].strip()
                yield _OpenRouterStreamActivity(
                    provider_event_type=activity_name or "sse.comment"
                )
                continue
            field_name, separator, value = line.partition(":")
            if separator != ":":
                continue
            if value.startswith(" "):
                value = value[1:]
            if field_name == "data":
                data_lines.append(value)

        if data_lines:
            yield "\n".join(data_lines)

    def _raise_for_status(self, response: requests.Response) -> None:
        status = response.status_code
        error = self._extract_http_error(response)
        message = self._extract_stream_error_message(error) if error is not None else response.text
        metadata = self._build_error_metadata(
            error=error,
            provider_name=None,
            response_id=None,
            response_headers=dict(response.headers),
            fallback_http_code=status,
        )
        if status == 429:
            raise ProviderRateLimitError(message, metadata=metadata)
        if status in {401, 403}:
            raise ProviderAuthenticationError(message, metadata=metadata)
        if status >= 500:
            raise ProviderTemporaryError(message, metadata=metadata)
        raise ProviderBadRequestError(message, metadata=metadata)

    def _normalize_chat_response(
        self,
        *,
        request: LLMRequest,
        response_json: dict[str, Any],
        response_headers: dict[str, str] | None = None,
    ) -> LLMResponse:
        choices = response_json.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})

        text = self._extract_text(message.get("content"))
        tool_calls = self._extract_tool_calls(
            message_tool_calls=message.get("tool_calls", []) or [],
            request_tools=request.tools,
        )

        finish_reason = str(choice.get("finish_reason", "unknown"))
        usage = self._normalize_usage(response_json.get("usage"))

        return LLMResponse(
            provider=self.name,
            model=response_json.get("model", request.model or ""),
            text=text,
            tool_calls=tool_calls,
            finish_reason=self._map_finish_reason(finish_reason),
            usage=usage,
            response_id=response_json.get("id"),
            provider_metadata={
                "finish_reason_raw": finish_reason,
                **self._extract_response_header_metadata(response_headers),
            },
        )

    def _extract_response_header_metadata(
        self,
        response_headers: dict[str, str] | None,
    ) -> dict[str, Any]:
        if response_headers is None:
            return {}

        headers = {key.lower(): value for key, value in response_headers.items()}
        metadata: dict[str, Any] = {}

        cache_status = headers.get("x-openrouter-cache-status")
        if cache_status:
            metadata["openrouter_cache_status"] = str(cache_status).upper()

        cache_age = _parse_header_int(headers.get("x-openrouter-cache-age"))
        if cache_age is not None:
            metadata["openrouter_cache_age_seconds"] = cache_age

        cache_ttl = _parse_header_int(headers.get("x-openrouter-cache-ttl"))
        if cache_ttl is not None:
            metadata["openrouter_cache_ttl_seconds"] = cache_ttl

        generation_id = headers.get("x-generation-id")
        if generation_id:
            metadata["openrouter_generation_id"] = str(generation_id)

        return metadata

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return "".join(text_parts)

    def _extract_stream_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return self._extract_text(content)

    def _extract_stream_tool_call_events(
        self,
        tool_call_deltas: Sequence[dict[str, Any]],
        *,
        tool_call_states: dict[int | str, dict[str, Any]],
    ) -> list[ToolCallDeltaEvent]:
        events: list[ToolCallDeltaEvent] = []
        for fallback_tool_index, tool_call_delta in enumerate(tool_call_deltas):
            tool_index = tool_call_delta.get("index", fallback_tool_index)
            state = tool_call_states.setdefault(
                tool_index,
                {
                    "id": tool_call_delta.get("id") or f"tool_call_{tool_index}",
                    "name": None,
                    "arguments_parts": [],
                },
            )

            call_id = tool_call_delta.get("id")
            if call_id:
                state["id"] = call_id

            function_obj = tool_call_delta.get("function") or {}
            tool_name = function_obj.get("name")
            if tool_name:
                state["name"] = tool_name

            arguments_delta = function_obj.get("arguments")
            if isinstance(arguments_delta, dict):
                arguments_delta = json.dumps(arguments_delta)
            if not arguments_delta:
                continue

            arguments_text = str(arguments_delta)
            state["arguments_parts"].append(arguments_text)
            events.append(
                ToolCallDeltaEvent(
                    call_id=state["id"],
                    tool_name=state["name"],
                    arguments_delta=arguments_text,
                )
            )
        return events

    def _materialize_stream_tool_calls(
        self,
        streamed_tool_calls: dict[int | str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        materialized: list[dict[str, Any]] = []
        for state in streamed_tool_calls.values():
            name = state.get("name")
            if not name:
                continue
            materialized.append(
                {
                    "id": state["id"],
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": "".join(state["arguments_parts"]),
                    },
                }
            )
        return materialized

    def _normalize_usage(self, usage_obj: Any) -> LLMUsage | None:
        if not usage_obj:
            return None
        return LLMUsage(
            input_tokens=usage_obj.get("prompt_tokens"),
            output_tokens=usage_obj.get("completion_tokens"),
            total_tokens=usage_obj.get("total_tokens"),
        )

    def _map_finish_reason(self, finish_reason: str | None) -> FinishReason:
        mapped_reasons: dict[str, FinishReason] = {
            "stop": "stop",
            "tool_calls": "tool_calls",
            "length": "length",
            "content_filter": "content_filter",
            "error": "error",
        }
        return mapped_reasons.get(finish_reason or "unknown", "unknown")

    def _decode_stream_chunk(self, sse_payload: str) -> dict[str, Any]:
        try:
            chunk = json.loads(sse_payload)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("OpenRouter returned malformed streaming JSON.") from exc
        if not isinstance(chunk, dict):
            raise ProviderResponseError("OpenRouter returned a non-object streaming chunk.")
        return chunk

    def _extract_stream_error_message(self, error: Any) -> str:
        if isinstance(error, str) and error:
            return error
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
            code = error.get("code")
            if code is not None:
                return f"OpenRouter streaming error ({code})."
        return "OpenRouter streaming response failed."

    def _map_stream_error(
        self,
        *,
        error: Any,
        chunk: dict[str, Any],
        response_id: str | None,
        response_header_metadata: dict[str, Any],
    ) -> Exception:
        metadata = self._build_error_metadata(
            error=error,
            provider_name=_optional_string(chunk.get("provider")),
            response_id=response_id,
            response_headers=response_header_metadata,
        )
        message = self._extract_stream_error_message(error)
        return ProviderResponseError(message, metadata=metadata)

    def _build_error_metadata(
        self,
        *,
        error: Any,
        provider_name: str | None,
        response_id: str | None,
        response_headers: dict[str, Any] | None,
        fallback_http_code: int | None = None,
    ) -> dict[str, Any]:
        error_obj = error if isinstance(error, dict) else {}
        error_metadata = error_obj.get("metadata")
        metadata_obj = error_metadata if isinstance(error_metadata, dict) else {}
        normalized_headers = {
            str(key).lower(): value
            for key, value in (response_headers or {}).items()
        }
        generation_id = (
            normalized_headers.get("openrouter_generation_id")
            or normalized_headers.get("x-generation-id")
            or response_id
        )
        return {
            "generation_id": _optional_string(generation_id),
            "response_id": response_id,
            "provider_name": (
                provider_name
                or _optional_string(metadata_obj.get("provider_name"))
            ),
            "http_code": (
                _parse_error_code(error_obj.get("code"))
                or fallback_http_code
            ),
            "error_type": _optional_string(metadata_obj.get("error_type")),
            "upstream_provider_code": _optional_string(
                metadata_obj.get("provider_code")
            ),
        }

    def _extract_http_error(self, response: requests.Response) -> Any:
        try:
            payload = response.json()
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload.get("error")

    def _map_request_exception(self, exc: Exception) -> Exception:
        if isinstance(
            exc,
            (
                LLMConfigurationError,
                ProviderAuthenticationError,
                ProviderBadRequestError,
                ProviderRateLimitError,
                ProviderResponseError,
                ProviderTemporaryError,
                ProviderTimeoutError,
                StreamProtocolError,
            ),
        ):
            return exc
        if isinstance(exc, requests.Timeout):
            return ProviderTimeoutError(
                str(exc),
                metadata=transport_timeout_metadata(
                    exc,
                    timeouts=self._transport_timeouts,
                ),
            )
        if isinstance(exc, requests.ConnectionError):
            return ProviderTemporaryError(str(exc))
        if isinstance(exc, requests.RequestException):
            return ProviderResponseError(str(exc))
        return ProviderResponseError(str(exc))

    def _extract_tool_calls(
        self,
        *,
        message_tool_calls: Sequence[dict[str, Any]],
        request_tools: Sequence[ToolDefinition],
    ) -> list[ToolCall]:
        tool_schemas = build_tool_schema_map(request_tools)
        parsed_calls: list[ToolCall] = []
        for index, call in enumerate(message_tool_calls):
            function_obj = call.get("function", {})
            name = function_obj.get("name")
            if not name:
                continue

            call_id = call.get("id") or f"{name}_{index}"
            raw_arguments = function_obj.get("arguments", "{}")
            if isinstance(raw_arguments, dict):
                raw_arguments = json.dumps(raw_arguments)

            parsed_calls.append(
                parse_and_validate_tool_call_or_recover(
                    call_id=call_id,
                    name=name,
                    raw_arguments=raw_arguments,
                    tool_schemas=tool_schemas,
                )
            )
        return parsed_calls


def _parse_header_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_error_code(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _openrouter_model_uses_anthropic_system_rules(model: str | None) -> bool:
    if model is None:
        return False
    normalized = model.lower()
    return normalized.startswith("anthropic/") or "claude" in normalized


def _openrouter_session_id(prompt_cache_key: str) -> str | None:
    normalized = prompt_cache_key.strip()
    if not normalized:
        return None
    if len(normalized) <= 256:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"jarvis-{digest}"


def _openrouter_system_message_is_global(message: LLMMessage) -> bool:
    metadata = message.metadata
    if not metadata:
        return True
    return bool(
        metadata.get("bootstrap_identity")
        or metadata.get("memory_bootstrap")
        or metadata.get("summary_seed")
    )


def _join_text_parts(parts: Sequence[Any], *, unsupported_message: str) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            text_parts.append(part.text)
            continue
        raise LLMConfigurationError(unsupported_message)
    return "\n".join(text_parts).strip()
