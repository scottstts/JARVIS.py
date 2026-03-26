"""LM Studio provider adapter using its local OpenAI-compatible HTTP APIs."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests

from ..config import LMStudioProviderSettings
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
from ..validation import build_tool_schema_map, parse_and_validate_tool_call

_HOST_DOCKER_INTERNAL = "host.docker.internal"


def _running_in_container() -> bool:
    return Path("/.dockerenv").exists()


class LMStudioProvider:
    """Provider implementation for a locally running LM Studio server."""

    def __init__(
        self,
        *,
        settings: LMStudioProviderSettings,
        default_timeout_seconds: float,
    ) -> None:
        self._settings = settings
        self._default_timeout_seconds = default_timeout_seconds

    @property
    def name(self) -> str:
        return "lmstudio"

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            streaming=True,
            tools=True,
            embeddings=False,
            image_input=True,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        resolved_request = await self._resolve_chat_request(request)
        payload = self._build_chat_payload(resolved_request, stream=False)
        data = await self._post_json(
            endpoint="/chat/completions",
            payload=payload,
            timeout_seconds=resolved_request.timeout_seconds,
        )
        return self._normalize_chat_response(request=resolved_request, response_json=data)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        resolved_request = await self._resolve_chat_request(request)
        payload = self._build_chat_payload(resolved_request, stream=True)
        accumulated_text: list[str] = []
        streamed_tool_calls: dict[int | str, dict[str, Any]] = {}
        usage: LLMUsage | None = None
        response_id: str | None = None
        response_model = resolved_request.model
        raw_finish_reason = "unknown"
        saw_done_sentinel = False
        saw_terminal_choice = False

        async for sse_payload in self._stream_sse_payloads(
            endpoint="/chat/completions",
            payload=payload,
            timeout_seconds=resolved_request.timeout_seconds,
        ):
            if sse_payload == "[DONE]":
                saw_done_sentinel = True
                break

            chunk = self._decode_stream_chunk(sse_payload)
            response_id = chunk.get("id") or response_id
            response_model = chunk.get("model") or response_model

            error = chunk.get("error")
            if error is not None:
                raise ProviderResponseError(self._extract_stream_error_message(error))

            chunk_usage = self._normalize_usage(chunk.get("usage"))
            if chunk_usage is not None:
                usage = chunk_usage
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
                    yield TextDeltaEvent(delta=text_delta)

                for tool_call_event in self._extract_stream_tool_call_events(
                    self._stream_tool_call_deltas(delta),
                    tool_call_states=streamed_tool_calls,
                ):
                    yield tool_call_event

                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    raw_finish_reason = str(finish_reason)
                    saw_terminal_choice = True

        if not saw_done_sentinel and not saw_terminal_choice:
            raise StreamProtocolError(
                "LM Studio stream closed without a terminal chunk or [DONE]."
            )

        response = LLMResponse(
            provider=self.name,
            model=response_model or resolved_request.model or "",
            text="".join(accumulated_text),
            tool_calls=self._extract_tool_calls(
                message_tool_calls=self._materialize_stream_tool_calls(streamed_tool_calls),
                request_tools=resolved_request.tools,
            ),
            finish_reason=self._map_finish_reason(raw_finish_reason),
            usage=usage,
            response_id=response_id,
            provider_metadata={"finish_reason_raw": raw_finish_reason},
        )
        yield DoneEvent(response=response)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise UnsupportedCapabilityError("Provider 'lmstudio' does not support embeddings.")

    async def aclose(self) -> None:
        return

    async def _resolve_chat_request(self, request: LLMRequest) -> LLMRequest:
        if request.model is not None:
            return request
        model = await self._select_loaded_chat_model(request)
        return replace(request, model=model)

    async def _select_loaded_chat_model(self, request: LLMRequest) -> str:
        models_payload = await self._get_json(
            endpoint="/models",
            timeout_seconds=request.timeout_seconds,
        )
        models = models_payload.get("models")
        if not isinstance(models, list):
            raise ProviderResponseError("LM Studio returned an invalid model listing.")

        loaded_llms = [model for model in models if self._is_loaded_chat_model(model)]
        if not loaded_llms:
            raise LLMConfigurationError(
                "LM Studio does not have a loaded LLM available. Load a model in LM Studio "
                "before using the lmstudio provider."
            )

        requires_tools = bool(request.tools)
        requires_vision = any(
            isinstance(part, ImagePart)
            for message in request.messages
            for part in message.parts
        )

        candidates = loaded_llms
        if requires_tools:
            tool_capable = [
                model for model in candidates if self._model_supports_tools(model)
            ]
            if not tool_capable:
                raise LLMConfigurationError(
                    "LM Studio does not have a loaded tool-capable LLM available for this request."
                )
            candidates = tool_capable

        if requires_vision:
            vision_capable = [
                model for model in candidates if self._model_supports_vision(model)
            ]
            if not vision_capable:
                raise LLMConfigurationError(
                    "LM Studio does not have a loaded vision-capable LLM available for this request."
                )
            candidates = vision_capable

        identifiers = [self._loaded_instance_identifier(model) for model in candidates]
        unique_identifiers = sorted({identifier for identifier in identifiers if identifier})
        if len(unique_identifiers) != 1:
            raise LLMConfigurationError(
                "LM Studio has multiple loaded LLMs that match this request. Keep exactly one "
                "matching model loaded, or pass request.model explicitly."
            )
        return unique_identifiers[0]

    def _is_loaded_chat_model(self, model: Any) -> bool:
        if not isinstance(model, dict):
            return False
        if model.get("type") != "llm":
            return False
        loaded_instances = model.get("loaded_instances")
        return isinstance(loaded_instances, list) and bool(loaded_instances)

    def _loaded_instance_identifier(self, model: dict[str, Any]) -> str | None:
        loaded_instances = model.get("loaded_instances")
        if not isinstance(loaded_instances, list) or not loaded_instances:
            return None

        first_instance = loaded_instances[0]
        if isinstance(first_instance, dict):
            instance_id = first_instance.get("id")
            if isinstance(instance_id, str) and instance_id.strip():
                return instance_id.strip()

        key = model.get("key")
        if not isinstance(key, str) or not key.strip():
            return None
        identifier = key.strip()
        if "/" in identifier:
            return identifier

        publisher = model.get("publisher")
        if isinstance(publisher, str) and publisher.strip():
            return f"{publisher.strip()}/{identifier}"
        return identifier

    def _model_supports_tools(self, model: dict[str, Any]) -> bool:
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, dict):
            return False
        return bool(capabilities.get("trained_for_tool_use"))

    def _model_supports_vision(self, model: dict[str, Any]) -> bool:
        capabilities = model.get("capabilities")
        if not isinstance(capabilities, dict):
            return False
        return bool(capabilities.get("vision"))

    def _build_chat_payload(self, request: LLMRequest, *, stream: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._to_lmstudio_message(message) for message in request.messages],
            "stream": stream,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        if request.tools:
            payload["tools"] = [self._to_lmstudio_tool(tool) for tool in request.tools]
            payload["tool_choice"] = self._to_lmstudio_tool_choice(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        return payload

    def _to_lmstudio_message(self, message: LLMMessage) -> dict[str, Any]:
        role = "system" if message.role in {"system", "developer"} else message.role
        if role == "tool":
            return self._to_lmstudio_tool_result_message(message)

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
                        "LM Studio provider supports image_url, not file_id."
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
                    f"Unsupported LM Studio message part type: {type(part).__name__}."
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

    def _to_lmstudio_tool_result_message(self, message: LLMMessage) -> dict[str, Any]:
        if len(message.parts) != 1 or not isinstance(message.parts[0], ToolResultPart):
            raise LLMConfigurationError(
                "Tool-role messages must contain exactly one tool result for LM Studio."
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

    def _to_lmstudio_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        function_obj: dict[str, Any] = {
            "name": tool.name,
            "parameters": dict(tool.input_schema),
        }
        if tool.description is not None:
            function_obj["description"] = tool.description
        return {"type": "function", "function": function_obj}

    def _to_lmstudio_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any] | str:
        if tool_choice.mode == ToolChoiceMode.AUTO:
            return "auto"
        if tool_choice.mode == ToolChoiceMode.REQUIRED:
            return "required"
        if tool_choice.mode == ToolChoiceMode.NONE:
            return "none"
        return {"type": "function", "function": {"name": tool_choice.tool_name}}

    async def _get_json(
        self,
        *,
        endpoint: str,
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        url, headers, timeout = self._build_request_context(
            endpoint=endpoint,
            api_prefix="/api/v1",
            timeout_seconds=timeout_seconds,
        )
        try:
            response = await asyncio.to_thread(
                requests.get,
                url,
                headers=headers,
                timeout=timeout,
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except requests.ConnectionError as exc:
            raise ProviderTemporaryError(str(exc)) from exc
        except requests.RequestException as exc:
            raise ProviderResponseError(str(exc)) from exc

        if response.status_code >= 400:
            self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError("LM Studio returned non-JSON response.") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError("LM Studio returned a non-object JSON response.")
        return data

    async def _post_json(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        url, headers, timeout = self._build_request_context(
            endpoint=endpoint,
            api_prefix="/v1",
            timeout_seconds=timeout_seconds,
        )

        try:
            response = await asyncio.to_thread(
                requests.post,
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.Timeout as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except requests.ConnectionError as exc:
            raise ProviderTemporaryError(str(exc)) from exc
        except requests.RequestException as exc:
            raise ProviderResponseError(str(exc)) from exc

        if response.status_code >= 400:
            self._raise_for_status(response)

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderResponseError("LM Studio returned non-JSON response.") from exc
        if not isinstance(data, dict):
            raise ProviderResponseError("LM Studio returned a non-object JSON response.")
        return data

    async def _stream_sse_payloads(
        self,
        *,
        endpoint: str,
        payload: dict[str, Any],
        timeout_seconds: float | None,
    ) -> AsyncIterator[str]:
        url, headers, timeout = self._build_request_context(
            endpoint=endpoint,
            api_prefix="/v1",
            timeout_seconds=timeout_seconds,
        )
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | Exception | object] = asyncio.Queue()
        done_sentinel = object()
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
            name="lmstudio-sse-reader",
            daemon=True,
        )
        thread.start()

        try:
            while True:
                item = await queue.get()
                if item is done_sentinel:
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
        api_prefix: str,
        timeout_seconds: float | None,
    ) -> tuple[str, dict[str, str], float]:
        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds
        server_base_url = self._server_base_url()
        url = f"{server_base_url}{api_prefix}{endpoint}"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        return url, headers, timeout

    def _server_base_url(self) -> str:
        base_url = self._settings.base_url.strip().rstrip("/")
        for suffix in ("/api/v1", "/v1"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        return _rewrite_loopback_base_url_for_container(base_url)

    def _iter_sse_payloads(self, response: requests.Response) -> Iterator[str]:
        data_lines: list[str] = []
        for raw_line in response.iter_lines(decode_unicode=False):
            if isinstance(raw_line, bytes):
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ProviderResponseError(
                        "LM Studio returned a non-UTF-8 streaming payload."
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
        message = response.text
        if status == 429:
            raise ProviderRateLimitError(message)
        if status in {401, 403}:
            raise ProviderAuthenticationError(message)
        if status >= 500:
            raise ProviderTemporaryError(message)
        raise ProviderBadRequestError(message)

    def _normalize_chat_response(
        self,
        *,
        request: LLMRequest,
        response_json: dict[str, Any],
    ) -> LLMResponse:
        choices = response_json.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})

        text = self._extract_text(message.get("content"))
        tool_calls = self._extract_tool_calls(
            message_tool_calls=self._message_tool_calls(message),
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
            provider_metadata={"finish_reason_raw": finish_reason},
        )

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

    def _stream_tool_call_deltas(self, delta: dict[str, Any]) -> list[dict[str, Any]]:
        tool_call_deltas = delta.get("tool_calls")
        if isinstance(tool_call_deltas, list):
            return tool_call_deltas

        legacy_function_call = delta.get("function_call")
        if isinstance(legacy_function_call, dict):
            return [
                {
                    "index": 0,
                    "function": legacy_function_call,
                }
            ]
        return []

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

    def _message_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        if not isinstance(message, dict):
            return []

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            return tool_calls

        legacy_function_call = message.get("function_call")
        if isinstance(legacy_function_call, dict):
            return [
                {
                    "id": message.get("tool_call_id") or "tool_call_0",
                    "type": "function",
                    "function": legacy_function_call,
                }
            ]
        return []

    def _normalize_usage(self, usage_obj: Any) -> LLMUsage | None:
        if not usage_obj:
            return None
        return LLMUsage(
            input_tokens=usage_obj.get("prompt_tokens"),
            output_tokens=usage_obj.get("completion_tokens"),
            total_tokens=usage_obj.get("total_tokens"),
        )

    def _map_finish_reason(self, finish_reason: str | None) -> str:
        return {
            "stop": "stop",
            "tool_calls": "tool_calls",
            "function_call": "tool_calls",
            "length": "length",
            "content_filter": "content_filter",
            "error": "error",
        }.get(finish_reason or "unknown", "unknown")

    def _decode_stream_chunk(self, sse_payload: str) -> dict[str, Any]:
        try:
            chunk = json.loads(sse_payload)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("LM Studio returned malformed streaming JSON.") from exc
        if not isinstance(chunk, dict):
            raise ProviderResponseError("LM Studio returned a non-object streaming chunk.")
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
                return f"LM Studio streaming error ({code})."
        return "LM Studio streaming response failed."

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
                UnsupportedCapabilityError,
            ),
        ):
            return exc
        if isinstance(exc, requests.Timeout):
            return ProviderTimeoutError(str(exc))
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
                parse_and_validate_tool_call(
                    call_id=call_id,
                    name=name,
                    raw_arguments=raw_arguments,
                    tool_schemas=tool_schemas,
                )
            )
        return parsed_calls


def _rewrite_loopback_base_url_for_container(base_url: str) -> str:
    if not _running_in_container():
        return base_url

    parsed = urlsplit(base_url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return base_url

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    port = f":{parsed.port}" if parsed.port is not None else ""
    rewritten_netloc = f"{userinfo}{_HOST_DOCKER_INTERNAL}{port}"
    return urlunsplit(
        (
            parsed.scheme,
            rewritten_netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )
