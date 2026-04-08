"""LM Studio provider adapter using its local OpenAI-compatible Responses API."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass, replace
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
from ..validation import build_tool_schema_map, parse_and_validate_tool_call_or_recover

_HOST_DOCKER_INTERNAL = "host.docker.internal"
_STATEFUL_HISTORY_CACHE_LIMIT = 1024


def _running_in_container() -> bool:
    return Path("/.dockerenv").exists()


@dataclass(slots=True, frozen=True)
class _ResolvedResponseRequest:
    request: LLMRequest
    full_input_items: tuple[dict[str, Any], ...]
    full_history_key: tuple[str, ...]
    previous_response_id: str | None
    payload: dict[str, Any]


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
        self._stateful_response_ids: OrderedDict[tuple[str, tuple[str, ...]], str] = OrderedDict()
        self._stateful_response_ids_lock = threading.Lock()

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
        resolved = await self._resolve_response_request(request, stream=False)
        data = await self._post_with_stateful_fallback(resolved)
        response = self._normalize_response(request=resolved.request, response_json=data)
        self._remember_stateful_response(resolved=resolved, response=response)
        return response

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        resolved = await self._resolve_response_request(request, stream=True)
        attempt_resolved = resolved
        retried_without_previous_response_id = False

        while True:
            emitted_any = False
            saw_completion = False
            tool_name_by_item_id: dict[str, str] = {}
            call_id_by_item_id: dict[str, str] = {}
            try:
                async for sse_payload in self._stream_sse_payloads(
                    endpoint="/responses",
                    payload=attempt_resolved.payload,
                    timeout_seconds=attempt_resolved.request.timeout_seconds,
                ):
                    if sse_payload == "[DONE]":
                        continue

                    emitted_any = True
                    event = self._decode_stream_event(sse_payload)
                    event_type = str(event.get("type", ""))

                    if event_type == "response.output_text.delta":
                        delta = event.get("delta")
                        if isinstance(delta, str) and delta:
                            yield TextDeltaEvent(delta=delta)
                        continue

                    if event_type in {"response.output_item.added", "response.output_item.done"}:
                        item = event.get("item")
                        if isinstance(item, dict) and item.get("type") == "function_call":
                            item_id = item.get("id")
                            if isinstance(item_id, str) and item_id:
                                name = item.get("name")
                                call_id = item.get("call_id")
                                if isinstance(name, str) and name:
                                    tool_name_by_item_id[item_id] = name
                                if isinstance(call_id, str) and call_id:
                                    call_id_by_item_id[item_id] = call_id
                        continue

                    if event_type == "response.function_call_arguments.delta":
                        item_id = str(event.get("item_id", "")).strip()
                        arguments_delta = self._normalize_function_call_arguments_chunk(
                            event.get("delta")
                        )
                        if not item_id or arguments_delta is None:
                            continue
                        yield ToolCallDeltaEvent(
                            call_id=call_id_by_item_id.get(item_id, item_id),
                            tool_name=tool_name_by_item_id.get(item_id),
                            arguments_delta=arguments_delta,
                        )
                        continue

                    if event_type == "response.function_call_arguments.done":
                        item_id = str(event.get("item_id", "")).strip()
                        arguments_text = self._normalize_function_call_arguments_chunk(
                            event.get("arguments")
                        )
                        if not item_id or arguments_text is None:
                            continue
                        tool_name = event.get("name")
                        yield ToolCallDeltaEvent(
                            call_id=call_id_by_item_id.get(item_id, item_id),
                            tool_name=(
                                tool_name
                                if isinstance(tool_name, str)
                                else tool_name_by_item_id.get(item_id)
                            ),
                            arguments_delta=arguments_text,
                        )
                        continue

                    if event_type == "response.completed":
                        response_json = event.get("response")
                        if not isinstance(response_json, dict):
                            raise ProviderResponseError(
                                "LM Studio response.completed event did not include a response object."
                            )
                        normalized = self._normalize_response(
                            request=attempt_resolved.request,
                            response_json=response_json,
                        )
                        self._remember_stateful_response(
                            resolved=attempt_resolved,
                            response=normalized,
                        )
                        if normalized.usage is not None:
                            yield UsageDeltaEvent(usage=normalized.usage)
                        yield DoneEvent(response=normalized)
                        saw_completion = True
                        continue

                    if event_type == "response.failed":
                        raise ProviderResponseError(self._extract_response_failed_message(event))

                    if event_type == "error":
                        raise ProviderResponseError(
                            self._extract_stream_error_message(event.get("error"))
                        )

                if not saw_completion:
                    raise StreamProtocolError(
                        "LM Studio stream closed without a response.completed event."
                    )
                return
            except ProviderBadRequestError as exc:
                if (
                    emitted_any
                    or retried_without_previous_response_id
                    or not self._should_retry_without_previous_response_id(
                        exc,
                        previous_response_id=attempt_resolved.previous_response_id,
                    )
                ):
                    raise
                self._forget_response_id(attempt_resolved.previous_response_id)
                attempt_resolved = self._without_previous_response_id(
                    attempt_resolved,
                    stream=True,
                )
                retried_without_previous_response_id = True

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise UnsupportedCapabilityError("Provider 'lmstudio' does not support embeddings.")

    async def aclose(self) -> None:
        return

    async def _resolve_response_request(
        self,
        request: LLMRequest,
        *,
        stream: bool,
    ) -> _ResolvedResponseRequest:
        resolved_request = await self._resolve_chat_request(request)
        full_input_items = tuple(
            item
            for message in resolved_request.messages
            for item in self._to_response_input_items(message)
        )
        full_history_key = tuple(
            self._serialize_history_item(item)
            for item in full_input_items
        )
        previous_response_id, prefix_length = self._find_previous_response_id(
            model=resolved_request.model or "",
            full_history_key=full_history_key,
        )
        input_items = full_input_items[prefix_length:] if previous_response_id else full_input_items
        payload = self._build_response_payload(
            resolved_request,
            input_items=input_items,
            previous_response_id=previous_response_id,
            stream=stream,
        )
        return _ResolvedResponseRequest(
            request=resolved_request,
            full_input_items=full_input_items,
            full_history_key=full_history_key,
            previous_response_id=previous_response_id,
            payload=payload,
        )

    async def _post_with_stateful_fallback(
        self,
        resolved: _ResolvedResponseRequest,
    ) -> dict[str, Any]:
        attempt_resolved = resolved
        retried_without_previous_response_id = False

        while True:
            try:
                return await self._post_json(
                    endpoint="/responses",
                    payload=attempt_resolved.payload,
                    timeout_seconds=attempt_resolved.request.timeout_seconds,
                )
            except ProviderBadRequestError as exc:
                if (
                    retried_without_previous_response_id
                    or not self._should_retry_without_previous_response_id(
                        exc,
                        previous_response_id=attempt_resolved.previous_response_id,
                    )
                ):
                    raise
                self._forget_response_id(attempt_resolved.previous_response_id)
                attempt_resolved = self._without_previous_response_id(
                    attempt_resolved,
                    stream=False,
                )
                retried_without_previous_response_id = True

    def _without_previous_response_id(
        self,
        resolved: _ResolvedResponseRequest,
        *,
        stream: bool,
    ) -> _ResolvedResponseRequest:
        payload = self._build_response_payload(
            resolved.request,
            input_items=resolved.full_input_items,
            previous_response_id=None,
            stream=stream,
        )
        return _ResolvedResponseRequest(
            request=resolved.request,
            full_input_items=resolved.full_input_items,
            full_history_key=resolved.full_history_key,
            previous_response_id=None,
            payload=payload,
        )

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

    def _build_response_payload(
        self,
        request: LLMRequest,
        *,
        input_items: Sequence[dict[str, Any]],
        previous_response_id: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "input": list(input_items),
            "stream": stream,
            "store": True,
            "parallel_tool_calls": request.parallel_tool_calls,
        }
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if request.instructions is not None:
            payload["instructions"] = request.instructions
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = request.max_output_tokens
        if request.metadata is not None:
            payload["metadata"] = dict(request.metadata)

        if request.tools:
            payload["tools"] = [self._to_response_tool(tool) for tool in request.tools]
            payload["tool_choice"] = self._to_response_tool_choice(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        return payload

    def _to_response_input_items(self, message: LLMMessage) -> list[dict[str, Any]]:
        if message.role == "tool":
            return self._to_response_tool_result_items(message)

        role = message.role
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
                        "LM Studio provider does not support assistant image history items."
                    )
                if part.file_id is not None:
                    raise LLMConfigurationError(
                        "LM Studio provider supports image_url, not file_id."
                    )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": part.image_url,
                        "detail": part.detail,
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
                    f"Unsupported LM Studio message part type: {type(part).__name__}."
                )

        items: list[dict[str, Any]] = []
        if content:
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": content,
                }
            )
        items.extend(tool_call_items)
        return items

    def _to_response_tool_result_items(self, message: LLMMessage) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for part in message.parts:
            if not isinstance(part, ToolResultPart):
                raise LLMConfigurationError(
                    "Tool-role messages can only contain tool results for LM Studio."
                )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": part.call_id,
                    "output": part.content,
                }
            )
        return items

    def _to_response_tool(self, tool: ToolDefinition) -> dict[str, Any]:
        function_obj: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "parameters": dict(tool.input_schema),
        }
        if tool.description is not None:
            function_obj["description"] = tool.description
        if tool.strict:
            function_obj["strict"] = True
        return function_obj

    def _to_response_tool_choice(self, tool_choice: ToolChoice) -> dict[str, Any] | str:
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

    def _normalize_response(
        self,
        *,
        request: LLMRequest,
        response_json: dict[str, Any],
    ) -> LLMResponse:
        output = response_json.get("output")
        if not isinstance(output, list):
            output = []

        tool_calls = self._extract_tool_calls(
            response_output=output,
            request_tools=request.tools,
        )
        usage = self._normalize_usage(response_json.get("usage"))

        return LLMResponse(
            provider=self.name,
            model=str(response_json.get("model") or request.model or ""),
            text=self._extract_response_text(response_json=response_json, response_output=output),
            tool_calls=tool_calls,
            finish_reason=self._infer_finish_reason(
                response_json=response_json,
                tool_calls=tool_calls,
            ),
            usage=usage,
            response_id=self._normalize_optional_string(response_json.get("id")),
            provider_metadata={
                "status": self._normalize_optional_string(response_json.get("status")),
                "previous_response_id": self._normalize_optional_string(
                    response_json.get("previous_response_id")
                ),
                "incomplete_reason": self._extract_incomplete_reason(
                    response_json.get("incomplete_details")
                ),
            },
        )

    def _extract_response_text(
        self,
        *,
        response_json: dict[str, Any],
        response_output: Sequence[dict[str, Any]],
    ) -> str:
        output_text = response_json.get("output_text")
        if isinstance(output_text, str):
            return output_text

        text_parts: list[str] = []
        for item in response_output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"}:
                    text = part.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
        return "".join(text_parts)

    def _extract_tool_calls(
        self,
        *,
        response_output: Sequence[dict[str, Any]],
        request_tools: Sequence[ToolDefinition],
    ) -> list[ToolCall]:
        tool_schemas = build_tool_schema_map(request_tools)
        parsed_calls: list[ToolCall] = []
        for index, item in enumerate(response_output):
            if not isinstance(item, dict) or item.get("type") != "function_call":
                continue

            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue

            call_id = item.get("call_id") or item.get("id") or f"{name}_{index}"
            raw_arguments = item.get("arguments", "{}")
            if isinstance(raw_arguments, dict):
                raw_arguments = json.dumps(raw_arguments)

            parsed_calls.append(
                parse_and_validate_tool_call_or_recover(
                    call_id=str(call_id),
                    name=name,
                    raw_arguments=str(raw_arguments),
                    tool_schemas=tool_schemas,
                )
            )
        return parsed_calls

    def _normalize_usage(self, usage_obj: Any) -> LLMUsage | None:
        if not isinstance(usage_obj, dict):
            return None

        input_tokens = self._coerce_optional_int(
            usage_obj.get("input_tokens", usage_obj.get("prompt_tokens"))
        )
        output_tokens = self._coerce_optional_int(
            usage_obj.get("output_tokens", usage_obj.get("completion_tokens"))
        )
        total_tokens = self._coerce_optional_int(usage_obj.get("total_tokens"))
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None
        return LLMUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _infer_finish_reason(
        self,
        *,
        response_json: dict[str, Any],
        tool_calls: Sequence[ToolCall],
    ) -> str:
        status = self._normalize_optional_string(response_json.get("status"))
        if status == "failed":
            return "error"
        if status == "incomplete":
            incomplete_reason = self._extract_incomplete_reason(
                response_json.get("incomplete_details")
            )
            if incomplete_reason == "max_output_tokens":
                return "length"
            if incomplete_reason == "content_filter":
                return "content_filter"
        if tool_calls:
            return "tool_calls"
        if status == "completed":
            return "stop"
        return "unknown"

    def _find_previous_response_id(
        self,
        *,
        model: str,
        full_history_key: tuple[str, ...],
    ) -> tuple[str | None, int]:
        if len(full_history_key) < 2:
            return None, 0

        with self._stateful_response_ids_lock:
            for prefix_length in range(len(full_history_key) - 1, 0, -1):
                cache_key = (model, full_history_key[:prefix_length])
                response_id = self._stateful_response_ids.get(cache_key)
                if response_id is None:
                    continue
                self._stateful_response_ids.move_to_end(cache_key)
                return response_id, prefix_length
        return None, 0

    def _remember_stateful_response(
        self,
        *,
        resolved: _ResolvedResponseRequest,
        response: LLMResponse,
    ) -> None:
        if response.response_id is None:
            return

        response_items = self._response_output_items(response)
        if not response_items:
            return

        history_key = resolved.full_history_key + tuple(
            self._serialize_history_item(item)
            for item in response_items
        )
        model = response.model or resolved.request.model or ""
        cache_key = (model, history_key)

        with self._stateful_response_ids_lock:
            self._stateful_response_ids[cache_key] = response.response_id
            self._stateful_response_ids.move_to_end(cache_key)
            while len(self._stateful_response_ids) > _STATEFUL_HISTORY_CACHE_LIMIT:
                self._stateful_response_ids.popitem(last=False)

    def _forget_response_id(self, response_id: str | None) -> None:
        if response_id is None:
            return

        with self._stateful_response_ids_lock:
            stale_keys = [
                cache_key
                for cache_key, cached_response_id in self._stateful_response_ids.items()
                if cached_response_id == response_id
            ]
            for cache_key in stale_keys:
                self._stateful_response_ids.pop(cache_key, None)

    def _response_output_items(self, response: LLMResponse) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if response.text:
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": response.text,
                        }
                    ],
                }
            )
        for tool_call in response.tool_calls:
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_call.call_id,
                    "name": tool_call.name,
                    "arguments": tool_call.raw_arguments,
                }
            )
        return items

    def _serialize_history_item(self, item: dict[str, Any]) -> str:
        return json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def _decode_stream_event(self, sse_payload: str) -> dict[str, Any]:
        try:
            chunk = json.loads(sse_payload)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("LM Studio returned malformed streaming JSON.") from exc
        if not isinstance(chunk, dict):
            raise ProviderResponseError("LM Studio returned a non-object streaming chunk.")
        return chunk

    def _extract_response_failed_message(self, event: dict[str, Any]) -> str:
        response = event.get("response")
        if isinstance(response, dict):
            return self._extract_stream_error_message(response.get("error"))
        return "LM Studio streaming response failed."

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

    def _normalize_function_call_arguments_chunk(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value)

    def _should_retry_without_previous_response_id(
        self,
        exc: ProviderBadRequestError,
        *,
        previous_response_id: str | None,
    ) -> bool:
        if previous_response_id is None:
            return False

        message = str(exc).lower()
        mentions_response_pointer = any(
            hint in message
            for hint in (
                "previous_response_id",
                "previous response id",
                "response_id",
                "response id",
            )
        )
        if not mentions_response_pointer:
            return False
        return any(
            hint in message
            for hint in (
                "not found",
                "unknown",
                "missing",
                "invalid",
                "expired",
                "no longer available",
            )
        )

    def _extract_incomplete_reason(self, incomplete_details: Any) -> str | None:
        if not isinstance(incomplete_details, dict):
            return None
        return self._normalize_optional_string(incomplete_details.get("reason"))

    def _normalize_optional_string(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _coerce_optional_int(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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
