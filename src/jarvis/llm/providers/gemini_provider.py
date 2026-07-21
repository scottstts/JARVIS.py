"""Gemini provider adapter using the google-genai SDK."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any, cast

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from ..config import GeminiProviderSettings
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

_GEMINI_3_MODEL_PREFIX = "gemini-3"
_GEMINI_25_MODEL_PREFIX = "gemini-2.5"
_GEMINI_CACHED_CONTENT_TTL = "3600s"


class GeminiProvider:
    """Provider implementation for Google Gemini via google-genai SDK."""

    def __init__(
        self,
        *,
        settings: GeminiProviderSettings,
        read_timeout_seconds: float,
        connect_timeout_seconds: float = 30.0,
    ) -> None:
        self._settings = settings
        self._transport_timeouts = ProviderTransportTimeouts(
            connect_seconds=connect_timeout_seconds,
            read_seconds=read_timeout_seconds,
        )
        self._client: genai.Client | None = None
        self._client_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "gemini"

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

        client = await self._client_instance()
        contents, config = self._build_generate_payload(request)
        cache_metadata = await self._ensure_cached_content(
            client=client,
            request=request,
            config=config,
        )

        try:
            response = await client.aio.models.generate_content(
                model=request.model,
                contents=cast(Any, contents),
                config=config or None,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        return self._normalize_generate_response(
            request=request,
            response=response,
            cache_metadata=cache_metadata,
        )

    async def stream_generate(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[ProviderStreamEvent]:
        if request.model is None:
            raise LLMConfigurationError("request.model must be set before provider dispatch.")

        client = await self._client_instance()
        contents, config = self._build_generate_payload(request)
        cache_metadata = await self._ensure_cached_content(
            client=client,
            request=request,
            config=config,
        )

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        seen_tool_calls: set[tuple[str, str, str]] = set()
        thought_signatures_b64: list[str] = []
        seen_thought_signatures: set[str] = set()
        usage: LLMUsage | None = None
        response_id: str | None = None
        model_version: str | None = None
        candidate_finish: str | None = None
        saw_chunk = False

        try:
            stream = await client.aio.models.generate_content_stream(
                model=request.model,
                contents=cast(Any, contents),
                config=config or None,
            )
            async for chunk in stream:
                saw_chunk = True
                chunk_response_id = _optional_str(getattr(chunk, "response_id", None))
                if chunk_response_id is not None:
                    response_id = chunk_response_id
                chunk_model_version = _optional_str(getattr(chunk, "model_version", None))
                if chunk_model_version is not None:
                    model_version = chunk_model_version

                chunk_usage = self._normalize_usage(getattr(chunk, "usage_metadata", None))
                if chunk_usage is not None:
                    usage = chunk_usage
                chunk_finish = self._candidate_finish_reason(chunk)
                if chunk_finish is not None:
                    candidate_finish = chunk_finish

                emitted_semantic_event = False
                chunk_text = self._extract_text_response(chunk)
                if chunk_text:
                    text_parts.append(chunk_text)
                    emitted_semantic_event = True
                    yield TextDeltaEvent(delta=chunk_text)

                for tool_call in self._extract_tool_calls(
                    candidates=getattr(chunk, "candidates", None) or [],
                    request_tools=request.tools,
                ):
                    identity = (
                        tool_call.call_id,
                        tool_call.name,
                        tool_call.raw_arguments,
                    )
                    if identity in seen_tool_calls:
                        continue
                    seen_tool_calls.add(identity)
                    tool_calls.append(tool_call)
                    emitted_semantic_event = True
                    yield ToolCallDeltaEvent(
                        call_id=tool_call.call_id,
                        tool_name=tool_call.name,
                        arguments_delta=tool_call.raw_arguments,
                    )

                for signature in self._extract_thought_signatures_b64(chunk):
                    if signature in seen_thought_signatures:
                        continue
                    seen_thought_signatures.add(signature)
                    thought_signatures_b64.append(signature)

                if not emitted_semantic_event:
                    yield ProviderActivityEvent(
                        provider_event_type="generate_content.chunk",
                        response_id=response_id,
                    )
        except Exception as exc:
            raise self._map_error(exc) from exc

        if not saw_chunk:
            raise StreamProtocolError("Gemini stream closed without any response chunks.")

        provider_metadata: dict[str, Any] = {
            "model_version": model_version,
            "finish_reason": candidate_finish,
            **cache_metadata,
        }
        if thought_signatures_b64:
            provider_metadata["thought_signatures_b64"] = thought_signatures_b64

        normalized = LLMResponse(
            provider=self.name,
            model=request.model,
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=self._normalize_finish_reason(
                tool_calls=tool_calls,
                candidate_finish=candidate_finish,
            ),
            usage=usage,
            response_id=response_id,
            provider_metadata=provider_metadata,
        )
        if usage is not None:
            yield UsageDeltaEvent(usage=usage)
        yield DoneEvent(response=normalized)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.model is None:
            raise LLMConfigurationError(
                "No embedding model configured for provider 'gemini'."
            )

        client = await self._client_instance()
        contents: Any = request.inputs if isinstance(request.inputs, str) else list(request.inputs)
        config: genai_types.EmbedContentConfigDict = {}
        if request.dimensions is not None:
            config["output_dimensionality"] = request.dimensions

        try:
            response = await client.aio.models.embed_content(
                model=request.model,
                contents=contents,
                config=config or None,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        embeddings = [
            list(item.values or [])
            for item in (response.embeddings or [])
        ]
        return EmbeddingResponse(
            provider=self.name,
            model=request.model,
            embeddings=embeddings,
            usage=None,
        )

    async def aclose(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is None:
            return

        async_client = getattr(client, "aio", None)
        if async_client is not None:
            aclose = getattr(async_client, "aclose", None)
            if aclose is not None:
                maybe = aclose()
                if inspect.isawaitable(maybe):
                    await maybe

        close = getattr(client, "close", None)
        if close is not None:
            maybe = close()
            if inspect.isawaitable(maybe):
                await maybe

    async def _client_instance(self) -> genai.Client:
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._client is not None:
                return self._client

            api_key = self._settings.api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise LLMConfigurationError(
                    "GOOGLE_API_KEY is required for the Gemini provider."
                )

            timeout_extension = {
                "connect": self._transport_timeouts.connect_seconds,
                "read": self._transport_timeouts.read_seconds,
                "write": self._transport_timeouts.connect_seconds,
                "pool": self._transport_timeouts.connect_seconds,
            }

            async def apply_transport_timeouts(request: httpx.Request) -> None:
                request.extensions["timeout"] = dict(timeout_extension)

            async_http_client = httpx.AsyncClient(
                timeout=self._transport_timeouts.as_httpx(),
                event_hooks={"request": [apply_transport_timeouts]},
            )
            try:
                self._client = genai.Client(
                    api_key=api_key,
                    http_options=genai_types.HttpOptions(
                        timeout=int(self._transport_timeouts.read_seconds * 1000),
                        httpx_async_client=async_http_client,
                    ),
                )
            except Exception:
                await async_http_client.aclose()
                raise
            return self._client

    def _build_generate_payload(self, request: LLMRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        uses_cached_context = bool(
            request.cached_content_name or request.cached_content_messages
        )
        contents, system_instruction = self._to_gemini_contents_and_system_instruction(
            request.messages,
            system_messages_as_content=uses_cached_context,
        )

        config: dict[str, Any] = {}
        cached_content_name = self._usable_cached_content_name(request)
        if cached_content_name is not None:
            config["cached_content"] = cached_content_name
        elif system_instruction:
            config["system_instruction"] = system_instruction
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            config["max_output_tokens"] = request.max_output_tokens

        thinking_config = self._to_gemini_thinking_config(model=request.model)
        if thinking_config:
            config["thinking_config"] = thinking_config

        if request.tools and not uses_cached_context:
            config["tools"] = self._to_gemini_tools(request.tools)
            config["tool_config"] = self._to_gemini_tool_config(request.tool_choice)
        elif not request.tools and request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        return contents, config

    async def _ensure_cached_content(
        self,
        *,
        client: genai.Client,
        request: LLMRequest,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        existing_name = config.get("cached_content")
        if isinstance(existing_name, str) and existing_name.strip():
            return {
                "cached_content_name": existing_name.strip(),
                "cached_content_model": request.cached_content_model,
                "cached_content_source_signature": request.cached_content_source_signature,
                "cached_content_source_record_ids": list(
                    request.cached_content_source_record_ids
                ),
                "cached_content_media_ids": list(request.cached_content_media_ids),
            }

        if not request.cached_content_messages:
            return {}

        if request.model is None:
            raise LLMConfigurationError("request.model must be set before creating Gemini cached content.")

        cache_config = self._build_cached_content_config(request)
        try:
            cache = await client.aio.caches.create(
                model=request.model,
                config=cache_config,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        cache_name = _optional_str(getattr(cache, "name", None))
        if cache_name is None:
            raise ProviderResponseError("Gemini cached content creation returned no cache name.")

        config["cached_content"] = cache_name
        return {
            "cached_content_name": cache_name,
            "cached_content_model": _optional_str(getattr(cache, "model", None)) or request.model,
            "cached_content_expires_at": _datetime_to_iso(getattr(cache, "expire_time", None)),
            "cached_content_source_signature": request.cached_content_source_signature,
            "cached_content_source_record_ids": list(request.cached_content_source_record_ids),
            "cached_content_media_ids": list(request.cached_content_media_ids),
        }

    def _build_cached_content_config(self, request: LLMRequest) -> dict[str, Any]:
        contents, system_instruction = self._to_gemini_contents_and_system_instruction(
            request.cached_content_messages,
            system_messages_as_content=False,
        )
        config: dict[str, Any] = {
            "ttl": _GEMINI_CACHED_CONTENT_TTL,
        }
        if contents:
            config["contents"] = contents
        if system_instruction:
            config["system_instruction"] = system_instruction
        if request.tools:
            config["tools"] = self._to_gemini_tools(request.tools)
            config["tool_config"] = self._to_gemini_tool_config(request.tool_choice)
        return config

    def _to_gemini_contents_and_system_instruction(
        self,
        messages: Sequence[Any],
        *,
        system_messages_as_content: bool,
    ) -> tuple[list[dict[str, Any]], str]:
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []
        pending_function_responses: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                text = _join_text_parts(
                    message.parts,
                    unsupported_message="Gemini system history only supports text parts.",
                )
                if not text:
                    continue
                if system_messages_as_content:
                    if pending_function_responses:
                        contents.append({"role": "user", "parts": pending_function_responses})
                        pending_function_responses = []
                    contents.append({"role": "user", "parts": [{"text": text}]})
                else:
                    system_parts.append(text)
                continue

            if message.role == "tool":
                pending_function_responses.extend(
                    self._to_gemini_function_response_parts(message)
                )
                continue

            if pending_function_responses:
                contents.append({"role": "user", "parts": pending_function_responses})
                pending_function_responses = []

            parts = self._to_gemini_content_parts(message)
            if not parts:
                continue
            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": parts})

        if pending_function_responses:
            contents.append({"role": "user", "parts": pending_function_responses})

        return contents, "\n\n".join(system_parts).strip()

    def _to_gemini_tools(self, tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
        declarations = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_json_schema": dict(tool.input_schema),
            }
            for tool in tools
        ]
        return [{"function_declarations": declarations}]

    def _usable_cached_content_name(self, request: LLMRequest) -> str | None:
        name = _optional_str(request.cached_content_name)
        if name is None:
            return None
        cached_model = _optional_str(request.cached_content_model)
        request_model = _optional_str(request.model)
        if cached_model is not None and request_model is not None and cached_model != request_model:
            return None
        return name

    def _to_gemini_thinking_config(self, *, model: str) -> dict[str, Any] | None:
        thinking_level = self._settings.thinking_level
        thinking_budget = self._settings.thinking_budget
        normalized_model = model.lower()

        config: dict[str, Any] = {}
        if normalized_model.startswith(_GEMINI_3_MODEL_PREFIX):
            if thinking_level is not None:
                config["thinking_level"] = thinking_level
            return config or None

        if normalized_model.startswith(_GEMINI_25_MODEL_PREFIX):
            if thinking_budget is not None:
                config["thinking_budget"] = thinking_budget
            return config or None

        if thinking_level is not None:
            config["thinking_level"] = thinking_level
        elif thinking_budget is not None:
            config["thinking_budget"] = thinking_budget
        return config or None

    def _to_gemini_tool_config(self, tool_choice: ToolChoice) -> dict[str, Any]:
        if tool_choice.mode == ToolChoiceMode.AUTO:
            mode = "AUTO"
            allowed: list[str] | None = None
        elif tool_choice.mode == ToolChoiceMode.REQUIRED:
            mode = "ANY"
            allowed = None
        elif tool_choice.mode == ToolChoiceMode.NONE:
            mode = "NONE"
            allowed = None
        else:
            mode = "ANY"
            allowed = [tool_choice.tool_name]

        function_calling_config: dict[str, Any] = {"mode": mode}
        if allowed:
            function_calling_config["allowed_function_names"] = allowed
        return {"function_calling_config": function_calling_config}

    def _to_gemini_content_parts(self, message: Any) -> list[Any]:
        parts: list[Any] = []
        for part in message.parts:
            if isinstance(part, TextPart):
                parts.append({"text": part.text})
            elif isinstance(part, ToolCall):
                if message.role != "assistant":
                    raise LLMConfigurationError(
                        "Tool call history can only appear on assistant messages."
                    )
                function_call_part: dict[str, Any] = {
                    "function_call": {
                        "name": part.name,
                        "args": part.arguments,
                    }
                }
                thought_signature = _decode_gemini_thought_signature(
                    part.provider_metadata
                )
                if thought_signature is not None:
                    function_call_part["thought_signature"] = thought_signature
                parts.append(function_call_part)
            elif isinstance(part, ImagePart):
                parts.append(_to_gemini_image_part(part))
            else:
                raise LLMConfigurationError(
                    f"Unsupported Gemini message part type: {type(part).__name__}."
                )
        return parts

    def _to_gemini_function_response_parts(self, message: Any) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        for part in message.parts:
            if not isinstance(part, ToolResultPart):
                raise LLMConfigurationError(
                    "Tool-role messages can only contain tool results for Gemini."
                )
            parts.append(
                {
                    "function_response": {
                        "name": part.name,
                        "response": {
                            "ok": not part.is_error,
                            "content": part.content,
                        },
                    }
                }
            )
        return parts

    def _normalize_generate_response(
        self,
        *,
        request: LLMRequest,
        response: Any,
        cache_metadata: dict[str, Any] | None = None,
    ) -> LLMResponse:
        tool_calls = self._extract_tool_calls(
            candidates=response.candidates or [],
            request_tools=request.tools,
        )

        usage = self._normalize_usage(getattr(response, "usage_metadata", None))
        candidate_finish = self._candidate_finish_reason(response)

        provider_metadata: dict[str, Any] = {
            "model_version": getattr(response, "model_version", None),
            "finish_reason": candidate_finish,
            **dict(cache_metadata or {}),
        }
        thought_signatures_b64 = self._extract_thought_signatures_b64(response)
        if thought_signatures_b64:
            provider_metadata["thought_signatures_b64"] = thought_signatures_b64

        return LLMResponse(
            provider=self.name,
            model=request.model or "",
            text=self._extract_text_response(response),
            tool_calls=tool_calls,
            finish_reason=self._normalize_finish_reason(
                tool_calls=tool_calls,
                candidate_finish=candidate_finish,
            ),
            usage=usage,
            response_id=getattr(response, "response_id", None),
            provider_metadata=provider_metadata,
        )

    def _normalize_usage(self, usage_metadata: Any) -> LLMUsage | None:
        if usage_metadata is None:
            return None
        return LLMUsage(
            input_tokens=getattr(usage_metadata, "prompt_token_count", None),
            output_tokens=getattr(usage_metadata, "candidates_token_count", None),
            total_tokens=getattr(usage_metadata, "total_token_count", None),
        )

    def _candidate_finish_reason(self, response: Any) -> str | None:
        first_candidate = (getattr(response, "candidates", None) or [None])[0]
        if first_candidate is None:
            return None
        finish_reason = getattr(first_candidate, "finish_reason", None)
        if finish_reason is None:
            return None
        value = getattr(finish_reason, "value", finish_reason)
        return str(value)

    def _normalize_finish_reason(
        self,
        *,
        tool_calls: Sequence[ToolCall],
        candidate_finish: str | None,
    ) -> FinishReason:
        if tool_calls:
            return "tool_calls"
        if candidate_finish == "MAX_TOKENS":
            return "length"
        if candidate_finish in {
            "SAFETY",
            "PROHIBITED_CONTENT",
            "BLOCKLIST",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
        }:
            return "content_filter"
        if candidate_finish == "STOP":
            return "stop"
        return "unknown"

    def _extract_text_response(self, response: Any) -> str:
        text_parts: list[str] = []
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                if getattr(part, "thought", False):
                    continue
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    text_parts.append(text)
        return "".join(text_parts)

    def _extract_thought_signatures_b64(self, response: Any) -> list[str]:
        signatures: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                signature = _extract_gemini_thought_signature(part)
                if signature is not None:
                    signatures.append(base64.b64encode(signature).decode("ascii"))
        return signatures

    def _extract_tool_calls(
        self,
        *,
        candidates: Sequence[Any],
        request_tools: Sequence[ToolDefinition],
    ) -> list[ToolCall]:
        tool_schemas = build_tool_schema_map(request_tools)
        tool_calls: list[ToolCall] = []

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                function_call = getattr(part, "function_call", None)
                if function_call is None or function_call.name is None:
                    continue

                call_id = getattr(function_call, "id", None) or function_call.name
                arguments = function_call.args or {}
                raw_arguments = json.dumps(arguments)
                thought_signature = _extract_gemini_thought_signature(part)
                provider_metadata = (
                    {
                        "thought_signature_b64": base64.b64encode(thought_signature).decode(
                            "ascii"
                        )
                    }
                    if thought_signature is not None
                    else {}
                )
                tool_calls.append(
                    parse_and_validate_tool_call_or_recover(
                        call_id=call_id,
                        name=function_call.name,
                        raw_arguments=raw_arguments,
                        tool_schemas=tool_schemas,
                        provider_metadata=provider_metadata,
                    )
                )

        return tool_calls

    def _map_error(self, error: Exception) -> Exception:
        if isinstance(error, (httpx.ReadTimeout, httpx.ConnectTimeout, TimeoutError)):
            return ProviderTimeoutError(
                str(error),
                metadata=transport_timeout_metadata(
                    error,
                    timeouts=self._transport_timeouts,
                ),
            )
        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            return ProviderTemporaryError(str(error))
        if isinstance(error, genai_errors.ServerError):
            return ProviderTemporaryError(str(error))
        if isinstance(error, genai_errors.ClientError):
            status_code = getattr(error, "code", None)
            if status_code == 429:
                return ProviderRateLimitError(str(error))
            if status_code in {401, 403}:
                return ProviderAuthenticationError(str(error))
            return ProviderBadRequestError(str(error))
        if isinstance(error, genai_errors.APIError):
            status_code = getattr(error, "code", None)
            if status_code == 429:
                return ProviderRateLimitError(str(error))
            if status_code in {401, 403}:
                return ProviderAuthenticationError(str(error))
            if isinstance(status_code, int) and status_code >= 500:
                return ProviderTemporaryError(str(error))
            return ProviderBadRequestError(str(error))
        return ProviderResponseError(str(error))


def _extract_gemini_thought_signature(part: Any) -> bytes | None:
    signature = getattr(part, "thought_signature", None)
    if signature is None:
        signature = getattr(part, "thoughtSignature", None)
    if isinstance(signature, bytes) and signature:
        return signature
    return None


def _decode_gemini_thought_signature(provider_metadata: dict[str, Any]) -> bytes | None:
    encoded = provider_metadata.get("thought_signature_b64")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        return base64.b64decode(encoded)
    except (ValueError, TypeError):
        return None


def _to_gemini_image_part(part: ImagePart) -> genai_types.Part:
    data_url_payload = part.data_url_payload()
    if data_url_payload is None:
        raise LLMConfigurationError(
            "Gemini image input in this layer requires a base64 data URL image."
        )
    media_type, data_base64 = data_url_payload
    try:
        data = base64.b64decode(data_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise LLMConfigurationError(
            "Gemini image input received invalid base64 image data."
        ) from exc
    return genai_types.Part.from_bytes(data=data, mime_type=media_type)


def _join_text_parts(parts: Sequence[Any], *, unsupported_message: str) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            text_parts.append(part.text)
            continue
        raise LLMConfigurationError(unsupported_message)
    return "\n".join(text_parts).strip()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _datetime_to_iso(value: Any) -> str | None:
    isoformat = getattr(value, "isoformat", None)
    if not callable(isoformat):
        return _optional_str(value)
    try:
        return _optional_str(isoformat())
    except TypeError:
        return _optional_str(value)
