"""Gemini provider adapter using the google-genai SDK."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors

from ..config import GeminiProviderSettings
from ..errors import (
    LLMConfigurationError,
    ProviderAuthenticationError,
    ProviderBadRequestError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTemporaryError,
    ProviderTimeoutError,
)
from ..protocols import ProviderCapabilities
from ..types import (
    DoneEvent,
    EmbeddingRequest,
    EmbeddingResponse,
    ImagePart,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
    LLMUsage,
    TextDeltaEvent,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    UsageDeltaEvent,
)
from ..validation import build_tool_schema_map, parse_and_validate_tool_call

_GEMINI_3_MODEL_PREFIX = "gemini-3"
_GEMINI_25_MODEL_PREFIX = "gemini-2.5"


class GeminiProvider:
    """Provider implementation for Google Gemini via google-genai SDK."""

    def __init__(
        self,
        *,
        settings: GeminiProviderSettings,
        default_timeout_seconds: float,
    ) -> None:
        self._settings = settings
        self._default_timeout_seconds = default_timeout_seconds
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
            image_input=False,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if request.model is None:
            raise LLMConfigurationError("request.model must be set before provider dispatch.")

        client = await self._client_instance()
        contents, config = self._build_generate_payload(request)

        try:
            response = await client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=config or None,
            )
        except Exception as exc:
            raise self._map_error(exc) from exc

        return self._normalize_generate_response(request=request, response=response)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        response = await self.generate(request)
        if response.text:
            yield TextDeltaEvent(delta=response.text)
        if response.usage is not None:
            yield UsageDeltaEvent(usage=response.usage)
        yield DoneEvent(response=response)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if request.model is None:
            raise LLMConfigurationError(
                "No embedding model configured for provider 'gemini'."
            )

        client = await self._client_instance()
        contents: Any = request.inputs if isinstance(request.inputs, str) else list(request.inputs)
        config: dict[str, Any] = {}
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

            self._client = genai.Client(api_key=api_key)
            return self._client

    def _build_generate_payload(self, request: LLMRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        system_parts: list[str] = []

        for message in request.messages:
            text_parts: list[str] = []
            for part in message.parts:
                if isinstance(part, ImagePart):
                    raise LLMConfigurationError("Gemini provider does not support image input in this layer yet.")
                if isinstance(part, TextPart):
                    text_parts.append(part.text)

            text = "\n".join(text_parts).strip()
            if not text:
                continue

            if message.role in {"system", "developer"}:
                system_parts.append(text)
                continue

            role = "model" if message.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": text}]})

        config: dict[str, Any] = {}
        system_instruction = "\n\n".join(system_parts).strip()
        if system_instruction:
            config["system_instruction"] = system_instruction
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            config["max_output_tokens"] = request.max_output_tokens

        thinking_config = self._to_gemini_thinking_config(model=request.model)
        if thinking_config:
            config["thinking_config"] = thinking_config

        if request.tools:
            declarations = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": dict(tool.input_schema),
                }
                for tool in request.tools
            ]
            config["tools"] = [{"function_declarations": declarations}]
            config["tool_config"] = self._to_gemini_tool_config(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        return contents, config

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

    def _normalize_generate_response(self, *, request: LLMRequest, response: Any) -> LLMResponse:
        tool_calls = self._extract_tool_calls(
            candidates=response.candidates or [],
            request_tools=request.tools,
        )

        usage_metadata = getattr(response, "usage_metadata", None)
        usage = None
        if usage_metadata is not None:
            usage = LLMUsage(
                input_tokens=getattr(usage_metadata, "prompt_token_count", None),
                output_tokens=getattr(usage_metadata, "candidates_token_count", None),
                total_tokens=getattr(usage_metadata, "total_token_count", None),
            )

        finish_reason = "unknown"
        first_candidate = (response.candidates or [None])[0]
        candidate_finish = None
        if first_candidate is not None:
            finish_reason_attr = getattr(first_candidate, "finish_reason", None)
            candidate_finish = (
                finish_reason_attr.value
                if hasattr(finish_reason_attr, "value")
                else str(finish_reason_attr) if finish_reason_attr is not None else None
            )

        if tool_calls:
            finish_reason = "tool_calls"
        elif candidate_finish == "MAX_TOKENS":
            finish_reason = "length"
        elif candidate_finish in {
            "SAFETY",
            "PROHIBITED_CONTENT",
            "BLOCKLIST",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
        }:
            finish_reason = "content_filter"
        elif candidate_finish == "STOP":
            finish_reason = "stop"

        return LLMResponse(
            provider=self.name,
            model=request.model or "",
            text=response.text or "",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            response_id=getattr(response, "response_id", None),
            provider_metadata={
                "model_version": getattr(response, "model_version", None),
                "finish_reason": candidate_finish,
            },
        )

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
                tool_calls.append(
                    parse_and_validate_tool_call(
                        call_id=call_id,
                        name=function_call.name,
                        raw_arguments=raw_arguments,
                        tool_schemas=tool_schemas,
                    )
                )

        return tool_calls

    def _map_error(self, error: Exception) -> Exception:
        if isinstance(error, (httpx.ReadTimeout, httpx.ConnectTimeout, TimeoutError)):
            return ProviderTimeoutError(str(error))
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
