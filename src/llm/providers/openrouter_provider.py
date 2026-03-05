"""OpenRouter provider adapter using direct HTTP requests."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator, Sequence
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
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    UsageDeltaEvent,
)
from ..validation import build_tool_schema_map, parse_and_validate_tool_call


class OpenRouterProvider:
    """Provider implementation for OpenRouter OpenAI-compatible HTTP APIs."""

    def __init__(
        self,
        *,
        settings: OpenRouterProviderSettings,
        default_timeout_seconds: float,
    ) -> None:
        self._settings = settings
        self._default_timeout_seconds = default_timeout_seconds

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

        payload = self._build_chat_payload(request)
        data = await self._post_json(
            endpoint="/chat/completions",
            payload=payload,
            timeout_seconds=request.timeout_seconds,
        )
        return self._normalize_chat_response(request=request, response_json=data)

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
            timeout_seconds=request.timeout_seconds,
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

    def _build_chat_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [self._to_openrouter_message(message) for message in request.messages],
            "stream": False,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens

        if request.tools:
            payload["tools"] = [self._to_openrouter_tool(tool) for tool in request.tools]
            payload["tool_choice"] = self._to_openrouter_tool_choice(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        return payload

    def _to_openrouter_message(self, message: LLMMessage) -> dict[str, Any]:
        role = "system" if message.role in {"system", "developer"} else message.role
        has_non_text = any(not isinstance(part, TextPart) for part in message.parts)

        if not has_non_text and len(message.parts) == 1:
            return {"role": role, "content": message.parts[0].text}

        content: list[dict[str, Any]] = []
        for part in message.parts:
            if isinstance(part, TextPart):
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

        return {"role": role, "content": content}

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
        timeout_seconds: float | None,
    ) -> dict[str, Any]:
        api_key = self._settings.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY is required for the OpenRouter provider.")

        timeout = timeout_seconds if timeout_seconds is not None else self._default_timeout_seconds
        url = f"{self._settings.base_url.rstrip('/')}{endpoint}"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self._settings.site_url:
            headers["HTTP-Referer"] = self._settings.site_url
        if self._settings.app_name:
            headers["X-Title"] = self._settings.app_name

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
            return response.json()
        except ValueError as exc:
            raise ProviderResponseError("OpenRouter returned non-JSON response.") from exc

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

    def _normalize_chat_response(self, *, request: LLMRequest, response_json: dict[str, Any]) -> LLMResponse:
        choices = response_json.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {})

        text = self._extract_text(message.get("content"))
        tool_calls = self._extract_tool_calls(
            message_tool_calls=message.get("tool_calls", []) or [],
            request_tools=request.tools,
        )

        finish_reason = choice.get("finish_reason", "unknown")
        mapped_finish_reason = {
            "stop": "stop",
            "tool_calls": "tool_calls",
            "length": "length",
            "content_filter": "content_filter",
        }.get(finish_reason, "unknown")

        usage_obj = response_json.get("usage", {})
        usage = None
        if usage_obj:
            usage = LLMUsage(
                input_tokens=usage_obj.get("prompt_tokens"),
                output_tokens=usage_obj.get("completion_tokens"),
                total_tokens=usage_obj.get("total_tokens"),
            )

        return LLMResponse(
            provider=self.name,
            model=response_json.get("model", request.model or ""),
            text=text,
            tool_calls=tool_calls,
            finish_reason=mapped_finish_reason,
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
