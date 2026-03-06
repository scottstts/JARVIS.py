"""Anthropic provider adapter for normalized LLM requests/responses."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

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
)

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
    LLMStreamEvent,
    LLMUsage,
    TextDeltaEvent,
    TextPart,
    ToolCall,
    ToolChoice,
    ToolChoiceMode,
    ToolDefinition,
    ToolResultPart,
    UsageDeltaEvent,
)
from ..validation import build_tool_schema_map, parse_and_validate_tool_call

_ANTHROPIC_ADAPTIVE_THINKING_MODEL_MARKERS = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)
_ANTHROPIC_EFFORT_MODEL_MARKERS = (
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)
_ANTHROPIC_MAX_EFFORT_MODEL_MARKERS = ("claude-opus-4-6",)


class AnthropicProvider:
    """Provider implementation for Anthropic Messages API."""

    def __init__(
        self,
        *,
        settings: AnthropicProviderSettings,
        default_timeout_seconds: float,
    ) -> None:
        self._settings = settings
        self._default_timeout_seconds = default_timeout_seconds
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
            image_input=False,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        kwargs = self._build_messages_create_kwargs(request)
        client = await self._client_instance()
        try:
            response = await client.messages.create(**kwargs)
        except Exception as exc:
            raise self._map_error(exc) from exc
        return self._normalize_message_response(request=request, response=response)

    async def stream_generate(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        response = await self.generate(request)
        if response.text:
            yield TextDeltaEvent(delta=response.text)
        if response.usage is not None:
            yield UsageDeltaEvent(usage=response.usage)
        yield DoneEvent(response=response)

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
                timeout=self._default_timeout_seconds,
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

        system_prompt, messages = self._to_anthropic_messages(request.messages)
        if not messages:
            raise LLMConfigurationError("Anthropic requests require at least one user/assistant message.")

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
        }

        if system_prompt is not None:
            kwargs["system"] = system_prompt
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.timeout_seconds is not None:
            kwargs["timeout"] = request.timeout_seconds
        if request.tools:
            kwargs["tools"] = [self._to_anthropic_tool(tool) for tool in request.tools]
            kwargs["tool_choice"] = self._to_anthropic_tool_choice(request.tool_choice)
        elif request.tool_choice.mode not in {ToolChoiceMode.AUTO, ToolChoiceMode.NONE}:
            raise LLMConfigurationError(
                "Specific tool-choice mode requires non-empty request.tools."
            )

        thinking = self._to_anthropic_thinking(model=request.model)
        if thinking is not None:
            kwargs["thinking"] = thinking

        output_config = self._to_anthropic_output_config(model=request.model)
        if output_config is not None:
            kwargs["output_config"] = output_config

        return kwargs

    def _to_anthropic_messages(
        self,
        messages: Sequence[LLMMessage],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        out_messages: list[dict[str, Any]] = []
        pending_tool_results: list[dict[str, Any]] = []

        for message in messages:
            if message.role in {"system", "developer"}:
                text = _join_text_parts(
                    message.parts,
                    unsupported_message=(
                        "System/developer Anthropic history only supports text parts."
                    ),
                )
                if text:
                    system_parts.append(text)
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

        system_prompt = "\n\n".join(system_parts).strip() or None
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
                raise LLMConfigurationError(
                    "Anthropic provider does not support image input in this layer yet."
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
            "input_schema": dict(tool.input_schema),
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
        supports_adaptive = self._supports_adaptive_thinking(model)

        if thinking_mode == "disabled":
            return None

        if thinking_mode == "adaptive":
            if supports_adaptive:
                return {"type": "adaptive"}
            if budget_tokens is not None:
                return {"type": "enabled", "budget_tokens": budget_tokens}
            return None

        if thinking_mode == "enabled":
            if budget_tokens is not None:
                return {"type": "enabled", "budget_tokens": budget_tokens}
            if supports_adaptive:
                return {"type": "adaptive"}
            return None

        if thinking_mode is None:
            if budget_tokens is not None:
                return {"type": "enabled", "budget_tokens": budget_tokens}
            return None

        payload: dict[str, Any] = {"type": thinking_mode}
        if budget_tokens is not None:
            payload["budget_tokens"] = budget_tokens
        return payload

    def _to_anthropic_output_config(self, *, model: str) -> dict[str, Any] | None:
        effort = self._settings.effort
        if effort is None:
            return None
        if not self._supports_effort(model):
            return None
        if effort == "max" and not self._supports_max_effort(model):
            return None
        return {"effort": effort}

    def _supports_adaptive_thinking(self, model: str) -> bool:
        normalized_model = model.lower()
        return any(
            marker in normalized_model
            for marker in _ANTHROPIC_ADAPTIVE_THINKING_MODEL_MARKERS
        )

    def _supports_effort(self, model: str) -> bool:
        normalized_model = model.lower()
        return any(marker in normalized_model for marker in _ANTHROPIC_EFFORT_MODEL_MARKERS)

    def _supports_max_effort(self, model: str) -> bool:
        normalized_model = model.lower()
        return any(
            marker in normalized_model
            for marker in _ANTHROPIC_MAX_EFFORT_MODEL_MARKERS
        )

    def _normalize_message_response(self, *, request: LLMRequest, response: Any) -> LLMResponse:
        tool_calls = self._extract_tool_calls(
            content_blocks=response.content,
            request_tools=request.tools,
        )
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        usage = LLMUsage(
            input_tokens=getattr(response.usage, "input_tokens", None),
            output_tokens=getattr(response.usage, "output_tokens", None),
            total_tokens=(
                (getattr(response.usage, "input_tokens", 0) or 0)
                + (getattr(response.usage, "output_tokens", 0) or 0)
            ),
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

        return LLMResponse(
            provider=self.name,
            model=response.model,
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            response_id=response.id,
            provider_metadata={"stop_reason": stop_reason},
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
                parse_and_validate_tool_call(
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
        if isinstance(error, AnthropicError):
            return ProviderResponseError(str(error))
        return ProviderResponseError(str(error))


def _join_text_parts(parts: Sequence[Any], *, unsupported_message: str) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            text_parts.append(part.text)
            continue
        raise LLMConfigurationError(unsupported_message)
    return "\n".join(text_parts).strip()
