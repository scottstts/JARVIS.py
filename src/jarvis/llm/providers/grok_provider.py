"""Grok provider adapter using xAI's OpenAI-compatible Responses API."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, replace
import inspect
import json
import os
from pathlib import Path
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

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
    FinishReason,
    ImagePart,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    LocalImagePart,
    ProviderActivityEvent,
    ProviderStreamEvent,
    StatefulContinuation,
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

_GROK_WEBSOCKET_MAX_AGE_SECONDS = 23 * 60
_GROK_WEBSOCKET_POOL_LIMIT = 32
_RESPONSE_STORAGE_OVERFLOW_HINT = "response is too large to store"


class GrokResponseStorageOverflowError(ProviderResponseError):
    """xAI could generate the response but refused its durable representation."""


class GrokWebSocketContinuationError(ProviderResponseError):
    """The requested xAI in-memory continuation is unavailable on this socket."""


@dataclass(slots=True)
class _GrokWebSocketSession:
    session_key: str
    websocket: Any
    lock: asyncio.Lock
    opened_at: float
    last_used_at: float
    generation: int
    live_response_id: str | None = None


class GrokProvider:
    """Provider implementation for xAI Grok Responses."""

    def __init__(
        self,
        *,
        settings: GrokProviderSettings,
        read_timeout_seconds: float,
        connect_timeout_seconds: float = 30.0,
    ) -> None:
        self._settings = settings
        self._transport_timeouts = ProviderTransportTimeouts(
            connect_seconds=connect_timeout_seconds,
            read_seconds=read_timeout_seconds,
        )
        self._client: AsyncOpenAI | None = None
        self._client_lock = asyncio.Lock()
        self._websocket_sessions: dict[str, _GrokWebSocketSession] = {}
        self._websocket_sessions_lock = asyncio.Lock()

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
        if _request_contains_image(request) and not _request_uses_ephemeral_storage(request):
            request = _request_with_ephemeral_fallback(request)
        if _request_uses_ephemeral_storage(request):
            return await self._generate_ephemeral(request)

        kwargs = self._build_response_create_kwargs(request, stream=False, store=True)
        client = await self._client_instance()
        try:
            response = await client.responses.create(**kwargs)
        except Exception as exc:
            mapped = self._map_error(exc)
            if isinstance(mapped, GrokResponseStorageOverflowError):
                return await self._generate_ephemeral(_request_with_ephemeral_fallback(request))
            raise mapped from exc
        return self._normalize_response(request=request, response=response)

    async def stream_generate(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[ProviderStreamEvent]:
        if _request_contains_image(request) and not _request_uses_ephemeral_storage(request):
            request = _request_with_ephemeral_fallback(request)
        if _request_uses_ephemeral_storage(request):
            async for event in self._stream_ephemeral_with_reconnect(request):
                yield event
            return

        kwargs = self._build_response_create_kwargs(request, stream=True, store=True)
        client = await self._client_instance()
        emitted_deltas: list[TextDeltaEvent | ToolCallDeltaEvent] = []

        try:
            stream = await client.responses.create(**kwargs)
            async for event in self._normalize_provider_stream(request=request, stream=stream):
                if isinstance(event, (TextDeltaEvent, ToolCallDeltaEvent)):
                    emitted_deltas.append(event)
                yield event
        except Exception as exc:
            mapped = self._map_error(exc)
            if not isinstance(mapped, GrokResponseStorageOverflowError):
                raise mapped from exc

            fallback = _request_with_ephemeral_fallback(request)
            async for event in self._suppress_replayed_deltas(
                self._stream_ephemeral_with_reconnect(fallback),
                emitted_deltas,
            ):
                yield event

    async def _generate_ephemeral(self, request: LLMRequest) -> LLMResponse:
        response: LLMResponse | None = None
        async for event in self._stream_ephemeral_with_reconnect(request):
            if isinstance(event, DoneEvent):
                response = event.response
        if response is None:
            raise StreamProtocolError("Grok WebSocket closed without a completed response.")
        return response

    async def _stream_ephemeral_with_reconnect(
        self,
        request: LLMRequest,
    ) -> AsyncIterator[ProviderStreamEvent]:
        emitted_deltas: list[TextDeltaEvent | ToolCallDeltaEvent] = []
        for attempt in range(2):
            try:
                stream = self._stream_ephemeral_once(
                    request,
                    force_reconnect=attempt > 0,
                )
                if attempt == 0:
                    async for event in stream:
                        if isinstance(event, (TextDeltaEvent, ToolCallDeltaEvent)):
                            emitted_deltas.append(event)
                        yield event
                else:
                    async for event in self._suppress_replayed_deltas(
                        stream,
                        emitted_deltas,
                    ):
                        yield event
                return
            except Exception as exc:
                mapped = self._map_error(exc)
                if attempt > 0 or not _is_reconnectable_websocket_error(mapped):
                    raise mapped from exc
                continuation = request.stateful_continuation
                if continuation is not None:
                    await self._drop_websocket_session(continuation.session_key)

        raise StreamProtocolError("Grok WebSocket reconnect loop exited unexpectedly.")

    async def _stream_ephemeral_once(
        self,
        request: LLMRequest,
        *,
        force_reconnect: bool,
    ) -> AsyncIterator[ProviderStreamEvent]:
        continuation = request.stateful_continuation
        if continuation is None:
            raise LLMConfigurationError(
                "Grok ephemeral continuation requires stateful_continuation."
            )
        if force_reconnect:
            await self._drop_websocket_session(continuation.session_key)
        session = await self._websocket_session(continuation)

        completed = False
        try:
            async with session.lock:
                previous_response_id = await self._prepare_websocket_continuation(
                    session=session,
                    request=request,
                )
                payload = self._build_websocket_payload(
                    request=request,
                    messages=request.messages,
                    previous_response_id=previous_response_id,
                    generate=True,
                )
                raw_stream = self._websocket_response_events(session, payload)
                async for event in self._normalize_provider_stream(
                    request=request,
                    stream=raw_stream,
                ):
                    if isinstance(event, DoneEvent):
                        session.live_response_id = event.response.response_id
                        session.last_used_at = asyncio.get_running_loop().time()
                        completed = True
                    yield event
        finally:
            if not completed:
                await self._drop_websocket_session(
                    continuation.session_key,
                    expected=session,
                )

    async def _prepare_websocket_continuation(
        self,
        *,
        session: _GrokWebSocketSession,
        request: LLMRequest,
    ) -> str | None:
        continuation = request.stateful_continuation
        if continuation is None:
            raise LLMConfigurationError("Missing Grok stateful continuation metadata.")

        requested_response_id = request.previous_response_id
        if session.live_response_id == requested_response_id:
            return requested_response_id

        durable_response_id = continuation.durable_response_id
        if requested_response_id == durable_response_id:
            session.live_response_id = durable_response_id
            return durable_response_id

        recovery_messages = continuation.materialize_recovery_messages()
        if not recovery_messages:
            raise GrokWebSocketContinuationError(
                "Grok live continuation is unavailable and no recovery tail was supplied.",
                metadata={
                    "code": "missing_recovery_tail",
                    "requested_response_id": requested_response_id,
                    "durable_response_id": durable_response_id,
                },
            )

        warmup_payload = self._build_websocket_payload(
            request=request,
            messages=recovery_messages,
            previous_response_id=durable_response_id,
            generate=False,
        )
        warmup_response_id: str | None = None
        async for raw_event in self._websocket_response_events(session, warmup_payload):
            event_type = str(_field(raw_event, "type", ""))
            if event_type == "error":
                raise self._stream_error_from_event(raw_event)
            if event_type == "response.failed":
                raise self._stream_error_from_failed_response(_field(raw_event, "response"))
            if event_type in {"response.completed", "response.incomplete"}:
                response = _field(raw_event, "response")
                warmup_response_id = _normalize_optional_string(_field(response, "id"))

        if warmup_response_id is None:
            raise StreamProtocolError(
                "Grok WebSocket recovery warmup completed without a response id."
            )
        session.live_response_id = warmup_response_id
        return warmup_response_id

    async def _websocket_session(
        self,
        continuation: StatefulContinuation,
    ) -> _GrokWebSocketSession:
        loop = asyncio.get_running_loop()
        now = loop.time()
        stale_sessions: list[_GrokWebSocketSession] = []
        async with self._websocket_sessions_lock:
            session = self._websocket_sessions.get(continuation.session_key)
            if (
                session is not None
                and not session.lock.locked()
                and now - session.opened_at >= _GROK_WEBSOCKET_MAX_AGE_SECONDS
            ):
                self._websocket_sessions.pop(continuation.session_key, None)
                stale_sessions.append(session)
                session = None

            if session is None:
                session = await self._open_websocket_session(continuation)
                self._websocket_sessions[continuation.session_key] = session

            if len(self._websocket_sessions) > _GROK_WEBSOCKET_POOL_LIMIT:
                candidates = [
                    candidate
                    for candidate in self._websocket_sessions.values()
                    if candidate is not session and not candidate.lock.locked()
                ]
                if candidates:
                    oldest = min(candidates, key=lambda candidate: candidate.last_used_at)
                    self._websocket_sessions.pop(oldest.session_key, None)
                    stale_sessions.append(oldest)

        for stale in stale_sessions:
            await self._close_websocket_session(stale)
        return session

    async def _open_websocket_session(
        self,
        continuation: StatefulContinuation,
    ) -> _GrokWebSocketSession:
        api_key = self._api_key()
        websocket = await connect(
            self._websocket_url(),
            additional_headers={"Authorization": f"Bearer {api_key}"},
            open_timeout=self._transport_timeouts.connect_seconds,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        )
        now = asyncio.get_running_loop().time()
        return _GrokWebSocketSession(
            session_key=continuation.session_key,
            websocket=websocket,
            lock=asyncio.Lock(),
            opened_at=now,
            last_used_at=now,
            generation=continuation.generation + 1,
        )

    async def _drop_websocket_session(
        self,
        session_key: str,
        *,
        expected: _GrokWebSocketSession | None = None,
    ) -> None:
        async with self._websocket_sessions_lock:
            session = self._websocket_sessions.get(session_key)
            if session is None or (expected is not None and session is not expected):
                return
            self._websocket_sessions.pop(session_key, None)
        await self._close_websocket_session(session)

    async def _close_websocket_session(self, session: _GrokWebSocketSession) -> None:
        close = getattr(session.websocket, "close", None)
        if not callable(close):
            return
        try:
            maybe = close()
            if inspect.isawaitable(maybe):
                await maybe
        except Exception:
            return

    async def _websocket_response_events(
        self,
        session: _GrokWebSocketSession,
        payload: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        await session.websocket.send(json.dumps(payload, separators=(",", ":")))
        while True:
            try:
                raw_event = await asyncio.wait_for(
                    session.websocket.recv(),
                    timeout=self._transport_timeouts.read_seconds,
                )
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    "Grok WebSocket read timed out.",
                    metadata={
                        "timeout_kind": "websocket_read",
                        "read_timeout_seconds": self._transport_timeouts.read_seconds,
                    },
                ) from exc
            if isinstance(raw_event, bytes):
                raw_event = raw_event.decode("utf-8")
            try:
                event = json.loads(raw_event)
            except (TypeError, ValueError) as exc:
                raise StreamProtocolError("Grok WebSocket returned invalid JSON.") from exc
            if not isinstance(event, dict):
                raise StreamProtocolError("Grok WebSocket returned a non-object event.")
            yield event
            if event.get("type") in {
                "response.completed",
                "response.incomplete",
                "response.failed",
                "error",
            }:
                return

    def _build_websocket_payload(
        self,
        *,
        request: LLMRequest,
        messages: Sequence[LLMMessage],
        previous_response_id: str | None,
        generate: bool,
    ) -> dict[str, Any]:
        payload_request = replace(
            request,
            messages=messages,
            previous_response_id=previous_response_id,
        )
        payload = self._build_response_create_kwargs(
            payload_request,
            stream=False,
            store=False,
        )
        payload.pop("stream", None)
        payload["type"] = "response.create"
        if not generate:
            payload["generate"] = False
        return payload

    async def _suppress_replayed_deltas(
        self,
        stream: AsyncIterator[ProviderStreamEvent],
        emitted_deltas: Sequence[TextDeltaEvent | ToolCallDeltaEvent],
    ) -> AsyncIterator[ProviderStreamEvent]:
        emitted_text_deltas = [
            event for event in emitted_deltas if isinstance(event, TextDeltaEvent)
        ]
        suppress_tool_deltas = any(
            isinstance(event, ToolCallDeltaEvent) for event in emitted_deltas
        )
        replay_index = 0
        replay_offset = 0
        async for event in stream:
            if isinstance(event, ToolCallDeltaEvent) and suppress_tool_deltas:
                # Tool execution is driven by DoneEvent.response, not stream deltas.
                # A retried response receives fresh call ids, so suppress every retry
                # delta after the first attempt already announced the tool name.
                continue
            if isinstance(event, TextDeltaEvent):
                remaining = _stream_delta_text(event)
                while remaining and replay_index < len(emitted_text_deltas):
                    expected = emitted_text_deltas[replay_index]
                    if not _same_stream_delta_channel(event, expected):
                        raise StreamProtocolError(
                            "Grok store=false retry diverged from the partial durable stream."
                        )
                    expected_remaining = _stream_delta_text(expected)[replay_offset:]
                    if expected_remaining.startswith(remaining):
                        replay_offset += len(remaining)
                        remaining = ""
                        if replay_offset == len(_stream_delta_text(expected)):
                            replay_index += 1
                            replay_offset = 0
                        break
                    if remaining.startswith(expected_remaining):
                        remaining = remaining[len(expected_remaining) :]
                        replay_index += 1
                        replay_offset = 0
                        continue
                    raise StreamProtocolError(
                        "Grok store=false retry diverged from the partial durable stream."
                    )
                if replay_index < len(emitted_text_deltas) or not remaining:
                    continue
                event = _replace_stream_delta_text(event, remaining)
            if replay_index < len(emitted_text_deltas) and isinstance(
                event, (UsageDeltaEvent, DoneEvent)
            ):
                raise StreamProtocolError(
                    "Grok store=false retry ended before replaying the partial durable stream."
                )
            yield event

    async def _normalize_provider_stream(
        self,
        *,
        request: LLMRequest,
        stream: AsyncIterator[Any],
    ) -> AsyncIterator[ProviderStreamEvent]:
        tool_name_by_item_id: dict[str, str] = {}
        call_id_by_item_id: dict[str, str] = {}
        saw_completion = False

        async for event in stream:
            event_type = str(_field(event, "type", ""))

            if event_type == "response.output_text.delta":
                yield TextDeltaEvent(delta=str(_field(event, "delta", "")))
                continue

            if event_type in {
                "response.output_item.added",
                "response.output_item.done",
            }:
                item = _field(event, "item")
                if _field(item, "type") == "function_call":
                    item_id = _normalize_optional_string(_field(item, "id"))
                    if item_id:
                        tool_name_by_item_id[item_id] = str(_field(item, "name", ""))
                        call_id_by_item_id[item_id] = str(_field(item, "call_id", item_id))
                yield _grok_activity_event(event_type, event)
                continue

            if event_type == "response.function_call_arguments.delta":
                item_id = str(_field(event, "item_id", ""))
                call_id = call_id_by_item_id.get(item_id, item_id)
                yield ToolCallDeltaEvent(
                    call_id=call_id,
                    tool_name=tool_name_by_item_id.get(item_id),
                    arguments_delta=str(_field(event, "delta", "")),
                )
                continue

            if event_type == "response.function_call_arguments.done":
                item_id = str(_field(event, "item_id", ""))
                call_id = call_id_by_item_id.get(item_id, item_id)
                yield ToolCallDeltaEvent(
                    call_id=call_id,
                    tool_name=str(_field(event, "name", "")) or None,
                    arguments_delta=str(_field(event, "arguments", "")),
                )
                continue

            if event_type in {"response.completed", "response.incomplete"}:
                normalized = self._normalize_response(
                    request=request,
                    response=_field(event, "response"),
                )
                if normalized.usage is not None:
                    yield UsageDeltaEvent(usage=normalized.usage)
                yield DoneEvent(response=normalized)
                saw_completion = True
                continue

            if event_type == "response.failed":
                yield _grok_activity_event(event_type, event)
                raise self._stream_error_from_failed_response(_field(event, "response"))

            if event_type == "error":
                yield _grok_activity_event(event_type, event)
                raise self._stream_error_from_event(event)

            yield _grok_activity_event(event_type or "unknown", event)

        if not saw_completion:
            raise StreamProtocolError("Grok stream closed without a response.completed event.")

    def _stream_error_from_failed_response(self, response: Any) -> ProviderResponseError:
        message = self._extract_stream_failed_message(response)
        error = _field(response, "error")
        return _grok_response_error(
            message,
            status=_extract_optional_int(response, "status"),
            code=_normalize_optional_string(_field(error, "code")),
            param=_normalize_optional_string(_field(error, "param")),
        )

    def _stream_error_from_event(self, event: Any) -> ProviderResponseError:
        error = _field(event, "error")
        message = _normalize_optional_string(_field(error, "message"))
        if message is None:
            message = _normalize_optional_string(_field(event, "message"))
        return _grok_response_error(
            message or "Grok streaming response failed.",
            status=_extract_optional_int(event, "status"),
            code=_normalize_optional_string(_field(error, "code")),
            param=_normalize_optional_string(_field(error, "param")),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise UnsupportedCapabilityError("Provider 'grok' does not support embeddings.")

    async def aclose(self) -> None:
        async with self._websocket_sessions_lock:
            websocket_sessions = tuple(self._websocket_sessions.values())
            self._websocket_sessions.clear()
        await asyncio.gather(
            *(self._close_websocket_session(session) for session in websocket_sessions),
            return_exceptions=True,
        )
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

            self._client = AsyncOpenAI(
                api_key=self._api_key(),
                base_url=self._settings.base_url.rstrip("/"),
                timeout=self._transport_timeouts.as_httpx(),
                max_retries=0,
            )
            return self._client

    def _api_key(self) -> str:
        api_key = self._settings.api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError("XAI_API_KEY is required for the Grok provider.")
        return api_key

    def _websocket_url(self) -> str:
        base_url = self._settings.base_url.rstrip("/")
        if base_url.startswith("https://"):
            base_url = "wss://" + base_url.removeprefix("https://")
        elif base_url.startswith("http://"):
            base_url = "ws://" + base_url.removeprefix("http://")
        if base_url.endswith("/responses"):
            return base_url
        return f"{base_url}/responses"

    def _build_response_create_kwargs(
        self,
        request: LLMRequest,
        *,
        stream: bool,
        store: bool | None = None,
    ) -> dict[str, Any]:
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
            "store": (store if store is not None else not _request_uses_ephemeral_storage(request)),
            "parallel_tool_calls": request.parallel_tool_calls,
        }

        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens
        if self._settings.reasoning_effort is not None:
            kwargs["reasoning"] = {"effort": self._settings.reasoning_effort}
        if request.previous_response_id is not None:
            kwargs["previous_response_id"] = request.previous_response_id
        if request.prompt_cache_key is not None:
            kwargs["prompt_cache_key"] = request.prompt_cache_key

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
                    raise LLMConfigurationError("Grok provider supports image_url, not file_id.")
                content.append(
                    {
                        "type": "input_image",
                        "image_url": part.image_url,
                        "detail": _normalize_grok_image_detail(part.detail),
                    }
                )
            elif isinstance(part, LocalImagePart):
                if message.role == "assistant":
                    raise LLMConfigurationError(
                        "Grok provider does not support assistant image history items."
                    )
                try:
                    image_bytes = Path(part.path).read_bytes()
                except OSError as exc:
                    raise LLMConfigurationError(
                        f"Grok recovery image is unavailable: {part.path}"
                    ) from exc
                content.append(
                    {
                        "type": "input_image",
                        "image_url": ImagePart.from_bytes(
                            media_type=part.media_type,
                            data=image_bytes,
                            detail=part.detail,
                        ).image_url,
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
        response_output = _serialize_response_output_items(_field(response, "output", []))
        tool_calls = self._extract_tool_calls(
            response_output=response_output,
            request_tools=request.tools,
        )
        usage_obj = _field(response, "usage")
        usage = self._normalize_usage(usage_obj)
        continuation = request.stateful_continuation
        storage_mode = continuation.storage_mode if continuation is not None else "durable"
        response_id = _normalize_optional_string(_field(response, "id"))
        durable_response_id = (
            response_id
            if storage_mode == "durable"
            else continuation.durable_response_id
            if continuation is not None
            else None
        )

        provider_metadata: dict[str, Any] = {
            "status": _field(response, "status"),
            "incomplete_reason": _normalize_optional_string(
                _field(_field(response, "incomplete_details"), "reason")
            ),
            "response_output": response_output,
            "response_storage_mode": storage_mode,
            "durable_response_id": durable_response_id,
        }
        if continuation is not None and storage_mode == "ephemeral":
            websocket_session = self._websocket_sessions.get(continuation.session_key)
            provider_metadata["websocket_generation"] = (
                websocket_session.generation
                if websocket_session is not None
                else continuation.generation
            )

        input_tokens_details = _serialize_optional_mapping(
            _field(usage_obj, "input_tokens_details")
        )
        if input_tokens_details is not None:
            provider_metadata["input_tokens_details"] = input_tokens_details
            cached_tokens = input_tokens_details.get("cached_tokens")
            if isinstance(cached_tokens, int):
                provider_metadata["cached_tokens"] = cached_tokens

        output_tokens_details = _serialize_optional_mapping(
            _field(usage_obj, "output_tokens_details")
        )
        if output_tokens_details is not None:
            provider_metadata["output_tokens_details"] = output_tokens_details
            reasoning_tokens = output_tokens_details.get("reasoning_tokens")
            if isinstance(reasoning_tokens, int):
                provider_metadata["reasoning_tokens"] = reasoning_tokens

        return LLMResponse(
            provider=self.name,
            model=_field(response, "model") or request.model or "",
            text=self._extract_response_text(response=response, response_output=response_output),
            tool_calls=tool_calls,
            finish_reason=self._infer_finish_reason(response=response, tool_calls=tool_calls),
            usage=usage,
            response_id=response_id,
            provider_metadata=provider_metadata,
        )

    def _extract_response_text(
        self,
        *,
        response: Any,
        response_output: Sequence[dict[str, Any]],
    ) -> str:
        output_text = _field(response, "output_text")
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
        error = _field(response, "error")
        if error is None:
            return "Grok streaming response failed."
        if isinstance(error, str) and error:
            return error
        message = _field(error, "message")
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

    def _infer_finish_reason(
        self, *, response: Any, tool_calls: Sequence[ToolCall]
    ) -> FinishReason:
        status = _field(response, "status")
        if status == "failed":
            return "error"
        if status == "incomplete":
            reason = _field(_field(response, "incomplete_details"), "reason")
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
        if isinstance(error, ConnectionClosed):
            return ProviderTemporaryError(
                str(error),
                metadata={"transport": "websocket"},
            )
        if _is_response_storage_overflow_message(str(error)):
            return _grok_response_error(str(error))
        if isinstance(error, (ProviderResponseError, StreamProtocolError)):
            return error
        if isinstance(error, (AuthenticationError, PermissionDeniedError)):
            return ProviderAuthenticationError(str(error))
        if isinstance(error, RateLimitError):
            return ProviderRateLimitError(str(error))
        if isinstance(error, APITimeoutError):
            return ProviderTimeoutError(
                str(error),
                metadata=transport_timeout_metadata(
                    error,
                    timeouts=self._transport_timeouts,
                ),
            )
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


def _grok_activity_event(event_type: str, event: Any) -> ProviderActivityEvent:
    response = _field(event, "response")
    response_id = _normalize_optional_string(_field(response, "id"))
    if response_id is None:
        response_id = _normalize_optional_string(_field(event, "response_id"))
    return ProviderActivityEvent(
        provider_event_type=event_type,
        response_id=response_id,
    )


def _normalize_grok_image_detail(detail: str) -> str:
    if detail == "original":
        return "high"
    if detail in {"auto", "low", "high"}:
        return detail
    raise LLMConfigurationError(f"Grok provider does not support image detail '{detail}'.")


def _request_uses_ephemeral_storage(request: LLMRequest) -> bool:
    continuation = request.stateful_continuation
    return continuation is not None and continuation.storage_mode == "ephemeral"


def _request_contains_image(request: LLMRequest) -> bool:
    return any(
        isinstance(part, (ImagePart, LocalImagePart))
        for message in request.messages
        for part in message.parts
    )


def _request_with_ephemeral_fallback(request: LLMRequest) -> LLMRequest:
    continuation = request.stateful_continuation
    if continuation is None:
        session_key = request.prompt_cache_key or f"grok-request-{id(request)}"
        continuation = StatefulContinuation(
            session_key=session_key,
            storage_mode="ephemeral",
            durable_response_id=request.previous_response_id,
        )
    else:
        continuation = replace(
            continuation,
            storage_mode="ephemeral",
            durable_response_id=(continuation.durable_response_id or request.previous_response_id),
        )
    return replace(request, stateful_continuation=continuation)


def _grok_response_error(
    message: str,
    *,
    status: int | None = None,
    code: str | None = None,
    param: str | None = None,
) -> ProviderResponseError:
    metadata: dict[str, Any] = {
        "status": status,
        "code": code,
        "param": param,
    }
    if _is_response_storage_overflow_message(message):
        metadata.update(
            {
                "code": code or "response_storage_too_large",
                "retryable_with_store_false": True,
            }
        )
        return GrokResponseStorageOverflowError(message, metadata=metadata)
    if code in {
        "previous_response_not_found",
        "websocket_connection_limit_reached",
    }:
        return GrokWebSocketContinuationError(message, metadata=metadata)
    return ProviderResponseError(message, metadata=metadata)


def _is_response_storage_overflow_message(message: str) -> bool:
    normalized = message.strip().lower()
    return _RESPONSE_STORAGE_OVERFLOW_HINT in normalized and "store" in normalized


def _is_reconnectable_websocket_error(error: Exception) -> bool:
    return isinstance(
        error,
        (
            GrokWebSocketContinuationError,
            ProviderTemporaryError,
            ProviderTimeoutError,
        ),
    )


def _field(source: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(field_name, default)
    return getattr(source, field_name, default)


def _stream_delta_text(event: TextDeltaEvent | ToolCallDeltaEvent) -> str:
    if isinstance(event, TextDeltaEvent):
        return event.delta
    return event.arguments_delta


def _same_stream_delta_channel(
    left: TextDeltaEvent | ToolCallDeltaEvent,
    right: TextDeltaEvent | ToolCallDeltaEvent,
) -> bool:
    if isinstance(left, TextDeltaEvent) or isinstance(right, TextDeltaEvent):
        return isinstance(left, TextDeltaEvent) and isinstance(right, TextDeltaEvent)
    return left.call_id == right.call_id and (
        left.tool_name is None or right.tool_name is None or left.tool_name == right.tool_name
    )


def _replace_stream_delta_text(
    event: TextDeltaEvent | ToolCallDeltaEvent,
    delta: str,
) -> TextDeltaEvent | ToolCallDeltaEvent:
    if isinstance(event, TextDeltaEvent):
        return TextDeltaEvent(delta=delta)
    return ToolCallDeltaEvent(
        call_id=event.call_id,
        tool_name=event.tool_name,
        arguments_delta=delta,
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
        return {str(key): _serialize_json_compatible(item) for key, item in value.items()}
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
            key: item for key, item in vars(value).items() if not key.startswith("_")
        }
        if public_attributes:
            return _serialize_json_compatible(public_attributes)

    return str(value)
