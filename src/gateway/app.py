"""Starlette websocket gateway for routing user turns to agent sessions."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from core import (
    AgentAssistantMessageEvent,
    AgentLoop,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentTurnDoneEvent,
    ContextBudgetError,
    CoreSettings,
)
from llm import LLMService, ProviderTimeoutError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from .config import GatewaySettings
from .protocol import (
    ClientStopTurn,
    ProtocolError,
    build_assistant_delta_event,
    build_assistant_message_event,
    build_error_event,
    build_ready_event,
    build_stop_ack_event,
    build_tool_call_event,
    build_turn_done_event,
    parse_client_event,
)
from .session_router import InvalidRouteIDError, SessionRouter, validate_route_id

_INTERNAL_ERROR_MESSAGE = "Internal error while processing message."
_PROVIDER_TIMEOUT_MESSAGE = "The model timed out while processing that message."
LOGGER = logging.getLogger(__name__)
_WEBSOCKET_CLOSED_SEND_ERROR = 'Cannot call "send" once a close message has been sent.'


async def _send_json_if_open(websocket: WebSocket, payload: dict[str, Any]) -> bool:
    try:
        await websocket.send_json(payload)
    except WebSocketDisconnect:
        return False
    except RuntimeError as exc:
        if _WEBSOCKET_CLOSED_SEND_ERROR in str(exc):
            return False
        raise
    return True


def create_app(
    *,
    gateway_settings: GatewaySettings | None = None,
    router: SessionRouter | None = None,
    core_settings: CoreSettings | None = None,
    llm_service: LLMService | None = None,
) -> Starlette:
    resolved_gateway_settings = gateway_settings or GatewaySettings.from_env()
    resolved_llm_service = llm_service
    owns_llm_service = False
    if router is None:
        if resolved_llm_service is None:
            resolved_llm_service = LLMService()
            owns_llm_service = True
        resolved_router = _build_default_router(
            core_settings=core_settings,
            llm_service=resolved_llm_service,
        )
    else:
        resolved_router = router

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if owns_llm_service and resolved_llm_service is not None:
                await resolved_llm_service.aclose()

    async def healthcheck(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def websocket_endpoint(websocket: WebSocket) -> None:
        route_id_raw = str(websocket.path_params.get("route_id", ""))
        await websocket.accept()

        try:
            route_id = validate_route_id(route_id_raw)
        except InvalidRouteIDError as exc:
            if not await _send_json_if_open(
                websocket,
                build_error_event(code="invalid_route_id", message=str(exc))
            ):
                return
            await websocket.close(code=1008)
            return

        if not await _send_json_if_open(
            websocket,
            build_ready_event(
                route_id=route_id,
                session_id=resolved_router.active_session_id(route_id),
            )
        ):
            return

        while True:
            try:
                raw_message = await websocket.receive_text()
            except WebSocketDisconnect:
                return

            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                if not await _send_json_if_open(
                    websocket,
                    build_error_event(
                        code="invalid_json",
                        message="Message must be valid JSON.",
                    )
                ):
                    return
                continue

            try:
                event = parse_client_event(
                    payload,
                    max_message_chars=resolved_gateway_settings.max_message_chars,
                )
            except ProtocolError as exc:
                if not await _send_json_if_open(
                    websocket,
                    build_error_event(code=exc.code, message=exc.message)
                ):
                    return
                continue

            if isinstance(event, ClientStopTurn):
                if not await _send_json_if_open(
                    websocket,
                    build_stop_ack_event(
                        stop_requested=resolved_router.request_stop(route_id)
                    ),
                ):
                    return
                continue

            try:
                async for turn_event in resolved_router.stream_turn(route_id, event.text):
                    if isinstance(turn_event, AgentTextDeltaEvent):
                        if not await _send_json_if_open(
                            websocket,
                            build_assistant_delta_event(
                                session_id=turn_event.session_id,
                                delta=turn_event.delta,
                            )
                        ):
                            return
                        continue

                    if isinstance(turn_event, AgentAssistantMessageEvent):
                        if not await _send_json_if_open(
                            websocket,
                            build_assistant_message_event(
                                session_id=turn_event.session_id,
                                text=turn_event.text,
                            )
                        ):
                            return
                        continue

                    if isinstance(turn_event, AgentToolCallEvent):
                        if not await _send_json_if_open(
                            websocket,
                            build_tool_call_event(
                                session_id=turn_event.session_id,
                                tool_names=turn_event.tool_names,
                            )
                        ):
                            return
                        continue

                    if isinstance(turn_event, AgentTurnDoneEvent):
                        if not await _send_json_if_open(
                            websocket,
                            build_turn_done_event(
                                session_id=turn_event.session_id,
                                response_text=turn_event.response_text,
                                command=turn_event.command,
                                compaction_performed=turn_event.compaction_performed,
                                interrupted=turn_event.interrupted,
                            )
                        ):
                            return
            except ContextBudgetError as exc:
                if not await _send_json_if_open(
                    websocket,
                    build_error_event(
                        code="context_budget_exceeded",
                        message=str(exc),
                    )
                ):
                    return
                continue
            except ProviderTimeoutError:
                if not await _send_json_if_open(
                    websocket,
                    build_error_event(
                        code="provider_timeout",
                        message=_PROVIDER_TIMEOUT_MESSAGE,
                    )
                ):
                    return
                continue
            except Exception:
                LOGGER.exception("Unhandled gateway turn error.")
                if not await _send_json_if_open(
                    websocket,
                    build_error_event(
                        code="internal_error",
                        message=_INTERNAL_ERROR_MESSAGE,
                    )
                ):
                    return
                continue

    return Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/healthz", healthcheck),
            WebSocketRoute(
                resolved_gateway_settings.websocket_route_path,
                websocket_endpoint,
            ),
        ],
    )


def build_asgi_app() -> Starlette:
    return create_app()


def _build_default_router(
    *,
    core_settings: CoreSettings | None,
    llm_service: LLMService,
) -> SessionRouter:
    resolved_core_settings = core_settings or CoreSettings.from_env()
    resolved_llm_service = llm_service
    base_transcript_archive_dir = resolved_core_settings.transcript_archive_dir

    def agent_loop_factory(route_id: str) -> AgentLoop:
        route_transcript_archive_dir = base_transcript_archive_dir / route_id
        route_core_settings = replace(
            resolved_core_settings,
            transcript_archive_dir=route_transcript_archive_dir,
        )
        return AgentLoop(
            llm_service=resolved_llm_service,
            settings=route_core_settings,
            route_id=route_id,
        )

    return SessionRouter(agent_loop_factory)
