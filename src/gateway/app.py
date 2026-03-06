"""Starlette websocket gateway for routing user turns to agent sessions."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace

from core import AgentLoop, AgentTextDeltaEvent, AgentTurnDoneEvent, ContextBudgetError, CoreSettings
from llm import LLMService
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from .config import GatewaySettings
from .protocol import (
    ProtocolError,
    build_assistant_delta_event,
    build_assistant_message_event,
    build_error_event,
    build_ready_event,
    parse_client_event,
)
from .session_router import InvalidRouteIDError, SessionRouter, validate_route_id

_INTERNAL_ERROR_MESSAGE = "Internal error while processing message."
LOGGER = logging.getLogger(__name__)


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
            await websocket.send_json(
                build_error_event(code="invalid_route_id", message=str(exc))
            )
            await websocket.close(code=1008)
            return

        await websocket.send_json(
            build_ready_event(
                route_id=route_id,
                session_id=resolved_router.active_session_id(route_id),
            )
        )

        while True:
            try:
                raw_message = await websocket.receive_text()
            except WebSocketDisconnect:
                return

            try:
                payload = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json(
                    build_error_event(
                        code="invalid_json",
                        message="Message must be valid JSON.",
                    )
                )
                continue

            try:
                event = parse_client_event(
                    payload,
                    max_message_chars=resolved_gateway_settings.max_message_chars,
                )
            except ProtocolError as exc:
                await websocket.send_json(
                    build_error_event(code=exc.code, message=exc.message)
                )
                continue

            try:
                async for turn_event in resolved_router.stream_turn(route_id, event.text):
                    if isinstance(turn_event, AgentTextDeltaEvent):
                        await websocket.send_json(
                            build_assistant_delta_event(
                                session_id=turn_event.session_id,
                                delta=turn_event.delta,
                            )
                        )
                        continue

                    if isinstance(turn_event, AgentTurnDoneEvent):
                        await websocket.send_json(
                            build_assistant_message_event(
                                session_id=turn_event.session_id,
                                text=turn_event.response_text,
                                command=turn_event.command,
                                compaction_performed=turn_event.compaction_performed,
                            )
                        )
            except ContextBudgetError as exc:
                await websocket.send_json(
                    build_error_event(
                        code="context_budget_exceeded",
                        message=str(exc),
                    )
                )
                continue
            except Exception:
                LOGGER.exception("Unhandled gateway turn error for route_id=%s", route_id)
                await websocket.send_json(
                    build_error_event(
                        code="internal_error",
                        message=_INTERNAL_ERROR_MESSAGE,
                    )
                )
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
    base_storage_dir = resolved_core_settings.storage_dir

    def agent_loop_factory(route_id: str) -> AgentLoop:
        route_storage_dir = base_storage_dir / "routes" / route_id
        route_core_settings = replace(
            resolved_core_settings,
            storage_dir=route_storage_dir,
        )
        return AgentLoop(
            llm_service=resolved_llm_service,
            settings=route_core_settings,
        )

    return SessionRouter(agent_loop_factory)
