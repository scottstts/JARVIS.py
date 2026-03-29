"""Starlette websocket gateway for persistent route-scoped agent sessions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from jarvis.core import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentTextDeltaEvent,
    AgentToolCallEvent,
    AgentTurnStartedEvent,
    AgentTurnDoneEvent,
    ContextBudgetError,
    CoreSettings,
)
from jarvis.llm import LLMService, ProviderTimeoutError
from jarvis.logging_setup import get_application_logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from jarvis.tools.config import ToolSettings
from jarvis.tools.remote_runtime_client import ensure_remote_tool_runtime_healthy

from .config import GatewaySettings
from .protocol import (
    ClientApprovalResponse,
    ClientStopTurn,
    ProtocolError,
    build_approval_ack_event,
    build_error_event,
    build_ready_event,
    build_route_event_payload,
    build_stop_ack_event,
    parse_client_event,
)
from .route_runtime import RouteRuntime
from .session_router import InvalidRouteIDError, SessionRouter, validate_route_id

LOGGER = get_application_logger(__name__)
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
    resolved_tool_settings: ToolSettings | None = None
    owns_llm_service = False
    if router is None:
        resolved_core_settings = core_settings or CoreSettings.from_env()
        if resolved_llm_service is None:
            resolved_llm_service = LLMService()
            owns_llm_service = True
        resolved_router = _build_default_router(
            core_settings=resolved_core_settings,
            llm_service=resolved_llm_service,
        )
        resolved_tool_settings = ToolSettings.from_workspace_dir(
            resolved_core_settings.workspace_dir
        )
    else:
        resolved_router = router

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            if resolved_tool_settings is not None:
                await ensure_remote_tool_runtime_healthy(resolved_tool_settings)
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
                build_error_event(code="invalid_route_id", message=str(exc)),
            ):
                return
            await websocket.close(code=1008)
            return

        if not hasattr(resolved_router, "subscribe") or not hasattr(
            resolved_router,
            "enqueue_message",
        ):
            await _serve_legacy_router_connection(
                websocket=websocket,
                route_id=route_id,
                router=resolved_router,
                max_message_chars=resolved_gateway_settings.max_message_chars,
            )
            return

        subscriber_id, event_queue = resolved_router.subscribe(route_id)

        if not await _send_json_if_open(
            websocket,
            build_ready_event(
                route_id=route_id,
                session_id=resolved_router.active_session_id(route_id),
            ),
        ):
            resolved_router.unsubscribe(route_id, subscriber_id)
            return

        async def _reader() -> None:
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
                        ),
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
                        build_error_event(code=exc.code, message=exc.message),
                    ):
                        return
                    continue

                if isinstance(event, ClientStopTurn):
                    if not await _send_json_if_open(
                        websocket,
                        build_stop_ack_event(
                            stop_requested=resolved_router.request_stop(route_id),
                        ),
                    ):
                        return
                    continue

                if isinstance(event, ClientApprovalResponse):
                    if not await _send_json_if_open(
                        websocket,
                        build_approval_ack_event(
                            resolved=resolved_router.resolve_approval(
                                route_id,
                                event.approval_id,
                                event.approved,
                            )
                        ),
                    ):
                        return
                    continue

                await resolved_router.enqueue_message(
                    route_id,
                    event.text,
                    client_message_id=event.client_message_id,
                )

        async def _writer() -> None:
            while True:
                event = await event_queue.get()
                if not event.public:
                    continue
                if not await _send_json_if_open(
                    websocket,
                    build_route_event_payload(event),
                ):
                    return

        reader_task = asyncio.create_task(_reader(), name=f"gateway-reader-{route_id}")
        writer_task = asyncio.create_task(_writer(), name=f"gateway-writer-{route_id}")
        done, pending = await asyncio.wait(
            {reader_task, writer_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
        for task in done:
            exception = task.exception()
            if exception is not None:
                raise exception
        resolved_router.unsubscribe(route_id, subscriber_id)

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

    def route_runtime_factory(route_id: str) -> RouteRuntime:
        route_transcript_archive_dir = base_transcript_archive_dir / route_id
        route_core_settings = replace(
            resolved_core_settings,
            transcript_archive_dir=route_transcript_archive_dir,
        )
        return RouteRuntime(
            route_id=route_id,
            llm_service=resolved_llm_service,
            core_settings=route_core_settings,
        )

    return SessionRouter(route_runtime_factory)


async def _serve_legacy_router_connection(
    *,
    websocket: WebSocket,
    route_id: str,
    router: Any,
    max_message_chars: int,
) -> None:
    if not await _send_json_if_open(
        websocket,
        build_ready_event(
            route_id=route_id,
            session_id=router.active_session_id(route_id),
        ),
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
                ),
            ):
                return
            continue

        try:
            event = parse_client_event(
                payload,
                max_message_chars=max_message_chars,
            )
        except ProtocolError as exc:
            if not await _send_json_if_open(
                websocket,
                build_error_event(code=exc.code, message=exc.message),
            ):
                return
            continue

        if isinstance(event, ClientStopTurn):
            if not await _send_json_if_open(
                websocket,
                build_stop_ack_event(
                    stop_requested=router.request_stop(route_id),
                ),
            ):
                return
            continue

        if isinstance(event, ClientApprovalResponse):
            if not await _send_json_if_open(
                websocket,
                build_approval_ack_event(
                    resolved=router.resolve_approval(
                        route_id,
                        event.approval_id,
                        event.approved,
                    )
                ),
            ):
                return
            continue

        try:
            async for turn_event in router.stream_turn(route_id, event.text):
                if isinstance(turn_event, AgentTurnStartedEvent):
                    if not await _send_json_if_open(
                        websocket,
                        {
                            "type": "turn_started",
                            "session_id": turn_event.session_id,
                            "turn_id": turn_event.turn_id,
                            "turn_kind": "user",
                            "client_message_id": event.client_message_id,
                        },
                    ):
                        return
                    continue
                if isinstance(turn_event, AgentTextDeltaEvent):
                    if not await _send_json_if_open(
                        websocket,
                        {
                            "type": "assistant_delta",
                            "session_id": turn_event.session_id,
                            "turn_id": turn_event.turn_id,
                            "turn_kind": "user",
                            "client_message_id": event.client_message_id,
                            "delta": turn_event.delta,
                        },
                    ):
                        return
                    continue
                if isinstance(turn_event, AgentAssistantMessageEvent):
                    if not await _send_json_if_open(
                        websocket,
                        {
                            "type": "assistant_message",
                            "session_id": turn_event.session_id,
                            "turn_id": turn_event.turn_id,
                            "turn_kind": "user",
                            "client_message_id": event.client_message_id,
                            "text": turn_event.text,
                        },
                    ):
                        return
                    continue
                if isinstance(turn_event, AgentToolCallEvent):
                    if not await _send_json_if_open(
                        websocket,
                        {
                            "type": "tool_call",
                            "session_id": turn_event.session_id,
                            "turn_id": turn_event.turn_id,
                            "turn_kind": "user",
                            "client_message_id": event.client_message_id,
                            "tool_names": list(turn_event.tool_names),
                        },
                    ):
                        return
                    continue
                if isinstance(turn_event, AgentApprovalRequestEvent):
                    if not await _send_json_if_open(
                        websocket,
                        {
                            "type": "approval_request",
                            "session_id": turn_event.session_id,
                            "turn_id": turn_event.turn_id,
                            "turn_kind": "user",
                            "client_message_id": event.client_message_id,
                            "approval_id": turn_event.approval_id,
                            "kind": turn_event.kind,
                            "summary": turn_event.summary,
                            "details": turn_event.details,
                            "command": turn_event.command,
                            "tool_name": turn_event.tool_name,
                            "inspection_url": turn_event.inspection_url,
                        },
                    ):
                        return
                    continue
                if isinstance(turn_event, AgentTurnDoneEvent):
                    if not await _send_json_if_open(
                        websocket,
                        {
                            "type": "turn_done",
                            "session_id": turn_event.session_id,
                            "turn_id": turn_event.turn_id,
                            "turn_kind": "user",
                            "client_message_id": event.client_message_id,
                            "response_text": turn_event.response_text,
                            "command": turn_event.command,
                            "compaction_performed": turn_event.compaction_performed,
                            "interrupted": turn_event.interrupted,
                            "approval_rejected": turn_event.approval_rejected,
                            "interruption_reason": turn_event.interruption_reason,
                        },
                    ):
                        return
        except ContextBudgetError as exc:
            if not await _send_json_if_open(
                websocket,
                build_error_event(
                    code="context_budget_exceeded",
                    message=str(exc),
                ),
            ):
                return
            continue
        except ProviderTimeoutError:
            if not await _send_json_if_open(
                websocket,
                build_error_event(
                    code="provider_timeout",
                    message="The model timed out while processing that message.",
                ),
            ):
                return
            continue
        except Exception:
            if not await _send_json_if_open(
                websocket,
                build_error_event(
                    code="internal_error",
                    message="Internal error while processing message.",
                ),
            ):
                return
