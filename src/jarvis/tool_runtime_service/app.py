"""ASGI app for isolated bash execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from jarvis.tools.config import ToolSettings
from jarvis.tools.types import ToolExecutionContext

from .bash_executor import build_service_bash_executor
from .models import (
    ToolRuntimeRequestError,
    parse_execute_request,
    serialize_execution_result,
)


def create_app(settings: ToolSettings | None = None) -> Starlette:
    resolved_settings = settings or ToolSettings.from_env()
    bash_executor = build_service_bash_executor(resolved_settings)

    async def healthcheck(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "workspace_dir": str(resolved_settings.workspace_dir),
            }
        )

    async def execute_bash(request: Request) -> JSONResponse:
        return await _execute_request(request, executor=bash_executor, workspace_dir=resolved_settings.workspace_dir)

    return Starlette(
        debug=False,
        routes=[
            Route("/health", healthcheck, methods=["GET"]),
            Route("/tools/bash/execute", execute_bash, methods=["POST"]),
        ],
    )


def build_asgi_app() -> Starlette:
    return create_app()


async def _execute_request(
    request: Request,
    *,
    executor: Callable[..., Any],
    workspace_dir,
) -> JSONResponse:
    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            {"error": "request body must be valid JSON."},
            status_code=400,
        )

    try:
        tool_request = parse_execute_request(payload)
    except ToolRuntimeRequestError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        resolved_workspace_dir = _resolve_workspace_dir(
            requested_path=tool_request.workspace_dir,
            shared_workspace_root=workspace_dir,
        )
    except ToolRuntimeRequestError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    context = ToolExecutionContext(
        workspace_dir=resolved_workspace_dir,
        route_id=tool_request.route_id,
        session_id=tool_request.session_id,
    )
    result = await executor(
        call_id=tool_request.call_id,
        arguments=tool_request.arguments,
        context=context,
    )
    return JSONResponse(serialize_execution_result(result))


def _resolve_workspace_dir(
    *,
    requested_path: str | None,
    shared_workspace_root: Path,
) -> Path:
    root = shared_workspace_root.resolve(strict=False)
    if requested_path is None:
        return root

    candidate = Path(requested_path).resolve(strict=False)
    if candidate != root and not candidate.is_relative_to(root):
        raise ToolRuntimeRequestError(
            "workspace_dir must stay inside the shared /workspace mount."
        )
    if not candidate.exists():
        raise ToolRuntimeRequestError("workspace_dir does not exist inside tool_runtime.")
    if not candidate.is_dir():
        raise ToolRuntimeRequestError("workspace_dir must point to a directory.")
    return candidate
