"""Request/response models for the isolated tool-runtime service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tools.types import ToolExecutionResult


class ToolRuntimeRequestError(ValueError):
    """Raised when a request payload is malformed."""


@dataclass(slots=True, frozen=True)
class ToolRuntimeExecuteRequest:
    call_id: str
    arguments: dict[str, Any]
    workspace_dir: str | None = None
    session_id: str | None = None
    route_id: str | None = None


def parse_execute_request(payload: object) -> ToolRuntimeExecuteRequest:
    if not isinstance(payload, dict):
        raise ToolRuntimeRequestError("request payload must be a JSON object.")

    call_id = payload.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        raise ToolRuntimeRequestError("'call_id' must be a non-empty string.")

    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ToolRuntimeRequestError("'arguments' must be a JSON object.")

    workspace_dir = payload.get("workspace_dir")
    if workspace_dir is not None and not isinstance(workspace_dir, str):
        raise ToolRuntimeRequestError("'workspace_dir' must be a string when provided.")

    session_id = payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise ToolRuntimeRequestError("'session_id' must be a string when provided.")

    route_id = payload.get("route_id")
    if route_id is not None and not isinstance(route_id, str):
        raise ToolRuntimeRequestError("'route_id' must be a string when provided.")

    return ToolRuntimeExecuteRequest(
        call_id=call_id.strip(),
        arguments=dict(arguments),
        workspace_dir=workspace_dir.strip() if isinstance(workspace_dir, str) and workspace_dir.strip() else None,
        session_id=session_id.strip() if isinstance(session_id, str) and session_id.strip() else None,
        route_id=route_id.strip() if isinstance(route_id, str) and route_id.strip() else None,
    )


def serialize_execution_result(result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "call_id": result.call_id,
        "name": result.name,
        "ok": result.ok,
        "content": result.content,
        "metadata": dict(result.metadata),
    }
