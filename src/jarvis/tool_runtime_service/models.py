"""Request/response models for the isolated tool-runtime service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from jarvis.tools.types import AgentToolAccess, ToolExecutionResult


class ToolRuntimeRequestError(ValueError):
    """Raised when a request payload is malformed."""


@dataclass(slots=True, frozen=True)
class ToolRuntimeExecuteRequest:
    call_id: str
    arguments: dict[str, Any]
    workspace_dir: str | None = None
    session_id: str | None = None
    route_id: str | None = None
    agent_kind: AgentToolAccess = "main"
    agent_name: str = "Jarvis"
    subagent_id: str | None = None
    workspace_write_allowed_paths: tuple[str, ...] = ()
    workspace_write_denied_paths: tuple[str, ...] = ()
    workspace_lease_generation: int | None = None


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

    agent_kind = payload.get("agent_kind", "main")
    if agent_kind not in {"main", "subagent"}:
        raise ToolRuntimeRequestError("'agent_kind' must be 'main' or 'subagent'.")
    agent_kind = cast(AgentToolAccess, agent_kind)
    agent_name = payload.get("agent_name", "Jarvis")
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ToolRuntimeRequestError("'agent_name' must be a non-empty string.")
    subagent_id = payload.get("subagent_id")
    if subagent_id is not None and not isinstance(subagent_id, str):
        raise ToolRuntimeRequestError("'subagent_id' must be a string when provided.")
    if agent_kind == "subagent" and not str(subagent_id or "").strip():
        raise ToolRuntimeRequestError("subagent requests require 'subagent_id'.")
    allowed_paths = _parse_path_list(
        payload.get("workspace_write_allowed_paths"),
        field="workspace_write_allowed_paths",
    )
    denied_paths = _parse_path_list(
        payload.get("workspace_write_denied_paths"),
        field="workspace_write_denied_paths",
    )
    lease_generation = payload.get("workspace_lease_generation")
    if lease_generation is not None and (
        isinstance(lease_generation, bool)
        or not isinstance(lease_generation, int)
        or lease_generation < 0
    ):
        raise ToolRuntimeRequestError(
            "'workspace_lease_generation' must be a non-negative integer when provided."
        )

    return ToolRuntimeExecuteRequest(
        call_id=call_id.strip(),
        arguments=dict(arguments),
        workspace_dir=workspace_dir.strip() if isinstance(workspace_dir, str) and workspace_dir.strip() else None,
        session_id=session_id.strip() if isinstance(session_id, str) and session_id.strip() else None,
        route_id=route_id.strip() if isinstance(route_id, str) and route_id.strip() else None,
        agent_kind=agent_kind,
        agent_name=agent_name.strip(),
        subagent_id=(
            subagent_id.strip()
            if isinstance(subagent_id, str) and subagent_id.strip()
            else None
        ),
        workspace_write_allowed_paths=allowed_paths,
        workspace_write_denied_paths=denied_paths,
        workspace_lease_generation=lease_generation,
    )


def _parse_path_list(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ToolRuntimeRequestError(
            f"'{field}' must be a list of non-empty path strings when provided."
        )
    return tuple(dict.fromkeys(item.strip() for item in value))


def serialize_execution_result(result: ToolExecutionResult) -> dict[str, Any]:
    return {
        "call_id": result.call_id,
        "name": result.name,
        "ok": result.ok,
        "content": result.content,
        "metadata": dict(result.metadata),
    }
