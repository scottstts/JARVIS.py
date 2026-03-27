"""Shared types for the tools layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from jarvis.llm import ToolDefinition

ToolExposure = Literal["basic", "discoverable"]
AgentToolAccess = Literal["main", "subagent"]

if TYPE_CHECKING:
    from jarvis.memory import MemoryService


@dataclass(slots=True, frozen=True)
class ToolExecutionContext:
    """Runtime context passed to tool executors and policy checks."""

    workspace_dir: Path
    route_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    agent_kind: AgentToolAccess = "main"
    agent_name: str = "Jarvis"
    subagent_id: str | None = None
    memory_service: "MemoryService | None" = None
    approved_action: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class ToolExecutionResult:
    """Normalized tool execution output written into transcript history."""

    call_id: str
    name: str
    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolPolicyDecision:
    """Result of a policy check before tool execution."""

    allowed: bool
    reason: str | None = None
    approval_request: dict[str, Any] | None = None


class ToolExecutor(Protocol):
    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        """Execute one tool call."""


@dataclass(slots=True, frozen=True)
class RegisteredTool:
    """Registry entry for an executable tool."""

    name: str
    exposure: ToolExposure
    definition: ToolDefinition
    executor: ToolExecutor
    allowed_agent_kinds: tuple[AgentToolAccess, ...] = ("main", "subagent")


@dataclass(slots=True, frozen=True)
class DiscoverableTool:
    """Catalog entry exposed through tool_search."""

    name: str
    purpose: str
    aliases: tuple[str, ...] = ()
    detailed_description: str | None = None
    usage: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    backing_tool_name: str | None = None
    allowed_agent_kinds: tuple[AgentToolAccess, ...] = ("main", "subagent")

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("DiscoverableTool.name cannot be empty.")
        if not self.purpose.strip():
            raise ValueError("DiscoverableTool.purpose cannot be empty.")
