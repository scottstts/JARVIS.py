"""Shared types for the tools layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from llm import ToolDefinition

ToolExposure = Literal["basic", "discoverable"]


@dataclass(slots=True, frozen=True)
class ToolExecutionContext:
    """Runtime context passed to tool executors and policy checks."""

    workspace_dir: Path
    route_id: str | None = None


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
