"""Registry of available tools and their exposure policies."""

from __future__ import annotations

from collections.abc import Iterable

from llm import ToolDefinition

from .bash import build_bash_tool
from .config import ToolSettings
from .send_file import build_send_file_tool
from .types import RegisteredTool
from .web_search import build_web_search_tool
from .view_image import build_view_image_tool


class ToolRegistry:
    """Stores registered tools and exposes them by visibility class."""

    def __init__(self, tools: Iterable[RegisteredTool] | None = None) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        if tools is not None:
            for tool in tools:
                self.register(tool)

    @classmethod
    def default(cls, settings: ToolSettings) -> "ToolRegistry":
        return cls(
            tools=(
                build_bash_tool(settings),
                build_web_search_tool(settings),
                build_view_image_tool(settings),
                build_send_file_tool(settings),
            )
        )

    def register(self, tool: RegisteredTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def require(self, name: str) -> RegisteredTool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return tool

    def basic_definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            tool.definition
            for tool in self._tools.values()
            if tool.exposure == "basic"
        )

    def search(self, query: str, *, include_basic: bool = False) -> tuple[RegisteredTool, ...]:
        normalized = query.strip().lower()
        if not normalized:
            matches = tuple(self._tools.values())
        else:
            matches = tuple(
                tool
                for tool in self._tools.values()
                if normalized in tool.name.lower()
                or normalized in (tool.definition.description or "").lower()
            )
        if include_basic:
            return matches
        return tuple(tool for tool in matches if tool.exposure != "basic")
