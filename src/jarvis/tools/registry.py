"""Registry of available tools and their exposure policies."""

from __future__ import annotations

from collections.abc import Iterable

from jarvis.llm import ToolDefinition
from jarvis.skills import SkillsSettings

from .basic.bash import build_bash_tool
from .basic.file_patch import build_file_patch_tool
from .basic.get_skills import build_get_skills_tool
from .basic.memory_get import build_memory_get_tool
from .basic.memory_search import build_memory_search_tool
from .basic.memory_write import build_memory_write_tool
from .basic.send_file import build_send_file_tool
from .basic.tool_register import build_tool_register_tool
from .basic.tool_search import build_tool_search_tool
from .basic.web_fetch import build_web_fetch_tool
from .basic.web_search import build_web_search_tool
from .basic.view_image import build_view_image_tool
from .config import ToolSettings
from .discoverable_search import search_discoverable_entries
from .discoverable.email import build_email_discoverable, build_email_tool
from .discoverable.ffmpeg import build_ffmpeg_discoverable
from .discoverable.generate_edit_image import (
    build_generate_edit_image_discoverable,
    build_generate_edit_image_tool,
)
from .discoverable.memory_admin import (
    build_memory_admin_discoverable,
    build_memory_admin_tool,
)
from .discoverable.transcribe import (
    build_transcribe_discoverable,
    build_transcribe_tool,
)
from .types import AgentToolAccess, DiscoverableTool, RegisteredTool


def _tool_visible_to_agent(
    tool: RegisteredTool,
    *,
    agent_kind: AgentToolAccess | None,
    hidden_tool_names: frozenset[str],
) -> bool:
    if tool.name in hidden_tool_names:
        return False
    if agent_kind is None:
        return True
    return agent_kind in tool.allowed_agent_kinds


def _discoverable_visible_to_agent(
    tool: DiscoverableTool,
    *,
    agent_kind: AgentToolAccess | None,
    hidden_tool_names: frozenset[str],
    visible_tool_names: frozenset[str],
) -> bool:
    if tool.name in hidden_tool_names:
        return False
    if agent_kind is not None and agent_kind not in tool.allowed_agent_kinds:
        return False
    backing_tool_name = tool.backing_tool_name
    if backing_tool_name is None:
        return True
    return backing_tool_name in visible_tool_names


class ToolRegistryView:
    """Filtered registry facade used for agent-scoped tool visibility."""

    def __init__(
        self,
        registry: "ToolRegistry",
        *,
        agent_kind: AgentToolAccess | None = None,
        hidden_tool_names: Iterable[str] = (),
    ) -> None:
        self._registry = registry
        self._agent_kind = agent_kind
        self._hidden_tool_names = frozenset(
            normalized
            for raw_name in hidden_tool_names
            if (normalized := str(raw_name).strip())
        )

    def get(self, name: str) -> RegisteredTool | None:
        tool = self._registry.get(name)
        if tool is None:
            return None
        if not _tool_visible_to_agent(
            tool,
            agent_kind=self._agent_kind,
            hidden_tool_names=self._hidden_tool_names,
        ):
            return None
        if name == "tool_search":
            return build_tool_search_tool(self)
        if name == "tool_register":
            return build_tool_register_tool(self)
        return tool

    def registered_tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._visible_tools()))

    def require(self, name: str) -> RegisteredTool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return tool

    def basic_definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            tool.definition
            for tool in self._visible_tools().values()
            if tool.exposure == "basic"
        )

    def discoverable_entries(self) -> tuple[DiscoverableTool, ...]:
        visible_tool_names = frozenset(self._visible_tools())
        return tuple(
            tool
            for name, tool in sorted(self._registry._discoverable_tools.items())
            if _discoverable_visible_to_agent(
                tool,
                agent_kind=self._agent_kind,
                hidden_tool_names=self._hidden_tool_names,
                visible_tool_names=visible_tool_names,
            )
        )

    def get_discoverable(self, name: str) -> DiscoverableTool | None:
        visible_tool_names = frozenset(self._visible_tools())
        tool = self._registry.get_discoverable(name)
        if tool is None:
            return None
        if not _discoverable_visible_to_agent(
            tool,
            agent_kind=self._agent_kind,
            hidden_tool_names=self._hidden_tool_names,
            visible_tool_names=visible_tool_names,
        ):
            return None
        return tool

    def search(self, query: str, *, include_basic: bool = False) -> tuple[RegisteredTool, ...]:
        normalized = query.strip().lower()
        visible_tools = tuple(self._visible_tools().values())
        if not normalized:
            matches = visible_tools
        else:
            matches = tuple(
                tool
                for tool in visible_tools
                if normalized in tool.name.lower()
                or normalized in (tool.definition.description or "").lower()
            )
        if include_basic:
            return matches
        return tuple(tool for tool in matches if tool.exposure != "basic")

    def search_discoverable(self, query: str) -> tuple[DiscoverableTool, ...]:
        return search_discoverable_entries(self.discoverable_entries(), query)

    def resolve_discoverable_tool_definitions(
        self,
        names: Iterable[str],
    ) -> tuple[ToolDefinition, ...]:
        definitions: list[ToolDefinition] = []
        seen_backing_tools: set[str] = set()
        for name in names:
            entry = self.get_discoverable(str(name).strip())
            if entry is None or entry.backing_tool_name is None:
                continue
            backing_tool_name = entry.backing_tool_name
            if backing_tool_name in seen_backing_tools:
                continue
            registered = self.require(backing_tool_name)
            definitions.append(registered.definition)
            seen_backing_tools.add(backing_tool_name)
        return tuple(definitions)

    def _visible_tools(self) -> dict[str, RegisteredTool]:
        visible: dict[str, RegisteredTool] = {}
        for name, tool in self._registry._tools.items():
            if not _tool_visible_to_agent(
                tool,
                agent_kind=self._agent_kind,
                hidden_tool_names=self._hidden_tool_names,
            ):
                continue
            resolved = self.get(name)
            if resolved is not None:
                visible[name] = resolved
        return visible


class ToolRegistry:
    """Stores registered tools and exposes them by visibility class."""

    def __init__(
        self,
        tools: Iterable[RegisteredTool] | None = None,
        discoverable_tools: Iterable[DiscoverableTool] | None = None,
    ) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self._discoverable_tools: dict[str, DiscoverableTool] = {}
        if tools is not None:
            for tool in tools:
                self.register(tool)
        if discoverable_tools is not None:
            for discoverable_tool in discoverable_tools:
                self.register_discoverable(discoverable_tool)

    @classmethod
    def default(cls, settings: ToolSettings) -> "ToolRegistry":
        registry = cls()
        registry.register(build_bash_tool(settings))
        registry.register(build_file_patch_tool(settings))
        registry.register(build_memory_search_tool())
        registry.register(build_memory_get_tool())
        registry.register(build_memory_write_tool())
        registry.register(
            build_get_skills_tool(SkillsSettings.from_workspace_dir(settings.workspace_dir))
        )
        registry.register(build_web_search_tool(settings))
        registry.register(build_web_fetch_tool(settings))
        registry.register(build_view_image_tool(settings))
        registry.register(build_send_file_tool(settings))
        registry.register(build_email_tool(settings))
        registry.register(build_generate_edit_image_tool(settings))
        registry.register(build_memory_admin_tool())
        registry.register(build_transcribe_tool(settings))
        registry.register(build_tool_search_tool(registry))
        registry.register(build_tool_register_tool(registry))
        registry.register_discoverable(build_email_discoverable())
        registry.register_discoverable(build_ffmpeg_discoverable())
        registry.register_discoverable(build_generate_edit_image_discoverable())
        registry.register_discoverable(build_memory_admin_discoverable())
        registry.register_discoverable(build_transcribe_discoverable())
        return registry

    def register(self, tool: RegisteredTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def register_discoverable(self, tool: DiscoverableTool) -> None:
        if tool.name in self._discoverable_tools:
            raise ValueError(f"Discoverable tool '{tool.name}' is already registered.")
        backing_tool_name = tool.backing_tool_name
        if backing_tool_name is not None and self.get(backing_tool_name) is None:
            raise KeyError(
                "Discoverable tool "
                f"'{tool.name}' references unknown backing tool '{backing_tool_name}'."
            )
        self._discoverable_tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def registered_tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def require(self, name: str) -> RegisteredTool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return tool

    def filtered_view(
        self,
        *,
        agent_kind: AgentToolAccess | None = None,
        hidden_tool_names: Iterable[str] = (),
    ) -> ToolRegistryView:
        return ToolRegistryView(
            self,
            agent_kind=agent_kind,
            hidden_tool_names=hidden_tool_names,
        )

    def basic_definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            tool.definition
            for tool in self._tools.values()
            if tool.exposure == "basic"
        )

    def discoverable_entries(self) -> tuple[DiscoverableTool, ...]:
        return tuple(
            self._discoverable_tools[name]
            for name in sorted(self._discoverable_tools)
        )

    def get_discoverable(self, name: str) -> DiscoverableTool | None:
        return self._discoverable_tools.get(name)

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

    def search_discoverable(self, query: str) -> tuple[DiscoverableTool, ...]:
        return search_discoverable_entries(self._discoverable_tools.values(), query)

    def resolve_discoverable_tool_definitions(
        self,
        names: Iterable[str],
    ) -> tuple[ToolDefinition, ...]:
        definitions: list[ToolDefinition] = []
        seen_backing_tools: set[str] = set()
        for name in names:
            entry = self.get_discoverable(str(name).strip())
            if entry is None or entry.backing_tool_name is None:
                continue
            backing_tool_name = entry.backing_tool_name
            if backing_tool_name in seen_backing_tools:
                continue
            registered = self.require(backing_tool_name)
            definitions.append(registered.definition)
            seen_backing_tools.add(backing_tool_name)
        return tuple(definitions)
