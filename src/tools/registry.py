"""Registry of available tools and their exposure policies."""

from __future__ import annotations

from collections.abc import Iterable
import re

from llm import ToolDefinition

from .basic.bash import build_bash_tool
from .basic.file_patch import build_file_patch_tool
from .basic.python_interpreter import build_python_interpreter_tool
from .basic.send_file import build_send_file_tool
from .basic.tool_search import build_tool_search_tool
from .basic.web_fetch import build_web_fetch_tool
from .basic.web_search import build_web_search_tool
from .basic.view_image import build_view_image_tool
from .config import ToolSettings
from .discoverable.ffmpeg_cli import build_ffmpeg_cli_discoverable
from .discoverable.generate_edit_image import (
    build_generate_edit_image_discoverable,
    build_generate_edit_image_tool,
)
from .discoverable.transcribe import (
    build_transcribe_discoverable,
    build_transcribe_tool,
)
from .discoverable.youtube import (
    build_youtube_discoverable,
    build_youtube_tool,
)
from .types import DiscoverableTool, RegisteredTool

_SEARCH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


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
        registry.register(build_python_interpreter_tool(settings))
        registry.register(build_web_search_tool(settings))
        registry.register(build_web_fetch_tool(settings))
        registry.register(build_view_image_tool(settings))
        registry.register(build_send_file_tool(settings))
        registry.register(build_generate_edit_image_tool(settings))
        registry.register(build_transcribe_tool(settings))
        registry.register(build_youtube_tool(settings))
        registry.register(build_tool_search_tool(registry))
        registry.register_discoverable(build_ffmpeg_cli_discoverable())
        registry.register_discoverable(build_generate_edit_image_discoverable())
        registry.register_discoverable(build_transcribe_discoverable())
        registry.register_discoverable(build_youtube_discoverable())
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
        normalized_query = _normalize_search_text(query)
        if not normalized_query:
            return self.discoverable_entries()

        query_tokens = _tokenize_search_text(normalized_query)
        matches: list[tuple[int, DiscoverableTool]] = []
        for entry in self._discoverable_tools.values():
            score = _score_discoverable_match(
                entry=entry,
                normalized_query=normalized_query,
                query_tokens=query_tokens,
            )
            if score <= 0:
                continue
            matches.append((score, entry))

        matches.sort(key=lambda item: (-item[0], item[1].name))
        return tuple(entry for _, entry in matches)

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


def _normalize_search_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _tokenize_search_text(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_SEARCH_TOKEN_PATTERN.findall(value)))


def _score_discoverable_match(
    *,
    entry: DiscoverableTool,
    normalized_query: str,
    query_tokens: tuple[str, ...],
) -> int:
    normalized_name = _normalize_search_text(entry.name)
    normalized_aliases = tuple(_normalize_search_text(alias) for alias in entry.aliases if alias.strip())
    normalized_purpose = _normalize_search_text(entry.purpose)
    normalized_description = _normalize_search_text(entry.detailed_description or "")
    combined_text = " ".join(
        part
        for part in (
            normalized_name,
            *normalized_aliases,
            normalized_purpose,
            normalized_description,
        )
        if part
    )

    if not combined_text:
        return 0

    score = 0
    if normalized_query == normalized_name:
        score += 200
    if normalized_query in normalized_name:
        score += 120
        if normalized_name.startswith(normalized_query):
            score += 20

    exact_alias_match = any(normalized_query == alias for alias in normalized_aliases)
    if exact_alias_match:
        score += 180

    alias_substring_match = any(normalized_query in alias for alias in normalized_aliases)
    if alias_substring_match:
        score += 110

    if normalized_query in normalized_purpose:
        score += 80
    if normalized_query in normalized_description:
        score += 40

    if query_tokens:
        token_hits = sum(1 for token in query_tokens if token in combined_text)
        if token_hits == 0:
            return 0
        score += token_hits * 15
        if token_hits == len(query_tokens):
            score += 25
    elif normalized_query not in combined_text:
        return 0

    return score
