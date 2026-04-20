"""Web-search tool package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .policy import WebSearchPolicy
    from .tool import build_web_search_tool

__all__ = ["WebSearchPolicy", "build_web_search_tool"]


def __getattr__(name: str) -> object:
    if name == "WebSearchPolicy":
        from .policy import WebSearchPolicy

        return WebSearchPolicy
    if name == "build_web_search_tool":
        from .tool import build_web_search_tool

        return build_web_search_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
