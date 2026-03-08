"""Web-search tool package."""

from .policy import WebSearchPolicy
from .tool import build_web_search_tool

__all__ = [
    "WebSearchPolicy",
    "build_web_search_tool",
]
