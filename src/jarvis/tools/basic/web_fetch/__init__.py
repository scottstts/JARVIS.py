"""Web-fetch tool package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .policy import WebFetchPolicy
    from .tool import build_web_fetch_tool

__all__ = ["WebFetchPolicy", "build_web_fetch_tool"]


def __getattr__(name: str) -> object:
    if name == "WebFetchPolicy":
        from .policy import WebFetchPolicy

        return WebFetchPolicy
    if name == "build_web_fetch_tool":
        from .tool import build_web_fetch_tool

        return build_web_fetch_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
