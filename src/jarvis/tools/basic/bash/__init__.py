"""Bash tool package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .policy import BashCommandPolicy
    from .tool import build_bash_tool

__all__ = ["BashCommandPolicy", "build_bash_tool"]


def __getattr__(name: str) -> object:
    if name == "BashCommandPolicy":
        from .policy import BashCommandPolicy

        return BashCommandPolicy
    if name == "build_bash_tool":
        from .tool import build_bash_tool

        return build_bash_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
