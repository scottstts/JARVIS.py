"""Email discoverable tool package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .policy import EmailPolicy
    from .tool import build_email_discoverable, build_email_tool

__all__ = ["EmailPolicy", "build_email_discoverable", "build_email_tool"]


def __getattr__(name: str) -> object:
    if name == "EmailPolicy":
        from .policy import EmailPolicy

        return EmailPolicy
    if name == "build_email_discoverable":
        from .tool import build_email_discoverable

        return build_email_discoverable
    if name == "build_email_tool":
        from .tool import build_email_tool

        return build_email_tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
