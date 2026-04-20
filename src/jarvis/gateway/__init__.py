"""Gateway public API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import build_asgi_app, create_app
    from .config import GatewaySettings
    from .route_runtime import RouteRuntime
    from .session_router import SessionRouter

__all__ = [
    "GatewaySettings",
    "RouteRuntime",
    "SessionRouter",
    "build_asgi_app",
    "create_app",
]

_EXPORTS = {
    "GatewaySettings": (".config", "GatewaySettings"),
    "RouteRuntime": (".route_runtime", "RouteRuntime"),
    "SessionRouter": (".session_router", "SessionRouter"),
    "build_asgi_app": (".app", "build_asgi_app"),
    "create_app": (".app", "create_app"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
