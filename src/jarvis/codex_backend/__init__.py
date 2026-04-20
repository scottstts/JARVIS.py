"""Codex app-server backend package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .actor_runtime import CodexActorRuntime
    from .config import CodexBackendSettings, CodexConfigurationError
    from .runtime import CodexRouteCoordinator
    from .types import CodexBackendError, CodexConnectionError

__all__ = [
    "CodexActorRuntime",
    "CodexBackendError",
    "CodexBackendSettings",
    "CodexConfigurationError",
    "CodexConnectionError",
    "CodexRouteCoordinator",
]

_EXPORTS = {
    "CodexActorRuntime": (".actor_runtime", "CodexActorRuntime"),
    "CodexBackendError": (".types", "CodexBackendError"),
    "CodexBackendSettings": (".config", "CodexBackendSettings"),
    "CodexConfigurationError": (".config", "CodexConfigurationError"),
    "CodexConnectionError": (".types", "CodexConnectionError"),
    "CodexRouteCoordinator": (".runtime", "CodexRouteCoordinator"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
