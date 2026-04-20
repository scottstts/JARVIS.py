"""UI package.

Telegram currently lives under `ui.telegram`. Top-level re-exports remain as
compatibility shims for existing callers.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .telegram import (
        TelegramGatewayBridge,
        UISettings,
        chat_id_for_route_id,
        run_telegram_ui,
        send_owner_telegram_file,
        send_telegram_file,
    )

__all__ = [
    "TelegramGatewayBridge",
    "UISettings",
    "chat_id_for_route_id",
    "run_telegram_ui",
    "send_owner_telegram_file",
    "send_telegram_file",
]

_EXPORTS = {
    "TelegramGatewayBridge": (".telegram", "TelegramGatewayBridge"),
    "UISettings": (".telegram", "UISettings"),
    "chat_id_for_route_id": (".telegram", "chat_id_for_route_id"),
    "run_telegram_ui": (".telegram", "run_telegram_ui"),
    "send_owner_telegram_file": (".telegram", "send_owner_telegram_file"),
    "send_telegram_file": (".telegram", "send_telegram_file"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
