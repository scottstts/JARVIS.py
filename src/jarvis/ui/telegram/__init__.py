"""Telegram UI integration package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bot import (
        TelegramGatewayBridge,
        chat_id_for_route_id,
        run_telegram_ui,
        send_owner_telegram_file,
        send_telegram_file,
    )
    from .config import UIConfigurationError, UISettings

__all__ = [
    "TelegramGatewayBridge",
    "UIConfigurationError",
    "UISettings",
    "chat_id_for_route_id",
    "run_telegram_ui",
    "send_owner_telegram_file",
    "send_telegram_file",
]

_EXPORTS = {
    "TelegramGatewayBridge": (".bot", "TelegramGatewayBridge"),
    "UIConfigurationError": (".config", "UIConfigurationError"),
    "UISettings": (".config", "UISettings"),
    "chat_id_for_route_id": (".bot", "chat_id_for_route_id"),
    "run_telegram_ui": (".bot", "run_telegram_ui"),
    "send_owner_telegram_file": (".bot", "send_owner_telegram_file"),
    "send_telegram_file": (".bot", "send_telegram_file"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
