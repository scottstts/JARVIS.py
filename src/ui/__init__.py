"""UI package.

Telegram currently lives under `ui.telegram`. Top-level re-exports remain as
compatibility shims for existing callers.
"""

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
