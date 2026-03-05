"""UI package.

Telegram currently lives under `ui.telegram`. Top-level re-exports remain as
compatibility shims for existing callers.
"""

from .telegram import TelegramGatewayBridge, UISettings, run_telegram_ui

__all__ = [
    "TelegramGatewayBridge",
    "UISettings",
    "run_telegram_ui",
]
