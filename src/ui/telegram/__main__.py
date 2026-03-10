"""Telegram UI process entrypoint."""

from __future__ import annotations

import asyncio

from logging_setup import configure_application_logging
from .bot import run_telegram_ui


def main() -> None:
    configure_application_logging()
    asyncio.run(run_telegram_ui())


if __name__ == "__main__":
    main()
