"""Telegram UI process entrypoint."""

from __future__ import annotations

import asyncio
import logging

from .bot import run_telegram_ui


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_telegram_ui())


if __name__ == "__main__":
    main()
