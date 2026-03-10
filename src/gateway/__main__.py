"""Gateway process entrypoint."""

from __future__ import annotations

import uvicorn

from .config import GatewaySettings


def main() -> None:
    settings = GatewaySettings.from_env()
    uvicorn.run(
        "gateway.app:build_asgi_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
