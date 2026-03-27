"""Process entrypoint for the isolated tool-runtime service."""

from __future__ import annotations

import os

import uvicorn

from jarvis import settings as app_settings


def main() -> None:
    host = os.getenv("JARVIS_TOOL_RUNTIME_SERVICE_HOST", app_settings.JARVIS_TOOL_RUNTIME_SERVICE_HOST)
    port = int(
        os.getenv(
            "JARVIS_TOOL_RUNTIME_SERVICE_PORT",
            str(app_settings.JARVIS_TOOL_RUNTIME_SERVICE_PORT),
        )
    )
    uvicorn.run(
        "jarvis.tool_runtime_service.app:build_asgi_app",
        factory=True,
        host=host,
        port=port,
        access_log=False,
    )


if __name__ == "__main__":
    main()
