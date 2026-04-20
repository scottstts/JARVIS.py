"""Process entrypoint for the isolated tool-runtime service."""

from __future__ import annotations

import os

import uvicorn


_DEFAULT_SERVICE_HOST = "0.0.0.0"
_DEFAULT_SERVICE_PORT = 8081


def main() -> None:
    host = os.getenv("JARVIS_TOOL_RUNTIME_SERVICE_HOST", _DEFAULT_SERVICE_HOST)
    port = int(
        os.getenv(
            "JARVIS_TOOL_RUNTIME_SERVICE_PORT",
            str(_DEFAULT_SERVICE_PORT),
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
