"""Shared compatibility contract for the isolated tool-runtime service."""

from __future__ import annotations

from typing import Any


TOOL_RUNTIME_PROTOCOL_VERSION = 1
TOOL_RUNTIME_CAPABILITIES: dict[str, dict[str, tuple[str, ...]]] = {
    "bash": {
        "modes": (
            "foreground",
            "background",
            "service",
            "status",
            "tail",
            "cancel",
        ),
    },
    "web_fetch": {},
}


def tool_runtime_capabilities_payload() -> dict[str, dict[str, list[str]]]:
    """Return a JSON-safe copy of the tool-runtime capabilities."""

    return {
        tool_name: {key: list(values) for key, values in capability.items()}
        for tool_name, capability in TOOL_RUNTIME_CAPABILITIES.items()
    }


def validate_tool_runtime_health_payload(payload: dict[str, Any]) -> str | None:
    """Return an actionable incompatibility description, or ``None`` when compatible."""

    version = payload.get("protocol_version")
    if version != TOOL_RUNTIME_PROTOCOL_VERSION:
        return (
            "protocol version mismatch "
            f"(expected {TOOL_RUNTIME_PROTOCOL_VERSION}, received {version!r})"
        )
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        return "health payload does not declare tool capabilities"
    for tool_name, expected in TOOL_RUNTIME_CAPABILITIES.items():
        actual = capabilities.get(tool_name)
        if not isinstance(actual, dict):
            return f"required tool capability {tool_name!r} is missing"
        for key, values in expected.items():
            actual_values = actual.get(key)
            if not isinstance(actual_values, list):
                return f"{tool_name}.{key} capability is missing"
            missing = sorted(set(values) - {str(value) for value in actual_values})
            if missing:
                return f"{tool_name}.{key} is missing required values: {', '.join(missing)}"
    return None
