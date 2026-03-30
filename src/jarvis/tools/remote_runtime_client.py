"""HTTP client for the isolated tool-runtime container."""

from __future__ import annotations

from typing import Any

import httpx

from .config import ToolSettings
from .types import ToolExecutionContext, ToolExecutionResult


class RemoteToolRuntimeError(RuntimeError):
    """Raised when the isolated tool runtime cannot be reached or parsed safely."""


class RemoteToolRuntimeClient:
    """Executes remote-capable tools inside the isolated tool-runtime container."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings
        self._base_url = settings.tool_runtime_base_url
        self._timeout_seconds = settings.tool_runtime_timeout_seconds
        self._healthcheck_timeout_seconds = settings.tool_runtime_healthcheck_timeout_seconds

    @property
    def enabled(self) -> bool:
        return self._base_url is not None

    async def healthcheck(self) -> dict[str, Any]:
        if not self.enabled:
            return {"configured": False}

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._healthcheck_timeout_seconds,
            ) as client:
                response = await client.get("/health")
        except httpx.HTTPError as exc:
            raise RemoteToolRuntimeError(
                f"tool_runtime healthcheck failed: {exc}"
            ) from exc

        if response.status_code != 200:
            raise RemoteToolRuntimeError(
                "tool_runtime healthcheck returned "
                f"HTTP {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteToolRuntimeError(
                "tool_runtime healthcheck returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise RemoteToolRuntimeError("tool_runtime healthcheck payload must be an object.")
        return payload

    async def execute(
        self,
        *,
        tool_name: str,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        if not self.enabled or self._base_url is None:
            raise RemoteToolRuntimeError("tool_runtime base URL is not configured.")

        payload: dict[str, Any] = {
            "call_id": call_id,
            "arguments": arguments,
            "workspace_dir": str(context.workspace_dir),
        }
        if context.session_id is not None:
            payload["session_id"] = context.session_id
        if context.route_id is not None:
            payload["route_id"] = context.route_id

        endpoint = f"/tools/{tool_name}/execute"
        request_timeout = _resolve_request_timeout_seconds(
            self._settings,
            tool_name=tool_name,
            arguments=arguments,
            base_timeout_seconds=self._timeout_seconds,
        )
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=request_timeout,
            ) as client:
                response = await client.post(endpoint, json=payload)
        except httpx.HTTPError as exc:
            raise RemoteToolRuntimeError(
                f"tool_runtime request failed for '{tool_name}': {exc}"
            ) from exc

        if response.status_code != 200:
            raise RemoteToolRuntimeError(
                "tool_runtime returned "
                f"HTTP {response.status_code} for '{tool_name}': {response.text}"
            )

        try:
            response_payload = response.json()
        except ValueError as exc:
            raise RemoteToolRuntimeError(
                f"tool_runtime returned invalid JSON for '{tool_name}'."
            ) from exc

        return _parse_execution_result(
            payload=response_payload,
            expected_tool_name=tool_name,
            expected_call_id=call_id,
        )


def _parse_execution_result(
    *,
    payload: object,
    expected_tool_name: str,
    expected_call_id: str,
) -> ToolExecutionResult:
    if not isinstance(payload, dict):
        raise RemoteToolRuntimeError("tool_runtime response payload must be an object.")

    call_id = payload.get("call_id")
    if not isinstance(call_id, str) or call_id != expected_call_id:
        raise RemoteToolRuntimeError(
            "tool_runtime response contained an unexpected call_id."
        )

    name = payload.get("name")
    if not isinstance(name, str) or name != expected_tool_name:
        raise RemoteToolRuntimeError(
            "tool_runtime response contained an unexpected tool name."
        )

    ok = payload.get("ok")
    if not isinstance(ok, bool):
        raise RemoteToolRuntimeError("tool_runtime response field 'ok' must be a boolean.")

    content = payload.get("content")
    if not isinstance(content, str):
        raise RemoteToolRuntimeError("tool_runtime response field 'content' must be a string.")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RemoteToolRuntimeError("tool_runtime response field 'metadata' must be an object.")
    return ToolExecutionResult(
        call_id=call_id,
        name=name,
        ok=ok,
        content=content,
        metadata=dict(metadata),
    )


def _resolve_effective_timeout(
    requested_timeout: object,
    *,
    default_timeout: float,
    max_timeout: float,
) -> float:
    if requested_timeout is None:
        return default_timeout

    effective_timeout = float(requested_timeout)
    if effective_timeout < 1:
        effective_timeout = 1.0
    if effective_timeout > max_timeout:
        effective_timeout = max_timeout
    return effective_timeout


def _resolve_request_timeout_seconds(
    settings: ToolSettings,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    base_timeout_seconds: float,
) -> float:
    requested_timeout = arguments.get("timeout_seconds")
    if tool_name == "bash":
        effective_timeout = _resolve_effective_timeout(
            requested_timeout,
            default_timeout=settings.bash_default_timeout_seconds,
            max_timeout=settings.bash_max_timeout_seconds,
        )
        return max(base_timeout_seconds, effective_timeout + 15.0)
    if tool_name == "web_fetch":
        return max(base_timeout_seconds, settings.web_fetch_timeout_seconds + 15.0)
    return base_timeout_seconds


async def ensure_remote_tool_runtime_healthy(settings: ToolSettings) -> None:
    """Fail fast when the app is configured to depend on the isolated runtime."""

    client = RemoteToolRuntimeClient(settings)
    if not client.enabled:
        return
    await client.healthcheck()
