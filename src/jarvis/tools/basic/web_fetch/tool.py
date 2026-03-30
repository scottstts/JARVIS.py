"""Web-fetch tool definition and isolated-runtime execution."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import subprocess
from typing import Any

from jarvis.llm import ToolDefinition

from ...config import ToolSettings
from ...remote_runtime_client import RemoteToolRuntimeClient
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_DEFUDDLE_NODE_CA_CERTS_PATH = "/etc/ssl/certs/ca-certificates.crt"


class WebFetchConfigurationError(RuntimeError):
    """Raised when the isolated runtime is missing Defuddle prerequisites."""


class WebFetchRequestError(RuntimeError):
    """Raised when Defuddle fails to fetch or parse the requested URL."""


class WebFetchToolExecutor:
    """Routes web_fetch execution into the isolated tool_runtime container."""

    def __init__(self, settings: ToolSettings) -> None:
        self._remote_client = RemoteToolRuntimeClient(settings)

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        return await self._remote_client.execute(
            tool_name="web_fetch",
            call_id=call_id,
            arguments=arguments,
            context=context,
        )


class DirectWebFetchToolExecutor:
    """Runs Defuddle directly inside tool_runtime."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        requested_url = str(arguments.get("url", "")).strip()
        try:
            return await asyncio.to_thread(
                _run_defuddle_web_fetch,
                call_id=call_id,
                requested_url=requested_url,
                workspace_dir=context.workspace_dir,
                settings=self._settings,
            )
        except (WebFetchConfigurationError, WebFetchRequestError) as exc:
            return _web_fetch_error(
                call_id=call_id,
                requested_url=requested_url,
                reason=str(exc),
            )


def build_service_web_fetch_executor(settings: ToolSettings) -> DirectWebFetchToolExecutor:
    """Build the web_fetch executor used by the tool_runtime service."""

    return DirectWebFetchToolExecutor(settings)


def build_web_fetch_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the web_fetch registry entry."""

    return RegisteredTool(
        name="web_fetch",
        exposure="basic",
        definition=ToolDefinition(
            name="web_fetch",
            description=_build_web_fetch_tool_description(),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Absolute public http:// or https:// URL to fetch and return as "
                            "clean markdown."
                        ),
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        ),
        executor=WebFetchToolExecutor(settings),
    )


def _build_web_fetch_tool_description() -> str:
    return (
        "Fetch one specific public URL and return clean markdown. "
        "Use it when you already have the exact URL, including normal webpages and "
        "YouTube video and X posts. "
        "Treat this tool as the universal web content fetch tool."
    )


def _run_defuddle_web_fetch(
    *,
    call_id: str,
    requested_url: str,
    workspace_dir: Path,
    settings: ToolSettings,
) -> ToolExecutionResult:
    if not requested_url:
        return _web_fetch_error(
            call_id=call_id,
            requested_url=requested_url,
            reason="web_fetch requires a non-empty 'url'.",
        )

    try:
        result = subprocess.run(
            [
                "npx",
                "defuddle",
                "parse",
                requested_url,
                "--markdown",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            cwd=str(workspace_dir),
            timeout=settings.web_fetch_timeout_seconds,
            env=_build_defuddle_environment(),
        )
    except FileNotFoundError as exc:
        raise WebFetchConfigurationError(
            "npx is required inside tool_runtime for web_fetch."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise WebFetchRequestError(
            f"Defuddle request timed out after {settings.web_fetch_timeout_seconds} seconds."
        ) from exc
    except OSError as exc:
        raise WebFetchRequestError(f"Defuddle execution failed: {exc}") from exc

    if result.returncode != 0:
        error_details = (
            _normalize_optional_text(result.stderr)
            or _normalize_optional_text(result.stdout)
            or f"defuddle exited with status {result.returncode}."
        )
        raise WebFetchRequestError(error_details)

    markdown = _normalize_optional_text(result.stdout)
    if markdown is None:
        raise WebFetchRequestError("Defuddle returned empty markdown.")

    truncated_markdown, markdown_truncated = _truncate_markdown(
        markdown=markdown,
        limit=settings.web_fetch_max_markdown_chars,
    )
    lines = [
        "Web fetch result",
        f"url: {requested_url}",
        "provider: defuddle",
    ]
    if markdown_truncated:
        lines.append("markdown_truncated: true")
    lines.extend(
        [
            "markdown:",
            truncated_markdown,
        ]
    )
    return ToolExecutionResult(
        call_id=call_id,
        name="web_fetch",
        ok=True,
        content="\n".join(lines),
        metadata={
            "requested_url": requested_url,
            "provider": "defuddle",
            "markdown_chars": len(markdown),
            "markdown_truncated": markdown_truncated,
            "target_runtime": "tool_runtime",
            "runtime_location": "tool_runtime_container",
            "runtime_transport": "http",
        },
    )


def _web_fetch_error(
    *,
    call_id: str,
    requested_url: str,
    reason: str,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id=call_id,
        name="web_fetch",
        ok=False,
        content="\n".join(
            [
                "Web fetch failed",
                f"url: {requested_url}",
                f"reason: {reason}",
            ]
        ),
        metadata={
            "requested_url": requested_url,
            "error": reason,
            "provider": "defuddle",
        },
    )


def _build_defuddle_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["NODE_EXTRA_CA_CERTS"] = _DEFUDDLE_NODE_CA_CERTS_PATH
    return environment


def _normalize_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    return normalized or None


def _truncate_markdown(*, markdown: str, limit: int) -> tuple[str, bool]:
    if len(markdown) <= limit:
        return markdown, False
    return markdown[:limit].rstrip(), True
