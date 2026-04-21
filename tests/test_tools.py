"""Unit tests for tool registry, policy, and runtime behavior."""

from __future__ import annotations

import asyncio
from base64 import b64decode, b64encode
from dataclasses import dataclass
from email import policy as email_parser_policy
from email.parser import BytesParser
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from jsonschema import Draft202012Validator
from jarvis.llm import ToolCall, ToolDefinition
from jarvis.memory import MemoryService, MemorySettings
from jarvis.tools import (
    DiscoverableTool,
    RegisteredTool,
    ToolExecutionResult,
    ToolExecutionContext,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
    ToolSettings,
)
from jarvis.tools.remote_runtime_client import RemoteToolRuntimeClient, RemoteToolRuntimeError
from jarvis.tools.basic.memory_write.tool import build_memory_write_tool
from jarvis.tools.basic.tool_search import build_tool_search_tool
from jarvis.tools.basic.tool_register.tool import build_tool_register_tool
from jarvis.tools.basic.memory_search.tool import _format_memory_search_result
from jarvis.tools.discoverable.memory_admin.tool import build_memory_admin_discoverable
from jarvis.tools.runtime_tool_manifest import (
    dump_runtime_tool_manifest,
    runtime_tool_manifest_path,
    validate_runtime_tool_manifest_payload,
)
from jarvis.tools.runtime_tools import load_runtime_tool_catalog
from jarvis.tools.basic.bash.local_executor import (
    _build_scrubbed_environment,
    DirectBashToolExecutor,
    format_bash_tool_description,
)
from jarvis.tools.basic.bash.jobs import load_job, sweep_job_artifacts
from jarvis.tools.basic.web_fetch.tool import (
    BrowserRenderResult,
    DirectWebFetchToolExecutor,
    HTTPFetchResult,
    MarkdownConversionResult,
    WebFetchRequestError,
    _markdown_is_usable,
    _render_page_html,
    _validate_browser_request_url,
)

_REMOTE_TOOL_RUNTIME_CONFIGURED = bool(os.getenv("JARVIS_TOOL_RUNTIME_BASE_URL"))
_CENTRAL_PYTHON_INTERPRETER = Path("/opt/venv/bin/python")
_BASH_RUNTIME_AVAILABLE = (
    _REMOTE_TOOL_RUNTIME_CONFIGURED
    or Path("/bin/bash").exists()
    or shutil.which("bash") is not None
)
_FFMPEG_BASH_RUNTIME_AVAILABLE = (
    _REMOTE_TOOL_RUNTIME_CONFIGURED
    or (_BASH_RUNTIME_AVAILABLE and shutil.which("ffmpeg") is not None)
)
_CENTRAL_PYTHON_RUNTIME_AVAILABLE = (
    _REMOTE_TOOL_RUNTIME_CONFIGURED
    or _CENTRAL_PYTHON_INTERPRETER.exists()
)
_BASH_RUNTIME_SKIP_REASON = (
    "bash runtime tests require a local bash binary or configured remote tool_runtime"
)
_REMOTE_TOOL_RUNTIME_SKIP_REASON = (
    "remote tool_runtime integration is only configured in jarvis_runtime"
)
_FFMPEG_BASH_RUNTIME_SKIP_REASON = (
    "ffmpeg bash runtime tests require local ffmpeg or configured remote tool_runtime"
)
_SAMPLE_JPEG_BASE64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a"
    "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAADAAIDASIA"
    "AhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQA"
    "AAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3O"
    "Dk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6"
    "ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEB"
    "AQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJB"
    "UQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVV"
    "ldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6ws"
    "PExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDxGiiitjI//9k="
)


def _bash_python_heredoc(code: str) -> str:
    normalized = code.rstrip("\n")
    return f"python - <<'PY'\n{normalized}\nPY"


def _build_web_fetch_tool_result(
    *,
    requested_url: str,
    markdown: str,
    markdown_truncated: bool = False,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id="unused",
        name="web_fetch",
        ok=True,
        content="\n".join(
            [
                "Web fetch result",
                f"url: {requested_url}",
                "markdown:",
                markdown,
            ]
        ),
        metadata={
            "requested_url": requested_url,
            "markdown_chars": len(markdown),
            "markdown_truncated": markdown_truncated,
        },
    )


def _build_http_fetch_result(
    *,
    requested_url: str,
    body_text: str,
    content_type: str,
    status_code: int = 200,
    final_url: str | None = None,
    headers: dict[str, str] | None = None,
) -> HTTPFetchResult:
    return HTTPFetchResult(
        requested_url=requested_url,
        final_url=final_url or requested_url,
        status_code=status_code,
        headers=headers or {},
        content_type=content_type,
        body_text=body_text,
        redirect_chain=(),
    )


class _FakeWebSearchResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: dict[str, object],
        headers: dict[str, str] | None = None,
        text: str | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = text or ""

    def json(self) -> dict[str, object]:
        return self._payload


@dataclass(slots=True)
class _FakeBrowserRequest:
    url: str


class _FakeBrowserRoute:
    def __init__(self, url: str) -> None:
        self.request = _FakeBrowserRequest(url=url)
        self.aborted = False
        self.abort_error_code: str | None = None
        self.continued = False

    async def abort(self, error_code: str | None = None) -> None:
        self.aborted = True
        self.abort_error_code = error_code

    async def continue_(self) -> None:
        self.continued = True


class _FakeBrowserPage:
    def __init__(
        self,
        *,
        final_url: str,
        html: str,
        request_urls: tuple[str, ...] = (),
    ) -> None:
        self.url = final_url
        self._html = html
        self._request_urls = request_urls
        self._route_handler = None
        self.handled_routes: list[_FakeBrowserRoute] = []

    async def route(self, pattern: str, handler) -> None:
        _ = pattern
        self._route_handler = handler

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> None:
        _ = url, wait_until, timeout
        if self._route_handler is None:
            return
        for request_url in self._request_urls:
            route = _FakeBrowserRoute(request_url)
            self.handled_routes.append(route)
            await self._route_handler(route)

    async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
        _ = state, timeout

    async def content(self) -> str:
        return self._html


class _FakeBrowserContext:
    def __init__(self, page: _FakeBrowserPage) -> None:
        self._page = page
        self.closed = False

    async def new_page(self) -> _FakeBrowserPage:
        return self._page

    async def close(self) -> None:
        self.closed = True


class _FakeChromiumBrowser:
    def __init__(self, page: _FakeBrowserPage) -> None:
        self._page = page
        self.context_kwargs: list[dict[str, object]] = []
        self.closed = False

    async def new_context(self, **kwargs: object) -> _FakeBrowserContext:
        self.context_kwargs.append(kwargs)
        return _FakeBrowserContext(self._page)

    async def close(self) -> None:
        self.closed = True


class _FakeChromium:
    def __init__(self, browser: _FakeChromiumBrowser) -> None:
        self._browser = browser

    async def launch(self, *, headless: bool, args: list[str]) -> _FakeChromiumBrowser:
        _ = headless, args
        return self._browser


class _FakePlaywrightManager:
    def __init__(self, page: _FakeBrowserPage) -> None:
        self.browser = _FakeChromiumBrowser(page)
        self.chromium = _FakeChromium(self.browser)

    async def __aenter__(self) -> "_FakePlaywrightManager":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        _ = exc_type, exc, tb


class _FakeAsyncHTTPResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: dict[str, object] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict[str, object]:
        if self._payload is None:
            raise ValueError("No JSON payload configured.")
        return self._payload


def _write_runtime_manifest(workspace_dir: Path, payload: dict[str, object]) -> Path:
    manifest = validate_runtime_tool_manifest_payload(payload)
    path = runtime_tool_manifest_path(workspace_dir, manifest.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_runtime_tool_manifest(manifest), encoding="utf-8")
    return path


class ToolSettingsTests(unittest.TestCase):
    def test_requires_agent_workspace_for_host_runs(self) -> None:
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ), patch("jarvis.workspace_paths._running_in_container", return_value=False):
            with self.assertRaisesRegex(
                ValueError,
                "AGENT_WORKSPACE must be explicitly set for host runs",
            ):
                ToolSettings.from_env()

    def test_uses_explicit_agent_workspace_for_host_runs(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENT_WORKSPACE": "/tmp/jarvis-host-workspace",
            },
            clear=True,
        ), patch("jarvis.workspace_paths._running_in_container", return_value=False):
            settings = ToolSettings.from_env()

        self.assertEqual(settings.workspace_dir, Path("/tmp/jarvis-host-workspace"))

    def test_uses_default_web_search_count_from_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(settings.web_search_result_count, 10)

    def test_allows_web_search_count_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                os.environ,
                {"JARVIS_TOOL_WEB_SEARCH_RESULT_COUNT": "7"},
                clear=False,
            ):
                settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(settings.web_search_result_count, 7)

    def test_allows_bash_permission_skip_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                os.environ,
                {"BASH_DANGEROUSLY_SKIP_PERMISSION": "true"},
                clear=False,
            ):
                settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertTrue(settings.bash_dangerously_skip_permission)

    def test_memory_write_tool_describes_superseding_rewrite_and_truth_contracts(self) -> None:
        tool = build_memory_write_tool()

        self.assertIn("upsert revises an existing canonical document", tool.definition.description.lower())
        self.assertIn("do not leave wrong active memory in place", tool.definition.description.lower())
        self.assertIn("daily corrections are section rewrites", tool.definition.description.lower())
        self.assertIn("memory_get", tool.definition.description.lower())
        self.assertIn("use append_daily only to add a new daily entry", tool.definition.description.lower())
        self.assertIn("rewrite the terminal summary and body_sections", tool.definition.description.lower())
        self.assertIn("facts and relations are explicit-decision fields", tool.definition.description.lower())
        self.assertIn('literal string "none"', tool.definition.description.lower())
        self.assertIn("structured subject-predicate-object claims", tool.definition.description.lower())
        self.assertIn("important narrative text in body_sections", tool.definition.description.lower())
        self.assertIn("subject-predicate-object", tool.definition.description.lower())
        operation_description = tool.definition.input_schema["properties"]["operation"]["description"].lower()
        summary_description = tool.definition.input_schema["properties"]["summary"]["description"].lower()
        facts_description = tool.definition.input_schema["properties"]["facts"]["description"].lower()
        relations_description = tool.definition.input_schema["properties"]["relations"]["description"].lower()
        body_description = tool.definition.input_schema["properties"]["body_sections"]["description"].lower()
        close_reason_description = tool.definition.input_schema["properties"]["close_reason"]["description"].lower()

        self.assertIn("close and archive are superseding transitions", operation_description)
        self.assertIn("upsert revises an existing canonical document", operation_description)
        self.assertIn("append_daily appends a new daily entry", operation_description)
        self.assertIn("rewrite terminal summary/body_sections first", operation_description)
        self.assertIn("summary is not a substitute", summary_description)
        self.assertIn("structured truth", summary_description)
        self.assertIn("mirror it into facts", summary_description)
        self.assertIn("durable fact objects", facts_description)
        self.assertIn("replaces the document's fact set", facts_description)
        self.assertIn("existing facts stay", facts_description)
        self.assertIn('{"text":"..."}', facts_description)
        self.assertIn("subject-predicate-object claims", relations_description)
        self.assertIn("preferences", relations_description)
        self.assertIn("replaces the document's relation set", relations_description)
        self.assertIn("existing relations stay", relations_description)
        self.assertIn("replace the active truth", relations_description)
        self.assertIn('{"subject":"...","predicate":"...","object":"..."}', relations_description)
        self.assertIn("main searchable body text", body_description)
        self.assertIn("overwrite matching canonical sections", body_description)
        self.assertIn("omitted sections stay unchanged", body_description)
        self.assertIn("include only sections you want to rewrite", body_description)
        self.assertIn("do not send blank placeholders", body_description)
        self.assertIn("rewrite stale narrative content", body_description)
        self.assertIn("daily corrections", body_description)
        self.assertIn("memory_get", body_description)
        self.assertIn("not a list of {heading, body} objects", body_description)
        self.assertIn('{"overview":"..."}', body_description)
        self.assertIn('{"notable events":"- went for a bike ride along the coast."}', body_description)
        self.assertIn("optional close/archive reason metadata", close_reason_description)

    def test_memory_write_schema_exposes_nested_truth_shapes(self) -> None:
        tool = build_memory_write_tool()
        facts_schema = tool.definition.input_schema["properties"]["facts"]["anyOf"][0]
        relations_schema = tool.definition.input_schema["properties"]["relations"]["anyOf"][0]
        body_sections_schema = tool.definition.input_schema["properties"]["body_sections"]

        self.assertEqual(facts_schema["type"], "array")
        self.assertEqual(facts_schema["minItems"], 1)
        self.assertEqual(facts_schema["items"]["required"], ["text"])
        self.assertEqual(
            facts_schema["items"]["properties"]["status"]["enum"],
            ["current", "past", "uncertain", "superseded"],
        )
        self.assertEqual(relations_schema["type"], "array")
        self.assertEqual(relations_schema["minItems"], 1)
        self.assertEqual(relations_schema["items"]["required"], ["subject", "predicate", "object"])
        self.assertEqual(
            relations_schema["items"]["properties"]["cardinality"]["enum"],
            ["single", "multi"],
        )
        self.assertEqual(body_sections_schema["type"], "object")
        self.assertEqual(body_sections_schema["additionalProperties"]["type"], "string")

    def test_memory_write_schema_rejects_string_items_in_truth_arrays(self) -> None:
        tool = build_memory_write_tool()
        arguments = {
            "operation": "create",
            "target_kind": "ongoing",
            "title": "Visual Pipeline Project",
            "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
            "facts": ["Scott is building a Three.js/WebGPU/TSL shader playground."],
            "relations": "None",
        }

        self.assertFalse(Draft202012Validator(tool.definition.input_schema).is_valid(arguments))

    def test_memory_write_schema_rejects_truth_objects(self) -> None:
        tool = build_memory_write_tool()
        arguments = {
            "operation": "create",
            "target_kind": "ongoing",
            "title": "Visual Pipeline Project",
            "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
            "facts": {"text": "Scott is building a Three.js/WebGPU/TSL shader playground."},
            "relations": "None",
        }

        self.assertFalse(Draft202012Validator(tool.definition.input_schema).is_valid(arguments))

    def test_memory_write_schema_rejects_body_sections_lists(self) -> None:
        tool = build_memory_write_tool()
        arguments = {
            "operation": "create",
            "target_kind": "ongoing",
            "title": "Visual Pipeline Project",
            "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
            "facts": "None",
            "relations": "None",
            "body_sections": [
                {
                    "title": "Overview",
                    "text": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                }
            ],
        }

        self.assertFalse(Draft202012Validator(tool.definition.input_schema).is_valid(arguments))

    def test_memory_admin_discoverable_stays_compact_and_explicit(self) -> None:
        discoverable = build_memory_admin_discoverable()

        self.assertIn("explicitly asks", discoverable.purpose.lower())
        self.assertIsNone(discoverable.detailed_description)
        self.assertIsInstance(discoverable.usage, str)
        self.assertIn("repair_canonical_drift", discoverable.usage)

    def test_tool_register_loose_json_fields_include_array_items(self) -> None:
        tool = build_tool_register_tool(ToolRegistry())
        manifest_schema = tool.definition.input_schema["properties"]["manifest"]["properties"]

        for field_name in (
            "usage",
            "notes",
            "invocation",
            "provisioning",
            "artifacts",
            "rebuild",
            "safety",
        ):
            field_schema = manifest_schema[field_name]
            self.assertIn("array", field_schema["type"])
            self.assertIn("items", field_schema)

    def test_uses_default_web_fetch_limits_from_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(settings.web_fetch_timeout_seconds, 20.0)
        self.assertEqual(settings.web_fetch_max_markdown_chars, 250_000)

    def test_uses_default_central_python_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                os.environ,
                {"JARVIS_TOOL_RUNTIME_BASE_URL": ""},
                clear=True,
            ):
                settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(
            settings.central_python_venv,
            Path("/opt/venv"),
        )
        self.assertIsNone(settings.tool_runtime_base_url)
        self.assertEqual(settings.tool_runtime_timeout_seconds, 135.0)
        self.assertEqual(settings.tool_runtime_healthcheck_timeout_seconds, 5.0)
        self.assertEqual(
            settings.central_python_starter_packages,
            (
                "python-dateutil",
                "pyyaml",
                "pymupdf",
                "pandas",
                "matplotlib",
                "pillow",
                "pydantic",
                "jinja2",
                "rapidfuzz",
                "markdown-it-py",
                "feedparser",
                "openpyxl",
                "python-docx",
                "icalendar",
            ),
        )

    def test_accepts_legacy_python_env_override_for_central_python_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            legacy_venv = Path(tmp) / "legacy-venv"

            with patch.dict(
                os.environ,
                {"JARVIS_TOOL_PYTHON_INTERPRETER_VENV": str(legacy_venv)},
                clear=False,
            ):
                settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(settings.central_python_venv, legacy_venv)

    def test_bash_scrubbed_environment_targets_central_python_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            settings = ToolSettings.from_workspace_dir(workspace_dir)

        with patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}, clear=False):
            environment = _build_scrubbed_environment(settings)

        self.assertEqual(environment["VIRTUAL_ENV"], "/opt/venv")
        self.assertEqual(environment["UV_PROJECT_ENVIRONMENT"], "/opt/venv")
        self.assertEqual(environment["PIP_REQUIRE_VIRTUALENV"], "1")
        self.assertEqual(environment["PATH"], "/opt/venv/bin:/usr/bin:/bin")

    def test_bash_description_explains_central_python_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            settings = ToolSettings.from_workspace_dir(workspace_dir)

        description = format_bash_tool_description(settings).lower()

        self.assertIn("built-ins:", description)
        self.assertIn("python:", description)
        self.assertIn("long-running jobs:", description)
        self.assertIn("approvals and installation:", description)
        self.assertIn("central `/opt/venv` environment", description)
        self.assertIn("mode='background'", description)
        self.assertIn("mode='status'", description)
        self.assertIn("soft timeout", description)
        self.assertIn("automatically moved to background mode", description)
        self.assertIn("detached jobs are monitored by the orchestrator", description)
        self.assertIn("proactive polling", description)
        self.assertIn("uv pip install --python /opt/venv/bin/python", description)
        self.assertIn("`rg`, `jq`, `yq`", description)
        self.assertIn("`node`, `npm`, `npx`, and `corepack`", description)
        self.assertIn("user approval is required", description)
        self.assertIn("install packages or tools", description)
        self.assertIn("system-level mutations outside `/workspace`", description)
        self.assertIn("approval_summary", description)
        self.assertIn("use bare `python`/`python3`", description)


class DirectBashToolExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_promotes_long_foreground_command_to_background(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                os.environ,
                {
                    "JARVIS_TOOL_BASH_FOREGROUND_SOFT_TIMEOUT_SECONDS": "1",
                    "JARVIS_TOOL_BASH_DEFAULT_TIMEOUT_SECONDS": "5",
                    "JARVIS_TOOL_BASH_MAX_TIMEOUT_SECONDS": "30",
                },
                clear=False,
            ):
                settings = ToolSettings.from_workspace_dir(workspace_dir)

            executor = DirectBashToolExecutor(
                settings,
                target_runtime="test_runtime",
                runtime_location="test_runtime",
                runtime_transport="inprocess",
                container_mutation_boundary="test_boundary",
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)

            result = await executor(
                call_id="call_bash_soft_timeout",
                arguments={
                    "command": "sleep 2; printf 'late\\n'",
                    "timeout_seconds": 5,
                },
                context=context,
            )

            self.assertTrue(result.ok)
            self.assertTrue(result.metadata["promoted_to_background"])
            self.assertEqual(result.metadata["mode"], "foreground")
            self.assertEqual(result.metadata["status"], "running")
            self.assertEqual(result.metadata["state"], "running")
            self.assertIsNotNone(result.metadata["started_at"])
            self.assertIsNotNone(result.metadata["last_update_at"])
            self.assertEqual(result.metadata["suggested_next_check_seconds"], 5)
            self.assertIn("moved to background", result.content)
            job_id = str(result.metadata["job_id"])

            try:
                status_result = await executor(
                    call_id="call_bash_soft_timeout_status",
                    arguments={"mode": "status", "job_id": job_id},
                    context=context,
                )
                self.assertTrue(status_result.ok)
                self.assertIn(status_result.metadata["status"], {"running", "finished"})

                tail_result = await executor(
                    call_id="call_bash_soft_timeout_tail",
                    arguments={"mode": "tail", "job_id": job_id},
                    context=context,
                )
                self.assertTrue(tail_result.ok)
            finally:
                cancel_result = await executor(
                    call_id="call_bash_soft_timeout_cancel",
                    arguments={"mode": "cancel", "job_id": job_id},
                    context=context,
                )
                self.assertTrue(cancel_result.ok)

    async def test_promotes_when_requested_timeout_matches_soft_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                os.environ,
                {
                    "JARVIS_TOOL_BASH_FOREGROUND_SOFT_TIMEOUT_SECONDS": "1",
                    "JARVIS_TOOL_BASH_DEFAULT_TIMEOUT_SECONDS": "5",
                    "JARVIS_TOOL_BASH_MAX_TIMEOUT_SECONDS": "30",
                },
                clear=False,
            ):
                settings = ToolSettings.from_workspace_dir(workspace_dir)

            executor = DirectBashToolExecutor(
                settings,
                target_runtime="test_runtime",
                runtime_location="test_runtime",
                runtime_transport="inprocess",
                container_mutation_boundary="test_boundary",
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)

            result = await executor(
                call_id="call_bash_soft_timeout_boundary",
                arguments={
                    "command": "sleep 2; printf 'late\\n'",
                    "timeout_seconds": 1,
                },
                context=context,
            )

            self.assertTrue(result.ok)
            self.assertTrue(result.metadata["promoted_to_background"])
            job_id = str(result.metadata["job_id"])

            cancel_result = await executor(
                call_id="call_bash_soft_timeout_boundary_cancel",
                arguments={"mode": "cancel", "job_id": job_id},
                context=context,
            )
            self.assertTrue(cancel_result.ok)

    async def test_background_job_output_is_capped_and_reports_dropped_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                os.environ,
                {
                    "JARVIS_TOOL_BASH_JOB_LOG_MAX_BYTES": "128",
                    "JARVIS_TOOL_BASH_JOB_TOTAL_STORAGE_BUDGET_BYTES": "1048576",
                    "JARVIS_TOOL_BASH_JOB_RETENTION_SECONDS": "3600",
                },
                clear=False,
            ):
                settings = ToolSettings.from_workspace_dir(workspace_dir)

            executor = DirectBashToolExecutor(
                settings,
                target_runtime="test_runtime",
                runtime_location="test_runtime",
                runtime_transport="inprocess",
                container_mutation_boundary="test_boundary",
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)

            start_result = await executor(
                call_id="call_bash_background_capped_start",
                arguments={
                    "mode": "background",
                    "command": (
                        "python3 -c \"import sys; sys.stdout.write('x' * 4096); sys.stdout.flush()\""
                    ),
                },
                context=context,
            )

            self.assertTrue(start_result.ok)
            self.assertEqual(start_result.metadata["status"], "running")
            self.assertEqual(start_result.metadata["state"], "running")
            self.assertIsNotNone(start_result.metadata["started_at"])
            self.assertIsNotNone(start_result.metadata["last_update_at"])
            self.assertEqual(start_result.metadata["suggested_next_check_seconds"], 5)
            job_id = str(start_result.metadata["job_id"])

            status_result = None
            for _ in range(20):
                status_result = await executor(
                    call_id="call_bash_background_capped_status",
                    arguments={"mode": "status", "job_id": job_id},
                    context=context,
                )
                self.assertTrue(status_result.ok)
                if status_result.metadata["status"] == "finished":
                    break
                await asyncio.sleep(0.05)

            self.assertIsNotNone(status_result)
            if status_result is None:
                self.fail("Expected a bash background status result.")
            self.assertEqual(status_result.metadata["status"], "finished")
            self.assertGreater(status_result.metadata["stdout_bytes_dropped"], 0)
            self.assertLessEqual(
                status_result.metadata["stdout_bytes_retained"],
                settings.bash_job_log_max_bytes,
            )

            tail_result = await executor(
                call_id="call_bash_background_capped_tail",
                arguments={"mode": "tail", "job_id": job_id},
                context=context,
            )

            self.assertTrue(tail_result.ok)
            self.assertGreater(tail_result.metadata["stdout_bytes_dropped"], 0)
            self.assertIn("earlier output dropped", tail_result.content)

            paths, _ = load_job(workspace_dir, job_id)
            self.assertLessEqual(paths.stdout_path.stat().st_size, settings.bash_job_log_max_bytes)

    async def test_background_job_metadata_tracks_child_process_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            settings = ToolSettings.from_workspace_dir(workspace_dir)

            executor = DirectBashToolExecutor(
                settings,
                target_runtime="test_runtime",
                runtime_location="test_runtime",
                runtime_transport="inprocess",
                container_mutation_boundary="test_boundary",
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)

            start_result = await executor(
                call_id="call_bash_child_pid_start",
                arguments={"mode": "background", "command": "sleep 5"},
                context=context,
            )

            self.assertTrue(start_result.ok)
            job_id = str(start_result.metadata["job_id"])

            try:
                status_result = await executor(
                    call_id="call_bash_child_pid_status",
                    arguments={"mode": "status", "job_id": job_id},
                    context=context,
                )
                self.assertTrue(status_result.ok)
                self.assertGreater(status_result.metadata["runner_pid"], 0)
                self.assertGreater(status_result.metadata["pid"], 0)
                self.assertNotEqual(
                    status_result.metadata["runner_pid"],
                    status_result.metadata["pid"],
                )
            finally:
                cancel_result = await executor(
                    call_id="call_bash_child_pid_cancel",
                    arguments={"mode": "cancel", "job_id": job_id},
                    context=context,
                )
                self.assertTrue(cancel_result.ok)

    async def test_background_job_budget_rejects_new_job_when_active_jobs_exhaust_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            with patch.dict(
                os.environ,
                {
                    "JARVIS_TOOL_BASH_JOB_LOG_MAX_BYTES": "1024",
                    "JARVIS_TOOL_BASH_JOB_TOTAL_STORAGE_BUDGET_BYTES": "265000",
                    "JARVIS_TOOL_BASH_JOB_RETENTION_SECONDS": "3600",
                },
                clear=False,
            ):
                settings = ToolSettings.from_workspace_dir(workspace_dir)

            executor = DirectBashToolExecutor(
                settings,
                target_runtime="test_runtime",
                runtime_location="test_runtime",
                runtime_transport="inprocess",
                container_mutation_boundary="test_boundary",
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)

            first_result = await executor(
                call_id="call_bash_budget_first",
                arguments={"mode": "background", "command": "sleep 5"},
                context=context,
            )
            self.assertTrue(first_result.ok)
            first_job_id = str(first_result.metadata["job_id"])

            try:
                second_result = await executor(
                    call_id="call_bash_budget_second",
                    arguments={"mode": "background", "command": "sleep 5"},
                    context=context,
                )
                self.assertFalse(second_result.ok)
                self.assertIn("storage budget", second_result.content)
            finally:
                cancel_result = await executor(
                    call_id="call_bash_budget_cancel",
                    arguments={"mode": "cancel", "job_id": first_job_id},
                    context=context,
                )
                self.assertTrue(cancel_result.ok)

    async def test_sweeper_removes_expired_cancelled_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            settings = ToolSettings.from_workspace_dir(workspace_dir)

            executor = DirectBashToolExecutor(
                settings,
                target_runtime="test_runtime",
                runtime_location="test_runtime",
                runtime_transport="inprocess",
                container_mutation_boundary="test_boundary",
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)

            start_result = await executor(
                call_id="call_bash_sweeper_start",
                arguments={"mode": "background", "command": "sleep 5"},
                context=context,
            )
            self.assertTrue(start_result.ok)
            job_id = str(start_result.metadata["job_id"])

            cancel_result = await executor(
                call_id="call_bash_sweeper_cancel",
                arguments={"mode": "cancel", "job_id": job_id},
                context=context,
            )
            self.assertTrue(cancel_result.ok)

            paths, _ = load_job(workspace_dir, job_id)
            stale_timestamp = "2000-01-01T00:00:00Z\n"
            paths.cancelled_at_path.write_text(stale_timestamp, encoding="utf-8")

            sweep_job_artifacts(
                workspace_dir=workspace_dir,
                retention_seconds=1.0,
                total_storage_budget_bytes=settings.bash_job_total_storage_budget_bytes,
            )

            self.assertFalse(paths.job_dir.exists())


class RemoteToolRuntimeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_serializes_request_and_parses_result(self) -> None:
        captured_requests: list[tuple[str, str, dict[str, object] | None]] = []
        captured_timeouts: list[float] = []

        class _FakeAsyncClient:
            def __init__(self, *, base_url: str, timeout: float) -> None:
                self.base_url = base_url
                self.timeout = timeout
                captured_timeouts.append(timeout)

            async def __aenter__(self) -> "_FakeAsyncClient":
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                _ = exc_type, exc, tb

            async def post(
                self,
                url: str,
                *,
                json: dict[str, object],
            ) -> _FakeAsyncHTTPResponse:
                captured_requests.append(("POST", url, json))
                return _FakeAsyncHTTPResponse(
                    status_code=200,
                    payload={
                        "call_id": "call_1",
                        "name": "bash",
                        "ok": True,
                        "content": "Bash execution result",
                        "metadata": {"runtime_location": "tool_runtime_container"},
                    },
                )

        with patch.dict(
            os.environ,
            {"JARVIS_TOOL_RUNTIME_BASE_URL": "http://tool_runtime:8081"},
            clear=False,
        ):
            settings = ToolSettings.from_workspace_dir(Path("/workspace"))

        client = RemoteToolRuntimeClient(settings)
        with patch("jarvis.tools.remote_runtime_client.httpx.AsyncClient", _FakeAsyncClient):
            result = await client.execute(
                tool_name="bash",
                call_id="call_1",
                arguments={"command": "pwd"},
                context=ToolExecutionContext(
                    workspace_dir=Path("/workspace"),
                    route_id="tg_123",
                    session_id="session_9",
                ),
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.name, "bash")
        self.assertEqual(
            captured_requests,
            [
                (
                    "POST",
                    "/tools/bash/execute",
                    {
                        "call_id": "call_1",
                        "arguments": {"command": "pwd"},
                        "workspace_dir": "/workspace",
                        "session_id": "session_9",
                        "route_id": "tg_123",
                    },
                )
            ],
        )
        self.assertEqual(captured_timeouts, [135.0])

    async def test_execute_uses_requested_tool_timeout_plus_headroom(self) -> None:
        captured_timeouts: list[float] = []

        class _FakeAsyncClient:
            def __init__(self, *, base_url: str, timeout: float) -> None:
                _ = base_url
                captured_timeouts.append(timeout)

            async def __aenter__(self) -> "_FakeAsyncClient":
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                _ = exc_type, exc, tb

            async def post(
                self,
                url: str,
                *,
                json: dict[str, object],
            ) -> _FakeAsyncHTTPResponse:
                _ = url, json
                return _FakeAsyncHTTPResponse(
                    status_code=200,
                    payload={
                        "call_id": "call_timeout",
                        "name": "bash",
                        "ok": True,
                        "content": "Bash execution result",
                        "metadata": {},
                    },
                )

        with patch.dict(
            os.environ,
            {"JARVIS_TOOL_RUNTIME_BASE_URL": "http://tool_runtime:8081"},
            clear=False,
        ):
            settings = ToolSettings.from_workspace_dir(Path("/workspace"))

        client = RemoteToolRuntimeClient(settings)
        with patch("jarvis.tools.remote_runtime_client.httpx.AsyncClient", _FakeAsyncClient):
            await client.execute(
                tool_name="bash",
                call_id="call_timeout",
                arguments={"command": "python -c 'print(\\\"ok\\\")'", "timeout_seconds": 600},
                context=ToolExecutionContext(workspace_dir=Path("/workspace")),
            )

        self.assertEqual(captured_timeouts, [615.0])

    async def test_execute_clamps_requested_timeout_to_tool_max_before_adding_headroom(self) -> None:
        captured_timeouts: list[float] = []

        class _FakeAsyncClient:
            def __init__(self, *, base_url: str, timeout: float) -> None:
                _ = base_url
                captured_timeouts.append(timeout)

            async def __aenter__(self) -> "_FakeAsyncClient":
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                _ = exc_type, exc, tb

            async def post(
                self,
                url: str,
                *,
                json: dict[str, object],
            ) -> _FakeAsyncHTTPResponse:
                _ = url, json
                return _FakeAsyncHTTPResponse(
                    status_code=200,
                    payload={
                        "call_id": "call_timeout_clamped",
                        "name": "bash",
                        "ok": True,
                        "content": "Bash execution result",
                        "metadata": {},
                    },
                )

        with patch.dict(
            os.environ,
            {"JARVIS_TOOL_RUNTIME_BASE_URL": "http://tool_runtime:8081"},
            clear=False,
        ):
            settings = ToolSettings.from_workspace_dir(Path("/workspace"))

        client = RemoteToolRuntimeClient(settings)
        with patch("jarvis.tools.remote_runtime_client.httpx.AsyncClient", _FakeAsyncClient):
            await client.execute(
                tool_name="bash",
                call_id="call_timeout_clamped",
                arguments={"command": "pwd", "timeout_seconds": 999999},
                context=ToolExecutionContext(workspace_dir=Path("/workspace")),
            )

        self.assertEqual(captured_timeouts, [1815.0])

    async def test_execute_raises_for_malformed_response_payload(self) -> None:
        class _FakeAsyncClient:
            def __init__(self, *, base_url: str, timeout: float) -> None:
                _ = base_url, timeout

            async def __aenter__(self) -> "_FakeAsyncClient":
                return self

            async def __aexit__(self, exc_type, exc, tb) -> None:
                _ = exc_type, exc, tb

            async def post(
                self,
                url: str,
                *,
                json: dict[str, object],
            ) -> _FakeAsyncHTTPResponse:
                _ = url, json
                return _FakeAsyncHTTPResponse(
                    status_code=200,
                    payload={
                        "call_id": "wrong_call",
                        "name": "bash",
                        "ok": True,
                        "content": "bad",
                        "metadata": {},
                    },
                )

        with patch.dict(
            os.environ,
            {"JARVIS_TOOL_RUNTIME_BASE_URL": "http://tool_runtime:8081"},
            clear=False,
        ):
            settings = ToolSettings.from_workspace_dir(Path("/workspace"))

        client = RemoteToolRuntimeClient(settings)
        with patch("jarvis.tools.remote_runtime_client.httpx.AsyncClient", _FakeAsyncClient):
            with self.assertRaises(RemoteToolRuntimeError):
                await client.execute(
                    tool_name="bash",
                    call_id="call_1",
                    arguments={"command": "pwd"},
                    context=ToolExecutionContext(workspace_dir=Path("/workspace")),
                )


class RuntimeToolManifestTests(unittest.TestCase):
    def test_validate_runtime_tool_manifest_normalizes_aliases(self) -> None:
        manifest = validate_runtime_tool_manifest_payload(
            {
                "name": "google_workspace_cli",
                "purpose": "Use the Google Workspace CLI from bash.",
                "operator": "bash",
                "aliases": [" google ", "admin", "google", ""],
                "invocation": {"command": "gws --help"},
            }
        )

        self.assertEqual(manifest.name, "google_workspace_cli")
        self.assertEqual(manifest.aliases, ("google", "admin"))
        self.assertEqual(manifest.to_dict()["invocation"], {"command": "gws --help"})
        self.assertRegex(manifest.manifest_hash(), r"^[0-9a-f]{64}$")

    def test_load_runtime_tool_catalog_reads_valid_manifests_and_reports_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            _write_runtime_manifest(
                workspace_dir,
                {
                    "name": "google_workspace_cli",
                    "purpose": "Use the Google Workspace CLI from bash.",
                    "operator": "bash",
                    "invocation": {"command": "gws --help"},
                },
            )
            runtime_dir = workspace_dir / "runtime_tools"
            (runtime_dir / "bad.json").write_text("{bad json", encoding="utf-8")
            _write_runtime_manifest(
                workspace_dir,
                {
                    "name": "bash",
                    "purpose": "Conflicting tool.",
                    "operator": "bash",
                },
            )

            loaded = load_runtime_tool_catalog(
                workspace_dir,
                reserved_names={"bash", "tool_search"},
            )

        self.assertEqual([entry.name for entry in loaded.entries], ["google_workspace_cli"])
        self.assertEqual([manifest.name for manifest in loaded.manifests], ["google_workspace_cli"])
        self.assertEqual(loaded.entries[0].metadata["source"], "runtime_tools")
        self.assertEqual(loaded.entries[0].usage["operator"], "bash")
        self.assertEqual(len(loaded.errors), 2)
        self.assertTrue(any("bad.json" in error for error in loaded.errors))
        self.assertTrue(any("conflicts with a built-in tool" in error for error in loaded.errors))


class ToolRegistryTests(unittest.TestCase):
    def test_memory_search_result_formats_sys_warning(self) -> None:
        rendered = _format_memory_search_result(
            query="current project",
            results=[],
            warnings=["semantic search was skipped because the embedding vector table is missing; used lexical+graph fallback"],
            semantic_disabled=True,
        )

        self.assertIn("semantic_disabled: true", rendered)
        self.assertIn("sys_warning: semantic search was skipped because the embedding vector table is missing; used lexical+graph fallback", rendered)

    def test_memory_search_result_includes_title_and_memory_get_hint(self) -> None:
        rendered = _format_memory_search_result(
            query="backend preference",
            results=[
                {
                    "document_id": "core_backend_pref",
                    "title": "Backend Preference",
                    "path": "/workspace/memory/core/backend-preference.md",
                    "kind": "core",
                    "chunk_id": "chunk_123",
                    "section_path": "facts",
                    "score": 0.98,
                    "snippet": "Scott prefers Python for backend work.",
                    "match_reasons": ["semantic_match", "graph_entity_match"],
                    "source_ref_ids": [],
                    "semantic_disabled": False,
                }
            ],
            warnings=[],
            semantic_disabled=False,
        )

        self.assertIn("title: Backend Preference", rendered)
        self.assertIn("section: facts (synthetic)", rendered)
        self.assertIn("next_step: use memory_get(document_id=..., section_path=...)", rendered)

    def test_default_registry_exposes_basic_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))

            basic_names = [tool.name for tool in registry.basic_definitions()]
            self.assertEqual(
                basic_names,
                [
                    "bash",
                    "file_patch",
                    "memory_search",
                    "memory_get",
                    "memory_write",
                    "get_skills",
                    "web_search",
                    "web_fetch",
                    "view_image",
                    "send_file",
                    "tool_search",
                    "tool_register",
                ],
            )
            self.assertEqual(registry.search("bash"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("bash", include_basic=True)],
                ["bash"],
            )
            self.assertEqual(registry.search("patch"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("patch", include_basic=True)],
                ["file_patch"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search("memory")],
                ["memory_admin"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search("memory", include_basic=True)],
                ["memory_search", "memory_get", "memory_write", "memory_admin"],
            )
            self.assertEqual(registry.search("python"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("python", include_basic=True)],
                ["bash"],
            )
            self.assertEqual(registry.search_discoverable("python"), ())
            self.assertEqual(registry.search("web_search"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("web_search", include_basic=True)],
                ["web_search"],
            )
            self.assertEqual(registry.search("web_fetch"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("web_fetch", include_basic=True)],
                ["web_fetch"],
            )
            self.assertEqual(registry.search("view_image"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("image", include_basic=True)],
                ["view_image", "generate_edit_image"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search("send")],
                ["email"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search("send", include_basic=True)],
                ["send_file", "email"],
            )
            self.assertEqual(registry.search_discoverable("archive"), ())
            self.assertEqual(
                [tool.name for tool in registry.search_discoverable("")],
                [
                    "email",
                    "ffmpeg",
                    "generate_edit_image",
                    "memory_admin",
                    "transcribe",
                ],
            )
            self.assertEqual(
                [tool.name for tool in registry.search_discoverable("email")],
                ["email"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search_discoverable("edit image")],
                ["generate_edit_image"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search_discoverable("memory")],
                ["memory_admin"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search_discoverable("transcribe")],
                ["transcribe"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search_discoverable("ffmpeg")],
                ["ffmpeg"],
            )
            self.assertEqual(
                [
                    tool.name
                    for tool in registry.resolve_discoverable_tool_definitions(["ffmpeg"])
                ],
                [],
            )

    def test_backed_discoverable_entries_reuse_executable_tool_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))

            for tool_name in ("email", "generate_edit_image", "transcribe"):
                registered = registry.require(tool_name)
                discoverable = registry.get_discoverable(tool_name)

                self.assertIsNotNone(discoverable)
                self.assertEqual(
                    registered.definition.description,
                    discoverable.detailed_description,
                )

            memory_admin = registry.get_discoverable("memory_admin")
            self.assertIsNotNone(memory_admin)
            self.assertIsNone(memory_admin.detailed_description)
            self.assertIn("explicitly asks", memory_admin.purpose.lower())

    def test_search_discoverable_matches_name_alias_and_description(self) -> None:
        registry = ToolRegistry()
        registry.register(
            RegisteredTool(
                name="archive",
                exposure="discoverable",
                definition=ToolDefinition(
                    name="archive",
                    description="Archive file helper.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
                executor=AsyncMock(),
            )
        )
        registry.register_discoverable(
            DiscoverableTool(
                name="archive",
                aliases=("zip_tools", "tar_tools"),
                purpose="List, extract, and create archive formats inside the workspace.",
                detailed_description="Handles zip and tar style archive workflows.",
                usage={"arguments": ["path"]},
                metadata={"family": "filesystem"},
                backing_tool_name="archive",
            )
        )

        self.assertEqual(
            [tool.name for tool in registry.search_discoverable("zip")],
            ["archive"],
        )
        self.assertEqual(
            [tool.name for tool in registry.search_discoverable("tar style")],
            ["archive"],
        )
        self.assertEqual(
            [tool.name for tool in registry.search_discoverable("")],
            ["archive"],
        )
        self.assertEqual(
            [tool.name for tool in registry.resolve_discoverable_tool_definitions(["archive"])],
            ["archive"],
        )


class ToolRegistryFilteredViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_subagent_view_hides_main_only_send_file_and_email_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
            main_view = registry.filtered_view(agent_kind="main")
            subagent_view = registry.filtered_view(agent_kind="subagent")

            self.assertIsNotNone(main_view.get("send_file"))
            self.assertIsNone(subagent_view.get("send_file"))
            self.assertIsNotNone(main_view.get("email"))
            self.assertIsNone(subagent_view.get("email"))

    async def test_subagent_filtered_view_hides_blocked_built_in_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
            subagent_view = registry.filtered_view(
                agent_kind="subagent",
                hidden_tool_names=(
                    "memory_search",
                    "memory_get",
                    "memory_write",
                    "memory_admin",
                    "send_file",
                    "email",
                ),
            )

            self.assertNotIn(
                "memory_search",
                [tool.name for tool in subagent_view.basic_definitions()],
            )
            self.assertNotIn(
                "memory_get",
                [tool.name for tool in subagent_view.basic_definitions()],
            )
            self.assertNotIn(
                "memory_write",
                [tool.name for tool in subagent_view.basic_definitions()],
            )
            self.assertNotIn(
                "send_file",
                [tool.name for tool in subagent_view.basic_definitions()],
            )
            self.assertNotIn(
                "email",
                [tool.name for tool in subagent_view.basic_definitions()],
            )
            self.assertEqual(
                [tool.name for tool in subagent_view.search_discoverable("memory")],
                [],
            )

    async def test_subagent_tool_search_keeps_runtime_manifests_visible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            _write_runtime_manifest(
                workspace_dir,
                {
                    "name": "memory_helper_cli",
                    "purpose": "Inspect exported memory reports through bash.",
                    "operator": "bash",
                    "invocation": {"command": "memory-helper --help"},
                },
            )
            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
            subagent_view = registry.filtered_view(
                agent_kind="subagent",
                hidden_tool_names=(
                    "memory_search",
                    "memory_get",
                    "memory_write",
                    "memory_admin",
                    "send_file",
                ),
            )
            tool_search = subagent_view.require("tool_search")

            result = await tool_search.executor(
                call_id="call_1",
                arguments={"query": "memory", "verbosity": "high"},
                context=ToolExecutionContext(
                    workspace_dir=workspace_dir,
                    agent_kind="subagent",
                    agent_name="Friday",
                ),
            )

            self.assertTrue(result.ok)
            self.assertEqual(
                [match["name"] for match in result.metadata["matches"]],
                ["memory_helper_cli"],
            )
            self.assertEqual(result.metadata["activated_discoverable_tool_names"], [])


class ToolPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace_dir = Path(self._tmp.name) / "workspace"
        self.workspace_dir.mkdir()
        self.context = ToolExecutionContext(workspace_dir=self.workspace_dir)
        self.policy = ToolPolicy()

    def test_bash_policy_allows_absolute_paths_for_runtime_to_handle(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "cat /etc/hosts"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_bash_policy_allows_normal_shell_syntax(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "printf 'hello' > note.txt && cat $(pwd)/note.txt"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_bash_policy_allows_workspace_write_commands(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "cp /etc/hosts notes.txt"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_bash_policy_allows_path_escape_attempts_for_runtime_to_handle(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "cat /repo/private.txt"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_bash_policy_denies_null_byte_commands(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "printf 'hello'\x00"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("null bytes", decision.reason or "")

    def test_bash_policy_requires_approval_for_system_write_command(self) -> None:
        command = "printf 'jarvis' > /etc/jarvis-bash-test"
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": command},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "bash command requires explicit approval.")
        self.assertIsNotNone(decision.approval_request)
        if decision.approval_request is None:
            self.fail("Expected approval request metadata.")
        self.assertEqual(decision.approval_request["kind"], "bash_command")
        self.assertEqual(decision.approval_request["command"], command)
        self.assertEqual(decision.approval_request["target_runtime"], "tool_runtime")

    def test_bash_policy_hard_denies_os_upgrade_commands(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "apt-get upgrade -y"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "tool_runtime OS upgrade commands are denied.")

    def test_bash_policy_hard_denies_service_control_commands(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "systemctl restart nginx"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(
            decision.reason,
            "tool_runtime service and init control commands are denied.",
        )

    def test_bash_policy_allows_direct_python_execution(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "python -c 'print(1)'"},
            context=self.context,
        )

        self.assertTrue(decision.allowed)

    def test_bash_policy_requires_job_id_for_status_mode(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"mode": "status"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("job_id", decision.reason or "")

    def test_bash_policy_hard_denies_noncentral_python_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "/usr/bin/python -c 'print(1)'"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("non-central interpreter path", decision.reason or "")
        self.assertIn("/opt/venv", decision.reason or "")
        self.assertIn("/opt/venv/bin/python", decision.reason or "")

    def test_bash_policy_hard_denies_uv_run_python_execution(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "uv run python script.py"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("must not route through `uv run python`", decision.reason or "")
        self.assertIn("/opt/venv/bin/python", decision.reason or "")

    def test_bash_policy_hard_denies_unknown_python_command_name(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "python9.9 -c 'print(1)'"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("does not resolve to the central interpreter set", decision.reason or "")
        self.assertIn("/opt/venv/bin/python", decision.reason or "")

    def test_bash_policy_allows_explicit_central_python_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "/opt/venv/bin/python -c 'print(1)'"},
            context=self.context,
        )

        self.assertTrue(decision.allowed)

    def test_bash_policy_hard_denies_workspace_venv_python_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "./.venv/bin/python -c 'print(1)'"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("non-central interpreter path", decision.reason or "")
        self.assertIn("/opt/venv/bin/python", decision.reason or "")

    def test_bash_policy_hard_denies_second_venv_creation(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "python -m venv .venv"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("must not create a second Python environment", decision.reason or "")
        self.assertIn("/opt/venv/bin/python", decision.reason or "")

    def test_bash_policy_hard_denies_uv_pip_install_targeting_noncentral_python(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={
                "command": "uv pip install --python /usr/bin/python requests"
            },
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("targets a non-central interpreter path", decision.reason or "")
        self.assertIn("/opt/venv/bin/python", decision.reason or "")

    def test_bash_policy_allows_background_mode_command_validation(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"mode": "background", "command": "python -c 'print(1)'"},
            context=self.context,
        )

        self.assertTrue(decision.allowed)

    def test_bash_policy_requires_approval_for_uv_pip_install(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "uv pip install requests"},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "bash command requires explicit approval.")
        self.assertIsNotNone(decision.approval_request)
        if decision.approval_request is None:
            self.fail("Expected approval request metadata.")
        self.assertEqual(
            decision.approval_request["detector_reason"],
            "matched install or package-manager mutation pattern for tool_runtime",
        )

    def test_bash_policy_requires_approval_for_background_system_write(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={
                "mode": "background",
                "command": "printf 'jarvis' > /etc/jarvis-bash-test",
            },
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "bash command requires explicit approval.")
        self.assertIsNotNone(decision.approval_request)

    def test_bash_policy_allows_exactly_approved_system_write_command(self) -> None:
        command = "printf 'jarvis' > /etc/jarvis-bash-test"
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": command},
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                approved_action={"kind": "bash_command", "command": command},
            ),
        )

        self.assertTrue(decision.allowed)

    def test_bash_policy_skip_permission_env_bypasses_detector(self) -> None:
        with patch.dict(
            os.environ,
            {"BASH_DANGEROUSLY_SKIP_PERMISSION": "true"},
            clear=False,
        ):
            decision = ToolPolicy().authorize(
                tool_name="bash",
                arguments={"command": "printf 'jarvis' > /etc/jarvis-bash-test"},
                context=self.context,
            )

        self.assertTrue(decision.allowed)

    def test_tool_register_requires_approval(self) -> None:
        manifest = {
            "name": "google_workspace_cli",
            "purpose": "Use the Google Workspace CLI from bash.",
            "operator": "bash",
            "invocation": {"command": "gws --help"},
        }

        decision = self.policy.authorize(
            tool_name="tool_register",
            arguments={"manifest": manifest},
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "tool_register requires explicit approval.")
        self.assertIsNotNone(decision.approval_request)
        if decision.approval_request is None:
            self.fail("Expected approval request metadata.")
        self.assertEqual(decision.approval_request["kind"], "register_runtime_tool")
        self.assertEqual(decision.approval_request["tool_name"], "google_workspace_cli")

    def test_tool_register_allows_exactly_approved_manifest(self) -> None:
        manifest = {
            "name": "google_workspace_cli",
            "purpose": "Use the Google Workspace CLI from bash.",
            "operator": "bash",
            "invocation": {"command": "gws --help"},
        }
        validated = validate_runtime_tool_manifest_payload(manifest)

        decision = self.policy.authorize(
            tool_name="tool_register",
            arguments={"manifest": manifest},
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                approved_action={
                    "kind": "register_runtime_tool",
                    "tool_name": validated.name,
                    "manifest_hash": validated.manifest_hash(),
                },
            ),
        )

        self.assertTrue(decision.allowed)

    def test_memory_write_policy_denies_summary_only_core_create(self) -> None:
        decision = self.policy.authorize(
            tool_name="memory_write",
            arguments={
                "operation": "create",
                "target_kind": "core",
                "title": "Backend Preference",
                "summary": "Scott prefers Python for backend APIs.",
            },
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("facts and relations", decision.reason or "")
        self.assertIn('literal string "None"', decision.reason or "")
        self.assertIn("summary is not a substitute", decision.reason or "")
        self.assertIn(
            "prefer real fact objects in facts whenever the user states an explicit durable fact",
            (decision.reason or "").lower(),
        )

    def test_memory_write_policy_denies_daily_upsert_without_body_sections(self) -> None:
        decision = self.policy.authorize(
            tool_name="memory_write",
            arguments={
                "operation": "upsert",
                "target_kind": "daily",
                "document_id": "daily_2026-04-08",
                "summary": "Scott went to Bray for a bike ride.",
            },
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("daily upsert", (decision.reason or "").lower())
        self.assertIn("memory_get", decision.reason or "")
        self.assertIn("body_sections", decision.reason or "")
        self.assertIn("summary alone does not rewrite prior daily content", (decision.reason or "").lower())
        self.assertIn("append_daily", decision.reason or "")

    def test_memory_write_policy_denies_empty_truth_arrays(self) -> None:
        decision = self.policy.authorize(
            tool_name="memory_write",
            arguments={
                "operation": "create",
                "target_kind": "ongoing",
                "title": "Visual Pipeline Project",
                "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                "facts": [],
                "relations": "None",
            },
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("empty arrays are not allowed", decision.reason or "")

    def test_memory_write_policy_allows_explicit_none_truth_decision(self) -> None:
        decision = self.policy.authorize(
            tool_name="memory_write",
            arguments={
                "operation": "create",
                "target_kind": "core",
                "title": "Visual Pipeline Project",
                "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                "facts": "None",
                "relations": "None",
            },
            context=self.context,
        )

        self.assertTrue(decision.allowed)

    def test_memory_write_policy_denies_list_body_sections(self) -> None:
        decision = self.policy.authorize(
            tool_name="memory_write",
            arguments={
                "operation": "create",
                "target_kind": "ongoing",
                "title": "Visual Pipeline Project",
                "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                "facts": "None",
                "relations": "None",
                "body_sections": [
                    {
                        "title": "Overview",
                        "text": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                    }
                ],
            },
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("body_sections", decision.reason or "")
        self.assertIn("Do not pass a list of section objects", decision.reason or "")

    def test_memory_write_policy_denies_blank_body_section_placeholders(self) -> None:
        decision = self.policy.authorize(
            tool_name="memory_write",
            arguments={
                "operation": "upsert",
                "target_kind": "daily",
                "document_id": "daily_2026-04-08",
                "body_sections": {
                    "Notable Events": "- Went for a bike ride along the coast.",
                    "Decisions": "",
                },
            },
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn('body_sections["Decisions"]', decision.reason or "")
        self.assertIn("omit untouched sections instead of passing empty strings", decision.reason or "")

    def test_memory_write_policy_reports_multiple_nested_errors_together(self) -> None:
        decision = self.policy.authorize(
            tool_name="memory_write",
            arguments={
                "operation": "create",
                "target_kind": "ongoing",
                "title": "Visual Pipeline Project",
                "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                "facts": [{"statement": "Scott is building a Three.js/WebGPU/TSL shader playground."}],
                "relations": [
                    {
                        "subject": "Scott",
                        "predicate": "is_building",
                        "object": "Visual Pipeline Project",
                        "status": "active",
                    }
                ],
                "body_sections": [
                    {
                        "title": "Overview",
                        "text": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                    }
                ],
            },
            context=self.context,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("facts[0].text", decision.reason or "")
        self.assertIn("relations[0].status", decision.reason or "")
        self.assertIn("body_sections", decision.reason or "")
        self.assertIn("Minimal valid example", decision.reason or "")

    def test_view_image_allows_workspace_relative_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="view_image",
            arguments={"path": "temp/sample.png"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_file_patch_allows_workspace_relative_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="file_patch",
            arguments={
                "path": "src/example.py",
                "operations": [{"type": "write", "content": "print('hello')\n"}],
            },
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_file_patch_denies_path_escape(self) -> None:
        outside = self.workspace_dir.parent / "outside.py"
        decision = self.policy.authorize(
            tool_name="file_patch",
            arguments={
                "path": str(outside),
                "operations": [{"type": "write", "content": "print('hello')\n"}],
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("inside", decision.reason or "")

    def test_file_patch_denies_dot_env_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="file_patch",
            arguments={
                "path": ".env",
                "operations": [{"type": "write", "content": "SECRET=1\n"}],
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn(".env", decision.reason or "")

    def test_view_image_denies_path_escape(self) -> None:
        outside = self.workspace_dir.parent / "outside.png"
        decision = self.policy.authorize(
            tool_name="view_image",
            arguments={"path": str(outside)},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("inside", decision.reason or "")

    def test_send_file_allows_workspace_relative_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="send_file",
            arguments={"path": "exports/report.pdf"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_send_file_denies_path_escape(self) -> None:
        outside = self.workspace_dir.parent / "outside.pdf"
        decision = self.policy.authorize(
            tool_name="send_file",
            arguments={"path": str(outside)},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("inside", decision.reason or "")

    def test_send_file_denies_dot_env_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="send_file",
            arguments={"path": ".env"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn(".env", decision.reason or "")

    def test_generate_edit_image_allows_prompt_only(self) -> None:
        decision = self.policy.authorize(
            tool_name="generate_edit_image",
            arguments={
                "prompt": "A minimal product photo of a coffee mug.",
                "output_path": "artifacts/mug.png",
            },
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_generate_edit_image_denies_missing_output_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="generate_edit_image",
            arguments={"prompt": "A minimal product photo of a coffee mug."},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("output_path", decision.reason or "")

    def test_generate_edit_image_denies_invalid_provider(self) -> None:
        decision = self.policy.authorize(
            tool_name="generate_edit_image",
            arguments={
                "prompt": "A minimal product photo of a coffee mug.",
                "output_path": "artifacts/mug.png",
                "provider": "stability",
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("provider", decision.reason or "")

    def test_generate_edit_image_denies_invalid_quality(self) -> None:
        decision = self.policy.authorize(
            tool_name="generate_edit_image",
            arguments={
                "prompt": "A minimal product photo of a coffee mug.",
                "output_path": "artifacts/mug.png",
                "quality": "ultra",
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("quality", decision.reason or "")

    def test_generate_edit_image_denies_invalid_resolution(self) -> None:
        decision = self.policy.authorize(
            tool_name="generate_edit_image",
            arguments={
                "prompt": "A minimal product photo of a coffee mug.",
                "output_path": "artifacts/mug.png",
                "resolution": "8K",
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("resolution", decision.reason or "")

    def test_generate_edit_image_denies_path_escape(self) -> None:
        outside = self.workspace_dir.parent / "outside.png"
        decision = self.policy.authorize(
            tool_name="generate_edit_image",
            arguments={
                "prompt": "Swap the background to blue.",
                "image_path": str(outside),
                "output_path": "artifacts/result.png",
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("inside", decision.reason or "")

    def test_generate_edit_image_denies_output_path_escape(self) -> None:
        outside = self.workspace_dir.parent / "outside.png"
        decision = self.policy.authorize(
            tool_name="generate_edit_image",
            arguments={
                "prompt": "A minimal product photo of a coffee mug.",
                "output_path": str(outside),
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("inside", decision.reason or "")

    def test_transcribe_allows_supported_audio_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="transcribe",
            arguments={"audio_path": "temp/meeting.wav"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_transcribe_denies_missing_audio_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="transcribe",
            arguments={},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("audio_path", decision.reason or "")

    def test_transcribe_denies_unsupported_extension(self) -> None:
        decision = self.policy.authorize(
            tool_name="transcribe",
            arguments={"audio_path": "temp/meeting.aac"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("supported extension", decision.reason or "")

    def test_transcribe_denies_path_escape(self) -> None:
        outside = self.workspace_dir.parent / "meeting.wav"
        decision = self.policy.authorize(
            tool_name="transcribe",
            arguments={"audio_path": str(outside)},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("inside", decision.reason or "")

    def test_email_requires_approval_for_valid_send(self) -> None:
        decision = self.policy.authorize(
            tool_name="email",
            arguments={
                "to_email": "user@example.com",
                "subject": "Weekly update",
                "body": "# Status\n\nAll tasks are complete.",
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "email requires explicit approval.")
        self.assertIsNotNone(decision.approval_request)
        if decision.approval_request is None:
            self.fail("Expected approval request metadata.")
        self.assertEqual(decision.approval_request["kind"], "send_email")
        self.assertEqual(decision.approval_request["to_email"], "user@example.com")
        self.assertIn("Weekly update", decision.approval_request["summary"])

    def test_email_allows_matching_approved_action(self) -> None:
        arguments = {
            "to_email": "user@example.com",
            "subject": "Weekly update",
            "body": "# Status\n\nAll tasks are complete.",
        }
        initial = self.policy.authorize(
            tool_name="email",
            arguments=arguments,
            context=self.context,
        )
        self.assertFalse(initial.allowed)
        if initial.approval_request is None:
            self.fail("Expected approval request metadata.")

        decision = self.policy.authorize(
            tool_name="email",
            arguments=arguments,
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                approved_action={
                    "kind": "send_email",
                    "request_hash": initial.approval_request["request_hash"],
                },
            ),
        )
        self.assertTrue(decision.allowed)

    def test_email_denies_invalid_to_email(self) -> None:
        decision = self.policy.authorize(
            tool_name="email",
            arguments={
                "to_email": "not-an-email",
                "subject": "Weekly update",
                "body": "Hello there.",
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("to_email", decision.reason or "")

    def test_email_denies_attachment_path_escape(self) -> None:
        outside = self.workspace_dir.parent / "outside.txt"
        decision = self.policy.authorize(
            tool_name="email",
            arguments={
                "to_email": "user@example.com",
                "subject": "Weekly update",
                "body": "Hello there.",
                "attachment_paths": [str(outside)],
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("inside", decision.reason or "")

    def test_web_search_allows_non_empty_query(self) -> None:
        decision = self.policy.authorize(
            tool_name="web_search",
            arguments={"query": "best coffee grinders"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_web_search_denies_empty_query(self) -> None:
        decision = self.policy.authorize(
            tool_name="web_search",
            arguments={"query": "   "},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("non-empty", decision.reason or "")

    def test_web_fetch_allows_public_https_url(self) -> None:
        decision = self.policy.authorize(
            tool_name="web_fetch",
            arguments={"url": "https://example.com/docs"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_web_fetch_denies_localhost_target(self) -> None:
        decision = self.policy.authorize(
            tool_name="web_fetch",
            arguments={"url": "http://localhost:8080/debug"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("localhost", decision.reason or "")

    def test_web_fetch_denies_private_ip_target(self) -> None:
        decision = self.policy.authorize(
            tool_name="web_fetch",
            arguments={"url": "http://127.0.0.1:8080/debug"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("private", decision.reason or "")

    def test_tool_search_allows_empty_query_and_defaults(self) -> None:
        decision = self.policy.authorize(
            tool_name="tool_search",
            arguments={},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_tool_search_denies_invalid_verbosity(self) -> None:
        decision = self.policy.authorize(
            tool_name="tool_search",
            arguments={"query": "archive", "verbosity": "verbose"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("verbosity", decision.reason or "")

class ToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        temp_dir_kwargs: dict[str, str] = {}
        if _REMOTE_TOOL_RUNTIME_CONFIGURED:
            temp_dir_kwargs["dir"] = "/workspace"
        self._tmp = tempfile.TemporaryDirectory(**temp_dir_kwargs)
        self.addAsyncCleanup(self._cleanup_tmpdir)
        self.workspace_dir = Path(self._tmp.name) / "workspace"
        self.workspace_dir.mkdir()
        settings = ToolSettings.from_workspace_dir(self.workspace_dir)
        self.settings = settings
        self.registry = ToolRegistry.default(settings)
        self.runtime = ToolRuntime(registry=self.registry)
        self.context = ToolExecutionContext(workspace_dir=self.workspace_dir)

    async def _cleanup_tmpdir(self) -> None:
        self._tmp.cleanup()

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_executes_pwd_inside_workspace(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_pwd",
                name="bash",
                arguments={"command": "pwd"},
                raw_arguments='{"command":"pwd"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("cwd: /workspace", result.content)
        self.assertEqual(result.metadata["cwd"], "/workspace")
        self.assertEqual(result.metadata["workspace_source_dir"], str(self.workspace_dir))

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_writes_file_inside_workspace(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_write",
                name="bash",
                arguments={"command": "printf 'hello' > note.txt"},
                raw_arguments='{"command":"printf \\"hello\\" > note.txt"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            (self.workspace_dir / "note.txt").read_text(encoding="utf-8"),
            "hello",
        )

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE and _CENTRAL_PYTHON_INTERPRETER.exists(),
        "bash Python execution requires the central /opt/venv interpreter",
    )
    async def test_bash_executes_python_in_the_central_venv(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_python",
                name="bash",
                arguments={"command": "python -c 'import sys; print(sys.executable)'"},
                raw_arguments=(
                    '{"command":"python -c \\"import sys; print(sys.executable)\\""}'
                ),
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("/opt/venv/bin/python", result.content)

    @unittest.skipUnless(
        _REMOTE_TOOL_RUNTIME_CONFIGURED,
        _REMOTE_TOOL_RUNTIME_SKIP_REASON,
    )
    async def test_remote_bash_foreground_soft_timeout_promotes_to_background(self) -> None:
        started_at = time.monotonic()
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_remote_bash_soft_timeout",
                name="bash",
                arguments={
                    "command": "sleep 35; printf 'late\\n'",
                    "timeout_seconds": 180,
                },
                raw_arguments=(
                    '{"command":"sleep 35; printf \\"late\\\\n\\"","timeout_seconds":180}'
                ),
            ),
            context=self.context,
        )
        duration_seconds = time.monotonic() - started_at

        self.assertTrue(result.ok)
        self.assertTrue(result.metadata["promoted_to_background"])
        self.assertIn("moved to background", result.content)
        self.assertLess(duration_seconds, 90.0)

        job_id = str(result.metadata["job_id"])
        try:
            status_result = await self.runtime.execute(
                tool_call=ToolCall(
                    call_id="call_remote_bash_soft_timeout_status",
                    name="bash",
                    arguments={"mode": "status", "job_id": job_id},
                    raw_arguments=f'{{"mode":"status","job_id":"{job_id}"}}',
                ),
                context=self.context,
            )
            self.assertTrue(status_result.ok)
            self.assertIn(status_result.metadata["status"], {"running", "finished"})
        finally:
            cancel_result = await self.runtime.execute(
                tool_call=ToolCall(
                    call_id="call_remote_bash_soft_timeout_cancel",
                    name="bash",
                    arguments={"mode": "cancel", "job_id": job_id},
                    raw_arguments=f'{{"mode":"cancel","job_id":"{job_id}"}}',
                ),
                context=self.context,
            )
            self.assertTrue(cancel_result.ok)

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_bash_background_job_lifecycle(self) -> None:
        start_result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_background_start",
                name="bash",
                arguments={
                    "mode": "background",
                    "command": "sleep 0.1; printf 'ready\\n'",
                },
                raw_arguments=(
                    '{"mode":"background","command":"sleep 0.1; printf \\"ready\\\\n\\""}'
                ),
            ),
            context=self.context,
        )

        self.assertTrue(start_result.ok)
        if not _REMOTE_TOOL_RUNTIME_CONFIGURED:
            self.assertEqual(start_result.metadata["status"], "running")
            self.assertEqual(start_result.metadata["state"], "running")
            self.assertIsNotNone(start_result.metadata["started_at"])
            self.assertIsNotNone(start_result.metadata["last_update_at"])
            self.assertEqual(start_result.metadata["suggested_next_check_seconds"], 5)
        job_id = str(start_result.metadata["job_id"])

        status_result = None
        for _ in range(10):
            status_result = await self.runtime.execute(
                tool_call=ToolCall(
                    call_id="call_bash_background_status",
                    name="bash",
                    arguments={"mode": "status", "job_id": job_id},
                    raw_arguments=f'{{"mode":"status","job_id":"{job_id}"}}',
                ),
                context=self.context,
            )
            self.assertTrue(status_result.ok)
            if status_result.metadata["status"] == "finished":
                break
            await asyncio.sleep(0.05)

        self.assertIsNotNone(status_result)
        if status_result is None:
            self.fail("Expected a bash background status result.")
        self.assertEqual(status_result.metadata["status"], "finished")
        self.assertEqual(status_result.metadata["exit_code"], 0)

        tail_result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_background_tail",
                name="bash",
                arguments={"mode": "tail", "job_id": job_id},
                raw_arguments=f'{{"mode":"tail","job_id":"{job_id}"}}',
            ),
            context=self.context,
        )

        self.assertTrue(tail_result.ok)
        self.assertIn("ready", tail_result.content)

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_bash_background_job_can_be_cancelled(self) -> None:
        start_result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_background_cancel_start",
                name="bash",
                arguments={
                    "mode": "background",
                    "command": "sleep 5",
                },
                raw_arguments='{"mode":"background","command":"sleep 5"}',
            ),
            context=self.context,
        )

        self.assertTrue(start_result.ok)
        job_id = str(start_result.metadata["job_id"])

        cancel_result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_background_cancel",
                name="bash",
                arguments={"mode": "cancel", "job_id": job_id},
                raw_arguments=f'{{"mode":"cancel","job_id":"{job_id}"}}',
            ),
            context=self.context,
        )

        self.assertTrue(cancel_result.ok)
        self.assertEqual(cancel_result.metadata["status"], "cancelled")

        status_result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_background_cancel_status",
                name="bash",
                arguments={"mode": "status", "job_id": job_id},
                raw_arguments=f'{{"mode":"status","job_id":"{job_id}"}}',
            ),
            context=self.context,
        )

        self.assertTrue(status_result.ok)
        self.assertEqual(status_result.metadata["status"], "cancelled")

    @unittest.skipUnless(
        _FFMPEG_BASH_RUNTIME_AVAILABLE,
        _FFMPEG_BASH_RUNTIME_SKIP_REASON,
    )
    async def test_bash_can_execute_ffmpeg_when_installed(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_ffmpeg_version",
                name="bash",
                arguments={"command": "ffmpeg -version | head -n 1"},
                raw_arguments='{"command":"ffmpeg -version | head -n 1"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("ffmpeg version", result.content)

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_bash_requires_approval_for_system_write_command(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_etc_read_only",
                name="bash",
                arguments={"command": "printf 'jarvis' > /etc/jarvis-bash-test"},
                raw_arguments='{"command":"printf \\"jarvis\\" > /etc/jarvis-bash-test"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("Approval required", result.content)
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(
            result.metadata["approval_request"]["kind"],
            "bash_command",
        )
        self.assertEqual(
            result.metadata["approval_request"]["command"],
            "printf 'jarvis' > /etc/jarvis-bash-test",
        )

    async def test_file_patch_creates_new_file_with_write_operation(self) -> None:
        scripts_dir = self.workspace_dir / "scripts"
        scripts_dir.mkdir()

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_file_patch_create",
                name="file_patch",
                arguments={
                    "path": "scripts/example.py",
                    "operations": [
                        {
                            "type": "write",
                            "content": "print('hello')\n",
                        }
                    ],
                },
                raw_arguments=(
                    '{"path":"scripts/example.py","operations":[{"type":"write",'
                    '"content":"print(\'hello\')\\n"}]}'
                ),
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("status: created", result.content)
        self.assertEqual(
            (scripts_dir / "example.py").read_text(encoding="utf-8"),
            "print('hello')\n",
        )
        self.assertEqual(result.metadata["status"], "created")
        self.assertTrue(result.metadata["file_created"])

    async def test_file_patch_applies_ordered_literal_edits(self) -> None:
        target_path = self.workspace_dir / "note.txt"
        target_path.write_text("alpha\nbeta\n", encoding="utf-8")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_file_patch_update",
                name="file_patch",
                arguments={
                    "path": "note.txt",
                    "operations": [
                        {
                            "type": "replace",
                            "old": "beta",
                            "new": "gamma",
                        },
                        {
                            "type": "insert_after",
                            "anchor": "gamma\n",
                            "text": "delta\n",
                        },
                    ],
                },
                raw_arguments=(
                    '{"path":"note.txt","operations":[{"type":"replace","old":"beta",'
                    '"new":"gamma"},{"type":"insert_after","anchor":"gamma\\n",'
                    '"text":"delta\\n"}]}'
                ),
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            target_path.read_text(encoding="utf-8"),
            "alpha\ngamma\ndelta\n",
        )
        self.assertEqual(result.metadata["status"], "updated")
        self.assertEqual(result.metadata["operation_types"], ["replace", "insert_after"])

    async def test_file_patch_fails_when_anchor_is_missing(self) -> None:
        target_path = self.workspace_dir / "note.txt"
        target_path.write_text("alpha\n", encoding="utf-8")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_file_patch_missing_anchor",
                name="file_patch",
                arguments={
                    "path": "note.txt",
                    "operations": [
                        {
                            "type": "insert_before",
                            "anchor": "beta\n",
                            "text": "delta\n",
                        }
                    ],
                },
                raw_arguments=(
                    '{"path":"note.txt","operations":[{"type":"insert_before",'
                    '"anchor":"beta\\n","text":"delta\\n"}]}'
                ),
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("could not find a unique anchor match", result.content)
        self.assertEqual(target_path.read_text(encoding="utf-8"), "alpha\n")

    async def test_file_patch_does_not_partially_write_on_failed_later_operation(self) -> None:
        target_path = self.workspace_dir / "note.txt"
        target_path.write_text("alpha\nbeta\n", encoding="utf-8")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_file_patch_atomic",
                name="file_patch",
                arguments={
                    "path": "note.txt",
                    "operations": [
                        {
                            "type": "replace",
                            "old": "beta",
                            "new": "gamma",
                        },
                        {
                            "type": "insert_after",
                            "anchor": "missing\n",
                            "text": "delta\n",
                        },
                    ],
                },
                raw_arguments=(
                    '{"path":"note.txt","operations":[{"type":"replace","old":"beta",'
                    '"new":"gamma"},{"type":"insert_after","anchor":"missing\\n",'
                    '"text":"delta\\n"}]}'
                ),
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("operation 2", result.content)
        self.assertEqual(target_path.read_text(encoding="utf-8"), "alpha\nbeta\n")

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_returns_policy_error_for_null_byte_command(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_denied",
                name="bash",
                arguments={"command": "printf 'hello'\x00"},
                raw_arguments='{"command":"printf \\"hello\\"\\u0000"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("denied", result.content.lower())
        self.assertFalse((self.workspace_dir / "note.txt").exists())

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_bash_cannot_read_repo(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_repo_blocked",
                name="bash",
                arguments={"command": "ls /repo"},
                raw_arguments='{"command":"ls /repo"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("/repo", result.content)
        self.assertIn("No such file", result.content)

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_bash_cannot_read_run_secrets(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_secrets_blocked",
                name="bash",
                arguments={"command": "ls /run/secrets"},
                raw_arguments='{"command":"ls /run/secrets"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("/run/secrets", result.content)
        self.assertTrue(
            "No such file" in result.content
            or "Permission denied" in result.content
        )

    async def test_tool_register_requires_approval_before_writing_manifest(self) -> None:
        manifest = {
            "name": "google_workspace_cli",
            "purpose": "Use the Google Workspace CLI from bash.",
            "operator": "bash",
            "invocation": {"command": "gws --help"},
        }

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_register_requires_approval",
                name="tool_register",
                arguments={"manifest": manifest},
                raw_arguments="{}",
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(
            result.metadata["approval_request"]["kind"],
            "register_runtime_tool",
        )
        self.assertFalse(
            (self.workspace_dir / "runtime_tools" / "google_workspace_cli.json").exists()
        )

    async def test_tool_register_writes_manifest_when_exact_approval_is_present(self) -> None:
        manifest_payload = {
            "name": "google_workspace_cli",
            "purpose": "Use the Google Workspace CLI from bash.",
            "operator": "bash",
            "invocation": {"command": "gws --help"},
        }
        manifest = validate_runtime_tool_manifest_payload(manifest_payload)

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_register_approved",
                name="tool_register",
                arguments={"manifest": manifest_payload},
                raw_arguments="{}",
            ),
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                approved_action={
                    "kind": "register_runtime_tool",
                    "tool_name": manifest.name,
                    "manifest_hash": manifest.manifest_hash(),
                },
            ),
        )

        self.assertTrue(result.ok)
        manifest_path = self.workspace_dir / "runtime_tools" / "google_workspace_cli.json"
        self.assertTrue(manifest_path.exists())
        self.assertIn("Runtime tool registered", result.content)
        self.assertEqual(result.metadata["manifest_hash"], manifest.manifest_hash())
        self.assertIn('"name": "google_workspace_cli"', manifest_path.read_text(encoding="utf-8"))

    async def test_tool_search_includes_runtime_registered_tools_from_workspace(self) -> None:
        _write_runtime_manifest(
            self.workspace_dir,
            {
                "name": "google_workspace_cli",
                "purpose": "Use the Google Workspace CLI from bash.",
                "operator": "bash",
                "invocation": {"command": "gws --help"},
                "rebuild": {"check": "command -v gws"},
            },
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_runtime",
                name="tool_search",
                arguments={"query": "google_workspace_cli", "verbosity": "high"},
                raw_arguments='{"query":"google_workspace_cli","verbosity":"high"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertGreaterEqual(result.metadata["match_count"], 1)
        self.assertEqual(result.metadata["matches"][0]["name"], "google_workspace_cli")
        self.assertNotIn(
            "google_workspace_cli",
            result.metadata["activated_discoverable_tool_names"],
        )
        self.assertEqual(
            result.metadata["matches"][0]["metadata"]["source"],
            "runtime_tools",
        )
        self.assertEqual(
            result.metadata["matches"][0]["usage"]["operator"],
            "bash",
        )
        self.assertIn("google_workspace_cli", result.content)
        self.assertIn("source: runtime_tools", result.content)
        self.assertIn('"operator": "bash"', result.content)
        self.assertNotIn("manifest_path", result.content)

    async def test_tool_search_reports_runtime_manifest_errors(self) -> None:
        runtime_dir = self.workspace_dir / "runtime_tools"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "broken.json").write_text("{bad json", encoding="utf-8")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_runtime_errors",
                name="tool_search",
                arguments={},
                raw_arguments="{}",
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(result.metadata["runtime_tool_errors"]), 1)
        self.assertIn("broken.json", result.metadata["runtime_tool_errors"][0])

    async def test_email_requires_approval_before_sending(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_email_requires_approval",
                name="email",
                arguments={
                    "to_email": "user@example.com",
                    "subject": "Weekly update",
                    "body": "Hello there.",
                },
                raw_arguments=(
                    '{"to_email":"user@example.com","subject":"Weekly update",'
                    '"body":"Hello there."}'
                ),
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.metadata["approval_required"])
        self.assertEqual(
            result.metadata["approval_request"]["kind"],
            "send_email",
        )
        self.assertIn("user@example.com", result.metadata["approval_request"]["details"])

    async def test_email_sends_with_html_footer_and_attachment_when_exact_approval_is_present(
        self,
    ) -> None:
        attachment_path = self.workspace_dir / "exports" / "report.txt"
        attachment_path.parent.mkdir(parents=True, exist_ok=True)
        attachment_path.write_text("ship it\n", encoding="utf-8")
        arguments = {
            "to_email": "yourmainemail@example.com",
            "subject": "Weekly update",
            "body": "# Status\n\n**Ready** for review.\n\n- Ship build\n- Send update",
            "attachment_paths": ["exports/report.txt"],
        }
        captured: dict[str, object] = {}

        class _FakeSMTPSSL:
            def __init__(self, host: str, port: int, timeout: float) -> None:
                captured["host"] = host
                captured["port"] = port
                captured["timeout"] = timeout

            def __enter__(self) -> "_FakeSMTPSSL":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = exc_type, exc, tb
                captured["closed"] = True

            def login(self, username: str, password: str) -> None:
                captured["login"] = (username, password)

            def send_message(self, message) -> None:
                captured["raw_message"] = message.as_bytes()

        with patch.dict(
            os.environ,
            {
                "SENDER_EMAIL_ADDRESS": "myagentalerts@gmail.com",
                "SMTP_PASSWORD": "test-app-password",
            },
            clear=False,
        ):
            settings = ToolSettings.from_workspace_dir(self.workspace_dir)
            runtime = ToolRuntime(registry=ToolRegistry.default(settings))
            approval_result = await runtime.execute(
                tool_call=ToolCall(
                    call_id="call_email_approval_hash",
                    name="email",
                    arguments=arguments,
                    raw_arguments="{}",
                ),
                context=ToolExecutionContext(workspace_dir=self.workspace_dir),
            )

            if not approval_result.metadata.get("approval_request"):
                self.fail("Expected approval request metadata for email.")
            request_hash = approval_result.metadata["approval_request"]["request_hash"]

            with patch(
                "jarvis.tools.discoverable.email.tool.smtplib.SMTP_SSL",
                _FakeSMTPSSL,
            ):
                result = await runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_email_send",
                        name="email",
                        arguments=arguments,
                        raw_arguments="{}",
                    ),
                    context=ToolExecutionContext(
                        workspace_dir=self.workspace_dir,
                        approved_action={
                            "kind": "send_email",
                            "request_hash": request_hash,
                        },
                    ),
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["from_email"], "myagentalerts@gmail.com")
        self.assertEqual(result.metadata["to_email"], "yourmainemail@example.com")
        self.assertEqual(result.metadata["attachment_count"], 1)
        self.assertEqual(
            captured["login"],
            ("myagentalerts@gmail.com", "test-app-password"),
        )
        self.assertEqual(captured["host"], "smtp.gmail.com")
        self.assertEqual(captured["port"], 465)
        self.assertTrue(captured["closed"])

        message = BytesParser(policy=email_parser_policy.default).parsebytes(
            captured["raw_message"]  # type: ignore[arg-type]
        )
        self.assertEqual(message["From"], "myagentalerts@gmail.com")
        self.assertEqual(message["To"], "yourmainemail@example.com")
        self.assertEqual(message["Subject"], "Weekly update")
        plain_part = message.get_body(preferencelist=("plain",))
        html_part = message.get_body(preferencelist=("html",))
        self.assertIsNotNone(plain_part)
        self.assertIsNotNone(html_part)
        if plain_part is None or html_part is None:
            self.fail("Expected both plain-text and HTML email bodies.")
        self.assertIn("Sent by Jarvis", plain_part.get_content())
        self.assertIn("Sent by Jarvis", html_part.get_content())
        self.assertIn("<strong>Ready</strong>", html_part.get_content())
        attachments = list(message.iter_attachments())
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0].get_filename(), "report.txt")
        self.assertEqual(attachments[0].get_content_type(), "text/plain")
        self.assertEqual(attachments[0].get_content().strip(), "ship it")

    @unittest.skipUnless(
        _BASH_RUNTIME_AVAILABLE,
        _BASH_RUNTIME_SKIP_REASON,
    )
    async def test_bash_scrubs_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "secret-key-marker",
                "CUSTOM_SECRET_FOR_TEST": "custom-secret-marker",
            },
            clear=False,
        ):
            result = await self.runtime.execute(
                tool_call=ToolCall(
                    call_id="call_env_scrubbed",
                    name="bash",
                    arguments={"command": "env | sort"},
                    raw_arguments='{"command":"env | sort"}',
                ),
                context=self.context,
            )

        self.assertTrue(result.ok)
        self.assertNotIn("OPENAI_API_KEY", result.content)
        self.assertNotIn("CUSTOM_SECRET_FOR_TEST", result.content)
        self.assertIn("HOME=/workspace", result.content)

    async def test_returns_tool_error_when_executor_raises(self) -> None:
        async def _failing_executor(*, call_id, arguments, context):
            _ = call_id, arguments, context
            raise RuntimeError("boom")

        registry = ToolRegistry(
            tools=(
                RegisteredTool(
                    name="bash",
                    exposure="basic",
                    definition=ToolDefinition(
                        name="bash",
                        description="Always fails.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "command": {"type": "string"},
                            },
                            "required": ["command"],
                            "additionalProperties": False,
                        },
                    ),
                    executor=_failing_executor,
                ),
            )
        )
        runtime = ToolRuntime(registry=registry)

        result = await runtime.execute(
            tool_call=ToolCall(
                call_id="call_fail",
                name="bash",
                arguments={"command": "pwd"},
                raw_arguments='{"command":"pwd"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("Tool execution failed", result.content)
        self.assertEqual(result.metadata["error_type"], "RuntimeError")

    async def test_memory_write_runtime_denies_summary_only_core_create(self) -> None:
        memory_service = MemoryService(
            settings=MemorySettings.from_workspace_dir(self.workspace_dir),
            llm_service=None,
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_memory_write_summary_only",
                name="memory_write",
                arguments={
                    "operation": "create",
                    "target_kind": "core",
                    "title": "Backend Preference",
                    "summary": "Scott prefers Python for backend APIs.",
                },
                raw_arguments='{"operation":"create","target_kind":"core","title":"Backend Preference","summary":"Scott prefers Python for backend APIs."}',
            ),
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                memory_service=memory_service,
            ),
        )

        self.assertFalse(result.ok)
        self.assertIn("Tool execution denied by policy", result.content)
        self.assertIn("summary is not a substitute", result.content)
        self.assertIn(
            "prefer real fact objects in facts whenever the user states an explicit durable fact",
            result.content.lower(),
        )

    async def test_memory_write_runtime_denies_daily_upsert_without_body_sections(self) -> None:
        memory_service = MemoryService(
            settings=MemorySettings.from_workspace_dir(self.workspace_dir),
            llm_service=None,
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_memory_write_daily_upsert_without_sections",
                name="memory_write",
                arguments={
                    "operation": "upsert",
                    "target_kind": "daily",
                    "document_id": "daily_2026-04-08",
                    "summary": "Scott went to Bray for a bike ride.",
                },
                raw_arguments='{"operation":"upsert","target_kind":"daily","document_id":"daily_2026-04-08","summary":"Scott went to Bray for a bike ride."}',
            ),
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                memory_service=memory_service,
            ),
        )

        self.assertFalse(result.ok)
        self.assertIn("Tool execution denied by policy", result.content)
        self.assertIn("daily upsert", result.content.lower())
        self.assertIn("memory_get", result.content)
        self.assertIn("body_sections", result.content)
        self.assertIn("summary alone does not rewrite prior daily content", result.content.lower())
        self.assertIn("append_daily", result.content)

    async def test_tool_runtime_returns_recoverable_error_for_invalid_model_tool_call(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_memory_write_invalid_shape",
                name="memory_write",
                arguments={
                    "operation": "upsert",
                    "target_kind": "daily",
                    "document_id": "daily_2026-04-08",
                    "body_sections": [
                        {
                            "heading": "Notable Events",
                            "body": "- Scott went to Bray for a bike ride.",
                        }
                    ],
                },
                raw_arguments=(
                    '{"operation":"upsert","target_kind":"daily","document_id":"daily_2026-04-08",'
                    '"body_sections":[{"heading":"Notable Events","body":"- Scott went to Bray for a bike ride."}]}'
                ),
                provider_metadata={
                    "tool_call_validation_error": (
                        "Tool 'memory_write' arguments failed schema validation: "
                        "[{'heading': 'Notable Events', 'body': '- Scott went to Bray for a bike ride.'}] "
                        "is not of type 'object'"
                    )
                },
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.name, "memory_write")
        self.assertIn("Tool execution failed", result.content)
        self.assertIn("ToolCallValidationError", result.content)
        self.assertIn("raw_arguments", result.content)
        self.assertIn("match the tool schema", result.content)
        self.assertTrue(result.metadata["tool_call_validation_failed"])

    async def test_memory_write_runtime_accepts_explicit_none_truth_decision(self) -> None:
        memory_service = MemoryService(
            settings=MemorySettings.from_workspace_dir(self.workspace_dir),
            llm_service=None,
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_memory_write_explicit_none",
                name="memory_write",
                arguments={
                    "operation": "create",
                    "target_kind": "core",
                    "title": "Backend Preference",
                    "summary": "Scott prefers Python for backend APIs.",
                    "facts": "None",
                    "relations": "None",
                },
                raw_arguments='{"operation":"create","target_kind":"core","title":"Backend Preference","summary":"Scott prefers Python for backend APIs.","facts":"None","relations":"None"}',
            ),
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                memory_service=memory_service,
            ),
        )

        self.assertTrue(result.ok)
        document = memory_service._store.read_document(
            self.workspace_dir / "memory" / "core" / "backend-preference.md"
        )
        self.assertEqual(document.facts, ())
        self.assertEqual(document.relations, ())

    async def test_memory_write_runtime_accepts_lowercase_none_truth_decision(self) -> None:
        memory_service = MemoryService(
            settings=MemorySettings.from_workspace_dir(self.workspace_dir),
            llm_service=None,
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_memory_write_lowercase_none",
                name="memory_write",
                arguments={
                    "operation": "create",
                    "target_kind": "core",
                    "title": "Backend Preference",
                    "summary": "Scott prefers Python for backend APIs.",
                    "facts": "none",
                    "relations": "none",
                },
                raw_arguments='{"operation":"create","target_kind":"core","title":"Backend Preference","summary":"Scott prefers Python for backend APIs.","facts":"none","relations":"none"}',
            ),
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                memory_service=memory_service,
            ),
        )

        self.assertTrue(result.ok)
        document = memory_service._store.read_document(
            self.workspace_dir / "memory" / "core" / "backend-preference.md"
        )
        self.assertEqual(document.facts, ())
        self.assertEqual(document.relations, ())

    async def test_memory_write_runtime_denies_string_items_in_truth_arrays(self) -> None:
        memory_service = MemoryService(
            settings=MemorySettings.from_workspace_dir(self.workspace_dir),
            llm_service=None,
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_memory_write_bad_truth_array",
                name="memory_write",
                arguments={
                    "operation": "create",
                    "target_kind": "ongoing",
                    "title": "Visual Pipeline Project",
                    "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                    "facts": ["Scott is building a Three.js/WebGPU/TSL shader playground."],
                    "relations": "None",
                },
                raw_arguments='{"operation":"create","target_kind":"ongoing","title":"Visual Pipeline Project","summary":"Scott is building a Three.js/WebGPU/TSL shader playground.","facts":["Scott is building a Three.js/WebGPU/TSL shader playground."],"relations":"None"}',
            ),
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                memory_service=memory_service,
            ),
        )

        self.assertFalse(result.ok)
        self.assertIn("Tool execution denied by policy", result.content)
        self.assertIn("array items must be objects", result.content)

    async def test_memory_write_runtime_denies_list_body_sections(self) -> None:
        memory_service = MemoryService(
            settings=MemorySettings.from_workspace_dir(self.workspace_dir),
            llm_service=None,
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_memory_write_bad_body_sections",
                name="memory_write",
                arguments={
                    "operation": "create",
                    "target_kind": "ongoing",
                    "title": "Visual Pipeline Project",
                    "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                    "facts": "None",
                    "relations": "None",
                    "body_sections": [
                        {
                            "title": "Overview",
                            "text": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                        }
                    ],
                },
                raw_arguments='{"operation":"create","target_kind":"ongoing","title":"Visual Pipeline Project","summary":"Scott is building a Three.js/WebGPU/TSL shader playground.","facts":"None","relations":"None","body_sections":[{"title":"Overview","text":"Scott is building a Three.js/WebGPU/TSL shader playground."}]}',
            ),
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                memory_service=memory_service,
            ),
        )

        self.assertFalse(result.ok)
        self.assertIn("Tool execution denied by policy", result.content)
        self.assertIn("body_sections", result.content)

    async def test_memory_write_runtime_reports_multiple_nested_errors_together(self) -> None:
        memory_service = MemoryService(
            settings=MemorySettings.from_workspace_dir(self.workspace_dir),
            llm_service=None,
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_memory_write_multi_error",
                name="memory_write",
                arguments={
                    "operation": "create",
                    "target_kind": "ongoing",
                    "title": "Visual Pipeline Project",
                    "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                    "facts": [{"statement": "Scott is building a Three.js/WebGPU/TSL shader playground."}],
                    "relations": [
                        {
                            "subject": "Scott",
                            "predicate": "is_building",
                            "object": "Visual Pipeline Project",
                            "status": "active",
                        }
                    ],
                    "body_sections": [
                        {
                            "title": "Overview",
                            "text": "Scott is building a Three.js/WebGPU/TSL shader playground.",
                        }
                    ],
                },
                raw_arguments=(
                    '{"operation":"create","target_kind":"ongoing","title":"Visual Pipeline Project",'
                    '"summary":"Scott is building a Three.js/WebGPU/TSL shader playground.",'
                    '"facts":[{"statement":"Scott is building a Three.js/WebGPU/TSL shader playground."}],'
                    '"relations":[{"subject":"Scott","predicate":"is_building","object":"Visual Pipeline Project","status":"active"}],'
                    '"body_sections":[{"title":"Overview","text":"Scott is building a Three.js/WebGPU/TSL shader playground."}]}'
                ),
            ),
            context=ToolExecutionContext(
                workspace_dir=self.workspace_dir,
                memory_service=memory_service,
            ),
        )

        self.assertFalse(result.ok)
        self.assertIn("Tool execution denied by policy", result.content)
        self.assertIn("facts[0].text", result.content)
        self.assertIn("relations[0].status", result.content)
        self.assertIn("body_sections", result.content)
        self.assertIn("Minimal valid example", result.content)

    @unittest.skipUnless(
        _CENTRAL_PYTHON_RUNTIME_AVAILABLE,
        "central Python runtime is only available when /opt/venv/bin/python exists",
    )
    async def test_bash_python_heredoc_executes_with_workspace_access(self) -> None:
        input_dir = self.workspace_dir / "inputs"
        output_dir = self.workspace_dir / "outputs"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "data.txt").write_text("hello", encoding="utf-8")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_python_inline",
                name="bash",
                arguments={
                    "command": _bash_python_heredoc(
                        "from pathlib import Path\n"
                        "text = Path('inputs/data.txt').read_text(encoding='utf-8')\n"
                        "Path('outputs/result.txt').write_text(text.upper(), encoding='utf-8')\n"
                        "print(text.upper())\n"
                    ),
                },
                raw_arguments='{"command":"python heredoc"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("Bash execution result", result.content)
        self.assertIn("HELLO", result.metadata["stdout"])
        self.assertEqual(result.metadata["cwd"], "/workspace")
        self.assertEqual(
            (output_dir / "result.txt").read_text(encoding="utf-8"),
            "HELLO",
        )

    @unittest.skipUnless(
        _CENTRAL_PYTHON_RUNTIME_AVAILABLE,
        "central Python runtime is only available when /opt/venv/bin/python exists",
    )
    async def test_bash_python_script_executes_with_curated_package(self) -> None:
        scripts_dir = self.workspace_dir / "scripts"
        exports_dir = self.workspace_dir / "exports"
        scripts_dir.mkdir()
        exports_dir.mkdir()
        script_path = scripts_dir / "emit_yaml.py"
        script_path.write_text(
            (
                "from pathlib import Path\n"
                "import sys\n"
                "import yaml\n"
                "\n"
                "payload = {'value': sys.argv[1]}\n"
                "target = Path('exports/payload.yml')\n"
                "target.write_text(yaml.safe_dump(payload), encoding='utf-8')\n"
                "print(payload['value'])\n"
            ),
            encoding="utf-8",
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_python_script",
                name="bash",
                arguments={"command": "python scripts/emit_yaml.py example"},
                raw_arguments='{"command":"python scripts/emit_yaml.py example"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("example", result.metadata["stdout"])
        self.assertIn(
            "value: example",
            (exports_dir / "payload.yml").read_text(encoding="utf-8"),
        )

    @unittest.skipUnless(
        _CENTRAL_PYTHON_RUNTIME_AVAILABLE,
        "central Python runtime is only available when /opt/venv/bin/python exists",
    )
    async def test_bash_python_allows_workspace_local_helper_imports(self) -> None:
        scripts_dir = self.workspace_dir / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "helper.py").write_text(
            "def message() -> str:\n"
            "    return 'local helper ok'\n",
            encoding="utf-8",
        )
        (scripts_dir / "main.py").write_text(
            "import helper\n"
            "print(helper.message())\n",
            encoding="utf-8",
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_python_local_import",
                name="bash",
                arguments={"command": "python scripts/main.py"},
                raw_arguments='{"command":"python scripts/main.py"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("local helper ok", result.metadata["stdout"])

    @unittest.skipUnless(
        _CENTRAL_PYTHON_RUNTIME_AVAILABLE,
        "central Python runtime is only available when /opt/venv/bin/python exists",
    )
    async def test_bash_python_supports_pymupdf_text_round_trip(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_python_pymupdf",
                name="bash",
                arguments={
                    "command": _bash_python_heredoc(
                        "import fitz\n"
                        "doc = fitz.open()\n"
                        "page = doc.new_page()\n"
                        "page.insert_text((72, 72), 'hello from pymupdf')\n"
                        "pdf_bytes = doc.tobytes()\n"
                        "reopened = fitz.open(stream=pdf_bytes, filetype='pdf')\n"
                        "print(reopened[0].cropbox)\n"
                        "print(reopened[0].get_text('text').strip())\n"
                    ),
                },
                raw_arguments='{"command":"python pymupdf heredoc"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("hello from pymupdf", result.metadata["stdout"])
        self.assertIn("Rect(0.0, 0.0, 595.0, 842.0)", result.metadata["stdout"])

    @unittest.skipUnless(
        _CENTRAL_PYTHON_RUNTIME_AVAILABLE,
        "central Python runtime is only available when /opt/venv/bin/python exists",
    )
    async def test_bash_python_can_edit_workspace_jpeg_with_pillow(self) -> None:
        input_path = self.workspace_dir / "cat_input.jpg"
        output_path = self.workspace_dir / "cat_output.jpg"
        input_path.write_bytes(b64decode(_SAMPLE_JPEG_BASE64))

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_python_jpeg_edit",
                name="bash",
                arguments={
                    "command": _bash_python_heredoc(
                        "from pathlib import Path\n"
                        "from PIL import Image, ImageFilter\n"
                        "source = Path('cat_input.jpg')\n"
                        "target = Path('cat_output.jpg')\n"
                        "with Image.open(source) as image:\n"
                        "    square = image.crop((0, 0, 2, 2)).convert('L')\n"
                        "    square.filter(ImageFilter.GaussianBlur(radius=1)).save(target)\n"
                        "print(target)\n"
                    ),
                },
                raw_arguments='{"command":"python pillow heredoc"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("cat_output.jpg", result.metadata["stdout"])
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    @unittest.skipUnless(
        _CENTRAL_PYTHON_RUNTIME_AVAILABLE,
        "central Python runtime is only available when /opt/venv/bin/python exists",
    )
    async def test_bash_python_preserves_non_sandboxed_runtime_capabilities(self) -> None:
        outside_path = self.workspace_dir.parent / "jarvis-python-outside.txt"
        outside_path.unlink(missing_ok=True)

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_bash_python_capabilities",
                name="bash",
                arguments={
                    "command": _bash_python_heredoc(
                        "from pathlib import Path\n"
                        "import ctypes\n"
                        "import socket\n"
                        "import subprocess\n"
                        "\n"
                        "target = Path('../jarvis-python-outside.txt')\n"
                        "target.write_text('allowed', encoding='utf-8')\n"
                        "sock = socket.socket()\n"
                        "sock.settimeout(1)\n"
                        "completed = subprocess.run(['echo', 'hello'], check=True, capture_output=True, text=True)\n"
                        "print(target.read_text(encoding='utf-8'))\n"
                        "print(ctypes.sizeof(ctypes.c_int))\n"
                        "print(type(sock).__name__)\n"
                        "print(completed.stdout.strip())\n"
                        "sock.close()\n"
                    ),
                },
                raw_arguments='{"command":"python capabilities heredoc"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("allowed", result.metadata["stdout"])
        self.assertIn("4", result.metadata["stdout"])
        self.assertIn("socket", result.metadata["stdout"].lower())
        self.assertIn("hello", result.metadata["stdout"])
        self.assertTrue(outside_path.exists())
        outside_path.unlink(missing_ok=True)

    async def test_send_file_executes_for_workspace_file(self) -> None:
        report_path = self.workspace_dir / "exports" / "report.txt"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("hello", encoding="utf-8")
        context = ToolExecutionContext(
            workspace_dir=self.workspace_dir,
            route_id="tg_123",
        )

        async def _fake_send_telegram_file(**kwargs):
            self.assertEqual(kwargs["route_id"], "tg_123")
            self.assertEqual(
                kwargs["file_path"].resolve(strict=False),
                report_path.resolve(strict=False),
            )
            self.assertEqual(kwargs["caption"], "Attached report")
            self.assertEqual(kwargs["filename"], "weekly.txt")
            return {"message_id": 9, "chat_id": 123}

        with patch(
            "jarvis.tools.basic.send_file.tool.send_telegram_file",
            side_effect=_fake_send_telegram_file,
        ):
            result = await self.runtime.execute(
                tool_call=ToolCall(
                    call_id="call_send_file",
                    name="send_file",
                    arguments={
                        "path": "exports/report.txt",
                        "caption": "Attached report",
                        "filename": "weekly.txt",
                    },
                    raw_arguments=(
                        '{"path":"exports/report.txt","caption":"Attached report",'
                        '"filename":"weekly.txt"}'
                    ),
                ),
                context=context,
            )

        self.assertTrue(result.ok)
        self.assertIn("File sent to Telegram", result.content)
        self.assertEqual(result.metadata["chat_id"], 123)

    async def test_view_image_prepares_image_attachment(self) -> None:
        image_path = self.workspace_dir / "temp" / "sample.png"
        image_path.parent.mkdir()
        image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake_png_payload")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_view_image",
                name="view_image",
                arguments={"path": "temp/sample.png", "detail": "high"},
                raw_arguments='{"path":"temp/sample.png","detail":"high"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("Image attachment prepared", result.content)
        self.assertEqual(result.metadata["media_type"], "image/png")
        self.assertEqual(result.metadata["detail"], "high")
        self.assertEqual(
            Path(result.metadata["image_attachment"]["path"]).resolve(strict=False),
            image_path.resolve(strict=False),
        )

    async def test_generate_edit_image_executes_openai_generation(self) -> None:
        captured: dict[str, object] = {}
        fake_image_bytes = b"openai-image-bytes"
        fake_image_b64 = b64encode(fake_image_bytes).decode("ascii")

        class _FakeUsage:
            def model_dump(self, *, exclude_none: bool = False) -> dict[str, int]:
                _ = exclude_none
                return {"total_tokens": 12}

        class _FakeImagesAPI:
            def generate(self, **kwargs):
                captured["generate_kwargs"] = kwargs
                return type(
                    "_FakeOpenAIResponse",
                    (),
                    {
                        "data": [
                            type(
                                "_FakeOpenAIImage",
                                (),
                                {
                                    "b64_json": fake_image_b64,
                                    "revised_prompt": "Refined studio product photo prompt.",
                                },
                            )()
                        ],
                        "usage": _FakeUsage(),
                    },
                )()

        class _FakeOpenAIClient:
            def __init__(self, **kwargs) -> None:
                captured["client_kwargs"] = kwargs
                self.images = _FakeImagesAPI()

            def close(self) -> None:
                captured["closed"] = True

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=False):
            with patch(
                "jarvis.tools.discoverable.generate_edit_image.tool.OpenAI",
                _FakeOpenAIClient,
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_generate_edit_image_openai",
                        name="generate_edit_image",
                        arguments={
                            "prompt": "A matte black coffee grinder on a white studio background.",
                            "output_path": "artifacts/openai/grinder",
                            "provider": "openai",
                        },
                        raw_arguments=(
                            '{"prompt":"A matte black coffee grinder on a white studio '
                            'background.","output_path":"artifacts/openai/grinder","provider":"openai"}'
                        ),
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["provider"], "openai")
        self.assertEqual(
            result.metadata["model"],
            self.settings.generate_edit_image_openai_model,
        )
        self.assertEqual(result.metadata["mime_type"], "image/png")
        self.assertEqual(result.metadata["file_size_bytes"], len(fake_image_bytes))
        self.assertEqual(
            captured["generate_kwargs"],
            {
                "model": self.settings.generate_edit_image_openai_model,
                "prompt": "A matte black coffee grinder on a white studio background.",
                "quality": "medium",
                "output_format": "png",
            },
        )
        self.assertTrue(captured["closed"])
        self.assertEqual(result.metadata["quality"], "medium")
        output_path = Path(result.metadata["output_path"])
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.name, "grinder.png")
        self.assertEqual(output_path.read_bytes(), fake_image_bytes)

    async def test_generate_edit_image_executes_gemini_edit(self) -> None:
        captured: dict[str, object] = {}
        input_path = self.workspace_dir / "temp" / "input.png"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_bytes(b"\x89PNG\r\n\x1a\nfake_png_payload")
        gemini_model = self.settings.generate_edit_image_gemini_model

        class _FakeUsageMetadata:
            def model_dump(self, *, exclude_none: bool = False) -> dict[str, int]:
                _ = exclude_none
                return {"total_token_count": 21}

        class _FakeGeminiModels:
            def generate_content(self, *, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return type(
                    "_FakeGeminiResponse",
                    (),
                    {
                        "model_version": gemini_model,
                        "response_id": "resp_gemini_123",
                        "usage_metadata": _FakeUsageMetadata(),
                        "candidates": [
                            type(
                                "_FakeCandidate",
                                (),
                                {
                                    "content": type(
                                        "_FakeContent",
                                        (),
                                        {
                                            "parts": [
                                                type(
                                                    "_FakeImagePart",
                                                    (),
                                                    {
                                                        "text": None,
                                                        "inline_data": type(
                                                            "_FakeInlineData",
                                                            (),
                                                            {
                                                                "data": b"gemini-image-bytes",
                                                                "mime_type": "image/png",
                                                            },
                                                        )(),
                                                    },
                                                )(),
                                            ],
                                        },
                                    )(),
                                },
                            )()
                        ],
                    },
                )()

        class _FakeGeminiClient:
            def __init__(self, *, api_key: str) -> None:
                captured["api_key"] = api_key
                self.models = _FakeGeminiModels()

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}, clear=False):
            with patch(
                "jarvis.tools.discoverable.generate_edit_image.tool.genai.Client",
                _FakeGeminiClient,
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_generate_edit_image_gemini",
                        name="generate_edit_image",
                        arguments={
                            "prompt": "Replace the background with a sunrise gradient.",
                            "image_path": "temp/input.png",
                            "output_path": "artifacts/gemini/edited-input.png",
                        },
                        raw_arguments=(
                            '{"prompt":"Replace the background with a sunrise gradient.",'
                            '"image_path":"temp/input.png","output_path":"artifacts/gemini/edited-input.png"}'
                        ),
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["provider"], "gemini")
        self.assertEqual(
            result.metadata["model"],
            self.settings.generate_edit_image_gemini_model,
        )
        self.assertEqual(
            Path(result.metadata["image_path"]).resolve(strict=False),
            input_path.resolve(strict=False),
        )
        self.assertEqual(result.metadata["input_media_type"], "image/png")
        self.assertNotIn("provider_text", result.metadata)
        self.assertEqual(captured["api_key"], "test-google-key")
        self.assertEqual(captured["model"], self.settings.generate_edit_image_gemini_model)
        config = captured["config"]
        self.assertEqual(config.response_modalities, ["IMAGE"])
        self.assertEqual(config.image_config.image_size, "1K")
        self.assertEqual(result.metadata["resolution"], "1K")
        contents = captured["contents"]
        self.assertEqual(len(contents), 2)
        self.assertEqual(contents[1], "Replace the background with a sunrise gradient.")
        output_path = Path(result.metadata["output_path"])
        self.assertTrue(output_path.exists())
        self.assertEqual(output_path.name, "edited-input.png")
        self.assertEqual(output_path.read_bytes(), b"gemini-image-bytes")

    async def test_generate_edit_image_reports_gemini_no_image_diagnostics(self) -> None:
        gemini_model = self.settings.generate_edit_image_gemini_model

        class _FakeGeminiModels:
            def generate_content(self, *, model, contents, config):
                _ = model, contents, config
                return type(
                    "_FakeGeminiResponse",
                    (),
                    {
                        "model_version": gemini_model,
                        "response_id": "resp_gemini_missing_image",
                        "prompt_feedback": type(
                            "_FakePromptFeedback",
                            (),
                            {
                                "model_dump": staticmethod(
                                    lambda *, exclude_none=False: {
                                        "block_reason": "IMAGE_OTHER",
                                    }
                                )
                            },
                        )(),
                        "candidates": [
                            type(
                                "_FakeCandidate",
                                (),
                                {
                                    "finish_reason": "NO_IMAGE",
                                    "finish_message": "The model did not generate an image.",
                                    "content": type(
                                        "_FakeContent",
                                        (),
                                        {
                                            "parts": [
                                                type(
                                                    "_FakeTextPart",
                                                    (),
                                                    {
                                                        "text": "No image was produced.",
                                                        "inline_data": None,
                                                    },
                                                )(),
                                            ],
                                        },
                                    )(),
                                },
                            )()
                        ],
                        "text": "No image was produced.",
                    },
                )()

        class _FakeGeminiClient:
            def __init__(self, *, api_key: str) -> None:
                _ = api_key
                self.models = _FakeGeminiModels()

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}, clear=False):
            with patch(
                "jarvis.tools.discoverable.generate_edit_image.tool.genai.Client",
                _FakeGeminiClient,
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_generate_edit_image_gemini_missing_image",
                        name="generate_edit_image",
                        arguments={
                            "prompt": "An abstract machine mind in darkness.",
                            "output_path": "artifacts/gemini/missing-image.png",
                        },
                        raw_arguments=(
                            '{"prompt":"An abstract machine mind in darkness.",'
                            '"output_path":"artifacts/gemini/missing-image.png"}'
                        ),
                    ),
                    context=self.context,
                )

        self.assertFalse(result.ok)
        self.assertIn("Gemini did not return an inline image payload.", result.content)
        self.assertIn("candidate_count=1", result.content)
        self.assertIn("prompt_block_reason=IMAGE_OTHER", result.content)
        self.assertIn("candidate_finish_reason=NO_IMAGE", result.content)
        self.assertIn("candidate_part_types=text", result.content)
        self.assertIn("response_text=No image was produced.", result.content)

    async def test_transcribe_executes_openai_transcription(self) -> None:
        captured: dict[str, object] = {}
        audio_path = self.workspace_dir / "temp" / "meeting.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        class _FakeTranscriptionResponse:
            text = "Hello from the meeting transcript."

            @staticmethod
            def model_dump(*, exclude_none: bool = False) -> dict[str, object]:
                _ = exclude_none
                return {
                    "text": "Hello from the meeting transcript.",
                    "language": "en",
                    "duration": 12.345,
                }

        class _FakeTranscriptionsAPI:
            def create(self, **kwargs):
                captured["model"] = kwargs["model"]
                captured["response_format"] = kwargs["response_format"]
                captured["uploaded_file_name"] = kwargs["file"].name
                return _FakeTranscriptionResponse()

        class _FakeAudioAPI:
            def __init__(self) -> None:
                self.transcriptions = _FakeTranscriptionsAPI()

        class _FakeOpenAIClient:
            def __init__(self, **kwargs) -> None:
                captured["client_kwargs"] = kwargs
                self.audio = _FakeAudioAPI()

            def close(self) -> None:
                captured["closed"] = True

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=False):
            with patch(
                "jarvis.tools.discoverable.transcribe.tool.OpenAI",
                _FakeOpenAIClient,
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_transcribe_openai",
                        name="transcribe",
                        arguments={"audio_path": "temp/meeting.wav"},
                        raw_arguments='{"audio_path":"temp/meeting.wav"}',
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["model"], "gpt-4o-mini-transcribe")
        self.assertEqual(result.metadata["response_format"], "json")
        self.assertEqual(result.metadata["input_format"], "wav")
        self.assertEqual(result.metadata["language"], "en")
        self.assertEqual(result.metadata["duration_seconds"], 12.345)
        self.assertEqual(
            result.metadata["transcript_char_count"],
            len("Hello from the meeting transcript."),
        )
        self.assertEqual(
            captured["client_kwargs"],
            {
                "api_key": "test-openai-key",
                "timeout": 60.0,
                "max_retries": 2,
            },
        )
        self.assertEqual(captured["model"], "gpt-4o-mini-transcribe")
        self.assertEqual(captured["response_format"], "json")
        self.assertEqual(
            Path(captured["uploaded_file_name"]).resolve(strict=False),
            audio_path.resolve(strict=False),
        )
        self.assertTrue(captured["closed"])
        self.assertIn("Audio transcription completed", result.content)
        self.assertIn("transcript:", result.content)
        self.assertIn("Hello from the meeting transcript.", result.content)

    async def test_transcribe_rejects_oversized_audio_before_openai_call(self) -> None:
        audio_path = self.workspace_dir / "temp" / "oversized.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with audio_path.open("wb") as handle:
            handle.truncate((25 * 1024 * 1024) + 1)

        class _FailIfCalledOpenAIClient:
            def __init__(self, **kwargs) -> None:
                _ = kwargs
                raise AssertionError("OpenAI client should not be created for oversized audio.")

        with patch(
            "jarvis.tools.discoverable.transcribe.tool.OpenAI",
            _FailIfCalledOpenAIClient,
        ):
            result = await self.runtime.execute(
                tool_call=ToolCall(
                    call_id="call_transcribe_oversized",
                    name="transcribe",
                    arguments={"audio_path": "temp/oversized.wav"},
                    raw_arguments='{"audio_path":"temp/oversized.wav"}',
                ),
                context=self.context,
            )

        self.assertFalse(result.ok)
        self.assertIn("25 MB limit", result.content)

    async def test_web_search_executes_and_normalizes_results(self) -> None:
        def _fake_requests_get(url, *, params, headers, timeout):
            self.assertEqual(url, "https://api.search.brave.com/res/v1/web/search")
            self.assertEqual(params["q"], "brave browser")
            self.assertEqual(params["count"], "10")
            self.assertEqual(params["result_filter"], "web")
            self.assertEqual(params["spellcheck"], "false")
            self.assertEqual(headers["X-Subscription-Token"], "test-brave-key")
            self.assertEqual(timeout, 15.0)
            return _FakeWebSearchResponse(
                status_code=200,
                headers={
                    "X-RateLimit-Limit": "1000",
                    "X-RateLimit-Remaining": "999",
                },
                payload={
                    "type": "search",
                    "query": {
                        "original": "brave browser",
                        "cleaned": "brave browser",
                        "more_results_available": True,
                    },
                    "web": {
                        "type": "search",
                        "family_friendly": True,
                        "results": [
                            {
                                "title": "Brave Browser",
                                "url": "https://brave.com/",
                                "description": "Browse privately with Brave.",
                                "language": "en",
                                "page_age": "2026-02-01T00:00:00Z",
                                "meta_url": {"hostname": "brave.com"},
                            },
                            {
                                "title": "Brave Search",
                                "url": "https://search.brave.com/",
                                "description": "Private independent search.",
                            },
                        ],
                    },
                },
            )

        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-brave-key"}, clear=False):
            with patch("jarvis.tools.basic.web_search.tool.requests.get", side_effect=_fake_requests_get):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_web_search",
                        name="web_search",
                        arguments={"query": "brave browser"},
                        raw_arguments='{"query":"brave browser"}',
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertIn("Web search results", result.content)
        self.assertIn("Brave Browser", result.content)
        self.assertEqual(result.metadata["configured_result_count"], 10)
        self.assertEqual(result.metadata["result_count"], 2)
        self.assertEqual(result.metadata["query"]["original"], "brave browser")
        self.assertTrue(result.metadata["query"]["more_results_available"])
        self.assertEqual(result.metadata["results"][0]["source"], "brave.com")

    async def test_web_search_returns_error_when_api_key_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            result = await self.runtime.execute(
                tool_call=ToolCall(
                    call_id="call_web_search_missing_key",
                    name="web_search",
                    arguments={"query": "brave browser"},
                    raw_arguments='{"query":"brave browser"}',
                ),
                context=self.context,
            )

        self.assertFalse(result.ok)
        self.assertIn("BRAVE_SEARCH_API_KEY", result.content)

    async def test_web_search_returns_api_error_details(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-brave-key"}, clear=False):
            with patch(
                "jarvis.tools.basic.web_search.tool.requests.get",
                return_value=_FakeWebSearchResponse(
                    status_code=429,
                    payload={"error": {"detail": "rate limit exceeded"}},
                    text='{"error":{"detail":"rate limit exceeded"}}',
                ),
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_web_search_rate_limit",
                        name="web_search",
                        arguments={"query": "brave browser"},
                        raw_arguments='{"query":"brave browser"}',
                    ),
                    context=self.context,
                )

        self.assertFalse(result.ok)
        self.assertEqual(result.metadata["status_code"], 429)
        self.assertIn("rate limit exceeded", result.content)

    async def test_tool_search_returns_empty_low_verbosity_listing_when_no_discoverables_exist(self) -> None:
        registry = ToolRegistry()
        registry.register(build_tool_search_tool(registry))
        runtime = ToolRuntime(registry=registry)

        result = await runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_empty",
                name="tool_search",
                arguments={},
                raw_arguments="{}",
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("Tool search result", result.content)
        self.assertIn("verbosity: low", result.content)
        self.assertEqual(result.metadata["match_count"], 0)
        self.assertEqual(result.metadata["activated_discoverable_tool_names"], [])

    async def test_tool_search_lists_default_generate_edit_image_without_activation_at_low_verbosity(
        self,
    ) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_generate_edit_image",
                name="tool_search",
                arguments={"query": "image", "verbosity": "low"},
                raw_arguments='{"query":"image","verbosity":"low"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("generate_edit_image", result.content)
        self.assertEqual(result.metadata["activated_discoverable_tool_names"], [])
        self.assertEqual(
            [match["name"] for match in result.metadata["matches"]],
            ["generate_edit_image"],
        )

    async def test_tool_search_returns_high_verbosity_details_and_activation_metadata(self) -> None:
        archive_tool = RegisteredTool(
            name="archive",
            exposure="discoverable",
            definition=ToolDefinition(
                name="archive",
                description="Archive workspace files.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            executor=AsyncMock(),
        )
        self.registry.register(archive_tool)
        self.registry.register_discoverable(
            DiscoverableTool(
                name="archive",
                aliases=("zip_tools",),
                purpose="List, extract, and create archive formats inside the workspace.",
                detailed_description="Use this for zip or tar workflows in the workspace.",
                usage={
                    "arguments": [
                        {
                            "name": "path",
                            "type": "string",
                        }
                    ]
                },
                metadata={"family": "filesystem"},
                backing_tool_name="archive",
            )
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_archive",
                name="tool_search",
                arguments={"query": "zip", "verbosity": "high"},
                raw_arguments='{"query":"zip","verbosity":"high"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("aliases: zip_tools", result.content)
        self.assertIn("detailed_description: Use this for zip or tar workflows in the workspace.", result.content)
        self.assertNotIn("usage:", result.content)
        self.assertNotIn("metadata:", result.content)
        self.assertNotIn("backing_tool_name:", result.content)
        self.assertEqual(result.metadata["match_count"], 1)
        self.assertEqual(
            result.metadata["activated_discoverable_tool_names"],
            ["archive"],
        )

    async def test_tool_search_high_verbosity_ffmpeg_is_docs_only_and_not_activated(
        self,
    ) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_ffmpeg",
                name="tool_search",
                arguments={"query": "ffmpeg", "verbosity": "high"},
                raw_arguments='{"query":"ffmpeg","verbosity":"high"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("ffmpeg", result.content)
        self.assertIn("Use the basic `bash` tool", result.content)
        self.assertEqual(result.metadata["activated_discoverable_tool_names"], [])
        self.assertEqual(
            [match["name"] for match in result.metadata["matches"]],
            ["ffmpeg"],
        )

    async def test_tool_search_high_verbosity_transcribe_activates_tool(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_transcribe",
                name="tool_search",
                arguments={"query": "transcribe", "verbosity": "high"},
                raw_arguments='{"query":"transcribe","verbosity":"high"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("transcribe", result.content)
        self.assertIn("25 MB", result.content)
        self.assertNotIn("usage:", result.content)
        self.assertEqual(result.metadata["activated_discoverable_tool_names"], ["transcribe"])
        self.assertEqual(
            [match["name"] for match in result.metadata["matches"]],
            ["transcribe"],
        )

    async def test_tool_search_high_verbosity_email_activates_tool(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_email",
                name="tool_search",
                arguments={"query": "email", "verbosity": "high"},
                raw_arguments='{"query":"email","verbosity":"high"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("email", result.content)
        self.assertIn("SMTP", result.content)
        self.assertNotIn("usage:", result.content)
        self.assertEqual(result.metadata["activated_discoverable_tool_names"], ["email"])
        self.assertEqual(
            [match["name"] for match in result.metadata["matches"]],
            ["email"],
        )

    async def test_web_fetch_returns_source_markdown_without_defuddle_fallback(self) -> None:
        requested_url = "https://example.com/page"
        tier1_result = _build_http_fetch_result(
            requested_url=requested_url,
            content_type="text/markdown",
            headers={
                "Content-Type": "text/markdown",
                "X-Markdown-Tokens": "41",
                "Content-Signal": "search=yes",
            },
            body_text="# Example\n\nHello from Tier 1.",
        )

        with patch(
            "jarvis.tools.basic.web_fetch.tool._fetch_http_text",
            return_value=tier1_result,
        ) as fetch_mock:
            with patch(
                "jarvis.tools.basic.web_fetch.tool.RemoteToolRuntimeClient.execute",
                new=AsyncMock(side_effect=AssertionError("Defuddle should not be used.")),
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_web_fetch_tier1",
                        name="web_fetch",
                        arguments={"url": requested_url},
                        raw_arguments='{"url":"https://example.com/page"}',
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertEqual(fetch_mock.call_count, 1)
        self.assertEqual(result.metadata["final_url"], requested_url)
        self.assertEqual(result.metadata["status_code"], 200)
        self.assertEqual(result.metadata["markdown_tokens"], 41)
        self.assertEqual(result.metadata["content_signal"], "search=yes")
        self.assertIn("Hello from Tier 1.", result.content)

    async def test_web_fetch_routes_youtube_x_and_reddit_urls_to_remote_defuddle_only(self) -> None:
        forced_urls = (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://x.com/openai/status/1234567890",
            "https://www.reddit.com/r/python/comments/abc123/example_post/",
        )

        for index, requested_url in enumerate(forced_urls, start=1):
            with self.subTest(url=requested_url):
                expected_result = _build_web_fetch_tool_result(
                    requested_url=requested_url,
                    markdown="# Example\n\nHello from Defuddle.",
                )
                expected_result = ToolExecutionResult(
                    call_id=f"call_web_fetch_remote_{index}",
                    name="web_fetch",
                    ok=expected_result.ok,
                    content=expected_result.content,
                    metadata=expected_result.metadata,
                )

                with patch(
                    "jarvis.tools.basic.web_fetch.tool.RemoteToolRuntimeClient.execute",
                    new=AsyncMock(return_value=expected_result),
                ) as execute_mock:
                    with patch(
                        "jarvis.tools.basic.web_fetch.tool._fetch_http_text",
                        side_effect=AssertionError("HTTP fallback should not run."),
                    ):
                        result = await self.runtime.execute(
                            tool_call=ToolCall(
                                call_id=f"call_web_fetch_remote_{index}",
                                name="web_fetch",
                                arguments={"url": requested_url},
                                raw_arguments=f'{{"url":"{requested_url}"}}',
                            ),
                            context=self.context,
                        )

                self.assertTrue(result.ok)
                execute_mock.assert_awaited_once()
                self.assertEqual(execute_mock.await_args.kwargs["tool_name"], "web_fetch")
                self.assertEqual(
                    execute_mock.await_args.kwargs["arguments"],
                    {"url": requested_url},
                )

    async def test_web_fetch_falls_back_from_defuddle_to_cloudflare_conversion(self) -> None:
        requested_url = "https://example.com/article"
        tier1_result = _build_http_fetch_result(
            requested_url=requested_url,
            content_type="text/html",
            body_text="<html><body><div>Not markdown</div></body></html>",
        )
        tier2_result = _build_http_fetch_result(
            requested_url=requested_url,
            content_type="text/html",
            body_text=(
                "<html><body><article><h1>Example</h1><p>Hello from HTML.</p>"
                "</article></body></html>"
            ),
        )
        defuddle_failure = ToolExecutionResult(
            call_id="web_fetch_defuddle_step",
            name="web_fetch",
            ok=False,
            content="\n".join(
                [
                    "Web fetch failed",
                    f"url: {requested_url}",
                    "reason: Defuddle fetch failed.",
                ]
            ),
            metadata={
                "requested_url": requested_url,
                "error": "Defuddle fetch failed.",
            },
        )
        converted_result = MarkdownConversionResult(
            markdown="# Example\n\nHello from HTML.",
            markdown_tokens=88,
        )

        with patch(
            "jarvis.tools.basic.web_fetch.tool._fetch_http_text",
            side_effect=[tier1_result, tier2_result],
        ) as fetch_mock:
            with patch(
                "jarvis.tools.basic.web_fetch.tool.RemoteToolRuntimeClient.execute",
                new=AsyncMock(return_value=defuddle_failure),
            ) as execute_mock:
                with patch(
                    "jarvis.tools.basic.web_fetch.tool._convert_html_to_markdown",
                    return_value=converted_result,
                ) as convert_mock:
                    result = await self.runtime.execute(
                        tool_call=ToolCall(
                            call_id="call_web_fetch_cloudflare_fallback",
                            name="web_fetch",
                            arguments={"url": requested_url},
                            raw_arguments='{"url":"https://example.com/article"}',
                        ),
                        context=self.context,
                    )

        self.assertTrue(result.ok)
        self.assertEqual(fetch_mock.call_count, 2)
        execute_mock.assert_awaited_once()
        convert_mock.assert_called_once()
        self.assertEqual(result.metadata["final_url"], requested_url)
        self.assertEqual(result.metadata["status_code"], 200)
        self.assertEqual(result.metadata["markdown_tokens"], 88)
        self.assertIn("Hello from HTML.", result.content)

    async def test_web_fetch_continues_after_tier1_http_error(self) -> None:
        requested_url = "https://example.com/protected"
        expected_result = _build_web_fetch_tool_result(
            requested_url=requested_url,
            markdown="# Example\n\nHello from Defuddle.",
        )
        expected_result = ToolExecutionResult(
            call_id="call_web_fetch_tier1_403",
            name="web_fetch",
            ok=expected_result.ok,
            content=expected_result.content,
            metadata=expected_result.metadata,
        )

        with patch(
            "jarvis.tools.basic.web_fetch.tool._fetch_http_text",
            side_effect=WebFetchRequestError(
                "request returned HTTP 403.",
                status_code=403,
            ),
        ):
            with patch(
                "jarvis.tools.basic.web_fetch.tool.RemoteToolRuntimeClient.execute",
                new=AsyncMock(return_value=expected_result),
            ) as execute_mock:
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_web_fetch_tier1_403",
                        name="web_fetch",
                        arguments={"url": requested_url},
                        raw_arguments='{"url":"https://example.com/protected"}',
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        execute_mock.assert_awaited_once()
        self.assertIn("Hello from Defuddle.", result.content)

    async def test_web_fetch_falls_back_to_playwright_and_remote_defuddle_html(self) -> None:
        requested_url = "https://example.com/protected"
        initial_defuddle_failure = ToolExecutionResult(
            call_id="call_web_fetch_browser_fallback",
            name="web_fetch",
            ok=False,
            content="\n".join(
                [
                    "Web fetch failed",
                    f"url: {requested_url}",
                    "reason: Defuddle fetch failed.",
                ]
            ),
            metadata={
                "requested_url": requested_url,
                "error": "Defuddle fetch failed.",
            },
        )
        rendered_defuddle_success = ToolExecutionResult(
            call_id="call_web_fetch_browser_fallback",
            name="web_fetch",
            ok=True,
            content="\n".join(
                [
                    "Web fetch result",
                    f"url: {requested_url}",
                    "markdown:",
                    "# Rendered Example\n\nHello from browser HTML.",
                ]
            ),
            metadata={
                "requested_url": requested_url,
                "markdown_chars": 37,
                "markdown_truncated": False,
            },
        )
        rendered_result = BrowserRenderResult(
            requested_url=requested_url,
            final_url=requested_url,
            html=(
                "<html><body><main><h1>Rendered Example</h1>"
                "<p>Hello from browser HTML.</p></main></body></html>"
            ),
        )

        with patch(
            "jarvis.tools.basic.web_fetch.tool._fetch_http_text",
            side_effect=[
                WebFetchRequestError("request returned HTTP 403.", status_code=403),
                WebFetchRequestError("request returned HTTP 403.", status_code=403),
            ],
        ):
            with patch(
                "jarvis.tools.basic.web_fetch.tool.RemoteToolRuntimeClient.execute",
                new=AsyncMock(
                    side_effect=[
                        initial_defuddle_failure,
                        rendered_defuddle_success,
                    ]
                ),
            ) as execute_mock:
                with patch(
                    "jarvis.tools.basic.web_fetch.tool._render_page_html",
                    new=AsyncMock(return_value=rendered_result),
                ) as render_mock:
                    result = await self.runtime.execute(
                        tool_call=ToolCall(
                            call_id="call_web_fetch_browser_fallback",
                            name="web_fetch",
                            arguments={"url": requested_url},
                            raw_arguments='{"url":"https://example.com/protected"}',
                        ),
                        context=self.context,
                    )

        self.assertTrue(result.ok)
        render_mock.assert_awaited_once()
        self.assertEqual(execute_mock.await_count, 2)
        first_call = execute_mock.await_args_list[0].kwargs
        second_call = execute_mock.await_args_list[1].kwargs
        self.assertEqual(first_call["arguments"], {"url": requested_url})
        self.assertEqual(second_call["arguments"]["url"], requested_url)
        self.assertIn("input_path", second_call["arguments"])
        self.assertEqual(
            list(self.workspace_dir.glob("web-fetch-render-*.html")),
            [],
        )
        self.assertIn("Hello from browser HTML.", result.content)

    def test_validate_browser_request_url_allows_data_scheme(self) -> None:
        _validate_browser_request_url("data:text/plain,hello")

    def test_validate_browser_request_url_denies_non_http_scheme(self) -> None:
        with self.assertRaises(WebFetchRequestError):
            _validate_browser_request_url("javascript:alert(1)")

    def test_markdown_is_usable_rejects_access_challenge_pages(self) -> None:
        markdown = (
            "## www.axios.com\n\n"
            "## Performing security verification\n\n"
            "This website uses a security service to protect against malicious bots. "
            "This page is displayed while the website verifies you are not a bot.\n\n"
            "Just a moment...\n"
        )
        self.assertFalse(_markdown_is_usable(markdown))

    async def test_render_page_html_blocks_non_public_browser_subrequests(self) -> None:
        requested_url = "https://example.com/app"
        page = _FakeBrowserPage(
            final_url=requested_url,
            html="<html><body><main>Rendered</main></body></html>",
            request_urls=(
                requested_url,
                "http://127.0.0.1:8080/debug",
            ),
        )
        manager = _FakePlaywrightManager(page)

        def validate_public_url(url: str) -> None:
            if "127.0.0.1" in url:
                raise WebFetchRequestError(
                    "web_fetch does not allow private, loopback, or reserved IP targets."
                )

        with patch("jarvis.tools.basic.web_fetch.tool.async_playwright", return_value=manager):
            with patch(
                "jarvis.tools.basic.web_fetch.tool._validate_public_url",
                side_effect=validate_public_url,
            ):
                result = await _render_page_html(
                    url=requested_url,
                    settings=self.settings,
                )

        self.assertEqual(result.final_url, requested_url)
        self.assertEqual(manager.browser.context_kwargs, [{"service_workers": "block"}])
        self.assertEqual(len(page.handled_routes), 2)
        self.assertTrue(page.handled_routes[0].continued)
        self.assertTrue(page.handled_routes[1].aborted)
        self.assertEqual(page.handled_routes[1].abort_error_code, "blockedbyclient")

    async def test_render_page_html_rejects_private_final_url(self) -> None:
        requested_url = "https://example.com/app"
        page = _FakeBrowserPage(
            final_url="http://127.0.0.1:8080/debug",
            html="<html><body><main>Blocked</main></body></html>",
            request_urls=(requested_url,),
        )
        manager = _FakePlaywrightManager(page)

        def validate_public_url(url: str) -> None:
            if "127.0.0.1" in url:
                raise WebFetchRequestError(
                    "web_fetch does not allow private, loopback, or reserved IP targets."
                )

        with patch("jarvis.tools.basic.web_fetch.tool.async_playwright", return_value=manager):
            with patch(
                "jarvis.tools.basic.web_fetch.tool._validate_public_url",
                side_effect=validate_public_url,
            ):
                with self.assertRaisesRegex(RuntimeError, "blocked final URL"):
                    await _render_page_html(
                        url=requested_url,
                        settings=self.settings,
                    )

    async def test_direct_web_fetch_executor_runs_defuddle(self) -> None:
        requested_url = "https://example.com/article"
        captured_call: dict[str, object] = {}
        executor = DirectWebFetchToolExecutor(self.settings)

        def _fake_run(command: list[str], **kwargs):
            captured_call["command"] = command
            captured_call["kwargs"] = kwargs
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="# Example\n\nHello from Defuddle.\n",
                stderr="",
            )

        with patch(
            "jarvis.tools.basic.web_fetch.tool.subprocess.run",
            side_effect=_fake_run,
        ):
            result = await executor(
                call_id="call_defuddle_success",
                arguments={"url": requested_url},
                context=self.context,
            )

        self.assertTrue(result.ok)
        self.assertEqual(
            captured_call["command"],
            ["npx", "defuddle", "parse", requested_url, "--markdown"],
        )
        kwargs = captured_call["kwargs"]
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["encoding"], "utf-8")
        self.assertFalse(kwargs["check"])
        self.assertEqual(kwargs["cwd"], str(self.workspace_dir))
        self.assertEqual(kwargs["timeout"], self.settings.web_fetch_timeout_seconds)
        self.assertEqual(
            kwargs["env"]["NODE_EXTRA_CA_CERTS"],
            "/etc/ssl/certs/ca-certificates.crt",
        )
        self.assertNotIn("provider: defuddle", result.content)
        self.assertIn("Hello from Defuddle.", result.content)

    async def test_direct_web_fetch_executor_truncates_markdown(self) -> None:
        requested_url = "https://example.com/long"
        markdown = "x" * (self.settings.web_fetch_max_markdown_chars + 25)
        executor = DirectWebFetchToolExecutor(self.settings)

        with patch(
            "jarvis.tools.basic.web_fetch.tool.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["npx", "defuddle"],
                returncode=0,
                stdout=markdown,
                stderr="",
            ),
        ):
            result = await executor(
                call_id="call_defuddle_truncated",
                arguments={"url": requested_url},
                context=self.context,
            )

        self.assertTrue(result.ok)
        self.assertTrue(result.metadata["markdown_truncated"])
        rendered_markdown = result.content.split("markdown:\n", 1)[1]
        self.assertLessEqual(
            len(rendered_markdown),
            self.settings.web_fetch_max_markdown_chars,
        )

    async def test_direct_web_fetch_executor_returns_tool_error_on_defuddle_failure(self) -> None:
        requested_url = "https://example.com/failure"
        executor = DirectWebFetchToolExecutor(self.settings)

        with patch(
            "jarvis.tools.basic.web_fetch.tool.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["npx", "defuddle"],
                returncode=1,
                stdout="",
                stderr="Error: fetch failed",
            ),
        ):
            result = await executor(
                call_id="call_defuddle_failure",
                arguments={"url": requested_url},
                context=self.context,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.metadata["error"], "Error: fetch failed")
        self.assertIn("Web fetch failed", result.content)
        self.assertIn("Error: fetch failed", result.content)

    async def test_view_image_rejects_non_universal_format(self) -> None:
        image_path = self.workspace_dir / "temp" / "sample.gif"
        image_path.parent.mkdir()
        image_path.write_bytes(b"GIF89afake_gif_payload")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_view_image_gif",
                name="view_image",
                arguments={"path": "temp/sample.gif"},
                raw_arguments='{"path":"temp/sample.gif"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("unsupported image type", result.content.lower())
