"""Unit tests for tool registry, policy, and runtime behavior."""

from __future__ import annotations

from base64 import b64decode, b64encode
from dataclasses import dataclass
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from jsonschema import Draft202012Validator
from llm import ToolCall, ToolDefinition
from memory import MemoryService, MemorySettings
from tools import (
    DiscoverableTool,
    RegisteredTool,
    ToolExecutionContext,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
    ToolSettings,
)
from tools.basic.memory_write.tool import build_memory_write_tool
from tools.basic.tool_search import build_tool_search_tool
from tools.basic.memory_search.tool import _format_memory_search_result
from tools.basic.web_fetch.tool import (
    BrowserRenderResult,
    HTTPFetchResult,
    MarkdownConversionResult,
    WebFetchRequestError,
    _render_page_html,
    _validate_browser_request_url,
)

_PYTHON_INTERPRETER_VENV = Path("/opt/jarvis-python-tool-venv/bin/python")
_BASH_SANDBOX_RUNTIME_AVAILABLE = shutil.which("bwrap") is not None
_FFMPEG_BASH_RUNTIME_AVAILABLE = (
    _BASH_SANDBOX_RUNTIME_AVAILABLE and shutil.which("ffmpeg") is not None
)
_PYTHON_INTERPRETER_RUNTIME_AVAILABLE = (
    shutil.which("bwrap") is not None and _PYTHON_INTERPRETER_VENV.exists()
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


def _build_http_fetch_result(
    *,
    requested_url: str = "https://example.com/page",
    final_url: str | None = None,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
    content_type: str | None = "text/html",
    body_text: str = "",
    redirect_chain: tuple[str, ...] = (),
) -> HTTPFetchResult:
    resolved_headers = dict(headers or {})
    if content_type is not None and "Content-Type" not in resolved_headers:
        resolved_headers["Content-Type"] = content_type
    return HTTPFetchResult(
        requested_url=requested_url,
        final_url=final_url or requested_url,
        status_code=status_code,
        headers=resolved_headers,
        content_type=content_type,
        body_text=body_text,
        redirect_chain=redirect_chain,
    )


class ToolSettingsTests(unittest.TestCase):
    def test_requires_agent_workspace_for_host_runs(self) -> None:
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ), patch("workspace_paths._running_in_container", return_value=False):
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
        ), patch("workspace_paths._running_in_container", return_value=False):
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

    def test_memory_write_tool_describes_superseding_rewrite_and_truth_contracts(self) -> None:
        tool = build_memory_write_tool()

        self.assertIn("rewrite the memory content", tool.definition.description.lower())
        self.assertIn("facts and relations are explicit-decision fields", tool.definition.description.lower())
        self.assertIn("explicit durable fact", tool.definition.description.lower())
        self.assertIn("subject-predicate-object", tool.definition.description.lower())
        self.assertIn("minimal valid shapes", tool.definition.description.lower())
        self.assertIn("example payload pieces", tool.definition.description.lower())
        operation_description = tool.definition.input_schema["properties"]["operation"]["description"].lower()
        summary_description = tool.definition.input_schema["properties"]["summary"]["description"].lower()
        facts_description = tool.definition.input_schema["properties"]["facts"]["description"].lower()
        relations_description = tool.definition.input_schema["properties"]["relations"]["description"].lower()
        body_description = tool.definition.input_schema["properties"]["body_sections"]["description"].lower()
        close_reason_description = tool.definition.input_schema["properties"]["close_reason"]["description"].lower()

        self.assertIn("close and archive are superseding transitions", operation_description)
        self.assertIn("durable fact", facts_description)
        self.assertIn("summary/body text", facts_description)
        self.assertIn('literal string "none"', facts_description)
        self.assertIn('{"text":"..."}', facts_description)
        self.assertIn("subject-predicate-object claims", relations_description)
        self.assertIn("preferences", relations_description)
        self.assertIn('literal string "none"', relations_description)
        self.assertIn("subject, predicate, and object", relations_description)
        self.assertIn("summary is not a substitute", summary_description)
        self.assertIn("rewritten terminal summary", summary_description)
        self.assertIn("rewritten terminal body", body_description)
        self.assertIn("do not pass a list of section objects", body_description)
        self.assertIn("not a substitute", close_reason_description)

    def test_memory_write_schema_exposes_nested_truth_shapes(self) -> None:
        tool = build_memory_write_tool()
        facts_schema = tool.definition.input_schema["properties"]["facts"]["anyOf"][0]
        relations_schema = tool.definition.input_schema["properties"]["relations"]["anyOf"][0]
        body_sections_schema = tool.definition.input_schema["properties"]["body_sections"]["anyOf"][0]

        self.assertEqual(facts_schema["type"], "array")
        self.assertEqual(facts_schema["items"]["required"], ["text"])
        self.assertEqual(
            facts_schema["items"]["properties"]["status"]["enum"],
            ["current", "past", "uncertain", "superseded"],
        )
        self.assertEqual(relations_schema["type"], "array")
        self.assertEqual(relations_schema["items"]["required"], ["subject", "predicate", "object"])
        self.assertEqual(
            relations_schema["items"]["properties"]["cardinality"]["enum"],
            ["single", "multi"],
        )
        self.assertEqual(body_sections_schema["type"], "object")
        self.assertEqual(body_sections_schema["additionalProperties"]["type"], "string")

    def test_memory_write_schema_accepts_broad_truth_arrays_for_runtime_rejection(self) -> None:
        tool = build_memory_write_tool()
        arguments = {
            "operation": "create",
            "target_kind": "ongoing",
            "title": "Visual Pipeline Project",
            "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
            "facts": ["Scott is building a Three.js/WebGPU/TSL shader playground."],
            "relations": "None",
        }

        Draft202012Validator(tool.definition.input_schema).validate(arguments)

    def test_memory_write_schema_accepts_truth_objects_for_runtime_rejection(self) -> None:
        tool = build_memory_write_tool()
        arguments = {
            "operation": "create",
            "target_kind": "ongoing",
            "title": "Visual Pipeline Project",
            "summary": "Scott is building a Three.js/WebGPU/TSL shader playground.",
            "facts": {"text": "Scott is building a Three.js/WebGPU/TSL shader playground."},
            "relations": "None",
        }

        Draft202012Validator(tool.definition.input_schema).validate(arguments)

    def test_memory_write_schema_accepts_broad_body_sections_for_runtime_rejection(self) -> None:
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

        Draft202012Validator(tool.definition.input_schema).validate(arguments)

    def test_uses_default_web_fetch_limits_from_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(settings.web_fetch_timeout_seconds, 20.0)
        self.assertEqual(settings.web_fetch_playwright_timeout_seconds, 20.0)
        self.assertEqual(settings.web_fetch_max_response_bytes, 2_097_152)
        self.assertEqual(settings.web_fetch_max_markdown_chars, 20_000)

    def test_uses_default_python_interpreter_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(
            settings.python_interpreter_venv,
            Path("/opt/jarvis-python-tool-venv"),
        )
        self.assertEqual(
            settings.python_interpreter_allowed_packages,
            (
                "python-dateutil",
                "pyyaml",
                "pymupdf",
                "pandas",
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
        self.assertEqual(settings.python_interpreter_default_timeout_seconds, 10.0)
        self.assertEqual(settings.python_interpreter_max_timeout_seconds, 30.0)


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
                    "python_interpreter",
                    "web_search",
                    "web_fetch",
                    "view_image",
                    "send_file",
                    "tool_search",
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
                ["python_interpreter"],
            )
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
                ["python_interpreter", "view_image", "generate_edit_image"],
            )
            self.assertEqual(registry.search("send"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("send", include_basic=True)],
                ["send_file"],
            )
            self.assertEqual(registry.search_discoverable("archive"), ())
            self.assertEqual(
                [tool.name for tool in registry.search_discoverable("")],
                ["ffmpeg_cli", "generate_edit_image", "memory_admin", "transcribe", "youtube"],
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
                ["ffmpeg_cli"],
            )
            self.assertEqual(
                [tool.name for tool in registry.search_discoverable("youtube")],
                ["youtube"],
            )
            self.assertEqual(
                [
                    tool.name
                    for tool in registry.resolve_discoverable_tool_definitions(["ffmpeg_cli"])
                ],
                [],
            )

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

    def test_youtube_allows_valid_youtube_urls(self) -> None:
        decision = self.policy.authorize(
            tool_name="youtube",
            arguments={
                "video_urls": [
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "https://youtu.be/3JZ_D3ELwOQ?t=43",
                ]
            },
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_youtube_denies_invalid_video_urls(self) -> None:
        decision = self.policy.authorize(
            tool_name="youtube",
            arguments={
                "video_urls": [
                    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "https://example.com/not-youtube",
                    "https://youtu.be/not-a-real-id",
                ]
            },
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("[2] https://example.com/not-youtube", decision.reason or "")
        self.assertIn("[3] https://youtu.be/not-a-real-id", decision.reason or "")

    def test_youtube_denies_more_than_ten_urls(self) -> None:
        video_urls = [
            f"https://www.youtube.com/watch?v=vid{i:08d}"
            for i in range(11)
        ]
        decision = self.policy.authorize(
            tool_name="youtube",
            arguments={"video_urls": video_urls},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("at most 10", decision.reason or "")

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

    def test_python_interpreter_allows_inline_code_without_declared_io_paths(self) -> None:
        decision = self.policy.authorize(
            tool_name="python_interpreter",
            arguments={
                "code": "print('hello')",
            },
            context=self.context,
        )

        self.assertTrue(decision.allowed)

    def test_python_interpreter_allows_scripts_from_temp_workspace_path(self) -> None:
        script_path = self.workspace_dir / "temp" / "script.py"
        script_path.parent.mkdir()
        script_path.write_text("print('hello')", encoding="utf-8")

        decision = self.policy.authorize(
            tool_name="python_interpreter",
            arguments={"script_path": "temp/script.py"},
            context=self.context,
        )

        self.assertTrue(decision.allowed)


class ToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
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
        _BASH_SANDBOX_RUNTIME_AVAILABLE,
        "bash sandbox runtime is only available when bubblewrap is installed",
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
        _BASH_SANDBOX_RUNTIME_AVAILABLE,
        "bash sandbox runtime is only available when bubblewrap is installed",
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
        _FFMPEG_BASH_RUNTIME_AVAILABLE,
        "ffmpeg bash runtime is only available when bubblewrap and ffmpeg are installed",
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
        _BASH_SANDBOX_RUNTIME_AVAILABLE,
        "bash sandbox runtime is only available when bubblewrap is installed",
    )
    async def test_bash_mounts_etc_read_only(self) -> None:
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
        self.assertTrue(
            "Read-only file system" in result.content
            or "Permission denied" in result.content
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
        _BASH_SANDBOX_RUNTIME_AVAILABLE,
        "bash sandbox runtime is only available when bubblewrap is installed",
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
        _BASH_SANDBOX_RUNTIME_AVAILABLE,
        "bash sandbox runtime is only available when bubblewrap is installed",
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
        _BASH_SANDBOX_RUNTIME_AVAILABLE,
        "bash sandbox runtime is only available when bubblewrap is installed",
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
        self.assertIn("No such file", result.content)

    @unittest.skipUnless(
        _BASH_SANDBOX_RUNTIME_AVAILABLE,
        "bash sandbox runtime is only available when bubblewrap is installed",
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
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_executes_inline_code_with_direct_workspace_access(self) -> None:
        input_dir = self.workspace_dir / "inputs"
        output_dir = self.workspace_dir / "outputs"
        input_dir.mkdir()
        output_dir.mkdir()
        (input_dir / "data.txt").write_text("hello", encoding="utf-8")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_python_inline",
                name="python_interpreter",
                arguments={
                    "code": (
                        "from pathlib import Path\n"
                        "text = Path('/workspace/inputs/data.txt').read_text(encoding='utf-8')\n"
                        "Path('/workspace/outputs/result.txt').write_text("
                        "text.upper(), encoding='utf-8')\n"
                        "print(text.upper())\n"
                    ),
                },
                raw_arguments=(
                    '{"code":"from pathlib import Path\\ntext = '
                    "Path('/workspace/inputs/data.txt').read_text(encoding='utf-8')\\n"
                    "Path('/workspace/outputs/result.txt').write_text(text.upper(), "
                    "encoding='utf-8')\\nprint(text.upper())\\n\"}"
                ),
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("HELLO", result.content)
        self.assertIn("workspace_mode: direct_bind", result.content)
        self.assertEqual(
            (output_dir / "result.txt").read_text(encoding="utf-8"),
            "HELLO",
        )
        self.assertIn("pyyaml", result.metadata["allowed_packages"])
        self.assertEqual(result.metadata["workspace_mode"], "direct_bind")

    @unittest.skipUnless(
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_executes_stored_script_with_curated_package(self) -> None:
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
                "target = Path('/workspace/exports/payload.yml')\n"
                "target.write_text(yaml.safe_dump(payload), encoding='utf-8')\n"
                "print(payload['value'])\n"
            ),
            encoding="utf-8",
        )

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_python_script",
                name="python_interpreter",
                arguments={
                    "script_path": "scripts/emit_yaml.py",
                    "args": ["example"],
                },
                raw_arguments=(
                    '{"script_path":"scripts/emit_yaml.py","args":["example"]}'
                ),
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("example", result.content)
        self.assertIn("value: example", (exports_dir / "payload.yml").read_text(encoding="utf-8"))

    @unittest.skipUnless(
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_allows_workspace_local_helper_imports(self) -> None:
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
                call_id="call_python_local_import",
                name="python_interpreter",
                arguments={
                    "script_path": "scripts/main.py",
                },
                raw_arguments='{"script_path":"scripts/main.py"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("local helper ok", result.content)

    @unittest.skipUnless(
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_supports_pymupdf_text_round_trip(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_python_pymupdf_round_trip",
                name="python_interpreter",
                arguments={
                    "code": (
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
                raw_arguments='{"code":"pymupdf round trip"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("hello from pymupdf", result.content)
        self.assertIn("Rect(0.0, 0.0, 595.0, 842.0)", result.content)

    @unittest.skipUnless(
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_can_edit_workspace_jpeg_with_pillow(self) -> None:
        input_path = self.workspace_dir / "cat_input.jpg"
        output_path = self.workspace_dir / "cat_output.jpg"
        input_path.write_bytes(b64decode(_SAMPLE_JPEG_BASE64))

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_python_jpeg_edit",
                name="python_interpreter",
                arguments={
                    "code": (
                        "from pathlib import Path\n"
                        "from PIL import Image, ImageFilter\n"
                        "source = Path('/workspace/cat_input.jpg')\n"
                        "target = Path('/workspace/cat_output.jpg')\n"
                        "with Image.open(source) as image:\n"
                        "    square = image.crop((0, 0, 2, 2)).convert('L')\n"
                        "    square.filter(ImageFilter.GaussianBlur(radius=1)).save(target)\n"
                        "print(target)\n"
                    ),
                },
                raw_arguments='{"code":"jpeg edit"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("/workspace/cat_output.jpg", result.content)
        self.assertTrue(output_path.exists())
        self.assertGreater(output_path.stat().st_size, 0)

    @unittest.skipUnless(
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_denies_writes_outside_workspace(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_python_outside_write",
                name="python_interpreter",
                arguments={
                    "code": (
                        "from pathlib import Path\n"
                        "Path('/tmp/outside.txt').write_text('blocked', encoding='utf-8')\n"
                    ),
                },
                raw_arguments=(
                    '{"code":"from pathlib import Path\\n'
                    "Path('/tmp/outside.txt').write_text('blocked', encoding='utf-8')\\n\"}"
                ),
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("/tmp", result.content)
        self.assertFalse((self.workspace_dir / "outside.txt").exists())

    @unittest.skipUnless(
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_blocks_disallowed_imports(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_python_ctypes",
                name="python_interpreter",
                arguments={"code": "import ctypes\n"},
                raw_arguments='{"code":"import ctypes\\n"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("blocked", result.content.lower())

    @unittest.skipUnless(
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_allows_socket_import_but_network_use_still_fails(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_python_socket_runtime",
                name="python_interpreter",
                arguments={
                    "code": (
                        "import socket\n"
                        "sock = socket.socket()\n"
                        "sock.settimeout(1)\n"
                        "sock.connect(('1.1.1.1', 80))\n"
                    ),
                },
                raw_arguments='{"code":"socket runtime"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertNotIn("blocked", result.content.lower())
        self.assertIn("OSError", result.content)

    @unittest.skipUnless(
        _PYTHON_INTERPRETER_RUNTIME_AVAILABLE,
        "python_interpreter runtime is only available in the dev container",
    )
    async def test_python_interpreter_blocks_subprocess_execution_even_when_imported(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_python_subprocess",
                name="python_interpreter",
                arguments={
                    "code": (
                        "import subprocess\n"
                        "subprocess.run(['echo', 'hello'], check=True)\n"
                    ),
                },
                raw_arguments='{"code":"import subprocess\\nsubprocess.run([\'echo\', \'hello\'], check=True)\\n"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("does not allow spawning child processes", result.content)

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
            self.assertEqual(kwargs["file_path"], report_path)
            self.assertEqual(kwargs["caption"], "Attached report")
            self.assertEqual(kwargs["filename"], "weekly.txt")
            return {"message_id": 9, "chat_id": 123}

        with patch(
            "tools.basic.send_file.tool.send_telegram_file",
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
            result.metadata["image_attachment"]["path"],
            str(image_path),
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
                "tools.discoverable.generate_edit_image.tool.OpenAI",
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
            "gpt-image-1.5",
        )
        self.assertEqual(result.metadata["mime_type"], "image/png")
        self.assertEqual(result.metadata["file_size_bytes"], len(fake_image_bytes))
        self.assertEqual(
            captured["generate_kwargs"],
            {
                "model": "gpt-image-1.5",
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
                        "model_version": "gemini-3.1-flash-image-preview",
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
                "tools.discoverable.generate_edit_image.tool.genai.Client",
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
            "gemini-3.1-flash-image-preview",
        )
        self.assertEqual(result.metadata["image_path"], str(input_path))
        self.assertEqual(result.metadata["input_media_type"], "image/png")
        self.assertNotIn("provider_text", result.metadata)
        self.assertEqual(captured["api_key"], "test-google-key")
        self.assertEqual(captured["model"], "gemini-3.1-flash-image-preview")
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
        class _FakeGeminiModels:
            def generate_content(self, *, model, contents, config):
                _ = model, contents, config
                return type(
                    "_FakeGeminiResponse",
                    (),
                    {
                        "model_version": "gemini-3.1-flash-image-preview",
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
                "tools.discoverable.generate_edit_image.tool.genai.Client",
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
                "tools.discoverable.transcribe.tool.OpenAI",
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
        self.assertEqual(captured["uploaded_file_name"], str(audio_path))
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
            "tools.discoverable.transcribe.tool.OpenAI",
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

    async def test_youtube_executes_with_default_objectives(self) -> None:
        captured: dict[str, object] = {}

        class _FakeUsageMetadata:
            def model_dump(self, *, exclude_none: bool = False) -> dict[str, int]:
                _ = exclude_none
                return {"total_token_count": 55}

        class _FakeGeminiModels:
            def generate_content(self, *, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return type(
                    "_FakeGeminiResponse",
                    (),
                    {
                        "model_version": "gemini-3-flash-preview",
                        "response_id": "resp_youtube_123",
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
                                                    "_FakeTextPart",
                                                    (),
                                                    {
                                                        "text": "Video one explains the launch plan.",
                                                    },
                                                )(),
                                                type(
                                                    "_FakeTextPart",
                                                    (),
                                                    {
                                                        "text": " Video two adds implementation details.",
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

            def close(self) -> None:
                captured["closed"] = True

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}, clear=False):
            with patch(
                "tools.discoverable.youtube.tool.genai.Client",
                _FakeGeminiClient,
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_youtube_default",
                        name="youtube",
                        arguments={
                            "video_urls": [
                                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                "https://youtu.be/3JZ_D3ELwOQ",
                            ]
                        },
                        raw_arguments=(
                            '{"video_urls":["https://www.youtube.com/watch?v=dQw4w9WgXcQ",'
                            '"https://youtu.be/3JZ_D3ELwOQ"]}'
                        ),
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["provider"], "gemini")
        self.assertEqual(result.metadata["model"], "gemini-3-flash-preview")
        self.assertEqual(result.metadata["video_count"], 2)
        self.assertEqual(result.metadata["objectives_source"], "default")
        self.assertEqual(captured["api_key"], "test-google-key")
        self.assertEqual(captured["model"], "gemini-3-flash-preview")
        self.assertTrue(captured["closed"])
        config = captured["config"]
        self.assertIn("You are helping another agent", config.system_instruction)
        self.assertIn("Summarize each provided video", config.system_instruction)
        contents = captured["contents"]
        self.assertEqual(len(contents), 3)
        self.assertEqual(
            [contents[0].file_data.file_uri, contents[1].file_data.file_uri],
            [
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "https://youtu.be/3JZ_D3ELwOQ",
            ],
        )
        self.assertEqual(
            contents[2],
            "Analyze the provided YouTube videos using the active system instruction.",
        )
        self.assertEqual(
            result.metadata["usage"],
            {"total_token_count": 55},
        )
        self.assertIn("YouTube analysis completed", result.content)
        self.assertIn("Video one explains the launch plan.", result.content)

    async def test_youtube_executes_with_custom_objectives(self) -> None:
        captured: dict[str, object] = {}

        class _FakeGeminiModels:
            def generate_content(self, *, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return type(
                    "_FakeGeminiResponse",
                    (),
                    {
                        "model_version": "gemini-3-flash-preview",
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
                                                    "_FakeTextPart",
                                                    (),
                                                    {
                                                        "text": "The release date claim appears at the start of the video.",
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
                _ = api_key
                self.models = _FakeGeminiModels()

        custom_objectives = (
            "Context: I do not need a general summary. I only need release-date evidence "
            "from this video for downstream planning. Task: extract only concrete claims "
            "about release timing, attribute each claim clearly, and separate direct "
            "statements from speculation."
        )

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}, clear=False):
            with patch(
                "tools.discoverable.youtube.tool.genai.Client",
                _FakeGeminiClient,
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_youtube_custom_objectives",
                        name="youtube",
                        arguments={
                            "video_urls": [
                                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                            ],
                            "objectives": custom_objectives,
                        },
                        raw_arguments=(
                            '{"video_urls":["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],'
                            '"objectives":"Context: I do not need a general summary. I only need release-date evidence from this video for downstream planning. Task: extract only concrete claims about release timing, attribute each claim clearly, and separate direct statements from speculation."}'
                        ),
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["objectives_source"], "provided")
        self.assertEqual(result.metadata["objectives"], custom_objectives)
        self.assertEqual(captured["model"], "gemini-3-flash-preview")
        self.assertIn("You are helping another agent", captured["config"].system_instruction)
        self.assertIn(custom_objectives, captured["config"].system_instruction)
        self.assertIn("release date", result.content)

    async def test_youtube_executes_transcript_mode_with_defuddle(self) -> None:
        captured_commands: list[list[str]] = []

        def _fake_curl_run(*args, **kwargs):
            command = args[0]
            captured_commands.append(command)
            self.assertEqual(
                command[:5],
                ["curl", "--fail", "--silent", "--show-error", "--location"],
            )
            self.assertTrue(command[5].startswith("https://defuddle.md/"))
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            self.assertEqual(kwargs["encoding"], "utf-8")
            self.assertFalse(kwargs["check"])
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="# Transcript\n\nThe speaker confirms the launch date.\n",
                stderr="",
            )

        class _FailIfCalledGeminiClient:
            def __init__(self, *, api_key: str) -> None:
                _ = api_key
                raise AssertionError(
                    "Gemini client should not be created in transcript mode."
                )

        with patch(
            "tools.discoverable.youtube.tool.subprocess.run",
            side_effect=_fake_curl_run,
        ):
            with patch(
                "tools.discoverable.youtube.tool.genai.Client",
                _FailIfCalledGeminiClient,
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_youtube_transcript",
                        name="youtube",
                        arguments={
                            "video_urls": [
                                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                "https://youtu.be/3JZ_D3ELwOQ?t=43",
                            ],
                            "objectives": "Only summarize release timing.",
                            "transcript": True,
                        },
                        raw_arguments=(
                            '{"video_urls":["https://www.youtube.com/watch?v=dQw4w9WgXcQ",'
                            '"https://youtu.be/3JZ_D3ELwOQ?t=43"],'
                            '"objectives":"Only summarize release timing.",'
                            '"transcript":true}'
                        ),
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertEqual(result.metadata["provider"], "defuddle")
        self.assertEqual(result.metadata["mode"], "transcript")
        self.assertTrue(result.metadata["transcript_requested"])
        self.assertTrue(result.metadata["objectives_ignored"])
        self.assertEqual(result.metadata["video_count"], 2)
        self.assertEqual(len(captured_commands), 2)
        self.assertEqual(
            captured_commands[0][5],
            "https://defuddle.md/https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DdQw4w9WgXcQ",
        )
        self.assertEqual(
            captured_commands[1][5],
            "https://defuddle.md/https%3A%2F%2Fyoutu.be%2F3JZ_D3ELwOQ%3Ft%3D43",
        )
        self.assertIn("YouTube transcript retrieval completed", result.content)
        self.assertIn("objectives_ignored: yes", result.content)
        self.assertIn("# Transcript", result.content)

    async def test_youtube_transcript_mode_reports_curl_failure(self) -> None:
        def _fake_curl_run(*args, **kwargs):
            _ = kwargs
            return subprocess.CompletedProcess(
                args=args[0],
                returncode=22,
                stdout="",
                stderr="HTTP 502 from defuddle",
            )

        with patch(
            "tools.discoverable.youtube.tool.subprocess.run",
            side_effect=_fake_curl_run,
        ):
            result = await self.runtime.execute(
                tool_call=ToolCall(
                    call_id="call_youtube_transcript_failure",
                    name="youtube",
                    arguments={
                        "video_urls": [
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        ],
                        "transcript": True,
                    },
                    raw_arguments=(
                        '{"video_urls":["https://www.youtube.com/watch?v=dQw4w9WgXcQ"],'
                        '"transcript":true}'
                    ),
                ),
                context=self.context,
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.metadata["provider"], "defuddle")
        self.assertEqual(result.metadata["mode"], "transcript")
        self.assertTrue(result.metadata["transcript_requested"])
        self.assertIn("YouTube transcript retrieval failed", result.content)
        self.assertIn("HTTP 502 from defuddle", result.content)

    async def test_youtube_invalid_urls_fail_before_execution(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_youtube_invalid",
                name="youtube",
                arguments={
                    "video_urls": [
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "https://example.com/not-youtube",
                    ]
                },
                raw_arguments=(
                    '{"video_urls":["https://www.youtube.com/watch?v=dQw4w9WgXcQ",'
                    '"https://example.com/not-youtube"]}'
                ),
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.metadata["policy_denied"])
        self.assertIn("[2] https://example.com/not-youtube", result.content)

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
            with patch("tools.basic.web_search.tool.requests.get", side_effect=_fake_requests_get):
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
                "tools.basic.web_search.tool.requests.get",
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
        self.assertIn("usage:", result.content)
        self.assertIn("metadata:", result.content)
        self.assertEqual(result.metadata["match_count"], 1)
        self.assertEqual(
            result.metadata["activated_discoverable_tool_names"],
            ["archive"],
        )

    async def test_tool_search_high_verbosity_ffmpeg_cli_is_docs_only_and_not_activated(
        self,
    ) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_ffmpeg_cli",
                name="tool_search",
                arguments={"query": "ffmpeg", "verbosity": "high"},
                raw_arguments='{"query":"ffmpeg","verbosity":"high"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("ffmpeg_cli", result.content)
        self.assertIn("Use the basic `bash` tool", result.content)
        self.assertEqual(result.metadata["activated_discoverable_tool_names"], [])
        self.assertEqual(
            [match["name"] for match in result.metadata["matches"]],
            ["ffmpeg_cli"],
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
        self.assertIn("gpt-4o-mini-transcribe", result.content)
        self.assertEqual(result.metadata["activated_discoverable_tool_names"], ["transcribe"])
        self.assertEqual(
            [match["name"] for match in result.metadata["matches"]],
            ["transcribe"],
        )

    async def test_tool_search_high_verbosity_youtube_activates_tool(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_tool_search_youtube",
                name="tool_search",
                arguments={"query": "youtube", "verbosity": "high"},
                raw_arguments='{"query":"youtube","verbosity":"high"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("youtube", result.content)
        self.assertIn("gemini-3-flash-preview", result.content)
        self.assertEqual(result.metadata["activated_discoverable_tool_names"], ["youtube"])
        self.assertEqual(
            [match["name"] for match in result.metadata["matches"]],
            ["youtube"],
        )

    async def test_web_fetch_uses_tier1_markdown_when_available(self) -> None:
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
            "tools.basic.web_fetch.tool._fetch_http_text",
            return_value=tier1_result,
        ) as fetch_mock:
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
        self.assertEqual(result.metadata["strategy"], "tier1_markdown_accept")
        self.assertEqual(result.metadata["markdown_tokens"], 41)
        self.assertIn("strategy: tier1_markdown_accept", result.content)

    async def test_web_fetch_converts_html_when_markdown_fast_path_fails(self) -> None:
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
        converted_result = MarkdownConversionResult(
            markdown="# Example\n\nHello from HTML.",
            markdown_tokens=88,
        )

        with patch(
            "tools.basic.web_fetch.tool._fetch_http_text",
            side_effect=[tier1_result, tier2_result],
        ) as fetch_mock:
            with patch(
                "tools.basic.web_fetch.tool._convert_html_to_markdown",
                return_value=converted_result,
            ) as convert_mock:
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_web_fetch_tier2",
                        name="web_fetch",
                        arguments={"url": requested_url},
                        raw_arguments='{"url":"https://example.com/article"}',
                    ),
                    context=self.context,
                )

        self.assertTrue(result.ok)
        self.assertEqual(fetch_mock.call_count, 2)
        convert_mock.assert_called_once()
        self.assertEqual(result.metadata["strategy"], "tier2_html_to_markdown")
        self.assertIn("Hello from HTML.", result.content)

    async def test_web_fetch_falls_back_to_playwright_for_js_heavy_page(self) -> None:
        requested_url = "https://example.com/app"
        app_shell_html = (
            "<html><body><div id=\"__next\"></div><script>window.__APP__ = {};</script>"
            "</body></html>"
        )
        tier1_result = _build_http_fetch_result(
            requested_url=requested_url,
            content_type="text/html",
            body_text=app_shell_html,
        )
        tier2_result = _build_http_fetch_result(
            requested_url=requested_url,
            content_type="text/html",
            body_text=app_shell_html,
        )
        low_signal_markdown = MarkdownConversionResult(
            markdown="Loading...",
            markdown_tokens=3,
        )
        rendered_result = BrowserRenderResult(
            requested_url=requested_url,
            final_url=requested_url,
            html=(
                "<html><body><main><h1>Rendered App</h1><p>The client content loaded."
                "</p></main></body></html>"
            ),
        )
        rendered_markdown = MarkdownConversionResult(
            markdown="# Rendered App\n\nThe client content loaded.",
            markdown_tokens=55,
        )

        with patch(
            "tools.basic.web_fetch.tool._fetch_http_text",
            side_effect=[tier1_result, tier2_result],
        ) as fetch_mock:
            with patch(
                "tools.basic.web_fetch.tool._convert_html_to_markdown",
                side_effect=[low_signal_markdown, rendered_markdown],
            ) as convert_mock:
                with patch(
                    "tools.basic.web_fetch.tool._render_page_html",
                    new=AsyncMock(return_value=rendered_result),
                ) as render_mock:
                    result = await self.runtime.execute(
                        tool_call=ToolCall(
                            call_id="call_web_fetch_tier3",
                            name="web_fetch",
                            arguments={"url": requested_url},
                            raw_arguments='{"url":"https://example.com/app"}',
                        ),
                        context=self.context,
                    )

        self.assertTrue(result.ok)
        self.assertEqual(fetch_mock.call_count, 2)
        self.assertEqual(convert_mock.call_count, 2)
        render_mock.assert_awaited_once()
        self.assertEqual(result.metadata["strategy"], "tier3_playwright_html_to_markdown")
        self.assertTrue(result.metadata["browser_rendered"])

    def test_validate_browser_request_url_allows_data_scheme(self) -> None:
        _validate_browser_request_url("data:text/plain,hello")

    def test_validate_browser_request_url_denies_non_http_scheme(self) -> None:
        with self.assertRaises(WebFetchRequestError):
            _validate_browser_request_url("javascript:alert(1)")

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

        with patch("tools.basic.web_fetch.tool.async_playwright", return_value=manager):
            with patch(
                "tools.basic.web_fetch.tool._validate_public_url",
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

        with patch("tools.basic.web_fetch.tool.async_playwright", return_value=manager):
            with patch(
                "tools.basic.web_fetch.tool._validate_public_url",
                side_effect=validate_public_url,
            ):
                with self.assertRaisesRegex(RuntimeError, "blocked final URL"):
                    await _render_page_html(
                        url=requested_url,
                        settings=self.settings,
                    )

    async def test_web_fetch_returns_configuration_error_when_account_id_missing(self) -> None:
        requested_url = "https://example.com/docs"
        tier1_result = _build_http_fetch_result(
            requested_url=requested_url,
            content_type="text/html",
            body_text="<html><body><div>fallback</div></body></html>",
        )
        tier2_result = _build_http_fetch_result(
            requested_url=requested_url,
            content_type="text/html",
            body_text="<html><body><article><h1>Docs</h1></article></body></html>",
        )

        with patch(
            "tools.basic.web_fetch.tool._fetch_http_text",
            side_effect=[tier1_result, tier2_result],
        ):
            with patch.dict(
                os.environ,
                {"CLOUDFLARE_AI_WORKERS_REST_API_KEY": "test-key"},
                clear=True,
            ):
                result = await self.runtime.execute(
                    tool_call=ToolCall(
                        call_id="call_web_fetch_missing_account",
                        name="web_fetch",
                        arguments={"url": requested_url},
                        raw_arguments='{"url":"https://example.com/docs"}',
                    ),
                    context=self.context,
                )

        self.assertFalse(result.ok)
        self.assertIn("CLOUDFLARE_ACCOUNT_ID", result.content)

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
