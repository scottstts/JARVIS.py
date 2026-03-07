"""Unit tests for tool registry, policy, and runtime behavior."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from llm import ToolCall, ToolDefinition
from tools import RegisteredTool, ToolExecutionContext, ToolPolicy, ToolRegistry, ToolRuntime, ToolSettings
from tools.web_fetch.tool import BrowserRenderResult, HTTPFetchResult, MarkdownConversionResult


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

    def test_uses_default_web_fetch_limits_from_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            settings = ToolSettings.from_workspace_dir(workspace_dir)

        self.assertEqual(settings.web_fetch_timeout_seconds, 20.0)
        self.assertEqual(settings.web_fetch_playwright_timeout_seconds, 20.0)
        self.assertEqual(settings.web_fetch_max_response_bytes, 2_097_152)
        self.assertEqual(settings.web_fetch_max_markdown_chars, 20_000)


class ToolRegistryTests(unittest.TestCase):
    def test_default_registry_exposes_basic_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))

            basic_names = [tool.name for tool in registry.basic_definitions()]
            self.assertEqual(
                basic_names,
                ["bash", "web_search", "web_fetch", "view_image", "send_file"],
            )
            self.assertEqual(registry.search("bash"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("bash", include_basic=True)],
                ["bash"],
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
                ["view_image"],
            )
            self.assertEqual(registry.search("send"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("send", include_basic=True)],
                ["send_file"],
            )


class ToolPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.workspace_dir = Path(self._tmp.name) / "workspace"
        self.workspace_dir.mkdir()
        self.context = ToolExecutionContext(workspace_dir=self.workspace_dir)
        self.policy = ToolPolicy()

    def test_allows_reads_outside_workspace(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "cat /etc/hosts"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_denies_redirect_syntax(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "printf 'hello' > note.txt"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("not allowed", decision.reason or "")

    def test_allows_copy_into_workspace(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "cp /etc/hosts notes.txt"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

    def test_denies_move_from_outside_workspace(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "mv /etc/hosts notes.txt"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("may only write inside", decision.reason or "")

    def test_denies_write_path_escape(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "printf 'hello' | tee ../note.txt"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("may only write inside", decision.reason or "")

    def test_denies_reading_dot_env_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "cat /tmp/.env"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn(".env", decision.reason or "")

    def test_denies_writing_dot_env_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "printf 'secret' | tee .env"},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn(".env", decision.reason or "")

    def test_denies_recursive_grep(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "grep -R secret ."},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("Recursive grep", decision.reason or "")

    def test_denies_rg_no_config_override(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "rg --no-config secret ."},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("--no-config", decision.reason or "")

    def test_denies_rg_glob_that_targets_dot_env(self) -> None:
        decision = self.policy.authorize(
            tool_name="bash",
            arguments={"command": "rg -g .env secret ."},
            context=self.context,
        )
        self.assertFalse(decision.allowed)
        self.assertIn(".env", decision.reason or "")

    def test_view_image_allows_workspace_relative_path(self) -> None:
        decision = self.policy.authorize(
            tool_name="view_image",
            arguments={"path": "temp/sample.png"},
            context=self.context,
        )
        self.assertTrue(decision.allowed)

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


class ToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addAsyncCleanup(self._cleanup_tmpdir)
        self.workspace_dir = Path(self._tmp.name) / "workspace"
        self.workspace_dir.mkdir()
        settings = ToolSettings.from_workspace_dir(self.workspace_dir)
        self.registry = ToolRegistry.default(settings)
        self.runtime = ToolRuntime(registry=self.registry)
        self.context = ToolExecutionContext(workspace_dir=self.workspace_dir)

    async def _cleanup_tmpdir(self) -> None:
        self._tmp.cleanup()

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
        self.assertIn(str(self.workspace_dir), result.content)

    async def test_writes_file_inside_workspace(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_write",
                name="bash",
                arguments={"command": "printf 'hello' | tee note.txt"},
                raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertEqual(
            (self.workspace_dir / "note.txt").read_text(encoding="utf-8"),
            "hello",
        )

    async def test_returns_policy_error_for_denied_command(self) -> None:
        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_denied",
                name="bash",
                arguments={"command": "printf 'hello' > note.txt"},
                raw_arguments='{"command":"printf \\"hello\\" > note.txt"}',
            ),
            context=self.context,
        )

        self.assertFalse(result.ok)
        self.assertIn("denied", result.content.lower())
        self.assertFalse((self.workspace_dir / "note.txt").exists())

    @unittest.skipUnless(shutil.which("rg"), "rg is not installed")
    async def test_rg_runtime_ignores_dot_env_files(self) -> None:
        (self.workspace_dir / ".env").write_text("secret\n", encoding="utf-8")
        (self.workspace_dir / "note.txt").write_text("secret\n", encoding="utf-8")

        result = await self.runtime.execute(
            tool_call=ToolCall(
                call_id="call_rg_env",
                name="bash",
                arguments={"command": "rg secret ."},
                raw_arguments='{"command":"rg secret ."}',
            ),
            context=self.context,
        )

        self.assertTrue(result.ok)
        self.assertIn("note.txt", result.content)
        self.assertNotIn(".env", result.content)

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
            "tools.send_file.tool.send_telegram_file",
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
            with patch("tools.web_search.tool.requests.get", side_effect=_fake_requests_get):
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
                "tools.web_search.tool.requests.get",
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
            "tools.web_fetch.tool._fetch_http_text",
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
            "tools.web_fetch.tool._fetch_http_text",
            side_effect=[tier1_result, tier2_result],
        ) as fetch_mock:
            with patch(
                "tools.web_fetch.tool._convert_html_to_markdown",
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
            "tools.web_fetch.tool._fetch_http_text",
            side_effect=[tier1_result, tier2_result],
        ) as fetch_mock:
            with patch(
                "tools.web_fetch.tool._convert_html_to_markdown",
                side_effect=[low_signal_markdown, rendered_markdown],
            ) as convert_mock:
                with patch(
                    "tools.web_fetch.tool._render_page_html",
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
            "tools.web_fetch.tool._fetch_http_text",
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
