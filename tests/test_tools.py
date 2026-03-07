"""Unit tests for tool registry, policy, and runtime behavior."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm import ToolCall, ToolDefinition
from tools import RegisteredTool, ToolExecutionContext, ToolPolicy, ToolRegistry, ToolRuntime, ToolSettings


class ToolRegistryTests(unittest.TestCase):
    def test_default_registry_exposes_basic_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()

            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))

            basic_names = [tool.name for tool in registry.basic_definitions()]
            self.assertEqual(basic_names, ["bash", "view_image", "send_file"])
            self.assertEqual(registry.search("bash"), ())
            self.assertEqual(
                [tool.name for tool in registry.search("bash", include_basic=True)],
                ["bash"],
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
