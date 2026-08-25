"""Regression coverage for non-blocking subagent handoff after limited verification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jarvis.core import AgentIdentity, AgentLoop
from jarvis.llm import LLMRequest, LLMResponse, LLMUsage, ToolCall
from jarvis.storage import SessionStorage
from jarvis.tools import ToolExecutionResult, ToolRegistry, ToolSettings
from tests.helpers import build_core_settings


def _response(text: str = "", *, tool: str | None = None, call_id: str = "") -> LLMResponse:
    calls = ()
    if tool is not None:
        calls = (
            ToolCall(
                call_id=call_id,
                name=tool,
                arguments={"command": "verify"} if tool == "bash" else {"path": "src/a.py", "operations": []},
                raw_arguments="{}",
            ),
        )
    return LLMResponse(
        provider="fake",
        model="fake-chat",
        text=text,
        tool_calls=list(calls),
        finish_reason="tool_calls" if calls else "stop",  # type: ignore[arg-type]
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        response_id="resp_fake",
    )


class _VerificationWorkaroundService:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return _response(tool="file_patch", call_id="edit")
        if self.calls == 2:
            return _response(tool="bash", call_id="literal")
        if self.calls == 3:
            return _response(tool="bash", call_id="workaround")
        return _response(
            "Implementation complete. The literal verification hit a read-only environment; "
            "the equivalent workaround passed."
        )

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("streaming not used")


class PassiveAcceptanceRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_environment_limited_verification_never_blocks_child_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            registry = ToolRegistry.default(ToolSettings.from_workspace_dir(settings.workspace_dir))

            async def _execute(tool_call, _context):
                if tool_call.call_id == "edit":
                    return ToolExecutionResult(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        ok=True,
                        content="edited",
                        metadata={"changed": True, "workspace_changed": True},
                    )
                if tool_call.call_id == "literal":
                    return ToolExecutionResult(
                        call_id=tool_call.call_id,
                        name=tool_call.name,
                        ok=False,
                        content="read-only environment",
                        metadata={"execution_failed": True, "exit_code": 1},
                    )
                return ToolExecutionResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    ok=True,
                    content="27 passed",
                    metadata={"mode": "foreground", "exit_code": 0},
                )

            service = _VerificationWorkaroundService()
            loop = AgentLoop(
                llm_service=service,  # type: ignore[arg-type]
                settings=settings,
                storage=storage,
                tool_registry=registry,
                identity=AgentIdentity(kind="subagent", name="Jocasta", subagent_id="child"),
                tool_executor=_execute,
            )

            result = await loop.handle_user_input("Start the assigned task now.")

            self.assertIn("equivalent workaround passed", result.response_text)
            self.assertFalse(result.completion_blocked)
            self.assertEqual(service.calls, 4)


if __name__ == "__main__":
    unittest.main()
