"""Cross-subsystem regression shaped like the bb3899 production session."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from jarvis.core import AgentLoop
from jarvis.core.task_contract import build_task_contract
from jarvis.llm import LLMRequest, LLMResponse, ToolCall
from jarvis.storage import SessionStorage
from jarvis.tools import ToolExecutionContext, ToolExecutionResult
from tests.helpers import build_core_settings


_USER_TEXT = "You must finish the project."
_CONTRACT_ITEM_ID = build_task_contract(
    task_id="irrelevant",
    origin_turn_id="irrelevant",
    user_text=_USER_TEXT,
).requirements[0].item_id


def _response(*, tool_name: str | None = None, call_id: str = "", text: str = "") -> LLMResponse:
    tool_calls = []
    if tool_name is not None:
        arguments: dict[str, object]
        if tool_name == "bash":
            arguments = {"command": "npx vitest run 2>&1"}
        elif tool_name == "file_write":
            arguments = {"path": "project/main.ts", "content": "implemented\n"}
        elif tool_name == "acceptance_run":
            arguments = {
                "scope": "project",
                "revision_paths": ["project"],
                "gates": [{"gate_id": "tests", "command": "npm test"}],
            }
        else:
            arguments = {
                "scope": "project",
                "workspace_revision": "revision",
                "revision_paths": ["project"],
                "checks": [
                    {
                        "item_id": _CONTRACT_ITEM_ID,
                        "criterion": _USER_TEXT,
                        "outcome": "passed",
                        "evidence_kind": "test_result",
                        "evidence": "The verification gate passed.",
                        "source_tool_call_ids": ["acceptance-run"],
                    }
                ],
            }
        tool_calls = [
            ToolCall(
                call_id=call_id,
                name=tool_name,
                arguments=arguments,
                raw_arguments="",
            )
        ]
    return LLMResponse(
        provider="fake",
        model="fake",
        text=text,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        usage=None,
    )


class _TranscriptScaleService:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        responses = {
            1: _response(tool_name="bash", call_id="validation-before"),
            2: _response(tool_name="file_write", call_id="mutation"),
            # This response crosses the configured tool-slice boundary. It retries the same
            # validation signature only after a confirmed workspace mutation.
            3: _response(tool_name="bash", call_id="validation-after"),
            4: _response(tool_name="bash", call_id="validation-after-rollover"),
            5: _response(tool_name="acceptance_run", call_id="acceptance-run"),
            6: _response(tool_name="acceptance_record", call_id="acceptance-record"),
            7: _response(tool_name="acceptance_record", call_id="acceptance-record-rollover"),
            8: _response(text="Finished with current acceptance evidence."),
        }
        return responses[self.calls]


async def _execute(
    tool_call: ToolCall,
    context: ToolExecutionContext,
) -> ToolExecutionResult:
    if tool_call.name == "bash":
        return ToolExecutionResult(
            call_id=tool_call.call_id,
            name="bash",
            ok=False,
            content="validation failed",
            metadata={
                "execution_failed": True,
                "reason": "exit 1",
                "mode": "foreground",
                "exit_code": 1,
                "workspace_changed": False,
            },
        )
    if tool_call.name == "file_write":
        path = context.workspace_dir / "project" / "main.ts"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("implemented\n")
        return ToolExecutionResult(
            call_id=tool_call.call_id,
            name="file_write",
            ok=True,
            content="changed",
            metadata={"changed": True, "path": str(path)},
        )
    if tool_call.name == "acceptance_run":
        return ToolExecutionResult(
            call_id=tool_call.call_id,
            name="acceptance_run",
            ok=True,
            content="tests passed",
            metadata={
                "changed": False,
                "acceptance_run": {
                    "passed": True,
                    "revision_paths": ["project"],
                    "workspace_revision_after": "revision",
                    "gates": [
                        {"gate_id": "tests", "command": "npm test", "passed": True}
                    ]
                },
            },
        )
    return ToolExecutionResult(
        call_id=tool_call.call_id,
        name="acceptance_record",
        ok=True,
        content="recorded",
        metadata={
            "acceptance_ledger": {
                "workspace_revision_verified": True,
                "workspace_revision": "revision",
                "revision_paths": ["project"],
                "complete": True,
                "checks": [
                    {
                        "item_id": _CONTRACT_ITEM_ID,
                        "required": True,
                        "outcome": "passed",
                        "evidence_kind": "test_result",
                        "source_tool_call_ids": ["acceptance-run"],
                        "artifact_paths": [],
                    }
                ],
            }
        },
    )


class TranscriptScaleSafetyRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_failure_does_not_stop_after_mutation_and_slice_rollover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            service = _TranscriptScaleService()
            with patch.dict(
                "os.environ",
                {
                    "JARVIS_TOOL_MAX_ROUNDS_PER_TURN": "2",
                    "JARVIS_TOOL_MAX_ROUNDS_PER_TASK": "20",
                },
            ):
                loop = AgentLoop(
                    llm_service=service,  # type: ignore[arg-type]
                    settings=settings,
                    storage=storage,
                    tool_executor=_execute,
                )

            result = await loop.handle_user_input(_USER_TEXT)

            self.assertFalse(result.completion_blocked)
            self.assertEqual(result.response_text, "Finished with current acceptance evidence.")
            self.assertEqual(service.calls, 8)
            records = storage.load_records(result.session_id)
            self.assertTrue(
                any(
                    record.metadata.get("boundary") == "tool_slice"
                    and "previous tool slice" in record.content.casefold()
                    for record in records
                )
            )
            self.assertFalse(
                any(record.metadata.get("tool_safety_stop") for record in records)
            )

    async def test_repeated_failure_suppresses_exact_reuse_then_reports_liveness_exhaustion(
        self,
    ) -> None:
        class RepeatingService:
            def __init__(self) -> None:
                self.calls = 0

            async def generate(self, request: LLMRequest) -> LLMResponse:
                self.calls += 1
                return _response(
                    tool_name="bash",
                    call_id=f"repeat-{self.calls}",
                )

        executions = 0

        async def execute_repeated(
            tool_call: ToolCall,
            context: ToolExecutionContext,
        ) -> ToolExecutionResult:
            nonlocal executions
            _ = context
            executions += 1
            return ToolExecutionResult(
                call_id=tool_call.call_id,
                name="bash",
                ok=False,
                content="same failure",
                metadata={
                    "execution_failed": True,
                    "reason": "exit 1",
                    "mode": "foreground",
                    "exit_code": 1,
                    "workspace_changed": False,
                },
            )

        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            service = RepeatingService()
            with patch.dict(
                "os.environ",
                {
                    "JARVIS_TOOL_MAX_ROUNDS_PER_TURN": "3",
                    "JARVIS_TOOL_MAX_ROUNDS_PER_TASK": "3",
                },
            ):
                loop = AgentLoop(
                    llm_service=service,  # type: ignore[arg-type]
                    settings=settings,
                    storage=storage,
                    tool_executor=execute_repeated,
                )

            result = await loop.handle_user_input("Investigate this failure.")

            self.assertTrue(result.completion_blocked)
            self.assertEqual(result.completion_block_reason, "tool_liveness_exhausted")
            self.assertIn("could not make stable tool progress", result.response_text)
            self.assertEqual(executions, 2)
            self.assertEqual(service.calls, 5)
            records = storage.load_records(result.session_id)
            self.assertTrue(
                any(record.metadata.get("tool_liveness_exhausted") for record in records)
            )
            suppressed_tool_records = [
                record
                for record in records
                if record.role == "tool"
                and record.metadata.get("tool_liveness_suppressed")
            ]
            self.assertEqual(len(suppressed_tool_records), 1)
            diagnostics = suppressed_tool_records[0].metadata.get(
                "tool_liveness_diagnostics"
            )
            self.assertIsInstance(diagnostics, dict)
            assert isinstance(diagnostics, dict)
            self.assertEqual(diagnostics.get("first_call_id"), "repeat-1")
            self.assertEqual(diagnostics.get("threshold_call_id"), "repeat-2")
            self.assertEqual(diagnostics.get("progress_epoch"), 0)
