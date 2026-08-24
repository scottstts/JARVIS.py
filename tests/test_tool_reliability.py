"""Regression tests for transcript and workspace tool reliability boundaries."""

from __future__ import annotations

import asyncio
from pathlib import Path
import socket
import tempfile
import unittest

from jarvis.core.tool_safety import ToolSafetyTracker
from jarvis.llm import ToolCall
from jarvis.tools import (
    ToolExecutionContext,
    ToolExecutionResult,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
    ToolSettings,
    WorkspaceAccessCoordinator,
    WorkspaceLeaseError,
)
from jarvis.tools.basic.bash.local_executor import DirectBashToolExecutor
from jarvis.tools.basic.acceptance_run.tool import (
    AcceptanceRunToolExecutor,
    _source_line_count_gate,
    _top_level_compound_operator,
)
from jarvis.tools.workspace_revision import scoped_workspace_revision, workspace_revision


def _tool_call(name: str, arguments: dict[str, object], *, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments="",
    )


class ToolReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_flat_file_edits_and_acceptance_ledger_are_durable_tool_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            runtime = ToolRuntime(
                registry=ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)

            write_result = await runtime.execute(
                tool_call=_tool_call(
                    "file_write",
                    {"path": "report.txt", "content": "before\n"},
                    call_id="write_1",
                ),
                context=context,
            )
            replace_result = await runtime.execute(
                tool_call=_tool_call(
                    "file_replace",
                    {"path": "report.txt", "match": "before", "replacement": "after"},
                    call_id="replace_1",
                ),
                context=context,
            )
            acceptance_result = await runtime.execute(
                tool_call=_tool_call(
                    "acceptance_record",
                    {
                        "scope": "report rewrite",
                        "workspace_revision": scoped_workspace_revision(
                            workspace_dir,
                            ("report.txt",),
                        ),
                        "revision_paths": ["report.txt"],
                        "checks": [
                            {
                                "criterion": "report contains the requested replacement",
                                "outcome": "passed",
                                "evidence_kind": "artifact_inspection",
                                "evidence": "report.txt contains 'after'.",
                                "artifact_paths": ["report.txt"],
                            },
                            {
                                "criterion": "focused edit command completed",
                                "outcome": "passed",
                                "evidence_kind": "test_result",
                                "evidence": "file_replace returned a changed file digest.",
                                "source_tool_call_ids": ["replace_1"],
                            },
                        ],
                    },
                    call_id="acceptance_1",
                ),
                context=context,
            )

            self.assertTrue(write_result.ok)
            self.assertTrue(replace_result.ok)
            self.assertEqual((workspace_dir / "report.txt").read_text(), "after\n")
            self.assertTrue(acceptance_result.ok)
            self.assertIn("Acceptance ledger recorded", acceptance_result.content)
            self.assertEqual(
                acceptance_result.metadata["acceptance_ledger"]["summary"]["passed"],
                2,
            )

    async def test_workspace_revision_ignores_runtime_transcripts_but_tracks_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            initial = workspace_revision(workspace_dir)
            transcript = workspace_dir / "archive" / "transcripts" / "session.jsonl"
            transcript.parent.mkdir(parents=True)
            transcript.write_text("runtime record\n")
            self.assertEqual(workspace_revision(workspace_dir), initial)

            artifact = workspace_dir / "artifacts" / "result.txt"
            artifact.parent.mkdir()
            artifact.write_text("result\n")
            self.assertNotEqual(workspace_revision(workspace_dir), initial)

    async def test_workspace_leases_block_conflicting_edits_and_unscoped_bash_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            coordinator = WorkspaceAccessCoordinator(workspace_dir=workspace_dir)
            await coordinator.claim_paths(owner="subagent:child_1", paths=("owned.py",))
            main_context = ToolExecutionContext(workspace_dir=workspace_dir)
            child_context = ToolExecutionContext(
                workspace_dir=workspace_dir,
                agent_kind="subagent",
                subagent_id="child_1",
            )

            with self.assertRaises(WorkspaceLeaseError):
                async with coordinator.execute(
                    tool_call=_tool_call("file_write", {"path": "owned.py", "content": "x"}),
                    context=main_context,
                ):
                    pass

            with self.assertRaises(WorkspaceLeaseError):
                async with coordinator.execute(
                    tool_call=_tool_call("bash", {"command": "printf x > other.py"}),
                    context=main_context,
                ):
                    pass

            async with coordinator.execute(
                tool_call=_tool_call(
                    "bash",
                    {
                        "command": "printf x > owned.py",
                        "write_paths": ["owned.py"],
                        "expected_lease_generation": await coordinator.lease_generation(),
                    },
                ),
                context=child_context,
            ):
                pass

    async def test_overlapping_directory_and_file_writes_are_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            coordinator = WorkspaceAccessCoordinator(workspace_dir=workspace_dir)
            context = ToolExecutionContext(workspace_dir=workspace_dir)
            first_entered = asyncio.Event()
            release_first = asyncio.Event()
            second_entered = asyncio.Event()

            async def first_write() -> None:
                async with coordinator.execute(
                    tool_call=_tool_call(
                        "bash",
                        {"command": "touch src/a.py", "write_paths": ["src"]},
                    ),
                    context=context,
                ):
                    first_entered.set()
                    await release_first.wait()

            async def second_write() -> None:
                await first_entered.wait()
                async with coordinator.execute(
                    tool_call=_tool_call(
                        "file_write",
                        {"path": "src/a.py", "content": "x"},
                    ),
                    context=context,
                ):
                    second_entered.set()

            first_task = asyncio.create_task(first_write())
            second_task = asyncio.create_task(second_write())
            await first_entered.wait()
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(second_entered.wait(), timeout=0.05)
            release_first.set()
            await asyncio.gather(first_task, second_task)
            self.assertTrue(second_entered.is_set())

    async def test_overlapping_path_read_and_write_are_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            coordinator = WorkspaceAccessCoordinator(workspace_dir=workspace_dir)
            context = ToolExecutionContext(workspace_dir=workspace_dir)
            read_entered = asyncio.Event()
            release_read = asyncio.Event()
            write_entered = asyncio.Event()

            async def read_file() -> None:
                async with coordinator.execute(
                    tool_call=_tool_call("view_image", {"path": "artifacts/image.png"}),
                    context=context,
                ):
                    read_entered.set()
                    await release_read.wait()

            async def write_file() -> None:
                await read_entered.wait()
                async with coordinator.execute(
                    tool_call=_tool_call(
                        "file_write",
                        {"path": "artifacts/image.png", "content": "replacement"},
                    ),
                    context=context,
                ):
                    write_entered.set()

            read_task = asyncio.create_task(read_file())
            write_task = asyncio.create_task(write_file())
            await read_entered.wait()
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(write_entered.wait(), timeout=0.05)
            release_read.set()
            await asyncio.gather(read_task, write_task)
            self.assertTrue(write_entered.is_set())

    async def test_workspace_file_consumers_reject_absolute_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            runtime = ToolRuntime(
                registry=ToolRegistry.default(ToolSettings.from_workspace_dir(workspace_dir))
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)

            view_result = await runtime.execute(
                tool_call=_tool_call("view_image", {"path": "/etc/passwd"}),
                context=context,
            )
            send_result = await runtime.execute(
                tool_call=_tool_call("send_file", {"path": "/etc/passwd"}),
                context=context,
            )

            self.assertFalse(view_result.ok)
            self.assertFalse(send_result.ok)
            self.assertIn("inside", view_result.content)
            self.assertIn("inside", send_result.content)

    async def test_stale_workspace_lease_generation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            coordinator = WorkspaceAccessCoordinator(workspace_dir=workspace_dir)
            observed = await coordinator.lease_generation()
            await coordinator.claim_paths(owner="subagent:child", paths=("other.py",))
            with self.assertRaises(WorkspaceLeaseError):
                async with coordinator.execute(
                    tool_call=_tool_call(
                        "file_write",
                        {
                            "path": "mine.py",
                            "content": "x",
                            "expected_lease_generation": observed,
                        },
                    ),
                    context=ToolExecutionContext(workspace_dir=workspace_dir),
                ):
                    pass

    async def test_workspace_access_observes_undeclared_bash_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            coordinator = WorkspaceAccessCoordinator(workspace_dir=workspace_dir)
            call = _tool_call("bash", {"command": "custom-generator"})

            async with coordinator.execute(
                tool_call=call,
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            ) as observation:
                (workspace_dir / "generated.txt").write_text("created before failure")

            self.assertEqual(observation.mode, "global_write")
            self.assertTrue(observation.changed)
            self.assertNotEqual(
                observation.revision_before,
                observation.revision_after,
            )

    async def test_compound_bash_failure_cannot_be_masked_by_a_later_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            executor = DirectBashToolExecutor(
                ToolSettings.from_workspace_dir(workspace_dir),
                target_runtime="test",
                runtime_location="test",
                runtime_transport="inprocess",
                container_mutation_boundary="test",
            )
            result = await executor(
                call_id="compound_failure",
                arguments={"command": "false; printf 'masked'"},
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

            self.assertFalse(result.ok)
            self.assertNotIn("masked", str(result.metadata["stdout"]))

    async def test_acceptance_gate_rejects_nested_shell_failure_masking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            executor = AcceptanceRunToolExecutor(
                ToolSettings.from_workspace_dir(workspace_dir)
            )
            result = await executor(
                call_id="acceptance",
                arguments={
                    "scope": "verification",
                    "revision_paths": ["."],
                    "gates": [
                        {
                            "gate_id": "masked",
                            "command": "bash -lc 'false; true'",
                        }
                    ],
                },
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

            self.assertFalse(result.ok)
            self.assertIn("shell command wrapper", result.content)

    async def test_acceptance_gate_allows_descriptor_redirection(self) -> None:
        self.assertIsNone(_top_level_compound_operator("printf passed 2>&1"))
        self.assertIsNone(_top_level_compound_operator("printf passed >| result.txt"))

    async def test_source_line_count_excludes_vendor_generated_tests_and_probes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            for relative in ("src", "vendor", "generated", "tests", "probes"):
                (workspace_dir / relative).mkdir(parents=True)
            (workspace_dir / "src" / "main.ts").write_text("one\ntwo\nthree\n")
            for relative in ("vendor", "generated", "tests", "probes"):
                (workspace_dir / relative / "ignored.ts").write_text("ignored\n" * 20)

            metric, reason = _source_line_count_gate(
                {"include_paths": ["."], "minimum": 3},
                workspace_dir=workspace_dir,
            )

            self.assertIsNone(reason)
            self.assertTrue(metric["passed"])
            self.assertEqual(metric["line_count"], 3)
            self.assertEqual(metric["file_count"], 1)

    async def test_tool_safety_stops_third_identical_no_progress_result(self) -> None:
        tracker = ToolSafetyTracker()
        tool_call = _tool_call("bash", {"command": "true"})
        result = ToolExecutionResult(
            call_id="call_1",
            name="bash",
            ok=True,
            content="Bash execution result\nexit_code: 0",
            metadata={"mode": "foreground", "exit_code": 0, "status": "finished"},
        )

        self.assertFalse(tracker.record(tool_call, result).repeated_no_progress)
        self.assertFalse(tracker.record(tool_call, result).repeated_no_progress)
        self.assertTrue(tracker.record(tool_call, result).repeated_no_progress)

    async def test_tool_safety_blocks_exact_call_after_second_identical_failure(self) -> None:
        tracker = ToolSafetyTracker()
        tool_call = _tool_call("bash", {"command": "false"})
        result = ToolExecutionResult(
            call_id="call_1",
            name="bash",
            ok=False,
            content="Bash failed",
            metadata={"execution_failed": True, "reason": "exit 1"},
        )
        self.assertFalse(tracker.record(tool_call, result).repeated_invalid_call)
        self.assertTrue(tracker.record(tool_call, result).repeated_invalid_call)
        self.assertEqual(tracker.blocked_call_reason(tool_call), "repeated_invalid_result")
        self.assertIsNone(
            tracker.blocked_call_reason(_tool_call("bash", {"command": "false --different"}))
        )

    async def test_tool_safety_normalizes_varied_workspace_lease_conflicts(self) -> None:
        tracker = ToolSafetyTracker()
        first_call = _tool_call("bash", {"command": "write a.py"}, call_id="lease_1")
        second_call = _tool_call(
            "file_write",
            {"path": "a.py", "content": "x"},
            call_id="lease_2",
        )

        def conflict(call_id: str, name: str) -> ToolExecutionResult:
            return ToolExecutionResult(
                call_id=call_id,
                name=name,
                ok=False,
                content="workspace lease conflict",
                metadata={
                    "execution_failed": True,
                    "error_code": "workspace_lease_conflict",
                    "conflict_class": "path_owned_by_other_actor",
                    "remediation": "Wait for or dispose the owner.",
                },
            )

        first = tracker.record(first_call, conflict("lease_1", "bash"))
        second = tracker.record(second_call, conflict("lease_2", "file_write"))

        self.assertFalse(first.repeated_invalid_call)
        self.assertTrue(second.repeated_invalid_call)
        self.assertTrue(second.blocked_invalid_signature)

        distinct_target = ToolExecutionResult(
            call_id="lease_3",
            name="file_write",
            ok=False,
            content="different workspace lease conflict",
            metadata={
                "execution_failed": True,
                "error_code": "workspace_lease_conflict",
                "conflict_class": "path_owned_by_other_actor",
                "conflict_key": "path_owned_by_other_actor:b.py:subagent_2",
            },
        )
        distinct = tracker.record(
            _tool_call(
                "file_write",
                {"path": "b.py", "content": "y"},
                call_id="lease_3",
            ),
            distinct_target,
        )
        self.assertFalse(distinct.repeated_invalid_call)

    async def test_scoped_revision_ignores_runtime_files_outside_material_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            source_dir = workspace_dir / "project" / "src"
            runtime_dir = workspace_dir / "runtime-logs"
            source_dir.mkdir(parents=True)
            runtime_dir.mkdir(parents=True)
            source_file = source_dir / "main.ts"
            source_file.write_text("export const value = 1;\n")
            runtime_log = runtime_dir / "npm-debug.log"
            runtime_log.write_text("first\n")

            before = scoped_workspace_revision(workspace_dir, ("project",))
            runtime_log.write_text("second\n")
            after_runtime_change = scoped_workspace_revision(workspace_dir, ("project",))
            source_file.write_text("export const value = 2;\n")
            after_source_change = scoped_workspace_revision(workspace_dir, ("project",))

            self.assertEqual(after_runtime_change, before)
            self.assertNotEqual(after_source_change, before)

    async def test_workspace_root_revision_ignores_archive_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            source_file = workspace_dir / "main.py"
            source_file.write_text("value = 1\n")
            before = scoped_workspace_revision(workspace_dir, (".",))

            error_dir = workspace_dir / "archive" / "error_logs"
            error_dir.mkdir(parents=True)
            (error_dir / "session.jsonl").write_text("runtime error\n")
            after_archive_change = scoped_workspace_revision(workspace_dir, (".",))

            self.assertEqual(after_archive_change, before)

    async def test_tool_safety_forgets_stale_failure_after_workspace_mutation(self) -> None:
        tracker = ToolSafetyTracker()
        failed_call = _tool_call("bash", {"command": "npx vitest run 2>&1"})
        failed_result = ToolExecutionResult(
            call_id="failed_1",
            name="bash",
            ok=False,
            content="policy denied",
            metadata={"policy_denied": True, "reason": "denied"},
        )
        tracker.record(failed_call, failed_result)
        tracker.record(
            _tool_call("bash", {"command": "generate files"}),
            ToolExecutionResult(
                call_id="mutation",
                name="bash",
                ok=False,
                content="command failed after writing files",
                metadata={
                    "execution_failed": True,
                    "workspace_changed": True,
                    "workspace_revision_before": "before",
                    "workspace_revision_after": "after",
                },
            ),
        )

        observation = tracker.record(failed_call, failed_result)

        self.assertFalse(observation.repeated_invalid_call)
        self.assertEqual(observation.progress_epoch, 1)
        self.assertIsNone(tracker.blocked_call_reason(failed_call))

    async def test_completion_requires_a_revision_bound_gate_ledger(self) -> None:
        tracker = ToolSafetyTracker()
        tracker.record(
            _tool_call("file_write", {"path": "a.py", "content": "x"}),
            ToolExecutionResult(
                call_id="write_1",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True},
            ),
        )
        acceptance = ToolExecutionResult(
            call_id="accept_1",
            name="acceptance_record",
            ok=True,
            content="recorded",
            metadata={
                "acceptance_ledger": {
                    "workspace_revision_verified": True,
                    "workspace_revision": "revision-1",
                    "revision_paths": ["src"],
                    "complete": True,
                    "checks": [
                        {
                            "item_id": "implementation",
                            "outcome": "fixed",
                            "source_tool_call_ids": ["gate_1"],
                        }
                    ],
                }
            },
        )
        tracker.record(_tool_call("acceptance_record", {}), acceptance)
        self.assertTrue(tracker.unverified_workspace_mutation)
        tracker.record(
            _tool_call("acceptance_run", {}),
            ToolExecutionResult(
                call_id="gate_1",
                name="acceptance_run",
                ok=True,
                content="passed",
                metadata={
                    "changed": False,
                    "acceptance_run": {
                        "passed": True,
                        "revision_paths": ["src"],
                        "workspace_revision_after": "revision-1",
                        "gates": [],
                    },
                },
            ),
        )
        tracker.record(_tool_call("acceptance_record", {}), acceptance)
        self.assertFalse(tracker.unverified_workspace_mutation)

        tracker.record(
            _tool_call("file_write", {"path": "a.py", "content": "y"}),
            ToolExecutionResult(
                call_id="write_2",
                name="file_write",
                ok=True,
                content="changed again",
                metadata={"changed": True},
            ),
        )
        tracker.record(_tool_call("acceptance_record", {}), acceptance)
        self.assertTrue(tracker.unverified_workspace_mutation)

    async def test_acceptance_ledger_keeps_prior_required_items_open(self) -> None:
        tracker = ToolSafetyTracker()
        tracker.record(
            _tool_call("file_write", {"path": "a.py", "content": "x"}),
            ToolExecutionResult(
                call_id="write_1",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True},
            ),
        )
        tracker.record(
            _tool_call("acceptance_run", {}),
            ToolExecutionResult(
                call_id="gate_1",
                name="acceptance_run",
                ok=True,
                content="passed",
                metadata={
                    "changed": False,
                    "acceptance_run": {
                        "passed": True,
                        "revision_paths": ["src"],
                        "workspace_revision_after": "revision-1",
                        "gates": [],
                    },
                },
            ),
        )

        def ledger(call_id: str, item_id: str, outcome: str) -> ToolExecutionResult:
            return ToolExecutionResult(
                call_id=call_id,
                name="acceptance_record",
                ok=True,
                content="recorded",
                metadata={
                    "acceptance_ledger": {
                        "workspace_revision_verified": True,
                        "workspace_revision": "revision-1",
                        "revision_paths": ["src"],
                        "complete": outcome != "open",
                        "checks": [
                            {
                                "item_id": item_id,
                                "required": True,
                                "outcome": outcome,
                                "source_tool_call_ids": ["gate_1"],
                            }
                        ],
                    }
                },
            )

        tracker.record(
            _tool_call("acceptance_record", {}),
            ledger("open_ledger", "issue-1", "open"),
        )
        tracker.record(
            _tool_call("acceptance_record", {}),
            ledger("other_ledger", "issue-2", "fixed"),
        )
        self.assertTrue(tracker.unverified_workspace_mutation)
        tracker.record(
            _tool_call("acceptance_record", {}),
            ledger("fixed_ledger", "issue-1", "fixed"),
        )
        self.assertFalse(tracker.unverified_workspace_mutation)

    async def test_acceptance_ledger_rejects_cited_gate_from_different_revision_scope(
        self,
    ) -> None:
        tracker = ToolSafetyTracker()
        tracker.record(
            _tool_call("file_write", {"path": "src/a.py", "content": "x"}),
            ToolExecutionResult(
                call_id="write_1",
                name="file_write",
                ok=True,
                content="changed",
                metadata={"changed": True},
            ),
        )
        tracker.record(
            _tool_call("acceptance_run", {}),
            ToolExecutionResult(
                call_id="gate_1",
                name="acceptance_run",
                ok=True,
                content="passed",
                metadata={
                    "acceptance_run": {
                        "passed": True,
                        "revision_paths": ["src"],
                        "workspace_revision_after": "revision-1",
                        "gates": [],
                    }
                },
            ),
        )
        tracker.record(
            _tool_call("acceptance_record", {}),
            ToolExecutionResult(
                call_id="accept_1",
                name="acceptance_record",
                ok=True,
                content="recorded",
                metadata={
                    "acceptance_ledger": {
                        "workspace_revision_verified": True,
                        "workspace_revision": "revision-1",
                        "revision_paths": ["tests"],
                        "complete": True,
                        "checks": [
                            {
                                "item_id": "implementation",
                                "required": True,
                                "outcome": "fixed",
                                "source_tool_call_ids": ["gate_1"],
                            }
                        ],
                    }
                },
            ),
        )

        self.assertTrue(tracker.unverified_workspace_mutation)

    async def test_managed_service_allocates_owned_port_and_can_be_cancelled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            executor = DirectBashToolExecutor(
                ToolSettings.from_workspace_dir(workspace_dir),
                target_runtime="test",
                runtime_location="test",
                runtime_transport="inprocess",
                container_mutation_boundary="test",
            )
            context = ToolExecutionContext(workspace_dir=workspace_dir)
            result = await executor(
                call_id="service",
                arguments={
                    "mode": "service",
                    "command": "python -m http.server {port} --bind 127.0.0.1",
                    "service_port": 0,
                    "readiness_timeout_seconds": 10,
                },
                context=context,
            )
            self.assertTrue(result.ok, result.content)
            self.assertTrue(result.metadata["readiness_verified"])
            self.assertGreater(int(result.metadata["service_port"]), 0)
            cancel = await executor(
                call_id="cancel_service",
                arguments={"mode": "cancel", "job_id": result.metadata["job_id"]},
                context=context,
            )
            self.assertTrue(cancel.ok)

    async def test_managed_service_rejects_preowned_port(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        ) as listener:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = int(listener.getsockname()[1])
            executor = DirectBashToolExecutor(
                ToolSettings.from_workspace_dir(workspace_dir),
                target_runtime="test",
                runtime_location="test",
                runtime_transport="inprocess",
                container_mutation_boundary="test",
            )
            result = await executor(
                call_id="service",
                arguments={
                    "mode": "service",
                    "command": f"python -m http.server {port} --bind 127.0.0.1",
                    "service_port": port,
                },
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )
            self.assertFalse(result.ok)
            self.assertIn("already owned", result.content)


class BashPolicyReliabilityTests(unittest.TestCase):
    def test_policy_rejects_unmanaged_shell_backgrounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            decision = ToolPolicy().authorize(
                tool_name="bash",
                arguments={"command": "sleep 10 &"},
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

            self.assertFalse(decision.allowed)
            self.assertIn("mode='background'", decision.reason or "")

    def test_policy_ignores_background_words_and_ampersands_inside_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            decision = ToolPolicy().authorize(
                tool_name="bash",
                arguments={"command": "printf '%s' 'nohup & disown'"},
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )
            self.assertTrue(decision.allowed, decision.reason)

    def test_policy_rejects_backgrounding_hidden_in_nested_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            decision = ToolPolicy().authorize(
                tool_name="bash",
                arguments={"command": "bash -lc 'sleep 30 &'"},
                context=ToolExecutionContext(workspace_dir=workspace_dir),
            )

            self.assertFalse(decision.allowed)
            self.assertIn("mode='background'", decision.reason or "")

    def test_policy_allows_shell_operators_that_are_not_backgrounding(self) -> None:
        commands = (
            "npx vitest run 2>&1 | tail -5",
            "printf error >&2",
            "read value <&3",
            "printf output &>result.log",
            "printf output &>>result.log",
            "printf output |& sed -n '1p'",
            "case x in x) printf yes ;& esac",
            "value=$((3 & 1)); printf '%s' \"$value\"",
            "(( value = 3 & 1 )); printf '%s' \"$value\"",
            "cat <<'EOF'\nbackground data & is not syntax\nEOF",
            "cat <<-EOF\n\tbackground data & is not syntax\n\tEOF",
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            for command in commands:
                with self.subTest(command=command):
                    decision = ToolPolicy().authorize(
                        tool_name="bash",
                        arguments={"command": command},
                        context=ToolExecutionContext(workspace_dir=workspace_dir),
                    )
                    self.assertTrue(decision.allowed, decision.reason)

    def test_policy_rejects_backgrounding_across_shell_contexts(self) -> None:
        commands = (
            "sleep 10 & wait",
            "printf ready; sleep 10 &",
            "cat <<EOF\ndata\nEOF\nsleep 10 &",
            "bash -lc 'printf ready; sleep 10 &'",
            "eval 'sleep 10 &'",
            "nohup sleep 10",
            "command setsid sleep 10",
        )
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            for command in commands:
                with self.subTest(command=command):
                    decision = ToolPolicy().authorize(
                        tool_name="bash",
                        arguments={"command": command},
                        context=ToolExecutionContext(workspace_dir=workspace_dir),
                    )
                    self.assertFalse(decision.allowed)
                    self.assertIn("mode='background'", decision.reason or "")
