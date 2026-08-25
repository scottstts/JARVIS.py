"""Regression tests for transcript and workspace tool reliability boundaries."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket
import tempfile
import unittest

from jarvis.core.tool_safety import ToolActivityTracker
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
from jarvis.tools.basic.bash.tool import BashToolExecutor
from jarvis.tools.workspace_revision import workspace_revision


def _tool_call(name: str, arguments: dict[str, object], *, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments="",
    )


class ToolReliabilityTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(
        os.getenv("JARVIS_TOOL_RUNTIME_BASE_URL"),
        "requires the isolated tool runtime",
    )
    async def test_isolated_bash_enforces_subagent_write_capabilities(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jarvis-test-", dir="/workspace") as tmp:
            workspace_dir = Path(tmp)
            owned_dir = workspace_dir / "owned"
            denied_dir = workspace_dir / "denied"
            owned_dir.mkdir()
            denied_dir.mkdir()
            executor = BashToolExecutor(
                ToolSettings.from_workspace_dir(workspace_dir)
            )
            context = ToolExecutionContext(
                workspace_dir=workspace_dir,
                agent_kind="subagent",
                agent_name="Friday",
                subagent_id="child",
                workspace_write_allowed_paths=(owned_dir,),
                workspace_write_denied_paths=(denied_dir,),
                workspace_lease_generation=1,
            )

            allowed = await executor(
                call_id="allowed",
                arguments={"command": "printf allowed > owned/result.txt"},
                context=context,
            )
            denied = await executor(
                call_id="denied",
                arguments={"command": "printf denied > denied/result.txt"},
                context=context,
            )
            unowned = await executor(
                call_id="unowned",
                arguments={"command": "printf unowned > outside.txt"},
                context=context,
            )

            self.assertTrue(allowed.ok, allowed.content)
            self.assertFalse(denied.ok)
            self.assertFalse(unowned.ok)
            self.assertEqual((owned_dir / "result.txt").read_text(), "allowed")
            self.assertFalse((denied_dir / "result.txt").exists())
            self.assertFalse((workspace_dir / "outside.txt").exists())

            background = await executor(
                call_id="background",
                arguments={
                    "mode": "background",
                    "command": "printf background > owned/background.txt",
                },
                context=context,
            )
            self.assertTrue(background.ok, background.content)
            job_id = str(background.metadata.get("job_id", ""))
            self.assertTrue(job_id)
            terminal: ToolExecutionResult | None = None
            for attempt in range(100):
                status = await executor(
                    call_id=f"status-{attempt}",
                    arguments={"mode": "status", "job_id": job_id},
                    context=context,
                )
                if status.metadata.get("status") != "running":
                    terminal = status
                    break
                await asyncio.sleep(0.02)
            self.assertIsNotNone(terminal)
            assert terminal is not None
            self.assertEqual(terminal.metadata.get("exit_code"), 0)
            self.assertEqual(
                (owned_dir / "background.txt").read_text(),
                "background",
            )

            main_context = ToolExecutionContext(
                workspace_dir=workspace_dir,
                workspace_write_denied_paths=(owned_dir,),
                workspace_lease_generation=2,
            )
            main_allowed = await executor(
                call_id="main-allowed",
                arguments={"command": "printf main > main.txt"},
                context=main_context,
            )
            main_denied = await executor(
                call_id="main-denied",
                arguments={"command": "printf main > owned/main.txt"},
                context=main_context,
            )
            self.assertTrue(main_allowed.ok, main_allowed.content)
            self.assertFalse(main_denied.ok)
            self.assertEqual((workspace_dir / "main.txt").read_text(), "main")
            self.assertFalse((owned_dir / "main.txt").exists())

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

    async def test_workspace_leases_enforce_actor_filesystem_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            coordinator = WorkspaceAccessCoordinator(workspace_dir=workspace_dir)
            (workspace_dir / "owned.py").write_text("original")
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

            async with coordinator.execute(
                tool_call=_tool_call("bash", {"command": "printf x > other.py"}),
                context=main_context,
            ) as main_observation:
                self.assertIn(
                    workspace_dir / "owned.py",
                    main_observation.write_denied_paths,
                )

            async with coordinator.execute(
                tool_call=_tool_call("bash", {"command": "printf x > owned.py"}),
                context=child_context,
            ) as child_observation:
                self.assertEqual(
                    child_observation.write_allowed_paths,
                    (workspace_dir / "owned.py",),
                )
            with self.assertRaises(WorkspaceLeaseError):
                async with coordinator.execute(
                    tool_call=_tool_call(
                        "file_write",
                        {"path": "outside.py", "content": "x"},
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
                        "file_write",
                        {"path": "src", "content": "directory scope"},
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

    async def test_model_supplied_lease_generation_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            coordinator = WorkspaceAccessCoordinator(workspace_dir=workspace_dir)
            observed = await coordinator.lease_generation()
            (workspace_dir / "other.py").write_text("other")
            await coordinator.claim_paths(owner="subagent:child", paths=("other.py",))
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

            self.assertEqual(observation.mode, "actor_workspace")
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

    async def test_tool_safety_stops_third_identical_no_progress_result(self) -> None:
        tracker = ToolActivityTracker()
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
        tracker = ToolActivityTracker()
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

    async def test_tool_liveness_does_not_merge_varied_workspace_conflicts(self) -> None:
        tracker = ToolActivityTracker()
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
        self.assertFalse(second.repeated_invalid_call)
        repeated_first = tracker.record(first_call, conflict("lease_1", "bash"))
        self.assertTrue(repeated_first.repeated_invalid_call)
        self.assertTrue(repeated_first.blocked_invalid_signature)

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

    async def test_workspace_root_revision_ignores_archive_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            workspace_dir.mkdir()
            source_file = workspace_dir / "main.py"
            source_file.write_text("value = 1\n")
            before = workspace_revision(workspace_dir)

            error_dir = workspace_dir / "archive" / "error_logs"
            error_dir.mkdir(parents=True)
            (error_dir / "session.jsonl").write_text("runtime error\n")
            after_archive_change = workspace_revision(workspace_dir)

            self.assertEqual(after_archive_change, before)

    async def test_tool_safety_forgets_stale_failure_after_workspace_mutation(self) -> None:
        tracker = ToolActivityTracker()
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
