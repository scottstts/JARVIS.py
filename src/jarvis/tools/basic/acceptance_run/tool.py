"""Independent verification gates with durable structured evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shlex
from time import perf_counter
from typing import Any

from jarvis.llm import ToolDefinition

from ...config import ToolSettings
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult
from ...workspace_revision import workspace_revision
from ..bash import BashCommandPolicy
from ..bash.tool import BashToolExecutor

_MAX_GATES = 24
_MAX_GATE_OUTPUT_CHARS = 2_000


class AcceptanceRunToolExecutor:
    """Runs every required gate separately so one success cannot mask another failure."""

    def __init__(self, settings: ToolSettings) -> None:
        self._settings = settings
        self._bash = BashToolExecutor(settings)
        self._policy = BashCommandPolicy(settings)

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        scope = str(arguments.get("scope", "")).strip()
        raw_gates = arguments.get("gates")
        if not scope:
            return _failure(call_id, "scope must be non-empty.")
        if not isinstance(raw_gates, list) or not raw_gates or len(raw_gates) > _MAX_GATES:
            return _failure(call_id, f"gates must contain between 1 and {_MAX_GATES} entries.")

        initial_revision = workspace_revision(context.workspace_dir)
        gate_results: list[dict[str, object]] = []
        for index, raw_gate in enumerate(raw_gates, start=1):
            if not isinstance(raw_gate, dict):
                return _failure(call_id, f"gates[{index}] must be an object.")
            gate_id = str(raw_gate.get("gate_id", "")).strip()
            command = str(raw_gate.get("command", "")).strip()
            if not gate_id or not command:
                return _failure(call_id, f"gates[{index}] needs gate_id and command.")
            operator = _top_level_compound_operator(command)
            if operator is not None:
                return _failure(
                    call_id,
                    f"gates[{index}] contains top-level shell operator {operator!r}; "
                    "split every required check into its own gate.",
                )
            if _uses_shell_command_wrapper(command):
                return _failure(
                    call_id,
                    f"gates[{index}] invokes a shell command wrapper; run the underlying "
                    "verification command directly so nested shell logic cannot mask failure.",
                )
            bash_arguments: dict[str, Any] = {
                "mode": "foreground",
                "command": command,
                "_disable_auto_promote": True,
            }
            if raw_gate.get("cwd") is not None:
                bash_arguments["cwd"] = raw_gate["cwd"]
            if raw_gate.get("timeout_seconds") is not None:
                bash_arguments["timeout_seconds"] = raw_gate["timeout_seconds"]
            decision = self._policy.authorize(arguments=bash_arguments, context=context)
            if not decision.allowed:
                reason = decision.reason or "gate command was denied by bash policy."
                if decision.approval_request is not None:
                    reason += " Run and approve that setup action separately before verification."
                return _failure(call_id, f"gates[{index}]: {reason}")

            revision_before = workspace_revision(context.workspace_dir)
            started = perf_counter()
            result = await self._bash(
                call_id=f"{call_id}:gate:{index}",
                arguments=bash_arguments,
                context=context,
            )
            duration = round(perf_counter() - started, 3)
            revision_after = workspace_revision(context.workspace_dir)
            gate_results.append(
                {
                    "gate_id": gate_id,
                    "command": command,
                    "command_sha256": sha256(command.encode("utf-8")).hexdigest(),
                    "cwd": str(result.metadata.get("cwd", bash_arguments.get("cwd", "/workspace"))),
                    "passed": bool(result.ok and result.metadata.get("exit_code") == 0),
                    "exit_code": result.metadata.get("exit_code"),
                    "duration_seconds": duration,
                    "workspace_revision_before": revision_before,
                    "workspace_revision_after": revision_after,
                    "timed_out": bool(result.metadata.get("timed_out", False)),
                    "output_tail": _tail(result.content, _MAX_GATE_OUTPUT_CHARS),
                    "output_sha256": sha256(result.content.encode("utf-8")).hexdigest(),
                }
            )

        passed = all(bool(gate["passed"]) for gate in gate_results)
        final_revision = workspace_revision(context.workspace_dir)
        lines = [
            "Acceptance gates completed",
            f"scope: {scope}",
            f"workspace_revision_before: {initial_revision or 'unavailable'}",
            f"workspace_revision_after: {final_revision or 'unavailable'}",
            f"passed: {str(passed).lower()}",
        ]
        for gate in gate_results:
            lines.append(
                f"- {gate['gate_id']}: passed={str(gate['passed']).lower()} "
                f"exit_code={gate['exit_code']} duration_seconds={gate['duration_seconds']}"
            )
            lines.append(str(gate["output_tail"]))
        return ToolExecutionResult(
            call_id=call_id,
            name="acceptance_run",
            ok=passed,
            content="\n".join(lines),
            metadata={
                "acceptance_run": {
                    "scope": scope,
                    "workspace_revision_before": initial_revision,
                    "workspace_revision_after": final_revision,
                    "passed": passed,
                    "gates": gate_results,
                },
                "changed": final_revision != initial_revision,
            },
        )


def build_acceptance_run_tool(settings: ToolSettings) -> RegisteredTool:
    return RegisteredTool(
        name="acceptance_run",
        exposure="basic",
        definition=ToolDefinition(
            name="acceptance_run",
            description=(
                "Run required verification gates independently and return one structured ledger. "
                "Each gate must be one command without top-level pipes or command chaining, so a "
                "later success cannot hide an earlier failure."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "minLength": 1, "maxLength": 800},
                    "gates": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_GATES,
                        "items": {
                            "type": "object",
                            "properties": {
                                "gate_id": {"type": "string", "minLength": 1, "maxLength": 120},
                                "command": {"type": "string", "minLength": 1},
                                "cwd": {"type": "string", "minLength": 1},
                                "timeout_seconds": {
                                    "type": "number",
                                    "minimum": 1,
                                    "maximum": settings.bash_max_timeout_seconds,
                                },
                            },
                            "required": ["gate_id", "command"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["scope", "gates"],
                "additionalProperties": False,
            },
        ),
        executor=AcceptanceRunToolExecutor(settings),
    )


def _failure(call_id: str, reason: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id=call_id,
        name="acceptance_run",
        ok=False,
        content=f"Acceptance gates were not run\nerror_code: invalid_acceptance_run\nreason: {reason}",
        metadata={"execution_failed": True, "error_code": "invalid_acceptance_run", "reason": reason},
    )


def _tail(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return "...[gate output truncated]...\n" + value[-limit:]


def _top_level_compound_operator(command: str) -> str | None:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(command):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "\n":
            return "newline"
        if char in {";", "|", "&"}:
            if char in {"|", "&"} and index + 1 < len(command) and command[index + 1] == char:
                return char * 2
            return char
    return None


def _uses_shell_command_wrapper(command: str) -> bool:
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return True
    index = 0
    while index < len(tokens):
        token = tokens[index]
        basename = Path(token).name
        if basename in {"command", "exec"}:
            index += 1
            continue
        if basename == "env":
            index += 1
            while index < len(tokens) and (
                tokens[index].startswith("-") or "=" in tokens[index]
            ):
                index += 1
            continue
        if basename in {"bash", "dash", "fish", "ksh", "sh", "zsh"}:
            return any(
                token == "-c" or (token.startswith("-") and "c" in token[1:])
                for token in tokens[index + 1 :]
            )
        return False
    return False
