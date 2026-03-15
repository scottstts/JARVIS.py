"""tool_register definition and executor."""

from __future__ import annotations

from typing import Any, Protocol

from llm import ToolDefinition

from ...runtime_tool_manifest import (
    RuntimeToolManifestError,
    dump_runtime_tool_manifest,
    runtime_tool_manifest_path,
    runtime_tools_dir,
    validate_runtime_tool_manifest_payload,
)
from ...types import RegisteredTool, ToolExecutionContext, ToolExecutionResult

_LOOSE_JSON_VALUE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "additionalProperties": {},
        },
        {
            "type": "array",
            "items": {},
        },
        {
            "type": "string",
        },
        {
            "type": "number",
        },
        {
            "type": "boolean",
        },
        {
            "type": "null",
        },
    ]
}

_MANIFEST_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Runtime tool name, using lowercase letters, digits, and underscores.",
        },
        "purpose": {
            "type": "string",
            "description": "One-line purpose surfaced by tool_search.",
        },
        "aliases": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional alternate search terms for the tool.",
        },
        "detailed_description": {
            "type": "string",
            "description": "Optional long-form description surfaced at high verbosity.",
        },
        "usage": _LOOSE_JSON_VALUE_SCHEMA,
        "notes": _LOOSE_JSON_VALUE_SCHEMA,
        "operator": {
            "type": "string",
            "description": "How the tool is normally operated, usually 'bash'.",
        },
        "invocation": _LOOSE_JSON_VALUE_SCHEMA,
        "provisioning": _LOOSE_JSON_VALUE_SCHEMA,
        "artifacts": _LOOSE_JSON_VALUE_SCHEMA,
        "rebuild": _LOOSE_JSON_VALUE_SCHEMA,
        "safety": _LOOSE_JSON_VALUE_SCHEMA,
    },
    "required": ["name", "purpose", "operator"],
    "additionalProperties": False,
}


class ToolRegisterRegistry(Protocol):
    def get(self, name: str) -> RegisteredTool | None:
        """Return a registered executable tool by name."""

    def get_discoverable(self, name: str):
        """Return a registered built-in discoverable entry by name."""


class ToolRegisterExecutor:
    """Writes runtime tool manifests into the workspace runtime_tools directory."""

    def __init__(self, registry: ToolRegisterRegistry) -> None:
        self._registry = registry

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        raw_manifest = arguments.get("manifest")
        replace_existing = bool(arguments.get("replace_existing", False))
        try:
            manifest = validate_runtime_tool_manifest_payload(raw_manifest)
        except RuntimeToolManifestError as exc:
            return _tool_register_error(call_id=call_id, reason=str(exc))

        if self._registry.get(manifest.name) is not None:
            return _tool_register_error(
                call_id=call_id,
                reason=f"tool name '{manifest.name}' conflicts with a built-in executable tool.",
            )
        built_in_discoverable = self._registry.get_discoverable(manifest.name)
        if built_in_discoverable is not None:
            return _tool_register_error(
                call_id=call_id,
                reason=(
                    f"tool name '{manifest.name}' conflicts with a built-in discoverable tool."
                ),
            )

        target_dir = runtime_tools_dir(context.workspace_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = runtime_tool_manifest_path(context.workspace_dir, manifest.name)
        existed_before = target_path.exists()
        if target_path.exists() and not replace_existing:
            return _tool_register_error(
                call_id=call_id,
                reason=(
                    f"runtime tool manifest '{target_path.name}' already exists; "
                    "set replace_existing=true to overwrite it."
                ),
            )

        temp_path = target_path.with_suffix(".json.tmp")
        temp_path.write_text(
            dump_runtime_tool_manifest(manifest),
            encoding="utf-8",
        )
        temp_path.replace(target_path)

        status = "updated" if existed_before else "registered"
        return ToolExecutionResult(
            call_id=call_id,
            name="tool_register",
            ok=True,
            content=(
                "Runtime tool registered\n"
                f"name: {manifest.name}\n"
                f"status: {status}\n"
                f"manifest_path: {target_path}"
            ),
            metadata={
                "tool_name": manifest.name,
                "status": status,
                "manifest_hash": manifest.manifest_hash(),
                "manifest_path": str(target_path),
                "operator": manifest.operator,
                "replace_existing": replace_existing,
            },
        )


def build_tool_register_tool(registry: ToolRegisterRegistry) -> RegisteredTool:
    """Build the tool_register registry entry."""

    return RegisteredTool(
        name="tool_register",
        exposure="basic",
        definition=ToolDefinition(
            name="tool_register",
            description=(
                "Register or update one runtime tool manifest in /workspace/runtime_tools so "
                "future tool_search calls can discover it. Registration itself requires "
                "explicit user approval."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "manifest": {
                        **_MANIFEST_INPUT_SCHEMA,
                        "description": (
                            "Runtime tool manifest payload. The manifest defines how the "
                            "runtime tool should be discovered and rebuilt."
                        ),
                    },
                    "replace_existing": {
                        "type": "boolean",
                        "description": (
                            "Optional overwrite flag for an existing runtime tool manifest "
                            "with the same name."
                        ),
                    },
                    "approval_summary": {
                        "type": "string",
                        "description": (
                            "Short approval summary shown to the user before registration."
                        ),
                    },
                    "approval_details": {
                        "type": "string",
                        "description": (
                            "Longer approval rationale shown to the user before registration."
                        ),
                    },
                    "inspection_url": {
                        "type": "string",
                        "description": (
                            "Optional URL the user can inspect before approving the registration."
                        ),
                    },
                },
                "required": ["manifest"],
                "additionalProperties": False,
            },
            strict=False,
        ),
        executor=ToolRegisterExecutor(registry),
    )


def _tool_register_error(*, call_id: str, reason: str) -> ToolExecutionResult:
    return ToolExecutionResult(
        call_id=call_id,
        name="tool_register",
        ok=False,
        content=(
            "Runtime tool registration failed\n"
            f"reason: {reason}"
        ),
        metadata={"error": reason},
    )
