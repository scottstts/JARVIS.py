"""Policy checks for the tool_register tool."""

from __future__ import annotations

from typing import Any

from ...runtime_tool_manifest import RuntimeToolManifestError, validate_runtime_tool_manifest_payload
from ...types import ToolExecutionContext, ToolPolicyDecision


class ToolRegisterPolicy:
    """Requires exact-action approval before writing runtime tool manifests."""

    def authorize(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        raw_manifest = arguments.get("manifest")
        try:
            manifest = validate_runtime_tool_manifest_payload(raw_manifest)
        except RuntimeToolManifestError as exc:
            return ToolPolicyDecision(allowed=False, reason=str(exc))

        approved_action = context.approved_action or {}
        if (
            approved_action.get("kind") == "register_runtime_tool"
            and approved_action.get("tool_name") == manifest.name
            and approved_action.get("manifest_hash") == manifest.manifest_hash()
        ):
            return ToolPolicyDecision(allowed=True)

        summary = str(arguments.get("approval_summary", "")).strip()
        details = str(arguments.get("approval_details", "")).strip()
        inspection_url = str(arguments.get("inspection_url", "")).strip()
        if not summary:
            summary = f"Register runtime tool '{manifest.name}'."
        if not details:
            details = (
                f"I want to register the runtime tool '{manifest.name}' so it can be discovered "
                "through tool_search in future turns."
            )

        return ToolPolicyDecision(
            allowed=False,
            reason="tool_register requires explicit approval.",
            approval_request={
                "kind": "register_runtime_tool",
                "tool_name": manifest.name,
                "summary": summary,
                "details": details,
                "inspection_url": inspection_url or None,
                "manifest_hash": manifest.manifest_hash(),
            },
        )
