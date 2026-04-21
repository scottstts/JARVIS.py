"""Policy checks for tool execution."""

from __future__ import annotations

from jarvis.skills import SkillsSettings

from .basic.bash import BashCommandPolicy
from .basic.file_patch import FilePatchPolicy
from .basic.get_skills import GetSkillsPolicy
from .basic.memory_get import MemoryGetPolicy
from .basic.memory_search import MemorySearchPolicy
from .basic.memory_write import MemoryWritePolicy
from .basic.send_file import SendFilePolicy
from .basic.tool_register import ToolRegisterPolicy
from .basic.tool_search import ToolSearchPolicy
from .basic.web_fetch import WebFetchPolicy
from .basic.web_search import WebSearchPolicy
from .basic.view_image import ViewImagePolicy
from .config import ToolSettings
from .discoverable.email import EmailPolicy
from .discoverable.generate_edit_image import GenerateEditImagePolicy
from .discoverable.memory_admin import MemoryAdminPolicy
from .discoverable.transcribe import TranscribePolicy
from .types import ToolExecutionContext, ToolPolicyDecision


class ToolPolicy:
    """Universal tool policy interface and router."""

    def authorize(
        self,
        *,
        tool_name: str,
        arguments: dict[str, object],
        context: ToolExecutionContext,
    ) -> ToolPolicyDecision:
        if tool_name == "bash":
            settings = ToolSettings.from_workspace_dir(context.workspace_dir)
            return BashCommandPolicy(settings).authorize(
                arguments=arguments,
                context=context,
            )

        if tool_name == "view_image":
            path = str(arguments.get("path", "")).strip()
            return ViewImagePolicy().authorize(path=path, context=context)

        if tool_name == "file_patch":
            path = str(arguments.get("path", "")).strip()
            return FilePatchPolicy().authorize(path=path, context=context)

        if tool_name == "memory_search":
            query = str(arguments.get("query", "")).strip()
            return MemorySearchPolicy().authorize(query=query, context=context)

        if tool_name == "memory_get":
            raw_document_id = arguments.get("document_id")
            document_id = None if raw_document_id is None else str(raw_document_id)
            raw_path = arguments.get("path")
            path = None if raw_path is None else str(raw_path)
            return MemoryGetPolicy().authorize(
                document_id=document_id,
                path=path,
                context=context,
            )

        if tool_name == "memory_write":
            operation = str(arguments.get("operation", "")).strip()
            target_kind = str(arguments.get("target_kind", "")).strip()
            return MemoryWritePolicy().authorize(
                operation=operation,
                target_kind=target_kind,
                arguments=arguments,
                context=context,
            )

        if tool_name == "send_file":
            path = str(arguments.get("path", "")).strip()
            return SendFilePolicy().authorize(path=path, context=context)

        if tool_name == "web_search":
            query = str(arguments.get("query", "")).strip()
            return WebSearchPolicy().authorize(query=query, context=context)

        if tool_name == "web_fetch":
            url = str(arguments.get("url", "")).strip()
            return WebFetchPolicy().authorize(url=url, context=context)

        if tool_name == "tool_search":
            raw_query = arguments.get("query")
            query = None if raw_query is None else str(raw_query)
            raw_verbosity = arguments.get("verbosity")
            verbosity = None if raw_verbosity is None else str(raw_verbosity)
            return ToolSearchPolicy().authorize(
                query=query,
                verbosity=verbosity,
                context=context,
            )

        if tool_name == "get_skills":
            settings = SkillsSettings.from_workspace_dir(context.workspace_dir)
            raw_query = arguments.get("query")
            query = None if raw_query is None else str(raw_query)
            raw_skill_id = arguments.get("skill_id")
            skill_id = None if raw_skill_id is None else str(raw_skill_id)
            return GetSkillsPolicy(settings).authorize(
                mode=str(arguments.get("mode", "")),
                query=query,
                skill_id=skill_id,
                context=context,
            )

        if tool_name == "tool_register":
            return ToolRegisterPolicy().authorize(
                arguments=arguments,
                context=context,
            )

        if tool_name == "generate_edit_image":
            return GenerateEditImagePolicy().authorize(
                arguments=arguments,
                context=context,
            )

        if tool_name == "email":
            settings = ToolSettings.from_workspace_dir(context.workspace_dir)
            return EmailPolicy(settings).authorize(
                arguments=arguments,
                context=context,
            )

        if tool_name == "memory_admin":
            action = str(arguments.get("action", "")).strip()
            return MemoryAdminPolicy().authorize(action=action, context=context)

        if tool_name == "transcribe":
            return TranscribePolicy().authorize(
                arguments=arguments,
                context=context,
            )

        if tool_name not in {
            "bash",
            "file_patch",
            "memory_search",
            "memory_get",
            "memory_write",
            "web_search",
            "web_fetch",
            "view_image",
            "send_file",
            "get_skills",
            "tool_search",
            "tool_register",
            "email",
            "generate_edit_image",
            "memory_admin",
            "transcribe",
        }:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is not implemented in this runtime.",
            )
        return ToolPolicyDecision(allowed=False, reason=f"Tool '{tool_name}' is not implemented.")
