"""Policy checks for tool execution."""

from __future__ import annotations

from .basic.bash import BashCommandPolicy
from .basic.file_patch import FilePatchPolicy
from .basic.python_interpreter import PythonInterpreterPolicy
from .basic.send_file import SendFilePolicy
from .basic.tool_search import ToolSearchPolicy
from .basic.web_fetch import WebFetchPolicy
from .basic.web_search import WebSearchPolicy
from .basic.view_image import ViewImagePolicy
from .config import ToolSettings
from .discoverable.generate_edit_image import GenerateEditImagePolicy
from .discoverable.transcribe import TranscribePolicy
from .discoverable.youtube import YouTubePolicy
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
            command = str(arguments.get("command", "")).strip()
            if not command:
                return ToolPolicyDecision(allowed=False, reason="bash command cannot be empty.")

            return BashCommandPolicy().authorize(command=command, context=context)

        if tool_name == "view_image":
            path = str(arguments.get("path", "")).strip()
            return ViewImagePolicy().authorize(path=path, context=context)

        if tool_name == "file_patch":
            path = str(arguments.get("path", "")).strip()
            return FilePatchPolicy().authorize(path=path, context=context)

        if tool_name == "python_interpreter":
            settings = ToolSettings.from_workspace_dir(context.workspace_dir)
            return PythonInterpreterPolicy(settings).authorize(
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

        if tool_name == "generate_edit_image":
            return GenerateEditImagePolicy().authorize(
                arguments=arguments,
                context=context,
            )

        if tool_name == "transcribe":
            return TranscribePolicy().authorize(
                arguments=arguments,
                context=context,
            )

        if tool_name == "youtube":
            return YouTubePolicy().authorize(
                arguments=arguments,
                context=context,
            )

        if tool_name not in {
            "bash",
            "file_patch",
            "python_interpreter",
            "web_search",
            "web_fetch",
            "view_image",
            "send_file",
            "tool_search",
            "generate_edit_image",
            "transcribe",
            "youtube",
        }:
            return ToolPolicyDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is not implemented in this runtime.",
            )
        return ToolPolicyDecision(allowed=False, reason=f"Tool '{tool_name}' is not implemented.")
