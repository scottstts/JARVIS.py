"""AgentLoop tests covering tool execution rounds."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from base64 import b64decode
from pathlib import Path
from unittest.mock import patch

from jarvis import settings as app_settings

from jarvis.core import (
    AgentApprovalRequestEvent,
    AgentAssistantMessageEvent,
    AgentLoop,
    AgentToolCallEvent,
    AgentTurnDoneEvent,
)
from jarvis.llm import (
    DoneEvent,
    ImagePart,
    LLMConfigurationError,
    LLMRequest,
    LLMResponse,
    ToolDefinition,
    LLMUsage,
    ProviderBadRequestError,
    TextDeltaEvent,
    TextPart,
    ToolCall,
    ToolCallDeltaEvent,
    ToolResultPart,
)
from jarvis.storage import SessionStorage
from tests.helpers import build_core_settings
from jarvis.tools import (
    DiscoverableTool,
    RegisteredTool,
    ToolExecutionResult,
    ToolPolicyDecision,
    ToolRegistry,
    ToolRuntime,
    ToolSettings,
)
from jarvis.tools.basic.web_fetch.tool import HTTPFetchResult

_EXPECTED_BASIC_TOOL_NAMES = [
    "bash",
    "file_patch",
    "memory_search",
    "memory_get",
    "memory_write",
    "web_search",
    "web_fetch",
    "view_image",
    "send_file",
    "tool_search",
    "tool_register",
]


def _bash_python_heredoc(code: str) -> str:
    normalized = code.rstrip("\n")
    return f"python - <<'PY'\n{normalized}\nPY"


def _build_response(
    text: str,
    *,
    tool_calls: list[ToolCall] | None = None,
    finish_reason: str = "stop",
) -> LLMResponse:
    return LLMResponse(
        provider="fake",
        model="fake-chat",
        text=text,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,  # type: ignore[arg-type]
        usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        response_id="resp_fake",
    )


def _is_compaction_request(request: LLMRequest) -> bool:
    return (
        len(request.messages) == 2
        and request.messages[0].role == "system"
        and request.messages[1].role == "user"
        and any(
            isinstance(part, TextPart)
            and "Compact the following transcript." in part.text
            for part in request.messages[1].parts
        )
    )


def _has_summary_seed_message(request: LLMRequest) -> bool:
    return any(
        message.role == "system"
        and isinstance(part, TextPart)
        and "Summarized context from previous session compaction." in part.text
        for message in request.messages
        for part in message.parts
    )


def _build_http_fetch_result(
    *,
    requested_url: str,
    content_type: str,
    body_text: str,
) -> HTTPFetchResult:
    return HTTPFetchResult(
        requested_url=requested_url,
        final_url=requested_url,
        status_code=200,
        headers={"Content-Type": content_type},
        content_type=content_type,
        body_text=body_text,
        redirect_chain=(),
    )


class _FakeToolLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.stream_calls = 0
        self._turn_context_text: str | None = None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        self._assert_bash_registered(request)
        self._assert_turn_context_present(request)

        if self.generate_calls == 1 and self.stream_calls == 0:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="bash_1",
                        name="bash",
                        arguments={"command": "printf 'hello' | tee note.txt"},
                        raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")

        assistant_tool_parts = [
            part for part in assistant_message.parts if isinstance(part, ToolCall)
        ]
        if not assistant_tool_parts:
            raise AssertionError("Expected assistant message to contain tool-call parts.")
        if assistant_tool_parts[0].name != "bash":
            raise AssertionError("Expected bash tool call in assistant history.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "note.txt" not in tool_result_parts[0].content:
            raise AssertionError("Expected tool result content to mention written file.")
        return _build_response("File written.")

    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        self._assert_bash_registered(request)
        self._assert_turn_context_present(request)
        if self.stream_calls == 1:
            yield TextDeltaEvent(delta="Working on it.")
            yield DoneEvent(
                response=_build_response(
                    "Working on it.",
                    tool_calls=[
                        ToolCall(
                            call_id="bash_stream_1",
                            name="bash",
                            arguments={"command": "printf 'hello' | tee note.txt"},
                            raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            )
            return

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "note.txt" not in tool_result_parts[0].content:
            raise AssertionError("Expected streamed tool result content to mention written file.")

        yield TextDeltaEvent(delta="File written.")
        yield DoneEvent(response=_build_response("File written."))

    def _assert_bash_registered(self, request: LLMRequest) -> None:
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

    def _assert_turn_context_present(self, request: LLMRequest) -> None:
        matching_contexts: list[str] = []
        for message in request.messages:
            if message.role != "system":
                continue
            for part in message.parts:
                if not isinstance(part, TextPart):
                    continue
                if "System context auto-appended for this turn only." in part.text:
                    matching_contexts.append(part.text)

        if len(matching_contexts) != 1:
            raise AssertionError(
                "Expected exactly one transient turn datetime context message in the request."
            )
        if f"| {app_settings.JARVIS_CORE_TIMEZONE} time" not in matching_contexts[0]:
            raise AssertionError("Expected turn datetime context to include the configured timezone.")
        if self._turn_context_text is None:
            self._turn_context_text = matching_contexts[0]
        elif matching_contexts[0] != self._turn_context_text:
            raise AssertionError(
                "Expected the same transient turn datetime context across tool follow-up rounds."
            )


class _FakeToolFirstLLMService(_FakeToolLLMService):
    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        self._assert_bash_registered(request)
        self._assert_turn_context_present(request)
        if self.stream_calls == 1:
            yield ToolCallDeltaEvent(
                call_id="bash_stream_1",
                tool_name="bash",
                arguments_delta='{"command":"printf \\"hello\\" | tee note.txt"}',
            )
            yield TextDeltaEvent(delta="Working on it.")
            yield DoneEvent(
                response=_build_response(
                    "Working on it.",
                    tool_calls=[
                        ToolCall(
                            call_id="bash_stream_1",
                            name="bash",
                            arguments={"command": "printf 'hello' | tee note.txt"},
                            raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            )
            return

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "note.txt" not in tool_result_parts[0].content:
            raise AssertionError("Expected streamed tool result content to mention written file.")

        yield TextDeltaEvent(delta="File written.")
        yield DoneEvent(response=_build_response("File written."))


class _ForegroundPromotionToolExecutor:
    async def __call__(self, *, call_id: str, arguments: dict[str, object], context) -> ToolExecutionResult:
        _ = arguments, context
        job_id = "deadbeefdeadbeefdeadbeefdeadbeef"
        return ToolExecutionResult(
            call_id=call_id,
            name="bash",
            ok=True,
            content=(
                "Bash foreground execution moved to background\n"
                f"job_id: {job_id}\n"
                "Use mode='status' or mode='tail'."
            ),
            metadata={
                "mode": "foreground",
                "promoted_to_background": True,
                "job_id": job_id,
                "soft_timeout_seconds": 30.0,
            },
        )


class _FakeForegroundPromotionLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != ["bash"]:
            raise AssertionError(f"Expected only bash to be registered, got {names}.")

        if self.generate_calls == 1:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="bash_promote_1",
                        name="bash",
                        arguments={"command": "long-running command"},
                        raw_arguments='{"command":"long-running command"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result before follow-up.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected exactly one bash tool result part.")
        if "moved to background" not in tool_result_parts[0].content:
            raise AssertionError("Expected tool result to mention background promotion.")

        return _build_response("Monitoring the promoted job.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeViewImageLLMService:
    def __init__(self, expected_image_bytes: bytes) -> None:
        self._expected_image_bytes = expected_image_bytes
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.generate_calls == 1:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="view_image_1",
                        name="view_image",
                        arguments={"path": "temp/sample.png", "detail": "high"},
                        raw_arguments='{"path":"temp/sample.png","detail":"high"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        assistant_message = request.messages[-3]
        tool_message = request.messages[-2]
        attachment_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before attachment follow-up.")
        if attachment_message.role != "user":
            raise AssertionError("Expected transient user attachment message before follow-up.")

        assistant_tool_parts = [
            part for part in assistant_message.parts if isinstance(part, ToolCall)
        ]
        if not assistant_tool_parts or assistant_tool_parts[0].name != "view_image":
            raise AssertionError("Expected view_image tool call in assistant history.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "Image attachment prepared" not in tool_result_parts[0].content:
            raise AssertionError("Expected tool result content to mention prepared image attachment.")

        image_parts = [
            part for part in attachment_message.parts if isinstance(part, ImagePart)
        ]
        text_parts = [
            part for part in attachment_message.parts if isinstance(part, TextPart)
        ]
        if len(image_parts) != 1:
            raise AssertionError("Expected one image attachment part in follow-up history.")
        if not text_parts:
            raise AssertionError("Expected the attachment follow-up message to include text context.")

        payload = image_parts[0].data_url_payload()
        if payload is None:
            raise AssertionError("Expected attachment image to be encoded as a data URL.")
        media_type, data_base64 = payload
        if media_type != "image/png":
            raise AssertionError(f"Expected image/png attachment, got {media_type}.")
        if image_parts[0].detail != "high":
            raise AssertionError(f"Expected detail=high, got {image_parts[0].detail}.")
        if b64decode(data_base64) != self._expected_image_bytes:
            raise AssertionError("Attachment bytes did not match the workspace file.")

        return _build_response("Image inspected.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeViewImageFailureLLMService:
    def __init__(self, expected_image_bytes: bytes) -> None:
        self._expected_image_bytes = expected_image_bytes
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.generate_calls == 1:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="view_image_1",
                        name="view_image",
                        arguments={"path": "temp/sample.png", "detail": "high"},
                        raw_arguments='{"path":"temp/sample.png","detail":"high"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        if self.generate_calls == 2:
            assistant_message = request.messages[-3]
            tool_message = request.messages[-2]
            if assistant_message.role != "assistant":
                raise AssertionError("Expected assistant tool-call message before tool result.")
            if tool_message.role != "tool":
                raise AssertionError("Expected tool result message before follow-up model call.")
            tool_result_parts = [
                part for part in tool_message.parts if isinstance(part, ToolResultPart)
            ]
            if len(tool_result_parts) != 1:
                raise AssertionError("Expected one tool result part before follow-up model call.")
            attachment_message = request.messages[-1]
            if attachment_message.role != "user":
                raise AssertionError("Expected transient user attachment message before follow-up.")
            if "Image attachment prepared" not in tool_result_parts[0].content:
                raise AssertionError("Expected optimistic view_image success before provider rejection.")
            image_parts = [
                part for part in attachment_message.parts if isinstance(part, ImagePart)
            ]
            if len(image_parts) != 1:
                raise AssertionError("Expected one image attachment part in follow-up history.")
            payload = image_parts[0].data_url_payload()
            if payload is None:
                raise AssertionError("Expected attachment image to be encoded as a data URL.")
            _media_type, data_base64 = payload
            if b64decode(data_base64) != self._expected_image_bytes:
                raise AssertionError("Attachment bytes did not match the workspace file.")
            raise LLMConfigurationError("Current model does not support image inputs.")

        if len(request.messages) < 2:
            raise AssertionError("Expected the recovered follow-up to end with a tool error.")
        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before recovered tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected the recovered follow-up to end with a tool error.")
        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one recovered tool result part before follow-up model call.")
        if tool_result_parts[0].is_error is not True:
            raise AssertionError("Expected recovered view_image tool result to be marked as an error.")
        if "Current model does not support image inputs." not in tool_result_parts[0].content:
            raise AssertionError("Expected recovered tool result to include the provider error.")
        if any(
            isinstance(part, ImagePart)
            for message in request.messages
            for part in message.parts
        ):
            raise AssertionError("Recovered follow-up request should not retain the image attachment.")
        return _build_response("Image unavailable.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeStreamingViewImageFailureLLMService:
    def __init__(self, expected_image_bytes: bytes) -> None:
        self._expected_image_bytes = expected_image_bytes
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("Non-streaming generation is not expected in this test.")

    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.stream_calls == 1:
            yield DoneEvent(
                response=_build_response(
                    "",
                    tool_calls=[
                        ToolCall(
                            call_id="view_image_1",
                            name="view_image",
                            arguments={"path": "temp/sample.png", "detail": "high"},
                            raw_arguments='{"path":"temp/sample.png","detail":"high"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            )
            return

        if self.stream_calls == 2:
            assistant_message = request.messages[-3]
            tool_message = request.messages[-2]
            if assistant_message.role != "assistant":
                raise AssertionError("Expected assistant tool-call message before tool result.")
            if tool_message.role != "tool":
                raise AssertionError("Expected tool result message before follow-up model call.")
            tool_result_parts = [
                part for part in tool_message.parts if isinstance(part, ToolResultPart)
            ]
            if len(tool_result_parts) != 1:
                raise AssertionError("Expected one tool result part before follow-up model call.")
            attachment_message = request.messages[-1]
            if attachment_message.role != "user":
                raise AssertionError("Expected transient user attachment message before follow-up.")
            if "Image attachment prepared" not in tool_result_parts[0].content:
                raise AssertionError("Expected optimistic view_image success before provider rejection.")
            image_parts = [
                part for part in attachment_message.parts if isinstance(part, ImagePart)
            ]
            if len(image_parts) != 1:
                raise AssertionError("Expected one image attachment part in follow-up history.")
            payload = image_parts[0].data_url_payload()
            if payload is None:
                raise AssertionError("Expected attachment image to be encoded as a data URL.")
            _media_type, data_base64 = payload
            if b64decode(data_base64) != self._expected_image_bytes:
                raise AssertionError("Attachment bytes did not match the workspace file.")
            raise LLMConfigurationError("Current model does not support image inputs.")

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before recovered tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected recovered tool result message before follow-up model call.")
        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one recovered tool result part before follow-up model call.")
        if tool_result_parts[0].is_error is not True:
            raise AssertionError("Expected recovered view_image tool result to be marked as an error.")
        if "Current model does not support image inputs." not in tool_result_parts[0].content:
            raise AssertionError("Expected recovered tool result to include the provider error.")
        if any(
            isinstance(part, ImagePart)
            for message in request.messages
            for part in message.parts
        ):
            raise AssertionError("Recovered follow-up request should not retain the image attachment.")
        yield DoneEvent(response=_build_response("Image unavailable."))


class _FakeFilePatchLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.generate_calls == 1:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="file_patch_1",
                        name="file_patch",
                        arguments={
                            "path": "notes/todo.txt",
                            "operations": [
                                {
                                    "type": "write",
                                    "content": "hello\n",
                                }
                            ],
                        },
                        raw_arguments=(
                            '{"path":"notes/todo.txt","operations":[{"type":"write",'
                            '"content":"hello\\n"}]}'
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "File patch applied" not in tool_result_parts[0].content:
            raise AssertionError("Expected file_patch tool result content.")
        if "notes/todo.txt" not in tool_result_parts[0].content:
            raise AssertionError("Expected file_patch result to mention the target file.")

        return _build_response("Patched file.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeBashPythonLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.generate_calls == 1:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="bash_python_1",
                        name="bash",
                        arguments={
                            "command": _bash_python_heredoc(
                                "from pathlib import Path\n"
                                "Path('exports/report.txt').write_text('hello', encoding='utf-8')\n"
                                "print('done')\n"
                            ),
                        },
                        raw_arguments='{"command":"python heredoc"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "Bash execution result" not in tool_result_parts[0].content:
            raise AssertionError("Expected bash tool result content.")
        if "done" not in tool_result_parts[0].content:
            raise AssertionError("Expected bash Python result to include command output.")

        return _build_response("Python task finished.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeSendFileLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.generate_calls == 1:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="send_file_1",
                        name="send_file",
                        arguments={
                            "path": "exports/report.txt",
                            "caption": "Weekly report",
                            "filename": "report.txt",
                        },
                        raw_arguments=(
                            '{"path":"exports/report.txt","caption":"Weekly report",'
                            '"filename":"report.txt"}'
                        ),
                    )
                ],
                finish_reason="tool_calls",
            )

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "File sent to Telegram" not in tool_result_parts[0].content:
            raise AssertionError("Expected send_file tool result content.")
        if "chat_id: 123" not in tool_result_parts[0].content:
            raise AssertionError("Expected send_file result to include the resolved Telegram chat.")

        return _build_response("Report delivered.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeWebFetchLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.generate_calls == 1:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="web_fetch_1",
                        name="web_fetch",
                        arguments={"url": "https://example.com/docs"},
                        raw_arguments='{"url":"https://example.com/docs"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        assistant_message = request.messages[-2]
        tool_message = request.messages[-1]
        if assistant_message.role != "assistant":
            raise AssertionError("Expected assistant tool-call message before tool result.")
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")

        tool_result_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_result_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "Web fetch result" not in tool_result_parts[0].content:
            raise AssertionError("Expected web_fetch tool result content.")
        if "strategy: tier1_markdown_accept" not in tool_result_parts[0].content:
            raise AssertionError("Expected web_fetch result to record the tier 1 strategy.")

        return _build_response("Fetched page.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeToolSearchActivationLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1

        if self.generate_calls == 1:
            names = [tool.name for tool in request.tools]
            if names != _EXPECTED_BASIC_TOOL_NAMES:
                raise AssertionError(
                    f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
                )
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="tool_search_1",
                        name="tool_search",
                        arguments={"query": "archive", "verbosity": "high"},
                        raw_arguments='{"query":"archive","verbosity":"high"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        names = [tool.name for tool in request.tools]
        expected = _EXPECTED_BASIC_TOOL_NAMES + ["archive"]
        if names != expected:
            raise AssertionError(f"Expected {expected} tools to be registered, got {names}.")

        tool_message = request.messages[-1]
        if tool_message.role != "tool":
            raise AssertionError("Expected tool_search tool result before follow-up.")
        tool_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_parts) != 1:
            raise AssertionError("Expected one tool result part after tool_search.")
        if "backing_tool_name: archive" not in tool_parts[0].content:
            raise AssertionError("Expected high-verbosity tool_search content for archive.")

        return _build_response("Archive tool surfaced.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeDiscoverableActivationTurnBoundaryLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        names = [tool.name for tool in request.tools]

        if self.generate_calls == 1:
            if names != _EXPECTED_BASIC_TOOL_NAMES:
                raise AssertionError(
                    f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
                )
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="tool_search_1",
                        name="tool_search",
                        arguments={"query": "archive", "verbosity": "high"},
                        raw_arguments='{"query":"archive","verbosity":"high"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        if self.generate_calls == 2:
            expected = _EXPECTED_BASIC_TOOL_NAMES + ["archive"]
            if names != expected:
                raise AssertionError(f"Expected {expected} tools to be registered, got {names}.")
            return _build_response("Archive tool surfaced.")

        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                "Discoverable activation should stay current-turn-only and must not persist "
                f"into a later user turn; got tools {names}."
            )
        return _build_response("New turn.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeToolRegisterApprovalLLMService:
    def __init__(self) -> None:
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("Non-streaming is not expected in this test.")

    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        names = [tool.name for tool in request.tools]
        if self.stream_calls == 1:
            if names != _EXPECTED_BASIC_TOOL_NAMES:
                raise AssertionError(
                    f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
                )
            yield DoneEvent(
                response=_build_response(
                    "",
                    tool_calls=[
                        ToolCall(
                            call_id="tool_register_1",
                            name="tool_register",
                            arguments={
                                "manifest": {
                                    "name": "google_workspace_cli",
                                    "purpose": "Use the Google Workspace CLI from bash.",
                                    "operator": "bash",
                                    "invocation": {"command": "gws --help"},
                                },
                                "approval_summary": "Register google_workspace_cli.",
                                "approval_details": (
                                    "I want to register the tool so future tool_search "
                                    "calls can discover it."
                                ),
                            },
                            raw_arguments="{}",
                        )
                    ],
                    finish_reason="tool_calls",
                )
            )
            return

        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )
        tool_message = request.messages[-1]
        if tool_message.role != "tool":
            raise AssertionError("Expected tool result message before follow-up model call.")
        tool_parts = [
            part for part in tool_message.parts if isinstance(part, ToolResultPart)
        ]
        if len(tool_parts) != 1:
            raise AssertionError("Expected one tool result part before follow-up model call.")
        if "Runtime tool registered" not in tool_parts[0].content:
            raise AssertionError("Expected approved tool_register result content.")

        yield DoneEvent(response=_build_response("Registered runtime tool."))


class _FakeToolRoundLimitLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls += 1
        if request.tool_choice.mode.value == "none":
            names = [tool.name for tool in request.tools]
            if names:
                raise AssertionError("Recovery request should disable all tools.")
            system_message = request.messages[-1]
            if system_message.role != "system":
                raise AssertionError("Expected tool-round-limit system recovery message.")
            return _build_response("Final answer without more tools.")

        if self.generate_calls > 3:
            raise AssertionError("Expected recovery request before another tool round executes.")

        return _build_response(
            "",
            tool_calls=[
                ToolCall(
                    call_id=f"bash_{self.generate_calls}",
                    name="bash",
                    arguments={"command": "printf 'hello' | tee note.txt"},
                    raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                )
            ],
            finish_reason="tool_calls",
        )

    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        if request.tool_choice.mode.value == "none":
            names = [tool.name for tool in request.tools]
            if names:
                raise AssertionError("Recovery request should disable all tools.")
            yield TextDeltaEvent(delta="Final ")
            yield TextDeltaEvent(delta="answer without more tools.")
            yield DoneEvent(response=_build_response("Final answer without more tools."))
            return

        if self.stream_calls > 3:
            raise AssertionError("Expected recovery request before another streamed tool round.")

        yield DoneEvent(
            response=_build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id=f"bash_stream_{self.stream_calls}",
                        name="bash",
                        arguments={"command": "printf 'hello' | tee note.txt"},
                        raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        )


class _FakeFollowupPreflightCompactionLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if _is_compaction_request(request):
            return _build_response("Compacted summary")

        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.generate_calls == 1:
            return _build_response("Seed history.")
        if self.generate_calls == 2:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="bash_compact_1",
                        name="bash",
                        arguments={"command": "printf 'hello' | tee note.txt"},
                        raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                    )
                ],
                finish_reason="tool_calls",
            )

        if not _has_summary_seed_message(request):
            raise AssertionError("Expected the compacted follow-up request to include a summary seed.")
        if request.messages[-2].role != "assistant" or request.messages[-1].role != "tool":
            raise AssertionError("Expected assistant/tool history to survive follow-up compaction.")
        return _build_response("Recovered after compaction.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _FakeStreamingFollowupOverflowCompactionLLMService:
    def __init__(self) -> None:
        self.stream_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if _is_compaction_request(request):
            return _build_response("Compacted summary")
        raise AssertionError("Non-compaction generate should not be used in this test.")

    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.stream_calls == 1:
            yield TextDeltaEvent(delta="Seed history.")
            yield DoneEvent(response=_build_response("Seed history."))
            return
        if self.stream_calls == 2:
            yield DoneEvent(
                response=_build_response(
                    "Working on it.",
                    tool_calls=[
                        ToolCall(
                            call_id="bash_stream_compact_1",
                            name="bash",
                            arguments={"command": "printf 'hello' | tee note.txt"},
                            raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            )
            return
        if self.stream_calls == 3:
            raise ProviderBadRequestError("context_length_exceeded")

        if not _has_summary_seed_message(request):
            raise AssertionError("Expected the compacted streamed follow-up request to include a summary seed.")
        if request.messages[-2].role != "assistant" or request.messages[-1].role != "tool":
            raise AssertionError("Expected assistant/tool history to survive streamed follow-up compaction.")
        yield TextDeltaEvent(delta="Recovered after compaction.")
        yield DoneEvent(response=_build_response("Recovered after compaction."))


class _FakeFollowupOverflowCompactionLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if _is_compaction_request(request):
            return _build_response("Compacted summary")

        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != _EXPECTED_BASIC_TOOL_NAMES:
            raise AssertionError(
                f"Expected {_EXPECTED_BASIC_TOOL_NAMES} tools to be registered, got {names}."
            )

        if self.generate_calls == 1:
            return _build_response("Seed history.")
        if self.generate_calls == 2:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="bash_compact_overflow_1",
                        name="bash",
                        arguments={"command": "printf 'hello' | tee note.txt"},
                        raw_arguments='{"command":"printf \\"hello\\" | tee note.txt"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        if self.generate_calls == 3:
            raise ProviderBadRequestError("context_length_exceeded")

        if not _has_summary_seed_message(request):
            raise AssertionError("Expected the compacted follow-up request to include a summary seed.")
        if request.messages[-2].role != "assistant" or request.messages[-1].role != "tool":
            raise AssertionError("Expected assistant/tool history to survive follow-up compaction.")
        return _build_response("Recovered after compaction.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _BlockingToolExecutor:
    def __init__(
        self,
        *,
        name: str = "slow_tool",
        content: str = "slow tool finished",
    ) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self._name = name
        self._content = content

    async def __call__(self, *, call_id: str, arguments: dict[str, object], context) -> ToolExecutionResult:
        _ = arguments, context
        self.started.set()
        await self.release.wait()
        return ToolExecutionResult(
            call_id=call_id,
            name=self._name,
            ok=True,
            content=self._content,
            metadata={"source": "test"},
        )


class _FakeStopDuringToolLLMService:
    def __init__(
        self,
        *,
        tool_name: str = "slow_tool",
        assistant_text: str = "Using the slow tool.",
    ) -> None:
        self.stream_calls = 0
        self._tool_name = tool_name
        self._assistant_text = assistant_text

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("Non-streaming generate is not expected in this test.")

    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        names = [tool.name for tool in request.tools]
        if names != [self._tool_name]:
            raise AssertionError(f"Expected only {self._tool_name} to be registered, got {names}.")
        if self.stream_calls > 1:
            raise AssertionError("Stop should prevent follow-up model calls after the tool batch.")
        yield DoneEvent(
            response=_build_response(
                self._assistant_text,
                tool_calls=[
                    ToolCall(
                        call_id=f"{self._tool_name}_1",
                        name=self._tool_name,
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
                finish_reason="tool_calls",
            )
        )


class _LargeOutputToolExecutor:
    def __init__(self, content: str) -> None:
        self._content = content

    async def __call__(self, *, call_id: str, arguments: dict[str, object], context) -> ToolExecutionResult:
        _ = arguments, context
        return ToolExecutionResult(
            call_id=call_id,
            name="large_tool",
            ok=True,
            content=self._content,
        )


class _FakeCurrentTurnResidualCompactionLLMService:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if _is_compaction_request(request):
            return _build_response("Compacted current-turn summary")

        self.generate_calls += 1
        names = [tool.name for tool in request.tools]
        if names != ["large_tool"]:
            raise AssertionError(f"Expected only large_tool to be registered, got {names}.")

        if self.generate_calls == 1:
            return _build_response(
                "",
                tool_calls=[
                    ToolCall(
                        call_id="large_tool_1",
                        name="large_tool",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
                finish_reason="tool_calls",
            )

        if not _has_summary_seed_message(request):
            raise AssertionError("Expected the compacted follow-up request to include a summary seed.")
        tool_messages = [message for message in request.messages if message.role == "tool"]
        if len(tool_messages) != 1:
            raise AssertionError("Expected one carried-forward tool result message.")
        tool_text_parts = [
            part.content
            for part in tool_messages[0].parts
            if isinstance(part, ToolResultPart)
        ]
        if len(tool_text_parts) != 1:
            raise AssertionError("Expected one carried-forward tool result part.")
        if len(tool_text_parts[0]) >= 2_500:
            raise AssertionError("Expected carried-forward tool output to be compacted.")
        return _build_response("Recovered after compacting the current turn.")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _InterruptedToolProposalContinuationLLMService:
    def __init__(self) -> None:
        self.stream_started = asyncio.Event()
        self.release_stream = asyncio.Event()
        self.generate_requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_requests.append(request)
        if any(
            isinstance(part, ToolCall) and part.name == "bash"
            for message in request.messages
            if message.role == "assistant"
            for part in message.parts
        ):
            raise AssertionError("Dangling interrupted bash tool calls should be normalized before prompt build.")
        if not any(
            isinstance(part, TextPart)
            and "bash" in part.text
            and "Treat them as not run." in part.text
            for message in request.messages
            if message.role == "system"
            for part in message.parts
        ):
            raise AssertionError("Expected a normalization note for the unexecuted interrupted tool call.")
        return _build_response("continued")

    async def stream_generate(self, request: LLMRequest):
        if "bash" not in [tool.name for tool in request.tools]:
            raise AssertionError("Expected bash to be available in the initial tool request.")
        yield TextDeltaEvent(delta="I'll use bash with Python.")
        self.stream_started.set()
        await self.release_stream.wait()
        yield DoneEvent(
            response=_build_response(
                "I'll use bash with Python.",
                tool_calls=[
                    ToolCall(
                        call_id="python_1",
                        name="bash",
                        arguments={"command": "python -c pass"},
                        raw_arguments='{"command":"python -c pass"}',
                    )
                ],
                finish_reason="tool_calls",
            )
        )


class _InterruptedCompletedToolContinuationLLMService:
    def __init__(self) -> None:
        self.stream_calls = 0
        self.generate_requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_requests.append(request)
        if not any(
            isinstance(part, ToolCall) and part.name == "slow_tool"
            for message in request.messages
            if message.role == "assistant"
            for part in message.parts
        ):
            raise AssertionError("Expected the interrupted turn's completed slow_tool call to remain in context.")
        if not any(
            isinstance(part, ToolResultPart) and part.content == "slow tool finished"
            for message in request.messages
            if message.role == "tool"
            for part in message.parts
        ):
            raise AssertionError("Expected the interrupted turn's completed tool result to remain in context.")
        return _build_response("continued after tool")

    async def stream_generate(self, request: LLMRequest):
        self.stream_calls += 1
        names = [tool.name for tool in request.tools]
        if names != ["slow_tool"]:
            raise AssertionError(f"Expected only slow_tool to be registered, got {names}.")
        if self.stream_calls > 1:
            raise AssertionError("Stop should prevent follow-up model calls after the tool batch.")
        yield DoneEvent(
            response=_build_response(
                "Using the slow tool.",
                tool_calls=[
                    ToolCall(
                        call_id="slow_tool_1",
                        name="slow_tool",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
                finish_reason="tool_calls",
            )
        )


class _OrphanedTurnRecoveryLLMService:
    def __init__(self, *, expect_unexecuted_tool_notice: bool) -> None:
        self.expect_unexecuted_tool_notice = expect_unexecuted_tool_notice
        self.generate_requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_requests.append(request)
        if not any(
            isinstance(part, TextPart) and part.text == "Continue after recovery."
            for message in request.messages
            if message.role == "user"
            for part in message.parts
        ):
            raise AssertionError("Expected the new user message in the recovered-turn request.")
        if not any(
            isinstance(part, TextPart) and part.text == "Work in progress."
            for message in request.messages
            if message.role == "user"
            for part in message.parts
        ):
            raise AssertionError("Expected the orphaned turn user message to remain in replay.")
        if not any(
            isinstance(part, TextPart) and part.text == "Partial answer."
            for message in request.messages
            if message.role == "assistant"
            for part in message.parts
        ):
            raise AssertionError("Expected the orphaned turn assistant text to remain in replay.")
        if not any(
            isinstance(part, TextPart)
            and "ended unexpectedly before it completed" in part.text
            for message in request.messages
            if message.role == "system"
            for part in message.parts
        ):
            raise AssertionError("Expected a persisted orphaned-turn recovery note in replay.")
        if self.expect_unexecuted_tool_notice:
            if any(
                isinstance(part, ToolCall)
                for message in request.messages
                if message.role == "assistant"
                for part in message.parts
            ):
                raise AssertionError(
                    "Recovered orphaned turns should not replay unresolved assistant tool calls."
                )
            if not any(
                isinstance(part, TextPart)
                and "Treat them as not run." in part.text
                for message in request.messages
                if message.role == "system"
                for part in message.parts
            ):
                raise AssertionError(
                    "Expected a persisted unexecuted-tool-call notice during orphaned-turn recovery."
                )
        return _build_response("continued")

    async def stream_generate(self, request: LLMRequest):
        raise AssertionError("Streaming is not expected in this test.")


class _StopMidToolCallStreamingLLMService:
    def __init__(self) -> None:
        self.tool_call_started = asyncio.Event()
        self.release_stream = asyncio.Event()
        self.stream_closed = asyncio.Event()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("Non-streaming generate is not expected in this test.")

    async def stream_generate(self, request: LLMRequest):
        try:
            names = [tool.name for tool in request.tools]
            if "bash" not in names:
                raise AssertionError("Expected bash to be available in the request.")
            yield TextDeltaEvent(delta="I'll use Python for this.")
            yield ToolCallDeltaEvent(
                call_id="python_1",
                tool_name="bash",
                arguments_delta='{"command":"python -c pass',
            )
            self.tool_call_started.set()
            await self.release_stream.wait()
            yield ToolCallDeltaEvent(
                call_id="python_1",
                tool_name="bash",
                arguments_delta='"}',
            )
            yield DoneEvent(
                response=_build_response(
                    "I'll use Python for this.",
                    tool_calls=[
                        ToolCall(
                            call_id="python_1",
                            name="bash",
                            arguments={"command": "python -c pass"},
                            raw_arguments='{"command":"python -c pass"}',
                        )
                    ],
                    finish_reason="tool_calls",
                )
            )
        finally:
            self.stream_closed.set()


class _AllowAllToolPolicy:
    def authorize(self, *, tool_name: str, arguments: dict[str, object], context) -> ToolPolicyDecision:
        _ = tool_name, arguments, context
        return ToolPolicyDecision(allowed=True)


class AgentLoopToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_user_input_recovers_when_tool_round_limit_is_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)

            with patch.dict("os.environ", {"JARVIS_TOOL_MAX_ROUNDS_PER_TURN": "2"}):
                loop = AgentLoop(
                    llm_service=_FakeToolRoundLimitLLMService(),
                    settings=settings,
                    storage=storage,
                )

            result = await loop.handle_user_input("Keep working until done.")

            self.assertEqual(result.response_text, "Final answer without more tools.")
            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            assistant_records = [record for record in message_records if record.role == "assistant"]
            self.assertEqual(len(assistant_records), 4)
            self.assertEqual(assistant_records[-1].content, "Final answer without more tools.")
            self.assertTrue(any(record.metadata.get("tool_round_limit") for record in message_records))

    async def test_stream_user_input_recovers_when_tool_round_limit_is_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)

            with patch.dict("os.environ", {"JARVIS_TOOL_MAX_ROUNDS_PER_TURN": "2"}):
                loop = AgentLoop(
                    llm_service=_FakeToolRoundLimitLLMService(),
                    settings=settings,
                    storage=storage,
                )

            events = [event async for event in loop.stream_user_input("Keep working until done.")]

            self.assertEqual(events[-1].type, "done")
            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertEqual(done.response_text, "Final answer without more tools.")
            assistant_messages = [
                event.text
                for event in events
                if isinstance(event, AgentAssistantMessageEvent)
            ]
            self.assertEqual(assistant_messages[-1], "Final answer without more tools.")

    async def test_handle_user_input_executes_bash_tool_round_and_persists_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeToolLLMService(),
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("Write hello into note.txt.")

            self.assertEqual(result.response_text, "File written.")
            self.assertEqual(
                (settings.workspace_dir / "note.txt").read_text(encoding="utf-8"),
                "hello",
            )

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(message_records[-3].content, "")
            self.assertEqual(message_records[-3].metadata["tool_calls"][0]["name"], "bash")
            self.assertIn("Bash execution result", message_records[-2].content)

    async def test_handle_user_input_handles_bash_foreground_promotion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            registry = ToolRegistry(
                tools=[
                    RegisteredTool(
                        name="bash",
                        exposure="basic",
                        definition=ToolDefinition(
                            name="bash",
                            input_schema={
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                },
                                "required": ["command"],
                                "additionalProperties": False,
                            },
                            description="Test bash tool.",
                        ),
                        executor=_ForegroundPromotionToolExecutor(),
                    )
                ]
            )
            loop = AgentLoop(
                llm_service=_FakeForegroundPromotionLLMService(),
                settings=settings,
                storage=storage,
                tool_registry=registry,
                tool_runtime=ToolRuntime(
                    registry=registry,
                    policy=_AllowAllToolPolicy(),
                ),
            )

            result = await loop.handle_user_input("Run the long bash command.")

            self.assertEqual(result.response_text, "Monitoring the promoted job.")

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertTrue(
                message_records[-2].metadata.get("promoted_to_background", False)
            )
            self.assertFalse(
                any(
                    record.role == "system"
                    and record.metadata.get("bash_background_promotion")
                    for record in message_records
                )
            )

    async def test_handle_user_input_auto_compacts_when_followup_preflight_budget_is_exceeded(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeFollowupPreflightCompactionLLMService(),
                settings=settings,
                storage=storage,
            )

            seeded = await loop.handle_user_input("Seed history.")

            def _estimate_with_followup_overflow(request: LLMRequest) -> int:
                has_tool_result = any(message.role == "tool" for message in request.messages)
                has_summary_seed = _has_summary_seed_message(request)
                if has_tool_result and not has_summary_seed:
                    return settings.context_policy.preflight_limit_tokens
                return 10

            with patch(
                "jarvis.core.agent_loop.estimate_request_input_tokens",
                side_effect=_estimate_with_followup_overflow,
            ):
                result = await loop.handle_user_input("Write hello into note.txt.")

            self.assertEqual(result.response_text, "Recovered after compaction.")
            self.assertTrue(result.compaction_performed)
            self.assertNotEqual(result.session_id, seeded.session_id)
            self.assertEqual(
                (settings.workspace_dir / "note.txt").read_text(encoding="utf-8"),
                "hello",
            )

            old_session = storage.get_session(seeded.session_id)
            self.assertIsNotNone(old_session)
            self.assertEqual(old_session.status, "archived")  # type: ignore[union-attr]

            new_records = storage.load_records(result.session_id)
            self.assertTrue(
                any(record.role == "system" and record.metadata.get("summary_seed") for record in new_records)
            )
            self.assertEqual(new_records[-1].role, "assistant")
            self.assertEqual(new_records[-1].content, "Recovered after compaction.")

    async def test_handle_user_input_auto_compacts_when_current_turn_itself_overflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            large_output = "L" * 8_000
            registry = ToolRegistry(
                tools=[
                    RegisteredTool(
                        name="large_tool",
                        exposure="basic",
                        definition=ToolDefinition(
                            name="large_tool",
                            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                            description="Large-output test tool.",
                        ),
                        executor=_LargeOutputToolExecutor(large_output),
                    )
                ]
            )
            loop = AgentLoop(
                llm_service=_FakeCurrentTurnResidualCompactionLLMService(),
                settings=settings,
                storage=storage,
                tool_registry=registry,
                tool_runtime=ToolRuntime(
                    registry=registry,
                    policy=_AllowAllToolPolicy(),
                ),
            )

            def _estimate_current_turn_overflow(request: LLMRequest) -> int:
                tool_parts = [
                    part.content
                    for message in request.messages
                    if message.role == "tool"
                    for part in message.parts
                    if isinstance(part, ToolResultPart)
                ]
                if not tool_parts:
                    return 10
                has_summary_seed = _has_summary_seed_message(request)
                if not has_summary_seed:
                    return settings.context_policy.preflight_limit_tokens
                if max(len(content) for content in tool_parts) >= 2_500:
                    return settings.context_policy.preflight_limit_tokens
                return 10

            with patch(
                "jarvis.core.agent_loop.estimate_request_input_tokens",
                side_effect=_estimate_current_turn_overflow,
            ):
                result = await loop.handle_user_input("Use the large tool.")

            self.assertEqual(result.response_text, "Recovered after compacting the current turn.")
            self.assertTrue(result.compaction_performed)

            new_session = storage.get_session(result.session_id)
            self.assertIsNotNone(new_session)
            self.assertEqual(new_session.start_reason, "compaction")  # type: ignore[union-attr]
            self.assertIsNotNone(new_session.parent_session_id)  # type: ignore[union-attr]

            new_records = storage.load_records(result.session_id, include_all_turns=True)
            self.assertTrue(
                any(record.role == "system" and record.metadata.get("summary_seed") for record in new_records)
            )
            new_tool_records = [record for record in new_records if record.role == "tool"]
            self.assertEqual(len(new_tool_records), 1)
            self.assertTrue(new_tool_records[0].metadata.get("carry_forward_compacted"))
            self.assertLess(len(new_tool_records[0].content), 2_500)

            archived_session_id = new_session.parent_session_id  # type: ignore[union-attr]
            archived_session = storage.get_session(archived_session_id)
            self.assertIsNotNone(archived_session)
            self.assertEqual(archived_session.status, "archived")  # type: ignore[union-attr]
            archived_records = storage.load_records(archived_session_id, include_all_turns=True)
            archived_tool_records = [record for record in archived_records if record.role == "tool"]
            self.assertEqual(len(archived_tool_records), 1)
            self.assertEqual(archived_tool_records[0].content, large_output)

    async def test_stream_user_input_completes_tool_round_and_emits_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeToolLLMService(),
                settings=settings,
                storage=storage,
            )

            events = [event async for event in loop.stream_user_input("Write hello into note.txt.")]

            self.assertEqual(
                [event.type for event in events],
                [
                    "turn_started",
                    "text_delta",
                    "assistant_message",
                    "tool_call",
                    "text_delta",
                    "assistant_message",
                    "done",
                ],
            )
            self.assertIsInstance(events[2], AgentAssistantMessageEvent)
            self.assertIsInstance(events[3], AgentToolCallEvent)
            self.assertIsInstance(events[5], AgentAssistantMessageEvent)
            first_message = events[2]
            if not isinstance(first_message, AgentAssistantMessageEvent):
                self.fail("Expected assistant message stream event before tool execution.")
            self.assertEqual(first_message.text, "Working on it.")
            tool_event = events[3]
            if not isinstance(tool_event, AgentToolCallEvent):
                self.fail("Expected tool-call stream event before tool execution.")
            self.assertEqual(tool_event.tool_names, ("bash",))
            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertEqual(done.response_text, "File written.")

    async def test_stream_user_input_preserves_tool_first_order_when_tool_delta_arrives_first(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeToolFirstLLMService(),
                settings=settings,
                storage=storage,
            )

            events = [event async for event in loop.stream_user_input("Write hello into note.txt.")]

            self.assertEqual(
                [event.type for event in events],
                [
                    "turn_started",
                    "tool_call",
                    "text_delta",
                    "assistant_message",
                    "text_delta",
                    "assistant_message",
                    "done",
                ],
            )
            tool_event = events[1]
            if not isinstance(tool_event, AgentToolCallEvent):
                self.fail("Expected first stream event to be a tool-call notice.")
            self.assertEqual(tool_event.tool_names, ("bash",))
            first_message = events[3]
            if not isinstance(first_message, AgentAssistantMessageEvent):
                self.fail("Expected the mixed response text to still be finalized.")
            self.assertEqual(first_message.text, "Working on it.")

    async def test_stream_user_input_interrupts_after_current_tool_batch_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            tool_executor = _BlockingToolExecutor()
            registry = ToolRegistry(
                tools=[
                    RegisteredTool(
                        name="slow_tool",
                        exposure="basic",
                        definition=ToolDefinition(
                            name="slow_tool",
                            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                            description="Slow test tool.",
                        ),
                        executor=tool_executor,
                    )
                ]
            )
            loop = AgentLoop(
                llm_service=_FakeStopDuringToolLLMService(),
                settings=settings,
                storage=storage,
                tool_registry=registry,
                tool_runtime=ToolRuntime(
                    registry=registry,
                    policy=_AllowAllToolPolicy(),
                ),
            )

            async def _collect_events():
                return [event async for event in loop.stream_user_input("Use the slow tool.")]

            task = asyncio.create_task(_collect_events())
            await tool_executor.started.wait()

            self.assertTrue(loop.request_stop())
            tool_executor.release.set()
            events = await task

            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertTrue(done.interrupted)
            self.assertEqual(done.response_text, "Using the slow tool.")

            all_records = storage.load_records(done.session_id, include_all_turns=True)
            message_records = [record for record in all_records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "system")
            self.assertEqual(message_records[-2].content, "slow tool finished")
            self.assertTrue(
                message_records[-2].metadata.get("completed_after_interrupt_request")
            )

    async def test_stream_user_input_persists_text_when_stop_interrupts_mid_tool_call_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _StopMidToolCallStreamingLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            async def _collect_events():
                return [event async for event in loop.stream_user_input("Use python.")]

            task = asyncio.create_task(_collect_events())
            await llm_service.tool_call_started.wait()

            self.assertTrue(loop.request_stop())
            llm_service.release_stream.set()
            events = await task
            await llm_service.stream_closed.wait()

            self.assertEqual(
                [event.type for event in events],
                ["turn_started", "text_delta", "tool_call", "done"],
            )
            tool_event = events[2]
            if not isinstance(tool_event, AgentToolCallEvent):
                self.fail("Expected the second event to be an AgentToolCallEvent.")
            self.assertEqual(tool_event.tool_names, ("bash",))

            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertTrue(done.interrupted)
            self.assertEqual(done.response_text, "I'll use Python for this.")

            records = storage.load_records(done.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(
                [record.role for record in message_records[-4:]],
                ["user", "assistant", "system", "system"],
            )
            self.assertEqual(message_records[-3].content, "I'll use Python for this.")
            self.assertEqual(message_records[-3].metadata.get("tool_calls"), None)
            self.assertTrue(message_records[-2].metadata.get("unexecuted_tool_call_notice"))
            self.assertIn("Treat them as not run.", message_records[-2].content)
            self.assertFalse(any(record.role == "tool" for record in message_records))

    async def test_next_turn_normalizes_unexecuted_interrupted_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            llm_service = _InterruptedToolProposalContinuationLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
            )

            async def _collect_events():
                return [event async for event in loop.stream_user_input("Use python.")]

            task = asyncio.create_task(_collect_events())
            await llm_service.stream_started.wait()

            self.assertTrue(loop.request_stop())
            llm_service.release_stream.set()
            events = await task

            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertTrue(done.interrupted)

            result = await loop.handle_user_input("continue")
            self.assertEqual(result.response_text, "continued")
            self.assertEqual(len(llm_service.generate_requests), 1)

    async def test_next_turn_keeps_completed_tool_results_from_interrupted_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            tool_executor = _BlockingToolExecutor()
            registry = ToolRegistry(
                tools=[
                    RegisteredTool(
                        name="slow_tool",
                        exposure="basic",
                        definition=ToolDefinition(
                            name="slow_tool",
                            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                            description="Slow test tool.",
                        ),
                        executor=tool_executor,
                    )
                ]
            )
            llm_service = _InterruptedCompletedToolContinuationLLMService()
            loop = AgentLoop(
                llm_service=llm_service,
                settings=settings,
                storage=storage,
                tool_registry=registry,
                tool_runtime=ToolRuntime(
                    registry=registry,
                    policy=_AllowAllToolPolicy(),
                ),
            )

            async def _collect_events():
                return [event async for event in loop.stream_user_input("Use the slow tool.")]

            task = asyncio.create_task(_collect_events())
            await tool_executor.started.wait()

            self.assertTrue(loop.request_stop())
            tool_executor.release.set()
            events = await task

            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertTrue(done.interrupted)

            result = await loop.handle_user_input("continue")
            self.assertEqual(result.response_text, "continued after tool")
            self.assertEqual(len(llm_service.generate_requests), 1)

    async def test_handle_user_input_recovers_orphaned_in_progress_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            setup_loop = AgentLoop(
                llm_service=_FakeToolLLMService(),
                settings=settings,
                storage=storage,
            )
            session_id = await setup_loop.prepare_session()
            turn_id = "orphan_turn"
            storage.append_record(
                session_id,
                setup_loop._build_message_record(
                    session_id=session_id,
                    role="user",
                    content="Work in progress.",
                    turn_id=turn_id,
                ),
            )
            storage.append_record(
                session_id,
                setup_loop._build_message_record(
                    session_id=session_id,
                    role="assistant",
                    content="Partial answer.",
                    turn_id=turn_id,
                ),
            )
            storage.set_turn_status(
                session_id,
                turn_id=turn_id,
                status="in_progress",
            )

            loop = AgentLoop(
                llm_service=_OrphanedTurnRecoveryLLMService(
                    expect_unexecuted_tool_notice=False
                ),
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("Continue after recovery.")

            self.assertEqual(result.response_text, "continued")
            session = storage.get_session(session_id)
            self.assertIsNotNone(session)
            self.assertEqual(session.turn_states[turn_id], "interrupted")  # type: ignore[index]

            all_records = storage.load_records(session_id, include_all_turns=True)
            recovered_records = [
                record
                for record in all_records
                if str(record.metadata.get("turn_id", "")).strip() == turn_id
            ]
            self.assertEqual(
                [record.role for record in recovered_records[-3:]],
                ["user", "assistant", "system"],
            )
            self.assertTrue(
                recovered_records[-1].metadata.get("orphaned_turn_recovery")
            )
            self.assertIn(
                "ended unexpectedly before it completed",
                recovered_records[-1].content,
            )

    async def test_handle_user_input_recovers_orphaned_unexecuted_tool_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            setup_loop = AgentLoop(
                llm_service=_FakeToolLLMService(),
                settings=settings,
                storage=storage,
            )
            session_id = await setup_loop.prepare_session()
            turn_id = "orphan_tool_turn"
            storage.append_record(
                session_id,
                setup_loop._build_message_record(
                    session_id=session_id,
                    role="user",
                    content="Work in progress.",
                    turn_id=turn_id,
                ),
            )
            storage.append_record(
                session_id,
                setup_loop._build_message_record(
                    session_id=session_id,
                    role="assistant",
                    content="Partial answer.",
                    metadata={
                        "tool_calls": [
                            {
                                "call_id": "bash_1",
                                "name": "bash",
                                "arguments": {"command": "printf hi"},
                                "raw_arguments": '{"command":"printf hi"}',
                                "provider_metadata": {},
                            }
                        ]
                    },
                    turn_id=turn_id,
                ),
            )
            storage.set_turn_status(
                session_id,
                turn_id=turn_id,
                status="in_progress",
            )

            loop = AgentLoop(
                llm_service=_OrphanedTurnRecoveryLLMService(
                    expect_unexecuted_tool_notice=True
                ),
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("Continue after recovery.")

            self.assertEqual(result.response_text, "continued")
            session = storage.get_session(session_id)
            self.assertIsNotNone(session)
            self.assertEqual(session.turn_states[turn_id], "interrupted")  # type: ignore[index]

            all_records = storage.load_records(session_id, include_all_turns=True)
            recovered_records = [
                record
                for record in all_records
                if str(record.metadata.get("turn_id", "")).strip() == turn_id
            ]
            self.assertEqual(
                [record.role for record in recovered_records[-4:]],
                ["user", "assistant", "system", "system"],
            )
            self.assertTrue(
                recovered_records[-2].metadata.get("unexecuted_tool_call_notice")
            )
            self.assertIn("Treat them as not run.", recovered_records[-2].content)
            self.assertTrue(
                recovered_records[-1].metadata.get("orphaned_turn_recovery")
            )

    async def test_stream_user_input_stops_after_large_tool_batch_without_compacting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            large_output = "tool-output-" * 800
            tool_executor = _BlockingToolExecutor(
                name="large_tool",
                content=large_output,
            )
            registry = ToolRegistry(
                tools=[
                    RegisteredTool(
                        name="large_tool",
                        exposure="basic",
                        definition=ToolDefinition(
                            name="large_tool",
                            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                            description="Large-output stop test tool.",
                        ),
                        executor=tool_executor,
                    )
                ]
            )
            loop = AgentLoop(
                llm_service=_FakeStopDuringToolLLMService(
                    tool_name="large_tool",
                    assistant_text="Using the large tool.",
                ),
                settings=settings,
                storage=storage,
                tool_registry=registry,
                tool_runtime=ToolRuntime(
                    registry=registry,
                    policy=_AllowAllToolPolicy(),
                ),
            )

            def _estimate_with_tool_overflow(request: LLMRequest) -> int:
                if any(message.role == "tool" for message in request.messages):
                    return settings.context_policy.preflight_limit_tokens
                return 10

            async def _collect_events():
                return [event async for event in loop.stream_user_input("Use the large tool.")]

            with patch(
                "jarvis.core.agent_loop.estimate_request_input_tokens",
                side_effect=_estimate_with_tool_overflow,
            ):
                task = asyncio.create_task(_collect_events())
                await tool_executor.started.wait()

                self.assertTrue(loop.request_stop())
                tool_executor.release.set()
                events = await task

            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertTrue(done.interrupted)
            self.assertFalse(done.compaction_performed)
            self.assertEqual(done.response_text, "Using the large tool.")

            session = storage.get_session(done.session_id)
            self.assertIsNotNone(session)
            self.assertEqual(session.compaction_count, 0)  # type: ignore[union-attr]

            all_records = storage.load_records(done.session_id, include_all_turns=True)
            tool_records = [record for record in all_records if record.role == "tool"]
            self.assertEqual(len(tool_records), 1)
            self.assertEqual(tool_records[0].content, large_output)

    async def test_handle_user_input_auto_compacts_when_followup_provider_overflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeFollowupOverflowCompactionLLMService(),
                settings=settings,
                storage=storage,
            )

            seeded = await loop.handle_user_input("Seed history.")
            result = await loop.handle_user_input("Write hello into note.txt.")

            self.assertEqual(result.response_text, "Recovered after compaction.")
            self.assertTrue(result.compaction_performed)
            self.assertNotEqual(result.session_id, seeded.session_id)
            self.assertEqual(
                (settings.workspace_dir / "note.txt").read_text(encoding="utf-8"),
                "hello",
            )

            old_session = storage.get_session(seeded.session_id)
            self.assertIsNotNone(old_session)
            self.assertEqual(old_session.status, "archived")  # type: ignore[union-attr]

            new_records = storage.load_records(result.session_id)
            self.assertTrue(
                any(record.role == "system" and record.metadata.get("summary_seed") for record in new_records)
            )
            self.assertEqual(new_records[-1].content, "Recovered after compaction.")

    async def test_stream_user_input_auto_compacts_when_followup_provider_overflows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeStreamingFollowupOverflowCompactionLLMService(),
                settings=settings,
                storage=storage,
            )

            seeded_events = [event async for event in loop.stream_user_input("Seed history.")]
            seeded_done = seeded_events[-1]
            if not isinstance(seeded_done, AgentTurnDoneEvent):
                self.fail("Expected the seeded turn to complete with a done event.")

            events = [event async for event in loop.stream_user_input("Write hello into note.txt.")]

            done = events[-1]
            if not isinstance(done, AgentTurnDoneEvent):
                self.fail("Expected final stream event to be AgentTurnDoneEvent.")
            self.assertEqual(done.response_text, "Recovered after compaction.")
            self.assertTrue(done.compaction_performed)
            self.assertNotEqual(done.session_id, seeded_done.session_id)
            self.assertEqual(
                (settings.workspace_dir / "note.txt").read_text(encoding="utf-8"),
                "hello",
            )

            old_session = storage.get_session(seeded_done.session_id)
            self.assertIsNotNone(old_session)
            self.assertEqual(old_session.status, "archived")  # type: ignore[union-attr]

            assistant_messages = [
                event.text
                for event in events
                if isinstance(event, AgentAssistantMessageEvent)
            ]
            self.assertEqual(assistant_messages[-1], "Recovered after compaction.")

            new_records = storage.load_records(done.session_id)
            self.assertTrue(
                any(record.role == "system" and record.metadata.get("summary_seed") for record in new_records)
            )
            self.assertEqual(new_records[-1].content, "Recovered after compaction.")

    async def test_handle_user_input_executes_view_image_tool_and_keeps_attachment_transient(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            image_path = settings.workspace_dir / "temp" / "sample.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_bytes = b"\x89PNG\r\n\x1a\nfake_png_payload"
            image_path.write_bytes(image_bytes)

            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeViewImageLLMService(image_bytes),
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input(
                "Inspect the image at local_path: temp/sample.png."
            )

            self.assertEqual(result.response_text, "Image inspected.")

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(message_records[-3].metadata["tool_calls"][0]["name"], "view_image")
            self.assertIn("Image attachment prepared", message_records[-2].content)
            self.assertTrue(all(not record.metadata.get("transient", False) for record in message_records))

    async def test_handle_user_input_converts_view_image_provider_rejection_into_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            image_path = settings.workspace_dir / "temp" / "sample.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_bytes = b"\x89PNG\r\n\x1a\nfake_png_payload"
            image_path.write_bytes(image_bytes)

            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeViewImageFailureLLMService(image_bytes),
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input(
                "Inspect the image at local_path: temp/sample.png."
            )

            self.assertEqual(result.response_text, "Image unavailable.")

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(message_records[-3].metadata["tool_calls"][0]["name"], "view_image")
            self.assertNotIn("Image attachment prepared", message_records[-2].content)
            self.assertIn("View image failed", message_records[-2].content)
            self.assertIn("Current model does not support image inputs.", message_records[-2].content)
            self.assertFalse(message_records[-2].metadata["ok"])

    async def test_stream_user_input_converts_view_image_provider_rejection_into_tool_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            image_path = settings.workspace_dir / "temp" / "sample.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_bytes = b"\x89PNG\r\n\x1a\nfake_png_payload"
            image_path.write_bytes(image_bytes)

            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeStreamingViewImageFailureLLMService(image_bytes),
                settings=settings,
                storage=storage,
            )

            events = [event async for event in loop.stream_user_input(
                "Inspect the image at local_path: temp/sample.png."
            )]

            done_events = [event for event in events if isinstance(event, AgentTurnDoneEvent)]
            self.assertEqual(len(done_events), 1)
            self.assertEqual(done_events[0].response_text, "Image unavailable.")

            records = storage.load_records(done_events[0].session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertIn("View image failed", message_records[-2].content)
            self.assertIn("Current model does not support image inputs.", message_records[-2].content)
            self.assertFalse(message_records[-2].metadata["ok"])

    async def test_handle_user_input_executes_file_patch_tool_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            notes_dir = settings.workspace_dir / "notes"
            notes_dir.mkdir(parents=True, exist_ok=True)

            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeFilePatchLLMService(),
                settings=settings,
                storage=storage,
            )

            result = await loop.handle_user_input("Create notes/todo.txt with hello.")

            self.assertEqual(result.response_text, "Patched file.")
            self.assertEqual(
                (notes_dir / "todo.txt").read_text(encoding="utf-8"),
                "hello\n",
            )

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(message_records[-3].metadata["tool_calls"][0]["name"], "file_patch")
            self.assertIn("File patch applied", message_records[-2].content)

    async def test_handle_user_input_executes_bash_python_tool_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            exports_dir = settings.workspace_dir / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)

            storage = SessionStorage(settings.transcript_archive_dir)
            with patch.dict(
                os.environ,
                {
                    "JARVIS_TOOL_CENTRAL_PYTHON_VENV": str(
                        Path(sys.executable).resolve().parent.parent
                    )
                },
                clear=False,
            ):
                loop = AgentLoop(
                    llm_service=_FakeBashPythonLLMService(),
                    settings=settings,
                    storage=storage,
                )
                result = await loop.handle_user_input(
                    "Use python to write hello into exports/report.txt."
                )

            self.assertEqual(result.response_text, "Python task finished.")
            self.assertEqual(
                (exports_dir / "report.txt").read_text(encoding="utf-8"),
                "hello",
            )

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(
                message_records[-3].metadata["tool_calls"][0]["name"],
                "bash",
            )
            self.assertIn("Bash execution result", message_records[-2].content)

    async def test_handle_user_input_executes_send_file_tool_with_route_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            report_path = settings.workspace_dir / "exports" / "report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("weekly report", encoding="utf-8")

            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeSendFileLLMService(),
                settings=settings,
                storage=storage,
                route_id="tg_123",
            )

            async def _fake_send_telegram_file(**kwargs):
                self.assertEqual(kwargs["route_id"], "tg_123")
                self.assertEqual(
                    kwargs["file_path"].resolve(strict=False),
                    report_path.resolve(strict=False),
                )
                return {"message_id": 7, "chat_id": 123}

            with patch(
                "jarvis.tools.basic.send_file.tool.send_telegram_file",
                side_effect=_fake_send_telegram_file,
            ):
                result = await loop.handle_user_input("Send me the report file.")

            self.assertEqual(result.response_text, "Report delivered.")

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-3].metadata["tool_calls"][0]["name"], "send_file")
            self.assertIn("File sent to Telegram", message_records[-2].content)

    async def test_handle_user_input_executes_web_fetch_tool_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeWebFetchLLMService(),
                settings=settings,
                storage=storage,
            )

            tier1_result = _build_http_fetch_result(
                requested_url="https://example.com/docs",
                content_type="text/markdown",
                body_text="# Example Docs\n\nHello from the fetched page.",
            )

            with patch(
                "jarvis.tools.basic.web_fetch.tool._fetch_http_text",
                return_value=tier1_result,
            ):
                result = await loop.handle_user_input("Fetch https://example.com/docs for me.")

            self.assertEqual(result.response_text, "Fetched page.")

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(message_records[-3].metadata["tool_calls"][0]["name"], "web_fetch")
            self.assertIn("Web fetch result", message_records[-2].content)
            self.assertEqual(message_records[-2].metadata["strategy"], "tier1_markdown_accept")

    async def test_handle_user_input_surfaces_discoverable_tools_after_high_verbosity_tool_search(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            registry = ToolRegistry.default(
                ToolSettings.from_workspace_dir(settings.workspace_dir)
            )

            async def _archive_executor(**kwargs):
                raise AssertionError("archive should not be executed in this test.")

            registry.register(
                RegisteredTool(
                    name="archive",
                    exposure="discoverable",
                    definition=ToolDefinition(
                        name="archive",
                        description="Archive workspace files.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    ),
                    executor=_archive_executor,
                )
            )
            registry.register_discoverable(
                DiscoverableTool(
                    name="archive",
                    aliases=("zip_tools",),
                    purpose="List, extract, and create archive formats inside the workspace.",
                    detailed_description="Use this to inspect or manipulate zip and tar archives.",
                    usage={"arguments": [{"name": "path", "type": "string"}]},
                    metadata={"family": "filesystem"},
                    backing_tool_name="archive",
                )
            )

            loop = AgentLoop(
                llm_service=_FakeToolSearchActivationLLMService(),
                settings=settings,
                storage=storage,
                tool_registry=registry,
            )

            result = await loop.handle_user_input("Find an archive tool.")

            self.assertEqual(result.response_text, "Archive tool surfaced.")

            records = storage.load_records(result.session_id)
            message_records = [record for record in records if record.kind == "message"]
            self.assertEqual(message_records[-4].role, "user")
            self.assertEqual(message_records[-3].role, "assistant")
            self.assertEqual(message_records[-2].role, "tool")
            self.assertEqual(message_records[-1].role, "assistant")
            self.assertEqual(message_records[-3].metadata["tool_calls"][0]["name"], "tool_search")
            self.assertEqual(
                message_records[-2].metadata["activated_discoverable_tool_names"],
                ["archive"],
            )

    async def test_handle_user_input_keeps_discoverable_activation_current_turn_only(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            registry = ToolRegistry.default(
                ToolSettings.from_workspace_dir(settings.workspace_dir)
            )

            async def _archive_executor(**kwargs):
                raise AssertionError("archive should not be executed in this test.")

            registry.register(
                RegisteredTool(
                    name="archive",
                    exposure="discoverable",
                    definition=ToolDefinition(
                        name="archive",
                        description="Archive workspace files.",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    ),
                    executor=_archive_executor,
                )
            )
            registry.register_discoverable(
                DiscoverableTool(
                    name="archive",
                    aliases=("zip_tools",),
                    purpose="List, extract, and create archive formats inside the workspace.",
                    detailed_description="Use this to inspect or manipulate zip and tar archives.",
                    usage={"arguments": [{"name": "path", "type": "string"}]},
                    metadata={"family": "filesystem"},
                    backing_tool_name="archive",
                )
            )

            loop = AgentLoop(
                llm_service=_FakeDiscoverableActivationTurnBoundaryLLMService(),
                settings=settings,
                storage=storage,
                tool_registry=registry,
            )

            first_result = await loop.handle_user_input("Find an archive tool.")
            second_result = await loop.handle_user_input("Continue.")

            self.assertEqual(first_result.response_text, "Archive tool surfaced.")
            self.assertEqual(second_result.response_text, "New turn.")

    async def test_stream_user_input_emits_approval_request_and_resumes_after_tool_register_approval(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeToolRegisterApprovalLLMService(),
                settings=settings,
                storage=storage,
            )

            events: list[object] = []

            async def _collect_events() -> None:
                async for event in loop.stream_user_input("Register a runtime tool."):
                    events.append(event)

            collector = asyncio.create_task(_collect_events())
            approval_event: AgentApprovalRequestEvent | None = None
            for _ in range(100):
                await asyncio.sleep(0.01)
                approval_event = next(
                    (
                        event
                        for event in events
                        if isinstance(event, AgentApprovalRequestEvent)
                    ),
                    None,
                )
                if approval_event is not None:
                    break

            self.assertIsNotNone(approval_event)
            if approval_event is None:
                self.fail("Expected approval request event before stream completion.")
            self.assertTrue(loop.resolve_approval(approval_event.approval_id, True))
            await asyncio.wait_for(collector, timeout=2.0)

            self.assertIsInstance(events[-1], AgentTurnDoneEvent)
            final_event = events[-1]
            if not isinstance(final_event, AgentTurnDoneEvent):
                self.fail("Expected streamed turn to finish with AgentTurnDoneEvent.")
            self.assertEqual(final_event.response_text, "Registered runtime tool.")
            self.assertTrue(
                (
                    settings.workspace_dir
                    / "runtime_tools"
                    / "google_workspace_cli.json"
                ).exists()
            )

    async def test_stream_user_input_returns_rejection_text_when_tool_register_approval_is_denied(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = build_core_settings(root_dir=Path(tmp))
            storage = SessionStorage(settings.transcript_archive_dir)
            loop = AgentLoop(
                llm_service=_FakeToolRegisterApprovalLLMService(),
                settings=settings,
                storage=storage,
            )

            events: list[object] = []

            async def _collect_events() -> None:
                async for event in loop.stream_user_input("Register a runtime tool."):
                    events.append(event)

            collector = asyncio.create_task(_collect_events())
            approval_event: AgentApprovalRequestEvent | None = None
            for _ in range(100):
                await asyncio.sleep(0.01)
                approval_event = next(
                    (
                        event
                        for event in events
                        if isinstance(event, AgentApprovalRequestEvent)
                    ),
                    None,
                )
                if approval_event is not None:
                    break

            self.assertIsNotNone(approval_event)
            if approval_event is None:
                self.fail("Expected approval request event before stream completion.")
            self.assertTrue(loop.resolve_approval(approval_event.approval_id, False))
            await asyncio.wait_for(collector, timeout=2.0)

            self.assertIsInstance(events[-1], AgentTurnDoneEvent)
            final_event = events[-1]
            if not isinstance(final_event, AgentTurnDoneEvent):
                self.fail("Expected streamed turn to finish with AgentTurnDoneEvent.")
            self.assertEqual(
                final_event.response_text,
                "Approval request was rejected. I did not execute the action.",
            )
            self.assertFalse(
                (
                    settings.workspace_dir
                    / "runtime_tools"
                    / "google_workspace_cli.json"
                ).exists()
            )
