"""YouTube discoverable tool definition and execution runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from llm import ToolDefinition

from ...config import ToolSettings
from ...types import (
    DiscoverableTool,
    RegisteredTool,
    ToolExecutionContext,
    ToolExecutionResult,
)
from .shared import (
    DEFAULT_YOUTUBE_OBJECTIVES,
    GEMINI_YOUTUBE_MODEL,
    MAX_YOUTUBE_VIDEO_URLS,
    build_youtube_system_instruction,
    collect_invalid_video_urls,
    format_invalid_video_urls,
    normalize_optional_string,
    normalize_video_urls,
)

_YOUTUBE_ANALYSIS_USER_PROMPT = (
    "Analyze the provided YouTube videos using the active system instruction."
)


class YouTubeConfigurationError(RuntimeError):
    """Raised when YouTube runtime configuration is missing."""


class YouTubeRequestError(RuntimeError):
    """Raised when the YouTube analysis request fails or returns no text."""


@dataclass(slots=True, frozen=True)
class YouTubeAnalysisPayload:
    text: str
    model: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None


class YouTubeToolExecutor:
    """Analyzes one or more public YouTube video URLs through Gemini."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        _ = context
        video_urls = normalize_video_urls(arguments.get("video_urls"))
        if video_urls is None:
            return _youtube_error(
                call_id=call_id,
                video_urls=None,
                objectives_source="default",
                reason="video_urls is required and must be a non-empty list.",
            )

        invalid_video_urls = collect_invalid_video_urls(video_urls)
        if invalid_video_urls:
            return _youtube_error(
                call_id=call_id,
                video_urls=video_urls,
                objectives_source="default",
                reason=(
                    "video_urls must all be valid YouTube video URLs. Invalid entries: "
                    f"{format_invalid_video_urls(invalid_video_urls)}."
                ),
                invalid_video_urls=invalid_video_urls,
            )

        if len(video_urls) > MAX_YOUTUBE_VIDEO_URLS:
            return _youtube_error(
                call_id=call_id,
                video_urls=video_urls,
                objectives_source="default",
                reason=(
                    "youtube supports at most "
                    f"{MAX_YOUTUBE_VIDEO_URLS} video_urls per call."
                ),
            )

        objectives = normalize_optional_string(arguments.get("objectives"))
        effective_objectives = objectives or DEFAULT_YOUTUBE_OBJECTIVES
        objectives_source = "provided" if objectives is not None else "default"
        system_instruction = build_youtube_system_instruction(effective_objectives)

        try:
            payload = await asyncio.to_thread(
                self._run_gemini_request,
                video_urls,
                system_instruction,
            )
        except (YouTubeConfigurationError, YouTubeRequestError) as exc:
            return _youtube_error(
                call_id=call_id,
                video_urls=video_urls,
                objectives_source=objectives_source,
                reason=str(exc),
            )

        content_lines = [
            "YouTube analysis completed",
            "provider: gemini",
            f"model: {payload.model}",
            f"video_count: {len(video_urls)}",
            f"objectives_source: {objectives_source}",
            "video_urls:",
        ]
        content_lines.extend(f"- {video_url}" for video_url in video_urls)
        content_lines.extend(["analysis:", payload.text])

        metadata: dict[str, Any] = {
            "provider": "gemini",
            "model": payload.model,
            "video_urls": list(video_urls),
            "video_count": len(video_urls),
            "objectives": effective_objectives,
            "objectives_source": objectives_source,
            "response_char_count": len(payload.text),
        }
        if payload.response_id is not None:
            metadata["response_id"] = payload.response_id
        if payload.usage is not None:
            metadata["usage"] = payload.usage

        return ToolExecutionResult(
            call_id=call_id,
            name="youtube",
            ok=True,
            content="\n".join(content_lines),
            metadata=metadata,
        )

    def _run_gemini_request(
        self,
        video_urls: list[str],
        system_instruction: str,
    ) -> YouTubeAnalysisPayload:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise YouTubeConfigurationError("GOOGLE_API_KEY is required for youtube.")

        client = genai.Client(api_key=api_key)
        contents: list[str | genai_types.Part] = [
            genai_types.Part(
                file_data=genai_types.FileData(
                    file_uri=video_url,
                )
            )
            for video_url in video_urls
        ]
        contents.append(_YOUTUBE_ANALYSIS_USER_PROMPT)

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
        )
        try:
            response = client.models.generate_content(
                model=GEMINI_YOUTUBE_MODEL,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise _map_gemini_error(exc) from exc
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()

        response_text = _extract_response_text(response)
        if response_text is None:
            raise YouTubeRequestError("Gemini did not return analysis text.")

        return YouTubeAnalysisPayload(
            text=response_text,
            model=str(getattr(response, "model_version", None) or GEMINI_YOUTUBE_MODEL),
            response_id=normalize_optional_string(getattr(response, "response_id", None)),
            usage=_model_dump(getattr(response, "usage_metadata", None)),
        )


def build_youtube_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the youtube registry entry."""

    _ = settings
    return RegisteredTool(
        name="youtube",
        exposure="discoverable",
        definition=ToolDefinition(
            name="youtube",
            description=(
                "Analyze one or more public YouTube videos by URL using Gemini "
                f"{GEMINI_YOUTUBE_MODEL}. Pass video_urls as a list of valid YouTube video "
                "URLs. The tool always keeps a built-in system instruction with universal "
                "context and guidelines. If objectives is omitted, it uses the default "
                "summary task; if objectives is provided, that custom task replaces only the "
                "default summary objective while preserving the universal system guidance."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "video_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": (
                            "Required list of one or more public YouTube video URLs. Each URL "
                            "must be a valid YouTube video link."
                        ),
                    },
                    "objectives": {
                        "type": "string",
                        "description": (
                            "Optional custom task-specific objective for how to analyze the "
                            "videos. If omitted, the tool uses its default summary objective. "
                            "The tool's universal system instruction remains in place either "
                            "way."
                        ),
                    },
                },
                "required": ["video_urls"],
                "additionalProperties": False,
            },
        ),
        executor=YouTubeToolExecutor(),
    )


def build_youtube_discoverable() -> DiscoverableTool:
    """Build the discoverable catalog entry for youtube."""

    return DiscoverableTool(
        name="youtube",
        aliases=(
            "view_youtube",
            "youtube video",
            "youtube summary",
            "video understanding",
            "summarize youtube",
        ),
        purpose=(
            "Understand one or more public YouTube videos by summarizing them or answering "
            "focused questions about them."
        ),
        detailed_description=(
            "Backed executable discoverable tool that sends public YouTube video URLs "
            f"directly to Gemini {GEMINI_YOUTUBE_MODEL}. The tool validates every URL before "
            "execution, always keeps a shared system instruction with universal context and "
            "guidelines, uses a default summary task when objectives is omitted, and lets "
            "the agent replace only that task portion with a custom objectives string for "
            "targeted detail extraction."
        ),
        usage={
            "arguments": [
                {
                    "name": "video_urls",
                    "type": "string[]",
                    "required": True,
                    "description": "List of one or more public YouTube video URLs.",
                },
                {
                    "name": "objectives",
                    "type": "string",
                    "required": False,
                    "default": "default_summary_objective",
                    "description": (
                        "Optional custom task objective. When provided, it replaces the "
                        "tool's default summary objective but keeps the shared system "
                        "instruction in place."
                    ),
                },
            ],
            "examples": [
                {
                    "mode": "default_summary",
                    "arguments": {
                        "video_urls": [
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        ],
                    },
                },
                {
                    "mode": "focused_extraction",
                    "arguments": {
                        "video_urls": [
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                            "https://youtu.be/3JZ_D3ELwOQ",
                        ],
                        "objectives": (
                            "Context: I do not need a general summary. I am trying to "
                            "reconcile release-timeline claims across these videos for "
                            "downstream planning. Task: compare the videos and extract only "
                            "concrete statements about launch timing, release windows, "
                            "milestones, or deadlines. For each claim, note which video it "
                            "came from and distinguish direct statements from speculation or "
                            "ambiguous hints. Ignore unrelated product details unless they "
                            "materially affect the timeline."
                        ),
                    },
                },
            ],
            "notes": [
                "All video_urls must be valid YouTube video URLs or the tool returns an error before calling Gemini.",
                "Only public YouTube URLs are supported by the upstream Gemini video-understanding path.",
                f"The current implementation allows at most {MAX_YOUTUBE_VIDEO_URLS} video URLs per call.",
                "If objectives is omitted, the tool defaults to a summary-oriented task objective.",
                "If objectives is provided, it replaces only the default task objective and keeps the shared system instruction.",
            ],
        },
        metadata={
            "provider": "gemini",
            "model": GEMINI_YOUTUBE_MODEL,
            "max_video_urls": MAX_YOUTUBE_VIDEO_URLS,
            "default_behavior": "summary",
            "system_instruction_mode": "shared_plus_task_objective",
        },
        backing_tool_name="youtube",
    )


def _extract_response_text(response: Any) -> str | None:
    text_parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                text_parts.append(text)
    if not text_parts:
        return None
    return normalize_optional_string("".join(text_parts))


def _model_dump(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
        return None
    if isinstance(value, dict):
        return dict(value)
    return None


def _map_gemini_error(error: Exception) -> YouTubeRequestError | YouTubeConfigurationError:
    if isinstance(error, (httpx.ReadTimeout, httpx.ConnectTimeout, TimeoutError)):
        return YouTubeRequestError(f"Gemini request timed out: {error}")
    if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
        return YouTubeRequestError(f"Gemini request failed: {error}")
    if isinstance(error, genai_errors.ServerError):
        return YouTubeRequestError(f"Gemini server error: {error}")
    if isinstance(error, genai_errors.ClientError):
        status_code = getattr(error, "code", None)
        if status_code in {401, 403}:
            return YouTubeConfigurationError(str(error))
        return YouTubeRequestError(f"Gemini request rejected: {error}")
    if isinstance(error, genai_errors.APIError):
        status_code = getattr(error, "code", None)
        if status_code in {401, 403}:
            return YouTubeConfigurationError(str(error))
        return YouTubeRequestError(f"Gemini API error: {error}")
    return YouTubeRequestError(f"Gemini request failed: {error}")


def _youtube_error(
    *,
    call_id: str,
    video_urls: list[str] | None,
    objectives_source: str,
    reason: str,
    invalid_video_urls: list[tuple[int, str]] | None = None,
) -> ToolExecutionResult:
    content_lines = [
        "YouTube analysis failed",
        "provider: gemini",
        f"model: {GEMINI_YOUTUBE_MODEL}",
        f"objectives_source: {objectives_source}",
    ]
    metadata: dict[str, Any] = {
        "provider": "gemini",
        "model": GEMINI_YOUTUBE_MODEL,
        "objectives_source": objectives_source,
        "error": reason,
    }

    if video_urls is not None:
        content_lines.append(f"video_count: {len(video_urls)}")
        content_lines.append("video_urls:")
        content_lines.extend(f"- {video_url}" for video_url in video_urls)
        metadata["video_urls"] = list(video_urls)
        metadata["video_count"] = len(video_urls)

    if invalid_video_urls:
        content_lines.append("invalid_video_urls:")
        content_lines.extend(
            f"- [{index}] {video_url or '<empty>'}"
            for index, video_url in invalid_video_urls
        )
        metadata["invalid_video_urls"] = [
            {"index": index, "url": video_url}
            for index, video_url in invalid_video_urls
        ]

    content_lines.append(f"reason: {reason}")

    return ToolExecutionResult(
        call_id=call_id,
        name="youtube",
        ok=False,
        content="\n".join(content_lines),
        metadata=metadata,
    )
