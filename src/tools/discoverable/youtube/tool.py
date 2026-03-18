"""YouTube discoverable tool definition and execution runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import subprocess
from typing import Any
from urllib.parse import quote

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
_DEFUDDLE_BASE_URL = "https://defuddle.md/"
_DEFUDDLE_TRANSCRIPT_TIMEOUT_SECONDS = 60


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


@dataclass(slots=True, frozen=True)
class YouTubeTranscriptPayload:
    video_url: str
    defuddle_url: str
    text: str


class YouTubeToolExecutor:
    """Analyzes YouTube videos through Gemini or fetches transcripts via Defuddle."""

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
                provider="gemini",
                mode="analysis",
                model=GEMINI_YOUTUBE_MODEL,
                objectives_source="default",
                reason="video_urls is required and must be a non-empty list.",
            )

        invalid_video_urls = collect_invalid_video_urls(video_urls)
        if invalid_video_urls:
            return _youtube_error(
                call_id=call_id,
                video_urls=video_urls,
                provider="gemini",
                mode="analysis",
                model=GEMINI_YOUTUBE_MODEL,
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
                provider="gemini",
                mode="analysis",
                model=GEMINI_YOUTUBE_MODEL,
                objectives_source="default",
                reason=(
                    "youtube supports at most "
                    f"{MAX_YOUTUBE_VIDEO_URLS} video_urls per call."
                ),
            )

        transcript_argument = arguments.get("transcript")
        if transcript_argument is None:
            transcript_requested = False
        elif isinstance(transcript_argument, bool):
            transcript_requested = transcript_argument
        else:
            return _youtube_error(
                call_id=call_id,
                video_urls=video_urls,
                provider="youtube",
                mode="validation",
                reason="transcript must be a boolean when provided.",
            )

        if transcript_requested:
            objectives_were_ignored = arguments.get("objectives") is not None
            try:
                transcripts = await asyncio.to_thread(
                    self._run_defuddle_transcript_request,
                    video_urls,
                )
            except (YouTubeConfigurationError, YouTubeRequestError) as exc:
                return _youtube_error(
                    call_id=call_id,
                    video_urls=video_urls,
                    provider="defuddle",
                    mode="transcript",
                    reason=str(exc),
                    transcript_requested=True,
                )

            total_transcript_chars = sum(len(item.text) for item in transcripts)
            content_lines = [
                "YouTube transcript retrieval completed",
                "provider: defuddle",
                "mode: transcript",
                f"video_count: {len(video_urls)}",
                f"objectives_ignored: {'yes' if objectives_were_ignored else 'no'}",
                "video_urls:",
            ]
            content_lines.extend(f"- {video_url}" for video_url in video_urls)
            for index, transcript in enumerate(transcripts, start=1):
                content_lines.extend(
                    [
                        f"video_{index}_url: {transcript.video_url}",
                        f"video_{index}_transcript:",
                        transcript.text,
                    ]
                )

            return ToolExecutionResult(
                call_id=call_id,
                name="youtube",
                ok=True,
                content="\n".join(content_lines),
                metadata={
                    "provider": "defuddle",
                    "mode": "transcript",
                    "transcript_requested": True,
                    "video_urls": list(video_urls),
                    "video_count": len(video_urls),
                    "response_char_count": total_transcript_chars,
                    "objectives_ignored": objectives_were_ignored,
                    "transcripts": [
                        {
                            "video_url": transcript.video_url,
                            "defuddle_url": transcript.defuddle_url,
                            "transcript_char_count": len(transcript.text),
                        }
                        for transcript in transcripts
                    ],
                },
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
                provider="gemini",
                mode="analysis",
                model=GEMINI_YOUTUBE_MODEL,
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

    def _run_defuddle_transcript_request(
        self,
        video_urls: list[str],
    ) -> list[YouTubeTranscriptPayload]:
        transcripts: list[YouTubeTranscriptPayload] = []
        for video_url in video_urls:
            defuddle_url = _build_defuddle_url(video_url)
            try:
                result = subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--silent",
                        "--show-error",
                        "--location",
                        defuddle_url,
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    check=False,
                    timeout=_DEFUDDLE_TRANSCRIPT_TIMEOUT_SECONDS,
                )
            except FileNotFoundError as exc:
                raise YouTubeConfigurationError(
                    "curl is required for youtube transcript mode."
                ) from exc
            except subprocess.TimeoutExpired as exc:
                raise YouTubeRequestError(
                    f"Defuddle transcript request timed out for {video_url}."
                ) from exc
            except OSError as exc:
                raise YouTubeRequestError(
                    f"Defuddle transcript request failed for {video_url}: {exc}"
                ) from exc

            if result.returncode != 0:
                error_details = (
                    normalize_optional_string(result.stderr)
                    or normalize_optional_string(result.stdout)
                    or f"curl exited with status {result.returncode}."
                )
                raise YouTubeRequestError(
                    f"Defuddle transcript request failed for {video_url}: "
                    f"{error_details}"
                )

            transcript_text = normalize_optional_string(result.stdout)
            if transcript_text is None:
                raise YouTubeRequestError(
                    f"Defuddle returned an empty transcript for {video_url}."
                )

            transcripts.append(
                YouTubeTranscriptPayload(
                    video_url=video_url,
                    defuddle_url=defuddle_url,
                    text=transcript_text,
                )
            )

        return transcripts

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
            description=_build_youtube_tool_description(),
            input_schema={
                "type": "object",
                "properties": {
                    "video_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Required list of 1 to 10 public YouTube video URLs.",
                    },
                    "objectives": {
                        "type": "string",
                        "description": "Optional analysis instructions.",
                    },
                    "transcript": {
                        "type": "boolean",
                        "description": "Transcript mode.",
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
            "youtube summary",
            "youtube transcript",
            "video understanding",
        ),
        purpose=(
            "Understand one or more public YouTube videos by summarizing them, answering "
            "focused questions, or retrieving transcript text for them."
        ),
        detailed_description=_build_youtube_tool_description(),
        usage={
            "arguments": [
                {
                    "name": "video_urls",
                    "type": "string[]",
                    "required": True,
                },
                {
                    "name": "objectives",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "transcript",
                    "type": "boolean",
                    "required": False,
                },
            ],
        },
        backing_tool_name="youtube",
    )


def _build_youtube_tool_description() -> str:
    return (
        "Analyze public YouTube videos, or fetch transcripts when exact wording matters. "
        "Set `transcript=true` for transcript mode; otherwise use optional `objectives` "
        "to steer analysis."
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
    provider: str,
    mode: str,
    reason: str,
    objectives_source: str | None = None,
    model: str | None = None,
    invalid_video_urls: list[tuple[int, str]] | None = None,
    transcript_requested: bool | None = None,
) -> ToolExecutionResult:
    content_lines = [
        (
            "YouTube transcript retrieval failed"
            if mode == "transcript"
            else "YouTube analysis failed"
        ),
        f"provider: {provider}",
        f"mode: {mode}",
    ]
    metadata: dict[str, Any] = {
        "provider": provider,
        "mode": mode,
        "error": reason,
    }
    if model is not None:
        content_lines.append(f"model: {model}")
        metadata["model"] = model
    if objectives_source is not None:
        content_lines.append(f"objectives_source: {objectives_source}")
        metadata["objectives_source"] = objectives_source
    if transcript_requested is not None:
        content_lines.append(f"transcript_requested: {str(transcript_requested).lower()}")
        metadata["transcript_requested"] = transcript_requested

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


def _build_defuddle_url(video_url: str) -> str:
    return f"{_DEFUDDLE_BASE_URL}{quote(video_url, safe='')}"
