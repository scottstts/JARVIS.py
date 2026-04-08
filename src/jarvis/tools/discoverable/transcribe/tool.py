"""Transcribe discoverable tool definition and execution runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from jarvis.llm import ToolDefinition

from ...config import ToolSettings
from ...types import (
    DiscoverableTool,
    RegisteredTool,
    ToolExecutionContext,
    ToolExecutionResult,
)
from .shared import (
    MAX_AUDIO_FILE_SIZE_BYTES,
    SUPPORTED_AUDIO_FORMATS,
    detect_supported_audio_format,
    normalize_audio_path,
    normalize_optional_string,
    resolve_workspace_relative_path,
)

OPENAI_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"
OPENAI_TRANSCRIPTION_RESPONSE_FORMAT = "json"


class TranscribeConfigurationError(RuntimeError):
    """Raised when transcription runtime configuration is missing."""


class TranscribeRequestError(RuntimeError):
    """Raised when the transcription request fails or returns no transcript."""


@dataclass(slots=True, frozen=True)
class TranscriptionPayload:
    text: str
    response_dump: dict[str, Any] | None = None


class TranscribeToolExecutor:
    """Transcribes one workspace audio file to plain text via OpenAI."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        raw_audio_path = normalize_audio_path(arguments.get("audio_path"))
        if raw_audio_path is None:
            return _transcribe_error(
                call_id=call_id,
                raw_audio_path=None,
                reason="audio_path is required.",
            )

        resolved_audio_path = resolve_workspace_relative_path(raw_audio_path, context)
        if not resolved_audio_path.exists():
            return _transcribe_error(
                call_id=call_id,
                raw_audio_path=raw_audio_path,
                reason="input audio file does not exist.",
            )
        if not resolved_audio_path.is_file():
            return _transcribe_error(
                call_id=call_id,
                raw_audio_path=raw_audio_path,
                reason="audio_path must point to a file.",
            )

        audio_format = detect_supported_audio_format(resolved_audio_path)
        if audio_format is None:
            supported = ", ".join(SUPPORTED_AUDIO_FORMATS)
            return _transcribe_error(
                call_id=call_id,
                raw_audio_path=raw_audio_path,
                reason=(
                    "unsupported audio format. Supported formats: "
                    f"{supported}."
                ),
            )

        try:
            file_size_bytes = resolved_audio_path.stat().st_size
        except OSError as exc:
            return _transcribe_error(
                call_id=call_id,
                raw_audio_path=raw_audio_path,
                audio_format=audio_format,
                reason=f"failed to inspect audio file: {exc}",
            )

        if file_size_bytes > MAX_AUDIO_FILE_SIZE_BYTES:
            return _transcribe_error(
                call_id=call_id,
                raw_audio_path=raw_audio_path,
                audio_format=audio_format,
                reason=(
                    "audio file exceeds the OpenAI Audio API 25 MB limit "
                    f"({file_size_bytes} bytes)."
                ),
            )

        try:
            payload = await asyncio.to_thread(
                self._run_openai_request,
                resolved_audio_path,
            )
        except (TranscribeConfigurationError, TranscribeRequestError) as exc:
            return _transcribe_error(
                call_id=call_id,
                raw_audio_path=raw_audio_path,
                audio_format=audio_format,
                reason=str(exc),
            )

        content_lines = [
            "Audio transcription completed",
            f"model: {OPENAI_TRANSCRIPTION_MODEL}",
            f"audio_path: {resolved_audio_path}",
            f"input_format: {audio_format}",
            f"file_size_bytes: {file_size_bytes}",
            "transcript:",
            payload.text,
        ]

        metadata: dict[str, Any] = {
            "model": OPENAI_TRANSCRIPTION_MODEL,
            "response_format": OPENAI_TRANSCRIPTION_RESPONSE_FORMAT,
            "audio_path": str(resolved_audio_path),
            "input_format": audio_format,
            "file_size_bytes": file_size_bytes,
            "transcript_char_count": len(payload.text),
        }
        response_dump = payload.response_dump or {}
        language = normalize_optional_string(response_dump.get("language"))
        if language is not None:
            metadata["language"] = language
            content_lines.insert(5, f"language: {language}")

        duration = response_dump.get("duration")
        if isinstance(duration, int | float):
            metadata["duration_seconds"] = float(duration)
            insert_index = 6 if language is not None else 5
            content_lines.insert(insert_index, f"duration_seconds: {float(duration):.3f}")

        return ToolExecutionResult(
            call_id=call_id,
            name="transcribe",
            ok=True,
            content="\n".join(content_lines),
            metadata=metadata,
        )

    def _run_openai_request(self, audio_path: Path) -> TranscriptionPayload:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise TranscribeConfigurationError(
                "OPENAI_API_KEY is required for transcribe."
            )

        client = OpenAI(
            api_key=api_key,
            timeout=60.0,
            max_retries=2,
        )
        try:
            with audio_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    file=audio_file,
                    model=OPENAI_TRANSCRIPTION_MODEL,
                    response_format=OPENAI_TRANSCRIPTION_RESPONSE_FORMAT,
                )
        except OSError as exc:
            raise TranscribeRequestError(f"failed to read audio file: {exc}") from exc
        except Exception as exc:
            raise _map_openai_error(exc) from exc
        finally:
            client.close()

        transcript_text = _extract_transcript_text(response)
        if transcript_text is None:
            raise TranscribeRequestError("OpenAI did not return transcription text.")

        return TranscriptionPayload(
            text=transcript_text,
            response_dump=_model_dump(response),
        )


def build_transcribe_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the transcribe registry entry."""

    _ = settings
    return RegisteredTool(
        name="transcribe",
        exposure="discoverable",
        definition=ToolDefinition(
            name="transcribe",
            description=_build_transcribe_tool_description(),
            input_schema={
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Required workspace audio or video file path.",
                    }
                },
                "required": ["audio_path"],
                "additionalProperties": False,
            },
        ),
        executor=TranscribeToolExecutor(),
    )


def build_transcribe_discoverable() -> DiscoverableTool:
    """Build the discoverable catalog entry for transcribe."""

    return DiscoverableTool(
        name="transcribe",
        aliases=(
            "speech to text",
            "audio transcription",
            "transcribe audio",
        ),
        purpose=(
            "Transcribe spoken audio from one workspace media file into plain text."
        ),
        detailed_description=_build_transcribe_tool_description(),
        backing_tool_name="transcribe",
    )


def _build_transcribe_tool_description() -> str:
    return (
        "Transcribe one workspace audio or video file to text with OpenAI. `audio_path` "
        "must be under 25 MB; convert or split first if needed."
    )


def _extract_transcript_text(response: Any) -> str | None:
    if isinstance(response, str):
        return normalize_optional_string(response)

    text = normalize_optional_string(getattr(response, "text", None))
    if text is not None:
        return text

    response_dump = _model_dump(response)
    if response_dump is None:
        return None

    for field_name in ("text", "transcript"):
        text = normalize_optional_string(response_dump.get(field_name))
        if text is not None:
            return text
    return None


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


def _map_openai_error(error: Exception) -> TranscribeRequestError:
    if isinstance(error, AuthenticationError):
        return TranscribeConfigurationError(f"OpenAI authentication failed: {error}")
    if isinstance(error, PermissionDeniedError):
        return TranscribeRequestError(f"OpenAI permission denied: {error}")
    if isinstance(error, RateLimitError):
        return TranscribeRequestError(f"OpenAI rate limit: {error}")
    if isinstance(error, APITimeoutError):
        return TranscribeRequestError(f"OpenAI request timed out: {error}")
    if isinstance(error, APIConnectionError):
        return TranscribeRequestError(f"OpenAI request failed: {error}")
    if isinstance(error, BadRequestError):
        return TranscribeRequestError(f"OpenAI rejected the request: {error}")
    if isinstance(error, (APIStatusError, InternalServerError)):
        return TranscribeRequestError(
            f"OpenAI API status {error.status_code}: {error}"
        )
    if isinstance(error, OpenAIError):
        return TranscribeRequestError(f"OpenAI error: {error}")
    return TranscribeRequestError(f"OpenAI request failed: {error}")


def _transcribe_error(
    *,
    call_id: str,
    raw_audio_path: str | None,
    reason: str,
    audio_format: str | None = None,
) -> ToolExecutionResult:
    metadata: dict[str, Any] = {
        "model": OPENAI_TRANSCRIPTION_MODEL,
        "error": reason,
    }
    if raw_audio_path is not None:
        metadata["audio_path"] = raw_audio_path
    if audio_format is not None:
        metadata["input_format"] = audio_format

    content_lines = [
        "Audio transcription failed",
        f"model: {OPENAI_TRANSCRIPTION_MODEL}",
        f"reason: {reason}",
    ]
    if raw_audio_path is not None:
        content_lines.append(f"audio_path: {raw_audio_path}")
    if audio_format is not None:
        content_lines.append(f"input_format: {audio_format}")

    return ToolExecutionResult(
        call_id=call_id,
        name="transcribe",
        ok=False,
        content="\n".join(content_lines),
        metadata=metadata,
    )
