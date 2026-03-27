"""Generate-edit-image tool definition and execution runtime."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
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

_DEFAULT_PROVIDER = "gemini"
_SUPPORTED_PROVIDERS = ("gemini", "openai")
_OPENAI_IMAGE_MODEL = "gpt-image-1.5"
_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
_OPENAI_QUALITY_VALUES = ("low", "medium", "high")
_DEFAULT_OPENAI_QUALITY = "medium"
_GEMINI_RESOLUTION_VALUES = ("512", "1K", "2K", "4K")
_DEFAULT_GEMINI_RESOLUTION = "1K"
_SUPPORTED_IMAGE_MEDIA_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class GenerateEditImageConfigurationError(RuntimeError):
    """Raised when a provider is missing required runtime configuration."""


class GenerateEditImageRequestError(RuntimeError):
    """Raised when a provider request fails or returns an invalid payload."""


@dataclass(slots=True, frozen=True)
class GeneratedImagePayload:
    provider: str
    model: str
    image_bytes: bytes
    mime_type: str
    provider_text: str | None = None
    revised_prompt: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None


class GenerateEditImageToolExecutor:
    """Generates or edits one image through Gemini or OpenAI."""

    async def __call__(
        self,
        *,
        call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolExecutionResult:
        prompt = str(arguments["prompt"]).strip()
        provider = _normalize_provider(arguments.get("provider"))
        quality = _normalize_openai_quality(arguments.get("quality")) or _DEFAULT_OPENAI_QUALITY
        resolution = (
            _normalize_gemini_resolution(arguments.get("resolution"))
            or _DEFAULT_GEMINI_RESOLUTION
        )
        raw_output_path = str(arguments["output_path"]).strip()
        resolved_output_path = _resolve_workspace_relative_path(raw_output_path, context)
        raw_image_path = _normalize_optional_string(arguments.get("image_path"))
        operation = "edit" if raw_image_path is not None else "generate"

        resolved_input_path: Path | None = None
        input_media_type: str | None = None
        if raw_image_path is not None:
            resolved_input_path = _resolve_workspace_relative_path(raw_image_path, context)
            if not resolved_input_path.exists():
                return _generate_edit_image_error(
                    call_id=call_id,
                    provider=provider,
                    operation=operation,
                    prompt=prompt,
                    raw_image_path=raw_image_path,
                    raw_output_path=raw_output_path,
                    reason="input image does not exist.",
                )
            if not resolved_input_path.is_file():
                return _generate_edit_image_error(
                    call_id=call_id,
                    provider=provider,
                    operation=operation,
                    prompt=prompt,
                    raw_image_path=raw_image_path,
                    raw_output_path=raw_output_path,
                    reason="image_path must point to a file.",
                )

            input_media_type = _detect_supported_image_media_type(resolved_input_path)
            if input_media_type is None:
                supported = ", ".join(sorted(_SUPPORTED_IMAGE_MEDIA_TYPES))
                return _generate_edit_image_error(
                    call_id=call_id,
                    provider=provider,
                    operation=operation,
                    prompt=prompt,
                    raw_image_path=raw_image_path,
                    raw_output_path=raw_output_path,
                    reason=(
                        "unsupported input image type. Supported types: "
                        f"{supported}."
                    ),
                )
            if input_media_type == "unreadable":
                return _generate_edit_image_error(
                    call_id=call_id,
                    provider=provider,
                    operation=operation,
                    prompt=prompt,
                    raw_image_path=raw_image_path,
                    raw_output_path=raw_output_path,
                    reason="input image could not be read.",
                )

        try:
            if provider == "openai":
                generated = await asyncio.to_thread(
                    self._run_openai_request,
                    prompt,
                    resolved_input_path,
                    quality,
                )
            else:
                generated = await asyncio.to_thread(
                    self._run_gemini_request,
                    prompt,
                    resolved_input_path,
                    input_media_type,
                    resolution,
                )
        except (GenerateEditImageConfigurationError, GenerateEditImageRequestError) as exc:
            return _generate_edit_image_error(
                call_id=call_id,
                provider=provider,
                operation=operation,
                prompt=prompt,
                raw_image_path=raw_image_path,
                raw_output_path=raw_output_path,
                reason=str(exc),
            )

        output_path = _finalize_output_path(
            requested_path=resolved_output_path,
            mime_type=generated.mime_type,
        )
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(generated.image_bytes)
        except OSError as exc:
            return _generate_edit_image_error(
                call_id=call_id,
                provider=provider,
                operation=operation,
                prompt=prompt,
                raw_image_path=raw_image_path,
                raw_output_path=raw_output_path,
                reason=f"failed to write output image: {exc}",
            )

        content_lines = [
            "Image generation completed",
            f"operation: {operation}",
            f"provider: {generated.provider}",
            f"model: {generated.model}",
            f"output_path: {output_path}",
            f"mime_type: {generated.mime_type}",
            f"file_size_bytes: {len(generated.image_bytes)}",
        ]
        if resolved_input_path is not None:
            content_lines.append(f"input_image_path: {resolved_input_path}")
        if generated.revised_prompt is not None:
            content_lines.append(f"revised_prompt: {generated.revised_prompt}")
        if generated.provider_text is not None:
            content_lines.append(f"provider_text: {generated.provider_text}")
        if provider == "openai":
            content_lines.append(f"quality: {quality}")
        else:
            content_lines.append(f"resolution: {resolution}")

        metadata: dict[str, Any] = {
            "operation": operation,
            "provider": generated.provider,
            "model": generated.model,
            "prompt": prompt,
            "image_path": str(resolved_input_path) if resolved_input_path is not None else None,
            "requested_output_path": raw_output_path,
            "output_path": str(output_path),
            "output_dir": str(output_path.parent),
            "mime_type": generated.mime_type,
            "file_size_bytes": len(generated.image_bytes),
        }
        if provider == "openai":
            metadata["quality"] = quality
        else:
            metadata["resolution"] = resolution
        if generated.revised_prompt is not None:
            metadata["revised_prompt"] = generated.revised_prompt
        if generated.provider_text is not None:
            metadata["provider_text"] = generated.provider_text
        if generated.response_id is not None:
            metadata["response_id"] = generated.response_id
        if generated.usage is not None:
            metadata["usage"] = generated.usage
        if input_media_type is not None:
            metadata["input_media_type"] = input_media_type

        return ToolExecutionResult(
            call_id=call_id,
            name="generate_edit_image",
            ok=True,
            content="\n".join(content_lines),
            metadata=metadata,
        )

    def _run_openai_request(
        self,
        prompt: str,
        image_path: Path | None,
        quality: str,
    ) -> GeneratedImagePayload:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise GenerateEditImageConfigurationError(
                "OPENAI_API_KEY is required for provider 'openai'."
            )

        client = OpenAI(
            api_key=api_key,
            timeout=180.0,
            max_retries=0,
        )
        try:
            if image_path is None:
                response = client.images.generate(
                    model=_OPENAI_IMAGE_MODEL,
                    prompt=prompt,
                    quality=quality,
                    output_format="png",
                )
            else:
                with image_path.open("rb") as image_handle:
                    response = client.images.edit(
                        model=_OPENAI_IMAGE_MODEL,
                        image=image_handle,
                        prompt=prompt,
                        quality=quality,
                        output_format="png",
                    )
        except Exception as exc:
            raise self._map_openai_error(exc) from exc
        finally:
            client.close()

        image_data = (response.data or [None])[0]
        if image_data is None or not image_data.b64_json:
            raise GenerateEditImageRequestError(
                "OpenAI did not return image bytes in b64_json format."
            )

        try:
            image_bytes = base64.b64decode(image_data.b64_json, validate=True)
        except (ValueError, TypeError) as exc:
            raise GenerateEditImageRequestError(
                "OpenAI returned invalid base64 image bytes."
            ) from exc

        return GeneratedImagePayload(
            provider="openai",
            model=_OPENAI_IMAGE_MODEL,
            image_bytes=image_bytes,
            mime_type="image/png",
            revised_prompt=_normalize_optional_string(image_data.revised_prompt),
            usage=_model_dump(response.usage),
        )

    def _run_gemini_request(
        self,
        prompt: str,
        image_path: Path | None,
        input_media_type: str | None,
        resolution: str,
    ) -> GeneratedImagePayload:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise GenerateEditImageConfigurationError(
                "GOOGLE_API_KEY is required for provider 'gemini'."
            )

        client = genai.Client(api_key=api_key)
        contents: list[str | genai_types.Part] = []
        if image_path is not None:
            if input_media_type is None:
                raise GenerateEditImageRequestError(
                    "Gemini image edits require a detected input image MIME type."
                )
            try:
                image_bytes = image_path.read_bytes()
            except OSError as exc:
                raise GenerateEditImageRequestError(
                    f"failed to read input image: {exc}"
                ) from exc
            contents.append(
                genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=input_media_type,
                )
            )
        contents.append(prompt)

        config = genai_types.GenerateContentConfig(
            response_modalities=[genai_types.Modality.IMAGE],
            image_config=genai_types.ImageConfig(
                image_size=resolution,
            ),
        )
        try:
            response = client.models.generate_content(
                model=_GEMINI_IMAGE_MODEL,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            raise self._map_gemini_error(exc) from exc

        provider_text, image_bytes, mime_type = _extract_gemini_response_payload(response)
        return GeneratedImagePayload(
            provider="gemini",
            model=str(getattr(response, "model_version", None) or _GEMINI_IMAGE_MODEL),
            image_bytes=image_bytes,
            mime_type=mime_type,
            provider_text=provider_text,
            response_id=_normalize_optional_string(getattr(response, "response_id", None)),
            usage=_model_dump(getattr(response, "usage_metadata", None)),
        )

    def _map_openai_error(self, error: Exception) -> Exception:
        if isinstance(error, (AuthenticationError, PermissionDeniedError)):
            return GenerateEditImageConfigurationError(str(error))
        if isinstance(error, RateLimitError):
            return GenerateEditImageRequestError(f"OpenAI rate limit: {error}")
        if isinstance(error, APITimeoutError):
            return GenerateEditImageRequestError(f"OpenAI request timed out: {error}")
        if isinstance(error, (APIConnectionError, InternalServerError)):
            return GenerateEditImageRequestError(f"OpenAI request failed: {error}")
        if isinstance(error, BadRequestError):
            return GenerateEditImageRequestError(f"OpenAI rejected the request: {error}")
        if isinstance(error, APIStatusError):
            return GenerateEditImageRequestError(
                f"OpenAI API status {error.status_code}: {error}"
            )
        if isinstance(error, OpenAIError):
            return GenerateEditImageRequestError(f"OpenAI error: {error}")
        return GenerateEditImageRequestError(f"OpenAI request failed: {error}")

    def _map_gemini_error(self, error: Exception) -> Exception:
        if isinstance(error, (httpx.ReadTimeout, httpx.ConnectTimeout, TimeoutError)):
            return GenerateEditImageRequestError(f"Gemini request timed out: {error}")
        if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
            return GenerateEditImageRequestError(f"Gemini request failed: {error}")
        if isinstance(error, genai_errors.ServerError):
            return GenerateEditImageRequestError(f"Gemini server error: {error}")
        if isinstance(error, genai_errors.ClientError):
            status_code = getattr(error, "code", None)
            if status_code in {401, 403}:
                return GenerateEditImageConfigurationError(str(error))
            return GenerateEditImageRequestError(f"Gemini request rejected: {error}")
        if isinstance(error, genai_errors.APIError):
            status_code = getattr(error, "code", None)
            if status_code in {401, 403}:
                return GenerateEditImageConfigurationError(str(error))
            return GenerateEditImageRequestError(f"Gemini API error: {error}")
        return GenerateEditImageRequestError(f"Gemini request failed: {error}")


def build_generate_edit_image_tool(settings: ToolSettings) -> RegisteredTool:
    """Build the generate_edit_image registry entry."""

    _ = settings
    return RegisteredTool(
        name="generate_edit_image",
        exposure="discoverable",
        definition=ToolDefinition(
            name="generate_edit_image",
            description=_build_generate_edit_image_tool_description(),
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Required image instruction.",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Optional workspace source image.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": (
                            "Required workspace output path. Parent directories are created "
                            "automatically, and a safe image extension is added if needed."
                        ),
                    },
                    "provider": {
                        "type": "string",
                        "enum": list(_SUPPORTED_PROVIDERS),
                        "description": "Optional provider: gemini or openai.",
                    },
                    "quality": {
                        "type": "string",
                        "enum": list(_OPENAI_QUALITY_VALUES),
                        "description": "Optional OpenAI-only quality override.",
                    },
                    "resolution": {
                        "type": "string",
                        "enum": list(_GEMINI_RESOLUTION_VALUES),
                        "description": "Optional Gemini-only resolution override.",
                    },
                },
                "required": ["prompt", "output_path"],
                "additionalProperties": False,
            },
        ),
        executor=GenerateEditImageToolExecutor(),
    )


def build_generate_edit_image_discoverable() -> DiscoverableTool:
    """Build the discoverable catalog entry for generate_edit_image."""

    return DiscoverableTool(
        name="generate_edit_image",
        aliases=(
            "generate image",
            "edit image",
            "create image",
        ),
        purpose=(
            "Generate a new image from a text prompt or edit an existing workspace image "
            "with a text prompt."
        ),
        detailed_description=_build_generate_edit_image_tool_description(),
        usage={
            "arguments": [
                {
                    "name": "prompt",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "image_path",
                    "type": "string",
                    "required": False,
                },
                {
                    "name": "output_path",
                    "type": "string",
                    "required": True,
                },
                {
                    "name": "provider",
                    "type": "string",
                    "required": False,
                    "enum": list(_SUPPORTED_PROVIDERS),
                },
                {
                    "name": "quality",
                    "type": "string",
                    "required": False,
                    "enum": list(_OPENAI_QUALITY_VALUES),
                },
                {
                    "name": "resolution",
                    "type": "string",
                    "required": False,
                    "enum": list(_GEMINI_RESOLUTION_VALUES),
                },
            ],
        },
        backing_tool_name="generate_edit_image",
    )


def _build_generate_edit_image_tool_description() -> str:
    return (
        "Generate from `prompt` or edit `image_path`, writing the result to workspace "
        "`output_path`. Default to `provider='gemini'`; use `openai` only when needed. "
        "Leave `quality` and `resolution` unset unless the user asks for a specific tradeoff."
    )


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_provider(value: Any) -> str:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return _DEFAULT_PROVIDER
    lowered = normalized.lower()
    if lowered in _SUPPORTED_PROVIDERS:
        return lowered
    return _DEFAULT_PROVIDER


def _normalize_openai_quality(value: Any) -> str | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered in _OPENAI_QUALITY_VALUES:
        return lowered
    return None


def _normalize_gemini_resolution(value: Any) -> str | None:
    normalized = _normalize_optional_string(value)
    if normalized is None:
        return None
    if normalized == "512":
        return normalized
    upper = normalized.upper()
    if upper in _GEMINI_RESOLUTION_VALUES:
        return upper
    return None


def _resolve_workspace_relative_path(raw_path: str, context: ToolExecutionContext) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = context.workspace_dir / candidate
    return candidate.resolve(strict=False)


def _detect_supported_image_media_type(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return "unreadable"

    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def _finalize_output_path(
    *,
    requested_path: Path,
    mime_type: str,
) -> Path:
    extension = _SUPPORTED_IMAGE_MEDIA_TYPES.get(mime_type, ".bin")
    if requested_path.suffix:
        return requested_path
    return requested_path.with_suffix(extension)


def _extract_gemini_response_payload(
    response: Any,
) -> tuple[str | None, bytes, str]:
    text_parts: list[str] = []
    candidate_summaries: list[str] = []
    for candidate in getattr(response, "candidates", ()) or ():
        candidate_details: list[str] = []
        finish_reason = _normalize_optional_string(
            getattr(candidate, "finish_reason", None)
        )
        if finish_reason is not None:
            candidate_details.append(f"candidate_finish_reason={finish_reason}")
        finish_message = _normalize_optional_string(
            getattr(candidate, "finish_message", None)
        )
        if finish_message is not None:
            candidate_details.append(
                f"candidate_finish_message={_truncate_for_error(finish_message)}"
            )
        content = getattr(candidate, "content", None)
        if content is None:
            if candidate_details:
                candidate_summaries.append(", ".join(candidate_details))
            continue
        part_types: list[str] = []
        for part in getattr(content, "parts", ()) or ():
            part_types.extend(_summarize_gemini_part_types(part))
            text = _normalize_optional_string(getattr(part, "text", None))
            if text is not None:
                text_parts.append(text)

            inline_data = getattr(part, "inline_data", None)
            if inline_data is None:
                continue
            data = getattr(inline_data, "data", None)
            mime_type = _normalize_optional_string(getattr(inline_data, "mime_type", None))
            if isinstance(data, bytes) and data and mime_type is not None:
                return ("\n".join(text_parts).strip() or None, data, mime_type)
        if part_types:
            candidate_details.append(
                "candidate_part_types=" + ",".join(dict.fromkeys(part_types))
            )
        if candidate_details:
            candidate_summaries.append(", ".join(candidate_details))

    diagnostics: list[str] = []
    candidate_count = len(getattr(response, "candidates", ()) or ())
    diagnostics.append(f"candidate_count={candidate_count}")
    prompt_feedback = _model_dump(getattr(response, "prompt_feedback", None))
    if prompt_feedback is not None:
        block_reason = _normalize_optional_string(prompt_feedback.get("block_reason"))
        if block_reason is not None:
            diagnostics.append(f"prompt_block_reason={block_reason}")
        block_reason_message = _normalize_optional_string(
            prompt_feedback.get("block_reason_message")
        )
        if block_reason_message is not None:
            diagnostics.append(
                "prompt_block_reason_message="
                + _truncate_for_error(block_reason_message)
            )
    if candidate_summaries:
        diagnostics.extend(candidate_summaries)
    response_text = _normalize_optional_string(getattr(response, "text", None))
    if response_text is not None:
        diagnostics.append(f"response_text={_truncate_for_error(response_text)}")

    reason = "Gemini did not return an inline image payload."
    if diagnostics:
        reason += " " + "; ".join(diagnostics)
    raise GenerateEditImageRequestError(reason)


def _summarize_gemini_part_types(part: Any) -> list[str]:
    part_types: list[str] = []
    text = _normalize_optional_string(getattr(part, "text", None))
    if text is not None:
        part_types.append("text")
    for field_name in (
        "inline_data",
        "file_data",
        "function_call",
        "function_response",
        "executable_code",
        "code_execution_result",
    ):
        if getattr(part, field_name, None) is not None:
            part_types.append(field_name)
    return part_types


def _truncate_for_error(value: str, *, limit: int = 160) -> str:
    collapsed = " ".join(value.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


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


def _generate_edit_image_error(
    *,
    call_id: str,
    provider: str,
    operation: str,
    prompt: str,
    raw_image_path: str | None,
    raw_output_path: str | None,
    reason: str,
) -> ToolExecutionResult:
    metadata: dict[str, Any] = {
        "provider": provider,
        "operation": operation,
        "prompt": prompt,
        "error": reason,
    }
    if raw_image_path is not None:
        metadata["image_path"] = raw_image_path
    if raw_output_path is not None:
        metadata["output_path"] = raw_output_path

    content_lines = [
        "Generate/edit image failed",
        f"operation: {operation}",
        f"provider: {provider}",
        f"reason: {reason}",
    ]
    if raw_image_path is not None:
        content_lines.append(f"image_path: {raw_image_path}")
    if raw_output_path is not None:
        content_lines.append(f"output_path: {raw_output_path}")

    return ToolExecutionResult(
        call_id=call_id,
        name="generate_edit_image",
        ok=False,
        content="\n".join(content_lines),
        metadata=metadata,
    )
