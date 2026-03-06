"""Provider-agnostic request/response/event types for the LLM layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Mapping, Sequence, TypeAlias

LLMRole: TypeAlias = Literal["system", "developer", "user", "assistant", "tool"]
ImageDetail: TypeAlias = Literal["low", "high", "auto"]
FinishReason: TypeAlias = Literal[
    "stop",
    "tool_calls",
    "length",
    "content_filter",
    "error",
    "unknown",
]


@dataclass(slots=True, frozen=True)
class TextPart:
    text: str
    type: Literal["text"] = "text"


@dataclass(slots=True, frozen=True)
class ImagePart:
    image_url: str | None = None
    file_id: str | None = None
    detail: ImageDetail = "auto"
    type: Literal["image"] = "image"

    def __post_init__(self) -> None:
        has_url = self.image_url is not None
        has_file = self.file_id is not None
        if has_url == has_file:
            raise ValueError("ImagePart requires exactly one of image_url or file_id.")

    @classmethod
    def from_base64(
        cls,
        *,
        media_type: str,
        data_base64: str,
        detail: ImageDetail = "auto",
    ) -> "ImagePart":
        return cls(
            image_url=f"data:{media_type};base64,{data_base64}",
            detail=detail,
        )


@dataclass(slots=True, frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ToolResultPart:
    call_id: str
    name: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


MessagePart: TypeAlias = TextPart | ImagePart | ToolCall | ToolResultPart


@dataclass(slots=True, frozen=True)
class LLMMessage:
    role: LLMRole
    parts: tuple[MessagePart, ...]

    def __post_init__(self) -> None:
        if not self.parts:
            raise ValueError("LLMMessage.parts cannot be empty.")

    @classmethod
    def text(cls, role: LLMRole, text: str) -> "LLMMessage":
        return cls(role=role, parts=(TextPart(text=text),))


class ToolChoiceMode(StrEnum):
    AUTO = "auto"
    REQUIRED = "required"
    NONE = "none"
    TOOL = "tool"


@dataclass(slots=True, frozen=True)
class ToolChoice:
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    tool_name: str | None = None

    def __post_init__(self) -> None:
        if self.mode == ToolChoiceMode.TOOL and not self.tool_name:
            raise ValueError("ToolChoice in TOOL mode requires tool_name.")
        if self.mode != ToolChoiceMode.TOOL and self.tool_name is not None:
            raise ValueError("tool_name can only be set when mode is TOOL.")

    @classmethod
    def auto(cls) -> "ToolChoice":
        return cls(mode=ToolChoiceMode.AUTO)

    @classmethod
    def required(cls) -> "ToolChoice":
        return cls(mode=ToolChoiceMode.REQUIRED)

    @classmethod
    def none(cls) -> "ToolChoice":
        return cls(mode=ToolChoiceMode.NONE)

    @classmethod
    def tool(cls, name: str) -> "ToolChoice":
        return cls(mode=ToolChoiceMode.TOOL, tool_name=name)


@dataclass(slots=True, frozen=True)
class LLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    name: str
    input_schema: Mapping[str, Any]
    description: str | None = None
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolDefinition.name cannot be empty.")


@dataclass(slots=True, frozen=True)
class LLMResponse:
    provider: str
    model: str
    text: str
    tool_calls: list[ToolCall]
    finish_reason: FinishReason
    usage: LLMUsage | None
    response_id: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class TextDeltaEvent:
    delta: str
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True, frozen=True)
class ToolCallDeltaEvent:
    call_id: str
    arguments_delta: str
    tool_name: str | None = None
    type: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass(slots=True, frozen=True)
class UsageDeltaEvent:
    usage: LLMUsage
    type: Literal["usage_delta"] = "usage_delta"


@dataclass(slots=True, frozen=True)
class DoneEvent:
    response: LLMResponse
    type: Literal["done"] = "done"


LLMStreamEvent: TypeAlias = (
    TextDeltaEvent
    | ToolCallDeltaEvent
    | UsageDeltaEvent
    | DoneEvent
)


@dataclass(slots=True, frozen=True)
class LLMRequest:
    messages: Sequence[LLMMessage]
    provider: str | None = None
    model: str | None = None
    instructions: str | None = None
    tools: Sequence[ToolDefinition] = field(default_factory=tuple)
    tool_choice: ToolChoice = field(default_factory=ToolChoice.auto)
    temperature: float | None = None
    max_output_tokens: int | None = None
    parallel_tool_calls: bool = True
    metadata: Mapping[str, str] | None = None
    safety_identifier: str | None = None
    prompt_cache_key: str | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("LLMRequest.messages cannot be empty.")


@dataclass(slots=True, frozen=True)
class EmbeddingRequest:
    inputs: str | Sequence[str]
    provider: str | None = None
    model: str | None = None
    dimensions: int | None = None
    user: str | None = None
    timeout_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class EmbeddingResponse:
    provider: str
    model: str
    embeddings: list[list[float]]
    usage: LLMUsage | None = None
