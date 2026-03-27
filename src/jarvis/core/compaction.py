"""Compaction orchestration for long-running session contexts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from jarvis.llm import LLMMessage, LLMRequest, LLMService
from jarvis.storage import ConversationRecord

from .config import ContextPolicySettings

_COMPACTION_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "COMPACTION.md"
_COMPACTION_SYSTEM_PROMPT = _COMPACTION_PROMPT_PATH.read_text(encoding="utf-8").strip()


@dataclass(slots=True, frozen=True)
class CompactionOutcome:
    summary_text: str
    model: str
    provider: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    response_id: str | None


class ContextCompactor:
    """Builds compacted carry-forward summaries from prior transcript records."""

    def __init__(
        self,
        *,
        llm_service: LLMService,
        context_policy: ContextPolicySettings,
        provider: str | None = None,
    ) -> None:
        self._llm_service = llm_service
        self._context_policy = context_policy
        self._provider = provider

    async def compact(
        self,
        records: Sequence[ConversationRecord],
        *,
        user_instruction: str | None = None,
    ) -> CompactionOutcome:
        transcript = _serialize_transcript(records)
        instruction = user_instruction.strip() if user_instruction else ""

        user_prompt = (
            "Compact the following transcript.\n\n"
            "Additional user instruction for this compaction:\n"
            f"{instruction or 'None'}\n\n"
            "Transcript:\n"
            f"{transcript}"
        )
        request = LLMRequest(
            messages=(
                LLMMessage.text("system", _COMPACTION_SYSTEM_PROMPT),
                LLMMessage.text("user", user_prompt),
            ),
            provider=self._provider,
            max_output_tokens=self._context_policy.compact_reserve_output_tokens,
        )
        response = await self._llm_service.generate(request)

        summary = response.text.strip()
        if not summary:
            summary = (
                "Current objective: continue prior session.\n"
                "Stable context: compaction model returned empty output.\n"
                "Decisions made: none extracted.\n"
                "Open items: continue from latest user request.\n"
                "Important artifacts: session transcript retained in the archive."
            )

        usage = response.usage
        return CompactionOutcome(
            summary_text=summary,
            model=response.model,
            provider=response.provider,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
            response_id=response.response_id,
        )


def _serialize_transcript(records: Sequence[ConversationRecord]) -> str:
    lines: list[str] = []
    for index, record in enumerate(records, start=1):
        content = record.content.strip()
        if not content:
            continue

        if record.role == "tool":
            content = _shrink_high_entropy_text(content)

        lines.append(f"[{index}] {record.role.upper()}: {content}")

    if not lines:
        return "(no transcript content)"
    return "\n\n".join(lines)


def _shrink_high_entropy_text(content: str) -> str:
    if len(content) <= 1800:
        return content
    head = content[:900]
    tail = content[-900:]
    return f"{head}\n...[tool output truncated for compaction]...\n{tail}"
