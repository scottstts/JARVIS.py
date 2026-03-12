"""Post-turn reflection planning for memory updates."""

from __future__ import annotations

import json
import re
from typing import Any

from llm import LLMMessage, LLMRequest, LLMService
from storage import ConversationRecord

from .config import MemorySettings
from .types import ReflectionAction, ReflectionPlan

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class MemoryReflectionPlanner:
    """Asks the maintenance model for structured post-turn memory actions."""

    def __init__(
        self,
        *,
        settings: MemorySettings,
        llm_service: LLMService | None,
    ) -> None:
        self._settings = settings
        self._llm_service = llm_service

    async def plan_turn(
        self,
        *,
        route_id: str | None,
        session_id: str,
        records: tuple[ConversationRecord, ...],
        active_core_titles: tuple[str, ...],
        active_ongoing_titles: tuple[str, ...],
    ) -> ReflectionPlan:
        if self._llm_service is None:
            return ReflectionPlan(actions=(), raw_text="")
        transcript = _render_records(records)
        if not transcript.strip():
            return ReflectionPlan(actions=(), raw_text="")
        prompt = (
            "You are Jarvis memory reflection. Produce compact JSON only.\n"
            "Decide whether the completed turn implies memory actions.\n"
            "Allowed actions: append_daily, create_ongoing, update_ongoing, close_ongoing, "
            "create_core, update_core, add_relation, supersede_relation, ignore.\n"
            "Use core promotions only when explicitly asked to remember or when obviously durable.\n"
            "Prefer append_daily for any notable event.\n"
            "For create/update actions, payload may include title, summary, facts, relations, "
            "body_sections, priority, pinned, locked, review_after, expires_at, source_refs.\n"
            "For append_daily, payload may include body_sections with section names and markdown bullets.\n"
            "For close_ongoing, payload should include document_id or title and close_reason.\n"
            "Respond as: {\"actions\": [{\"action\": ..., \"confidence\": ..., \"payload\": {...}, \"rationale\": ...}]}\n\n"
            f"route_id: {route_id}\n"
            f"session_id: {session_id}\n"
            f"active_core_titles: {json.dumps(list(active_core_titles), ensure_ascii=True)}\n"
            f"active_ongoing_titles: {json.dumps(list(active_ongoing_titles), ensure_ascii=True)}\n\n"
            "completed_turn_transcript:\n"
            f"{transcript}"
        )
        response = await self._llm_service.generate(
            LLMRequest(
                provider=self._settings.maintenance_provider,
                model=self._settings.maintenance_model,
                max_output_tokens=self._settings.maintenance_max_output_tokens,
                messages=(
                    LLMMessage.text("developer", prompt),
                ),
            )
        )
        return ReflectionPlan(actions=_parse_actions(response.text), raw_text=response.text)


def _render_records(records: tuple[ConversationRecord, ...]) -> str:
    lines: list[str] = []
    for record in records:
        if record.role == "tool":
            tool_name = str(record.metadata.get("tool_name", "tool")).strip() or "tool"
            lines.append(f"[tool:{tool_name}] {record.content.strip()}")
            continue
        content = record.content.strip()
        if not content:
            continue
        lines.append(f"[{record.role}] {content}")
    return "\n".join(lines).strip()


def _parse_actions(raw_text: str) -> tuple[ReflectionAction, ...]:
    if not raw_text.strip():
        return ()
    payload = _extract_json_object(raw_text)
    actions = payload.get("actions")
    if not isinstance(actions, list):
        return ()
    parsed: list[ReflectionAction] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip()
        confidence = str(item.get("confidence", "")).strip().lower()
        if action not in {
            "append_daily",
            "create_ongoing",
            "update_ongoing",
            "close_ongoing",
            "create_core",
            "update_core",
            "add_relation",
            "supersede_relation",
            "ignore",
        }:
            continue
        if confidence not in {"low", "medium", "high"}:
            continue
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}
        rationale = item.get("rationale")
        parsed.append(
            ReflectionAction(
                action=action,  # type: ignore[arg-type]
                confidence=confidence,  # type: ignore[arg-type]
                payload=dict(payload),
                rationale=str(rationale).strip() if isinstance(rationale, str) and rationale.strip() else None,
            )
        )
    return tuple(parsed)


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_PATTERN.search(raw_text)
    if match is None:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}

