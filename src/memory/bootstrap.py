"""Deterministic runtime bootstrap rendering for active memory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from .types import MemoryDocument


def render_core_bootstrap(
    documents: tuple[MemoryDocument, ...],
    *,
    token_budget: int,
    reference_time: datetime | None = None,
) -> str:
    return _render_bootstrap(
        documents=documents,
        token_budget=token_budget,
        reference_time=reference_time or datetime.now(timezone.utc),
        include_current_state=False,
        include_open_loops=False,
        include_relations=True,
    )


def render_ongoing_bootstrap(
    documents: tuple[MemoryDocument, ...],
    *,
    token_budget: int,
    reference_time: datetime | None = None,
) -> str:
    return _render_bootstrap(
        documents=documents,
        token_budget=token_budget,
        reference_time=reference_time or datetime.now(timezone.utc),
        include_current_state=True,
        include_open_loops=True,
        include_relations=False,
    )


def _render_bootstrap(
    *,
    documents: tuple[MemoryDocument, ...],
    token_budget: int,
    reference_time: datetime,
    include_current_state: bool,
    include_open_loops: bool,
    include_relations: bool,
) -> str:
    blocks: list[str] = []
    tokens_used = 0
    for document in documents:
        block_lines = _document_block_lines(
            document=document,
            reference_time=reference_time,
            include_current_state=include_current_state,
            include_open_loops=include_open_loops,
            include_relations=include_relations,
        )
        accepted_lines: list[str] = []
        for line in block_lines:
            next_block = "\n".join(accepted_lines + [line]).strip()
            prospective = "\n\n".join(blocks + [next_block]).strip()
            if _estimate_tokens(prospective) > token_budget:
                break
            accepted_lines.append(line)
        if not accepted_lines:
            break
        blocks.append("\n".join(accepted_lines).strip())
        tokens_used = _estimate_tokens("\n\n".join(blocks).strip())
        if tokens_used >= token_budget:
            break
    return "\n\n".join(blocks).strip()


def _document_block_lines(
    *,
    document: MemoryDocument,
    reference_time: datetime,
    include_current_state: bool,
    include_open_loops: bool,
    include_relations: bool,
) -> list[str]:
    summary = document.summary or _first_non_empty(document.sections.get("Summary", ""))
    lines = [f"- {document.title}"]
    freshness = _freshness_hint(document.updated_at, reference_time=reference_time)
    if freshness is not None:
        lines.append(f"  freshness: {freshness}")
    if summary:
        lines.append(f"  summary: {summary}")

    summary_fingerprint = (summary or "").lower()
    for fact in document.facts:
        if fact.status != "current":
            continue
        if fact.text.lower() in summary_fingerprint:
            continue
        lines.append(f"  fact: {fact.text}")

    if include_relations:
        seen_relations = {line.lower() for line in lines}
        for relation in document.relations:
            if relation.status != "current":
                continue
            text = relation.textualization
            if text.lower() in summary_fingerprint or text.lower() in seen_relations:
                continue
            lines.append(f"  relation: {text}")
            seen_relations.add(text.lower())

    if include_current_state:
        for line in _section_lines(document.sections.get("Current State", "")):
            lines.append(f"  current_state: {line}")
    if include_open_loops:
        for line in _section_lines(document.sections.get("Open Loops", "")):
            lines.append(f"  open_loop: {line}")
    return lines


def _section_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _first_non_empty(value: str) -> str:
    for paragraph in value.split("\n\n"):
        normalized = paragraph.strip()
        if normalized:
            return normalized
    return ""


def _freshness_hint(updated_at: str, *, reference_time: datetime) -> str | None:
    parsed = _parse_iso(updated_at)
    if parsed is None:
        return None
    delta = max(reference_time - parsed, timedelta(0))
    total_hours = int(delta.total_seconds() // 3600)
    if total_hours < 24:
        return f"updated {total_hours}h ago"
    return f"updated {delta.days}d ago"


def _estimate_tokens(value: str) -> int:
    return max(1, len(value) // 4) if value else 0


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def checksum_bundle_for_documents(documents: tuple[MemoryDocument, ...]) -> str:
    return "|".join(f"{Path(document.path).name}:{document.checksum}" for document in documents)
