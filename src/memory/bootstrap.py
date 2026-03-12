"""Deterministic runtime bootstrap rendering for active memory."""

from __future__ import annotations

from pathlib import Path

from .types import MemoryDocument


def render_core_bootstrap(
    documents: tuple[MemoryDocument, ...],
    *,
    token_budget: int,
) -> str:
    return _render_bootstrap(documents=documents, token_budget=token_budget, include_open_loops=False)


def render_ongoing_bootstrap(
    documents: tuple[MemoryDocument, ...],
    *,
    token_budget: int,
) -> str:
    return _render_bootstrap(documents=documents, token_budget=token_budget, include_open_loops=True)


def _render_bootstrap(
    *,
    documents: tuple[MemoryDocument, ...],
    token_budget: int,
    include_open_loops: bool,
) -> str:
    lines: list[str] = []
    tokens_used = 0
    for document in documents:
        block_lines = [f"- {document.title}"]
        if document.summary:
            block_lines.append(f"  summary: {document.summary}")
        if document.kind == "core":
            for fact in document.facts:
                if fact.status != "current":
                    continue
                block_lines.append(f"  fact: {fact.text}")
        if include_open_loops and "Open Loops" in document.sections:
            for line in document.sections["Open Loops"].splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                block_lines.append(f"  open_loop: {stripped}")
        block = "\n".join(block_lines)
        block_tokens = max(1, len(block) // 4)
        if tokens_used + block_tokens > token_budget:
            break
        lines.append(block)
        tokens_used += block_tokens
    return "\n\n".join(lines).strip()


def checksum_bundle_for_documents(documents: tuple[MemoryDocument, ...]) -> str:
    return "|".join(f"{Path(document.path).name}:{document.checksum}" for document in documents)

