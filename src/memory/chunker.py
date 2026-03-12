"""Deterministic Markdown chunking for search indexing."""

from __future__ import annotations

import math
import re
from uuid import uuid5, NAMESPACE_URL

from .types import DocumentChunk, MemoryDocument

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_BULLET_PATTERN = re.compile(r"^(?:[-*]|\d+\.)\s+")


def chunk_document(document: MemoryDocument) -> tuple[DocumentChunk, ...]:
    if document.kind == "daily":
        return _chunk_daily_document(document)
    return _chunk_structured_document(document)


def _chunk_structured_document(document: MemoryDocument) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    ordinal = 0
    for heading, content in document.sections.items():
        section_text = content.strip()
        if heading == "Summary" and not section_text and document.summary:
            section_text = document.summary.strip()
        if not section_text:
            continue
        pieces = _split_text(section_text, target_min=600, target_max=1200, hard_max=1600)
        for piece_index, piece in enumerate(pieces, start=1):
            section_path = heading if len(pieces) == 1 else f"{heading}/{piece_index}"
            chunks.append(
                _build_chunk(
                    document=document,
                    ordinal=ordinal,
                    section_path=section_path,
                    text=f"{document.title}\n{heading}\n{piece}".strip(),
                )
            )
            ordinal += 1
    if not chunks and document.title.strip():
        chunks.append(
            _build_chunk(
                document=document,
                ordinal=ordinal,
                section_path="Title",
                text=document.title.strip(),
            )
        )
    return tuple(chunks)


def _chunk_daily_document(document: MemoryDocument) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    ordinal = 0
    for heading, content in document.sections.items():
        groups = _daily_groups(content)
        if not groups:
            continue
        current: list[str] = []
        for group in groups:
            proposed = "\n\n".join(current + [group]) if current else group
            if len(proposed) > 2000 and current:
                section_path = heading if len(chunks) == 0 else f"{heading}/{ordinal + 1}"
                chunks.append(
                    _build_chunk(
                        document=document,
                        ordinal=ordinal,
                        section_path=section_path,
                        text=f"{document.title}\n{heading}\n" + "\n\n".join(current),
                    )
                )
                ordinal += 1
                current = [group]
                continue
            current.append(group)
            if len(proposed) >= 800:
                section_path = heading if len(current) == 1 else f"{heading}/{ordinal + 1}"
                chunks.append(
                    _build_chunk(
                        document=document,
                        ordinal=ordinal,
                        section_path=section_path,
                        text=f"{document.title}\n{heading}\n" + "\n\n".join(current),
                    )
                )
                ordinal += 1
                current = []
        if current:
            section_path = heading if len(current) == 1 else f"{heading}/{ordinal + 1}"
            chunks.append(
                _build_chunk(
                    document=document,
                    ordinal=ordinal,
                    section_path=section_path,
                    text=f"{document.title}\n{heading}\n" + "\n\n".join(current),
                )
            )
            ordinal += 1
    return tuple(chunks)


def _split_text(text: str, *, target_min: int, target_max: int, hard_max: int) -> list[str]:
    if len(text) <= target_max:
        return [text.strip()]
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        return [text.strip()]
    groups = _pack_segments(paragraphs, target_min=target_min, target_max=target_max, hard_max=hard_max)
    flattened: list[str] = []
    for group in groups:
        if len(group) <= hard_max:
            flattened.append(group)
            continue
        sentences = [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(group) if sentence.strip()]
        flattened.extend(
            _pack_segments(sentences, target_min=target_min, target_max=target_max, hard_max=hard_max)
        )
    return flattened


def _pack_segments(
    segments: list[str],
    *,
    target_min: int,
    target_max: int,
    hard_max: int,
) -> list[str]:
    groups: list[str] = []
    current: list[str] = []
    current_length = 0
    for segment in segments:
        joiner = "\n\n" if current else ""
        projected_length = current_length + len(joiner) + len(segment)
        if current and projected_length > hard_max:
            groups.append("\n\n".join(current).strip())
            current = [segment]
            current_length = len(segment)
            continue
        current.append(segment)
        current_length = projected_length
        if current_length >= target_min and current_length <= target_max:
            groups.append("\n\n".join(current).strip())
            current = []
            current_length = 0
    if current:
        if groups and len(groups[-1]) < target_min and len(groups[-1]) + 2 + current_length <= hard_max:
            groups[-1] = groups[-1].rstrip() + "\n\n" + "\n\n".join(current).strip()
        else:
            groups.append("\n\n".join(current).strip())
    return groups


def _daily_groups(content: str) -> list[str]:
    lines = [line.rstrip() for line in content.splitlines()]
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                current.append("")
            continue
        if _BULLET_PATTERN.match(line) and current and not _BULLET_PATTERN.match(current[0]):
            groups.append(current)
            current = [line]
            continue
        if _BULLET_PATTERN.match(line) and current and current[-1] == "":
            groups.append(current[:-1])
            current = [line]
            continue
        current.append(line)
    if current:
        groups.append(current)
    return ["\n".join(line for line in group if line is not None).strip() for group in groups if any(line.strip() for line in group)]


def _build_chunk(
    *,
    document: MemoryDocument,
    ordinal: int,
    section_path: str,
    text: str,
) -> DocumentChunk:
    chunk_id = uuid5(NAMESPACE_URL, f"{document.document_id}:{ordinal}:{section_path}:{text}").hex
    token_estimate = max(1, math.ceil(len(text) / 4))
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document.document_id,
        path=document.path,
        kind=document.kind,
        ordinal=ordinal,
        section_path=section_path,
        text=text.strip(),
        token_estimate=token_estimate,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )
