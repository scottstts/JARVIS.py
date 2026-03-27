"""YAML frontmatter and Markdown section parsing for memory documents."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
import re

import yaml

from .types import ParsedMarkdownDocument

_FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


def parse_markdown_document(path: Path) -> ParsedMarkdownDocument:
    raw_text = path.read_text(encoding="utf-8")
    checksum = checksum_text(raw_text)
    frontmatter, body = _split_frontmatter(raw_text)
    title, sections = _parse_body_sections(body)
    return ParsedMarkdownDocument(
        path=path,
        raw_text=raw_text,
        frontmatter=frontmatter,
        title=title,
        sections=sections,
        checksum=checksum,
    )


def checksum_file(path: Path) -> str:
    return checksum_text(path.read_text(encoding="utf-8"))


def checksum_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_frontmatter(raw_text: str) -> tuple[dict[str, object], str]:
    match = _FRONTMATTER_PATTERN.match(raw_text)
    if match is None:
        raise ValueError("Memory document is missing required YAML frontmatter.")
    frontmatter_text = match.group(1)
    body = raw_text[match.end():]
    payload = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(payload, dict):
        raise ValueError("Memory frontmatter must parse to an object.")
    return dict(payload), body


def _parse_body_sections(body: str) -> tuple[str, OrderedDict[str, str]]:
    stripped = body.strip("\n")
    if not stripped:
        raise ValueError("Memory document body is empty.")

    lines = stripped.splitlines()
    title_index = next((index for index, line in enumerate(lines) if line.startswith("# ")), None)
    if title_index is None:
        raise ValueError("Memory document body must start with a '# <Title>' heading.")
    title = lines[title_index][2:].strip()
    if not title:
        raise ValueError("Top-level memory title cannot be blank.")

    sections: OrderedDict[str, str] = OrderedDict()
    current_heading: str | None = None
    buffer: list[str] = []

    def flush_current() -> None:
        if current_heading is None:
            return
        sections[current_heading] = "\n".join(buffer).strip()

    for line in lines[title_index + 1:]:
        if line.startswith("## "):
            flush_current()
            current_heading = line[3:].strip()
            if not current_heading:
                raise ValueError("Section headings cannot be blank.")
            buffer = []
            continue
        if current_heading is None:
            if line.strip():
                raise ValueError("Memory document body must use explicit '##' sections.")
            continue
        buffer.append(line)

    flush_current()
    return title, sections

