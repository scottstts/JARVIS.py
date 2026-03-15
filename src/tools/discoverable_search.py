"""Shared search helpers for discoverable tool entries."""

from __future__ import annotations

from collections.abc import Iterable
import re

from .types import DiscoverableTool

_SEARCH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def search_discoverable_entries(
    entries: Iterable[DiscoverableTool],
    query: str,
) -> tuple[DiscoverableTool, ...]:
    normalized_query = normalize_search_text(query)
    all_entries = tuple(entries)
    if not normalized_query:
        return tuple(sorted(all_entries, key=lambda entry: entry.name))

    query_tokens = tokenize_search_text(normalized_query)
    matches: list[tuple[int, DiscoverableTool]] = []
    for entry in all_entries:
        score = score_discoverable_match(
            entry=entry,
            normalized_query=normalized_query,
            query_tokens=query_tokens,
        )
        if score <= 0:
            continue
        matches.append((score, entry))

    matches.sort(key=lambda item: (-item[0], item[1].name))
    return tuple(entry for _, entry in matches)


def normalize_search_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def tokenize_search_text(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(_SEARCH_TOKEN_PATTERN.findall(value)))


def score_discoverable_match(
    *,
    entry: DiscoverableTool,
    normalized_query: str,
    query_tokens: tuple[str, ...],
) -> int:
    normalized_name = normalize_search_text(entry.name)
    normalized_aliases = tuple(
        normalize_search_text(alias)
        for alias in entry.aliases
        if alias.strip()
    )
    normalized_purpose = normalize_search_text(entry.purpose)
    normalized_description = normalize_search_text(entry.detailed_description or "")
    combined_text = " ".join(
        part
        for part in (
            normalized_name,
            *normalized_aliases,
            normalized_purpose,
            normalized_description,
        )
        if part
    )

    if not combined_text:
        return 0

    score = 0
    if normalized_query == normalized_name:
        score += 200
    if normalized_query in normalized_name:
        score += 120
        if normalized_name.startswith(normalized_query):
            score += 20

    if any(normalized_query == alias for alias in normalized_aliases):
        score += 180
    if any(normalized_query in alias for alias in normalized_aliases):
        score += 110

    if normalized_query in normalized_purpose:
        score += 80
    if normalized_query in normalized_description:
        score += 40

    if query_tokens:
        token_hits = sum(1 for token in query_tokens if token in combined_text)
        if token_hits == 0:
            return 0
        score += token_hits * 15
        if token_hits == len(query_tokens):
            score += 25
    elif normalized_query not in combined_text:
        return 0

    return score
