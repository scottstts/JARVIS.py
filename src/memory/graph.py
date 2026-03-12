"""Entity resolution and graph expansion for memory search."""

from __future__ import annotations

import json
from typing import Any

from rapidfuzz import fuzz

from .types import SearchCandidate


def expand_graph_candidates(
    *,
    query: str,
    entity_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    expand: int,
) -> tuple[SearchCandidate, ...]:
    if expand <= 0:
        return ()
    matched_entities = _resolve_entities(query=query, entity_rows=entity_rows)
    if not matched_entities:
        return ()

    frontier = {entity["canonical_name"] for entity in matched_entities}
    seen_entities = set(frontier)
    candidates: dict[tuple[str, str], SearchCandidate] = {}
    for depth in range(1, expand + 1):
        next_frontier: set[str] = set()
        for relation in relation_rows:
            subject = str(relation["subject"])
            obj = str(relation["object"])
            if subject not in frontier and obj not in frontier:
                continue
            next_frontier.update({subject, obj})
            document_id = str(relation["document_id"])
            section_path = "relations"
            key = (document_id, section_path)
            snippet = f"{subject} {relation['predicate']} {obj}"
            score = 1.0 if depth == 1 else 0.7
            candidate = SearchCandidate(
                document_id=document_id,
                path=_as_path(relation["path"]),
                kind=str(relation["kind"]),  # type: ignore[arg-type]
                section_path=section_path,
                snippet=snippet,
                source_ref_ids=tuple(json.loads(str(relation["source_ref_ids_json"]))),
                graph_score=score,
                match_reasons=("graph_entity_match",),
            )
            existing = candidates.get(key)
            if existing is None or candidate.graph_score > existing.graph_score:
                candidates[key] = candidate
        frontier = {name for name in next_frontier if name not in seen_entities}
        seen_entities.update(frontier)
        if not frontier:
            break
    return tuple(candidates.values())


def _resolve_entities(
    *,
    query: str,
    entity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_query = _normalize(query)
    query_tokens = tuple(token for token in normalized_query.split() if token)
    matches: list[tuple[float, dict[str, Any]]] = []
    for row in entity_rows:
        canonical_name = str(row["canonical_name"])
        aliases = tuple(json.loads(str(row["aliases_json"])))
        best_score = 0.0
        if _normalize(canonical_name) == normalized_query:
            best_score = 1.0
        elif normalized_query in _normalize(canonical_name):
            best_score = 0.96
        else:
            for alias in aliases:
                normalized_alias = _normalize(alias)
                if normalized_alias == normalized_query:
                    best_score = max(best_score, 0.99)
                elif normalized_query and normalized_query in normalized_alias:
                    best_score = max(best_score, 0.95)
                else:
                    fuzzy_score = fuzz.ratio(normalized_query, normalized_alias) / 100.0
                    if fuzzy_score >= 0.92:
                        best_score = max(best_score, fuzzy_score)
        if best_score == 0.0 and query_tokens:
            normalized_name_tokens = set(_normalize(canonical_name).split())
            if set(query_tokens).issubset(normalized_name_tokens):
                best_score = 0.93
        if best_score > 0.0:
            matches.append((best_score, row))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [row for _score, row in matches[:5]]


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _as_path(value: Any):
    from pathlib import Path

    return Path(str(value))

