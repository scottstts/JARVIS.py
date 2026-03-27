"""Entity resolution and graph expansion for memory search."""

from __future__ import annotations

import json
from typing import Any

from rapidfuzz import fuzz

from .types import SearchCandidate


def expand_graph_candidates(
    *,
    query: str,
    normalized_query: str,
    entity_terms: tuple[str, ...],
    entity_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    expand: int,
    limit: int,
) -> tuple[SearchCandidate, ...]:
    if expand <= 0 or limit <= 0:
        return ()
    matched_entities = _resolve_entities(
        query=query,
        normalized_query=normalized_query,
        entity_terms=entity_terms,
        entity_rows=entity_rows,
    )
    if not matched_entities:
        return ()

    frontier = {item["canonical_key"]: float(item["match_score"]) for item in matched_entities}
    seen_entities = set(frontier)
    candidates: dict[tuple[str, str], SearchCandidate] = {}
    for depth in range(1, expand + 1):
        next_frontier: dict[str, float] = {}
        for relation in relation_rows:
            subject = str(relation["subject"])
            obj = str(relation["object"])
            subject_key = _normalize(subject)
            object_key = _normalize(obj)
            subject_match = frontier.get(subject_key, 0.0)
            object_match = frontier.get(object_key, 0.0)
            if subject_match <= 0.0 and object_match <= 0.0:
                continue

            entity_score = max(subject_match, object_match)
            if subject_match > 0.0 and object_match > 0.0:
                entity_score = min(1.0, entity_score * 1.05)
            score = min(
                1.0,
                entity_score
                * _depth_multiplier(depth)
                * _relation_status_multiplier(str(relation["status"]))
                * _confidence_multiplier(str(relation["confidence"])),
            )
            document_id = str(relation["document_id"])
            section_path = "relations"
            key = (document_id, section_path)
            snippet = f"{subject} {relation['predicate']} {obj}"
            reason = (
                "graph_entity_exact"
                if entity_score >= 0.98
                else "graph_entity_alias"
                if entity_score >= 0.95
                else "graph_entity_match"
            )
            candidate = SearchCandidate(
                document_id=document_id,
                title=str(relation["title"]),
                path=_as_path(relation["path"]),
                kind=str(relation["kind"]),  # type: ignore[arg-type]
                chunk_id=f"relation:{relation['relation_id']}",
                section_path=section_path,
                snippet=snippet,
                source_ref_ids=tuple(json.loads(str(relation["source_ref_ids_json"]))),
                updated_at=str(relation["updated_at"]),
                status=str(relation["document_status"]),
                pinned=bool(relation["pinned"]),
                priority=int(relation["priority"]) if relation["priority"] is not None else None,
                review_after=_optional_str(relation.get("review_after")),
                expires_at=_optional_str(relation.get("expires_at")),
                archived_at=_optional_str(relation.get("archived_at")),
                truth_status=str(relation["status"]),
                support_count=int(relation["support_count"] or 0),
                contradiction_count=int(relation["contradiction_count"] or 0),
                last_confirmed_at=_optional_str(relation.get("last_confirmed_at")),
                last_contradicted_at=_optional_str(relation.get("last_contradicted_at")),
                graph_score=score,
                match_reasons=(reason, f"graph_relation_{relation['status']}"),
            )
            existing = candidates.get(key)
            if existing is None or candidate.graph_score > existing.graph_score:
                candidates[key] = candidate

            next_frontier[subject_key] = max(next_frontier.get(subject_key, 0.0), entity_score)
            next_frontier[object_key] = max(next_frontier.get(object_key, 0.0), entity_score)

        frontier = {
            entity_key: score
            for entity_key, score in next_frontier.items()
            if entity_key not in seen_entities
        }
        seen_entities.update(frontier)
        if not frontier:
            break

    ranked = sorted(
        candidates.values(),
        key=lambda item: (item.graph_score, item.pinned, item.priority or 0, item.updated_at),
        reverse=True,
    )
    return tuple(ranked[:limit])


def _resolve_entities(
    *,
    query: str,
    normalized_query: str,
    entity_terms: tuple[str, ...],
    entity_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del query
    terms = [term for term in entity_terms if term]
    if not terms and normalized_query:
        terms = [normalized_query]

    matches: dict[str, dict[str, Any]] = {}
    for row in entity_rows:
        canonical_name = str(row["canonical_name"])
        canonical_key = _normalize(canonical_name)
        aliases = tuple(json.loads(str(row["aliases_json"])))
        alias_keys = tuple(_normalize(alias) for alias in aliases if alias.strip())
        best_score = 0.0
        for term in terms:
            if term == canonical_key:
                best_score = max(best_score, 1.0)
                continue
            if term in alias_keys:
                best_score = max(best_score, 0.99)
                continue
            token_score = _token_match_score(term, canonical_key)
            if token_score > 0.0:
                best_score = max(best_score, token_score)
            for alias_key in alias_keys:
                token_score = _token_match_score(term, alias_key)
                if token_score > 0.0:
                    best_score = max(best_score, min(0.97, token_score + 0.01))
                if len(term) >= 4 and len(alias_key) >= 4:
                    fuzzy_score = fuzz.ratio(term, alias_key) / 100.0
                    if fuzzy_score >= 0.92:
                        best_score = max(best_score, fuzzy_score)
        if best_score <= 0.0:
            continue
        enriched = dict(row)
        enriched["canonical_key"] = canonical_key
        enriched["match_score"] = best_score
        existing = matches.get(str(row["entity_id"]))
        if existing is None or float(existing["match_score"]) < best_score:
            matches[str(row["entity_id"])] = enriched

    ranked = sorted(matches.values(), key=lambda item: float(item["match_score"]), reverse=True)
    return ranked[:5]


def _token_match_score(term: str, candidate: str) -> float:
    if not term or not candidate:
        return 0.0
    if term == candidate:
        return 1.0
    candidate_tokens = candidate.split()
    if term in candidate_tokens:
        return 0.94
    if " " in term:
        candidate_bigrams = {
            " ".join((candidate_tokens[index], candidate_tokens[index + 1]))
            for index in range(max(0, len(candidate_tokens) - 1))
        }
        if term in candidate_bigrams:
            return 0.96
    return 0.0


def _relation_status_multiplier(status: str) -> float:
    if status == "current":
        return 1.00
    if status == "past":
        return 0.70
    if status == "uncertain":
        return 0.55
    return 0.30


def _confidence_multiplier(confidence: str) -> float:
    if confidence == "high":
        return 1.00
    if confidence == "medium":
        return 0.92
    return 0.82


def _depth_multiplier(depth: int) -> float:
    return 1.0 if depth == 1 else 0.72


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _as_path(value: Any):
    from pathlib import Path

    return Path(str(value))
