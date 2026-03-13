"""Lexical, semantic, graph, and hybrid ranking for memory search."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from llm import EmbeddingRequest, LLMService

from .config import MemorySettings
from .graph import expand_graph_candidates
from .index_db import MemoryIndexDB
from .types import MemorySearchResponse, MemorySearchResult, SearchCandidate

_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "know",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "remember",
    "s",
    "should",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "t",
    "to",
    "us",
    "ve",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}
_MEANINGFUL_SHORT_TOKENS = {
    "ai",
    "c",
    "go",
    "js",
    "ml",
    "r",
    "ts",
    "ui",
    "ux",
}
_FALLBACK_GENERIC_TOKENS = {
    "again",
    "detail",
    "details",
    "info",
    "remember",
    "remind",
    "thing",
    "things",
}
_RAW_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9+#./_-]*")
_FTS_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class _LexicalQueryVariant:
    reason: str
    match_query: str


@dataclass(slots=True, frozen=True)
class _QueryPlan:
    normalized_query: str
    keyword_tokens: tuple[str, ...]
    lexical_variants: tuple[_LexicalQueryVariant, ...]
    entity_terms: tuple[str, ...]


class MemoryRetriever:
    """Runs the memory retrieval pipeline against the derived sidecar index."""

    def __init__(
        self,
        *,
        index_db: MemoryIndexDB,
        llm_service: LLMService | None,
        settings: MemorySettings,
    ) -> None:
        self._index_db = index_db
        self._llm_service = llm_service
        self._settings = settings

    async def search(
        self,
        *,
        query: str,
        mode: str,
        scopes: tuple[str, ...],
        top_k: int,
        daily_lookback_days: int,
        expand: int,
        include_expired: bool,
    ) -> MemorySearchResponse:
        normalized_mode = "hybrid" if mode == "auto" else mode
        query_plan = _build_query_plan(query)
        candidates, warnings, semantic_disabled = await self._collect_candidates(
            query=query,
            query_plan=query_plan,
            mode=normalized_mode,
            scopes=scopes,
            daily_lookback_days=daily_lookback_days,
            expand=expand,
            include_expired=include_expired,
        )

        if normalized_mode == "hybrid" and self._settings.retrieval_fallback_max_queries > 0:
            initial_fused = _prune_weak_semantic_tail(
                _fuse_candidates(candidates.values(), query=query, settings=self._settings),
                settings=self._settings,
            )
            if _should_attempt_fallback(initial_fused, settings=self._settings):
                used_queries = {
                    _normalize_query_text(query),
                    _normalize_query_text(query_plan.normalized_query),
                }
                fallback_used = False
                for fallback_query in _fallback_queries(query=query, query_plan=query_plan)[
                    : self._settings.retrieval_fallback_max_queries
                ]:
                    normalized_fallback = _normalize_query_text(fallback_query)
                    if not normalized_fallback or normalized_fallback in used_queries:
                        continue
                    used_queries.add(normalized_fallback)
                    fallback_used = True
                    fallback_candidates, fallback_warnings, fallback_semantic_disabled = await self._collect_candidates(
                        query=fallback_query,
                        query_plan=_build_query_plan(fallback_query),
                        mode=normalized_mode,
                        scopes=scopes,
                        daily_lookback_days=daily_lookback_days,
                        expand=expand,
                        include_expired=include_expired,
                    )
                    _merge_candidate_sets(
                        candidates,
                        fallback_candidates.values(),
                        reason="retrieval_fallback",
                    )
                    warnings.extend(fallback_warnings)
                    semantic_disabled = semantic_disabled and fallback_semantic_disabled
                if fallback_used:
                    warnings.append(
                        "hybrid retrieval looked weak on the first pass and used a bounded fallback query pass"
                    )

        final_limit = max(top_k, self._settings.hybrid_result_count)
        if normalized_mode == "lexical":
            final = sorted(
                candidates.values(),
                key=lambda item: (item.lexical_score, item.pinned, item.priority or 0, item.updated_at),
                reverse=True,
            )
        elif normalized_mode == "semantic":
            final = sorted(
                candidates.values(),
                key=lambda item: (item.semantic_score, item.pinned, item.priority or 0, item.updated_at),
                reverse=True,
            )
        elif normalized_mode == "graph":
            final = sorted(
                candidates.values(),
                key=lambda item: (item.graph_score, item.pinned, item.priority or 0, item.updated_at),
                reverse=True,
            )
        else:
            final = _prune_weak_semantic_tail(
                _fuse_candidates(candidates.values(), query=query, settings=self._settings),
                settings=self._settings,
            )[:final_limit]
            if not final and candidates:
                warnings.append(
                    "memory retrieval found only weak low-confidence candidates and returned no matches"
                )

        return MemorySearchResponse(
            results=tuple(
                MemorySearchResult(
                    document_id=item.document_id,
                    title=item.title,
                    path=item.path,
                    kind=item.kind,
                    chunk_id=item.chunk_id,
                    section_path=item.section_path,
                    score=round(
                        item.fused_score
                        if normalized_mode == "hybrid"
                        else _mode_score(item, normalized_mode),
                        6,
                    ),
                    snippet=item.snippet,
                    match_reasons=item.match_reasons,
                    source_ref_ids=item.source_ref_ids,
                    semantic_disabled=semantic_disabled,
                )
                for item in final[:top_k]
            ),
            warnings=tuple(dict.fromkeys(warnings)),
            semantic_disabled=semantic_disabled,
        )

    async def _collect_candidates(
        self,
        *,
        query: str,
        query_plan: _QueryPlan,
        mode: str,
        scopes: tuple[str, ...],
        daily_lookback_days: int,
        expand: int,
        include_expired: bool,
    ) -> tuple[OrderedDict[tuple[str, str], SearchCandidate], list[str], bool]:
        warnings: list[str] = []
        lexical_variant_rows: list[tuple[_LexicalQueryVariant, list[dict[str, Any]]]] = []
        if mode in {"lexical", "hybrid"}:
            for variant in query_plan.lexical_variants:
                rows = self._index_db.lexical_candidates(
                    query=variant.match_query,
                    scopes=scopes,
                    include_expired=include_expired,
                    daily_lookback_days=daily_lookback_days,
                    limit=self._settings.lexical_candidate_count,
                )
                if rows:
                    lexical_variant_rows.append((variant, rows))

        semantic_requested = mode in {"semantic", "hybrid"}
        semantic_disabled = False
        semantic_ready = False
        semantic_reason: str | None = None
        if semantic_requested:
            semantic_disabled = True
            semantic_ready, semantic_reason = self._index_db.semantic_search_status()
            if self._llm_service is None:
                semantic_ready = False
                semantic_reason = "the embedding service is not configured"

        semantic_rows: list[dict[str, Any]] = []
        if semantic_requested and semantic_ready and self._llm_service is not None:
            semantic_disabled = False
            try:
                embedding_response = await self._llm_service.embed(EmbeddingRequest(inputs=query))
                if embedding_response.embeddings:
                    semantic_rows = self._index_db.semantic_candidates(
                        embedding=embedding_response.embeddings[0],
                        scopes=scopes,
                        include_expired=include_expired,
                        daily_lookback_days=daily_lookback_days,
                        limit=self._settings.semantic_candidate_count,
                    )
                    semantic_rows = [
                        row
                        for row in semantic_rows
                        if _optional_float(row.get("semantic_similarity")) is not None
                        and float(row["semantic_similarity"]) >= self._settings.semantic_score_floor
                    ]
            except Exception as exc:
                semantic_disabled = True
                semantic_rows = []
                if mode == "hybrid":
                    warnings.append(
                        f"semantic search failed at runtime and was skipped; used lexical+graph fallback: {exc}"
                    )
                else:
                    warnings.append(f"semantic search failed at runtime and was skipped: {exc}")
        elif semantic_requested and semantic_reason is not None:
            if mode == "hybrid":
                warnings.append(
                    "semantic search was skipped because "
                    f"{semantic_reason}; used lexical+graph fallback"
                )
            else:
                warnings.append("semantic search was skipped because " f"{semantic_reason}")

        graph_candidates = (
            expand_graph_candidates(
                query=query,
                normalized_query=query_plan.normalized_query,
                entity_terms=query_plan.entity_terms,
                entity_rows=self._index_db.graph_entities(),
                relation_rows=self._index_db.graph_relations(
                    scopes=scopes,
                    include_expired=include_expired,
                    daily_lookback_days=daily_lookback_days,
                ),
                expand=expand,
                limit=self._settings.graph_candidate_count,
            )
            if mode in {"graph", "hybrid"} and expand > 0
            else ()
        )

        candidates: OrderedDict[tuple[str, str], SearchCandidate] = OrderedDict()
        for variant, rows in lexical_variant_rows:
            _merge_ranked_rows(
                candidates,
                rows,
                score_field="lexical_score",
                raw_score_field="bm25_score",
                reason=variant.reason,
                lower_is_better=True,
            )
        _merge_ranked_rows(
            candidates,
            semantic_rows,
            score_field="semantic_score",
            raw_score_field="semantic_similarity",
            reason="semantic_match",
            lower_is_better=False,
            normalize_scores=False,
            distance_field="distance",
        )
        _merge_graph_candidates(candidates, graph_candidates)
        return candidates, warnings, semantic_disabled


def _build_query_plan(query: str) -> _QueryPlan:
    raw_tokens = [token.lower() for token in _RAW_TOKEN_PATTERN.findall(query.lower())]
    safe_tokens = _safe_query_tokens(raw_tokens)
    meaningful_tokens = [
        token
        for token in safe_tokens
        if token in _MEANINGFUL_SHORT_TOKENS or token not in _STOP_WORDS
    ]
    if not meaningful_tokens:
        meaningful_tokens = safe_tokens

    normalized_query = " ".join(meaningful_tokens).strip()
    phrase_tokens = meaningful_tokens[:6]
    keyword_tokens = _dedupe_preserve_order(meaningful_tokens[:8])
    prefix_tokens = [token for token in keyword_tokens if len(token) >= 3]

    variants: list[_LexicalQueryVariant] = []
    if phrase_tokens:
        if len(phrase_tokens) == 1:
            variants.append(_LexicalQueryVariant("lexical_phrase", phrase_tokens[0]))
        else:
            variants.append(
                _LexicalQueryVariant(
                    "lexical_phrase",
                    '"' + " ".join(phrase_tokens) + '"',
                )
            )
    if keyword_tokens:
        variants.append(
            _LexicalQueryVariant(
                "lexical_keywords",
                " OR ".join(keyword_tokens),
            )
        )
    if prefix_tokens:
        variants.append(
            _LexicalQueryVariant(
                "lexical_prefix",
                " OR ".join(f"{token}*" for token in prefix_tokens),
            )
        )

    if not variants and safe_tokens:
        variants.append(_LexicalQueryVariant("lexical_keywords", " OR ".join(_dedupe_preserve_order(safe_tokens))))

    entity_terms: list[str] = []
    base_terms = keyword_tokens or safe_tokens
    entity_terms.extend(base_terms[:8])
    entity_terms.extend(
        " ".join((base_terms[index], base_terms[index + 1]))
        for index in range(max(0, len(base_terms) - 1))
    )
    if normalized_query:
        entity_terms.insert(0, normalized_query)
    return _QueryPlan(
        normalized_query=normalized_query,
        keyword_tokens=tuple(keyword_tokens),
        lexical_variants=tuple(_dedupe_variants(variants)),
        entity_terms=tuple(_dedupe_preserve_order(term for term in entity_terms if term.strip())),
    )


def _safe_query_tokens(raw_tokens: Sequence[str]) -> list[str]:
    safe_tokens: list[str] = []
    for token in raw_tokens:
        safe_tokens.extend(part for part in _FTS_TOKEN_PATTERN.findall(token) if part)
    return safe_tokens


def _dedupe_variants(variants: Sequence[_LexicalQueryVariant]) -> list[_LexicalQueryVariant]:
    seen: set[str] = set()
    result: list[_LexicalQueryVariant] = []
    for variant in variants:
        if not variant.match_query or variant.match_query in seen:
            continue
        seen.add(variant.match_query)
        result.append(variant)
    return result


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _merge_ranked_rows(
    candidates: OrderedDict[tuple[str, str], SearchCandidate],
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    raw_score_field: str,
    reason: str,
    lower_is_better: bool,
    normalize_scores: bool = True,
    distance_field: str | None = None,
) -> None:
    if not rows:
        return
    normalized_scores = (
        _normalize_channel_scores(
            rows=rows,
            raw_score_field=raw_score_field,
            lower_is_better=lower_is_better,
        )
        if normalize_scores
        else _raw_channel_scores(rows=rows, raw_score_field=raw_score_field)
    )
    for row, normalized_score in zip(rows, normalized_scores, strict=True):
        key = (str(row["document_id"]), str(row["section_path"]))
        candidate = candidates.get(key) or _candidate_from_row(row)
        candidate = _apply_score(
            candidate,
            row=row,
            score_field=score_field,
            raw_score_field=raw_score_field,
            value=normalized_score,
            reason=reason,
            distance_field=distance_field,
        )
        candidates[key] = candidate


def _merge_graph_candidates(
    candidates: OrderedDict[tuple[str, str], SearchCandidate],
    graph_candidates: tuple[SearchCandidate, ...],
) -> None:
    for graph_candidate in graph_candidates:
        key = (graph_candidate.document_id, graph_candidate.section_path)
        candidate = candidates.get(key)
        if candidate is None:
            candidates[key] = graph_candidate
            continue
        next_candidate = candidate
        if graph_candidate.graph_score > candidate.graph_score:
            next_candidate = _replace_candidate_content(candidate, graph_candidate)
        candidates[key] = SearchCandidate(
            document_id=next_candidate.document_id,
            title=next_candidate.title,
            path=next_candidate.path,
            kind=next_candidate.kind,
            chunk_id=next_candidate.chunk_id,
            section_path=next_candidate.section_path,
            snippet=next_candidate.snippet,
            source_ref_ids=tuple(
                dict.fromkeys(candidate.source_ref_ids + graph_candidate.source_ref_ids)
            ),
            updated_at=next_candidate.updated_at,
            status=next_candidate.status,
            pinned=next_candidate.pinned,
            priority=next_candidate.priority,
            review_after=next_candidate.review_after,
            expires_at=next_candidate.expires_at,
            archived_at=next_candidate.archived_at,
            truth_status=next_candidate.truth_status,
            support_count=next_candidate.support_count,
            contradiction_count=next_candidate.contradiction_count,
            last_confirmed_at=next_candidate.last_confirmed_at,
            last_contradicted_at=next_candidate.last_contradicted_at,
            lexical_raw_score=next_candidate.lexical_raw_score,
            lexical_score=next_candidate.lexical_score,
            semantic_distance=next_candidate.semantic_distance,
            semantic_score=next_candidate.semantic_score,
            graph_score=max(candidate.graph_score, graph_candidate.graph_score),
            recency_score=next_candidate.recency_score,
            fused_score=next_candidate.fused_score,
            match_reasons=tuple(
                dict.fromkeys(candidate.match_reasons + graph_candidate.match_reasons)
            ),
        )


def _candidate_from_row(row: dict[str, Any]) -> SearchCandidate:
    source_ref_ids = tuple(json.loads(str(row["source_ref_ids_json"])))
    return SearchCandidate(
        document_id=str(row["document_id"]),
        title=str(row["title"]),
        path=Path(str(row["path"])),
        kind=str(row["kind"]),  # type: ignore[arg-type]
        chunk_id=str(row["chunk_id"]),
        section_path=str(row["section_path"]),
        snippet=_compact_snippet(str(row["snippet"])),
        source_ref_ids=source_ref_ids,
        updated_at=str(row["updated_at"]),
        status=str(row["status"]),
        pinned=bool(row["pinned"]),
        priority=int(row["priority"]) if row["priority"] is not None else None,
        review_after=_optional_str(row.get("review_after")),
        expires_at=_optional_str(row.get("expires_at")),
        archived_at=_optional_str(row.get("archived_at")),
        truth_status=_optional_str(row.get("truth_status")),
        support_count=_optional_int(row.get("support_count")) or 0,
        contradiction_count=_optional_int(row.get("contradiction_count")) or 0,
        last_confirmed_at=_optional_str(row.get("last_confirmed_at")),
        last_contradicted_at=_optional_str(row.get("last_contradicted_at")),
        match_reasons=(),
    )


def _apply_score(
    candidate: SearchCandidate,
    *,
    row: dict[str, Any],
    score_field: str,
    raw_score_field: str,
    value: float,
    reason: str,
    distance_field: str | None,
) -> SearchCandidate:
    lexical_raw_score = candidate.lexical_raw_score
    lexical_score = candidate.lexical_score
    semantic_distance = candidate.semantic_distance
    semantic_score = candidate.semantic_score
    graph_score = candidate.graph_score
    next_candidate = candidate

    if score_field == "lexical_score":
        raw_value = _optional_float(row.get(raw_score_field))
        if lexical_score <= value:
            next_candidate = _replace_candidate_content(candidate, _candidate_from_row(row))
        if raw_value is not None:
            lexical_raw_score = raw_value if lexical_raw_score is None else min(lexical_raw_score, raw_value)
        lexical_score = max(lexical_score, value)
    elif score_field == "semantic_score":
        raw_value = _optional_float(row.get(raw_score_field))
        if semantic_score <= value:
            next_candidate = _replace_candidate_content(candidate, _candidate_from_row(row))
        distance_value = _optional_float(row.get(distance_field)) if distance_field is not None else raw_value
        if distance_value is not None:
            semantic_distance = distance_value if semantic_distance is None else min(semantic_distance, distance_value)
        semantic_score = max(semantic_score, value)
    else:
        if graph_score <= value:
            next_candidate = _replace_candidate_content(candidate, _candidate_from_row(row))
        graph_score = max(graph_score, value)

    return SearchCandidate(
        document_id=next_candidate.document_id,
        title=next_candidate.title,
        path=next_candidate.path,
        kind=next_candidate.kind,
        chunk_id=next_candidate.chunk_id,
        section_path=next_candidate.section_path,
        snippet=next_candidate.snippet,
        source_ref_ids=next_candidate.source_ref_ids,
        updated_at=next_candidate.updated_at,
        status=next_candidate.status,
        pinned=next_candidate.pinned,
        priority=next_candidate.priority,
        review_after=next_candidate.review_after,
        expires_at=next_candidate.expires_at,
        archived_at=next_candidate.archived_at,
        truth_status=next_candidate.truth_status,
        support_count=next_candidate.support_count,
        contradiction_count=next_candidate.contradiction_count,
        last_confirmed_at=next_candidate.last_confirmed_at,
        last_contradicted_at=next_candidate.last_contradicted_at,
        lexical_raw_score=lexical_raw_score,
        lexical_score=lexical_score,
        semantic_distance=semantic_distance,
        semantic_score=semantic_score,
        graph_score=graph_score,
        recency_score=next_candidate.recency_score,
        fused_score=next_candidate.fused_score,
        match_reasons=tuple(dict.fromkeys(next_candidate.match_reasons + (reason,))),
    )


def _replace_candidate_content(candidate: SearchCandidate, replacement: SearchCandidate) -> SearchCandidate:
    return SearchCandidate(
        document_id=candidate.document_id,
        title=replacement.title,
        path=replacement.path,
        kind=replacement.kind,
        chunk_id=replacement.chunk_id,
        section_path=replacement.section_path,
        snippet=replacement.snippet,
        source_ref_ids=replacement.source_ref_ids,
        updated_at=replacement.updated_at,
        status=replacement.status,
        pinned=replacement.pinned,
        priority=replacement.priority,
        review_after=replacement.review_after,
        expires_at=replacement.expires_at,
        archived_at=replacement.archived_at,
        truth_status=replacement.truth_status,
        support_count=replacement.support_count,
        contradiction_count=replacement.contradiction_count,
        last_confirmed_at=replacement.last_confirmed_at,
        last_contradicted_at=replacement.last_contradicted_at,
        lexical_raw_score=candidate.lexical_raw_score,
        lexical_score=candidate.lexical_score,
        semantic_distance=candidate.semantic_distance,
        semantic_score=candidate.semantic_score,
        graph_score=candidate.graph_score,
        recency_score=candidate.recency_score,
        fused_score=candidate.fused_score,
        match_reasons=candidate.match_reasons,
    )


def _normalize_channel_scores(
    *,
    rows: Sequence[dict[str, Any]],
    raw_score_field: str,
    lower_is_better: bool,
) -> list[float]:
    raw_values = [_optional_float(row.get(raw_score_field)) for row in rows]
    if any(value is None for value in raw_values):
        return _rank_fallback_scores(len(rows))

    usable_values = [value for value in raw_values if value is not None]
    min_value = min(usable_values)
    max_value = max(usable_values)
    if math.isclose(min_value, max_value):
        return _rank_fallback_scores(len(rows))

    normalized: list[float] = []
    for raw_value in usable_values:
        scale = (raw_value - min_value) / (max_value - min_value)
        normalized.append(1.0 - scale if lower_is_better else scale)
    return normalized


def _raw_channel_scores(
    *,
    rows: Sequence[dict[str, Any]],
    raw_score_field: str,
) -> list[float]:
    raw_values = [_optional_float(row.get(raw_score_field)) for row in rows]
    if any(value is None for value in raw_values):
        return _rank_fallback_scores(len(rows))
    return [_clamp_score(value) for value in raw_values if value is not None]


def _rank_fallback_scores(count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    return [1.0 - (index / (count - 1)) for index in range(count)]


def _merge_candidate_sets(
    candidates: OrderedDict[tuple[str, str], SearchCandidate],
    additions: Iterable[SearchCandidate],
    *,
    reason: str,
) -> None:
    for addition in additions:
        key = (addition.document_id, addition.section_path)
        existing = candidates.get(key)
        tagged_addition = SearchCandidate(
            document_id=addition.document_id,
            title=addition.title,
            path=addition.path,
            kind=addition.kind,
            chunk_id=addition.chunk_id,
            section_path=addition.section_path,
            snippet=addition.snippet,
            source_ref_ids=addition.source_ref_ids,
            updated_at=addition.updated_at,
            status=addition.status,
            pinned=addition.pinned,
            priority=addition.priority,
            review_after=addition.review_after,
            expires_at=addition.expires_at,
            archived_at=addition.archived_at,
            truth_status=addition.truth_status,
            support_count=addition.support_count,
            contradiction_count=addition.contradiction_count,
            last_confirmed_at=addition.last_confirmed_at,
            last_contradicted_at=addition.last_contradicted_at,
            lexical_raw_score=addition.lexical_raw_score,
            lexical_score=addition.lexical_score,
            semantic_distance=addition.semantic_distance,
            semantic_score=addition.semantic_score,
            graph_score=addition.graph_score,
            recency_score=addition.recency_score,
            fused_score=addition.fused_score,
            match_reasons=tuple(dict.fromkeys(addition.match_reasons + (reason,))),
        )
        if existing is None:
            candidates[key] = tagged_addition
            continue
        replacement = (
            tagged_addition
            if _candidate_strength(tagged_addition) > _candidate_strength(existing)
            else existing
        )
        candidates[key] = SearchCandidate(
            document_id=existing.document_id,
            title=replacement.title,
            path=replacement.path,
            kind=replacement.kind,
            chunk_id=replacement.chunk_id,
            section_path=replacement.section_path,
            snippet=replacement.snippet,
            source_ref_ids=tuple(dict.fromkeys(existing.source_ref_ids + tagged_addition.source_ref_ids)),
            updated_at=replacement.updated_at,
            status=replacement.status,
            pinned=replacement.pinned,
            priority=replacement.priority,
            review_after=replacement.review_after,
            expires_at=replacement.expires_at,
            archived_at=replacement.archived_at,
            truth_status=replacement.truth_status,
            support_count=max(existing.support_count, tagged_addition.support_count),
            contradiction_count=max(existing.contradiction_count, tagged_addition.contradiction_count),
            last_confirmed_at=_latest_timestamp(existing.last_confirmed_at, tagged_addition.last_confirmed_at),
            last_contradicted_at=_latest_timestamp(
                existing.last_contradicted_at,
                tagged_addition.last_contradicted_at,
            ),
            lexical_raw_score=_min_optional(existing.lexical_raw_score, tagged_addition.lexical_raw_score),
            lexical_score=max(existing.lexical_score, tagged_addition.lexical_score),
            semantic_distance=_min_optional(existing.semantic_distance, tagged_addition.semantic_distance),
            semantic_score=max(existing.semantic_score, tagged_addition.semantic_score),
            graph_score=max(existing.graph_score, tagged_addition.graph_score),
            recency_score=max(existing.recency_score, tagged_addition.recency_score),
            fused_score=max(existing.fused_score, tagged_addition.fused_score),
            match_reasons=tuple(
                dict.fromkeys(existing.match_reasons + tagged_addition.match_reasons)
            ),
        )


def _fuse_candidates(
    candidates: Iterable[SearchCandidate],
    *,
    query: str,
    settings: MemorySettings,
) -> list[SearchCandidate]:
    now = datetime.now(timezone.utc)
    present_state_query = _query_implies_present_state(query)
    fused: list[SearchCandidate] = []
    for candidate in candidates:
        recency_score = _recency_score(candidate.updated_at, candidate.kind, now=now)
        base_score = (
            0.40 * candidate.semantic_score
            + 0.35 * candidate.lexical_score
            + 0.15 * candidate.graph_score
            + 0.10 * recency_score
        )
        kind_multiplier = 1.20 if candidate.kind == "core" else 1.10 if candidate.kind == "ongoing" else 1.00
        modifier = kind_multiplier
        if candidate.pinned:
            modifier *= 1.15
        if candidate.archived_at or candidate.status == "archived":
            modifier *= 0.85
        if _is_expired(candidate, now=now):
            modifier *= 0.50
        if _is_stale_ongoing(candidate, now=now):
            modifier *= 0.90 if candidate.pinned else 0.80
        if _is_semantic_only(candidate):
            if candidate.semantic_score < settings.semantic_only_score_floor:
                modifier *= 0.55
            elif candidate.support_count > 0:
                modifier *= 0.96
            else:
                modifier *= 0.84
        if candidate.support_count > 0:
            modifier *= 1.0 + 0.04 * min(candidate.support_count, 5)
        if candidate.contradiction_count > 0:
            modifier *= max(0.50, 1.0 - 0.12 * min(candidate.contradiction_count, 4))
            if _is_recent(candidate.last_contradicted_at, now=now, days=30):
                modifier *= 0.88
        if candidate.last_confirmed_at is not None:
            modifier *= 1.0 + (0.06 * _confirmation_freshness_score(candidate.last_confirmed_at, now=now))
        modifier *= _truth_status_modifier(
            candidate.truth_status,
            present_state_query=present_state_query,
        )
        fused.append(
            SearchCandidate(
                document_id=candidate.document_id,
                title=candidate.title,
                path=candidate.path,
                kind=candidate.kind,
                chunk_id=candidate.chunk_id,
                section_path=candidate.section_path,
                snippet=candidate.snippet,
                source_ref_ids=candidate.source_ref_ids,
                updated_at=candidate.updated_at,
                status=candidate.status,
                pinned=candidate.pinned,
                priority=candidate.priority,
                review_after=candidate.review_after,
                expires_at=candidate.expires_at,
                archived_at=candidate.archived_at,
                truth_status=candidate.truth_status,
                support_count=candidate.support_count,
                contradiction_count=candidate.contradiction_count,
                last_confirmed_at=candidate.last_confirmed_at,
                last_contradicted_at=candidate.last_contradicted_at,
                lexical_raw_score=candidate.lexical_raw_score,
                lexical_score=candidate.lexical_score,
                semantic_distance=candidate.semantic_distance,
                semantic_score=candidate.semantic_score,
                graph_score=candidate.graph_score,
                recency_score=recency_score,
                fused_score=base_score * modifier,
                match_reasons=candidate.match_reasons,
            )
        )
    fused.sort(
        key=lambda item: (
            item.fused_score,
            item.pinned,
            item.priority or 0,
            item.updated_at,
        ),
        reverse=True,
    )
    return fused


def _prune_weak_semantic_tail(
    candidates: list[SearchCandidate],
    *,
    settings: MemorySettings,
) -> list[SearchCandidate]:
    if not candidates:
        return []
    top_window = candidates[:3]
    if top_window and all(
        _is_weak_semantic_candidate(item, settings=settings)
        for item in top_window
    ):
        return [
            item
            for item in candidates
            if not _is_weak_semantic_candidate(item, settings=settings)
        ]
    has_anchor_result = any(
        item.fused_score >= settings.weak_result_score_threshold
        and (_has_non_semantic_support(item) or item.semantic_score >= settings.semantic_only_score_floor)
        for item in top_window
    )
    if not has_anchor_result:
        return [
            item
            for item in candidates
            if not _is_weak_semantic_candidate(item, settings=settings)
        ]
    return [
        item
        for item in candidates
        if not (
            _is_weak_semantic_candidate(item, settings=settings)
            or (
                item.support_count <= 0
                and item.fused_score < settings.weak_result_score_threshold
                and (
                    item.lexical_score <= 0.0
                    and item.graph_score <= 0.0
                    and item.semantic_score < settings.semantic_score_floor
                )
            )
        )
    ]


def _should_attempt_fallback(
    candidates: Sequence[SearchCandidate],
    *,
    settings: MemorySettings,
) -> bool:
    if not candidates:
        return True
    top = candidates[0]
    if top.fused_score < settings.weak_result_score_threshold:
        return True
    top_window = candidates[:3]
    if all(_is_semantic_only(item) for item in top_window):
        return any(item.semantic_score < settings.semantic_only_score_floor for item in top_window)
    return False


def _fallback_queries(*, query: str, query_plan: _QueryPlan) -> tuple[str, ...]:
    del query
    queries: list[str] = []
    queries.extend(
        term
        for term in query_plan.entity_terms
        if " " in term and term != query_plan.normalized_query and _is_fallback_bigram(term)
    )
    if query_plan.normalized_query:
        queries.append(query_plan.normalized_query)
    if query_plan.keyword_tokens:
        queries.append(" ".join(query_plan.keyword_tokens[:3]))
    return tuple(
        _dedupe_preserve_order(
            candidate.strip()
            for candidate in queries
            if candidate.strip()
        )
    )


def _has_non_semantic_support(candidate: SearchCandidate) -> bool:
    return candidate.lexical_score > 0.0 or candidate.graph_score > 0.0 or candidate.support_count > 0


def _is_semantic_only(candidate: SearchCandidate) -> bool:
    return candidate.semantic_score > 0.0 and candidate.lexical_score <= 0.0 and candidate.graph_score <= 0.0


def _is_weak_semantic_candidate(
    candidate: SearchCandidate,
    *,
    settings: MemorySettings,
) -> bool:
    return (
        _is_semantic_only(candidate)
        and candidate.support_count <= 0
        and candidate.fused_score < settings.weak_result_score_threshold
        and candidate.semantic_score < settings.semantic_only_score_floor
    )


def _truth_status_modifier(status: str | None, *, present_state_query: bool) -> float:
    if status is None:
        return 1.0
    if status == "current":
        return 1.08 if present_state_query else 1.0
    if status == "past":
        return 0.82 if present_state_query else 1.0
    if status == "uncertain":
        return 0.76 if present_state_query else 0.92
    if status == "superseded":
        return 0.55 if present_state_query else 0.78
    return 1.0


def _confirmation_freshness_score(confirmed_at: str, *, now: datetime) -> float:
    parsed = _parse_iso(confirmed_at)
    if parsed is None:
        return 0.0
    age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    return math.exp(-math.log(2) * age_days / 21.0)


def _is_recent(value: str | None, *, now: datetime, days: int) -> bool:
    parsed = _parse_iso(value)
    if parsed is None:
        return False
    return (now - parsed).total_seconds() <= days * 86400


def _query_implies_present_state(query: str) -> bool:
    normalized_tokens = set(_safe_query_tokens(_RAW_TOKEN_PATTERN.findall(query.lower())))
    return bool(
        normalized_tokens.intersection(
            {
                "active",
                "currently",
                "current",
                "doing",
                "does",
                "is",
                "now",
                "ongoing",
                "prefer",
                "prefers",
                "status",
                "still",
                "think",
                "using",
                "uses",
                "working",
            }
        )
    )


def _candidate_strength(candidate: SearchCandidate) -> float:
    return max(candidate.lexical_score, candidate.semantic_score, candidate.graph_score)


def _min_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


def _latest_timestamp(*values: str | None) -> str | None:
    normalized = [value for value in values if value]
    return max(normalized) if normalized else None


def _normalize_query_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _is_fallback_bigram(value: str) -> bool:
    tokens = value.split()
    return len(tokens) == 2 and all(token not in _FALLBACK_GENERIC_TOKENS for token in tokens)


def _recency_score(updated_at: str, kind: str, *, now: datetime) -> float:
    parsed = _parse_iso(updated_at)
    if parsed is None:
        return 0.0
    if kind == "core":
        return 0.5
    age_days = max(0.0, (now - parsed).total_seconds() / 86400.0)
    half_life = 14.0 if kind == "ongoing" else 7.0
    return math.exp(-math.log(2) * age_days / half_life)


def _is_expired(candidate: SearchCandidate, *, now: datetime) -> bool:
    if not candidate.expires_at:
        return False
    expires_at = _parse_iso(candidate.expires_at)
    return expires_at is not None and expires_at <= now


def _is_stale_ongoing(candidate: SearchCandidate, *, now: datetime) -> bool:
    if candidate.kind != "ongoing" or candidate.status != "active" or candidate.archived_at:
        return False
    review_after = _parse_iso(candidate.review_after) if candidate.review_after else None
    if review_after is not None and review_after <= now:
        return True
    updated_at = _parse_iso(candidate.updated_at)
    if updated_at is None:
        return False
    return (now - updated_at).total_seconds() >= 30 * 86400


def _mode_score(candidate: SearchCandidate, mode: str) -> float:
    if mode == "lexical":
        return candidate.lexical_score
    if mode == "semantic":
        return candidate.semantic_score
    if mode == "graph":
        return candidate.graph_score
    return candidate.fused_score


def _compact_snippet(value: str, *, limit: int = 240) -> str:
    normalized = _WHITESPACE_PATTERN.sub(" ", value).strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
