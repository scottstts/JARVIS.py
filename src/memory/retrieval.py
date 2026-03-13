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
        warnings: list[str] = []
        query_plan = _build_query_plan(query)

        lexical_variant_rows: list[tuple[_LexicalQueryVariant, list[dict[str, Any]]]] = []
        if normalized_mode in {"lexical", "hybrid"}:
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

        semantic_requested = normalized_mode in {"semantic", "hybrid"}
        semantic_disabled = not semantic_requested
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
            except Exception as exc:
                semantic_disabled = True
                semantic_rows = []
                if normalized_mode == "hybrid":
                    warnings.append(
                        f"semantic search failed at runtime and was skipped; used lexical+graph fallback: {exc}"
                    )
                else:
                    warnings.append(f"semantic search failed at runtime and was skipped: {exc}")
        elif semantic_requested and semantic_reason is not None:
            if normalized_mode == "hybrid":
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
            if normalized_mode in {"graph", "hybrid"} and expand > 0
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
            raw_score_field="distance",
            reason="semantic_match",
            lower_is_better=True,
        )
        _merge_graph_candidates(candidates, graph_candidates)

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
            final = _fuse_candidates(candidates.values())[:final_limit]

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


def _build_query_plan(query: str) -> _QueryPlan:
    raw_tokens = [token.lower() for token in _RAW_TOKEN_PATTERN.findall(query.lower())]
    safe_tokens = _safe_query_tokens(raw_tokens)
    meaningful_tokens = [
        token
        for token in safe_tokens
        if token in _MEANINGFUL_SHORT_TOKENS or token not in _STOP_WORDS or len(token) > 2
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
) -> None:
    if not rows:
        return
    normalized_scores = _normalize_channel_scores(
        rows=rows,
        raw_score_field=raw_score_field,
        lower_is_better=lower_is_better,
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
        if raw_value is not None:
            semantic_distance = raw_value if semantic_distance is None else min(semantic_distance, raw_value)
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


def _rank_fallback_scores(count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    return [1.0 - (index / (count - 1)) for index in range(count)]


def _fuse_candidates(candidates: Iterable[SearchCandidate]) -> list[SearchCandidate]:
    now = datetime.now(timezone.utc)
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


def _parse_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
