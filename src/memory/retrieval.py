"""Lexical, semantic, graph, and hybrid ranking for memory search."""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from llm import EmbeddingRequest, LLMService

from .graph import expand_graph_candidates
from .index_db import MemoryIndexDB
from .types import MemorySearchResult, SearchCandidate


class MemoryRetriever:
    """Runs the memory retrieval pipeline against the derived sidecar index."""

    def __init__(
        self,
        *,
        index_db: MemoryIndexDB,
        llm_service: LLMService | None,
    ) -> None:
        self._index_db = index_db
        self._llm_service = llm_service

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
    ) -> tuple[MemorySearchResult, ...]:
        normalized_mode = "hybrid" if mode == "auto" else mode

        lexical_rows = (
            self._index_db.lexical_candidates(
                query=query,
                scopes=scopes,
                include_expired=include_expired,
                daily_lookback_days=daily_lookback_days,
                limit=max(30, top_k),
            )
            if normalized_mode in {"lexical", "hybrid"}
            else []
        )
        semantic_disabled = not self._index_db.semantic_enabled
        semantic_rows: list[dict[str, Any]] = []
        if normalized_mode in {"semantic", "hybrid"} and not semantic_disabled and self._llm_service is not None:
            embedding_response = await self._llm_service.embed(EmbeddingRequest(inputs=query))
            if embedding_response.embeddings:
                semantic_rows = self._index_db.semantic_candidates(
                    embedding=embedding_response.embeddings[0],
                    scopes=scopes,
                    include_expired=include_expired,
                    daily_lookback_days=daily_lookback_days,
                    limit=max(30, top_k),
                )

        graph_candidates = (
            expand_graph_candidates(
                query=query,
                entity_rows=self._index_db.graph_entities(),
                relation_rows=self._index_db.graph_relations(
                    scopes=scopes,
                    include_expired=include_expired,
                    daily_lookback_days=daily_lookback_days,
                ),
                expand=expand,
            )
            if normalized_mode in {"graph", "hybrid"} and expand > 0
            else ()
        )

        candidates: "OrderedDict[tuple[str, str], SearchCandidate]" = OrderedDict()
        _merge_ranked_rows(candidates, lexical_rows, score_field="lexical_score", reason="lexical_match")
        _merge_ranked_rows(candidates, semantic_rows, score_field="semantic_score", reason="semantic_match")
        _merge_graph_candidates(candidates, graph_candidates)

        if normalized_mode == "lexical":
            final = sorted(candidates.values(), key=lambda item: item.lexical_score, reverse=True)
        elif normalized_mode == "semantic":
            final = sorted(candidates.values(), key=lambda item: item.semantic_score, reverse=True)
        elif normalized_mode == "graph":
            final = sorted(candidates.values(), key=lambda item: item.graph_score, reverse=True)
        else:
            final = _fuse_candidates(candidates.values())

        return tuple(
            MemorySearchResult(
                document_id=item.document_id,
                path=item.path,
                kind=item.kind,
                section_path=item.section_path,
                score=round(item.fused_score if normalized_mode == "hybrid" else _mode_score(item, normalized_mode), 6),
                snippet=item.snippet,
                match_reasons=item.match_reasons,
                source_ref_ids=item.source_ref_ids,
                semantic_disabled=semantic_disabled,
            )
            for item in final[:top_k]
        )


def _merge_ranked_rows(
    candidates: "OrderedDict[tuple[str, str], SearchCandidate]",
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    reason: str,
) -> None:
    if not rows:
        return
    count = len(rows)
    for index, row in enumerate(rows):
        normalized_score = 1.0 - (index / max(1, count))
        key = (str(row["document_id"]), str(row["section_path"]))
        candidate = candidates.get(key)
        source_ref_ids = tuple(json.loads(str(row["source_ref_ids_json"])))
        if candidate is None:
            candidate = SearchCandidate(
                document_id=str(row["document_id"]),
                path=Path(str(row["path"])),
                kind=str(row["kind"]),  # type: ignore[arg-type]
                section_path=str(row["section_path"]),
                snippet=str(row["snippet"]),
                source_ref_ids=source_ref_ids,
                match_reasons=(),
            )
        candidate = _apply_score(candidate, score_field=score_field, value=normalized_score, reason=reason)
        candidates[key] = candidate


def _merge_graph_candidates(
    candidates: "OrderedDict[tuple[str, str], SearchCandidate]",
    graph_candidates: tuple[SearchCandidate, ...],
) -> None:
    for graph_candidate in graph_candidates:
        key = (graph_candidate.document_id, graph_candidate.section_path)
        candidate = candidates.get(key)
        if candidate is None:
            candidates[key] = graph_candidate
            continue
        candidates[key] = SearchCandidate(
            document_id=candidate.document_id,
            path=candidate.path,
            kind=candidate.kind,
            section_path=candidate.section_path,
            snippet=candidate.snippet,
            source_ref_ids=tuple(dict.fromkeys(candidate.source_ref_ids + graph_candidate.source_ref_ids)),
            lexical_score=candidate.lexical_score,
            semantic_score=candidate.semantic_score,
            graph_score=max(candidate.graph_score, graph_candidate.graph_score),
            recency_score=candidate.recency_score,
            match_reasons=tuple(dict.fromkeys(candidate.match_reasons + graph_candidate.match_reasons)),
        )


def _apply_score(candidate: SearchCandidate, *, score_field: str, value: float, reason: str) -> SearchCandidate:
    lexical = candidate.lexical_score
    semantic = candidate.semantic_score
    graph = candidate.graph_score
    if score_field == "lexical_score":
        lexical = max(lexical, value)
    elif score_field == "semantic_score":
        semantic = max(semantic, value)
    else:
        graph = max(graph, value)
    return SearchCandidate(
        document_id=candidate.document_id,
        path=candidate.path,
        kind=candidate.kind,
        section_path=candidate.section_path,
        snippet=candidate.snippet,
        source_ref_ids=candidate.source_ref_ids,
        lexical_score=lexical,
        semantic_score=semantic,
        graph_score=graph,
        recency_score=candidate.recency_score,
        match_reasons=tuple(dict.fromkeys(candidate.match_reasons + (reason,))),
    )


def _fuse_candidates(candidates: Any) -> list[SearchCandidate]:
    fused: list[SearchCandidate] = []
    for candidate in candidates:
        recency_score = _recency_score(candidate.path, candidate.kind)
        base_score = (
            0.40 * candidate.semantic_score
            + 0.35 * candidate.lexical_score
            + 0.15 * candidate.graph_score
            + 0.10 * recency_score
        )
        kind_multiplier = 1.20 if candidate.kind == "core" else 1.10 if candidate.kind == "ongoing" else 1.00
        final_score = base_score * kind_multiplier
        fused.append(
            SearchCandidate(
                document_id=candidate.document_id,
                path=candidate.path,
                kind=candidate.kind,
                section_path=candidate.section_path,
                snippet=candidate.snippet,
                source_ref_ids=candidate.source_ref_ids,
                lexical_score=candidate.lexical_score,
                semantic_score=candidate.semantic_score,
                graph_score=candidate.graph_score,
                recency_score=recency_score,
                fused_score=final_score,
                match_reasons=candidate.match_reasons,
            )
        )
    fused.sort(key=lambda item: item.fused_score, reverse=True)
    return fused


def _recency_score(path: Path, kind: str) -> float:
    if kind == "core":
        return 0.5
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except FileNotFoundError:
        return 0.0
    age_days = max(0.0, (datetime.now(timezone.utc) - modified_at).total_seconds() / 86400.0)
    half_life = 14.0 if kind == "ongoing" else 7.0
    return math.exp(-math.log(2) * age_days / half_life)


def _mode_score(candidate: SearchCandidate, mode: str) -> float:
    if mode == "lexical":
        return candidate.lexical_score
    if mode == "semantic":
        return candidate.semantic_score
    if mode == "graph":
        return candidate.graph_score
    return candidate.fused_score
