"""Hybrid 3-stage search with Personalized PageRank."""

from __future__ import annotations

import math
from time import time
from typing import TYPE_CHECKING

from synaptic.models import ActivatedNode, EdgeKind, Node, NodeKind, SearchResult
from synaptic.ppr import personalized_pagerank, personalized_pagerank_v2
from synaptic.protocols import QueryRewriter, StorageBackend
from synaptic.resonance import ResonanceScorer
from synaptic.synonyms import expand_synonyms

if TYPE_CHECKING:
    from synaptic.extensions.chunk_entity_index import ChunkEntityIndex

# Kind-query keyword mapping (boost the matching kind when these words appear in query)
_KIND_QUERY_HINTS: dict[NodeKind, list[str]] = {
    NodeKind.LESSON: [
        "실패",
        "에러",
        "오류",
        "장애",
        "교훈",
        "배운",
        "주의",
        "failure",
        "error",
        "incident",
        "lesson",
        "postmortem",
    ],
    NodeKind.RULE: [
        "규칙",
        "정책",
        "규정",
        "금지",
        "필수",
        "가이드",
        "rule",
        "policy",
        "constraint",
        "must",
        "forbidden",
    ],
    NodeKind.DECISION: [
        "결정",
        "선택",
        "판단",
        "채택",
        "어떻게",
        "decision",
        "choice",
        "decided",
        "approach",
    ],
    NodeKind.ARTIFACT: [
        "api",
        "엔드포인트",
        "스키마",
        "명세",
        "코드",
        "endpoint",
        "schema",
        "spec",
        "interface",
    ],
    NodeKind.ENTITY: [
        "회사",
        "조직",
        "제품",
        "서비스",
        "시스템",
        "company",
        "organization",
        "product",
        "service",
    ],
}
_KIND_BOOST = 0.05  # search_score boost amount on kind match (conservative)


def _rank_to_score(
    rank: int, *, top: float = 0.95, step: float = 0.05, floor: float = 0.3
) -> float:
    """Rank-based score conversion: rank 1 = top, decreasing by step per rank, clamped at floor."""
    return max(floor, top - rank * step)


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score: 1/(k + rank + 1). k=60 is standard."""
    return 1.0 / (k + rank + 1)


def _rrf_fusion(*rankings: dict[str, float], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion — robust score fusion across different distributions.

    RRF(d) = sum(1 / (k + rank_i)) for each ranking.
    More stable than linear alpha-blending when score distributions differ.

    Args:
        *rankings: Each ranking is {node_id: score}, higher score = better.
        k: RRF constant (default 60, standard value).

    Returns:
        {node_id: rrf_score} — unified scores, higher = better.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        # Sort by score descending to get ranks
        sorted_ids = sorted(ranking, key=lambda nid: ranking[nid], reverse=True)
        for rank, nid in enumerate(sorted_ids):
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
    return scores


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HybridSearch:
    """3-stage fallback search: FTS+vector → synonym expansion → query rewrite."""

    __slots__ = ("_chunk_entity_index", "_ppr_damping", "_query_rewriter", "_scorer")

    def __init__(
        self,
        *,
        scorer: ResonanceScorer | None = None,
        query_rewriter: QueryRewriter | None = None,
        spread_decay: float = 0.25,  # deprecated, kept for compat
        spread_depth: int = 1,  # deprecated, kept for compat
        ppr_damping: float = 0.85,
        chunk_entity_index: ChunkEntityIndex | None = None,
    ) -> None:
        self._scorer = scorer or ResonanceScorer()
        self._query_rewriter = query_rewriter
        self._ppr_damping = ppr_damping
        self._chunk_entity_index = chunk_entity_index

    async def search(
        self,
        backend: StorageBackend,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
        node_kinds: list[NodeKind] | None = None,
        corpus_size: int = 0,
    ) -> SearchResult:
        start = time()
        stages_used: list[str] = []
        all_nodes: dict[str, tuple[Node, float]] = {}

        # Stage 1: FTS-primary + vector cascade
        # FTS 결과는 rank 기반 스코어 유지, vector는 보완 역할
        # 실험 결과: fusion(blend, RRF) 방식은 소규모 corpus에서 FTS 순위 교란 → cascade가 최적
        fts_nodes = await backend.search_fts(query, limit=limit * 2)
        stages_used.append("fts")
        fts_ids: set[str] = set()
        for rank, node in enumerate(fts_nodes):
            score = _rank_to_score(rank)
            fts_ids.add(node.id)
            all_nodes[node.id] = (node, score)

        vec_cosine: dict[str, float] = {}
        if embedding:
            vec_nodes = await backend.search_vector(embedding, limit=limit * 2)
            stages_used.append("vector")

            for rank, node in enumerate(vec_nodes):
                if node.embedding and embedding:
                    vec_cosine[node.id] = _cosine_sim(embedding, node.embedding)

            # Corpus-size adaptive vector integration:
            # FTS 순위 보존 + vector는 FTS가 놓친 결과만 보완
            # vec_alpha: 소규모(0.3) → 대규모(0.85) 점진 증가
            # 실험 기록:
            #   - fusion(blend, RRF) → 소규모에서 FTS 순위 교란, 폐기 (2026-03-23)
            #   - FTS+vector 중복 boost → 모든 규모에서 노이즈 유입, 폐기 (2026-03-26)
            #   - threshold 0.40 → 대규모에서도 recall 개선 없음, 0.45 유지 (2026-03-26)
            vec_alpha = min(0.85, max(0.3, (corpus_size - 500) / 5000 + 0.3))

            for node in vec_nodes:
                nid = node.id
                cos = vec_cosine.get(nid, 0.0)
                if nid not in fts_ids and cos >= 0.45:
                    # Vector-only: FTS에 없는 결과만 삽입
                    vec_score = cos * vec_alpha
                    all_nodes[nid] = (node, vec_score)

        # Stage 2: Synonym expansion (if insufficient results)
        if len(all_nodes) < limit:
            expansions = expand_synonyms(query)
            for expanded_query in expansions[:3]:
                extra = await backend.search_fts(expanded_query, limit=limit)
                for node in extra:
                    if node.id not in all_nodes:
                        all_nodes[node.id] = (node, 0.5)
            if expansions:
                stages_used.append("synonym")

        # Stage 3: Query rewriter fallback (LLM-based)
        if len(all_nodes) < limit and self._query_rewriter is not None:
            rewritten = await self._query_rewriter.rewrite(query)
            for rq in rewritten[:2]:
                extra = await backend.search_fts(rq, limit=limit)
                for node in extra:
                    if node.id not in all_nodes:
                        all_nodes[node.id] = (node, 0.4)
            stages_used.append("rewriter")

        # PPR: graph-aware discovery + mild re-ranking
        total_candidates = len(all_nodes)
        if all_nodes:
            seed_scores = {nid: score for nid, (_node, score) in all_nodes.items()}
            # Use PPR v2 (noise-reduced) when chunk-entity index is available
            if self._chunk_entity_index is not None:
                ppr_results = await personalized_pagerank_v2(
                    backend,
                    seed_scores,
                    chunk_entity_index=self._chunk_entity_index,
                    damping=self._ppr_damping,
                    top_k=limit * 2,
                )
            else:
                ppr_results = await personalized_pagerank(
                    backend,
                    seed_scores,
                    damping=self._ppr_damping,
                    top_k=limit * 2,
                )
            for node_id, ppr_score in ppr_results:
                if node_id not in all_nodes:
                    # Node discovered by PPR — reachable only through graph paths
                    node = await backend.get_node(node_id)
                    if node:
                        all_nodes[node_id] = (node, ppr_score * 0.8)
                else:
                    # Existing FTS result — only mild PPR boost (preserve FTS ranking)
                    existing = all_nodes[node_id]
                    boosted = min(1.0, existing[1] + ppr_score * 0.1)
                    if boosted > existing[1]:
                        all_nodes[node_id] = (existing[0], boosted)

        # Chunk-entity expansion: when entity nodes are found, pull in their
        # source chunks so the final result includes grounded passages.
        if self._chunk_entity_index is not None:
            entity_ids = [
                nid
                for nid, (node, _) in all_nodes.items()
                if node.kind == NodeKind.ENTITY and "_phrase" not in (node.tags or [])
            ]
            if entity_ids:
                chunk_scores = self._chunk_entity_index.chunks_for_entities(entity_ids)
                for chunk_id, overlap_count in list(chunk_scores.items())[:limit * 2]:
                    if chunk_id not in all_nodes:
                        chunk_node = await backend.get_node(chunk_id)
                        if chunk_node:
                            # Score based on entity overlap (normalized)
                            chunk_score = min(0.85, 0.4 + overlap_count * 0.15)
                            all_nodes[chunk_id] = (chunk_node, chunk_score)
                stages_used.append("chunk_expansion")

        # Soft boost for preferred node_kinds (instead of hard filtering)
        if node_kinds:
            kind_set = set(node_kinds)
            for nid, (node, score) in all_nodes.items():
                if node.kind in kind_set:
                    all_nodes[nid] = (node, min(1.0, score * 1.5))

        # Kind-intent boost: boost kinds matching query keywords
        preferred_kinds: set[NodeKind] = set()
        q_lower = query.lower()
        for kind, hints in _KIND_QUERY_HINTS.items():
            if any(h in q_lower for h in hints):
                preferred_kinds.add(kind)

        # Tag-query boost: boost when query keywords appear in node tags
        query_terms_set = set(query.lower().split())

        # Score with resonance
        now = time()
        activated: list[ActivatedNode] = []
        for _nid, (node, search_score) in all_nodes.items():
            # kind boost
            if preferred_kinds and node.kind in preferred_kinds:
                search_score = min(1.0, search_score + _KIND_BOOST)
            # tag boost (exact match only — tags with 2+ characters)
            if node.tags and query_terms_set:
                tag_set = {t.lower() for t in node.tags if len(t) >= 2}
                tag_overlap = len(query_terms_set & tag_set)
                if tag_overlap > 0:
                    search_score = min(1.0, search_score + tag_overlap * 0.03)

            resonance = self._scorer.score(node, search_score=search_score, now=now)
            activated.append(
                ActivatedNode(
                    node=node,
                    activation=search_score,
                    resonance=resonance,
                    path=[],
                )
            )

        # Sort by resonance descending
        activated.sort(key=lambda a: a.resonance, reverse=True)

        # Filter out internal phrase nodes (_phrase tag) from final results.
        final: list[ActivatedNode] = [a for a in activated if "_phrase" not in (a.node.tags or [])]

        # Supersede: same-title AND similar-content nodes → keep only the newest.
        # This ensures knowledge updates are reflected: latest info wins.
        # Title만 같고 content가 다른 노드(예: 같은 파일의 다른 섹션)는 유지.
        seen_titles: dict[str, list[int]] = {}  # normalized_title → [indices in deduped]
        deduped: list[ActivatedNode] = []
        for a in final:
            title_key = a.node.title.strip().lower()
            if not title_key or len(title_key) < 4:
                deduped.append(a)
                continue
            if title_key in seen_titles:
                # Check if content is similar to any existing with same title
                content_snippet = a.node.content[:200].strip().lower()
                replaced = False
                for idx in seen_titles[title_key]:
                    existing_snippet = deduped[idx].node.content[:200].strip().lower()
                    if content_snippet == existing_snippet:
                        # Same content — supersede (keep newer)
                        if a.node.updated_at > deduped[idx].node.updated_at:
                            deduped[idx] = a
                        replaced = True
                        break
                if not replaced:
                    # Same title, different content — keep both
                    seen_titles[title_key].append(len(deduped))
                    deduped.append(a)
            else:
                seen_titles[title_key] = [len(deduped)]
                deduped.append(a)
        final = deduped

        elapsed_ms = (time() - start) * 1000
        return SearchResult(
            query=query,
            nodes=final[:limit],
            total_candidates=total_candidates,
            search_time_ms=elapsed_ms,
            stages_used=stages_used,
        )
