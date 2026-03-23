"""Hybrid 3-stage search with Personalized PageRank."""

from __future__ import annotations

import math
from time import time

from synaptic.models import ActivatedNode, Node, NodeKind, SearchResult
from synaptic.ppr import personalized_pagerank
from synaptic.protocols import QueryRewriter, StorageBackend
from synaptic.resonance import ResonanceScorer
from synaptic.synonyms import expand_synonyms

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

    __slots__ = ("_ppr_damping", "_query_rewriter", "_scorer")

    def __init__(
        self,
        *,
        scorer: ResonanceScorer | None = None,
        query_rewriter: QueryRewriter | None = None,
        spread_decay: float = 0.25,  # deprecated, kept for compat
        spread_depth: int = 1,  # deprecated, kept for compat
        ppr_damping: float = 0.85,
    ) -> None:
        self._scorer = scorer or ResonanceScorer()
        self._query_rewriter = query_rewriter
        self._ppr_damping = ppr_damping

    async def search(
        self,
        backend: StorageBackend,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
        node_kinds: list[NodeKind] | None = None,
    ) -> SearchResult:
        start = time()
        stages_used: list[str] = []
        all_nodes: dict[str, tuple[Node, float]] = {}

        # Stage 1: FTS-primary + vector cascade
        # FTS 결과는 rank 기반 스코어 유지, vector는 FTS 미스 보완 역할만
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

            for node in vec_nodes:
                nid = node.id
                cos = vec_cosine.get(nid, 0.0)
                if nid not in fts_ids and cos >= 0.5:
                    # Vector-only: FTS에 없는 결과만 삽입, 최하위보다 낮은 스코어
                    fts_floor = 0.3 if not fts_nodes else _rank_to_score(len(fts_nodes))
                    all_nodes[nid] = (node, fts_floor * cos)

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
