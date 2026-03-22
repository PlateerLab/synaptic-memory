"""Hybrid 3-stage search with Personalized PageRank."""

from __future__ import annotations

import math
from time import time

from synaptic.models import ActivatedNode, Node, NodeKind, SearchResult
from synaptic.ppr import personalized_pagerank
from synaptic.protocols import QueryRewriter, StorageBackend
from synaptic.resonance import ResonanceScorer
from synaptic.synonyms import expand_synonyms

# Kind-query 키워드 매핑 (쿼리에 이런 단어가 있으면 해당 kind 부스트)
_KIND_QUERY_HINTS: dict[NodeKind, list[str]] = {
    NodeKind.LESSON: [
        "실패", "에러", "오류", "장애", "교훈", "배운", "주의",
        "failure", "error", "incident", "lesson", "postmortem",
    ],
    NodeKind.RULE: [
        "규칙", "정책", "규정", "금지", "필수", "가이드",
        "rule", "policy", "constraint", "must", "forbidden",
    ],
    NodeKind.DECISION: [
        "결정", "선택", "판단", "채택", "어떻게",
        "decision", "choice", "decided", "approach",
    ],
    NodeKind.ARTIFACT: [
        "api", "엔드포인트", "스키마", "명세", "코드",
        "endpoint", "schema", "spec", "interface",
    ],
    NodeKind.ENTITY: [
        "회사", "조직", "제품", "서비스", "시스템",
        "company", "organization", "product", "service",
    ],
}
_KIND_BOOST = 0.05  # kind 매칭 시 search_score 부스트량 (보수적)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도."""
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

        # Stage 1: FTS + vector hybrid scoring
        fts_scores: dict[str, float] = {}
        fts_nodes = await backend.search_fts(query, limit=limit * 2)
        stages_used.append("fts")
        for rank, node in enumerate(fts_nodes):
            # FTS 순위 기반 점수: 1위=0.95, 감소율 0.05
            score = max(0.3, 0.95 - rank * 0.05)
            fts_scores[node.id] = score
            all_nodes[node.id] = (node, score)

        vec_scores: dict[str, float] = {}
        if embedding:
            vec_nodes = await backend.search_vector(embedding, limit=limit * 2)
            stages_used.append("vector")
            for rank, node in enumerate(vec_nodes):
                # Vector 순위 기반 점수 + 실제 cosine similarity 반영
                rank_score = max(0.3, 0.95 - rank * 0.05)
                # cosine similarity 직접 계산 (가능한 경우)
                if node.embedding and embedding:
                    sim = _cosine_sim(embedding, node.embedding)
                    vec_score = sim * 0.7 + rank_score * 0.3  # sim 우선
                else:
                    vec_score = rank_score
                vec_scores[node.id] = vec_score

            # FTS + vector 하이브리드 점수 합산
            alpha = 0.5  # FTS vs vector 가중치 (0.5 = 동등)
            for nid, node in {n.id: n for n in vec_nodes}.items():
                fts_s = fts_scores.get(nid, 0.0)
                vec_s = vec_scores.get(nid, 0.0)
                if nid in all_nodes:
                    # 양쪽 다 있으면 하이브리드 점수
                    hybrid = alpha * fts_s + (1 - alpha) * vec_s + 0.1  # 양쪽 매칭 보너스
                    all_nodes[nid] = (all_nodes[nid][0], min(1.0, hybrid))
                else:
                    # vector only
                    all_nodes[nid] = (node, vec_s * 0.9)  # FTS 매칭 없으면 약간 감쇠

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
                    # PPR이 새로 발견한 노드 — 그래프 경로로만 도달 가능
                    node = await backend.get_node(node_id)
                    if node:
                        all_nodes[node_id] = (node, ppr_score * 0.8)
                else:
                    # 기존 FTS 결과 — PPR로 미세 부스트만 (FTS 랭킹 보존)
                    existing = all_nodes[node_id]
                    boosted = min(1.0, existing[1] + ppr_score * 0.1)
                    if boosted > existing[1]:
                        all_nodes[node_id] = (existing[0], boosted)

        # Filter by node_kinds if specified
        if node_kinds:
            kind_set = set(node_kinds)
            all_nodes = {
                nid: (node, score)
                for nid, (node, score) in all_nodes.items()
                if node.kind in kind_set
            }

        # Kind-intent boost: 쿼리 키워드와 매칭되는 kind에 부스트
        preferred_kinds: set[NodeKind] = set()
        q_lower = query.lower()
        for kind, hints in _KIND_QUERY_HINTS.items():
            if any(h in q_lower for h in hints):
                preferred_kinds.add(kind)

        # Tag-query boost: 쿼리 키워드가 노드 태그에 있으면 부스트
        query_terms_set = set(query.lower().split())

        # Score with resonance
        now = time()
        activated: list[ActivatedNode] = []
        for _nid, (node, search_score) in all_nodes.items():
            # kind 부스트
            if preferred_kinds and node.kind in preferred_kinds:
                search_score = min(1.0, search_score + _KIND_BOOST)
            # tag 부스트 (정확 매칭만 — 2글자 이상 태그만)
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
        # Phrase nodes serve as PPR bridge nodes but should not appear in
        # user-facing search results — they carry no passage content.
        final: list[ActivatedNode] = []
        fallback: list[ActivatedNode] = []
        for a in activated:
            if "_phrase" in (a.node.tags or []):
                fallback.append(a)  # keep as last resort
            else:
                final.append(a)
        # If filtering removed too many, pad back with phrase nodes
        if len(final) < limit and fallback:
            final.extend(fallback[: limit - len(final)])

        elapsed_ms = (time() - start) * 1000
        return SearchResult(
            query=query,
            nodes=final[:limit],
            total_candidates=total_candidates,
            search_time_ms=elapsed_ms,
            stages_used=stages_used,
        )
