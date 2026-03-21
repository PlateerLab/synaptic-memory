"""Hybrid 3-stage search with spreading activation."""

from __future__ import annotations

from time import time

from synaptic.models import ActivatedNode, Node, SearchResult
from synaptic.protocols import QueryRewriter, StorageBackend
from synaptic.resonance import ResonanceScorer
from synaptic.synonyms import expand_synonyms


class HybridSearch:
    """3-stage fallback search: FTS+fuzzy → synonym expansion → query rewrite."""

    __slots__ = ("_query_rewriter", "_scorer")

    def __init__(
        self,
        *,
        scorer: ResonanceScorer | None = None,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        self._scorer = scorer or ResonanceScorer()
        self._query_rewriter = query_rewriter

    async def search(
        self,
        backend: StorageBackend,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        start = time()
        stages_used: list[str] = []
        all_nodes: dict[str, tuple[Node, float]] = {}

        # Stage 1: FTS + fuzzy + vector (parallel candidates)
        fts_nodes = await backend.search_fts(query, limit=limit * 2)
        stages_used.append("fts")
        for node in fts_nodes:
            if node.id not in all_nodes:
                all_nodes[node.id] = (node, 0.8)

        fuzzy_nodes = await backend.search_fuzzy(query, limit=limit * 2)
        stages_used.append("fuzzy")
        for node in fuzzy_nodes:
            if node.id not in all_nodes:
                all_nodes[node.id] = (node, 0.6)
            else:
                # Boost score if found in multiple stages
                existing = all_nodes[node.id]
                all_nodes[node.id] = (existing[0], min(1.0, existing[1] + 0.2))

        if embedding:
            vec_nodes = await backend.search_vector(embedding, limit=limit * 2)
            stages_used.append("vector")
            for node in vec_nodes:
                if node.id not in all_nodes:
                    all_nodes[node.id] = (node, 0.7)
                else:
                    existing = all_nodes[node.id]
                    all_nodes[node.id] = (existing[0], min(1.0, existing[1] + 0.2))

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

        # Spreading activation: expand from top candidates
        total_candidates = len(all_nodes)
        top_ids = sorted(all_nodes, key=lambda nid: all_nodes[nid][1], reverse=True)[:5]
        for nid in top_ids:
            neighbors = await backend.get_neighbors(nid, depth=1)
            for neighbor_node, edge in neighbors:
                if neighbor_node.id not in all_nodes:
                    activation = all_nodes[nid][1] * edge.weight * 0.5
                    all_nodes[neighbor_node.id] = (neighbor_node, max(0.0, min(1.0, activation)))

        # Score with resonance
        now = time()
        activated: list[ActivatedNode] = []
        for _nid, (node, search_score) in all_nodes.items():
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

        elapsed_ms = (time() - start) * 1000
        return SearchResult(
            query=query,
            nodes=activated[:limit],
            total_candidates=total_candidates,
            search_time_ms=elapsed_ms,
            stages_used=stages_used,
        )
