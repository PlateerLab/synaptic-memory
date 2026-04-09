"""Dual-level search — LightRAG-inspired local + global retrieval.

Low-level (local): Entity/chunk search for specific questions
  → "PostgreSQL의 환불 정책은?"

High-level (global): Community summary search for abstract questions
  → "전체적으로 어떤 트렌드가 있나?"

Auto mode detects query specificity via heuristics:
  - Named entities present → local
  - Abstract/summary keywords → global
  - Default → hybrid (both levels, RRF merged)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from synaptic.models import ActivatedNode, NodeKind, SearchResult
from synaptic.search import HybridSearch, _rrf_fusion

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

# Keywords that suggest global/abstract queries
_GLOBAL_HINTS = {
    # Korean
    "전체",
    "요약",
    "개요",
    "트렌드",
    "경향",
    "패턴",
    "주요",
    "핵심",
    "전반적",
    "대략",
    "종합",
    "정리",
    "무엇이",
    "어떤",
    # English
    "overview",
    "summary",
    "trend",
    "overall",
    "general",
    "main",
    "key",
    "major",
    "pattern",
    "comprehensive",
}

# Named entity indicators (suggest local search)
_RE_SPECIFIC = re.compile(
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|"  # English proper nouns
    r"[A-Z]{2,}|"  # Abbreviations
    r"\d{4}[-년]"  # Years
)


class DualLevelSearch:
    """LightRAG-inspired dual-level retrieval.

    Example::

        search = DualLevelSearch(hybrid=HybridSearch())

        # Auto-detect level
        result = await search.search(backend, "전체 트렌드 요약")  # → global
        result = await search.search(backend, "PostgreSQL 설정")   # → local

        # Force level
        result = await search.search(backend, query, level="low")
        result = await search.search(backend, query, level="high")
    """

    __slots__ = ("_hybrid",)

    def __init__(self, hybrid: HybridSearch) -> None:
        self._hybrid = hybrid

    async def search(
        self,
        backend: StorageBackend,
        query: str,
        *,
        level: str = "auto",
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        """Search at the appropriate level.

        Args:
            backend: Storage backend.
            query: Search query.
            level: "auto", "low", "high", or "hybrid".
            limit: Max results.
            embedding: Optional query embedding.

        Returns:
            SearchResult with level info in stages_used.
        """
        if level == "auto":
            level = self._classify_query_level(query)

        if level == "low":
            return await self._search_local(backend, query, limit=limit, embedding=embedding)
        elif level == "high":
            return await self._search_global(backend, query, limit=limit, embedding=embedding)
        else:  # hybrid
            return await self._search_hybrid(backend, query, limit=limit, embedding=embedding)

    async def _search_local(
        self,
        backend: StorageBackend,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        """Local search: entity/chunk level, exclude community nodes."""
        result = await self._hybrid.search(
            backend,
            query,
            limit=limit,
            embedding=embedding,
        )
        # Filter out community nodes
        filtered = [a for a in result.nodes if a.node.kind != NodeKind.COMMUNITY]
        return SearchResult(
            query=result.query,
            nodes=filtered[:limit],
            total_candidates=result.total_candidates,
            search_time_ms=result.search_time_ms,
            stages_used=result.stages_used + ["local"],
        )

    async def _search_global(
        self,
        backend: StorageBackend,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        """Global search: community summaries only."""
        result = await self._hybrid.search(
            backend,
            query,
            limit=limit * 2,
            embedding=embedding,
            node_kinds=[NodeKind.COMMUNITY],
        )
        return SearchResult(
            query=result.query,
            nodes=result.nodes[:limit],
            total_candidates=result.total_candidates,
            search_time_ms=result.search_time_ms,
            stages_used=result.stages_used + ["global"],
        )

    async def _search_hybrid(
        self,
        backend: StorageBackend,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        """Hybrid: merge local + global results with RRF."""
        import asyncio
        from time import time as _time

        start = _time()
        local_task = self._search_local(backend, query, limit=limit, embedding=embedding)
        global_task = self._search_global(backend, query, limit=limit, embedding=embedding)
        local_result, global_result = await asyncio.gather(local_task, global_task)

        # RRF merge
        local_ranking = {a.node.id: a.resonance for a in local_result.nodes}
        global_ranking = {a.node.id: a.resonance for a in global_result.nodes}

        node_map: dict[str, ActivatedNode] = {}
        for a in local_result.nodes + global_result.nodes:
            existing = node_map.get(a.node.id)
            if existing is None or a.resonance > existing.resonance:
                node_map[a.node.id] = a

        rrf_scores = _rrf_fusion(local_ranking, global_ranking)
        max_rrf = max(rrf_scores.values()) if rrf_scores else 1.0

        merged: list[ActivatedNode] = []
        for nid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:limit]:
            an = node_map[nid]
            merged.append(
                ActivatedNode(
                    node=an.node,
                    activation=an.activation,
                    resonance=score / max_rrf if max_rrf > 0 else 0.0,
                    path=an.path,
                )
            )

        elapsed = (_time() - start) * 1000
        return SearchResult(
            query=query,
            nodes=merged,
            total_candidates=local_result.total_candidates + global_result.total_candidates,
            search_time_ms=elapsed,
            stages_used=["dual_level", "local", "global"],
        )

    def _classify_query_level(self, query: str) -> str:
        """Heuristic: classify query as local, global, or hybrid.

        Returns "low" for specific entity queries, "high" for abstract,
        "hybrid" for ambiguous.
        """
        q_lower = query.lower()

        # Check for global/abstract keywords
        global_score = sum(1 for hint in _GLOBAL_HINTS if hint in q_lower)

        # Check for specific entity indicators
        local_score = len(_RE_SPECIFIC.findall(query))

        if global_score >= 2 and local_score == 0:
            return "high"
        elif local_score >= 1 and global_score == 0:
            return "low"
        elif global_score > local_score:
            return "high"
        elif local_score > global_score:
            return "low"
        else:
            return "low"  # Default to local (more precise)
