"""Agent-optimized search with intent-based strategies."""

from __future__ import annotations

from enum import StrEnum
from time import time

from synaptic.models import (
    ActivatedNode,
    EdgeKind,
    Node,
    NodeKind,
    SearchResult,
)
from synaptic.protocols import StorageBackend
from synaptic.resonance import ResonanceScorer, ResonanceWeights
from synaptic.search import HybridSearch


class SearchIntent(StrEnum):
    """What kind of knowledge the agent needs."""

    SIMILAR_DECISIONS = "similar_decisions"
    PAST_FAILURES = "past_failures"
    RELATED_RULES = "related_rules"
    REASONING_CHAIN = "reasoning_chain"
    CONTEXT_EXPLORE = "context_explore"
    GENERAL = "general"


# Weights tuned per intent
_INTENT_WEIGHTS: dict[SearchIntent, ResonanceWeights] = {
    SearchIntent.SIMILAR_DECISIONS: ResonanceWeights(
        relevance=0.45, importance=0.20, recency=0.15, vitality=0.05, context=0.15,
    ),
    SearchIntent.PAST_FAILURES: ResonanceWeights(
        relevance=0.40, importance=0.25, recency=0.20, vitality=0.05, context=0.10,
    ),
    SearchIntent.RELATED_RULES: ResonanceWeights(
        relevance=0.45, importance=0.20, recency=0.10, vitality=0.10, context=0.15,
    ),
    SearchIntent.REASONING_CHAIN: ResonanceWeights(
        relevance=0.40, importance=0.20, recency=0.20, vitality=0.05, context=0.15,
    ),
    SearchIntent.CONTEXT_EXPLORE: ResonanceWeights(
        relevance=0.30, importance=0.15, recency=0.15, vitality=0.10, context=0.30,
    ),
}


_INTENT_HINTS: dict[SearchIntent, list[str]] = {
    SearchIntent.PAST_FAILURES: [
        "실패", "에러", "오류", "버그", "장애", "incident", "fail", "error", "bug",
        "crash", "broke", "broken", "problem", "issue", "wrong", "잘못",
    ],
    SearchIntent.SIMILAR_DECISIONS: [
        "결정", "선택", "판단", "채택", "decide", "decision", "choose", "chose",
        "선택지", "대안", "alternative", "approach", "어떻게", "how to", "should",
    ],
    SearchIntent.RELATED_RULES: [
        "규칙", "정책", "규정", "제약", "금지", "필수", "rule", "policy", "constraint",
        "must", "should not", "required", "mandatory", "forbidden", "하면 안",
    ],
    SearchIntent.REASONING_CHAIN: [
        "결과", "원인", "왜", "이유", "교훈", "배운", "outcome", "result", "why",
        "because", "lesson", "learned", "led to", "caused", "이어", "때문",
    ],
    SearchIntent.CONTEXT_EXPLORE: [
        "관련", "주변", "연관", "비슷한", "explore", "related", "around", "context",
        "neighborhood", "연결", "linked",
    ],
}


def suggest_intent(query: str) -> SearchIntent:
    """Infer the most likely search intent from query keywords.

    Returns the intent with the highest keyword match count,
    or GENERAL if no keywords match.
    """
    q = query.lower()
    best_intent = SearchIntent.GENERAL
    best_score = 0
    for intent, keywords in _INTENT_HINTS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent


class AgentSearch:
    """Agent-optimized search with intent, graph awareness, and context."""

    __slots__ = ("_hybrid", "_scorer")

    def __init__(
        self,
        *,
        hybrid: HybridSearch | None = None,
        scorer: ResonanceScorer | None = None,
    ) -> None:
        self._hybrid = hybrid or HybridSearch()
        self._scorer = scorer or ResonanceScorer()

    async def search(
        self,
        backend: StorageBackend,
        query: str,
        *,
        intent: SearchIntent = SearchIntent.GENERAL,
        context_tags: list[str] | None = None,
        node_kinds: list[NodeKind] | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
        depth: int = 2,
    ) -> SearchResult:
        """Intent-aware search dispatching to specialized strategies."""
        match intent:
            case SearchIntent.SIMILAR_DECISIONS:
                return await self._search_similar_decisions(
                    backend, query, limit, embedding, context_tags,
                )
            case SearchIntent.PAST_FAILURES:
                return await self._search_past_failures(
                    backend, query, limit, context_tags,
                )
            case SearchIntent.RELATED_RULES:
                return await self._search_related_rules(
                    backend, query, limit, embedding, context_tags,
                )
            case SearchIntent.REASONING_CHAIN:
                return await self._search_reasoning_chain(
                    backend, query, limit, context_tags,
                )
            case SearchIntent.CONTEXT_EXPLORE:
                return await self._explore_context(
                    backend, query, limit, embedding, context_tags, depth,
                )
            case _:
                return await self._hybrid.search(
                    backend, query, limit=limit, embedding=embedding,
                    node_kinds=node_kinds,
                )

    async def _search_similar_decisions(
        self,
        backend: StorageBackend,
        query: str,
        limit: int,
        embedding: list[float] | None,
        context_tags: list[str] | None,
    ) -> SearchResult:
        """Find decisions on similar problems, expand to outcomes."""
        start = time()
        weights = _INTENT_WEIGHTS[SearchIntent.SIMILAR_DECISIONS]

        # Search filtered to decision nodes, fallback to unfiltered
        result = await self._hybrid.search(
            backend, query, limit=limit * 2, embedding=embedding,
            node_kinds=[NodeKind.DECISION],
        )
        if len(result.nodes) < 2:
            result = await self._hybrid.search(
                backend, query, limit=limit * 2, embedding=embedding,
            )

        # Expand: follow RESULTED_IN edges to include outcomes
        expanded: dict[str, tuple[Node, float]] = {}
        for an in result.nodes:
            expanded[an.node.id] = (an.node, an.activation)
            edges = await backend.get_edges(an.node.id, direction="outgoing")
            for edge in edges:
                if edge.kind == EdgeKind.RESULTED_IN:
                    outcome = await backend.get_node(edge.target_id)
                    if outcome and outcome.id not in expanded:
                        expanded[outcome.id] = (outcome, an.activation * 0.7)

        # Score with decision-tuned weights
        activated = self._score_candidates(expanded, weights, context_tags)
        return SearchResult(
            query=query,
            nodes=activated[:limit],
            total_candidates=len(expanded),
            search_time_ms=(time() - start) * 1000,
            stages_used=["similar_decisions", *result.stages_used],
        )

    async def _search_past_failures(
        self,
        backend: StorageBackend,
        query: str,
        limit: int,
        context_tags: list[str] | None,
    ) -> SearchResult:
        """Find failed outcomes, lessons, and their decision context."""
        start = time()
        weights = _INTENT_WEIGHTS[SearchIntent.PAST_FAILURES]

        # Search broadly — OUTCOME, DECISION, LESSON all relevant to failures
        result = await self._hybrid.search(
            backend, query, limit=limit * 3,
            node_kinds=[NodeKind.OUTCOME, NodeKind.DECISION, NodeKind.LESSON],
        )

        expanded: dict[str, tuple[Node, float]] = {}
        for an in result.nodes:
            node = an.node
            # LESSON 노드는 장애 교훈이므로 직접 포함
            if node.kind == NodeKind.LESSON:
                expanded[node.id] = (node, an.activation)
            elif node.kind == NodeKind.OUTCOME and node.failure_count > 0:
                expanded[node.id] = (node, an.activation)
                # Backtrack to decision
                edges = await backend.get_edges(node.id, direction="incoming")
                for edge in edges:
                    if edge.kind == EdgeKind.RESULTED_IN:
                        decision = await backend.get_node(edge.source_id)
                        if decision and decision.id not in expanded:
                            expanded[decision.id] = (decision, an.activation * 0.8)
            elif node.kind == NodeKind.DECISION and node.failure_count > 0:
                expanded[node.id] = (node, an.activation)

        # Also find lessons learned from failures via graph edges
        for node_id in list(expanded.keys()):
            edges = await backend.get_edges(node_id, direction="incoming")
            for edge in edges:
                if edge.kind == EdgeKind.LEARNED_FROM:
                    lesson = await backend.get_node(edge.source_id)
                    if lesson and lesson.id not in expanded:
                        expanded[lesson.id] = (lesson, 0.6)

        # If still empty, fall back to general search (no kind filter)
        if not expanded:
            result = await self._hybrid.search(
                backend, query, limit=limit * 2,
            )
            for an in result.nodes:
                expanded[an.node.id] = (an.node, an.activation)

        activated = self._score_candidates(expanded, weights, context_tags)
        return SearchResult(
            query=query,
            nodes=activated[:limit],
            total_candidates=len(expanded),
            search_time_ms=(time() - start) * 1000,
            stages_used=["past_failures"],
        )

    async def _search_related_rules(
        self,
        backend: StorageBackend,
        query: str,
        limit: int,
        embedding: list[float] | None,
        context_tags: list[str] | None,
    ) -> SearchResult:
        """Find RULE and LESSON nodes related to the query topic."""
        start = time()
        weights = _INTENT_WEIGHTS[SearchIntent.RELATED_RULES]

        result = await self._hybrid.search(
            backend, query, limit=limit * 2, embedding=embedding,
            node_kinds=[NodeKind.RULE, NodeKind.LESSON],
        )
        if len(result.nodes) < 2:
            result = await self._hybrid.search(
                backend, query, limit=limit * 2, embedding=embedding,
            )

        # Expand via graph traversal
        expanded: dict[str, tuple[Node, float]] = {}
        for an in result.nodes:
            expanded[an.node.id] = (an.node, an.activation)
            neighbors = await backend.get_neighbors(an.node.id, depth=1)
            for neighbor, edge in neighbors:
                if neighbor.kind in (NodeKind.RULE, NodeKind.LESSON, NodeKind.CONCEPT):
                    if neighbor.id not in expanded:
                        expanded[neighbor.id] = (neighbor, an.activation * edge.weight * 0.5)

        activated = self._score_candidates(expanded, weights, context_tags)
        return SearchResult(
            query=query,
            nodes=activated[:limit],
            total_candidates=len(expanded),
            search_time_ms=(time() - start) * 1000,
            stages_used=["related_rules", *result.stages_used],
        )

    async def _search_reasoning_chain(
        self,
        backend: StorageBackend,
        query: str,
        limit: int,
        context_tags: list[str] | None,
    ) -> SearchResult:
        """Traverse decision → outcome → lesson chains."""
        start = time()
        weights = _INTENT_WEIGHTS[SearchIntent.REASONING_CHAIN]

        # Find seed decisions, fallback to unfiltered
        result = await self._hybrid.search(
            backend, query, limit=limit,
            node_kinds=[NodeKind.DECISION],
        )
        if len(result.nodes) < 2:
            result = await self._hybrid.search(
                backend, query, limit=limit,
            )

        expanded: dict[str, tuple[Node, float]] = {}
        for an in result.nodes:
            expanded[an.node.id] = (an.node, an.activation)
            # Follow chain: decision → outcome → lesson
            edges = await backend.get_edges(an.node.id, direction="outgoing")
            for edge in edges:
                if edge.kind == EdgeKind.RESULTED_IN:
                    outcome = await backend.get_node(edge.target_id)
                    if outcome:
                        expanded[outcome.id] = (outcome, an.activation * 0.8)
                        # Find lessons from outcome
                        lesson_edges = await backend.get_edges(outcome.id, direction="incoming")
                        for le in lesson_edges:
                            if le.kind == EdgeKind.LEARNED_FROM:
                                lesson = await backend.get_node(le.source_id)
                                if lesson and lesson.id not in expanded:
                                    expanded[lesson.id] = (lesson, an.activation * 0.6)

        activated = self._score_candidates(expanded, weights, context_tags)
        # Preserve chain ordering: decisions first, then outcomes, then lessons
        kind_order = {NodeKind.DECISION: 0, NodeKind.OUTCOME: 1, NodeKind.LESSON: 2}
        activated.sort(
            key=lambda a: (kind_order.get(a.node.kind, 99), -a.resonance),
        )
        return SearchResult(
            query=query,
            nodes=activated[:limit],
            total_candidates=len(expanded),
            search_time_ms=(time() - start) * 1000,
            stages_used=["reasoning_chain"],
        )

    async def _explore_context(
        self,
        backend: StorageBackend,
        query: str,
        limit: int,
        embedding: list[float] | None,
        context_tags: list[str] | None,
        depth: int,
    ) -> SearchResult:
        """BFS expansion from seed nodes."""
        start = time()
        weights = _INTENT_WEIGHTS[SearchIntent.CONTEXT_EXPLORE]

        # Find seed nodes
        result = await self._hybrid.search(
            backend, query, limit=5, embedding=embedding,
        )

        expanded: dict[str, tuple[Node, float]] = {}
        for an in result.nodes:
            expanded[an.node.id] = (an.node, an.activation)

        # BFS expand
        for an in result.nodes:
            neighbors = await backend.get_neighbors(an.node.id, depth=depth)
            for neighbor, edge in neighbors:
                if neighbor.id not in expanded:
                    decay = 0.5 ** (1)  # distance-based decay
                    score = an.activation * edge.weight * decay
                    expanded[neighbor.id] = (neighbor, max(0.0, min(1.0, score)))

        activated = self._score_candidates(expanded, weights, context_tags)
        return SearchResult(
            query=query,
            nodes=activated[:limit],
            total_candidates=len(expanded),
            search_time_ms=(time() - start) * 1000,
            stages_used=["context_explore", *result.stages_used],
        )

    def _score_candidates(
        self,
        candidates: dict[str, tuple[Node, float]],
        weights: ResonanceWeights,
        context_tags: list[str] | None,
    ) -> list[ActivatedNode]:
        """Score and sort candidates."""
        now = time()
        activated: list[ActivatedNode] = []
        for _nid, (node, search_score) in candidates.items():
            resonance = self._scorer.score(
                node,
                search_score=search_score,
                now=now,
                weights=weights,
                context_tags=context_tags,
            )
            activated.append(ActivatedNode(
                node=node,
                activation=search_score,
                resonance=resonance,
                path=[],
            ))
        activated.sort(key=lambda a: a.resonance, reverse=True)
        return activated
