"""Tests for EvidenceSearch — end-to-end over a tiny in-memory graph.

We build the same fixture used by the GraphExpander tests but add
content to chunk nodes so the FTS seed step and the aggregator's
Jaccard dedup both have something to work with. Each test asserts
on a specific stage of the pipeline so a regression points at the
offending module quickly.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.evidence_search import (
    EvidenceSearch,
    _aggregate_candidate_pool_limit,
    _bounded_ppr_seed_scores,
)
from synaptic.extensions.graph_expander import ExpansionBudget
from synaptic.indexing import (
    CandidateScoreSource,
    CandidateSearchRequest,
    CandidateSearchResult,
    IndexLagReport,
)
from synaptic.indexing import (
    ScoredCandidate as RoutedCandidate,
)
from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)


async def _seed_graph(backend: MemoryBackend) -> None:
    """Build a minimal Category→Doc→Chunk graph with distinct content.

    Layout::

        Category: "규정"    ← Doc_R ← Chunk_R1 (규정 준수 의무)
                                   ← Chunk_R2 (위반 시 제재 조치)
        Category: "운영"    ← Doc_O ← Chunk_O1 (경마 운영계획 수립)
                                   ← Chunk_O2 (예산 편성 지침)

    Four chunks in total, all with different content so FTS + MMR
    both have something to discriminate.
    """

    async def _save(node: Node):
        await backend.save_node(node)

    async def _edge(eid: str, src: str, dst: str, kind: EdgeKind):
        await backend.save_edge(Edge(id=eid, source_id=src, target_id=dst, kind=kind, weight=1.0))

    def _mk(
        id_: str,
        kind: NodeKind,
        title: str,
        content: str,
        *,
        tags: list[str],
        category: str = "",
        doc_id: str = "",
    ):
        props: dict[str, str] = {}
        if category:
            props["category"] = category
        if doc_id:
            props["doc_id"] = doc_id
        return Node(
            id=id_,
            kind=kind,
            title=title,
            content=content,
            tags=tags,
            properties=props,
            level=ConsolidationLevel.L0_RAW,
        )

    await _save(
        _mk("cat_rule", NodeKind.CONCEPT, "규정 및 지침", "규정 및 지침", tags=["category"])
    )
    await _save(_mk("cat_ops", NodeKind.CONCEPT, "운영계획", "운영계획", tags=["category"]))

    await _save(
        _mk(
            "doc_r",
            NodeKind.ENTITY,
            "규정 문서",
            "규정 준수 의무 관련 문서",
            tags=["document"],
            category="규정 및 지침",
            doc_id="doc_r",
        )
    )
    await _save(
        _mk(
            "doc_o",
            NodeKind.ENTITY,
            "운영 문서",
            "경마 운영계획 전반 문서",
            tags=["document"],
            category="운영계획",
            doc_id="doc_o",
        )
    )

    await _save(
        _mk(
            "chunk_r1",
            NodeKind.CHUNK,
            "규정 준수 의무",
            "규정 준수 의무 사항 명시",
            tags=["chunk"],
            category="규정 및 지침",
            doc_id="doc_r",
        )
    )
    await _save(
        _mk(
            "chunk_r2",
            NodeKind.CHUNK,
            "위반 시 제재 조치",
            "규정 위반 시 제재 조치 절차",
            tags=["chunk"],
            category="규정 및 지침",
            doc_id="doc_r",
        )
    )
    await _save(
        _mk(
            "chunk_o1",
            NodeKind.CHUNK,
            "경마 운영계획 수립",
            "경마산업 운영계획 수립 기준",
            tags=["chunk"],
            category="운영계획",
            doc_id="doc_o",
        )
    )
    await _save(
        _mk(
            "chunk_o2",
            NodeKind.CHUNK,
            "예산 편성 지침",
            "운영 예산 편성 세부 지침",
            tags=["chunk"],
            category="운영계획",
            doc_id="doc_o",
        )
    )

    # PART_OF: doc → category
    await _edge("po_r", "doc_r", "cat_rule", EdgeKind.PART_OF)
    await _edge("po_o", "doc_o", "cat_ops", EdgeKind.PART_OF)

    # CONTAINS: doc → chunk
    await _edge("co_r1", "doc_r", "chunk_r1", EdgeKind.CONTAINS)
    await _edge("co_r2", "doc_r", "chunk_r2", EdgeKind.CONTAINS)
    await _edge("co_o1", "doc_o", "chunk_o1", EdgeKind.CONTAINS)
    await _edge("co_o2", "doc_o", "chunk_o2", EdgeKind.CONTAINS)

    # NEXT_CHUNK sequences
    await _edge("nx_r", "chunk_r1", "chunk_r2", EdgeKind.NEXT_CHUNK)
    await _edge("nx_o", "chunk_o1", "chunk_o2", EdgeKind.NEXT_CHUNK)


# --- Single-category query ---


@pytest.mark.asyncio
class TestSingleCategoryQuery:
    async def test_query_with_one_category_returns_matching_evidence(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        result = await searcher.search("규정 준수", k=4)

        assert "규정 및 지침" in result.anchors.categories
        # At least one evidence node comes from the rule category
        cats = {e.category for e in result.evidence}
        assert "규정 및 지침" in cats

    async def test_anchors_populated(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        result = await searcher.search("규정 및 지침 준수 의무")

        assert result.anchors.query
        assert result.anchors.categories
        assert result.anchors.keywords


# --- Multi-category (cross-document) query ---


@pytest.mark.asyncio
class TestMultiCategoryQuery:
    async def test_category_coverage_delivers_both_sides(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        result = await searcher.search(
            "규정 및 지침과 운영계획의 관계",
            k=4,
        )

        # Both categories should have been detected in anchors
        assert "규정 및 지침" in result.anchors.categories
        assert "운영계획" in result.anchors.categories

        # And both should have at least one representative in the evidence
        evidence_cats = {e.category for e in result.evidence}
        assert "규정 및 지침" in evidence_cats
        assert "운영계획" in evidence_cats


# --- Pipeline shape ---


@pytest.mark.asyncio
class TestPipelineShape:
    async def test_elapsed_time_recorded(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        result = await searcher.search("규정")
        assert result.elapsed_ms > 0
        assert result.timings_ms
        assert result.diagnostics
        assert {
            "anchor",
            "fts",
            "expand",
            "expand_graph",
            "expand_graph_seed",
            "expand_graph_seed_prefetch",
            "expand_graph_references",
            "expand_graph_category",
            "expand_graph_document",
            "expand_graph_chunk_next",
            "expand_graph_entity",
            "expand_graph_related",
            "expand_ppr",
            "rerank",
            "aggregate",
        }.issubset(set(result.timings_ms))
        assert {
            "seed_count",
            "expanded_count_before_ppr",
            "ppr_seed_cap",
            "ppr_seed_count",
            "ppr_result_count",
            "ppr_missing_count",
            "ppr_added_count",
            "ppr_skipped_saturated",
            "expanded_count",
            "scored_count",
            "evidence_count",
        }.issubset(set(result.diagnostics))

    async def test_cross_rerank_top_n_env_controls_candidate_count(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class _CountingReranker:
            def __init__(self) -> None:
                self.count = 0

            async def rerank(self, query: str, documents: list[str]) -> list[float]:
                self.count = len(documents)
                return [0.0] * len(documents)

        backend = MemoryBackend()
        await backend.connect()
        for i in range(30):
            await backend.save_node(
                Node(
                    id=f"n{i}",
                    kind=NodeKind.CONCEPT,
                    title=f"Alpha {i}",
                    content=f"alpha candidate {i}",
                )
            )
        reranker = _CountingReranker()
        monkeypatch.setenv("SYNAPTIC_CROSS_RERANK_TOP_N", "25")

        searcher = EvidenceSearch(
            backend=backend,
            reranker=reranker,
            graph_expansion=False,
            aggregate_candidate_pool_limit=0,
        )
        await searcher.search("alpha", k=5, fts_seed_limit=30)

        assert reranker.count == 25

    async def test_invalid_cross_rerank_top_n_env_falls_back_to_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        class _CountingReranker:
            def __init__(self) -> None:
                self.count = 0

            async def rerank(self, query: str, documents: list[str]) -> list[float]:
                self.count = len(documents)
                return [0.0] * len(documents)

        backend = MemoryBackend()
        await backend.connect()
        for i in range(30):
            await backend.save_node(
                Node(
                    id=f"n{i}",
                    kind=NodeKind.CONCEPT,
                    title=f"Beta {i}",
                    content=f"beta candidate {i}",
                )
            )
        reranker = _CountingReranker()
        monkeypatch.setenv("SYNAPTIC_CROSS_RERANK_TOP_N", "not-an-int")

        searcher = EvidenceSearch(
            backend=backend,
            reranker=reranker,
            graph_expansion=False,
            aggregate_candidate_pool_limit=0,
        )
        await searcher.search("beta", k=5, fts_seed_limit=30)

        assert reranker.count == 20

    async def test_fts_embedding_materialization_tracks_query_embedding_need(self):
        class CountingBackend(MemoryBackend):
            def __init__(self) -> None:
                super().__init__()
                self.include_embedding_calls: list[bool] = []

            async def search_fts(
                self,
                query: str,
                *,
                limit: int = 20,
                include_embedding: bool = True,
            ) -> list[Node]:
                self.include_embedding_calls.append(include_embedding)
                return await super().search_fts(query, limit=limit)

        backend = CountingBackend()
        await backend.connect()
        await backend.save_node(
            Node(
                id="alpha",
                kind=NodeKind.CHUNK,
                title="alpha",
                content="alpha content",
                embedding=[0.1, 0.2],
            )
        )

        searcher = EvidenceSearch(backend=backend)
        await searcher.search("alpha", k=1)
        assert backend.include_embedding_calls[0] is False

        backend.include_embedding_calls.clear()
        await searcher.search("alpha", k=1, query_embedding=[0.1, 0.2])
        assert backend.include_embedding_calls[0] is True

    async def test_index_router_seeds_bypass_backend_fts(self):
        class CountingBackend(MemoryBackend):
            def __init__(self) -> None:
                super().__init__()
                self.fts_calls = 0

            async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
                self.fts_calls += 1
                return await super().search_fts(query, limit=limit)

        class FakeRouter:
            def __init__(self) -> None:
                self.requests: list[CandidateSearchRequest] = []

            async def search_candidates(
                self,
                request: CandidateSearchRequest,
            ) -> CandidateSearchResult:
                self.requests.append(request)
                return CandidateSearchResult(
                    candidates=[
                        RoutedCandidate(
                            node_id="alpha",
                            score=0.77,
                            score_source=CandidateScoreSource.LEXICAL,
                            rank=1,
                            query_variant=request.query,
                            provider="fake",
                        )
                    ],
                    provider_counts={"fake": 1},
                    score_ranges={"fake": 0.77},
                )

            async def index_health(self) -> list[IndexLagReport]:
                return [IndexLagReport(provider="fake", status="ok")]

        backend = CountingBackend()
        await backend.connect()
        await backend.save_node(
            Node(
                id="alpha",
                kind=NodeKind.CHUNK,
                title="alpha",
                content="alpha routed content",
            )
        )
        router = FakeRouter()

        searcher = EvidenceSearch(backend=backend, index_router=router, graph_expansion=False)
        result = await searcher.search("alpha", k=1)

        assert backend.fts_calls == 0
        assert router.requests[0].query == "alpha"
        assert result.evidence[0].node.id == "alpha"
        assert result.diagnostics["router_candidate_count"] == 1.0
        assert result.diagnostics["router_provider_fake_count"] == 1.0

    async def test_ppr_seed_scores_are_bounded_by_relevance(self):
        scores = {f"n{i}": float(i) for i in range(80)}

        bounded = _bounded_ppr_seed_scores(scores, k=6)

        assert len(bounded) == 32
        assert "n79" in bounded
        assert "n48" in bounded
        assert "n47" not in bounded

    async def test_aggregate_candidate_pool_limit_scales_with_k(self):
        assert _aggregate_candidate_pool_limit(6) == 24
        assert _aggregate_candidate_pool_limit(30) == 30
        assert _aggregate_candidate_pool_limit(60) == 60

    async def test_aggregate_candidate_pool_limit_is_recorded(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend, aggregate_candidate_pool_limit=7)
        result = await searcher.search("규정", k=2)

        assert result.diagnostics["aggregate_pool_limit"] == 7.0

    async def test_aggregate_candidate_pool_limit_default_is_recorded(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        result = await EvidenceSearch(backend=backend).search("규정", k=40)

        assert result.diagnostics["aggregate_pool_limit"] == 40.0

    async def test_aggregate_candidate_pool_limit_env_wins(self, monkeypatch):
        monkeypatch.setenv("SYNAPTIC_AGGREGATE_CANDIDATE_POOL_LIMIT", "13")
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend, aggregate_candidate_pool_limit=7)
        result = await searcher.search("규정", k=40)

        assert result.diagnostics["aggregate_pool_limit"] == 13.0

    async def test_expanded_larger_than_seeds(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        result = await searcher.search("규정 운영계획", k=4)
        # Expansion should have surfaced at least as many candidates as seeds
        assert len(result.expanded) >= len(result.seeds)

    async def test_k_bounds_evidence_set(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        result = await searcher.search("규정", k=2)
        assert len(result.evidence) <= 2

    async def test_empty_query_returns_empty_evidence(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        result = await searcher.search("", k=4)
        assert result.evidence == []

    async def test_openie_semantic_relation_surfaces_relation_only_target(self):
        backend = MemoryBackend()
        await backend.connect()
        await backend.save_node(
            Node(
                id="ent_acme",
                kind=NodeKind.ENTITY,
                title="Acme",
                content="Acme depends on an external plan.",
                tags=["_openie", "_openie_entity"],
            )
        )
        await backend.save_node(
            Node(
                id="ent_roadmap",
                kind=NodeKind.ENTITY,
                title="Roadmap",
                content="Release calendar and milestones.",
                tags=["_openie", "_openie_entity"],
            )
        )
        for idx in range(4):
            await backend.save_node(
                Node(
                    id=f"chunk_distractor_{idx}",
                    kind=NodeKind.CHUNK,
                    title=f"Acme depends distractor {idx}",
                    content="Acme depends appears here lexically, but no relation target lives here.",
                    tags=["chunk"],
                )
            )
        await backend.save_edge(
            Edge(
                id="openie_acme_depends_roadmap",
                source_id="ent_acme",
                target_id="ent_roadmap",
                kind=EdgeKind.DEPENDS_ON,
                weight=0.9,
                properties={"is_openie": "true", "relation": "depends_on", "confidence": "0.9"},
            )
        )

        no_graph = await EvidenceSearch(backend=backend, graph_expansion=False).search(
            "Acme depends",
            k=3,
        )
        assert "ent_roadmap" not in {item.node.id for item in no_graph.expanded}

        with_graph = await EvidenceSearch(
            backend=backend,
            expansion_budget=ExpansionBudget(max_total_expanded=10),
        ).search("Acme depends", k=3)

        expanded_reasons = {item.node.id: item.reason for item in with_graph.expanded}
        assert expanded_reasons["ent_roadmap"] == "semantic_relation"
        relation_expansion = next(
            item for item in with_graph.expanded if item.node.id == "ent_roadmap"
        )
        assert relation_expansion.edge_kind == "depends_on"
        assert relation_expansion.edge_confidence == pytest.approx(0.9)
        assert "ent_roadmap" in {item.node.id for item in with_graph.evidence}

    async def test_ppr_discovery_stays_one_hop_from_search_seed(self):
        backend = MemoryBackend()
        await backend.connect()
        seed = Node(
            id="seed",
            kind=NodeKind.CHUNK,
            title="seedonly",
            content="seedonly lexical anchor",
        )
        direct = Node(
            id="direct",
            kind=NodeKind.CHUNK,
            title="direct",
            content="direct graph neighbour",
        )
        indirect = Node(
            id="indirect",
            kind=NodeKind.CHUNK,
            title="indirect",
            content="indirect second layer",
        )
        await backend.save_node(seed)
        await backend.save_node(direct)
        await backend.save_node(indirect)
        await backend.save_edge(
            Edge(
                id="seed_direct",
                source_id=seed.id,
                target_id=direct.id,
                kind=EdgeKind.RELATED,
            )
        )
        await backend.save_edge(
            Edge(
                id="direct_indirect",
                source_id=direct.id,
                target_id=indirect.id,
                kind=EdgeKind.RELATED,
            )
        )

        result = await EvidenceSearch(backend=backend).search("seedonly", k=3)

        expanded_reasons = {item.node.id: item.reason for item in result.expanded}
        assert expanded_reasons["direct"] == "ppr_discovery"
        assert "indirect" not in expanded_reasons

    async def test_ppr_result_count_is_bounded_to_k(self):
        backend = MemoryBackend()
        await backend.connect()
        seed = Node(
            id="seed",
            kind=NodeKind.CHUNK,
            title="seedonly",
            content="seedonly lexical anchor",
        )
        await backend.save_node(seed)
        for idx in range(6):
            node = Node(
                id=f"direct_{idx}",
                kind=NodeKind.CHUNK,
                title=f"direct {idx}",
                content=f"graph neighbour {idx}",
            )
            await backend.save_node(node)
            await backend.save_edge(
                Edge(
                    id=f"seed_direct_{idx}",
                    source_id=seed.id,
                    target_id=node.id,
                    kind=EdgeKind.RELATED,
                )
            )

        result = await EvidenceSearch(backend=backend).search("seedonly", k=3)

        assert result.diagnostics["ppr_result_count"] == 3.0
        assert result.diagnostics["ppr_missing_count"] <= 3.0

    async def test_ppr_skips_when_expanded_saturates_aggregate_pool(self):
        backend = MemoryBackend()
        await backend.connect()
        seed = Node(
            id="seed",
            kind=NodeKind.CHUNK,
            title="seedonly",
            content="seedonly lexical anchor",
        )
        direct = Node(
            id="direct",
            kind=NodeKind.CHUNK,
            title="direct",
            content="direct graph neighbour",
        )
        await backend.save_node(seed)
        await backend.save_node(direct)
        await backend.save_edge(
            Edge(
                id="seed_direct",
                source_id=seed.id,
                target_id=direct.id,
                kind=EdgeKind.RELATED,
            )
        )

        result = await EvidenceSearch(
            backend=backend,
            aggregate_candidate_pool_limit=1,
        ).search("seedonly", k=3)

        assert result.diagnostics["ppr_skipped_saturated"] == 1.0
        assert result.diagnostics["ppr_result_count"] == 0.0
        assert "direct" not in {item.node.id for item in result.expanded}

    async def test_openie_entity_seed_does_not_crowd_source_evidence(self):
        backend = MemoryBackend()
        await backend.connect()
        await backend.save_node(
            Node(
                id="doc_record",
                kind=NodeKind.ENTITY,
                title="Record management plan document",
                content="Annual record management plan under public records law.",
                tags=["document"],
                properties={"doc_id": "doc_record"},
            )
        )
        await backend.save_node(
            Node(
                id="ent_record_plan",
                kind=NodeKind.ENTITY,
                title="Record management plan",
                content="Record management plan record management plan.",
                tags=["_openie", "_openie_entity"],
            )
        )
        await backend.save_edge(
            Edge(
                id="mention_record_plan",
                source_id="doc_record",
                target_id="ent_record_plan",
                kind=EdgeKind.MENTIONS,
                properties={"is_openie": "true"},
            )
        )

        result = await EvidenceSearch(backend=backend).search("record management plan", k=2)

        assert "ent_record_plan" in set(result.seeds)
        evidence_ids = {item.node.id for item in result.evidence}
        assert "doc_record" in evidence_ids
        assert "ent_record_plan" not in evidence_ids


# --- Per-document cap end-to-end ---


@pytest.mark.asyncio
class TestPerDocumentCapE2E:
    async def test_cap_enforced_through_full_pipeline(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        # Force only rule docs to match and check that we don't take
        # more than cap=1 from doc_r even though both chunks qualify.
        result = await searcher.search("규정 위반", k=4, per_document_cap=1)
        rule_docs = [e for e in result.evidence if e.document_id == "doc_r"]
        assert len(rule_docs) <= 1


# --- Query decomposer integration ---


class _StubDecomposer:
    """Deterministic decomposer for integration tests.

    Returns the preset list on any input, so tests can verify that
    EvidenceSearch reacts to a >1-element decomposition without coupling
    to the rule-based decomposer's pattern library.
    """

    def __init__(self, subs: list[str]) -> None:
        self._subs = subs

    async def decompose(self, query: str) -> list[str]:
        return self._subs


@pytest.mark.asyncio
class TestDecomposerIntegration:
    async def test_no_decomposer_leaves_sub_queries_empty(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(backend=backend)
        result = await searcher.search("규정 운영계획", k=4)
        assert result.sub_queries == []

    async def test_single_subquery_treated_as_no_decomposition(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        searcher = EvidenceSearch(
            backend=backend,
            decomposer=_StubDecomposer(["규정 운영계획"]),
        )
        result = await searcher.search("규정 운영계획", k=4)
        # len==1 means "not decomposable" → no fusion attempted.
        assert result.sub_queries == []

    async def test_multi_subquery_surfaces_both_categories(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        # Deliberately use a compound query whose terms alone don't hit
        # both categories; the sub-queries do.
        searcher = EvidenceSearch(
            backend=backend,
            decomposer=_StubDecomposer(["규정 준수", "경마 운영계획"]),
        )
        result = await searcher.search("두 영역을 함께 살펴보자", k=6)

        assert result.sub_queries == ["규정 준수", "경마 운영계획"]
        evidence_cats = {e.category for e in result.evidence}
        # RRF fusion over sub-query FTS results should bring both sides.
        assert "규정 및 지침" in evidence_cats
        assert "운영계획" in evidence_cats

    async def test_decomposer_failure_is_non_fatal(self):
        backend = MemoryBackend()
        await backend.connect()
        await _seed_graph(backend)

        class _BrokenDecomposer:
            async def decompose(self, query: str) -> list[str]:
                raise RuntimeError("boom")

        searcher = EvidenceSearch(backend=backend, decomposer=_BrokenDecomposer())
        result = await searcher.search("규정", k=4)
        # Pipeline must degrade to the original-query path.
        assert result.sub_queries == []
        assert result.evidence
