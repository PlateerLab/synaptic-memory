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
from synaptic.extensions.evidence_search import EvidenceSearch
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
        await backend.save_edge(
            Edge(id=eid, source_id=src, target_id=dst, kind=kind, weight=1.0)
        )

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

    await _save(_mk("cat_rule", NodeKind.CONCEPT, "규정 및 지침", "규정 및 지침",
                    tags=["category"]))
    await _save(_mk("cat_ops", NodeKind.CONCEPT, "운영계획", "운영계획",
                    tags=["category"]))

    await _save(_mk("doc_r", NodeKind.ENTITY, "규정 문서",
                    "규정 준수 의무 관련 문서",
                    tags=["document"], category="규정 및 지침", doc_id="doc_r"))
    await _save(_mk("doc_o", NodeKind.ENTITY, "운영 문서",
                    "경마 운영계획 전반 문서",
                    tags=["document"], category="운영계획", doc_id="doc_o"))

    await _save(_mk("chunk_r1", NodeKind.CHUNK, "규정 준수 의무",
                    "규정 준수 의무 사항 명시",
                    tags=["chunk"], category="규정 및 지침", doc_id="doc_r"))
    await _save(_mk("chunk_r2", NodeKind.CHUNK, "위반 시 제재 조치",
                    "규정 위반 시 제재 조치 절차",
                    tags=["chunk"], category="규정 및 지침", doc_id="doc_r"))
    await _save(_mk("chunk_o1", NodeKind.CHUNK, "경마 운영계획 수립",
                    "경마산업 운영계획 수립 기준",
                    tags=["chunk"], category="운영계획", doc_id="doc_o"))
    await _save(_mk("chunk_o2", NodeKind.CHUNK, "예산 편성 지침",
                    "운영 예산 편성 세부 지침",
                    tags=["chunk"], category="운영계획", doc_id="doc_o"))

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
