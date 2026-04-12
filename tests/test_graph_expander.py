"""Tests for GraphExpander — shallow 1-hop expansion.

Each test builds a small Category → Document → Chunk graph in
``MemoryBackend``, runs the expander, and asserts on the set of
``ExpandedNode`` that comes back. We care about:

- Seeds are always present and tagged ``"seed"``.
- Category siblings surface documents the caller didn't start with.
- Document scope pulls chunk siblings via CONTAINS / PART_OF edges.
- NEXT_CHUNK walks the sequence.
- Budgets clamp fan-out so one path can't blow up the result set.
"""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.graph_expander import (
    ExpansionBudget,
    GraphExpander,
)
from synaptic.extensions.query_anchor import QueryAnchors
from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)

# --- Fixture helpers ---
#
# We build a tiny but realistic graph:
#
#   Category: "규정"  ←PART_OF──  Doc_R1  ─CONTAINS→  Chunk_R1a ─NEXT→ Chunk_R1b
#                    ←PART_OF──  Doc_R2  ─CONTAINS→  Chunk_R2a
#   Category: "운영"  ←PART_OF──  Doc_O1  ─CONTAINS→  Chunk_O1a
#
# Seven nodes total so the tests can enumerate what the expander sees.


async def _build_fixture(backend: MemoryBackend) -> dict[str, Node]:
    nodes: dict[str, Node] = {}

    def _make(id_: str, kind: NodeKind, title: str, tags: list[str] | None = None):
        node = Node(
            id=id_,
            kind=kind,
            title=title,
            content=title,
            tags=tags or [],
            level=ConsolidationLevel.L0_RAW,
        )
        nodes[id_] = node
        return node

    cat_rule = _make("cat_rule", NodeKind.CONCEPT, "규정", tags=["category"])
    cat_ops = _make("cat_ops", NodeKind.CONCEPT, "운영", tags=["category"])
    doc_r1 = _make("doc_r1", NodeKind.ENTITY, "규정 문서 1", tags=["document"])
    doc_r2 = _make("doc_r2", NodeKind.ENTITY, "규정 문서 2", tags=["document"])
    doc_o1 = _make("doc_o1", NodeKind.ENTITY, "운영 문서 1", tags=["document"])
    chunk_r1a = _make("chunk_r1a", NodeKind.CHUNK, "규정 문서 1 #0", tags=["chunk"])
    chunk_r1b = _make("chunk_r1b", NodeKind.CHUNK, "규정 문서 1 #1", tags=["chunk"])
    chunk_r2a = _make("chunk_r2a", NodeKind.CHUNK, "규정 문서 2 #0", tags=["chunk"])
    chunk_o1a = _make("chunk_o1a", NodeKind.CHUNK, "운영 문서 1 #0", tags=["chunk"])

    for n in nodes.values():
        await backend.save_node(n)

    async def _edge(eid: str, src: str, dst: str, kind: EdgeKind):
        await backend.save_edge(
            Edge(id=eid, source_id=src, target_id=dst, kind=kind, weight=1.0)
        )

    # PART_OF: doc → category
    await _edge("po_r1", doc_r1.id, cat_rule.id, EdgeKind.PART_OF)
    await _edge("po_r2", doc_r2.id, cat_rule.id, EdgeKind.PART_OF)
    await _edge("po_o1", doc_o1.id, cat_ops.id, EdgeKind.PART_OF)

    # CONTAINS: doc → chunk
    await _edge("co_r1a", doc_r1.id, chunk_r1a.id, EdgeKind.CONTAINS)
    await _edge("co_r1b", doc_r1.id, chunk_r1b.id, EdgeKind.CONTAINS)
    await _edge("co_r2a", doc_r2.id, chunk_r2a.id, EdgeKind.CONTAINS)
    await _edge("co_o1a", doc_o1.id, chunk_o1a.id, EdgeKind.CONTAINS)

    # NEXT_CHUNK: chunk sequence
    await _edge("nx_r1", chunk_r1a.id, chunk_r1b.id, EdgeKind.NEXT_CHUNK)

    return nodes


# --- Seeds only ---


@pytest.mark.asyncio
class TestSeedPath:
    async def test_seeds_included_as_first_entries(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(query="test")
        results = await expander.expand(
            anchors=anchors,
            seed_nodes=[nodes["chunk_r1a"]],
        )
        assert results[0].node.id == "chunk_r1a"
        assert results[0].reason == "seed"
        assert results[0].hops == 0


# --- Category sibling expansion ---


@pytest.mark.asyncio
class TestCategorySiblings:
    async def test_category_anchor_surfaces_sibling_docs(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(
            query="규정 관련",
            categories=["규정"],
            category_node_ids=["cat_rule"],
        )
        results = await expander.expand(
            anchors=anchors,
            seed_nodes=[],
        )
        ids = {r.node.id for r in results}
        # Both regulation documents surfaced via category expansion
        assert "doc_r1" in ids
        assert "doc_r2" in ids
        # Operations doc must NOT surface — different category
        assert "doc_o1" not in ids

    async def test_category_sibling_reason_tagged(self):
        backend = MemoryBackend()
        await backend.connect()
        await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(
            query="규정", categories=["규정"], category_node_ids=["cat_rule"]
        )
        results = await expander.expand(anchors=anchors, seed_nodes=[])
        doc_results = [r for r in results if r.node.id.startswith("doc_r")]
        for r in doc_results:
            assert r.reason == "category_sibling"
            assert r.hops == 1
            assert r.anchor_hit == "cat_rule"

    async def test_category_sibling_limit_respected(self):
        backend = MemoryBackend()
        await backend.connect()
        await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(
            query="규정", categories=["규정"], category_node_ids=["cat_rule"]
        )
        budget = ExpansionBudget(category_sibling_limit=1)
        results = await expander.expand(
            anchors=anchors, seed_nodes=[], budget=budget
        )
        doc_count = sum(1 for r in results if r.reason == "category_sibling")
        assert doc_count == 1


# --- Document scope ---


@pytest.mark.asyncio
class TestDocumentScope:
    async def test_seed_document_pulls_its_chunks(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(query="test")
        results = await expander.expand(
            anchors=anchors,
            seed_nodes=[nodes["doc_r1"]],
        )
        ids = {r.node.id for r in results}
        assert "chunk_r1a" in ids
        assert "chunk_r1b" in ids
        # Other document's chunks must NOT leak in
        assert "chunk_r2a" not in ids

    async def test_seed_chunk_pulls_parent_document(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(query="test")
        results = await expander.expand(
            anchors=anchors,
            seed_nodes=[nodes["chunk_r1a"]],
        )
        ids = {r.node.id for r in results}
        assert "doc_r1" in ids


# --- Chunk-next walk ---


@pytest.mark.asyncio
class TestChunkNext:
    async def test_next_chunk_surfaces(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(query="test")
        results = await expander.expand(
            anchors=anchors,
            seed_nodes=[nodes["chunk_r1a"]],
        )
        next_chunk_results = [r for r in results if r.reason == "chunk_next"]
        assert any(r.node.id == "chunk_r1b" for r in next_chunk_results)


# --- Budget cap ---


@pytest.mark.asyncio
class TestBudget:
    async def test_total_cap_enforced(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(
            query="규정", categories=["규정"], category_node_ids=["cat_rule"]
        )
        budget = ExpansionBudget(max_total_expanded=2)
        results = await expander.expand(
            anchors=anchors,
            seed_nodes=[nodes["chunk_r1a"]],
            budget=budget,
        )
        assert len(results) == 2

    async def test_no_duplicates(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        # chunk_r1a is both a seed and potentially reachable via chunk-next
        anchors = QueryAnchors(
            query="규정",
            categories=["규정"],
            category_node_ids=["cat_rule"],
        )
        results = await expander.expand(
            anchors=anchors,
            seed_nodes=[nodes["chunk_r1a"], nodes["doc_r1"]],
        )
        ids = [r.node.id for r in results]
        assert len(ids) == len(set(ids))


# --- No-op paths ---


@pytest.mark.asyncio
class TestNoOpPaths:
    async def test_empty_seeds_and_anchors_return_empty(self):
        backend = MemoryBackend()
        await backend.connect()
        await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        results = await expander.expand(
            anchors=QueryAnchors(query=""),
            seed_nodes=[],
        )
        assert results == []

    async def test_unknown_category_id_is_gracefully_skipped(self):
        backend = MemoryBackend()
        await backend.connect()
        await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(
            query="unknown",
            categories=["unknown"],
            category_node_ids=["does_not_exist"],
        )
        results = await expander.expand(anchors=anchors, seed_nodes=[])
        # No crash, just an empty expansion
        assert results == []
