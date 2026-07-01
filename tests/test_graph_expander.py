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
        await backend.save_edge(Edge(id=eid, source_id=src, target_id=dst, kind=kind, weight=1.0))

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

    async def test_optional_timings_break_down_expansion_paths(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)
        timings: dict[str, float] = {}

        await expander.expand(
            anchors=QueryAnchors(query="규정", categories=["규정"], category_node_ids=["cat_rule"]),
            seed_nodes=[nodes["chunk_r1a"]],
            timings_ms=timings,
        )

        assert {
            "expand_graph_seed",
            "expand_graph_seed_prefetch",
            "expand_graph_references",
            "expand_graph_category",
            "expand_graph_document",
            "expand_graph_chunk_next",
            "expand_graph_entity",
            "expand_graph_related",
        }.issubset(timings)
        assert all(value >= 0 for value in timings.values())


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

        anchors = QueryAnchors(query="규정", categories=["규정"], category_node_ids=["cat_rule"])
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

        anchors = QueryAnchors(query="규정", categories=["규정"], category_node_ids=["cat_rule"])
        budget = ExpansionBudget(category_sibling_limit=1)
        results = await expander.expand(anchors=anchors, seed_nodes=[], budget=budget)
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
    async def test_default_total_cap_is_bounded(self):
        assert ExpansionBudget().max_total_expanded == 40

    async def test_total_cap_enforced(self):
        backend = MemoryBackend()
        await backend.connect()
        nodes = await _build_fixture(backend)
        expander = GraphExpander(backend=backend)

        anchors = QueryAnchors(query="규정", categories=["규정"], category_node_ids=["cat_rule"])
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


@pytest.mark.asyncio
async def test_references_expansion_surfaces_cited_document():
    """A REFERENCES edge from a seed pulls the cited document into the
    expansion with reason ``"references"`` (v0.24 WS-A)."""
    backend = MemoryBackend()
    await backend.connect()
    a = Node(id="art_a", kind=NodeKind.ENTITY, title="Article A", content="cites B")
    b = Node(id="art_b", kind=NodeKind.ENTITY, title="Article B", content="body")
    await backend.save_node(a)
    await backend.save_node(b)
    await backend.save_edge(Edge(source_id="art_a", target_id="art_b", kind=EdgeKind.REFERENCES))

    expander = GraphExpander(backend=backend)
    results = await expander.expand(anchors=QueryAnchors(query="q"), seed_nodes=[a])

    by_id = {r.node.id: r for r in results}
    assert "art_b" in by_id
    assert by_id["art_b"].reason == "references"
    assert by_id["art_b"].anchor_hit == "art_a"


@pytest.mark.asyncio
async def test_references_expansion_noop_without_edges():
    """No REFERENCES edges → expansion is unaffected (no crash, no extras)."""
    backend = MemoryBackend()
    await backend.connect()
    a = Node(id="solo", kind=NodeKind.ENTITY, title="Solo", content="x")
    await backend.save_node(a)
    expander = GraphExpander(backend=backend)
    results = await expander.expand(anchors=QueryAnchors(query="q"), seed_nodes=[a])
    assert [r.reason for r in results] == ["seed"]


@pytest.mark.asyncio
async def test_seed_edges_are_cached_across_expansion_paths():
    """A seed can be inspected by REFERENCES, document-scope, and RELATED
    paths. The per-search cache should make that one backend edge read."""

    class CountingMemoryBackend(MemoryBackend):
        def __init__(self) -> None:
            super().__init__()
            self.edge_calls: list[tuple[str, str]] = []
            self.edge_batch_calls: list[tuple[tuple[str, ...], str]] = []
            self.edge_filtered_batch_calls: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = []
            self.node_calls: list[str] = []
            self.node_batch_calls: list[tuple[str, ...]] = []

        async def get_node(self, node_id: str) -> Node | None:
            self.node_calls.append(node_id)
            return await super().get_node(node_id)

        async def get_nodes_batch(self, node_ids: list[str]) -> list[Node]:
            self.node_batch_calls.append(tuple(node_ids))
            return await super().get_nodes_batch(node_ids)

        async def get_edges(self, node_id: str, *, direction: str = "both") -> list[Edge]:
            self.edge_calls.append((node_id, direction))
            return await super().get_edges(node_id, direction=direction)

        async def get_edges_batch(
            self, node_ids: list[str], *, direction: str = "both"
        ) -> dict[str, list[Edge]]:
            self.edge_batch_calls.append((tuple(node_ids), direction))
            return await super().get_edges_batch(node_ids, direction=direction)

        async def get_edges_batch_filtered(
            self,
            node_ids: list[str],
            *,
            direction: str = "both",
            kinds: list[str | EdgeKind],
        ) -> dict[str, list[Edge]]:
            kind_values = tuple(sorted(str(kind) for kind in kinds))
            self.edge_filtered_batch_calls.append((tuple(node_ids), direction, kind_values))
            return await super().get_edges_batch_filtered(
                node_ids, direction=direction, kinds=kinds
            )

    backend = CountingMemoryBackend()
    await backend.connect()
    seed = Node(id="seed", kind=NodeKind.ENTITY, title="Seed", content="seed")
    cited = Node(id="cited", kind=NodeKind.ENTITY, title="Cited", content="cited")
    related = Node(id="related", kind=NodeKind.ENTITY, title="Related", content="related")
    for node in (seed, cited, related):
        await backend.save_node(node)
    await backend.save_edge(Edge(source_id=seed.id, target_id=cited.id, kind=EdgeKind.REFERENCES))
    await backend.save_edge(Edge(source_id=seed.id, target_id=related.id, kind=EdgeKind.RELATED))

    expander = GraphExpander(backend=backend)
    results = await expander.expand(anchors=QueryAnchors(query="q"), seed_nodes=[seed])

    assert {"cited", "related"}.issubset({r.node.id for r in results})
    assert backend.edge_calls == []
    assert backend.edge_batch_calls == []
    assert any(
        call == (("seed",), "both", (str(EdgeKind.REFERENCES),))
        for call in backend.edge_filtered_batch_calls
    )
    assert any(
        call[0] == ("seed",) and call[1] == "both" and str(EdgeKind.RELATED) in call[2]
        for call in backend.edge_filtered_batch_calls
    )
    assert backend.node_calls == []
    assert ("cited",) in backend.node_batch_calls
    assert ("related",) in backend.node_batch_calls


@pytest.mark.asyncio
async def test_document_scope_batches_candidate_node_fetches():
    """Document-scope expansion should batch-load sibling nodes instead of
    issuing one backend node read per edge."""

    class CountingMemoryBackend(MemoryBackend):
        def __init__(self) -> None:
            super().__init__()
            self.node_calls: list[str] = []
            self.node_batch_calls: list[tuple[str, ...]] = []

        async def get_node(self, node_id: str) -> Node | None:
            self.node_calls.append(node_id)
            return await super().get_node(node_id)

        async def get_nodes_batch(self, node_ids: list[str]) -> list[Node]:
            self.node_batch_calls.append(tuple(node_ids))
            return await super().get_nodes_batch(node_ids)

    backend = CountingMemoryBackend()
    await backend.connect()
    doc = Node(id="doc", kind=NodeKind.ENTITY, title="Doc", content="doc", tags=["document"])
    chunk_a = Node(id="chunk_a", kind=NodeKind.CHUNK, title="A", content="a", tags=["chunk"])
    chunk_b = Node(id="chunk_b", kind=NodeKind.CHUNK, title="B", content="b", tags=["chunk"])
    for node in (doc, chunk_a, chunk_b):
        await backend.save_node(node)
    await backend.save_edge(Edge(source_id=doc.id, target_id=chunk_a.id, kind=EdgeKind.CONTAINS))
    await backend.save_edge(Edge(source_id=doc.id, target_id=chunk_b.id, kind=EdgeKind.CONTAINS))

    expander = GraphExpander(backend=backend)
    results = await expander.expand(anchors=QueryAnchors(query="q"), seed_nodes=[doc])

    assert {"chunk_a", "chunk_b"}.issubset({r.node.id for r in results})
    assert backend.node_calls == []
    assert ("chunk_a", "chunk_b") in backend.node_batch_calls


@pytest.mark.asyncio
async def test_openie_part_of_seed_is_not_document_scope():
    backend = MemoryBackend()
    await backend.connect()
    source = Node(
        id="ent_source",
        kind=NodeKind.ENTITY,
        title="Source",
        content="source",
        tags=["_openie", "_openie_entity"],
    )
    target = Node(
        id="ent_target",
        kind=NodeKind.ENTITY,
        title="Target",
        content="target",
        tags=["_openie", "_openie_entity"],
    )
    await backend.save_node(source)
    await backend.save_node(target)
    await backend.save_edge(
        Edge(
            source_id=source.id,
            target_id=target.id,
            kind=EdgeKind.PART_OF,
            properties={"is_openie": "true", "confidence": "0.8"},
        )
    )

    expander = GraphExpander(backend=backend)
    results = await expander.expand(anchors=QueryAnchors(query="q"), seed_nodes=[source])

    by_id = {r.node.id: r for r in results}
    assert by_id["ent_target"].reason == "semantic_relation"
    assert by_id["ent_target"].edge_kind == "part_of"
    assert by_id["ent_target"].edge_confidence == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_non_openie_entity_seed_can_still_use_document_scope():
    backend = MemoryBackend()
    await backend.connect()
    phrase = Node(
        id="phrase",
        kind=NodeKind.ENTITY,
        title="Phrase",
        content="phrase",
        tags=["phrase"],
    )
    chunk = Node(id="chunk", kind=NodeKind.CHUNK, title="Chunk", content="chunk", tags=["chunk"])
    await backend.save_node(phrase)
    await backend.save_node(chunk)
    await backend.save_edge(Edge(source_id=chunk.id, target_id=phrase.id, kind=EdgeKind.CONTAINS))

    expander = GraphExpander(backend=backend)
    results = await expander.expand(anchors=QueryAnchors(query="q"), seed_nodes=[phrase])

    by_id = {r.node.id: r for r in results}
    assert by_id["chunk"].reason == "document_chunk"


# --- relevance-aware budget (opt-in) -----------------------------------


def _mk_expanded(nid: str, *, title: str = "", content: str = "", hops: int = 1):
    from synaptic.extensions.graph_expander import ExpandedNode

    return ExpandedNode(
        node=Node(id=nid, kind=NodeKind.CHUNK, title=title, content=content),
        reason="seed" if hops == 0 else "doc_sibling",
        hops=hops,
    )


def test_relevance_budget_keeps_most_relevant_over_budget():
    """When the neighbourhood exceeds the budget, the most query-relevant
    non-seed neighbours win the slots — not the first-visited ones."""
    from synaptic.extensions.graph_expander import ExpansionBudget, _ExpansionState

    st = _ExpansionState(ExpansionBudget(max_total_expanded=3), q_terms=frozenset({"e217"}))
    st.add(_mk_expanded("seed", hops=0))  # protected
    st.add(_mk_expanded("noise1", content="unrelated"))  # rel 0
    st.add(_mk_expanded("noise2", content="also unrelated"))  # rel 0, budget now full
    # newcomer mentions the query term → must evict a rel-0 noise node
    st.add(_mk_expanded("hit", title="E217 spec"))  # rel 2
    ids = {r.node.id for r in st.results()}
    assert "seed" in ids  # never evicted
    assert "hit" in ids  # relevant newcomer admitted
    assert len(ids) == 3
    assert ("noise1" in ids) != ("noise2" in ids)  # exactly one noise evicted


def test_relevance_budget_never_evicts_seeds():
    from synaptic.extensions.graph_expander import ExpansionBudget, _ExpansionState

    st = _ExpansionState(ExpansionBudget(max_total_expanded=2), q_terms=frozenset({"x"}))
    st.add(_mk_expanded("s1", hops=0))
    st.add(_mk_expanded("s2", hops=0))  # budget full with two seeds
    st.add(_mk_expanded("relevant", title="x x x"))  # rel high but only seeds held
    ids = {r.node.id for r in st.results()}
    assert ids == {"s1", "s2"}  # seeds protected, newcomer rejected


def test_empty_q_terms_preserves_first_come():
    """No query terms → original behaviour: drop the newcomer when full."""
    from synaptic.extensions.graph_expander import ExpansionBudget, _ExpansionState

    st = _ExpansionState(ExpansionBudget(max_total_expanded=2))
    st.add(_mk_expanded("a", title="anything"))
    st.add(_mk_expanded("b", title="anything"))
    st.add(_mk_expanded("c", title="very relevant whatever"))  # dropped, not evicting
    assert [r.node.id for r in st.results()] == ["a", "b"]
