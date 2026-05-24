"""Tests for the editable-graph facade: unlink, update, update_edge."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.embedder import MockEmbeddingProvider
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind


class TestUnlink:
    async def test_unlink_all_between_pair(self, graph: SynapticGraph) -> None:
        a = await graph.add("A", "x")
        b = await graph.add("B", "y")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)
        await graph.link(a.id, b.id, kind=EdgeKind.CAUSED)
        removed = await graph.unlink(a.id, b.id)
        assert removed == 2
        edges = await graph.backend.get_edges(a.id, direction="outgoing")
        assert all(e.target_id != b.id for e in edges)

    async def test_unlink_filtered_by_kind(self, graph: SynapticGraph) -> None:
        a = await graph.add("A", "x")
        b = await graph.add("B", "y")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)
        await graph.link(a.id, b.id, kind=EdgeKind.CAUSED)
        removed = await graph.unlink(a.id, b.id, kind=EdgeKind.CAUSED)
        assert removed == 1
        edges = await graph.backend.get_edges(a.id, direction="outgoing")
        kinds = [str(e.kind) for e in edges if e.target_id == b.id]
        assert kinds == ["related"]

    async def test_unlink_no_match_returns_zero(self, graph: SynapticGraph) -> None:
        a = await graph.add("A", "x")
        b = await graph.add("B", "y")
        assert await graph.unlink(a.id, b.id) == 0


class TestUpdateEdge:
    async def test_update_weight(self, graph: SynapticGraph) -> None:
        a = await graph.add("A", "x")
        b = await graph.add("B", "y")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED, weight=1.0)
        updated = await graph.update_edge(a.id, b.id, new_weight=3.0)
        assert updated == 1
        edges = await graph.backend.get_edges(a.id, direction="outgoing")
        assert edges[0].weight == 3.0

    async def test_update_kind(self, graph: SynapticGraph) -> None:
        a = await graph.add("A", "x")
        b = await graph.add("B", "y")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)
        updated = await graph.update_edge(
            a.id, b.id, new_kind=EdgeKind.CAUSED
        )
        assert updated == 1
        edges = await graph.backend.get_edges(a.id, direction="outgoing")
        assert edges[0].kind == EdgeKind.CAUSED

    async def test_update_filtered_by_kind(self, graph: SynapticGraph) -> None:
        a = await graph.add("A", "x")
        b = await graph.add("B", "y")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED, weight=1.0)
        await graph.link(a.id, b.id, kind=EdgeKind.CAUSED, weight=1.0)
        updated = await graph.update_edge(
            a.id, b.id, kind=EdgeKind.CAUSED, new_weight=5.0
        )
        assert updated == 1
        edges = await graph.backend.get_edges(a.id, direction="outgoing")
        for e in edges:
            if str(e.kind) == "caused":
                assert e.weight == 5.0
            elif str(e.kind) == "related":
                assert e.weight == 1.0

    async def test_update_noop_returns_zero(self, graph: SynapticGraph) -> None:
        a = await graph.add("A", "x")
        b = await graph.add("B", "y")
        await graph.link(a.id, b.id)
        assert await graph.update_edge(a.id, b.id) == 0


class TestMergeNodes:
    async def test_repoint_outgoing_edges(self, graph: SynapticGraph) -> None:
        keep = await graph.add("keep", "x")
        drop = await graph.add("drop", "y")
        other = await graph.add("other", "z")
        await graph.link(drop.id, other.id, kind=EdgeKind.RELATED)
        merged = await graph.merge_nodes(keep.id, drop.id)
        assert merged is not None
        assert await graph.get(drop.id) is None
        edges = await graph.backend.get_edges(keep.id, direction="outgoing")
        assert any(
            e.target_id == other.id and str(e.kind) == "related" for e in edges
        )

    async def test_repoint_incoming_edges(self, graph: SynapticGraph) -> None:
        keep = await graph.add("keep", "x")
        drop = await graph.add("drop", "y")
        other = await graph.add("other", "z")
        await graph.link(other.id, drop.id, kind=EdgeKind.RELATED)
        await graph.merge_nodes(keep.id, drop.id)
        edges = await graph.backend.get_edges(keep.id, direction="incoming")
        assert any(e.source_id == other.id for e in edges)

    async def test_dedupe_keeps_higher_weight(self, graph: SynapticGraph) -> None:
        keep = await graph.add("keep", "x")
        drop = await graph.add("drop", "y")
        other = await graph.add("other", "z")
        await graph.link(keep.id, other.id, kind=EdgeKind.RELATED, weight=1.0)
        await graph.link(drop.id, other.id, kind=EdgeKind.RELATED, weight=3.0)
        await graph.merge_nodes(keep.id, drop.id)
        edges = [
            e
            for e in await graph.backend.get_edges(keep.id, direction="outgoing")
            if e.target_id == other.id and str(e.kind) == "related"
        ]
        assert len(edges) == 1
        assert edges[0].weight == 3.0

    async def test_self_loop_dropped(self, graph: SynapticGraph) -> None:
        keep = await graph.add("keep", "x")
        drop = await graph.add("drop", "y")
        await graph.link(drop.id, keep.id, kind=EdgeKind.RELATED)
        await graph.merge_nodes(keep.id, drop.id)
        edges = await graph.backend.get_edges(keep.id, direction="outgoing")
        assert all(e.target_id != keep.id for e in edges)

    async def test_merge_tags_and_properties(self, graph: SynapticGraph) -> None:
        keep = await graph.add("keep", "x", tags=["a"])
        keep.properties["k1"] = "keep-v1"
        await graph.backend.update_node(keep)
        drop = await graph.add("drop", "y", tags=["a", "b"])
        drop.properties["k1"] = "drop-v1"  # conflict — keep wins
        drop.properties["k2"] = "drop-v2"  # fill-in
        await graph.backend.update_node(drop)

        merged = await graph.merge_nodes(keep.id, drop.id)
        assert merged is not None
        assert set(merged.tags) == {"a", "b"}
        assert merged.properties["k1"] == "keep-v1"
        assert merged.properties["k2"] == "drop-v2"

    async def test_same_id_raises(self, graph: SynapticGraph) -> None:
        n = await graph.add("x", "y")
        with pytest.raises(ValueError):
            await graph.merge_nodes(n.id, n.id)

    async def test_missing_node_returns_none(self, graph: SynapticGraph) -> None:
        n = await graph.add("x", "y")
        assert await graph.merge_nodes(n.id, "nonexistent") is None


class TestNodeUpdate:
    async def test_update_partial_fields(self, graph: SynapticGraph) -> None:
        node = await graph.add("Old Title", "Old content", tags=["a"])
        updated = await graph.update(node.id, title="New Title")
        assert updated is not None
        assert updated.title == "New Title"
        assert updated.content == "Old content"
        assert updated.tags == ["a"]

    async def test_update_missing_returns_none(self, graph: SynapticGraph) -> None:
        result = await graph.update("nonexistent", title="x")
        assert result is None


@pytest.fixture
async def graph_with_embedder() -> SynapticGraph:
    backend = MemoryBackend()
    await backend.connect()
    g = SynapticGraph(backend, embedder=MockEmbeddingProvider(dim=8))
    yield g
    await backend.close()


class TestAutoReembedOnUpdate:
    async def test_content_change_reembeds(
        self, graph_with_embedder: SynapticGraph
    ) -> None:
        node = await graph_with_embedder.add("t", "old content")
        before = list(node.embedding)
        assert before  # was embedded at add() time

        await graph_with_embedder.update(node.id, content="completely different text")
        after = await graph_with_embedder.get(node.id)
        assert after is not None
        assert after.embedding != before

    async def test_title_change_reembeds(
        self, graph_with_embedder: SynapticGraph
    ) -> None:
        node = await graph_with_embedder.add("orig title", "body")
        before = list(node.embedding)
        await graph_with_embedder.update(node.id, title="new title entirely")
        after = await graph_with_embedder.get(node.id)
        assert after is not None
        assert after.embedding != before

    async def test_explicit_embedding_wins(
        self, graph_with_embedder: SynapticGraph
    ) -> None:
        node = await graph_with_embedder.add("t", "body")
        forced = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        await graph_with_embedder.update(
            node.id, content="changed", embedding=forced
        )
        after = await graph_with_embedder.get(node.id)
        assert after is not None
        assert after.embedding == forced

    async def test_reembed_false_keeps_stale_vector(
        self, graph_with_embedder: SynapticGraph
    ) -> None:
        node = await graph_with_embedder.add("t", "body")
        before = list(node.embedding)
        await graph_with_embedder.update(
            node.id, content="changed", reembed=False
        )
        after = await graph_with_embedder.get(node.id)
        assert after is not None
        assert after.embedding == before

    async def test_kind_only_change_no_reembed(
        self, graph_with_embedder: SynapticGraph
    ) -> None:
        node = await graph_with_embedder.add("t", "body")
        before = list(node.embedding)
        await graph_with_embedder.update(node.id, kind="concept")
        after = await graph_with_embedder.get(node.id)
        assert after is not None
        assert after.embedding == before

    async def test_update_without_embedder_does_not_error(
        self, graph: SynapticGraph
    ) -> None:
        # default `graph` fixture has no embedder; update should still work.
        node = await graph.add("t", "body")
        result = await graph.update(node.id, content="new")
        assert result is not None
        assert result.content == "new"
