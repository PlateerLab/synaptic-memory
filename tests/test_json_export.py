"""Tests for JSON export and node merge."""

from __future__ import annotations

import json

from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind


class TestJSONExport:
    async def test_export_empty(self, graph: SynapticGraph) -> None:
        result = await graph.export_json()
        data = json.loads(result)
        assert data["nodes"] == []
        assert data["edges"] == []

    async def test_export_with_data(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.export_json()
        data = json.loads(result)
        assert len(data["nodes"]) == 5
        assert len(data["edges"]) > 0

        # Check node structure
        node = data["nodes"][0]
        assert "id" in node
        assert "kind" in node
        assert "title" in node
        assert "tags" in node

        # Check edge structure
        edge = data["edges"][0]
        assert "source_id" in edge
        assert "target_id" in edge
        assert "kind" in edge
        assert "weight" in edge


class TestNodeMerge:
    async def test_merge_nodes(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("Node A", "Content from A", kind=NodeKind.LESSON, tags=["tag1"])
        n2 = await graph.add("Node B", "Content from B", kind=NodeKind.LESSON, tags=["tag2"])

        # Add some stats to source
        await graph.reinforce([n1.id], success=True)

        merged = await graph.merge(n1.id, n2.id)
        assert merged is not None
        assert "Content from A" in merged.content
        assert "Content from B" in merged.content
        assert "tag1" in merged.tags
        assert "tag2" in merged.tags

        # Source should be deleted
        assert await graph.get(n1.id) is None
        # Target should still exist
        assert await graph.get(n2.id) is not None

    async def test_merge_nonexistent(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("A", "a")
        result = await graph.merge(n1.id, "nonexistent")
        assert result is None

    async def test_merge_repoints_edges(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("A", "a")
        n2 = await graph.add("B", "b")
        n3 = await graph.add("C", "c")
        await graph.link(n1.id, n3.id, kind=EdgeKind.RELATED)

        await graph.merge(n1.id, n2.id)
        # n2 should now have edge to n3
        edges = await graph.backend.get_edges(n2.id)
        target_ids = {e.target_id for e in edges}
        assert n3.id in target_ids


class TestFindDuplicates:
    async def test_find_similar_titles(self, graph: SynapticGraph) -> None:
        await graph.add("배포 자동화 가이드", "content 1", kind=NodeKind.LESSON)
        await graph.add("배포 자동화 가이드라인", "content 2", kind=NodeKind.LESSON)
        await graph.add("완전히 다른 주제", "content 3", kind=NodeKind.LESSON)

        dupes = await graph.find_duplicates(threshold=0.7)
        assert len(dupes) >= 1
        titles = {dupes[0][0].title, dupes[0][1].title}
        assert "배포 자동화 가이드" in titles or "배포 자동화 가이드라인" in titles

    async def test_no_cross_kind_duplicates(self, graph: SynapticGraph) -> None:
        await graph.add("Same Title", "a", kind=NodeKind.LESSON)
        await graph.add("Same Title", "b", kind=NodeKind.RULE)

        dupes = await graph.find_duplicates(threshold=0.9)
        assert len(dupes) == 0  # Different kinds, not considered duplicates


class TestCacheIntegration:
    async def test_cache_hit(self, graph: SynapticGraph) -> None:
        n = await graph.add("Cached", "content")

        # First get — from backend (but already in cache from add)
        got = await graph.get(n.id)
        assert got is not None

        # Should be cached now
        assert graph.cache.get(n.id) is not None
        assert graph.cache.stats()["hits"] >= 1

    async def test_cache_invalidated_on_remove(self, graph: SynapticGraph) -> None:
        n = await graph.add("ToRemove", "content")
        await graph.remove(n.id)
        assert graph.cache.get(n.id) is None
