"""Tests for SynapticGraph facade."""

from __future__ import annotations

from synaptic.graph import SynapticGraph
from synaptic.models import DigestResult, EdgeKind, NodeKind


class TestGraphCRUD:
    async def test_add_and_get(self, graph: SynapticGraph) -> None:
        node = await graph.add("Test Node", "Test content", kind=NodeKind.LESSON)
        assert node.title == "Test Node"
        assert node.kind == NodeKind.LESSON

        fetched = await graph.get(node.id)
        assert fetched is not None
        assert fetched.title == "Test Node"

    async def test_get_nonexistent(self, graph: SynapticGraph) -> None:
        result = await graph.get("nonexistent")
        assert result is None

    async def test_remove(self, graph: SynapticGraph) -> None:
        node = await graph.add("To Remove", "Content")
        assert await graph.remove(node.id) is True
        assert await graph.get(node.id) is None

    async def test_remove_nonexistent(self, graph: SynapticGraph) -> None:
        assert await graph.remove("nonexistent") is False

    async def test_link(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("Node A", "Content A")
        n2 = await graph.add("Node B", "Content B")
        edge = await graph.link(n1.id, n2.id, kind=EdgeKind.CAUSED)
        assert edge.source_id == n1.id
        assert edge.target_id == n2.id
        assert edge.kind == EdgeKind.CAUSED


class TestGraphSearch:
    async def test_search_by_title(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("배포")
        assert len(result.nodes) > 0
        assert any("배포" in n.node.title for n in result.nodes)

    async def test_search_by_content(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("N+1")
        assert len(result.nodes) > 0

    async def test_search_empty_query(self, graph: SynapticGraph) -> None:
        result = await graph.search("")
        assert result.nodes == []

    async def test_search_no_results(self, graph: SynapticGraph) -> None:
        result = await graph.search("completely_nonexistent_term_xyz")
        assert result.nodes == []

    async def test_search_stages(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("테스트")
        assert "fts" in result.stages_used


class TestGraphReinforce:
    async def test_reinforce_success(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("Node A", "Content")
        n2 = await graph.add("Node B", "Content")
        await graph.reinforce([n1.id, n2.id], success=True)

        updated = await graph.get(n1.id)
        assert updated is not None
        assert updated.success_count >= 1

    async def test_reinforce_failure(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("Node A", "Content")
        await graph.reinforce([n1.id], success=False)

        updated = await graph.get(n1.id)
        assert updated is not None
        assert updated.failure_count >= 1


class TestGraphConsolidate:
    async def test_consolidate_without_digester(self, graph: SynapticGraph) -> None:
        result = await graph.consolidate()
        assert isinstance(result, DigestResult)


class TestGraphMaintenance:
    async def test_prune(self, graph: SynapticGraph) -> None:
        n1 = await graph.add("A", "a")
        n2 = await graph.add("B", "b")
        edge = await graph.link(n1.id, n2.id)
        # Manually set low weight
        edge.weight = 0.01
        await graph.backend.update_edge(edge)
        pruned = await graph.prune()
        assert pruned == 1

    async def test_decay(self, populated_graph: SynapticGraph) -> None:
        count = await populated_graph.decay()
        assert count > 0

    async def test_stats(self, populated_graph: SynapticGraph) -> None:
        stats = await populated_graph.stats()
        assert stats["total_nodes"] == 5
        assert stats.get("kind_lesson", 0) == 2
        assert stats.get("kind_rule", 0) == 2

    async def test_export_markdown(self, populated_graph: SynapticGraph) -> None:
        md = await populated_graph.export_markdown()
        assert "# Knowledge Graph" in md
        assert "배포 자동화" in md
