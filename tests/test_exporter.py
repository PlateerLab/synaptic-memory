"""Tests for Markdown exporter."""

from __future__ import annotations

from synaptic.graph import SynapticGraph


class TestMarkdownExporter:
    async def test_export_empty(self, graph: SynapticGraph) -> None:
        md = await graph.export_markdown()
        assert "No nodes found" in md

    async def test_export_with_nodes(self, populated_graph: SynapticGraph) -> None:
        md = await populated_graph.export_markdown()
        assert "# Knowledge Graph" in md
        assert "배포 자동화" in md
        assert "테스트 커버리지" in md
        assert "**Level**" in md

    async def test_export_specific_nodes(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("배포")
        if result.nodes:
            node_id = result.nodes[0].node.id
            md = await populated_graph.export_markdown(node_ids=[node_id])
            assert "배포" in md

    async def test_export_grouped_by_kind(self, populated_graph: SynapticGraph) -> None:
        md = await populated_graph.export_markdown()
        # Should have section headers for different kinds
        assert "## Lesson" in md or "## lesson" in md.lower()
        assert "## Rule" in md or "## rule" in md.lower()
