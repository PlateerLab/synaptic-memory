"""Tests for hybrid search."""

from __future__ import annotations

from synaptic.graph import SynapticGraph


class TestHybridSearch:
    async def test_fts_search(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("배포 자동화", limit=5)
        assert result.total_candidates > 0
        assert "fts" in result.stages_used

    async def test_fuzzy_search(self, populated_graph: SynapticGraph) -> None:
        # Fuzzy should catch partial matches
        result = await populated_graph.search("API 설계", limit=5)
        assert len(result.nodes) > 0

    async def test_synonym_expansion(self, populated_graph: SynapticGraph) -> None:
        # "deploy" should expand to "배포" via synonym map
        result = await populated_graph.search("deploy", limit=5)
        assert len(result.nodes) > 0 or "synonym" in result.stages_used

    async def test_spreading_activation(self, populated_graph: SynapticGraph) -> None:
        # Search for one node should discover neighbors
        result = await populated_graph.search("테스트 커버리지", limit=10)
        # Should find test coverage node + potentially neighbors
        assert len(result.nodes) >= 1

    async def test_search_time_recorded(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("보안", limit=5)
        assert result.search_time_ms > 0

    async def test_resonance_ordering(self, populated_graph: SynapticGraph) -> None:
        result = await populated_graph.search("배포", limit=5)
        if len(result.nodes) >= 2:
            # Should be sorted by resonance descending
            for i in range(len(result.nodes) - 1):
                assert result.nodes[i].resonance >= result.nodes[i + 1].resonance
