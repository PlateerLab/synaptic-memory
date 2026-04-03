"""Tests for QueryDecomposer — rule-based Korean/English query decomposition."""

import pytest

from synaptic import SynapticGraph, NodeKind
from synaptic.extensions.query_decomposer import QueryDecomposer


class TestRuleDecompose:
    """Test rule-based decomposition (no LLM)."""

    @pytest.fixture
    def decomposer(self):
        return QueryDecomposer()

    async def test_korean_wa(self, decomposer):
        """와 conjunction."""
        result = await decomposer.decompose("매출과 직원수")
        assert len(result) == 2
        assert any("매출" in r for r in result)
        assert any("직원수" in r for r in result)

    async def test_korean_rang(self, decomposer):
        """랑 conjunction."""
        result = await decomposer.decompose("서비스A랑 서비스B")
        assert len(result) == 2

    async def test_korean_geurigo(self, decomposer):
        """그리고 conjunction."""
        result = await decomposer.decompose("매출 그리고 직원수")
        assert len(result) == 2

    async def test_korean_and(self, decomposer):
        """및 conjunction."""
        result = await decomposer.decompose("매출 및 직원수 변화")
        assert len(result) == 2

    async def test_english_and(self, decomposer):
        result = await decomposer.decompose("revenue and headcount")
        assert len(result) == 2

    async def test_english_vs(self, decomposer):
        result = await decomposer.decompose("PostgreSQL vs MongoDB")
        assert len(result) == 2

    async def test_single_topic_no_split(self, decomposer):
        """Single topic should not be split."""
        result = await decomposer.decompose("PostgreSQL 성능 최적화")
        assert len(result) == 1
        assert result[0] == "PostgreSQL 성능 최적화"

    async def test_short_query_no_split(self, decomposer):
        """Very short queries should not be decomposed."""
        result = await decomposer.decompose("매출")
        assert len(result) == 1

    async def test_empty_query(self, decomposer):
        result = await decomposer.decompose("")
        assert result == []

    async def test_temporal_range(self, decomposer):
        """Temporal range decomposition."""
        result = await decomposer.decompose("매출 변화 2020년부터 2024년까지")
        assert len(result) == 2
        assert any("2020" in r for r in result)
        assert any("2024" in r for r in result)

    async def test_comparison_context_preserved(self, decomposer):
        """Comparison queries should split items."""
        result = await decomposer.decompose("PostgreSQL과 MongoDB 비교")
        assert len(result) == 2


class TestDecomposerIntegration:
    """Test query decomposition integrated with SynapticGraph.search()."""

    async def test_search_with_decomposer(self):
        from synaptic.backends.memory import MemoryBackend

        decomposer = QueryDecomposer()
        graph = SynapticGraph(
            MemoryBackend(),
            query_decomposer=decomposer,
        )

        await graph.add("PostgreSQL Guide", "PostgreSQL is a relational database")
        await graph.add("MongoDB Guide", "MongoDB is a document database")
        await graph.add("Redis Guide", "Redis is an in-memory cache")

        result = await graph.search("PostgreSQL과 MongoDB")
        assert result is not None
        assert "decompose" in result.stages_used

    async def test_search_without_decomposer(self):
        """Without decomposer, search should work normally."""
        graph = SynapticGraph.memory()

        await graph.add("Test", "content about databases")
        result = await graph.search("databases")
        assert "decompose" not in result.stages_used

    async def test_simple_query_not_decomposed(self):
        """Single-topic query should not trigger decomposition."""
        from synaptic.backends.memory import MemoryBackend

        decomposer = QueryDecomposer()
        graph = SynapticGraph(
            MemoryBackend(),
            query_decomposer=decomposer,
        )

        await graph.add("PostgreSQL", "database content")
        result = await graph.search("PostgreSQL 성능")
        # Single topic → no decomposition → no "decompose" stage
        assert "decompose" not in result.stages_used
