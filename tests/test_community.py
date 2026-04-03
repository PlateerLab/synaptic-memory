"""Tests for CommunityDetector and DualLevelSearch."""

import pytest

from synaptic import SynapticGraph, NodeKind, EdgeKind
from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.community import CommunityDetector, Community
from synaptic.extensions.dual_level_search import DualLevelSearch
from synaptic.search import HybridSearch


# --- CommunityDetector ---


class TestCommunityDetector:
    @pytest.fixture
    async def graph_with_clusters(self):
        """Create a graph with two distinct clusters."""
        graph = SynapticGraph.memory()

        # Cluster 1: database related
        n1 = await graph.add("PostgreSQL", "relational database system")
        n2 = await graph.add("MySQL", "relational database")
        n3 = await graph.add("SQL Queries", "select from where")
        await graph.link(n1.id, n2.id, kind=EdgeKind.RELATED, weight=0.9)
        await graph.link(n1.id, n3.id, kind=EdgeKind.RELATED, weight=0.8)
        await graph.link(n2.id, n3.id, kind=EdgeKind.RELATED, weight=0.7)

        # Cluster 2: cache related
        n4 = await graph.add("Redis", "in-memory cache")
        n5 = await graph.add("Memcached", "distributed cache")
        n6 = await graph.add("Cache Strategy", "TTL and eviction")
        await graph.link(n4.id, n5.id, kind=EdgeKind.RELATED, weight=0.9)
        await graph.link(n4.id, n6.id, kind=EdgeKind.RELATED, weight=0.8)
        await graph.link(n5.id, n6.id, kind=EdgeKind.RELATED, weight=0.7)

        return graph

    async def test_detect_finds_communities(self, graph_with_clusters):
        graph = graph_with_clusters
        detector = CommunityDetector(min_community_size=2)

        communities = await detector.detect(graph)

        assert len(communities) >= 1
        # All communities should have at least min_size members
        for comm in communities:
            assert len(comm.member_ids) >= 2

    async def test_detect_creates_community_nodes(self, graph_with_clusters):
        graph = graph_with_clusters
        detector = CommunityDetector(min_community_size=2)

        await detector.detect(graph)

        # Should have COMMUNITY nodes in the graph
        comm_nodes = await graph.backend.list_nodes(kind=NodeKind.COMMUNITY)
        assert len(comm_nodes) >= 1
        for cn in comm_nodes:
            assert "_community" in cn.tags

    async def test_detect_extractive_summary(self, graph_with_clusters):
        graph = graph_with_clusters
        detector = CommunityDetector(min_community_size=2)

        communities = await detector.detect(graph)

        for comm in communities:
            assert comm.summary  # Should have extractive summary
            assert "관련 노드:" in comm.summary

    async def test_detect_too_few_nodes(self):
        graph = SynapticGraph.memory()
        await graph.add("Only One", "single node")

        detector = CommunityDetector(min_community_size=3)
        communities = await detector.detect(graph)
        assert communities == []

    async def test_community_data_class(self):
        comm = Community(
            id="c1",
            member_ids=["n1", "n2", "n3"],
            summary="Test summary",
            level=0,
        )
        assert len(comm.member_ids) == 3
        assert comm.summary == "Test summary"


# --- DualLevelSearch ---


class TestDualLevelSearch:
    @pytest.fixture
    def dual_search(self):
        return DualLevelSearch(hybrid=HybridSearch())

    def test_classify_local(self, dual_search):
        assert dual_search._classify_query_level("PostgreSQL 설정 방법") == "low"

    def test_classify_global(self, dual_search):
        assert dual_search._classify_query_level("전체 트렌드 요약 정리") == "high"

    def test_classify_default_local(self, dual_search):
        # Ambiguous query defaults to local
        level = dual_search._classify_query_level("데이터 처리")
        assert level == "low"

    async def test_search_local(self, dual_search):
        backend = MemoryBackend()
        from synaptic.models import Node
        await backend.save_node(
            Node(title="PostgreSQL Guide", content="database guide", kind=NodeKind.CONCEPT)
        )

        result = await dual_search.search(backend, "PostgreSQL", level="low")
        assert "local" in result.stages_used

    async def test_search_global(self, dual_search):
        backend = MemoryBackend()
        from synaptic.models import Node
        await backend.save_node(
            Node(title="Community Overview", content="overall summary", kind=NodeKind.COMMUNITY)
        )

        result = await dual_search.search(backend, "전체 요약", level="high")
        assert "global" in result.stages_used

    async def test_search_hybrid_level(self, dual_search):
        backend = MemoryBackend()
        from synaptic.models import Node
        await backend.save_node(
            Node(title="Test", content="content", kind=NodeKind.CONCEPT)
        )

        result = await dual_search.search(backend, "test query", level="hybrid")
        assert "dual_level" in result.stages_used

    async def test_search_auto(self, dual_search):
        backend = MemoryBackend()
        from synaptic.models import Node
        await backend.save_node(
            Node(title="PostgreSQL", content="database", kind=NodeKind.CONCEPT)
        )

        result = await dual_search.search(backend, "PostgreSQL 성능")
        assert result is not None

    async def test_local_filters_community_nodes(self, dual_search):
        backend = MemoryBackend()
        from synaptic.models import Node
        await backend.save_node(
            Node(title="Entity", content="real content", kind=NodeKind.ENTITY)
        )
        await backend.save_node(
            Node(title="Community 1", content="summary", kind=NodeKind.COMMUNITY)
        )

        result = await dual_search.search(backend, "content", level="low")
        for a in result.nodes:
            assert a.node.kind != NodeKind.COMMUNITY
