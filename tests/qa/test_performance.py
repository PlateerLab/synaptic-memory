"""Performance tests — latency and throughput with real data."""

from __future__ import annotations

from time import time

import pytest

from synaptic.graph import SynapticGraph

pytestmark = pytest.mark.qa


class TestSearchPerformance:
    """Search latency with 100+ nodes of real data."""

    async def test_search_latency_p95(self, combined_graph: SynapticGraph) -> None:
        """95th percentile search latency should be under 100ms."""
        queries = [
            "데이터베이스",
            "프로그래밍",
            "네트워크",
            "보안",
            "알고리즘",
            "웹",
            "클라우드",
            "인공지능",
            "fix bug",
            "deploy",
            "performance",
            "refactor",
            "API",
            "테스트",
            "Python",
            "설계",
        ]
        latencies: list[float] = []

        for query in queries:
            start = time()
            await combined_graph.search(query, limit=10)
            elapsed = (time() - start) * 1000
            latencies.append(elapsed)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]
        avg = sum(latencies) / len(latencies)

        assert p95 < 100, f"P95 latency = {p95:.1f}ms, expected < 100ms"
        assert avg < 50, f"Avg latency = {avg:.1f}ms, expected < 50ms"

    async def test_batch_ingestion_throughput(self, combined_graph: SynapticGraph) -> None:
        """Should ingest at least 50 nodes/second."""
        from synaptic.models import Node, NodeKind  # noqa: PLC0415

        nodes = [
            Node(
                kind=NodeKind.CONCEPT,
                title=f"Batch node {i}",
                content=f"Content for batch node {i} about software engineering topic {i % 10}",
            )
            for i in range(100)
        ]

        start = time()
        await combined_graph.backend.save_nodes_batch(nodes)
        elapsed = time() - start

        throughput = 100 / elapsed if elapsed > 0 else float("inf")
        assert throughput > 50, f"Throughput = {throughput:.0f} nodes/sec, expected > 50"

    async def test_cache_effectiveness(self, combined_graph: SynapticGraph) -> None:
        """Cache should improve repeated access performance."""
        result = await combined_graph.search("프로그래밍", limit=5)
        if not result.nodes:
            pytest.skip("No results")

        node_id = result.nodes[0].node.id

        # First access (cache miss)
        start1 = time()
        await combined_graph.get(node_id)
        t1 = (time() - start1) * 1000

        # Second access (cache hit)
        start2 = time()
        await combined_graph.get(node_id)
        t2 = (time() - start2) * 1000

        # Cache hit should be faster (or at least not slower)
        # With MemoryBackend both are fast, but cache hit skips backend call
        assert t2 <= t1 + 1.0  # Allow 1ms tolerance


class TestGraphScale:
    """Test behavior at scale."""

    async def test_stats_with_real_data(self, combined_graph: SynapticGraph) -> None:
        """Stats should reflect the ingested data."""
        stats = await combined_graph.stats()
        total = stats["total_nodes"]
        assert total > 10, f"Expected 10+ nodes, got {total}"

    async def test_consolidation_at_scale(self, combined_graph: SynapticGraph) -> None:
        """Consolidation should complete in reasonable time."""
        start = time()
        await combined_graph.consolidate()
        elapsed = (time() - start) * 1000
        assert elapsed < 5000, f"Consolidation took {elapsed:.0f}ms, expected < 5000ms"

    async def test_find_duplicates_at_scale(self, combined_graph: SynapticGraph) -> None:
        """Duplicate detection should work with real data."""
        dupes = await combined_graph.find_duplicates(threshold=0.8, limit=10)
        # Just verify it doesn't crash and returns a list
        assert isinstance(dupes, list)
