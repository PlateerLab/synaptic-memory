"""Tests for Personalized PageRank (PPR) engine."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.graph import SynapticGraph
from synaptic.models import EdgeKind, NodeKind
from synaptic.ppr import personalized_pagerank


@pytest.fixture
async def backend() -> MemoryBackend:
    b = MemoryBackend()
    await b.connect()
    return b


@pytest.fixture
async def graph(backend: MemoryBackend) -> SynapticGraph:
    return SynapticGraph(backend)


class TestPPRBasic:
    """Basic PPR computation tests."""

    async def test_empty_seeds(self, backend: MemoryBackend) -> None:
        """Empty seed_scores should return empty list."""
        result = await personalized_pagerank(backend, {})
        assert result == []

    async def test_single_seed_no_edges(self, graph: SynapticGraph) -> None:
        """Single seed node with no edges returns the seed itself."""
        n = await graph.add("Lonely", "No connections")
        result = await personalized_pagerank(graph.backend, {n.id: 1.0})
        assert len(result) == 1
        assert result[0][0] == n.id
        assert result[0][1] == pytest.approx(1.0)

    async def test_single_seed_with_neighbor(self, graph: SynapticGraph) -> None:
        """Single seed with one neighbor: both nodes appear in results."""
        a = await graph.add("A", "Node A")
        b = await graph.add("B", "Node B")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(graph.backend, {a.id: 1.0})
        result_dict = dict(result)

        assert a.id in result_dict
        assert b.id in result_dict
        # Seed should have higher score than neighbor
        assert result_dict[a.id] > result_dict[b.id]

    async def test_two_seeds(self, graph: SynapticGraph) -> None:
        """Two seeds connected via a bridge: all three nodes appear."""
        a = await graph.add("A", "Content A")
        bridge = await graph.add("Bridge", "Bridge node")
        b = await graph.add("B", "Content B")
        await graph.link(a.id, bridge.id, kind=EdgeKind.RELATED)
        await graph.link(bridge.id, b.id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {a.id: 1.0, b.id: 1.0}
        )
        result_ids = {nid for nid, _ in result}
        assert a.id in result_ids
        assert b.id in result_ids
        assert bridge.id in result_ids

    async def test_multiple_seeds_different_weights(self, graph: SynapticGraph) -> None:
        """Seeds with different weights produce different PPR scores."""
        a = await graph.add("A", "Important")
        b = await graph.add("B", "Less important")
        c = await graph.add("C", "Shared neighbor")
        await graph.link(a.id, c.id, kind=EdgeKind.RELATED)
        await graph.link(b.id, c.id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {a.id: 10.0, b.id: 1.0}
        )
        result_dict = dict(result)
        # A should score higher than B due to heavier seed weight
        assert result_dict[a.id] > result_dict[b.id]


class TestPPRDamping:
    """Tests for damping factor effects."""

    async def test_low_damping_favors_seeds(self, graph: SynapticGraph) -> None:
        """Low damping (high teleport) concentrates score on seeds."""
        a = await graph.add("Seed", "I am the seed")
        b = await graph.add("Neighbor", "I am a neighbor")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)

        result_low = await personalized_pagerank(
            graph.backend, {a.id: 1.0}, damping=0.1
        )
        result_high = await personalized_pagerank(
            graph.backend, {a.id: 1.0}, damping=0.95
        )

        low_dict = dict(result_low)
        high_dict = dict(result_high)

        # With low damping, seed dominates more
        seed_ratio_low = low_dict[a.id] / (low_dict[a.id] + low_dict[b.id])
        seed_ratio_high = high_dict[a.id] / (high_dict[a.id] + high_dict[b.id])
        assert seed_ratio_low > seed_ratio_high

    async def test_high_damping_spreads_more(self, graph: SynapticGraph) -> None:
        """High damping distributes more score to neighbors."""
        a = await graph.add("Seed", "Content")
        b = await graph.add("Hop1", "Content")
        c = await graph.add("Hop2", "Content")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)
        await graph.link(b.id, c.id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {a.id: 1.0}, damping=0.95
        )
        result_dict = dict(result)
        # Hop2 should have non-trivial score with high damping
        assert result_dict.get(c.id, 0.0) > 0.01


class TestPPRConvergence:
    """Tests for convergence behavior."""

    async def test_converges_with_few_iterations(self, graph: SynapticGraph) -> None:
        """PPR should converge even with max_iter=5 on a small graph."""
        a = await graph.add("A", "A")
        b = await graph.add("B", "B")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {a.id: 1.0}, max_iter=5
        )
        assert len(result) >= 2
        # Scores should be positive
        for _, score in result:
            assert score > 0.0

    async def test_tight_tolerance(self, graph: SynapticGraph) -> None:
        """Very tight tolerance still produces valid results."""
        a = await graph.add("A", "A")
        b = await graph.add("B", "B")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {a.id: 1.0}, tol=1e-12, max_iter=200
        )
        assert len(result) >= 2


class TestPPRGraph:
    """Tests for graph structure handling."""

    async def test_no_edges_graph(self, graph: SynapticGraph) -> None:
        """Multiple disconnected nodes: only seeds returned."""
        a = await graph.add("A", "A")
        b = await graph.add("B", "B")
        # No edges between them

        result = await personalized_pagerank(
            graph.backend, {a.id: 1.0}
        )
        result_dict = dict(result)
        # Only seed is returned (b is unreachable)
        assert a.id in result_dict
        assert b.id not in result_dict

    async def test_bfs_depth_limiting(self, graph: SynapticGraph) -> None:
        """BFS explores only depth 2 from seeds — nodes beyond depth 2 may not appear."""
        nodes = []
        for i in range(5):
            n = await graph.add(f"N{i}", f"Content {i}")
            nodes.append(n)
        # Chain: N0 -- N1 -- N2 -- N3 -- N4
        for i in range(4):
            await graph.link(nodes[i].id, nodes[i + 1].id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {nodes[0].id: 1.0}
        )
        result_ids = {nid for nid, _ in result}

        # N0 and N1 are within depth 1, N2 within depth 2
        assert nodes[0].id in result_ids
        assert nodes[1].id in result_ids
        assert nodes[2].id in result_ids

    async def test_cycle_graph(self, graph: SynapticGraph) -> None:
        """PPR handles cycles gracefully."""
        a = await graph.add("A", "A")
        b = await graph.add("B", "B")
        c = await graph.add("C", "C")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)
        await graph.link(b.id, c.id, kind=EdgeKind.RELATED)
        await graph.link(c.id, a.id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {a.id: 1.0}
        )
        assert len(result) == 3
        for _, score in result:
            assert score > 0.0


class TestPPRTopK:
    """Tests for top-k limiting."""

    async def test_top_k_limits_results(self, graph: SynapticGraph) -> None:
        """top_k parameter limits the number of returned results."""
        nodes = []
        for i in range(6):
            n = await graph.add(f"N{i}", f"Content {i}")
            nodes.append(n)
        # Star topology: N0 connected to all others
        for i in range(1, 6):
            await graph.link(nodes[0].id, nodes[i].id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {nodes[0].id: 1.0}, top_k=3
        )
        assert len(result) <= 3

    async def test_top_k_returns_highest_scores(self, graph: SynapticGraph) -> None:
        """top_k returns nodes with the highest PPR scores."""
        a = await graph.add("Seed", "Seed node")
        b = await graph.add("Direct", "Direct neighbor")
        c = await graph.add("Indirect", "Indirect neighbor")
        await graph.link(a.id, b.id, kind=EdgeKind.RELATED)
        await graph.link(b.id, c.id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {a.id: 1.0}, top_k=2
        )
        assert len(result) == 2
        # Results should be sorted descending
        assert result[0][1] >= result[1][1]


class TestPPREdgeWeights:
    """Tests for edge weight normalization."""

    async def test_heavier_edge_transfers_more(self, graph: SynapticGraph) -> None:
        """Higher edge weight transfers more rank to the target."""
        seed = await graph.add("Seed", "Seed")
        heavy = await graph.add("Heavy", "Heavy connection")
        light = await graph.add("Light", "Light connection")

        # Create edges with different weights
        await graph.link(seed.id, heavy.id, kind=EdgeKind.RELATED, weight=5.0)
        await graph.link(seed.id, light.id, kind=EdgeKind.RELATED, weight=1.0)

        result = await personalized_pagerank(
            graph.backend, {seed.id: 1.0}
        )
        result_dict = dict(result)

        # Heavy neighbor should receive more rank
        assert result_dict[heavy.id] > result_dict[light.id]

    async def test_zero_seed_scores(self, backend: MemoryBackend) -> None:
        """All-zero seed scores returns zero scores for all nodes."""
        result = await personalized_pagerank(
            backend, {"a": 0.0, "b": 0.0}
        )
        # Zero seeds → all scores are zero
        for _, score in result:
            assert score == 0.0

    async def test_results_sorted_descending(self, graph: SynapticGraph) -> None:
        """Results are always sorted by score in descending order."""
        nodes = []
        for i in range(4):
            n = await graph.add(f"N{i}", f"Content {i}")
            nodes.append(n)
        for i in range(3):
            await graph.link(nodes[i].id, nodes[i + 1].id, kind=EdgeKind.RELATED)

        result = await personalized_pagerank(
            graph.backend, {nodes[0].id: 1.0}
        )
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
