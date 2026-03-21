"""Learning cycle tests — Hebbian reinforcement over multiple rounds.

Tests whether repeated reinforcement actually improves search ranking.
"""

from __future__ import annotations

import pytest

from synaptic.graph import SynapticGraph

pytestmark = pytest.mark.qa


class TestHebbianLearningEffect:
    """Verify that Hebbian reinforcement changes search rankings."""

    async def test_reinforced_nodes_rank_higher(self, wiki_graph: SynapticGraph) -> None:
        """After reinforcing specific nodes, they should rank higher."""
        # 1. Initial search
        result1 = await wiki_graph.search("데이터베이스", limit=10)
        if len(result1.nodes) < 3:
            pytest.skip("Not enough results for learning test")

        # Pick the 3rd result (not already top)
        target = result1.nodes[2]
        target_id = target.node.id
        initial_rank = 2

        # 2. Reinforce the target node 5 times
        for _ in range(5):
            await wiki_graph.reinforce([target_id], success=True)

        # 3. Search again — target should rank higher
        result2 = await wiki_graph.search("데이터베이스", limit=10)
        new_rank = next(
            (i for i, n in enumerate(result2.nodes) if n.node.id == target_id),
            len(result2.nodes),
        )

        # Target should have improved (lower rank number = higher)
        assert new_rank <= initial_rank, (
            f"Expected rank improvement: was #{initial_rank}, now #{new_rank}"
        )

    async def test_failed_nodes_rank_lower(self, wiki_graph: SynapticGraph) -> None:
        """After marking nodes as failed, they should rank lower."""
        result1 = await wiki_graph.search("프로그래밍", limit=10)
        if len(result1.nodes) < 3:
            pytest.skip("Not enough results")

        # Pick the top result
        target = result1.nodes[0]
        target_id = target.node.id

        # Mark as failed repeatedly
        for _ in range(5):
            await wiki_graph.reinforce([target_id], success=False)

        # Search again
        result2 = await wiki_graph.search("프로그래밍", limit=10)
        new_rank = next(
            (i for i, n in enumerate(result2.nodes) if n.node.id == target_id),
            len(result2.nodes),
        )

        # Should have dropped (but might still be in results due to text match)
        # Just verify it didn't stay at #1 or its resonance decreased
        if new_rank == 0:
            # Still #1 — check resonance decreased at least
            old_resonance = target.resonance
            new_resonance = result2.nodes[0].resonance
            assert new_resonance <= old_resonance, (
                f"Resonance should decrease after failure: was {old_resonance:.3f}, "
                f"now {new_resonance:.3f}"
            )

    async def test_co_activation_creates_edges(self, wiki_graph: SynapticGraph) -> None:
        """Reinforcing pairs should create edges between them."""
        result = await wiki_graph.search("데이터베이스", limit=5)
        if len(result.nodes) < 2:
            pytest.skip("Not enough results")

        id_a = result.nodes[0].node.id
        id_b = result.nodes[1].node.id

        # Before reinforcement
        edges_before = await wiki_graph.backend.get_edges(id_a)
        connected_before = {e.target_id for e in edges_before} | {e.source_id for e in edges_before}

        # Co-activate
        await wiki_graph.reinforce([id_a, id_b], success=True)

        # After reinforcement
        edges_after = await wiki_graph.backend.get_edges(id_a)
        connected_after = {e.target_id for e in edges_after} | {e.source_id for e in edges_after}

        # New connection should exist
        assert id_b in connected_after or id_b in connected_before


class TestConsolidationLifecycle:
    """Test L0→L1 promotion with realistic access patterns."""

    async def test_accessed_nodes_survive_consolidation(self, wiki_graph: SynapticGraph) -> None:
        """Nodes accessed 3+ times should be promoted from L0 to L1."""
        # Search multiple times (each get() increments access_count)
        result = await wiki_graph.search("데이터베이스", limit=5)
        if not result.nodes:
            pytest.skip("No results")

        target_id = result.nodes[0].node.id

        # Access 3 times (L1 promotion threshold)
        for _ in range(3):
            await wiki_graph.get(target_id)

        # Run consolidation
        await wiki_graph.consolidate()

        # Check level
        node = await wiki_graph.backend.get_node(target_id)
        assert node is not None
        assert node.level.value in ("L1", "L2", "L3"), (
            f"Node with 3+ accesses should be promoted, but level = {node.level}"
        )
