"""Tests for Hebbian learning engine."""

from __future__ import annotations

from synaptic.backends.memory import MemoryBackend
from synaptic.hebbian import MAX_WEIGHT, HebbianEngine
from synaptic.models import Edge, EdgeKind, Node, NodeKind


class TestHebbianEngine:
    async def test_reinforce_success_updates_counts(self, backend: MemoryBackend) -> None:
        engine = HebbianEngine()
        n1 = Node(title="A", kind=NodeKind.LESSON)
        n2 = Node(title="B", kind=NodeKind.LESSON)
        await backend.save_node(n1)
        await backend.save_node(n2)

        await engine.reinforce(backend, [n1.id, n2.id], success=True)

        updated = await backend.get_node(n1.id)
        assert updated is not None
        assert updated.success_count == 1
        assert updated.access_count == 1

    async def test_reinforce_failure_updates_counts(self, backend: MemoryBackend) -> None:
        engine = HebbianEngine()
        n1 = Node(title="A")
        await backend.save_node(n1)

        await engine.reinforce(backend, [n1.id], success=False)

        updated = await backend.get_node(n1.id)
        assert updated is not None
        assert updated.failure_count == 1

    async def test_reinforce_creates_edge_on_success(self, backend: MemoryBackend) -> None:
        engine = HebbianEngine()
        n1 = Node(title="A")
        n2 = Node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)

        await engine.reinforce(backend, [n1.id, n2.id], success=True)

        edges = await backend.get_edges(n1.id)
        assert len(edges) == 1
        assert edges[0].kind == EdgeKind.RELATED

    async def test_reinforce_strengthens_existing_edge(self, backend: MemoryBackend) -> None:
        engine = HebbianEngine()
        n1 = Node(title="A")
        n2 = Node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        edge = Edge(source_id=n1.id, target_id=n2.id, weight=1.0)
        await backend.save_edge(edge)

        await engine.reinforce(backend, [n1.id, n2.id], success=True)

        updated_edge = await backend.get_edges(n1.id, direction="outgoing")
        assert len(updated_edge) == 1
        assert updated_edge[0].weight > 1.0

    async def test_reinforce_weakens_on_failure(self, backend: MemoryBackend) -> None:
        engine = HebbianEngine()
        n1 = Node(title="A")
        n2 = Node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        edge = Edge(source_id=n1.id, target_id=n2.id, weight=1.0)
        await backend.save_edge(edge)

        await engine.reinforce(backend, [n1.id, n2.id], success=False)

        updated_edge = await backend.get_edges(n1.id, direction="outgoing")
        assert len(updated_edge) == 1
        assert updated_edge[0].weight < 1.0

    async def test_weight_clamp_max(self, backend: MemoryBackend) -> None:
        engine = HebbianEngine()
        n1 = Node(title="A")
        n2 = Node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        edge = Edge(source_id=n1.id, target_id=n2.id, weight=MAX_WEIGHT)
        await backend.save_edge(edge)

        await engine.reinforce(backend, [n1.id, n2.id], success=True)

        updated = await backend.get_edges(n1.id, direction="outgoing")
        assert updated[0].weight <= MAX_WEIGHT

    async def test_empty_node_ids(self, backend: MemoryBackend) -> None:
        engine = HebbianEngine()
        # Should not raise
        await engine.reinforce(backend, [], success=True)
