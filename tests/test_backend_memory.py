"""Tests for in-memory backend."""

from __future__ import annotations

from synaptic.backends.memory import MemoryBackend
from synaptic.models import ConsolidationLevel, Edge, EdgeKind, Node, NodeKind


class TestMemoryBackendNodes:
    async def test_save_and_get(self, backend: MemoryBackend) -> None:
        node = Node(title="Test", content="Content", kind=NodeKind.LESSON)
        await backend.save_node(node)
        fetched = await backend.get_node(node.id)
        assert fetched is not None
        assert fetched.title == "Test"

    async def test_get_nonexistent(self, backend: MemoryBackend) -> None:
        assert await backend.get_node("missing") is None

    async def test_update(self, backend: MemoryBackend) -> None:
        node = Node(title="Original")
        await backend.save_node(node)
        node.title = "Updated"
        await backend.update_node(node)
        fetched = await backend.get_node(node.id)
        assert fetched is not None
        assert fetched.title == "Updated"

    async def test_delete(self, backend: MemoryBackend) -> None:
        node = Node(title="ToDelete")
        await backend.save_node(node)
        await backend.delete_node(node.id)
        assert await backend.get_node(node.id) is None

    async def test_delete_cascades_edges(self, backend: MemoryBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        edge = Edge(source_id=n1.id, target_id=n2.id)
        await backend.save_edge(edge)

        await backend.delete_node(n1.id)
        edges = await backend.get_edges(n2.id)
        assert len(edges) == 0

    async def test_list_by_kind(self, backend: MemoryBackend) -> None:
        await backend.save_node(Node(title="A", kind=NodeKind.LESSON))
        await backend.save_node(Node(title="B", kind=NodeKind.RULE))
        await backend.save_node(Node(title="C", kind=NodeKind.LESSON))

        lessons = await backend.list_nodes(kind=NodeKind.LESSON)
        assert len(lessons) == 2

    async def test_list_by_level(self, backend: MemoryBackend) -> None:
        await backend.save_node(Node(title="A", level=ConsolidationLevel.L0_RAW))
        await backend.save_node(Node(title="B", level=ConsolidationLevel.L3_PERMANENT))

        l0 = await backend.list_nodes(level=ConsolidationLevel.L0_RAW)
        assert len(l0) == 1

    async def test_list_with_limit(self, backend: MemoryBackend) -> None:
        for i in range(10):
            await backend.save_node(Node(title=f"Node {i}"))
        limited = await backend.list_nodes(limit=3)
        assert len(limited) == 3


class TestMemoryBackendEdges:
    async def test_save_and_get(self, backend: MemoryBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        edge = Edge(source_id=n1.id, target_id=n2.id, kind=EdgeKind.CAUSED)
        await backend.save_edge(edge)

        edges = await backend.get_edges(n1.id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0].kind == EdgeKind.CAUSED

    async def test_direction_filter(self, backend: MemoryBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        edge = Edge(source_id=n1.id, target_id=n2.id)
        await backend.save_edge(edge)

        outgoing = await backend.get_edges(n1.id, direction="outgoing")
        incoming = await backend.get_edges(n1.id, direction="incoming")
        both = await backend.get_edges(n1.id, direction="both")
        assert len(outgoing) == 1
        assert len(incoming) == 0
        assert len(both) == 1


class TestMemoryBackendSearch:
    async def test_fts_basic(self, backend: MemoryBackend) -> None:
        await backend.save_node(Node(title="Python programming", content="Learn Python"))
        await backend.save_node(Node(title="Java basics", content="Learn Java"))

        results = await backend.search_fts("Python")
        assert len(results) == 1
        assert results[0].title == "Python programming"

    async def test_fts_korean(self, backend: MemoryBackend) -> None:
        await backend.save_node(Node(title="배포 자동화", content="CI/CD 파이프라인"))
        results = await backend.search_fts("배포")
        assert len(results) == 1

    async def test_fuzzy(self, backend: MemoryBackend) -> None:
        await backend.save_node(Node(title="Performance optimization", content="Speed up queries"))
        results = await backend.search_fuzzy("Performance")
        assert len(results) >= 1


class TestMemoryBackendTraversal:
    async def test_neighbors_depth_1(self, backend: MemoryBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        n3 = Node(title="C")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_node(n3)
        await backend.save_edge(Edge(source_id=n1.id, target_id=n2.id))
        await backend.save_edge(Edge(source_id=n2.id, target_id=n3.id))

        neighbors = await backend.get_neighbors(n1.id, depth=1)
        neighbor_ids = {n.id for n, _ in neighbors}
        assert n2.id in neighbor_ids
        assert n3.id not in neighbor_ids

    async def test_neighbors_depth_2(self, backend: MemoryBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        n3 = Node(title="C")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_node(n3)
        await backend.save_edge(Edge(source_id=n1.id, target_id=n2.id))
        await backend.save_edge(Edge(source_id=n2.id, target_id=n3.id))

        neighbors = await backend.get_neighbors(n1.id, depth=2)
        neighbor_ids = {n.id for n, _ in neighbors}
        assert n2.id in neighbor_ids
        assert n3.id in neighbor_ids


class TestMemoryBackendMaintenance:
    async def test_prune_edges(self, backend: MemoryBackend) -> None:
        n1 = Node(title="A")
        n2 = Node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_edge(Edge(source_id=n1.id, target_id=n2.id, weight=0.05))
        await backend.save_edge(Edge(source_id=n2.id, target_id=n1.id, weight=0.5))

        pruned = await backend.prune_edges(weight_below=0.1)
        assert pruned == 1

    async def test_decay_vitality(self, backend: MemoryBackend) -> None:
        node = Node(title="Test", vitality=1.0)
        await backend.save_node(node)

        count = await backend.decay_vitality(factor=0.9)
        assert count == 1

        updated = await backend.get_node(node.id)
        assert updated is not None
        assert abs(updated.vitality - 0.9) < 0.01
