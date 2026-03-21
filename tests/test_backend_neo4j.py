"""Neo4j backend integration tests.

Requires running Neo4j: docker compose up neo4j
Run with: pytest tests/test_backend_neo4j.py -v -m neo4j
"""

import pytest

from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
)

try:
    from synaptic.backends.neo4j import Neo4jBackend

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

pytestmark = [
    pytest.mark.neo4j,
    pytest.mark.skipif(not HAS_NEO4J, reason="neo4j driver not installed"),
]


@pytest.fixture
async def backend():
    """Connect to Neo4j, clear data, yield backend, then clean up."""
    b = Neo4jBackend("bolt://localhost:7687", auth=("neo4j", "password"))
    try:
        await b.connect()
    except Exception:
        pytest.skip("Neo4j server not available")
    await b.clear_all()
    yield b
    await b.clear_all()
    await b.close()


def _make_node(**kwargs) -> Node:
    defaults = {"title": "Test", "content": "Test content"}
    defaults.update(kwargs)
    return Node(**defaults)


def _make_edge(src: str, tgt: str, **kwargs) -> Edge:
    defaults = {"source_id": src, "target_id": tgt}
    defaults.update(kwargs)
    return Edge(**defaults)


# --- Node CRUD ---


class TestNodeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, backend: Neo4jBackend) -> None:
        node = _make_node(title="Hello", content="World")
        await backend.save_node(node)
        got = await backend.get_node(node.id)
        assert got is not None
        assert got.title == "Hello"
        assert got.content == "World"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, backend: Neo4jBackend) -> None:
        got = await backend.get_node("nonexistent")
        assert got is None

    @pytest.mark.asyncio
    async def test_update(self, backend: Neo4jBackend) -> None:
        node = _make_node(title="Original")
        await backend.save_node(node)
        node.title = "Updated"
        node.access_count = 5
        await backend.update_node(node)
        got = await backend.get_node(node.id)
        assert got is not None
        assert got.title == "Updated"
        assert got.access_count == 5

    @pytest.mark.asyncio
    async def test_delete(self, backend: Neo4jBackend) -> None:
        node = _make_node()
        await backend.save_node(node)
        await backend.delete_node(node.id)
        assert await backend.get_node(node.id) is None

    @pytest.mark.asyncio
    async def test_list_nodes(self, backend: Neo4jBackend) -> None:
        await backend.save_node(_make_node(kind=NodeKind.CONCEPT, title="A"))
        await backend.save_node(_make_node(kind=NodeKind.DECISION, title="B"))
        await backend.save_node(_make_node(kind=NodeKind.CONCEPT, title="C"))

        all_nodes = await backend.list_nodes()
        assert len(all_nodes) == 3

        concepts = await backend.list_nodes(kind=NodeKind.CONCEPT)
        assert len(concepts) == 2

    @pytest.mark.asyncio
    async def test_properties_roundtrip(self, backend: Neo4jBackend) -> None:
        node = _make_node(
            kind=NodeKind.TOOL_CALL,
            properties={"tool_name": "search", "success": "true"},
        )
        await backend.save_node(node)
        got = await backend.get_node(node.id)
        assert got is not None
        assert got.properties["tool_name"] == "search"

    @pytest.mark.asyncio
    async def test_tags_roundtrip(self, backend: Neo4jBackend) -> None:
        node = _make_node(tags=["deploy", "ci/cd", "한국어"])
        await backend.save_node(node)
        got = await backend.get_node(node.id)
        assert got is not None
        assert got.tags == ["deploy", "ci/cd", "한국어"]


# --- Edge CRUD ---


class TestEdgeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get_edges(self, backend: Neo4jBackend) -> None:
        n1 = _make_node(title="A")
        n2 = _make_node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)

        edge = _make_edge(n1.id, n2.id, kind=EdgeKind.RELATED)
        await backend.save_edge(edge)

        edges = await backend.get_edges(n1.id, direction="outgoing")
        assert len(edges) == 1
        assert edges[0].target_id == n2.id

    @pytest.mark.asyncio
    async def test_edge_directions(self, backend: Neo4jBackend) -> None:
        n1 = _make_node(title="A")
        n2 = _make_node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_edge(_make_edge(n1.id, n2.id))

        out = await backend.get_edges(n1.id, direction="outgoing")
        assert len(out) == 1
        inc = await backend.get_edges(n1.id, direction="incoming")
        assert len(inc) == 0
        both = await backend.get_edges(n1.id, direction="both")
        assert len(both) == 1

    @pytest.mark.asyncio
    async def test_typed_relationships(self, backend: Neo4jBackend) -> None:
        n1 = _make_node(kind=NodeKind.DECISION, title="Deploy v2")
        n2 = _make_node(kind=NodeKind.OUTCOME, title="Success")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_edge(_make_edge(n1.id, n2.id, kind=EdgeKind.RESULTED_IN))

        edges = await backend.get_edges(n1.id, direction="outgoing")
        assert edges[0].kind == EdgeKind.RESULTED_IN

    @pytest.mark.asyncio
    async def test_update_edge(self, backend: Neo4jBackend) -> None:
        n1 = _make_node()
        n2 = _make_node()
        await backend.save_node(n1)
        await backend.save_node(n2)
        edge = _make_edge(n1.id, n2.id, weight=0.5)
        await backend.save_edge(edge)
        edge.weight = 2.0
        await backend.update_edge(edge)
        edges = await backend.get_edges(n1.id)
        assert edges[0].weight == 2.0

    @pytest.mark.asyncio
    async def test_delete_edge(self, backend: Neo4jBackend) -> None:
        n1 = _make_node()
        n2 = _make_node()
        await backend.save_node(n1)
        await backend.save_node(n2)
        edge = _make_edge(n1.id, n2.id)
        await backend.save_edge(edge)
        await backend.delete_edge(edge.id)
        assert await backend.get_edges(n1.id) == []


# --- Search ---


class TestSearch:
    @pytest.mark.asyncio
    async def test_fts_search(self, backend: Neo4jBackend) -> None:
        await backend.save_node(_make_node(title="Docker deployment", content="K8s rollout"))
        await backend.save_node(_make_node(title="Python testing", content="pytest async"))

        results = await backend.search_fts("Docker")
        assert len(results) >= 1
        assert any("Docker" in n.title for n in results)

    @pytest.mark.asyncio
    async def test_fuzzy_search(self, backend: Neo4jBackend) -> None:
        await backend.save_node(_make_node(title="배포 자동화", content="CI/CD 파이프라인"))
        results = await backend.search_fuzzy("배포")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_vector_search_returns_empty(self, backend: Neo4jBackend) -> None:
        results = await backend.search_vector([0.1] * 10)
        assert results == []


# --- Graph Traversal ---


class TestGraphTraversal:
    @pytest.mark.asyncio
    async def test_get_neighbors_depth1(self, backend: Neo4jBackend) -> None:
        n1 = _make_node(title="A")
        n2 = _make_node(title="B")
        n3 = _make_node(title="C")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_node(n3)
        await backend.save_edge(_make_edge(n1.id, n2.id))
        await backend.save_edge(_make_edge(n2.id, n3.id))

        neighbors = await backend.get_neighbors(n1.id, depth=1)
        neighbor_ids = {n.id for n, _ in neighbors}
        assert n2.id in neighbor_ids
        assert n3.id not in neighbor_ids

    @pytest.mark.asyncio
    async def test_get_neighbors_depth2(self, backend: Neo4jBackend) -> None:
        n1 = _make_node(title="A")
        n2 = _make_node(title="B")
        n3 = _make_node(title="C")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_node(n3)
        await backend.save_edge(_make_edge(n1.id, n2.id))
        await backend.save_edge(_make_edge(n2.id, n3.id))

        neighbors = await backend.get_neighbors(n1.id, depth=2)
        neighbor_ids = {n.id for n, _ in neighbors}
        assert n2.id in neighbor_ids
        assert n3.id in neighbor_ids

    @pytest.mark.asyncio
    async def test_shortest_path(self, backend: Neo4jBackend) -> None:
        n1 = _make_node(title="Start")
        n2 = _make_node(title="Mid")
        n3 = _make_node(title="End")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_node(n3)
        await backend.save_edge(_make_edge(n1.id, n2.id))
        await backend.save_edge(_make_edge(n2.id, n3.id))

        path = await backend.shortest_path(n1.id, n3.id)
        assert len(path) > 0
        path_node_ids = {n.id for n, _ in path}
        assert n2.id in path_node_ids or n3.id in path_node_ids


# --- Batch ---


class TestBatch:
    @pytest.mark.asyncio
    async def test_save_nodes_batch(self, backend: Neo4jBackend) -> None:
        nodes = [_make_node(title=f"Node {i}") for i in range(5)]
        await backend.save_nodes_batch(nodes)
        all_nodes = await backend.list_nodes()
        assert len(all_nodes) == 5

    @pytest.mark.asyncio
    async def test_save_edges_batch(self, backend: Neo4jBackend) -> None:
        n1 = _make_node(title="A")
        n2 = _make_node(title="B")
        n3 = _make_node(title="C")
        await backend.save_nodes_batch([n1, n2, n3])
        edges = [
            _make_edge(n1.id, n2.id),
            _make_edge(n2.id, n3.id),
        ]
        await backend.save_edges_batch(edges)
        all_edges = await backend.get_edges(n2.id)
        assert len(all_edges) == 2


# --- Maintenance ---


class TestMaintenance:
    @pytest.mark.asyncio
    async def test_prune_edges(self, backend: Neo4jBackend) -> None:
        n1 = _make_node()
        n2 = _make_node()
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_edge(_make_edge(n1.id, n2.id, weight=0.05))
        pruned = await backend.prune_edges(weight_below=0.1)
        assert pruned >= 1
        assert await backend.get_edges(n1.id) == []

    @pytest.mark.asyncio
    async def test_decay_vitality(self, backend: Neo4jBackend) -> None:
        node = _make_node(vitality=1.0)
        await backend.save_node(node)
        count = await backend.decay_vitality(factor=0.9)
        assert count >= 1
        got = await backend.get_node(node.id)
        assert got is not None
        assert got.vitality == pytest.approx(0.9, abs=0.01)
