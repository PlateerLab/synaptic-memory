"""Composite backend integration tests.

Requires: Neo4j + Qdrant + MinIO all running.
"""

import pytest

from synaptic.models import Edge, EdgeKind, Node

try:
    from synaptic.backends.composite import CompositeBackend
    from synaptic.backends.minio_store import MinIOBackend
    from synaptic.backends.neo4j import Neo4jBackend
    from synaptic.backends.qdrant import QdrantBackend

    HAS_ALL = True
except ImportError:
    HAS_ALL = False

pytestmark = [
    pytest.mark.composite,
    pytest.mark.skipif(not HAS_ALL, reason="Missing neo4j/qdrant/minio deps"),
]

TEST_DIM = 4


@pytest.fixture
async def backend():
    graph = Neo4jBackend("bolt://localhost:7687", auth=("neo4j", "password"))
    vector = QdrantBackend("http://localhost:6333", collection="test_composite", dimension=TEST_DIM)
    blob = MinIOBackend(
        "localhost:9000",
        bucket="test-composite",
        access_key="minio",
        secret_key="minio123",
        secure=False,
    )
    composite = CompositeBackend(graph, vector=vector, blob=blob, blob_threshold=50)

    try:
        await composite.connect()
    except Exception as e:
        pytest.skip(f"Infrastructure not available: {e}")

    yield composite
    await composite.clear_all()
    # Cleanup MinIO bucket
    try:
        client = blob._get_client()
        objects = await client.list_objects("test-composite")
        async for obj in objects:
            await client.remove_object("test-composite", obj.object_name)
        await client.remove_bucket("test-composite")
    except Exception:
        pass
    await composite.close()


def _make_node(**kwargs) -> Node:
    defaults = {"title": "Test", "content": "Short content"}
    defaults.update(kwargs)
    return Node(**defaults)


class TestCompositeNodeCRUD:
    @pytest.mark.asyncio
    async def test_save_and_get(self, backend: CompositeBackend) -> None:
        node = _make_node(title="Hello", content="World")
        await backend.save_node(node)
        got = await backend.get_node(node.id)
        assert got is not None
        assert got.title == "Hello"

    @pytest.mark.asyncio
    async def test_save_with_embedding_routes_to_qdrant(self, backend: CompositeBackend) -> None:
        node = _make_node(
            title="Vectorized",
            embedding=[1.0, 0.0, 0.0, 0.0],
        )
        await backend.save_node(node)

        # Should be findable via vector search
        results = await backend.search_vector([1.0, 0.0, 0.0, 0.0], limit=1)
        assert len(results) == 1
        assert results[0].title == "Vectorized"

    @pytest.mark.asyncio
    async def test_save_large_content_routes_to_minio(self, backend: CompositeBackend) -> None:
        # blob_threshold=50 in fixture
        large_content = "x" * 100
        node = _make_node(title="Large", content=large_content)
        await backend.save_node(node)

        # Content should be restored from MinIO on get
        got = await backend.get_node(node.id)
        assert got is not None
        assert got.content == large_content

    @pytest.mark.asyncio
    async def test_delete_cleans_all_backends(self, backend: CompositeBackend) -> None:
        node = _make_node(
            title="To Delete",
            content="x" * 100,  # will go to MinIO
            embedding=[1.0, 0.0, 0.0, 0.0],  # will go to Qdrant
        )
        await backend.save_node(node)
        await backend.delete_node(node.id)

        assert await backend.get_node(node.id) is None
        # Vector should also be gone
        results = await backend.search_vector([1.0, 0.0, 0.0, 0.0], limit=1)
        assert len(results) == 0


class TestCompositeSearch:
    @pytest.mark.asyncio
    async def test_fts_routes_to_neo4j(self, backend: CompositeBackend) -> None:
        await backend.save_node(_make_node(title="Docker deployment", content="K8s"))
        results = await backend.search_fts("Docker")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_fuzzy_routes_to_neo4j(self, backend: CompositeBackend) -> None:
        await backend.save_node(_make_node(title="배포 자동화", content="CI/CD"))
        results = await backend.search_fuzzy("배포")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_vector_routes_to_qdrant(self, backend: CompositeBackend) -> None:
        await backend.save_node(
            _make_node(
                title="Node A",
                embedding=[1.0, 0.0, 0.0, 0.0],
            )
        )
        await backend.save_node(
            _make_node(
                title="Node B",
                embedding=[0.0, 1.0, 0.0, 0.0],
            )
        )

        results = await backend.search_vector([0.9, 0.1, 0.0, 0.0], limit=1)
        assert len(results) == 1
        assert results[0].title == "Node A"

    @pytest.mark.asyncio
    async def test_vector_without_qdrant_returns_empty(self) -> None:
        """CompositeBackend without Qdrant should return empty for vector search."""
        graph = Neo4jBackend("bolt://localhost:7687", auth=("neo4j", "password"))
        composite = CompositeBackend(graph)  # no vector backend
        try:
            await composite.connect()
        except Exception:
            pytest.skip("Neo4j not available")
        try:
            results = await composite.search_vector([1.0, 0.0, 0.0, 0.0])
            assert results == []
        finally:
            await graph.clear_all()
            await composite.close()


class TestCompositeGraphTraversal:
    @pytest.mark.asyncio
    async def test_neighbors(self, backend: CompositeBackend) -> None:
        n1 = _make_node(title="A")
        n2 = _make_node(title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_edge(Edge(source_id=n1.id, target_id=n2.id, kind=EdgeKind.RELATED))

        neighbors = await backend.get_neighbors(n1.id, depth=1)
        assert len(neighbors) == 1
        assert neighbors[0][0].title == "B"

    @pytest.mark.asyncio
    async def test_shortest_path(self, backend: CompositeBackend) -> None:
        n1 = _make_node(title="Start")
        n2 = _make_node(title="Mid")
        n3 = _make_node(title="End")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_node(n3)
        await backend.save_edge(Edge(source_id=n1.id, target_id=n2.id))
        await backend.save_edge(Edge(source_id=n2.id, target_id=n3.id))

        path = await backend.shortest_path(n1.id, n3.id)
        assert len(path) > 0


class TestCompositeMaintenance:
    @pytest.mark.asyncio
    async def test_decay_vitality(self, backend: CompositeBackend) -> None:
        node = _make_node(vitality=1.0)
        await backend.save_node(node)
        count = await backend.decay_vitality(factor=0.9)
        assert count >= 1

    @pytest.mark.asyncio
    async def test_prune_edges(self, backend: CompositeBackend) -> None:
        n1 = _make_node()
        n2 = _make_node()
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_edge(Edge(source_id=n1.id, target_id=n2.id, weight=0.05))
        pruned = await backend.prune_edges(weight_below=0.1)
        assert pruned >= 1
