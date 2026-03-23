"""Qdrant backend integration tests.

Requires running Qdrant: docker start qdrant
"""

import pytest

try:
    from synaptic.backends.qdrant import QdrantBackend

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

pytestmark = [
    pytest.mark.qdrant,
    pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed"),
]

TEST_DIM = 4  # Small dimension for testing


@pytest.fixture
async def backend():
    # Clean slate: delete leftover collection before connect
    from qdrant_client import AsyncQdrantClient as _AQC

    _tmp = _AQC(url="http://localhost:6333")
    try:
        await _tmp.delete_collection("test_synaptic")
    except Exception:
        pass
    await _tmp.close()

    b = QdrantBackend("http://localhost:6333", collection="test_synaptic", dimension=TEST_DIM)
    try:
        await b.connect()
    except Exception:
        pytest.skip("Qdrant server not available")
    yield b
    try:
        await b.delete_collection()
    except Exception:
        pass
    await b.close()


class TestQdrantLifecycle:
    @pytest.mark.asyncio
    async def test_connect_creates_collection(self, backend: QdrantBackend) -> None:
        # Collection should exist after connect
        client = backend._get_client()
        collections = await client.get_collections()
        names = {c.name for c in collections.collections}
        assert "test_synaptic" in names


class TestQdrantUpsertSearch:
    @pytest.mark.asyncio
    async def test_upsert_and_search(self, backend: QdrantBackend) -> None:
        await backend.upsert("node1", [1.0, 0.0, 0.0, 0.0])
        await backend.upsert("node2", [0.0, 1.0, 0.0, 0.0])
        await backend.upsert("node3", [0.9, 0.1, 0.0, 0.0])  # similar to node1

        results = await backend.search([1.0, 0.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0] == "node1"  # closest match
        assert results[1] == "node3"  # second closest

    @pytest.mark.asyncio
    async def test_upsert_with_metadata(self, backend: QdrantBackend) -> None:
        await backend.upsert(
            "node1",
            [1.0, 0.0, 0.0, 0.0],
            metadata={"title": "Test", "kind": "concept"},
        )
        results = await backend.search([1.0, 0.0, 0.0, 0.0], limit=1)
        assert results == ["node1"]

    @pytest.mark.asyncio
    async def test_search_empty_collection(self, backend: QdrantBackend) -> None:
        results = await backend.search([1.0, 0.0, 0.0, 0.0])
        assert results == []

    @pytest.mark.asyncio
    async def test_upsert_overwrites(self, backend: QdrantBackend) -> None:
        await backend.upsert("node1", [1.0, 0.0, 0.0, 0.0])
        await backend.upsert("node1", [0.0, 1.0, 0.0, 0.0])  # overwrite

        results = await backend.search([0.0, 1.0, 0.0, 0.0], limit=1)
        assert results == ["node1"]


class TestQdrantDelete:
    @pytest.mark.asyncio
    async def test_delete(self, backend: QdrantBackend) -> None:
        await backend.upsert("node1", [1.0, 0.0, 0.0, 0.0])
        await backend.delete("node1")
        results = await backend.search([1.0, 0.0, 0.0, 0.0])
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, backend: QdrantBackend) -> None:
        # Should not raise
        await backend.delete("nonexistent")
