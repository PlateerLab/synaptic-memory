"""Tests for HNSW disk persistence in SQLiteBackend.

Covers the build / load / invalidate cycle of the on-disk vector
index sidecar so cold starts don't pay the rebuild cost twice.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from synaptic.backends.sqlite import SQLiteBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind

# Skip the whole module when usearch isn't available — disk persistence
# only matters for the HNSW path.
usearch = pytest.importorskip("usearch")


def _node(node_id: str, embedding: list[float]) -> Node:
    return Node(
        id=node_id,
        kind=NodeKind.CHUNK,
        title=f"node {node_id}",
        content=f"content for {node_id}",
        tags=[],
        level=ConsolidationLevel.L0_RAW,
        vitality=1.0,
        embedding=embedding,
    )


@pytest.fixture
async def backend_with_vectors():
    """Build a SQLiteBackend with 120 embedding nodes (above HNSW threshold)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "vec.db")
        backend = SQLiteBackend(path)
        await backend.connect()

        # 120 4-dim vectors arranged on a ring so cosine search is meaningful
        import math

        for i in range(120):
            theta = 2 * math.pi * i / 120
            emb = [math.cos(theta), math.sin(theta), 0.0, 0.1]
            await backend.save_node(_node(f"n{i:03d}", emb))

        yield backend, path
        await backend.close()


class TestHNSWDiskPersistence:
    async def test_build_creates_sidecar_files(self, backend_with_vectors):
        backend, path = backend_with_vectors

        # First search builds the index and persists the sidecar
        results = await backend.search_vector([1.0, 0.0, 0.0, 0.1], limit=5)
        assert len(results) > 0

        idx_path = Path(f"{path}.hnsw")
        meta_path = Path(f"{path}.hnsw.meta.json")
        assert idx_path.exists(), "HNSW binary sidecar missing"
        assert meta_path.exists(), "HNSW meta sidecar missing"

        with meta_path.open() as f:
            meta = json.load(f)
        assert meta["count"] == 120
        assert meta["ndim"] == 4
        assert "id_map" in meta and len(meta["id_map"]) == 120

    async def test_second_process_loads_from_disk(self, backend_with_vectors):
        backend, path = backend_with_vectors
        # Build + persist
        await backend.search_vector([1.0, 0.0, 0.0, 0.1], limit=5)
        await backend.close()

        # Fresh backend = simulated cold start
        cold = SQLiteBackend(path)
        await cold.connect()
        # Sanity: in-memory cache empty
        assert cold._hnsw_index is None
        # First search should hit disk, not rebuild
        results = await cold.search_vector([1.0, 0.0, 0.0, 0.1], limit=5)
        assert len(results) > 0
        # Confirm signature loaded
        assert cold._hnsw_meta.get("count") == 120
        await cold.close()

    async def test_signature_invalidates_stale_cache(self, backend_with_vectors):
        backend, path = backend_with_vectors
        # Build + persist
        await backend.search_vector([1.0, 0.0, 0.0, 0.1], limit=5)

        # Add a new embedding node — count should change → cache stale
        await backend.save_node(_node("n_extra", [0.5, 0.5, 0.0, 0.1]))

        # In-memory cache invalidated by save_node
        assert backend._hnsw_index is None

        # Next search should rebuild from scratch and re-persist
        results = await backend.search_vector([0.5, 0.5, 0.0, 0.1], limit=5)
        assert len(results) > 0

        # New count reflected in sidecar
        with Path(f"{path}.hnsw.meta.json").open() as f:
            meta = json.load(f)
        assert meta["count"] == 121

    async def test_delete_hnsw_disk_cache_removes_files(self, backend_with_vectors):
        backend, path = backend_with_vectors
        await backend.search_vector([1.0, 0.0, 0.0, 0.1], limit=5)
        assert Path(f"{path}.hnsw").exists()

        backend.delete_hnsw_disk_cache()
        assert not Path(f"{path}.hnsw").exists()
        assert not Path(f"{path}.hnsw.meta.json").exists()

    async def test_below_threshold_skips_hnsw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "small.db")
            backend = SQLiteBackend(path)
            await backend.connect()
            # Only 10 nodes — below the 100 threshold
            for i in range(10):
                await backend.save_node(_node(f"s{i}", [float(i), 0.0, 0.0, 0.1]))

            # search_vector should fall back to brute force, no sidecar created
            await backend.search_vector([1.0, 0.0, 0.0, 0.1], limit=3)
            assert not Path(f"{path}.hnsw").exists()
            await backend.close()
