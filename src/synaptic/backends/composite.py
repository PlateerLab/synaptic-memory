"""Composite backend — routes operations to Kuzu + Qdrant + MinIO."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from synaptic.models import (
    ConsolidationLevel,
    Edge,
    Node,
    NodeKind,
)

logger = logging.getLogger(__name__)

# Import types conditionally for type checking
try:
    from synaptic.backends.kuzu import KuzuBackend
except ImportError:
    KuzuBackend = None  # type: ignore[assignment,misc]

try:
    from synaptic.backends.qdrant import QdrantBackend
except ImportError:
    QdrantBackend = None  # type: ignore[assignment,misc]

try:
    from synaptic.backends.minio_store import MinIOBackend
except ImportError:
    MinIOBackend = None  # type: ignore[assignment,misc]

_BLOB_PREFIX = "blob://"


class CompositeBackend:
    """Routes storage operations to specialized backends.

    - Kuzu: node/edge CRUD, FTS, fuzzy, graph traversal (embedded)
    - Qdrant: vector embedding search (optional)
    - MinIO: large content blob storage (optional)

    Implements the full StorageBackend protocol + GraphTraversal extensions.
    """

    __slots__ = ("_blob", "_blob_threshold", "_graph", "_vector")

    def __init__(
        self,
        graph: Any,  # KuzuBackend or any StorageBackend
        *,
        vector: Any | None = None,  # QdrantBackend
        blob: Any | None = None,  # MinIOBackend
        blob_threshold: int = 100_000,  # 100KB
    ) -> None:
        self._graph = graph
        self._vector = vector
        self._blob = blob
        self._blob_threshold = blob_threshold

    # --- Lifecycle ---

    async def connect(self) -> None:
        await self._graph.connect()
        if self._vector is not None:
            await self._vector.connect()
        if self._blob is not None:
            await self._blob.connect()
        logger.info(
            "CompositeBackend connected (graph=%s, vector=%s, blob=%s)",
            type(self._graph).__name__,
            type(self._vector).__name__ if self._vector else "None",
            type(self._blob).__name__ if self._blob else "None",
        )

    async def close(self) -> None:
        await self._graph.close()
        if self._vector is not None:
            await self._vector.close()
        if self._blob is not None:
            await self._blob.close()

    # --- Node CRUD ---

    async def save_node(self, node: Node) -> None:
        # Blob offload: large content → MinIO
        if self._blob is not None and len(node.content) > self._blob_threshold:
            await self._blob.upload(node.id, node.content)
            node.content = f"{_BLOB_PREFIX}{node.id}"

        # Vector upsert: embedding → Qdrant
        if self._vector is not None and node.embedding:
            await self._vector.upsert(
                node.id,
                node.embedding,
                metadata={"title": node.title, "kind": str(node.kind)},
            )

        # Graph: always save to Neo4j
        await self._graph.save_node(node)

    async def get_node(self, node_id: str) -> Node | None:
        node = await self._graph.get_node(node_id)
        if node is None:
            return None

        # Blob restore: fetch from MinIO if content is a blob reference
        if self._blob is not None and node.content.startswith(_BLOB_PREFIX):
            try:
                data = await self._blob.download(node_id)
                node.content = data.decode("utf-8")
            except Exception:
                logger.warning("Failed to download blob for node %s", node_id)

        return node

    async def update_node(self, node: Node) -> None:
        # Re-upload blob if content changed and is large
        if self._blob is not None and len(node.content) > self._blob_threshold:
            if not node.content.startswith(_BLOB_PREFIX):
                await self._blob.upload(node.id, node.content)
                node.content = f"{_BLOB_PREFIX}{node.id}"

        # Re-upsert vector if embedding changed
        if self._vector is not None and node.embedding:
            await self._vector.upsert(
                node.id,
                node.embedding,
                metadata={"title": node.title, "kind": str(node.kind)},
            )

        await self._graph.update_node(node)

    async def delete_node(self, node_id: str) -> None:
        await self._graph.delete_node(node_id)
        if self._vector is not None:
            try:
                await self._vector.delete(node_id)
            except Exception:
                pass  # Vector may not exist
        if self._blob is not None:
            try:
                await self._blob.delete(node_id)
            except Exception:
                pass  # Blob may not exist

    async def list_nodes(
        self,
        *,
        kind: str | NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        return await self._graph.list_nodes(kind=kind, level=level, limit=limit)

    # --- Edge CRUD (all to Neo4j) ---

    async def save_edge(self, edge: Edge) -> None:
        await self._graph.save_edge(edge)

    async def get_edges(self, node_id: str, *, direction: str = "both") -> list[Edge]:
        return await self._graph.get_edges(node_id, direction=direction)

    async def update_edge(self, edge: Edge) -> None:
        await self._graph.update_edge(edge)

    async def delete_edge(self, edge_id: str) -> None:
        await self._graph.delete_edge(edge_id)

    # --- Search ---

    async def search_fts(self, query: str, *, limit: int = 20) -> list[Node]:
        return await self._graph.search_fts(query, limit=limit)

    async def search_fuzzy(
        self, query: str, *, limit: int = 20, threshold: float = 0.3
    ) -> list[Node]:
        return await self._graph.search_fuzzy(query, limit=limit, threshold=threshold)

    async def search_vector(self, embedding: list[float], *, limit: int = 20) -> list[Node]:
        if self._vector is None:
            return []
        # Get node IDs from Qdrant, then fetch full nodes from Neo4j
        node_ids = await self._vector.search(embedding, limit=limit)
        nodes: list[Node] = []
        for nid in node_ids:
            node = await self._graph.get_node(nid)
            if node is not None:
                nodes.append(node)
        return nodes

    # --- Graph traversal (Neo4j native) ---

    async def get_neighbors(self, node_id: str, *, depth: int = 1) -> list[tuple[Node, Edge]]:
        return await self._graph.get_neighbors(node_id, depth=depth)

    # --- Batch ---

    async def save_nodes_batch(self, nodes: Sequence[Node]) -> None:
        for node in nodes:
            await self.save_node(node)

    async def save_edges_batch(self, edges: Sequence[Edge]) -> None:
        await self._graph.save_edges_batch(edges)

    # --- Maintenance ---

    async def prune_edges(self, *, weight_below: float = 0.1) -> int:
        return await self._graph.prune_edges(weight_below=weight_below)

    async def decay_vitality(self, *, factor: float = 0.95) -> int:
        return await self._graph.decay_vitality(factor=factor)

    # --- GraphTraversal extensions (passthrough to Neo4j) ---

    async def shortest_path(
        self, from_id: str, to_id: str, *, max_depth: int = 5
    ) -> list[tuple[Node, Edge]]:
        return await self._graph.shortest_path(from_id, to_id, max_depth=max_depth)

    async def pattern_match(self, pattern: str, *, limit: int = 20) -> list[dict[str, object]]:
        return await self._graph.pattern_match(pattern, limit=limit)

    async def find_by_type_hierarchy(self, type_name: str, *, limit: int = 50) -> list[Node]:
        return await self._graph.find_by_type_hierarchy(type_name, limit=limit)

    # --- Admin ---

    async def clear_all(self) -> None:
        """Delete all data. For testing only."""
        await self._graph.clear_all()
        if self._vector is not None:
            try:
                await self._vector.delete_collection()
            except Exception:
                pass
