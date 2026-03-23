"""Qdrant vector search backend — embedding storage and ANN search."""

from __future__ import annotations

import logging
import uuid

try:
    from qdrant_client import AsyncQdrantClient, models
except ImportError as e:
    msg = "Install synaptic-memory[qdrant] for Qdrant backend: pip install synaptic-memory[qdrant]"
    raise ImportError(msg) from e

logger = logging.getLogger(__name__)


def _node_id_to_uuid(node_id: str) -> str:
    """Convert a node_id string to a deterministic UUID for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, node_id))


class QdrantBackend:
    """Vector-only backend for embedding storage and approximate nearest neighbor search.

    This is NOT a StorageBackend — it's a helper used by CompositeBackend
    to handle the vector search portion.

    Qdrant requires point IDs to be UUIDs or unsigned integers.
    Node IDs (hex strings) are converted to deterministic UUIDs via uuid5,
    and the original node_id is stored in the payload for retrieval.
    """

    __slots__ = ("_client", "_collection", "_dimension", "_url")

    def __init__(
        self,
        url: str = "http://localhost:6333",
        *,
        collection: str = "synaptic",
        dimension: int = 1536,
    ) -> None:
        self._url = url
        self._collection = collection
        self._dimension = dimension
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(url=self._url)
        collections = await self._client.get_collections()
        existing = {c.name for c in collections.collections}
        if self._collection not in existing:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                self._collection,
                self._dimension,
            )
        else:
            logger.info("Qdrant collection '%s' already exists", self._collection)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            msg = "Not connected. Call connect() first."
            raise RuntimeError(msg)
        return self._client

    async def upsert(
        self,
        node_id: str,
        embedding: list[float],
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Store or update a vector with its node ID."""
        client = self._get_client()
        point_id = _node_id_to_uuid(node_id)
        payload: dict[str, object] = {"node_id": node_id}
        if metadata:
            payload.update(metadata)
        await client.upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                ),
            ],
        )

    async def search(
        self,
        embedding: list[float],
        *,
        limit: int = 20,
    ) -> list[str]:
        """Search for nearest vectors. Returns list of node IDs."""
        client = self._get_client()
        try:
            results = await client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=limit,
            )
        except Exception:
            # Empty collection or other query errors → return empty
            return []
        # Extract original node_id from payload
        node_ids: list[str] = []
        for point in results.points:
            if point.payload and "node_id" in point.payload:
                node_ids.append(str(point.payload["node_id"]))
        return node_ids

    async def delete(self, node_id: str) -> None:
        """Delete a vector by node ID."""
        client = self._get_client()
        point_id = _node_id_to_uuid(node_id)
        try:
            await client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=[point_id]),
            )
        except Exception:
            pass  # Idempotent: ignore if point doesn't exist or replica transient error

    async def delete_collection(self) -> None:
        """Delete the entire collection. For testing only."""
        client = self._get_client()
        await client.delete_collection(self._collection)
